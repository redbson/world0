"""Projection engine — generates LLM-prompt-ready cognitive views.

Uses MMR (Maximal Marginal Relevance) selection to balance activation
score against diversity, preventing hub concepts from monopolizing
the projection.

Task affinity and temporal freshness are integrated into MMR scoring
so that different tasks produce different *sets* of concepts, and
recently active concepts are preferred over stale ones.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from world0.schemas.clock import CognitiveClock
from world0.schemas.types import Projection

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# MMR diversity weight. 0 = pure score ranking, 1 = pure diversity.
# Calibrated on two scenarios (docs/world0-cognitive-dynamics-analysis.md
# §7.5): the cognitive benchmark is insensitive to λ in [0.1, 0.5], while a
# hub with a cluster of six near-identical siblings plus a three-concept
# chain needs λ ≥ 0.5 before the projection covers both regions instead of
# filling half its slots with siblings.
MMR_LAMBDA: float = 0.5

# Task affinity discount applied in MMR when a candidate has *no*
# association with the current task.  Reduces the effective relevance
# so task-aligned concepts are preferred during selection.
# 1.0 = no penalty, 0.0 = completely exclude non-matching concepts.
TASK_AFFINITY_DISCOUNT: float = 0.6

# Temporal freshness weight in MMR relevance computation.
# Controls how much temporal_relevance influences concept selection.
# 0.0 = no time influence, 1.0 = freshness equally weighted as score.
TEMPORAL_WEIGHT: float = 0.3

# Half-life used for temporal relevance in projection, in ticks
# (observations) of cognitive time.
PROJECTION_TEMPORAL_HL: float = 168.0

# Candidate cut as a fraction of the strongest activation.  Applied as
# ``min(min_activation, RELATIVE_MIN_ACTIVATION × peak)`` so it only ever
# *loosens* the absolute floor (for weak seeds), never tightens it.
RELATIVE_MIN_ACTIVATION: float = 0.02


class ProjectionEngine:
    """Generates a Projection from activation scores.

    The projection is the operational output — it is what gets injected
    into the Agent's prompt to shape its reasoning.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
        clock: CognitiveClock | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._clock = clock or CognitiveClock()

    def project(
        self,
        activations: dict[str, float],
        *,
        max_concepts: int = 15,
        min_activation: float = 0.01,
        task: str = "",
    ) -> Projection:
        """Build a cognitive projection from activation scores.

        1. Filter by minimum activation
        2. Compute per-candidate task affinity
        3. MMR greedy selection: balance score × task_affinity vs diversity
        4. Include relations between selected concepts
        5. Return LLM-prompt-ready Projection
        """
        # Filter candidates.  The cut is the *lower* of the absolute floor
        # and a fraction of the strongest activation, so a projection from
        # low-confidence seeds keeps its horizon instead of being emptied
        # by a threshold calibrated for confident seeds.
        peak = max(activations.values(), default=0.0)
        cut = min(min_activation, RELATIVE_MIN_ACTIVATION * peak)
        candidates = {
            cid: score
            for cid, score in activations.items()
            if score >= cut
        }

        if not candidates:
            return Projection(task=task)

        # Normalize scores to [0, 1] for MMR
        max_score = max(candidates.values()) if candidates else 1.0
        if max_score == 0:
            max_score = 1.0

        # Pre-compute task affinity and temporal freshness for each
        # candidate.  Both are multiplied into the MMR relevance score.
        task_lower = task.strip().lower()
        task_affinity: dict[str, float] = {}
        # 1 − raw affinity: how far a candidate is from the task (0 when
        # no task is given).  Used as a redundancy floor below.
        task_mismatch: dict[str, float] = {}
        temporal_freshness: dict[str, float] = {}
        now_tick = self._clock.tick
        now = datetime.now(timezone.utc)
        for cid in candidates:
            node = self._concepts.get(cid)

            # Task affinity — graded association from the concept's task
            # profile (exact task → 1.0, partial word overlap → fraction,
            # unrelated → 0), blended above the discount floor.
            if not task_lower:
                task_affinity[cid] = 1.0
                task_mismatch[cid] = 0.0
            elif node:
                affinity = node.task_affinity(task_lower)
                task_affinity[cid] = (
                    TASK_AFFINITY_DISCOUNT
                    + (1.0 - TASK_AFFINITY_DISCOUNT) * affinity
                )
                task_mismatch[cid] = 1.0 - affinity
            else:
                task_affinity[cid] = TASK_AFFINITY_DISCOUNT
                task_mismatch[cid] = 1.0

            # Temporal freshness: blend 1.0 (ignore time) with the
            # concept's salience (freshness, or evidence-backed persistence
            # for well-established dormant concepts) using TEMPORAL_WEIGHT.
            if node:
                raw_freshness = node.salience(
                    PROJECTION_TEMPORAL_HL, now_tick=now_tick, now=now
                )
                temporal_freshness[cid] = (
                    (1.0 - TEMPORAL_WEIGHT) + TEMPORAL_WEIGHT * raw_freshness
                )
            else:
                temporal_freshness[cid] = 1.0

        # Build neighbor sets for the redundancy term.  A coupling-weighted
        # Jaccard was evaluated against the cognitive benchmark and did not
        # improve precision/recall at any λ (it merely traded ML for Ops
        # precision at λ=0.3 and lost at λ≥0.4), so the plain set overlap
        # stays — see docs/world0-cognitive-dynamics-analysis.md §7.5.
        neighbor_sets: dict[str, set[str]] = {}
        for cid in candidates:
            neighbor_sets[cid] = set(self._relations.neighbors(cid))

        # MMR greedy selection.  Candidates are visited in a stable order
        # (score desc, then id) so exact ties resolve identically in every
        # process — a projection must never depend on PYTHONHASHSEED.
        selected: list[str] = []
        remaining = sorted(candidates, key=lambda cid: (-candidates[cid], cid))

        while remaining and len(selected) < max_concepts:
            best_id = None
            best_mmr = -float("inf")
            # The best task match still available: candidates further
            # from the task than this are "redundant with the task".
            min_mismatch = min(task_mismatch[cid] for cid in remaining)

            for cid in remaining:
                # Relevance incorporates task affinity and temporal
                # freshness so that concepts matching the current task
                # and recently active concepts are preferred during
                # selection, not just ranked higher.
                relevance = (
                    candidates[cid] / max_score
                    * task_affinity[cid]
                    * temporal_freshness[cid]
                )

                # Redundancy: max Jaccard similarity to any already-selected
                if selected:
                    neighbors_c = neighbor_sets.get(cid, set())
                    max_sim = 0.0
                    for sid in selected:
                        neighbors_s = neighbor_sets.get(sid, set())
                        union = neighbors_c | neighbors_s
                        if union:
                            sim = len(neighbors_c & neighbors_s) / len(union)
                        else:
                            sim = 0.0
                        if sim > max_sim:
                            max_sim = sim
                    redundancy = max_sim
                    # Off-task candidates are redundant with the task
                    # itself while better-matching candidates remain:
                    # otherwise a dense on-task cluster (whose members
                    # share one neighbourhood, sim ≈ 1) loses slots to an
                    # unrelated cluster that merely looks "diverse".  An
                    # off-task candidate can still enter when its raw
                    # relevance beats an on-task one by more than the
                    # task discount, and it is unconstrained once no
                    # better match is left (docs §7.12).
                    if task_mismatch[cid] > min_mismatch:
                        redundancy = max(redundancy, task_mismatch[cid])
                else:
                    redundancy = 0.0

                mmr = (1 - MMR_LAMBDA) * relevance - MMR_LAMBDA * redundancy

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = cid

            if best_id is None:
                break

            selected.append(best_id)
            remaining.remove(best_id)

        selected_ids = set(selected)
        selected_scores = {cid: candidates[cid] for cid in selected}

        # Resolve concepts in selection order (deterministic)
        concepts = []
        for cid in selected:
            node = self._concepts.get(cid)
            if node:
                concepts.append(node)

        # Gather relations between selected concepts
        relations = []
        seen: set[str] = set()
        for cid in selected:
            for rel in self._relations.for_concept(cid):
                if rel.id in seen:
                    continue
                if rel.source_id in selected_ids and rel.target_id in selected_ids:
                    relations.append(rel)
                    seen.add(rel.id)
        # Relation-index order depends on filesystem load order; sort so
        # the rendered projection is identical across processes.
        relations.sort(key=lambda r: (-r.weight, r.id))

        return Projection(
            concepts=concepts,
            relations=relations,
            activation_scores=selected_scores,
            task=task,
        )
