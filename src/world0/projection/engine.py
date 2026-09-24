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
# 0.3 gives a mild diversity nudge while keeping relevance primary.
MMR_LAMBDA: float = 0.3

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
        # Filter candidates
        candidates = {
            cid: score
            for cid, score in activations.items()
            if score >= min_activation
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
            elif node:
                affinity = node.task_affinity(task_lower)
                task_affinity[cid] = (
                    TASK_AFFINITY_DISCOUNT
                    + (1.0 - TASK_AFFINITY_DISCOUNT) * affinity
                )
            else:
                task_affinity[cid] = TASK_AFFINITY_DISCOUNT

            # Temporal freshness: blend 1.0 (ignore time) with the
            # actual temporal_relevance using TEMPORAL_WEIGHT.
            if node:
                raw_freshness = node.temporal_relevance(
                    PROJECTION_TEMPORAL_HL, now_tick=now_tick, now=now
                )
                temporal_freshness[cid] = (
                    (1.0 - TEMPORAL_WEIGHT) + TEMPORAL_WEIGHT * raw_freshness
                )
            else:
                temporal_freshness[cid] = 1.0

        # Build neighbor sets for similarity computation
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
