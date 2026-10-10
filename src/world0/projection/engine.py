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

from world0.context import IGNITION_THRESHOLD, Focus, ground_task
from world0.dynamics.coefficients import PROPAGATION_MIN_RATIO
from world0.dynamics.decay import (
    RELATION_PRUNE_THRESHOLD,
    concept_expired,
    projected_relation_weight,
)
from world0.projection.metacognition import assess
from world0.schemas.clock import CognitiveClock
from world0.schemas.types import AttentionTrace, Projection

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

# A relation is *in the task's context* when it was observed under a task
# matching the current one at least this well (distinctiveness-weighted,
# ``TaskVocabulary``).  When a concept in view has claims in the current
# context, its claims observed only under other tasks describe it in another
# context (another sense of a polysemous name, another project's wiring):
# they move to ``Projection.other_contexts`` instead of mixing into the view.
# A concept with no claims in the current context keeps all of them — there
# is no evidence that its other claims belong elsewhere (docs §7.24).
CONTEXT_MATCH: float = 0.5

# Temporal freshness weight in MMR relevance computation.
# Controls how much temporal_relevance influences concept selection.
# 0.0 = no time influence, 1.0 = freshness equally weighted as score.
TEMPORAL_WEIGHT: float = 0.3

# Half-life used for temporal relevance in projection, in ticks
# (observations) of cognitive time.
PROJECTION_TEMPORAL_HL: float = 168.0

# Relevance multiplier for a candidate fully in the sustained focus:
# ``relevance × (1 + FOCUS_GAIN × focus_affinity)`` (docs/mc/03-workspace.md).
FOCUS_GAIN: float = 0.5

# Candidate cut as a fraction of the strongest activation.  Applied as
# ``min(min_activation, RELATIVE_MIN_ACTIVATION × peak)`` so it only ever
# *loosens* the absolute floor (for weak seeds), never tightens it.
RELATIVE_MIN_ACTIVATION: float = 0.02

# Scores are compared at 1e-6.  Two concepts with identical evidence differ
# by ~1e-11 of wall-clock drift noise (a few microseconds between two
# activations of one observation), and ordering them by that noise made a
# projection depend on scheduling jitter — and on how much of the jitter a
# maturity-dependent half-life forgets.  Below the quantum they are tied
# and resolved by recency (the concept activated last first — the order the
# noise used to impose, now decided by the timestamps themselves), then
# name, then id.  (A quantum five orders of magnitude coarser than the
# noise keeps a tie group from straddling a rounding boundary in practice.)
# The quantum is relative to the strongest activation of the view: a view
# built from faded seeds has peak scores of 1e-6 and below, where an
# absolute quantum would tie everything and discard the ranking.
SCORE_DIGITS: int = 6


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

    def _tiebreak(self, concept_id: str) -> tuple[float, str, str]:
        node = self._concepts.get(concept_id)
        if node is None:
            return (0.0, "", concept_id)
        return (-node.last_activated.timestamp(), node.normalized_name(), concept_id)

    def project(
        self,
        activations: dict[str, float],
        *,
        max_concepts: int = 15,
        min_activation: float = 0.01,
        task: str = "",
        seed_ids: list[str] | None = None,
        focus: Focus | None = None,
    ) -> Projection:
        """Build a cognitive projection from activation scores.

        ``focus`` (optional) is the sustained workspace: candidates it
        holds, or that neighbour what it holds, get a relevance bonus.

        1. Filter by minimum activation (seeds are never filtered out)
        2. Compute per-candidate task affinity: the larger of the
           concept's task history and the task's lexical grounding
        3. Seeds first: what the Agent asked about is always in its
           projection, ordered by score and capped by ``max_concepts``
        4. MMR greedy selection for the remaining slots: balance
           score × task_affinity vs diversity.  ``max_concepts`` is a
           ceiling, not a target: a candidate whose known context is
           another task and whose activation sits in the floor band is
           filler and never selected (see ``_off_task``).  Candidates in
           the floor band carry no evidential strength to trade against
           diversity, so MMR runs over the candidates above the band and
           the band fills what is left in activation order
        5. Include relations between selected concepts
        6. Return LLM-prompt-ready Projection
        """
        # Filter candidates.  The cut is the *lower* of the absolute floor
        # and a fraction of the strongest activation, so a projection from
        # low-confidence seeds keeps its horizon instead of being emptied
        # by a threshold calibrated for confident seeds.
        peak = max(activations.values(), default=0.0)
        cut = min(min_activation, RELATIVE_MIN_ACTIVATION * peak)
        seed_set = set(seed_ids or ())
        # One reference instant; liveness is read from settled state so the
        # view does not depend on whether a reflect already pruned the dead
        # (docs/paper Corollary 3.2').
        now_tick = self._clock.tick
        now = datetime.now(timezone.utc)
        weights: dict[str, float] = {}

        def live_weight(rel) -> float:
            """Settled weight of a live relation, 0 for withdrawn / dead ones."""
            if rel.id not in weights:
                w = 0.0 if rel.is_retracted else projected_relation_weight(rel, now_tick, now)
                weights[rel.id] = w if w >= RELATION_PRUNE_THRESHOLD else 0.0
            return weights[rel.id]

        activations = {
            cid: score
            for cid, score in activations.items()
            if cid in seed_set
            or (node := self._concepts.get(cid)) is None
            or not concept_expired(node, now_tick, now)
        }
        quantum = peak if peak > 0.0 else 1.0
        candidates = {
            cid: round(score / quantum, SCORE_DIGITS) * quantum
            for cid, score in activations.items()
            if score >= cut or cid in seed_set
        }

        if not candidates:
            return Projection(task=task)

        # Normalize scores to [0, 1] for MMR
        max_score = max(candidates.values()) if candidates else 1.0
        if max_score == 0:
            max_score = 1.0

        # Build neighbor sets for the redundancy term.  A coupling-weighted
        # Jaccard was evaluated against the cognitive benchmark and did not
        # improve precision/recall at any λ (it merely traded ML for Ops
        # precision at λ=0.3 and lost at λ≥0.4), so the plain set overlap
        # stays — see docs/world0-cognitive-dynamics-analysis.md §7.5.
        neighbor_sets: dict[str, set[str]] = {}
        for cid in candidates:
            neighbor_sets[cid] = {
                other
                for rel in self._relations.for_concept(cid)
                if live_weight(rel) > 0.0 and (other := rel.other_end(cid)) is not None
            }

        # Pre-compute task affinity and temporal freshness for each
        # candidate.  Both are multiplied into the MMR relevance score.
        task_lower = task.strip().lower()
        vocabulary = getattr(self._concepts, "task_vocabulary", None)
        # Grounded affinity: the concepts the task names and their direct
        # neighbours.  Without it a task never seen in an observation
        # (or any task in a world built from unlabelled observations)
        # changed nothing — docs §7.17.
        grounded = (
            ground_task(task_lower, candidates, self._concepts, neighbor_sets)
            if task_lower
            else {}
        )
        focus_affinity = (
            focus.affinity(candidates, neighbor_sets, exclude=seed_set)
            if focus is not None
            else {}
        )
        history_affinity: dict[str, float] = {}
        task_affinity: dict[str, float] = {}
        # 1 − raw affinity: how far a candidate is from the task (0 when
        # no task is given).  Used as a redundancy floor below.
        task_mismatch: dict[str, float] = {}
        temporal_freshness: dict[str, float] = {}
        for cid in candidates:
            node = self._concepts.get(cid)

            # Task affinity — graded association from the concept's task
            # profile (exact task → 1.0, partial word overlap → fraction,
            # unrelated → 0), blended above the discount floor.
            if not task_lower:
                task_affinity[cid] = 1.0
                task_mismatch[cid] = 0.0
            elif node:
                history_affinity[cid] = node.task_affinity(task_lower, vocabulary)
                affinity = max(history_affinity[cid], grounded.get(cid, 0.0))
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

        # Relevance incorporates task affinity, temporal freshness and the
        # sustained focus so that concepts matching the current task,
        # recently active concepts and concepts continuing the current
        # line of attention are preferred during selection, not just
        # ranked higher.
        relevance: dict[str, float] = {
            cid: round(
                candidates[cid] / max_score
                * task_affinity[cid]
                * temporal_freshness[cid]
                * (1.0 + FOCUS_GAIN * focus_affinity.get(cid, 0.0)),
                SCORE_DIGITS,
            )
            for cid in candidates
        }

        # MMR greedy selection.  Candidates are visited in a stable order
        # (score desc, then id) so exact ties resolve identically in every
        # process — a projection must never depend on PYTHONHASHSEED.
        remaining = sorted(candidates, key=lambda cid: (-candidates[cid], *self._tiebreak(cid)))
        # Seeds first.  The seeds are the Agent's explicit focus; MMR must
        # never trade one away for a "more diverse" or better-matching
        # neighbour (it did: a cross-domain second seed under a task, or
        # several seeds from one cluster, were displaced — docs §7.13).
        selected: list[str] = [cid for cid in remaining if cid in seed_set][:max_concepts]
        for cid in selected:
            remaining.remove(cid)

        # Filler: a candidate whose known context is another task *and*
        # whose activation was lifted into the floor band — kept by the
        # activation engine for horizon completeness, not on the strength
        # of its evidence.  Filling the remaining slots with such candidates
        # is what made a task-conditioned view wander across a polysemous
        # bridge into the other domain once the task's own neighbourhood was
        # exhausted (docs §7.29).  An off-task concept the seeds reach
        # strongly (a direct claim, say) still competes on its merit; when
        # no candidate is in the task's context (a label the world has
        # never seen, or a wrong one) nothing is filler and the view is
        # filled as before.
        floor_band = PROPAGATION_MIN_RATIO * peak
        off_task = self._off_task(remaining, task_lower, task_mismatch)
        if any(cid not in off_task for cid in remaining):
            remaining = [
                cid for cid in remaining
                if cid not in off_task or candidates[cid] >= floor_band
            ]
        # Diversity is traded against evidence.  Candidates lifted into the
        # floor band carry no evidential strength to trade — the band keeps
        # only their order (closer, stronger first) — so MMR runs over the
        # candidates above the band and the band then fills what is left in
        # that order.  Before, the redundancy term dominated the band's
        # ~1 % relevance differences and bought "coverage" of third-hop
        # concepts at the expense of second-hop ones (docs §7.29).
        weak = [cid for cid in remaining if candidates[cid] < floor_band]
        remaining = [cid for cid in remaining if candidates[cid] >= floor_band]

        while remaining and len(selected) < max_concepts:
            best_id = None
            best_mmr = -float("inf")
            # The best task match still available: candidates further
            # from the task than this are "redundant with the task".
            min_mismatch = min(task_mismatch[cid] for cid in remaining)

            for cid in remaining:
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

                mmr = (1 - MMR_LAMBDA) * relevance[cid] - MMR_LAMBDA * redundancy

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = cid

            if best_id is None:
                break

            selected.append(best_id)
            remaining.remove(best_id)

        for cid in weak:
            if len(selected) >= max_concepts:
                break
            selected.append(cid)

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
        retracted = []
        seen: set[str] = set()
        for cid in selected:
            for rel in self._relations.for_concept(cid):
                if rel.id in seen:
                    continue
                if rel.is_retracted:
                    # Withdrawn claims are history, not part of the view;
                    # those about a seed are reported (until they are dead)
                    # so the Agent knows what no longer holds.
                    if (rel.source_id in seed_set or rel.target_id in seed_set) and (
                        projected_relation_weight(rel, now_tick, now) >= RELATION_PRUNE_THRESHOLD
                    ):
                        retracted.append(rel)
                        seen.add(rel.id)
                    continue
                if live_weight(rel) <= 0.0:
                    continue
                if rel.source_id in selected_ids and rel.target_id in selected_ids:
                    relations.append(rel)
                    seen.add(rel.id)
        retracted.sort(key=lambda r: (-(r.retracted_tick or 0), r.id))
        outside_names: dict[str, str] = {}
        outside_senses: dict[str, str] = {}
        for rel in retracted:
            for end in (rel.source_id, rel.target_id):
                if end not in selected_ids and end not in outside_names:
                    node = self._concepts.get(end)
                    if node is not None:
                        outside_names[end] = node.name
                        outside_senses[end] = node.representation_feature()
        # Relation-index order depends on filesystem load order; sort so
        # the rendered projection is identical across processes.  Weights
        # are compared at the same relative quantum as activation scores:
        # two claims stated the same way differ only by the wall-clock term
        # of their settled weight (minutes between their statements), and
        # that must not decide which of them the full render's first ten
        # relations show from one run to the next.
        peak_weight = max((live_weight(r) for r in relations), default=0.0) or 1.0
        relations.sort(key=lambda r: (-round(live_weight(r) / peak_weight, SCORE_DIGITS), r.id))
        relations, other_contexts = self._split_by_context(
            relations, selected, task_lower, vocabulary, live_weight
        )

        return Projection(
            concepts=concepts,
            relations=relations,
            other_contexts=other_contexts,
            retracted=retracted,
            outside_names=outside_names,
            outside_senses=outside_senses,
            activation_scores=selected_scores,
            task=task,
            # Claims moved to other contexts still compete with the ones in
            # view: a contested pair must not look settled because its
            # leading claim was stated under another task.
            epistemic=assess(concepts, relations + other_contexts),
            attention=self._attention(
                selected,
                seed_set,
                candidates,
                relevance,
                grounded,
                history_affinity,
                focus_affinity,
                set(focus.items()) if focus is not None else set(),
                live_weight,
            ),
        )

    def _off_task(self, candidates, task_lower: str, task_mismatch: dict[str, float]) -> set[str]:
        """Candidates with a *known, different* context.

        A concept has a known context when it carries a task profile; it
        is off-task when its affinity for the current task (history or
        grounding) is below ``CONTEXT_MATCH``, the same line that decides
        whether a claim is in the task's context.  A concept with no task
        profile at all is neutral — there is no evidence it belongs
        elsewhere — and without a task nothing is off-task.
        """
        if not task_lower:
            return set()
        off: set[str] = set()
        for cid in candidates:
            node = self._concepts.get(cid)
            if node is None or not node.task_profile:
                continue
            if 1.0 - task_mismatch[cid] < CONTEXT_MATCH:
                off.add(cid)
        return off

    def _split_by_context(self, relations, selected, task_lower, vocabulary, live_weight):
        """(in-context relations, relations that describe a concept elsewhere).

        See ``CONTEXT_MATCH``.  A claim's context is the tasks it was
        *stated* under (``RelationEdge.claim_tasks``); co-occurrence edges and
        claims stated without a task are neutral.  Only live claims count as
        evidence that a concept has claims in the current context.
        """
        if not task_lower or not relations:
            return relations, []

        def context_of(rel) -> bool | None:
            if not rel.is_explicit:
                return None
            affinity = rel.claim_affinity(task_lower, vocabulary)
            return None if affinity is None else affinity >= CONTEXT_MATCH

        in_context: set[str] = set()
        for cid in selected:
            if any(
                live_weight(rel) > 0.0 and context_of(rel)
                for rel in self._relations.for_concept(cid)
            ):
                in_context.add(cid)
        if not in_context:
            return relations, []
        kept, moved = [], []
        for rel in relations:
            if context_of(rel) is False and (
                rel.source_id in in_context or rel.target_id in in_context
            ):
                moved.append(rel)
            else:
                kept.append(rel)
        return kept, moved

    def _attention(
        self,
        selected: list[str],
        seed_set: set[str],
        candidates: dict[str, float],
        relevance: dict[str, float],
        grounded: dict[str, float],
        history_affinity: dict[str, float],
        focus_affinity: dict[str, float],
        focus_members: set[str],
        live_weight=None,
    ) -> dict[str, AttentionTrace]:
        """The attention schema of this view (docs/mc/03-workspace.md).

        Ignition is all-or-none: seeds always ignite; another selected
        concept ignites when its relevance reaches ``IGNITION_THRESHOLD``
        of the strongest non-seed relevance in the view.
        """
        non_seed = [relevance[cid] for cid in selected if cid not in seed_set]
        ignition_level = IGNITION_THRESHOLD * max(non_seed, default=0.0)
        traces: dict[str, AttentionTrace] = {}
        for cid in selected:
            if cid in seed_set:
                traces[cid] = AttentionTrace(kind="seed", ignited=True)
                continue
            via, via_relation, best = "", "", 0.0
            for rel in self._relations.for_concept(cid):
                other = rel.other_end(cid)
                if other is None or other not in candidates:
                    continue
                weight = live_weight(rel) if live_weight is not None else rel.weight
                if weight <= 0.0:
                    continue
                contribution = candidates[other] * weight
                if contribution > best or (contribution == best and other < via):
                    via, via_relation, best = other, rel.semantic_relation, contribution
            traces[cid] = AttentionTrace(
                kind="reached",
                via=via,
                relation=via_relation,
                task_named=grounded.get(cid, 0.0) > 0.0,
                task_history=history_affinity.get(cid, 0.0) > 0.0,
                sustained=focus_affinity.get(cid, 0.0) > 0.0,
                in_focus=cid in focus_members,
                ignited=relevance[cid] >= ignition_level > 0.0,
            )
        return traces
