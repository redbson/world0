"""Projection engine — generates LLM-prompt-ready cognitive views.

Uses MMR (Maximal Marginal Relevance) selection to balance activation
score against diversity, preventing hub concepts from monopolizing
the projection.

Task affinity and temporal freshness are integrated into MMR scoring
so that different tasks produce different *sets* of concepts, and
recently active concepts are preferred over stale ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from world0.dynamics.affinity import (
    concept_in_perspective_domain,
    task_affinity,
)
from world0.dynamics.coefficients import (
    MMR_LAMBDA,
    PROJECTION_TEMPORAL_HL,
    RELATION_TYPE_FACTOR,
    TASK_AFFINITY_DISCOUNT,
    TEMPORAL_WEIGHT,
    ProjectionConfig,
)
from world0.schemas.context import Perspective
from world0.schemas.relation import RelationEdge
from world0.schemas.types import ActivationTrace, Projection

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# MMR_LAMBDA / TASK_AFFINITY_DISCOUNT / TEMPORAL_WEIGHT /
# PROJECTION_TEMPORAL_HL are re-exported from
# ``dynamics.coefficients`` for backward compatibility; tunable copies
# live on ``ProjectionConfig``.
__all__ = [
    "ProjectionEngine",
    "ProjectionConfig",
    "MMR_LAMBDA",
    "TASK_AFFINITY_DISCOUNT",
    "TEMPORAL_WEIGHT",
    "PROJECTION_TEMPORAL_HL",
]


class ProjectionEngine:
    """Generates a Projection from activation scores.

    The projection is the operational output — it is what gets injected
    into the Agent's prompt to shape its reasoning.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
        config: ProjectionConfig | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._cfg = config if config is not None else ProjectionConfig()

    def project(
        self,
        activations: dict[str, float],
        *,
        max_concepts: int = 15,
        min_activation: float = 0.01,
        task: str = "",
        perspective: Perspective | None = None,
        traces: dict[str, ActivationTrace] | None = None,
        max_relations: int | None = None,
        min_relation_signal: float | None = None,
        now: datetime | None = None,
    ) -> Projection:
        """Build a cognitive projection from activation scores.

        1. Filter by minimum activation
        2. Compute per-candidate task affinity (graded) and, when a
           perspective is active, domain affinity
        3. MMR greedy selection: balance relevance vs diversity
        4. Include relations between selected concepts, ranked by
           perspective-weighted signal; optionally filtered/capped via
           ``min_relation_signal`` / ``max_relations`` (defaults keep
           every relation, the historical behavior)
        5. Return LLM-prompt-ready Projection carrying provenance traces
        """
        # Config coefficients (call-level params override config).
        cfg = self._cfg
        mmr_lambda = cfg.mmr_lambda
        task_discount = cfg.task_affinity_discount
        temporal_weight = cfg.temporal_weight
        temporal_hl = cfg.temporal_hl
        if max_relations is None:
            max_relations = cfg.max_relations
        if min_relation_signal is None:
            min_relation_signal = cfg.min_relation_signal

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

        # Pre-compute task affinity, domain affinity and temporal
        # freshness for each candidate.  All are multiplied into the
        # MMR relevance score.
        task_lower = task.strip().lower()
        task_affinities: dict[str, float] = {}
        domain_affinities: dict[str, float] = {}
        temporal_freshness: dict[str, float] = {}
        for cid in candidates:
            node = self._concepts.get(cid)

            # Task affinity — graded: substring hit scores 1.0 exactly
            # like the legacy boolean check, partial token overlap earns
            # a proportional multiplier between the discount and 1.0.
            if not task_lower:
                task_affinities[cid] = 1.0
            elif node:
                affinity = task_affinity(
                    task_lower,
                    (entry.task for entry in node.reinforcement_log),
                )
                task_affinities[cid] = (
                    task_discount + (1.0 - task_discount) * affinity
                )
            else:
                task_affinities[cid] = task_discount

            # Domain affinity — concepts whose dominant domain is in
            # focus for the perspective are preferred during selection,
            # mirroring the boost they already receive in activation.
            if node is not None and concept_in_perspective_domain(
                node, perspective
            ):
                domain_affinities[cid] = perspective.domain_affinity_boost
            else:
                domain_affinities[cid] = 1.0

            # Temporal freshness: blend 1.0 (ignore time) with the
            # actual temporal_relevance using the temporal weight.
            if node:
                raw_freshness = node.temporal_relevance(
                    temporal_hl, now=now
                )
                temporal_freshness[cid] = (
                    (1.0 - temporal_weight)
                    + temporal_weight * raw_freshness
                )
            else:
                temporal_freshness[cid] = 1.0

        # Build neighbor sets for similarity computation
        neighbor_sets: dict[str, set[str]] = {}
        for cid in candidates:
            neighbor_sets[cid] = set(self._relations.neighbors(cid))

        # MMR greedy selection
        selected: list[str] = []
        remaining = set(candidates.keys())

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
                    * task_affinities[cid]
                    * domain_affinities[cid]
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

                mmr = (1 - mmr_lambda) * relevance - mmr_lambda * redundancy

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = cid

            if best_id is None:
                break

            selected.append(best_id)
            remaining.discard(best_id)

        selected_ids = set(selected)
        selected_scores = {cid: candidates[cid] for cid in selected_ids}

        # Resolve concepts
        concepts = []
        for cid in selected_ids:
            node = self._concepts.get(cid)
            if node:
                concepts.append(node)

        # Gather relations between selected concepts, ranked by their
        # perspective-weighted signal so the strongest edges *for this
        # view* render first.  Defaults (0.0 / None) keep every relation
        # — the historical behavior.
        scored_relations: list[tuple[float, RelationEdge]] = []
        seen: set[str] = set()
        for cid in selected_ids:
            for rel in self._relations.for_concept(cid):
                if rel.id in seen:
                    continue
                if rel.source_id in selected_ids and rel.target_id in selected_ids:
                    seen.add(rel.id)
                    default_factor = RELATION_TYPE_FACTOR.get(
                        rel.relation_type, 0.5
                    )
                    if perspective is not None:
                        factor = perspective.weight_for_relation(
                            rel.semantic_relation,
                            rel.relation_type.value,
                            default_factor,
                        )
                    else:
                        factor = default_factor
                    scored_relations.append((rel.weight * factor, rel))

        scored_relations.sort(key=lambda item: item[0], reverse=True)
        relations = [
            rel
            for signal, rel in scored_relations
            if signal >= min_relation_signal
        ]
        if max_relations is not None:
            relations = relations[:max_relations]

        # Restrict traces to the selected concepts — the projection
        # explains what it shows, nothing more.
        selected_traces = {
            cid: traces[cid]
            for cid in selected_ids
            if traces is not None and cid in traces
        }

        return Projection(
            concepts=concepts,
            relations=relations,
            activation_scores=selected_scores,
            task=task,
            perspective_name=perspective.name if perspective else "",
            traces=selected_traces,
        )
