"""Spreading activation — propagation through the relation network.

Activation factors:
  - concept confidence and maturity
  - relation weight, confidence, and type (type factor is resolved
    *through* the active Perspective, so different frames can produce
    different projections over the same world)
  - task affinity: relations/concepts associated with the current task
    propagate more strongly
  - domain affinity: concepts whose dominant domain is "in focus" for
    the Perspective receive an extra boost
  - temporal relevance: recently active concepts and relations propagate
    more strongly than stale ones
  - seed specificity: rarer seed concepts (attested by fewer distinct
    sources) are more discriminative, so common seeds are blended down
    relative to the rarest seed in the set (IDF-like, HippoRAG-style)
  - fan dilution: high-fan hub concepts spread less per edge (ACT-R fan
    effect), preventing over-connected nodes from flooding the network
  - depth decay with configurable falloff
  - propagation floor to prevent low-confidence nodes from blocking spread

Inhibition:
  Negative-axis edges spread *negative* activation that accumulates on a
  separate inhibition channel.  The final score returned for each
  concept is ``max(0, activation - inhibition)``, so an aggressively
  repelled neighbor can disappear from projections even if it is
  also weakly activated through other paths.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from world0.dynamics.affinity import (
    concept_in_perspective_domain,
    task_affinity,
)
from world0.dynamics.coefficients import (
    CONCEPT_TEMPORAL_HL,
    CONTRASTS_INHIBITION_FACTOR,
    FAN_DILUTION_STRENGTH,
    FAN_DILUTION_THRESHOLD,
    MIN_TASK_AFFINITY,
    PROPAGATION_FLOOR,
    PROPAGATION_MIN_RATIO,
    RELATION_TEMPORAL_HL,
    RELATION_TYPE_FACTOR,
    SEED_SPECIFICITY_WEIGHT,
    TASK_AFFINITY_BOOST,
    ActivationConfig,
)
from world0.schemas.context import Perspective
from world0.schemas.relation import RelationType
from world0.schemas.types import ActivationStep, ActivationTrace

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Re-exported for backwards compatibility.  The constants live in
# ``dynamics.coefficients`` so multiple engines can share them without
# importing each other's modules.
__all__ = [
    "ActivationEngine",
    "ActivationConfig",
    "RELATION_TYPE_FACTOR",
    "CONCEPT_TEMPORAL_HL",
    "RELATION_TEMPORAL_HL",
    "CONTRASTS_INHIBITION_FACTOR",
    "TASK_AFFINITY_BOOST",
    "MIN_TASK_AFFINITY",
    "PROPAGATION_FLOOR",
    "PROPAGATION_MIN_RATIO",
    "SEED_SPECIFICITY_WEIGHT",
    "FAN_DILUTION_THRESHOLD",
    "FAN_DILUTION_STRENGTH",
]


class ActivationEngine:
    """Spreads activation from seed concepts through the relation network.

    Implements the ``ActivationProvider`` Protocol from ``world0.core``.
    Depends only on ``ConceptStore`` / ``RelationStore`` Protocols.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
        config: ActivationConfig | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._cfg = config if config is not None else ActivationConfig()

    def activate(
        self,
        seed_ids: list[str],
        *,
        max_depth: int = 2,
        decay: float = 0.6,
        min_activation: float = 0.01,
        source: str = "",
        task: str = "",
        record: bool = True,
        perspective: Perspective | None = None,
        now: datetime | None = None,
    ) -> dict[str, float]:
        """Spread activation from seeds; scores only (no traces).

        Thin wrapper over :meth:`activate_traced` — see there for the
        propagation model.
        """
        net, _ = self.activate_traced(
            seed_ids,
            max_depth=max_depth,
            decay=decay,
            min_activation=min_activation,
            source=source,
            task=task,
            record=record,
            perspective=perspective,
            now=now,
        )
        return net

    def activate_traced(
        self,
        seed_ids: list[str],
        *,
        max_depth: int = 2,
        decay: float = 0.6,
        min_activation: float = 0.01,
        source: str = "",
        task: str = "",
        record: bool = True,
        perspective: Perspective | None = None,
        now: datetime | None = None,
    ) -> tuple[dict[str, float], dict[str, ActivationTrace]]:
        """Spread activation from seeds with excitation + inhibition.

        Propagation strength (excitatory edges) =
            source_score
            * relation.weight * perspective.weight_for(relation.type)
            * max(neighbor.confidence, PROPAGATION_FLOOR)
            * depth_decay
            * task_affinity
            * domain_affinity
            * relation.temporal_relevance
            * neighbor.temporal_relevance
            * fan_dilution(source)   [hubs past FAN_DILUTION_THRESHOLD
                                      spread less per edge]

        Seed scores are additionally scaled by relative seed
        specificity: rarer seeds (fewer distinct evidence sources) stay
        at full strength while ubiquitous seeds are blended down by
        SEED_SPECIFICITY_WEIGHT.

        Negative-axis edges use the same multiplicative chain but feed an
        independent inhibition channel multiplied by
        CONTRASTS_INHIBITION_FACTOR.  The returned score for each
        concept is ``max(0, activation - inhibition)``.

        Args:
            record: If True, touched concepts get their activation_count
                incremented and last_activated updated. Set to False for
                read-only operations like projection.
            perspective: Task/role view.  Relation type weights, active
                domains and the task label are all sourced from it.
                Plain ``task`` is honored for backward compatibility
                when no perspective is passed.

        Returns ``(net_scores, traces)`` where ``net_scores`` maps
        concept_id → net activation and ``traces`` maps concept_id →
        the best activation path that produced that score.
        """
        # Unify legacy (task: str) and new (Perspective) arguments.
        if perspective is None:
            perspective = Perspective(task=task)
        task_lower = (perspective.task or task).strip().lower()

        # Hot-loop coefficients as locals (config reads once per call).
        cfg = self._cfg
        relation_type_factor = cfg.relation_type_factor
        inhibition_factor = cfg.inhibition_factor
        boost_amplitude = cfg.task_affinity_boost - 1.0
        min_task_affinity = cfg.min_task_affinity
        propagation_floor = cfg.propagation_floor
        propagation_min_ratio = cfg.propagation_min_ratio
        concept_temporal_hl = cfg.concept_temporal_hl
        relation_temporal_hl = cfg.relation_temporal_hl
        fan_threshold = cfg.fan_dilution_threshold
        fan_strength = cfg.fan_dilution_strength

        activations: dict[str, float] = {}
        # concept_id → (inhibition value, inhibiting source concept id)
        inhibitions: dict[str, tuple[float, str]] = {}
        traces: dict[str, ActivationTrace] = {}

        # Seed concepts activate at their own confidence level
        seeded: list[tuple[str, object, float]] = []
        for cid in seed_ids:
            node = self._concepts.get(cid)
            if not node:
                continue
            score = node.confidence
            # Boost seeds that have task affinity (graded)
            if task_lower:
                affinity = self._concept_task_affinity(node, task_lower)
                if affinity >= min_task_affinity:
                    boost = 1.0 + boost_amplitude * affinity
                    score = min(1.0, score * boost)
            # Domain affinity stacks on top — a concept whose dominant
            # domain is "in focus" for the perspective is boosted too.
            if self._concept_in_perspective_domain(node, perspective):
                score = min(
                    1.0, score * perspective.domain_affinity_boost
                )
            seeded.append((cid, node, score))

        # Seed specificity: rarer seeds (fewer distinct evidence
        # sources) are more discriminative.  Normalized across the seed
        # set so the rarest seed keeps full score and ubiquitous seeds
        # are blended toward (1 - weight).  Neutral for a single seed.
        specificity_weight = cfg.seed_specificity_weight
        if specificity_weight > 0.0 and len(seeded) > 1:
            raw_specs = [
                1.0 / self._source_count(node) for _, node, _ in seeded
            ]
            max_spec = max(raw_specs)
            seeded = [
                (
                    cid,
                    node,
                    score
                    * (
                        1.0
                        - specificity_weight
                        + specificity_weight * (raw / max_spec)
                    ),
                )
                for (cid, node, score), raw in zip(seeded, raw_specs)
            ]

        seed_score_max = 0.0
        for cid, node, score in seeded:
            activations[cid] = score
            traces[cid] = ActivationTrace(seed_id=cid, score=score)
            if score > seed_score_max:
                seed_score_max = score
            if record:
                node.activate(source=source, task=task)

        # Propagation floor: minimum signal that can still pass through
        prop_floor = seed_score_max * propagation_min_ratio

        # BFS propagation with decay
        frontier = list(seed_ids)
        for depth in range(max_depth):
            depth_factor = decay ** (depth + 1)
            next_frontier: list[str] = []

            for cid in frontier:
                source_score = activations.get(cid, 0.0)
                if source_score < min_activation:
                    continue

                rels = self._relations.for_concept(cid)

                # Fan dilution (ACT-R fan effect): a hub with many
                # edges spreads less per edge.  Smooth log dilution —
                # unlike ACT-R's smax - ln(fan) it never flips negative.
                fan_factor = 1.0
                fan = len(rels)
                if fan_strength > 0.0 and fan > fan_threshold:
                    fan_factor = 1.0 / (
                        1.0 + fan_strength * math.log(fan / fan_threshold)
                    )

                for rel in rels:
                    neighbor_id = rel.other_end(cid)
                    if neighbor_id is None:
                        continue

                    neighbor = self._concepts.get(neighbor_id)
                    if neighbor is None:
                        continue

                    default_type_factor = relation_type_factor.get(
                        rel.relation_type, 0.5
                    )
                    type_factor = perspective.weight_for_relation(
                        rel.semantic_relation,
                        rel.relation_type.value,
                        default_type_factor,
                    )
                    edge_strength = rel.weight * type_factor

                    neighbor_readiness = max(
                        neighbor.confidence, propagation_floor
                    )

                    task_boost = 1.0
                    if task_lower:
                        affinity = max(
                            task_affinity(task_lower, rel.task_history),
                            self._concept_task_affinity(
                                neighbor, task_lower
                            ),
                        )
                        if affinity >= min_task_affinity:
                            task_boost = 1.0 + boost_amplitude * affinity

                    # Domain affinity boost for perspective-focused domains
                    domain_boost = 1.0
                    if self._concept_in_perspective_domain(
                        neighbor, perspective
                    ):
                        domain_boost = perspective.domain_affinity_boost

                    rel_freshness = rel.temporal_relevance(
                        relation_temporal_hl, now=now
                    )
                    neighbor_freshness = neighbor.temporal_relevance(
                        concept_temporal_hl, now=now
                    )

                    raw = (
                        source_score
                        * edge_strength
                        * neighbor_readiness
                        * depth_factor
                        * task_boost
                        * domain_boost
                        * rel_freshness
                        * neighbor_freshness
                        * fan_factor
                    )

                    if rel.relation_type == RelationType.NEGATIVE:
                        # Inhibitory channel: negative-axis links spread negative
                        # activation instead of weak excitation.
                        inhibition = raw * inhibition_factor
                        if inhibition < min_activation:
                            continue
                        old_inhibition, _ = inhibitions.get(
                            neighbor_id, (0.0, "")
                        )
                        if inhibition > old_inhibition:
                            inhibitions[neighbor_id] = (inhibition, cid)
                        continue

                    propagated = raw
                    floored = False
                    # Apply propagation minimum floor — ensures distant
                    # but structurally connected concepts still receive
                    # enough signal to participate in projections.
                    if propagated < prop_floor and propagated > 0:
                        propagated = prop_floor
                        floored = True

                    if propagated < min_activation:
                        continue

                    old = activations.get(neighbor_id, 0.0)
                    if propagated > old:
                        activations[neighbor_id] = propagated
                        source_trace = traces.get(cid)
                        if source_trace is not None:
                            step = ActivationStep(
                                from_id=cid,
                                relation_id=rel.id,
                                semantic_relation=rel.semantic_relation,
                                axis=rel.relation_type.value,
                                contribution=(
                                    propagated / source_score
                                    if source_score > 0
                                    else 0.0
                                ),
                                floored=floored,
                            )
                            traces[neighbor_id] = source_trace.extended_with(
                                step, propagated
                            )
                        next_frontier.append(neighbor_id)
                        if record:
                            neighbor.activate(source=source, task=task)

            frontier = next_frontier

        # Subtract inhibition from excitation; drop concepts driven to
        # zero or below so they vanish from the projection entirely.
        # Iterate ``activations`` in insertion order (BFS order) so
        # projection selection is deterministic across process runs.
        net: dict[str, float] = {}
        for cid, excitation in activations.items():
            inhibition, inhibitor = inhibitions.get(cid, (0.0, ""))
            score = excitation - inhibition
            if score > min_activation:
                net[cid] = score
                trace = traces.get(cid)
                if trace is not None:
                    trace.score = score
                    if inhibition > 0:
                        trace.inhibition = inhibition
                        trace.inhibition_source = inhibitor
        for cid, (inhibition, _) in inhibitions.items():
            if cid in activations:
                continue
            score = -inhibition
            if score > min_activation:
                net[cid] = score
        # Only return traces for concepts that survived netting.
        surviving_traces = {
            cid: trace for cid, trace in traces.items() if cid in net
        }
        return net, surviving_traces

    @staticmethod
    def _source_count(node) -> int:
        """Distinct evidence sources attesting a concept (always ≥ 1).

        Prefers ``source_refs`` (deduplicated by source id at record
        time); falls back to distinct sources in the reinforcement log
        for concepts ingested before source tracking existed.
        """
        if node.source_refs:
            return len(node.source_refs)
        sources = {
            entry.source for entry in node.reinforcement_log if entry.source
        }
        return max(1, len(sources))

    @staticmethod
    def _concept_in_perspective_domain(
        node, perspective: Perspective
    ) -> bool:
        return concept_in_perspective_domain(node, perspective)

    @staticmethod
    def _concept_task_affinity(node, task_lower: str) -> float:
        """Graded affinity between the task and the concept's history."""
        return task_affinity(
            task_lower, (entry.task for entry in node.reinforcement_log)
        )
