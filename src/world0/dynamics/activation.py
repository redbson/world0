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
  - depth decay with configurable falloff
  - propagation floor to prevent low-confidence nodes from blocking spread

Inhibition:
  CONTRASTS edges spread *negative* activation that accumulates on a
  separate inhibition channel.  The final score returned for each
  concept is ``max(0, activation - inhibition)``, so an aggressively
  contrasted neighbor can disappear from projections even if it is
  also weakly activated through other paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.dynamics.coefficients import (
    CONCEPT_TEMPORAL_HL,
    RELATION_TEMPORAL_HL,
    RELATION_TYPE_FACTOR,
)
from world0.schemas.context import Perspective
from world0.schemas.relation import RelationType

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Re-exported for backwards compatibility.  The constants live in
# ``dynamics.coefficients`` so multiple engines can share them without
# importing each other's modules.
__all__ = [
    "ActivationEngine",
    "RELATION_TYPE_FACTOR",
    "CONCEPT_TEMPORAL_HL",
    "RELATION_TEMPORAL_HL",
    "CONTRASTS_INHIBITION_FACTOR",
    "TASK_AFFINITY_BOOST",
    "PROPAGATION_FLOOR",
    "PROPAGATION_MIN_RATIO",
]

# ── Inhibition coefficient for CONTRASTS ──────────────────────────────
# CONTRASTS is treated as an *inhibitory* relation: activating the
# source produces negative activation on the target (which subtracts
# from any positive spread on the same target).  The scalar below is
# multiplied by the same edge/depth/task/temporal factors used for
# excitation — contrast strength tracks the evidence behind it.
CONTRASTS_INHIBITION_FACTOR: float = 0.6

# ── Task affinity boost ──────────────────────────────────────────────
# When the current task matches a concept's or relation's history,
# propagation is multiplied by this factor.
TASK_AFFINITY_BOOST: float = 1.5

# ── Propagation floor ────────────────────────────────────────────────
# Low-confidence nodes still allow propagation to pass through at
# this minimum readiness level, preventing "dead node" blockage.
PROPAGATION_FLOOR: float = 0.3

# ── Propagation minimum ratio ────────────────────────────────────────
# Ensures propagated score is at least this fraction of the *seed*
# score at each depth step, preventing the multiplicative chain from
# zeroing out signal too early.  This widens the cognitive horizon
# from ~1 hop to 3-4 hops.
PROPAGATION_MIN_RATIO: float = 0.03


class ActivationEngine:
    """Spreads activation from seed concepts through the relation network.

    Implements the ``ActivationProvider`` Protocol from ``world0.core``.
    Depends only on ``ConceptStore`` / ``RelationStore`` Protocols.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
    ) -> None:
        self._concepts = concepts
        self._relations = relations

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
    ) -> dict[str, float]:
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

        CONTRASTS edges use the same multiplicative chain but feed an
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

        Returns concept_id → *net* activation score mapping.
        """
        # Unify legacy (task: str) and new (Perspective) arguments.
        if perspective is None:
            perspective = Perspective(task=task)
        task_lower = (perspective.task or task).strip().lower()

        activations: dict[str, float] = {}
        inhibitions: dict[str, float] = {}

        # Seed concepts activate at their own confidence level
        seed_score_max = 0.0
        for cid in seed_ids:
            node = self._concepts.get(cid)
            if not node:
                continue
            score = node.confidence
            # Boost seeds that have task affinity
            if task_lower and self._concept_has_task(node, task_lower):
                score = min(1.0, score * TASK_AFFINITY_BOOST)
            # Domain affinity stacks on top — a concept whose dominant
            # domain is "in focus" for the perspective is boosted too.
            if self._concept_in_perspective_domain(node, perspective):
                score = min(
                    1.0, score * perspective.domain_affinity_boost
                )
            activations[cid] = score
            if score > seed_score_max:
                seed_score_max = score
            if record:
                node.activate(source=source, task=task)

        # Propagation floor: minimum signal that can still pass through
        prop_floor = seed_score_max * PROPAGATION_MIN_RATIO

        # BFS propagation with decay
        frontier = list(seed_ids)
        for depth in range(max_depth):
            depth_factor = decay ** (depth + 1)
            next_frontier: list[str] = []

            for cid in frontier:
                source_score = activations.get(cid, 0.0)
                if source_score < min_activation:
                    continue

                for rel in self._relations.for_concept(cid):
                    neighbor_id = rel.other_end(cid)
                    if neighbor_id is None:
                        continue

                    neighbor = self._concepts.get(neighbor_id)
                    if neighbor is None:
                        continue

                    default_type_factor = RELATION_TYPE_FACTOR.get(
                        rel.relation_type, 0.5
                    )
                    type_factor = perspective.weight_for(
                        rel.relation_type.value, default_type_factor
                    )
                    edge_strength = rel.weight * type_factor

                    neighbor_readiness = max(
                        neighbor.confidence, PROPAGATION_FLOOR
                    )

                    task_boost = 1.0
                    if task_lower:
                        rel_match = any(
                            task_lower in t.lower()
                            for t in rel.task_history
                        )
                        node_match = self._concept_has_task(
                            neighbor, task_lower
                        )
                        if rel_match or node_match:
                            task_boost = TASK_AFFINITY_BOOST

                    # Domain affinity boost for perspective-focused domains
                    domain_boost = 1.0
                    if self._concept_in_perspective_domain(
                        neighbor, perspective
                    ):
                        domain_boost = perspective.domain_affinity_boost

                    rel_freshness = rel.temporal_relevance(RELATION_TEMPORAL_HL)
                    neighbor_freshness = neighbor.temporal_relevance(
                        CONCEPT_TEMPORAL_HL
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
                    )

                    if rel.relation_type == RelationType.CONTRASTS:
                        # Inhibitory channel: contrast spreads negative
                        # activation instead of weak excitation.
                        inhibition = raw * CONTRASTS_INHIBITION_FACTOR
                        if inhibition < min_activation:
                            continue
                        old = inhibitions.get(neighbor_id, 0.0)
                        if inhibition > old:
                            inhibitions[neighbor_id] = inhibition
                        continue

                    propagated = raw
                    # Apply propagation minimum floor — ensures distant
                    # but structurally connected concepts still receive
                    # enough signal to participate in projections.
                    if propagated < prop_floor and propagated > 0:
                        propagated = prop_floor

                    if propagated < min_activation:
                        continue

                    old = activations.get(neighbor_id, 0.0)
                    if propagated > old:
                        activations[neighbor_id] = propagated
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
            score = excitation - inhibitions.get(cid, 0.0)
            if score > min_activation:
                net[cid] = score
        for cid, inhibition in inhibitions.items():
            if cid in activations:
                continue
            score = -inhibition
            if score > min_activation:
                net[cid] = score
        return net

    @staticmethod
    def _concept_in_perspective_domain(
        node, perspective: Perspective
    ) -> bool:
        if not perspective.active_domains:
            return False
        # Prefer the dominant domain in the concept's profile, fall
        # back to its static ``domain`` field.
        if node.domain_profile:
            top_domain, _ = max(
                node.domain_profile.items(), key=lambda item: item[1]
            )
            if perspective.domain_match(top_domain):
                return True
        return perspective.domain_match(node.domain)

    @staticmethod
    def _concept_has_task(node, task_lower: str) -> bool:
        """Check if a concept has been activated under a matching task."""
        for entry in node.reinforcement_log:
            if task_lower in entry.task.lower():
                return True
        return False
