"""Spreading activation — propagation through the relation network.

Activation factors:
  - concept confidence and maturity
  - relation weight, confidence, and type
  - task affinity: relations/concepts associated with the current task
    propagate more strongly
  - depth decay with configurable falloff
  - propagation floor to prevent low-confidence nodes from blocking spread
"""

from __future__ import annotations

from world0.concepts.manager import ConceptManager
from world0.relations.manager import RelationManager
from world0.schemas.relation import RelationType

# ── Relation type propagation coefficients ────────────────────────────
# Stronger semantic relations propagate more activation.
# "related_to" (the generic fallback) gets the lowest weight.
RELATION_TYPE_FACTOR: dict[RelationType, float] = {
    RelationType.DEPENDS_ON: 1.0,
    RelationType.CONTAINS: 0.95,
    RelationType.PART_OF: 0.95,
    RelationType.SUPPORTS: 0.85,
    RelationType.ACTIVATES: 0.90,
    RelationType.PRECEDES: 0.80,
    RelationType.DERIVED_FROM: 0.80,
    RelationType.SIMILAR_TO: 0.70,
    RelationType.CONTRASTS: 0.40,
    RelationType.RELATED_TO: 0.50,
}

# ── Task affinity boost ──────────────────────────────────────────────
# When the current task matches a concept's or relation's history,
# propagation is multiplied by this factor.
TASK_AFFINITY_BOOST: float = 1.5

# ── Propagation floor ────────────────────────────────────────────────
# Low-confidence nodes still allow propagation to pass through at
# this minimum readiness level, preventing "dead node" blockage.
PROPAGATION_FLOOR: float = 0.3


class ActivationEngine:
    """Spreads activation from seed concepts through the relation network."""

    def __init__(
        self, concepts: ConceptManager, relations: RelationManager
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def activate(
        self,
        seed_ids: list[str],
        *,
        max_depth: int = 2,
        decay: float = 0.5,
        min_activation: float = 0.01,
        source: str = "",
        task: str = "",
        record: bool = True,
    ) -> dict[str, float]:
        """Spread activation from seeds.

        Propagation strength =
            source_score
            * relation.weight * relation.confidence
            * RELATION_TYPE_FACTOR[relation.type]
            * max(neighbor.confidence, PROPAGATION_FLOOR)
            * depth_decay
            * task_affinity

        Args:
            record: If True, touched concepts get their activation_count
                incremented and last_activated updated. Set to False for
                read-only operations like projection.

        Returns concept_id → activation_score mapping.
        """
        activations: dict[str, float] = {}
        task_lower = task.strip().lower()

        # Seed concepts activate at their own confidence level
        for cid in seed_ids:
            node = self._concepts.get(cid)
            if not node:
                continue
            score = node.confidence
            # Boost seeds that have task affinity
            if task_lower and self._concept_has_task(node, task_lower):
                score = min(1.0, score * TASK_AFFINITY_BOOST)
            activations[cid] = score
            if record:
                node.activate(source=source, task=task)

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

                    # Edge strength: use weight as primary signal
                    # (weight and confidence track nearly identically on
                    # relations; multiplying them squares the attenuation)
                    type_factor = RELATION_TYPE_FACTOR.get(
                        rel.relation_type, 0.5
                    )
                    edge_strength = rel.weight * type_factor

                    # Propagation floor prevents dead-node blockage
                    neighbor_readiness = max(
                        neighbor.confidence, PROPAGATION_FLOOR
                    )

                    # Task affinity: boost if relation or neighbor has
                    # history with the current task
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

                    propagated = (
                        source_score
                        * edge_strength
                        * neighbor_readiness
                        * depth_factor
                        * task_boost
                    )

                    if propagated < min_activation:
                        continue

                    old = activations.get(neighbor_id, 0.0)
                    if propagated > old:
                        activations[neighbor_id] = propagated
                        next_frontier.append(neighbor_id)
                        if record:
                            neighbor.activate(source=source, task=task)

            frontier = next_frontier

        return activations

    @staticmethod
    def _concept_has_task(node, task_lower: str) -> bool:
        """Check if a concept has been activated under a matching task."""
        for entry in node.reinforcement_log:
            if task_lower in entry.task.lower():
                return True
        return False
