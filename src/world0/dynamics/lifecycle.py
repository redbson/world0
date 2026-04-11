"""Lifecycle management — maturity promotion and demotion rules.

Promotion rules:
  embryonic → developing:   activation_count >= 3 and confidence >= 0.3
  developing → established: activation_count >= 10 and confidence >= 0.6
  established → core:       activation_count >= 30 and connections >= 5

Demotion:
  any → fading: handled by decay engine (confidence < 0.05)
  fading → developing: handled by ConceptNode.activate() (on re-activation)
"""

from __future__ import annotations

from world0.concepts.manager import ConceptManager
from world0.relations.manager import RelationManager
from world0.schemas.concept import Maturity


class LifecycleEngine:
    """Evaluates and applies maturity transitions for concepts."""

    def __init__(
        self, concepts: ConceptManager, relations: RelationManager
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def evaluate(self) -> tuple[list[str], list[str]]:
        """Evaluate all concepts for promotion or demotion.

        Returns (promoted_ids, demoted_ids).
        """
        promoted: list[str] = []
        demoted: list[str] = []

        for node in self._concepts.all():
            new_maturity = self._evaluate_one(node)
            if new_maturity and new_maturity != node.maturity:
                old = node.maturity
                self._concepts.update_maturity(node.id, new_maturity)
                if self._is_promotion(old, new_maturity):
                    promoted.append(node.id)
                else:
                    demoted.append(node.id)

        return promoted, demoted

    def _evaluate_one(self, node) -> Maturity | None:
        """Determine if a concept should change maturity."""
        connections = len(self._relations.for_concept(node.id))

        if node.maturity == Maturity.EMBRYONIC:
            if node.activation_count >= 3 and node.confidence >= 0.3:
                return Maturity.DEVELOPING
            return None

        if node.maturity == Maturity.DEVELOPING:
            if node.activation_count >= 10 and node.confidence >= 0.6:
                return Maturity.ESTABLISHED
            return None

        if node.maturity == Maturity.ESTABLISHED:
            if node.activation_count >= 30 and connections >= 5:
                return Maturity.CORE
            return None

        return None

    @staticmethod
    def _is_promotion(old: Maturity, new: Maturity) -> bool:
        order = {
            Maturity.FADING: 0,
            Maturity.EMBRYONIC: 1,
            Maturity.DEVELOPING: 2,
            Maturity.ESTABLISHED: 3,
            Maturity.CORE: 4,
        }
        return order.get(new, 0) > order.get(old, 0)
