"""Time-based decay — unused concepts and relations fade over time.

Decay rates are maturity-dependent:
  - core concepts decay very slowly
  - embryonic concepts decay fast
  - relations decay inversely with reinforcement count

Optimization: items activated/reinforced within a grace period are
skipped entirely, avoiding unnecessary floating-point work.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from world0.schemas.concept import Maturity

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Half-life in hours per maturity level
CONCEPT_HALF_LIFE: dict[Maturity, float] = {
    Maturity.EMBRYONIC: 24.0,       # 1 day
    Maturity.DEVELOPING: 168.0,     # 1 week
    Maturity.ESTABLISHED: 720.0,    # 1 month
    Maturity.CORE: 2160.0,          # 3 months
    Maturity.FADING: 24.0,          # 1 day (already fading)
}

# Base half-life for relations (modified by reinforcement count)
RELATION_BASE_HALF_LIFE: float = 72.0  # 3 days

# Skip decay for items activated/reinforced within this many hours
DECAY_GRACE_HOURS: float = 1.0


class DecayEngine:
    """Applies time-based decay to concepts and relations.

    Implements the ``DecayPolicy`` Protocol from ``world0.core``.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def decay_concepts(self) -> list[str]:
        """Apply time decay to all concept confidences.

        Returns list of concept ids that fell below the fading threshold.
        """
        newly_fading: list[str] = []

        for node in self._concepts.all():
            if node.maturity == Maturity.FADING and node.confidence <= 0.0:
                continue

            hours = node.hours_since_activation()

            # Skip recently activated concepts
            if hours < DECAY_GRACE_HOURS:
                continue

            half_life = CONCEPT_HALF_LIFE.get(node.maturity, 168.0)
            decay_factor = math.pow(0.5, hours / half_life)

            node.confidence = max(0.0, node.confidence * decay_factor)
            self._concepts.mark_dirty(node.id)

            if node.confidence < 0.05 and node.maturity != Maturity.FADING:
                node.maturity = Maturity.FADING
                newly_fading.append(node.id)

        return newly_fading

    def decay_relations(self) -> list[str]:
        """Apply time decay to all relation weights.

        Returns list of relation ids that fell below threshold.
        """
        weak_relations: list[str] = []

        for edge in self._relations.all():
            hours = edge.hours_since_reinforced()

            # Skip recently reinforced relations
            if hours < DECAY_GRACE_HOURS:
                continue

            # More reinforced relations decay slower
            half_life = RELATION_BASE_HALF_LIFE * (
                1.0 + edge.reinforcement_count * 0.5
            )
            decay_factor = math.pow(0.5, hours / half_life)

            edge.weight = max(0.0, edge.weight * decay_factor)
            edge.confidence = max(0.0, edge.confidence * decay_factor)
            edge.probability = edge.confidence
            self._relations.mark_dirty(edge.id)

            if edge.weight < 0.02:
                weak_relations.append(edge.id)

        return weak_relations

    def prune_concepts(self, threshold: float = 0.02) -> list[str]:
        """Remove concepts that have decayed beyond recovery (batch)."""
        to_prune = [
            n.id
            for n in self._concepts.all()
            if n.maturity == Maturity.FADING and n.confidence < threshold
        ]
        for cid in to_prune:
            self._relations.remove_for_concept(cid)
            self._concepts.remove(cid)
        return to_prune

    def prune_relations(self, threshold: float = 0.02) -> list[str]:
        """Remove relations that have decayed beyond recovery (batch)."""
        to_prune = [
            e.id for e in self._relations.all() if e.weight < threshold
        ]
        for rid in to_prune:
            self._relations.remove(rid)
        return to_prune
