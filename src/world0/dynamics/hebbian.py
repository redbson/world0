"""Hebbian learning — "concepts that fire together, wire together."

When multiple concepts co-occur in a single observation, their connections
are automatically strengthened (or created if they don't exist).
This is the primary mechanism by which relations are *discovered*.

Optimization: a co-occurrence threshold prevents O(n²) relation explosion.
Only concept pairs that have co-occurred >= COOCCURRENCE_THRESHOLD times
actually produce a RelationEdge. Below that, only a counter is incremented.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

from world0.schemas.relation import RelationType

if TYPE_CHECKING:
    from world0.core import RelationStore

# Minimum co-occurrence count before a Hebbian relation is created.
# Prevents noise relations from a single shared observation.
COOCCURRENCE_THRESHOLD: int = 2

# Maximum concept pairs to process per learn() call.
# When an observation contains many concepts, only the first MAX_PAIRS
# pairs (sorted by ID for determinism) are considered.
MAX_PAIRS: int = 30


class HebbianEngine:
    """Implements Hebbian co-activation learning for relation discovery.

    Implements the ``HebbianLearner`` Protocol from ``world0.core``.
    """

    def __init__(self, relations: "RelationStore") -> None:
        self._relations = relations
        # Tracks co-occurrence counts for pairs that don't yet have a relation.
        # Key: frozenset({id_a, id_b}), Value: count
        self._cooccurrence: dict[frozenset[str], int] = defaultdict(int)

    def learn(
        self,
        concept_ids: list[str],
        *,
        provenance: str = "",
    ) -> list[str]:
        """Process co-occurring concepts: strengthen or create relations.

        For every pair of concepts in the list:
          - If a relation exists → reinforce it
          - If no relation exists and co-occurrence < threshold → increment counter
          - If no relation exists and co-occurrence >= threshold → create RELATED_TO

        Returns list of relation ids that were created (not reinforced).
        """
        new_relation_ids: list[str] = []

        pairs = list(combinations(concept_ids, 2))
        if len(pairs) > MAX_PAIRS:
            pairs = sorted(pairs)[:MAX_PAIRS]

        for id_a, id_b in pairs:
            existing = self._relations.find_any_between(id_a, id_b)
            if existing:
                for rel in existing:
                    self._relations.reinforce(rel.id, provenance=provenance)
            else:
                key = frozenset((id_a, id_b))
                self._cooccurrence[key] += 1
                if self._cooccurrence[key] >= COOCCURRENCE_THRESHOLD:
                    edge, is_new = self._relations.discover(
                        id_a,
                        id_b,
                        RelationType.RELATED_TO,
                        provenance=provenance,
                        is_explicit=False,
                    )
                    if is_new:
                        new_relation_ids.append(edge.id)
                    # Clear counter once relation is created
                    del self._cooccurrence[key]

        return new_relation_ids

    @property
    def pending_pairs(self) -> int:
        """Number of concept pairs awaiting threshold before relation creation."""
        return len(self._cooccurrence)
