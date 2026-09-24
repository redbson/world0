"""Hebbian learning — "concepts that fire together, wire together."

When multiple concepts co-occur in a single observation, their connections
are automatically strengthened (or created if they don't exist).
This is the primary mechanism by which relations are *discovered*.

Optimization: a co-occurrence threshold prevents O(n²) relation explosion.
Only concept pairs that have co-occurred >= COOCCURRENCE_THRESHOLD times
actually produce a RelationEdge. Below that, only a counter is incremented.

The pending counters are part of the world's learning state: ``snapshot``
/ ``restore`` let the owning ``World`` persist them alongside the rest of
its cross-cycle state, so a pair first seen in one session and again in
the next still crosses the threshold.  Without this, every restart would
silently reset co-occurrence learning.
"""

from __future__ import annotations

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
# pairs in *observation order* are considered.  Extraction lists concepts
# by salience, so this keeps the pairs around the most salient concepts
# rather than the pairs whose ids happen to sort first.
MAX_PAIRS: int = 30

# Upper bound on pending (sub-threshold) pairs kept in memory and in the
# persisted snapshot.  Oldest pairs are evicted first.
MAX_PENDING_PAIRS: int = 50_000

_SNAPSHOT_SEPARATOR = "|"


class HebbianEngine:
    """Implements Hebbian co-activation learning for relation discovery.

    Implements the ``HebbianLearner`` Protocol from ``world0.core``.
    """

    def __init__(self, relations: "RelationStore") -> None:
        self._relations = relations
        # Tracks co-occurrence counts for pairs that don't yet have a relation.
        # Key: frozenset({id_a, id_b}), Value: count.  Insertion-ordered so
        # eviction under MAX_PENDING_PAIRS drops the oldest pairs first.
        self._cooccurrence: dict[frozenset[str], int] = {}

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
          - If no relation exists and co-occurrence >= threshold → create PARALLEL

        Returns list of relation ids that were created (not reinforced).
        """
        new_relation_ids: list[str] = []

        pairs = list(combinations(dict.fromkeys(concept_ids), 2))
        if len(pairs) > MAX_PAIRS:
            pairs = pairs[:MAX_PAIRS]

        for id_a, id_b in pairs:
            existing = self._relations.find_any_between(id_a, id_b)
            if existing:
                for rel in existing:
                    self._relations.reinforce(rel.id, provenance=provenance)
            else:
                key = frozenset((id_a, id_b))
                count = self._cooccurrence.get(key, 0) + 1
                if count >= COOCCURRENCE_THRESHOLD:
                    edge, is_new = self._relations.discover(
                        id_a,
                        id_b,
                        RelationType.PARALLEL,
                        provenance=provenance,
                        is_explicit=False,
                    )
                    if is_new:
                        new_relation_ids.append(edge.id)
                    # Clear counter once relation is created
                    self._cooccurrence.pop(key, None)
                else:
                    self._cooccurrence[key] = count

        self._evict_overflow()
        return new_relation_ids

    @property
    def pending_pairs(self) -> int:
        """Number of concept pairs awaiting threshold before relation creation."""
        return len(self._cooccurrence)

    # ── persistence of learning state ────────────────────────────────

    def snapshot(self) -> dict[str, int]:
        """JSON-serialisable view of the pending co-occurrence counters."""
        return {
            _SNAPSHOT_SEPARATOR.join(sorted(key)): count
            for key, count in self._cooccurrence.items()
        }

    def restore(self, snapshot: dict[str, int] | None) -> None:
        """Replace the pending counters with a previously saved snapshot."""
        self._cooccurrence.clear()
        if not snapshot:
            return
        for joined, count in snapshot.items():
            parts = joined.split(_SNAPSHOT_SEPARATOR)
            if len(parts) != 2 or not all(parts):
                continue
            try:
                value = int(count)
            except (TypeError, ValueError):
                continue
            if value > 0:
                self._cooccurrence[frozenset(parts)] = value
        self._evict_overflow()

    def forget_concept(self, concept_id: str) -> int:
        """Drop pending pairs that reference a removed concept."""
        stale = [key for key in self._cooccurrence if concept_id in key]
        for key in stale:
            del self._cooccurrence[key]
        return len(stale)

    def _evict_overflow(self) -> None:
        overflow = len(self._cooccurrence) - MAX_PENDING_PAIRS
        if overflow <= 0:
            return
        for key in list(self._cooccurrence)[:overflow]:
            del self._cooccurrence[key]
