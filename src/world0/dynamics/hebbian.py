"""Hebbian learning — "concepts that fire together, wire together."

When multiple concepts co-occur in a single observation, their connections
are automatically strengthened (or created if they don't exist).
This is the primary mechanism by which relations are *discovered*.

Two gates keep discovery from degenerating into a clique:

1. **Count.**  A pair must co-occur at least ``COOCCURRENCE_THRESHOLD``
   times; below that only a counter is incremented.
2. **Association.**  Co-occurring twice is not evidence of a relation
   when both concepts are mentioned all the time — in a world of 60
   concepts observed six at a time every pair co-occurs by chance every
   ~100 observations, and with the count gate alone 83 % of all pairs
   were linked after 400 observations (analysis doc §7.11).  A pair is
   therefore linked only when its Jaccard association over observations
   (``co-occurrences / (mentions_a + mentions_b − co-occurrences)``) is
   at least ``HEBBIAN_MIN_ASSOCIATION``: "of the observations mentioning
   either concept, this share mentions both".  Concepts that always
   appear together score 1.0; a hub mentioned everywhere is not linked to
   a concept it merely happens to share a few observations with.

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

# Minimum Jaccard association over observations before a pair that has
# crossed COOCCURRENCE_THRESHOLD is linked.  Calibrated in
# scripts/sweep_hebbian.py: uniformly random co-mention scores ≈ 0.05–0.1,
# concepts drawn from one topic ≈ 0.4, always-together pairs 1.0.
HEBBIAN_MIN_ASSOCIATION: float = 0.2

# ``revalidate()`` (run by reflect) removes an auto-discovered generic
# edge whose association, recomputed from its reinforcement count and the
# current mention statistics, has fallen below
# HEBBIAN_MIN_ASSOCIATION × REVALIDATION_HYSTERESIS — the gap between the
# creation and removal thresholds stops a pair from flapping.  Edges are
# judged only once both concepts have been mentioned often enough
# (REVALIDATION_MIN_MENTIONS, summed) for the association to mean
# anything: two concepts seen twice, together both times, stay linked.
REVALIDATION_HYSTERESIS: float = 0.5
REVALIDATION_MIN_MENTIONS: int = 20

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
        # Association statistics: observations seen and mentions per
        # concept (unique per observation).  Persisted with the pending
        # counters so the association gate survives restarts.
        self._observations: int = 0
        self._mentions: dict[str, int] = {}

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
          - If no relation exists, co-occurrence >= threshold and the
            pair's association >= HEBBIAN_MIN_ASSOCIATION → create PARALLEL

        Returns list of relation ids that were created (not reinforced).
        """
        new_relation_ids: list[str] = []

        unique_ids = list(dict.fromkeys(concept_ids))
        if unique_ids:
            self._observations += 1
            for cid in unique_ids:
                self._mentions[cid] = self._mentions.get(cid, 0) + 1

        pairs = list(combinations(unique_ids, 2))
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
                if (
                    count >= COOCCURRENCE_THRESHOLD
                    and self._association(id_a, id_b, count)
                    >= HEBBIAN_MIN_ASSOCIATION
                ):
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

    def _association(self, id_a: str, id_b: str, cooccurrences: int) -> float:
        """Jaccard association of two concepts over observations."""
        either = (
            self._mentions.get(id_a, 0) + self._mentions.get(id_b, 0) - cooccurrences
        )
        return cooccurrences / max(either, cooccurrences, 1)

    def association(self, id_a: str, id_b: str) -> float:
        """Current association of a still-pending pair (0.0 if none)."""
        count = self._cooccurrence.get(frozenset((id_a, id_b)), 0)
        return self._association(id_a, id_b, count) if count else 0.0

    @property
    def observations(self) -> int:
        """Observations processed by ``learn()`` (for association stats)."""
        return self._observations

    def mentions(self, concept_id: str) -> int:
        """Observations in which ``concept_id`` was mentioned."""
        return self._mentions.get(concept_id, 0)

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
        """Drop pending pairs (and mention stats) of a removed concept."""
        stale = [key for key in self._cooccurrence if concept_id in key]
        for key in stale:
            del self._cooccurrence[key]
        self._mentions.pop(concept_id, None)
        return len(stale)

    def revalidate(self) -> list[str]:
        """Remove auto-discovered generic edges that no longer pass the gate.

        Early in a world's life the association gate sees thin statistics
        (two mentions, both shared → J = 1), so chance pairs get linked
        and then keep being reinforced by further chance co-occurrences.
        Reflect calls this to re-judge every non-explicit
        ``generic_relation`` edge against the accumulated statistics.
        Explicit relations and edges that were upgraded to a typed
        semantic relation are never touched.

        Returns the ids of the removed relations.
        """
        cutoff = HEBBIAN_MIN_ASSOCIATION * REVALIDATION_HYSTERESIS
        removed: list[str] = []
        for edge in list(self._relations.all()):
            if edge.is_explicit or edge.semantic_relation != "generic_relation":
                continue
            n_a = self._mentions.get(edge.source_id, 0)
            n_b = self._mentions.get(edge.target_id, 0)
            if n_a + n_b < REVALIDATION_MIN_MENTIONS:
                continue
            # Created at the COOCCURRENCE_THRESHOLD-th co-occurrence and
            # reinforced on each later one.
            cooccurrences = edge.reinforcement_count + COOCCURRENCE_THRESHOLD
            if self._association(edge.source_id, edge.target_id, cooccurrences) < cutoff:
                if self._relations.remove(edge.id):
                    removed.append(edge.id)
        return removed

    # ── persistence of association statistics ────────────────────────

    def stats_snapshot(self) -> dict:
        """JSON-serialisable observation / mention counts."""
        return {
            "observations": self._observations,
            "mentions": dict(self._mentions),
        }

    def restore_stats(self, snapshot: dict | None) -> None:
        """Replace the association statistics with a saved snapshot.

        A world persisted before association statistics existed starts
        counting from zero; until statistics accumulate the gate then
        sees ``cooccurrences ≥ mentions`` and behaves like the old
        count-only rule.
        """
        self._observations = 0
        self._mentions.clear()
        if not snapshot or not isinstance(snapshot, dict):
            return
        try:
            self._observations = max(0, int(snapshot.get("observations", 0)))
        except (TypeError, ValueError):
            self._observations = 0
        mentions = snapshot.get("mentions") or {}
        if isinstance(mentions, dict):
            for cid, count in mentions.items():
                try:
                    value = int(count)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    self._mentions[str(cid)] = value

    def _evict_overflow(self) -> None:
        overflow = len(self._cooccurrence) - MAX_PENDING_PAIRS
        if overflow <= 0:
            return
        for key in list(self._cooccurrence)[:overflow]:
            del self._cooccurrence[key]
