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

import math
from itertools import combinations
from typing import TYPE_CHECKING

from world0.schemas.relation import RelationType
from world0.schemas.types import PredictionError

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

# ── Prediction (docs/mc/04-prediction.md) ─────────────────────────────
# A concept predicts a linked companion b once it has been mentioned at
# least PREDICTION_MIN_SUPPORT times and P(b | a) = co-occurrences /
# mentions(a) reaches PREDICTION_MIN_PROBABILITY.  Two concepts that each
# have that support and were never seen together form a novel pair.
PREDICTION_MIN_SUPPORT: int = 5
PREDICTION_MIN_PROBABILITY: float = 0.5


def _pair_key(id_a: str, id_b: str) -> str:
    """Canonical, order-independent key for a concept pair."""
    return _SNAPSHOT_SEPARATOR.join(sorted((id_a, id_b)))


class HebbianEngine:
    """Implements Hebbian co-activation learning for relation discovery.

    Implements the ``HebbianLearner`` Protocol from ``world0.core``.
    """

    def __init__(self, relations: "RelationStore") -> None:
        self._relations = relations
        # Tracks co-occurrence counts for pairs that don't yet have a relation.
        # Key: the canonical pair key ``"<id_a>|<id_b>"`` (ids sorted), so
        # the snapshot is a plain copy instead of an O(pairs) rebuild.
        # Insertion-ordered so eviction under MAX_PENDING_PAIRS drops the
        # oldest pairs first.
        self._cooccurrence: dict[str, int] = {}
        # Association statistics: observations seen and mentions per
        # concept (unique per observation).  Persisted with the pending
        # counters so the association gate survives restarts.
        self._observations: int = 0
        self._mentions: dict[str, int] = {}
        # Co-occurrence counts of pairs that *do* have a relation (the
        # pending counter is dropped when a pair is linked).  Bounded by
        # the number of related pairs; the basis for predictions.
        self._linked: dict[str, int] = {}

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
            # An edge that is already dead is absent, not revivable
            # (RelationManager.reap_dead_between).
            self._relations.reap_dead_between(id_a, id_b)
            existing = self._relations.find_any_between(id_a, id_b)
            key = _pair_key(id_a, id_b)
            if existing:
                for rel in existing:
                    self._relations.reinforce(rel.id, provenance=provenance)
                self._linked[key] = (
                    self._linked.get(key, 0) + 1 + self._cooccurrence.pop(key, 0)
                )
            else:
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
                    # Move the counter to the linked statistics once the
                    # relation exists.
                    self._cooccurrence.pop(key, None)
                    self._linked[key] = count
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

    def cooccurrences(self, id_a: str, id_b: str) -> int:
        """Observations in which both concepts appeared (linked or not)."""
        key = _pair_key(id_a, id_b)
        return self._linked.get(key, 0) + self._cooccurrence.get(key, 0)

    def prediction_error(self, concept_ids: list[str]) -> PredictionError:
        """Score an observation against learned co-occurrence.

        Call *before* ``learn()`` for the same observation: a prediction
        must not see the evidence it is judged on.  Both errors are
        measured *relative to what the model itself expects*, so the
        ordinary variability of a loose cluster is not surprise:

        - **missing**: each predicted companion ``b`` of ``a`` appears with
          probability ``p = P(b | a)``; its absence costs ``p``.  The
          model already expects a missed mass of ``Σ p(1 − p)``, so the
          error is ``(missed − expected) / (Σ p − expected)``, clipped to
          ``[0, 1]`` — absent 0.6-companions are routine, an absent
          1.0-companion is not;
        - **novelty**: a pair of well-known concepts never seen together
          counts by ``1 − exp(−λ)``, ``λ = n_a · n_b / N`` — the chance
          they would already have met if mentions were independent.  Two
          concepts that each fill half the history and never met are
          surprising together; two that each appear rarely are not.
        """
        observed = list(dict.fromkeys(concept_ids))
        present = set(observed)
        error = PredictionError()
        predicted_mass = 0.0
        expected_missed = 0.0
        missed_mass = 0.0
        for cid in observed:
            support = self._mentions.get(cid, 0)
            if support < PREDICTION_MIN_SUPPORT:
                continue
            # One prediction per companion, however many relations (e.g.
            # opposing explicit claims) link the pair (docs/paper §9).
            companions: set[str] = set()
            for rel in self._relations.for_concept(cid):
                other = rel.other_end(cid)
                if other is None or other == cid or other in companions:
                    continue
                companions.add(other)
                p = min(1.0, self._linked.get(_pair_key(cid, other), 0) / support)
                if p < PREDICTION_MIN_PROBABILITY:
                    continue
                predicted_mass += p
                expected_missed += p * (1.0 - p)
                if other not in present:
                    missed_mass += p
                    entry = (cid, other, round(p, 4))
                    if entry not in error.missing:
                        error.missing.append(entry)
        # Relation-index order depends on load order; sort so the report
        # is identical across processes.
        error.missing.sort(key=lambda m: (-m[2], m[0], m[1]))
        span = predicted_mass - expected_missed
        if span > 1e-9:
            error.missing_ratio = round(
                min(1.0, max(0.0, (missed_mass - expected_missed) / span)), 4
            )
        known = [
            cid for cid in observed
            if self._mentions.get(cid, 0) >= PREDICTION_MIN_SUPPORT
        ]
        known_pairs = list(combinations(known, 2))
        weight = 0.0
        n_obs = max(1, self._observations)
        for id_a, id_b in known_pairs:
            if self.cooccurrences(id_a, id_b) == 0:
                lam = self._mentions[id_a] * self._mentions[id_b] / n_obs
                weight += 1.0 - math.exp(-lam)
                error.novel_pairs.append((id_a, id_b))
        error.novelty = round(weight / len(known_pairs), 4) if known_pairs else 0.0
        error.surprise = max(error.missing_ratio, error.novelty)
        return error

    def association(self, id_a: str, id_b: str) -> float:
        """Current association of a still-pending pair (0.0 if none)."""
        count = self._cooccurrence.get(_pair_key(id_a, id_b), 0)
        return self._association(id_a, id_b, count) if count else 0.0

    @property
    def observations(self) -> int:
        """Observations processed by ``learn()`` (for association stats)."""
        return self._observations

    def mentions(self, concept_id: str) -> int:
        """Observations in which ``concept_id`` was mentioned."""
        return self._mentions.get(concept_id, 0)

    @property
    def tracked_concepts(self) -> int:
        """Concepts with a mention count (size of the statistics)."""
        return len(self._mentions)

    @property
    def pending_pairs(self) -> int:
        """Number of concept pairs awaiting threshold before relation creation."""
        return len(self._cooccurrence)

    # ── persistence of learning state ────────────────────────────────

    def snapshot(self) -> dict[str, int]:
        """JSON-serialisable view of the pending co-occurrence counters."""
        return dict(self._cooccurrence)

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
                self._cooccurrence[_pair_key(*parts)] = value
        self._evict_overflow()

    def forget_concept(self, concept_id: str) -> int:
        """Drop pending pairs (and mention stats) of a removed concept."""
        stale = [
            key
            for key in self._cooccurrence
            if concept_id in key.split(_SNAPSHOT_SEPARATOR)
        ]
        for key in stale:
            del self._cooccurrence[key]
        for key in [k for k in self._linked if concept_id in k.split(_SNAPSHOT_SEPARATOR)]:
            del self._linked[key]
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
            # Exact count when tracked (``_linked``); otherwise the lower
            # bound for an edge created at the COOCCURRENCE_THRESHOLD-th
            # co-occurrence and reinforced on each later one.  The bound
            # undercounts edges whose creation the association gate
            # delayed, which biased revalidation toward removal
            # (docs/paper, Proposition 5.4).
            key = _pair_key(edge.source_id, edge.target_id)
            cooccurrences = max(
                edge.reinforcement_count + COOCCURRENCE_THRESHOLD,
                self._linked.get(key, 0),
            )
            if self._association(edge.source_id, edge.target_id, cooccurrences) < cutoff:
                if self._relations.remove(edge.id):
                    removed.append(edge.id)
                    # Back to the pending counter: the pair is unlinked
                    # again but its co-occurrence history is real.
                    count = self._linked.pop(key, 0)
                    if count:
                        self._cooccurrence[key] = count
        return removed

    # ── persistence of association statistics ────────────────────────

    def stats_snapshot(self) -> dict:
        """JSON-serialisable observation / mention counts."""
        return {
            "observations": self._observations,
            "mentions": dict(self._mentions),
            "linked": dict(self._linked),
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
        self._linked.clear()
        if not snapshot or not isinstance(snapshot, dict):
            return
        linked = snapshot.get("linked") or {}
        if isinstance(linked, dict):
            for key, count in linked.items():
                try:
                    value = int(count)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    self._linked[str(key)] = value
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
