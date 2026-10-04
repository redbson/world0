"""Lifecycle management — maturity promotion rules.

Maturity is a statement about *durability* — has this concept earned its
place — so every promotion gate is a function of quantities that only
move at events (an activation, a disconfirmation, a connection) and that
the passage of time cannot revoke: confirmation count ``n``, spaced
recurrence ``ρ``, disconfirmation balance, evidence ``e(n, d)``.  The
decayed ``confidence`` is a statement about *salience* and may appear
only where the event itself is the evidence (a dense burst that just
drove confidence up), never as the arbiter of a concept that is used
sparsely.

Promotion rules (either gate suffices)::

  embryonic → developing:
      dense    n ≥ 3            and confidence ≥ 0.3
      spaced   ρ ≥ 3            and e(n, d) ≥ 0.15          ("tentative")
  developing → established:
      dense    n ≥ 10, ρ ≥ 3    and confidence ≥ 0.6
      spaced   ρ ≥ 10           and e(n, d) ≥ 0.5           ("well evidenced")
                                and balance(n, d) ≥ 0.8
  established → core:
      n ≥ 30 and live connections ≥ dynamic_threshold

Why these gates (docs/paper §3.3):

* ``ρ`` counts activations at least 24 ticks apart, so thirty mentions in
  one burst count once (twice at most when the burst is longer than a
  window); every path to ESTABLISHED needs spacing (the dense path a
  minimum of three windows, the spaced path ten).  Intensity alone is not
  durability.
* ``e(n, d) ≥ 0.5`` is exactly the metacognition layer's *well evidenced*
  line (``projection.metacognition.WELL_EVIDENCED``): a concept the Agent
  is shown as well evidenced is not still labelled DEVELOPING, and a
  concept labelled ESTABLISHED is one the Agent is told to trust (for
  spaced use: a burst can be well evidenced and still DEVELOPING, since
  its ρ is too low).  It is reached after 12 confirmations at any cadence
  up to the prune grace (720 observations), where the old confidence gate
  needed 324 at a 720-observation cadence — the confidence equilibrium of
  a sparse cadence is capped by the evidence floor, which is a fact about
  *decay*, not about whether the concept is real.  Beyond the grace the
  concept (n ≤ 6 keeps its floor under the fading line) is forgotten as
  noise between uses, in every world.
* Records with ``ρ == 0`` predate recurrence tracking; the dense gate does
  not ask them for spacing.
* ``balance ≥ 0.8`` keeps contested concepts out: confirmations must
  outnumber disconfirmations about four to one.

The ESTABLISHED → CORE connection threshold is dynamic::

  required_connections = max(MIN_CORE_CONNECTIONS,
                             BASE_CORE_CONNECTIONS - (n - 30) // ACTIVATION_REDUCTION_STEP)

Heavily activated concepts need fewer connections, acknowledging that
frequency of use is itself evidence of centrality; the minimum keeps
isolated concepts from reaching CORE.  A connection counts while its
(settled) weight is at least the prune threshold and the concept at the
other end is not expired (``concept_expired``) — so whether a reflect has
already pruned a dead edge or a dead neighbour does not change the count.

**When eligibility is evaluated.**  A promotion changes the half-life
used for every later decay interval, so *when* it is applied matters as
much as *whether*.  The gates only change at events, hence they are
evaluated at the events themselves:

* after every activation (``ConceptManager.reinforce``),
* on every connection event — relation created, reinforced — for both
  endpoints (``RelationManager``), after settling the decay owed under the
  old half-life,

each climbing the ladder to a fixpoint.  ``evaluate()`` (reflect) is only
a catch-up for state changed outside these paths (merge, imported or
hand-edited records); for a world driven through ``ingest`` it changes
nothing.  Consequently the maturity trajectory — and with it the half-life
history — is a function of the observation stream alone.  The same holds
for existence: a concept or relation that is already *dead* (settled
state past recovery, ``concept_expired`` / ``relation_dead``) is treated as
absent at every event, so physical deletion at reflect is garbage
collection and cannot change what a later mention meets (docs/paper
Definition 3.6, Theorem 3.7).

Demotion:
  any → fading: exactly at the instant confidence crosses 0.05
                (``dynamics.decay.settle_concept``)
  fading → developing / embryonic: ``ConceptNode.activate`` (re-activation)
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

from world0.dynamics.decay import (
    RELATION_PRUNE_THRESHOLD,
    concept_expired,
    projected_relation_weight,
    settle_concept,
    settle_relation,
)
from world0.schemas.clock import CognitiveClock, wall_now
from world0.schemas.concept import ConceptNode, Maturity

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# ── ESTABLISHED → CORE promotion parameters ─────────────────────────
CORE_MIN_ACTIVATIONS: int = 30
BASE_CORE_CONNECTIONS: int = 5       # default connection requirement
MIN_CORE_CONNECTIONS: int = 2        # absolute minimum connections
ACTIVATION_REDUCTION_STEP: int = 20  # every N extra activations reduces
                                      # connection requirement by 1

# ── EMBRYONIC → DEVELOPING ──────────────────────────────────────────
DENSE_DEVELOPING_ACTIVATIONS: int = 3
DENSE_DEVELOPING_CONFIDENCE: float = 0.3
RECURRENCE_FOR_DEVELOPING: int = 3
# Mirrors ``projection.metacognition.TENTATIVE_EVIDENCE`` (checked by a
# test; ``dynamics`` must not import the projection layer).
SPACED_DEVELOPING_EVIDENCE: float = 0.15

# ── DEVELOPING → ESTABLISHED ────────────────────────────────────────
DENSE_ESTABLISHED_ACTIVATIONS: int = 10
DENSE_ESTABLISHED_RECURRENCE: int = 3
DENSE_ESTABLISHED_CONFIDENCE: float = 0.6
RECURRENCE_FOR_ESTABLISHED: int = 10
# Mirrors ``projection.metacognition.WELL_EVIDENCED``.
SPACED_ESTABLISHED_EVIDENCE: float = 0.5
SPACED_ESTABLISHED_BALANCE: float = 0.8

_ORDER = {
    Maturity.FADING: 0,
    Maturity.EMBRYONIC: 1,
    Maturity.DEVELOPING: 2,
    Maturity.ESTABLISHED: 3,
    Maturity.CORE: 4,
}


# ── Long-term memory: the consolidation gate ─────────────────────────
# A concept enters long-term memory (``ConceptNode.consolidated_tick``;
# the slow curve of ``dynamics/decay``) when it is well evidenced
# (e(n, d) ≥ LONG_TERM_EVIDENCE — metacognition's WELL_EVIDENCED line, so
# n ≥ 10 confirmations with none against), has recurred in at least
# LONG_TERM_RECURRENCE spaced windows (uses ≥ 24 observations apart: a
# burst, however large, does not consolidate — spacing is the signal of
# durability), and is not contested (β ≥ LONG_TERM_BALANCE).  All three
# are event-time counters, so the gate turns true only at an activation
# (Prop. 3.8).  It leaves the mode when a disconfirmation takes β below
# the balance gate, or when it finally fades (``ConceptNode.long_term``).
# ``LONG_TERM_MEMORY_DEFAULT`` is what ``World(long_term_memory=None)``
# means; with the mode off nothing consolidates (concepts consolidated
# earlier keep their curve).
LONG_TERM_EVIDENCE: float = SPACED_ESTABLISHED_EVIDENCE
LONG_TERM_RECURRENCE: int = 5
LONG_TERM_BALANCE: float = SPACED_ESTABLISHED_BALANCE
# The environment variable WORLD0_LONG_TERM_MEMORY=0 turns the default off
# (used by the benchmark to measure the mode; a World's own argument wins).
LONG_TERM_MEMORY_DEFAULT: bool = os.environ.get("WORLD0_LONG_TERM_MEMORY", "1").strip().lower() not in (
    "0", "false", "off", "no",
)


def consolidation_gate(node: ConceptNode) -> bool:
    """Whether ``node`` meets the long-term memory gate now (event-time counters only)."""
    return (
        node.maturity != Maturity.FADING
        and node.recurrence_count >= LONG_TERM_RECURRENCE
        and node.evidence() >= LONG_TERM_EVIDENCE
        and node.evidence_balance() >= LONG_TERM_BALANCE
    )


def core_connections_required(activation_count: int) -> int:
    """Connections an ESTABLISHED concept needs for CORE at ``n`` activations."""
    reduction = max(0, activation_count - CORE_MIN_ACTIVATIONS) // ACTIVATION_REDUCTION_STEP
    return max(MIN_CORE_CONNECTIONS, BASE_CORE_CONNECTIONS - reduction)


class LifecycleEngine:
    """Evaluates and applies maturity transitions for concepts.

    Implements the ``LifecyclePolicy`` Protocol from ``world0.core``.
    ``clock`` is optional: without it connections are counted without the
    weight-liveness filter and the era-time coordinate is unavailable.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
        clock: CognitiveClock | None = None,
        long_term_memory: bool | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._clock = clock
        self.long_term_memory = (
            LONG_TERM_MEMORY_DEFAULT if long_term_memory is None else bool(long_term_memory)
        )
        # Promotions applied at events since the last ``evaluate()``, so
        # a reflect still reports what rose during its consolidation period.
        self._pending_promoted: dict[str, None] = {}
        # Consolidations applied at events and not yet reported (``ingest``
        # drains them into ``IngestResult``; ``evaluate`` reports the rest).
        self._pending_consolidated: dict[str, None] = {}
        self.last_consolidated: list[str] = []

    # ── event-time evaluation ────────────────────────────────────────

    def promote(self, node: ConceptNode) -> bool:
        """Climb the ladder to a fixpoint; returns True when maturity rose.

        Called at events (see module docstring).  Idempotent: the gates
        are monotone in the counters, so a second call changes nothing.
        """
        risen = False
        while True:
            new_maturity = self._evaluate_one(node)
            if new_maturity is None or new_maturity == node.maturity:
                return risen
            self._concepts.update_maturity(node.id, new_maturity)
            self._pending_promoted[node.id] = None
            risen = True

    def on_activation(self, node: ConceptNode) -> None:
        """Hook: a concept was just activated (ConceptManager.reinforce)."""
        self.promote(node)
        self.consolidate(node)
        # A revival cleared the flag, or a consolidation set it: the
        # claims around the concept follow (settled under the old profile).
        self._sync_edges(node)

    def on_weaken(self, node: ConceptNode) -> None:
        """Hook: a concept was just disconfirmed (ConceptManager.weaken).

        A long-term concept argued below the balance gate leaves long-term
        memory; the decay owed under the slow curve was settled by the
        manager before the disconfirmation."""
        if node.consolidated_tick is not None and node.evidence_balance() < LONG_TERM_BALANCE:
            node.consolidated_tick = None
            self._concepts.mark_dirty(node.id)
            self._sync_edges(node)

    # ── long-term memory ─────────────────────────────────────────────

    def consolidate(self, node: ConceptNode) -> bool:
        """Move ``node`` into long-term memory if it meets the gate; returns
        True when it just did.  Idempotent; a no-op with the mode off."""
        if not self.long_term_memory or node.consolidated_tick is not None:
            return False
        if not consolidation_gate(node):
            return False
        node.consolidated_tick = self._clock.tick if self._clock is not None else node.last_activated_tick
        self._concepts.mark_dirty(node.id)
        self._pending_consolidated[node.id] = None
        return True

    def _sync_edges(self, node: ConceptNode) -> int:
        """Re-judge ``long_term`` on the explicit claims around ``node``: true
        when both endpoints are in long-term memory.  A claim whose profile
        changes is settled under the old one first (the same discipline as
        a concept's half-life change at promotion), so the floor era in
        force during any gap is the one at the gap's start.  Returns how
        many edges changed."""
        changed = 0
        for edge in self._relations.for_concept(node.id):
            if not edge.is_explicit:
                continue
            other_id = edge.other_end(node.id)
            other = self._concepts.get(other_id) if other_id else None
            want = bool(node.long_term and other is not None and other.long_term)
            if want == edge.long_term:
                continue
            if self._clock is not None:
                settle_relation(edge, self._clock.tick)
            edge.long_term = want
            self._relations.mark_dirty(edge.id)
            changed += 1
        return changed

    def pop_consolidated(self) -> list[str]:
        """Ids consolidated at events since the last call (for ``IngestResult``)."""
        out = [cid for cid in self._pending_consolidated if self._concepts.get(cid) is not None]
        self._pending_consolidated.clear()
        return out

    def on_connection(self, *concept_ids: str) -> None:
        """Hook: a relation touching these concepts was created / reinforced.

        Only ESTABLISHED concepts with enough activations can be affected;
        the cheap pre-check keeps the hook free for everything else.  The
        decay owed under the old half-life is settled before the half-life
        can change.
        """
        for cid in concept_ids:
            node = self._concepts.get(cid)
            if (
                node is None
                or node.maturity != Maturity.ESTABLISHED
                or node.activation_count < CORE_MIN_ACTIVATIONS
            ):
                continue
            if self._connections(node) >= core_connections_required(node.activation_count):
                self._settle(node)
                self.promote(node)

    def _settle(self, node: ConceptNode) -> None:
        if self._clock is None:
            return
        settle_concept(node, self._clock.tick)
        self._concepts.mark_dirty(node.id)

    # ── reflect-time catch-up ────────────────────────────────────────

    def evaluate(self) -> tuple[list[str], list[str]]:
        """Catch-up evaluation of all concepts (reflect).

        Returns (promoted_ids, demoted_ids).  For a world driven through
        ``ingest`` every eligible promotion was already applied at its
        event, so nothing new happens here; the promoted ids are those
        that rose since the previous ``evaluate()`` (at events or now).
        The whole ladder is climbed in one call.  Nothing is ever demoted
        here: FADING is entered by decay, exactly.
        """
        for node in self._concepts.all():
            self.promote(node)
            self.consolidate(node)
            self._sync_edges(node)
        promoted = [
            cid for cid in self._pending_promoted if self._concepts.get(cid) is not None
        ]
        self._pending_promoted.clear()
        self.last_consolidated = self.pop_consolidated()
        return promoted, []

    # ── gates ────────────────────────────────────────────────────────

    def _connections(self, node: ConceptNode) -> int:
        """Live connections: alive edges to concepts that are not expired.

        An edge is alive while its settled weight is at least the prune
        line; a neighbour is alive until it is past recovery
        (``concept_expired``).  Both are pure predicates of settled state,
        so the count is the same whether a reflect has already deleted the
        dead edges and neighbours or not.
        """
        edges = self._relations.for_concept(node.id)
        if self._clock is None:
            return len(edges)
        now, tick = wall_now(), self._clock.tick
        live: set[str] = set()
        for edge in edges:
            if edge.is_retracted:
                continue
            if projected_relation_weight(edge, tick, now) < RELATION_PRUNE_THRESHOLD:
                continue
            other_id = edge.other_end(node.id)
            other = self._concepts.get(other_id) if other_id else None
            if other is None or concept_expired(other, tick, now):
                continue
            live.add(other_id)  # a neighbour counts once, however many claims link it
        return len(live)

    def _evaluate_one(self, node: ConceptNode) -> Maturity | None:
        """The maturity ``node`` is eligible to rise to next, if any."""
        if node.maturity == Maturity.EMBRYONIC:
            if (
                node.activation_count >= DENSE_DEVELOPING_ACTIVATIONS
                and node.confidence >= DENSE_DEVELOPING_CONFIDENCE
            ):
                return Maturity.DEVELOPING
            # Spaced recurrence is evidence of durability even when the
            # confidence equilibrium of a sparse cadence stays low.
            if (
                node.recurrence_count >= RECURRENCE_FOR_DEVELOPING
                and node.evidence() >= SPACED_DEVELOPING_EVIDENCE
            ):
                return Maturity.DEVELOPING
            return None

        if node.maturity == Maturity.DEVELOPING:
            # ρ == 0 marks a record that predates recurrence tracking
            # (or was built without a tick); it cannot be judged on
            # spacing, so it keeps the original dense rule.
            spaced_enough = (
                node.recurrence_count == 0
                or node.recurrence_count >= DENSE_ESTABLISHED_RECURRENCE
            )
            if (
                node.activation_count >= DENSE_ESTABLISHED_ACTIVATIONS
                and spaced_enough
                and node.confidence >= DENSE_ESTABLISHED_CONFIDENCE
            ):
                return Maturity.ESTABLISHED
            if (
                node.recurrence_count >= RECURRENCE_FOR_ESTABLISHED
                and node.evidence() >= SPACED_ESTABLISHED_EVIDENCE
                and node.evidence_balance() >= SPACED_ESTABLISHED_BALANCE
            ):
                return Maturity.ESTABLISHED
            return None

        if node.maturity == Maturity.ESTABLISHED:
            if node.activation_count >= CORE_MIN_ACTIVATIONS:
                if self._connections(node) >= core_connections_required(
                    node.activation_count
                ):
                    return Maturity.CORE
            return None

        return None

    @staticmethod
    def _is_promotion(old: Maturity, new: Maturity) -> bool:
        return _ORDER.get(new, 0) > _ORDER.get(old, 0)
