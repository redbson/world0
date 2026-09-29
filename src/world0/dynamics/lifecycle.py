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

* ``ρ`` counts distinct 24-tick windows, so thirty mentions in one burst
  count once; every path to ESTABLISHED needs spacing (the dense path a
  minimum of three windows, the spaced path ten).  Intensity alone is not
  durability.
* ``e(n, d) ≥ 0.5`` is exactly the metacognition layer's *well evidenced*
  line (``projection.metacognition.WELL_EVIDENCED``): a concept the Agent
  is shown as well evidenced is not still labelled DEVELOPING, and a
  concept labelled ESTABLISHED is one the Agent is told to trust.  It is
  reached after 12 confirmations at any cadence, where the old confidence
  gate needed 324 at a 720-observation cadence — the confidence
  equilibrium of a sparse cadence is capped by the evidence floor, which
  is a fact about *decay*, not about whether the concept is real.
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
(settled) weight is at least the prune threshold — so whether a reflect
has already pruned a dead edge does not change the count.

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
history — is a function of the observation stream alone.

Demotion:
  any → fading: exactly at the instant confidence crosses 0.05
                (``dynamics.decay.settle_concept``)
  fading → developing / embryonic: ``ConceptNode.activate`` (re-activation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.dynamics.decay import (
    RELATION_PRUNE_THRESHOLD,
    projected_relation_weight,
    settle_concept,
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
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._clock = clock
        # Promotions applied at events since the last ``evaluate()``, so
        # a reflect still reports what rose during its consolidation period.
        self._pending_promoted: dict[str, None] = {}

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
        promoted = [
            cid for cid in self._pending_promoted if self._concepts.get(cid) is not None
        ]
        self._pending_promoted.clear()
        return promoted, []

    # ── gates ────────────────────────────────────────────────────────

    def _connections(self, node: ConceptNode) -> int:
        """Live connections: edges whose settled weight is above the prune line."""
        edges = self._relations.for_concept(node.id)
        if self._clock is None:
            return len(edges)
        now, tick = wall_now(), self._clock.tick
        return sum(
            1
            for edge in edges
            if projected_relation_weight(edge, tick, now) >= RELATION_PRUNE_THRESHOLD
        )

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
