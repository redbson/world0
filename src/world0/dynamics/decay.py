"""Time-based decay — unused concepts and relations fade over time.

Time here is **cognitive time** (see ``schemas/clock.py``): one tick per
observation ingested, plus a slow wall-clock drift while the world is
idle.  Half-lives below are therefore expressed in observations — an
embryonic concept halves after 24 further observations that do not
mention it, not after 24 hours.

Decay rates are maturity-dependent:
  - core concepts decay very slowly
  - embryonic concepts decay fast
  - relations decay inversely with reinforcement count

Three guarantees this engine makes (analysis and probe evidence in
``docs/world0-cognitive-dynamics-analysis.md``):

1. **Idempotent in cognitive time.**  Decay is applied to the interval
   since the *last decay application* (or the last activation, whichever
   is later), never to the whole span since the last activation.  Calling
   ``reflect()`` ten times between two observations therefore leaves the
   same confidence as calling it once.

2. **Evidence-anchored (concepts).**  Repeated confirmation buys
   persistence in two ways.  The effective half-life stretches with
   activation count (as it already did for relations), and confidence
   relaxes toward an *evidence floor* instead of toward zero — an
   Ornstein–Uhlenbeck-style mean reversion.  The floor itself forgets on
   an "era" scale (thousands of observations), so nothing is immortal,
   but a concept confirmed thirty times no longer evaporates after a
   short burst of unrelated observations the way a one-off mention does.
   Disconfirmation lowers the floor through ``evidence_balance()``.

3. **Semantic probability is not time-decayed (relations).**  ``weight``
   and ``confidence`` are operational traversal strengths and do decay;
   ``probability`` is the belief that the typed relation is *correct*,
   which only evidence (reinforce / weaken / extraction) may change.

4. **Schedule-independent (lazy settlement).**  Activation and
   reinforcement move the decay reference point to "now", so the decay
   still owed for the interval before them must be applied first — or it
   is lost, and how much a concept forgets would depend on how often
   ``reflect()`` happened to run between two uses (it did: a concept
   re-used every 720 observations with no reflect in between never
   decayed at all).  ``settle_concept`` / ``settle_relation`` apply that
   owed decay; the managers call them right before every reinforcement.
   Together with guarantee 1, forgetting then depends only on cognitive
   time, never on the reflect cadence (docs/paper, Theorem 3.2).

5. **Exact FADING boundary.**  Entering FADING switches the half-life,
   so *when* a concept is judged to have crossed ``FADING_THRESHOLD``
   used to depend on when settlement happened to run (a reflect in the
   middle of a gap changed the decay rate for the rest of it).  A settle
   now splits the interval at the analytically solved crossing instant:
   before it the maturity's half-life applies, after it the FADING one.
   The crossing time is a function of the state at the start of the gap
   only, so any subdivision of the gap gives the same result (exactly,
   Proposition 3.1).  Disconfirmation (``weaken``)
   settles first and is judged at the event for the same reason.

Optimization: items touched less than one tick ago are skipped entirely,
avoiding unnecessary floating-point work.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from world0.schemas.clock import CognitiveClock, wall_now
from world0.schemas.concept import LONG_TERM_ERA_HL, SALIENCE_ERA_HL, ConceptNode, Maturity
from world0.schemas.relation import RelationEdge

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Half-life in ticks (observations) per maturity level, for a concept
# backed by a single confirming observation.  ``concept_half_life``
# stretches it with additional evidence.
CONCEPT_HALF_LIFE: dict[Maturity, float] = {
    Maturity.EMBRYONIC: 24.0,
    Maturity.DEVELOPING: 168.0,
    Maturity.ESTABLISHED: 720.0,
    Maturity.CORE: 2160.0,
    Maturity.FADING: 24.0,
}

# Base half-life for relations in ticks (modified by reinforcement count)
RELATION_BASE_HALF_LIFE: float = 72.0

# Skip decay for items touched less than this many ticks ago
DECAY_GRACE_TICKS: float = 1.0

# ── Evidence-scaled concept half-life ─────────────────────────────────
# Every confirming activation beyond the first stretches the maturity
# half-life by this fraction, capped at CONCEPT_EVIDENCE_HL_MAX_SCALE.
# Mirrors the ``1 + 0.5 × reinforcement_count`` rule relations already
# use, with a gentler slope because maturity stages already encode a
# large share of a concept's evidence history.
#
# Calibration (docs/world0-cognitive-dynamics-analysis.md §5, since revised
# by event-time promotion, §7.22 and docs/paper §3.5): a concept re-observed
# every 24 / 168 / 720 observations is ``developing`` at its 3rd use and
# ``established`` at its 12th (ticks 265 / 1849 / 7921; settled confidence
# 0.76 / 0.48 / 0.30 after 30 uses); a one-off mention fades after ~50
# observations and is deleted after 720; a concept confirmed 30× and then
# abandoned needs ~26 000 observations to fade.
CONCEPT_EVIDENCE_HL_GAIN: float = 0.20
CONCEPT_EVIDENCE_HL_MAX_SCALE: float = 8.0
# Absolute ceiling on the effective half-life so heavily used core
# concepts still forget on a bounded scale instead of becoming immortal.
CONCEPT_MAX_HALF_LIFE: float = 8760.0

# ── Long-term memory: the slow curve ──────────────────────────────────
# A consolidated concept (``ConceptNode.long_term``; gate in
# ``dynamics/lifecycle``) relaxes with this half-life whatever its maturity,
# and its evidence floor forgets on the same era (``floor_era``).  Four
# times the ceiling above: a concept consolidated at confidence 0.6 after
# 12 spaced uses and never mentioned again stays above the FADING line for
# ~145 000 observations (its floor alone holds it ~33 000), against
# ~11 000 for the same concept on the ESTABLISHED curve (verify.py §3.6).
# Once it does cross, the FADING profile applies.  Consolidation is an event
# (Prop. 3.8), so the curve in force during a gap is fixed at the gap's
# start and the settle operator stays an exact semigroup.
LONG_TERM_HALF_LIFE: float = LONG_TERM_ERA_HL

# ── Evidence floor (mean-reversion target) ────────────────────────────
#   floor = EVIDENCE_FLOOR_MAX × evidence_balance × (n / (n + K))²
#           × 0.5 ** (ticks_since_activation / EVIDENCE_FLOOR_ERA_HL)
# The squared saturation keeps the floor below FADING_THRESHOLD for
# lightly evidenced concepts (n ≲ 6) so genuine noise still fades and is
# pruned, while a concept confirmed ~30 times holds ≈ 0.20 and one
# confirmed 100 times ≈ 0.29.  The era half-life makes the floor itself
# forget.
EVIDENCE_FLOOR_MAX: float = 0.35
EVIDENCE_FLOOR_K: float = 10.0
EVIDENCE_FLOOR_ERA_HL: float = SALIENCE_ERA_HL  # one era for both halves of belief

# Confidence below which a concept is marked FADING.
FADING_THRESHOLD: float = 0.05

# ── Prune grace: fade fast, delete slowly ─────────────────────────────
# A FADING concept is deleted only once it has also been idle for this
# many observations.  Fading is reversible (a re-mention revives the
# same node with its relations, task profile and sources); deletion is
# not.  Without the grace a concept mentioned once was deleted unless
# re-mentioned within ~80 observations (embryonic half-life 24, prune at
# confidence < 0.02), and the second mention created a fresh node from
# zero — in a 100-topic world with periodic light reflects 1 937 created
# concepts shrank to 797 (analysis doc §7.16).  720 observations is the
# "established" half-life: a one-off trace survives about a month of
# observations at 24 per day, then goes.
PRUNE_MIN_IDLE_TICKS: float = 720.0

# ── Probability-anchored relation floor ───────────────────────────────
# An *explicit* relation's traversal weight (and structural confidence)
# relax toward RELATION_FLOOR_SHARE × probability instead of toward 0,
# on the same era scale as the concept floor.  ``probability`` is the
# belief that the typed relation is correct and is never time-decayed
# (guarantee 3) — but a belief attached to an edge that has been pruned
# is lost all the same.  Before this floor an explicit relation stated
# once (p = 0.70) was pruned after ~1 000 idle observations and one
# re-stated five times (p = 0.76) after ~3 000; now the floor keeps it
# above the prune threshold for ~7 900 / ~8 400 observations, after which
# era forgetting lets it go.  Auto-discovered (Hebbian) edges get no
# floor: they live on reinforcement and are revalidated by reflect.
#
# The anchor is a *belief* on every axis.  A stated negative claim used to
# be seeded from its inhibition gain (0.05-0.12), so its floor (0.005-0.012)
# sat below the 0.02 prune threshold and it was forgotten ~18x faster than
# a stated dependence; it is now seeded from ``NEGATIVE_CLAIM_PRIOR`` (0.70,
# ``schemas/relation.py``) and gets the same floor.  A claim whose belief
# is below 0.2 (an extractor's "probably not", or one disconfirmed that far)
# still has a floor under the threshold and is not protected.
RELATION_FLOOR_SHARE: float = 0.1

# Weight below which a relation is considered gone (pruned at reflect).
RELATION_PRUNE_THRESHOLD: float = 0.02


def floor_era(node: ConceptNode) -> float:
    """Era half-life on which this concept's evidence floor forgets."""
    return LONG_TERM_ERA_HL if node.long_term else EVIDENCE_FLOOR_ERA_HL


def concept_half_life(node: ConceptNode) -> float:
    """Effective half-life in ticks: maturity base × evidence scale, or the
    long-term curve for a consolidated concept."""
    if node.long_term:
        return LONG_TERM_HALF_LIFE
    base = CONCEPT_HALF_LIFE.get(node.maturity, 168.0)
    extra_evidence = max(0, node.activation_count - 1)
    scale = min(
        CONCEPT_EVIDENCE_HL_MAX_SCALE,
        1.0 + CONCEPT_EVIDENCE_HL_GAIN * extra_evidence,
    )
    return min(CONCEPT_MAX_HALF_LIFE, base * scale)


def relation_floor(
    edge: RelationEdge,
    *,
    now_tick: int | None = None,
    now: datetime | None = None,
) -> float:
    """Weight floor an explicit relation relaxes toward (0 for Hebbian edges).

    Anchored on ``edge.probability`` — the belief that the claim is correct,
    seeded from the claim prior and never time-decayed — not on the edge's
    propagation / inhibition gain (``weight``).
    """
    if not edge.is_explicit or edge.is_retracted:
        return 0.0
    floor = RELATION_FLOOR_SHARE * edge.probability
    elapsed = edge.elapsed_since_reinforced(now_tick, now)
    if elapsed > 0 and EVIDENCE_FLOOR_ERA_HL > 0:
        floor *= math.pow(0.5, elapsed / EVIDENCE_FLOOR_ERA_HL)
    return floor


def evidence_floor(
    node: ConceptNode,
    *,
    now_tick: int | None = None,
    now: datetime | None = None,
) -> float:
    """Confidence level that accumulated evidence keeps a concept above.

    Zero for a never-activated concept; grows with confirmations, shrinks
    with disconfirmations, and forgets on the era scale.
    """
    n = node.activation_count
    if n <= 0:
        return 0.0
    saturation = (n / (n + EVIDENCE_FLOOR_K)) ** 2
    floor = EVIDENCE_FLOOR_MAX * node.evidence_balance() * saturation
    elapsed = node.elapsed_since_activation(now_tick, now)
    era = floor_era(node)
    if elapsed > 0 and era > 0:
        floor *= math.pow(0.5, elapsed / era)
    return floor


_LN2 = math.log(2.0)


def relax_confidence(
    confidence: float, floor: float, half_life: float, dt: float, era: float = EVIDENCE_FLOOR_ERA_HL
) -> float:
    """Exact solution of ``c' = -λ · max(0, c − f(t))`` with ``f(t) = floor · 2^(-t/E)``.

    ``λ = ln 2 / half_life`` and ``E = era`` is the era half-life of the
    evidence floor (``floor_era``: the long-term era for a consolidated
    concept), so the floor keeps *moving* while confidence relaxes toward it.
    The flow of an ODE is a semigroup, which is why settling in any number
    of pieces gives the same confidence (docs/paper, Proposition 3.1):

    * ``c ≤ f``: nothing to forget; confidence stays put until the floor,
      still falling, meets it (at ``t_x = E · log2(f / c)``);
    * ``c > f``: ``c(t) = A e^(-μt) + (c − A) e^(-λt)`` with
      ``A = λ f / (λ − μ)`` (``μ = ln 2 / E``); ``c ≥ f(t)`` throughout.
    """
    lam = _LN2 / half_life
    mu = _LN2 / era if era > 0 else 0.0
    if confidence <= floor:
        if floor <= 0.0 or mu <= 0.0 or confidence <= 0.0:
            return confidence
        meet = math.log(floor / confidence) / mu
        if meet >= dt:
            return confidence
        floor, dt = confidence, dt - meet
    if floor <= 0.0:
        return confidence * math.exp(-lam * dt)
    if abs(lam - mu) < 1e-9 * max(lam, mu):
        return max(0.0, math.exp(-lam * dt) * (confidence + lam * floor * dt))
    amp = lam * floor / (lam - mu)
    return max(0.0, amp * math.exp(-mu * dt) + (confidence - amp) * math.exp(-lam * dt))


def _settled_state(
    node: ConceptNode, now_tick: int, now: datetime
) -> tuple[float, Maturity, bool] | None:
    """(confidence, maturity, entered_fading) that settling to ``now_tick``
    leaves, without mutating ``node``; None when nothing is owed."""
    if node.maturity == Maturity.FADING and node.confidence <= 0.0:
        return None
    elapsed = node.decay_elapsed(now_tick, now)
    # Skip recently activated (or recently decayed) concepts; the
    # un-applied interval is not lost — it is measured from the unchanged
    # reference point on the next call.
    if elapsed < DECAY_GRACE_TICKS:
        return None
    maturity = node.maturity
    entered_fading = False
    if node.confidence < FADING_THRESHOLD and maturity != Maturity.FADING:
        maturity = Maturity.FADING
        entered_fading = True
    floor = evidence_floor(
        node,
        now_tick=node.decay_reference_tick(),
        now=node.decay_reference_time(),
    )
    confidence = node.confidence
    half_life, era = _profile_for(node, maturity)
    end = relax_confidence(confidence, floor, half_life, elapsed, era)
    if maturity != Maturity.FADING and confidence >= FADING_THRESHOLD > end:
        # Crossing inside the interval: bisect for t* with c(t*) = θ.
        lo, hi = 0.0, elapsed
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if relax_confidence(confidence, floor, half_life, mid, era) >= FADING_THRESHOLD:
                lo = mid
            else:
                hi = mid
        maturity = Maturity.FADING
        entered_fading = True
        floor_at_cross = floor * math.pow(0.5, hi / era) if era > 0 else floor
        # After the crossing the FADING profile applies (a long-term concept
        # that fades is back on the fast curve and the ordinary era).
        tail_half_life, tail_era = _profile_for(node, maturity)
        end = relax_confidence(FADING_THRESHOLD, floor_at_cross, tail_half_life, elapsed - hi, tail_era)
    if end < FADING_THRESHOLD and maturity != Maturity.FADING:
        maturity = Maturity.FADING
        entered_fading = True
    return end, maturity, entered_fading


def _half_life_for(node: ConceptNode, maturity: Maturity) -> float:
    return _profile_for(node, maturity)[0]


def _profile_for(node: ConceptNode, maturity: Maturity) -> tuple[float, float]:
    """(half-life, floor era) the concept would have at ``maturity``."""
    probe = node if maturity == node.maturity else node.model_copy(update={"maturity": maturity})
    return concept_half_life(probe), floor_era(probe)


def settled_confidence(
    node: ConceptNode, now_tick: int, now: datetime | None = None
) -> float:
    """The confidence ``settle_concept`` would leave at ``now_tick``, read-only.

    Readers (activation, projection) use it so that what a view shows does
    not depend on when a reflect last settled the stored value (docs/paper
    Theorem 3.7, read path).
    """
    state = _settled_state(node, now_tick, now or wall_now())
    return node.confidence if state is None else state[0]


def settle_concept(
    node: ConceptNode, now_tick: int, now: datetime | None = None
) -> bool:
    """Apply the decay a concept owes up to ``now_tick``.

    Relaxes confidence toward the (moving) evidence floor over the
    cognitive time since decay was last applied (or since the last
    activation, whichever is later) and moves the reference point to now.
    Idempotent — calling it again at the same instant changes nothing —
    and an exact semigroup: settling ``[t0, t1]`` in any number of pieces
    gives the same confidence.  Returns True when the concept has just
    crossed into FADING.

    The FADING boundary is exact too.  Confidence only falls between
    events, so the interval is split at the instant it crosses
    ``FADING_THRESHOLD`` (found by bisection on the monotone solution): the
    maturity's half-life governs up to it, the FADING half-life after.
    Whether or when settlement runs therefore cannot change the outcome.
    """
    now = now or wall_now()
    state = _settled_state(node, now_tick, now)
    if state is None:
        return False
    node.confidence, node.maturity, entered_fading = state
    node.last_decayed_tick = now_tick
    node.last_decayed_at = now
    return entered_fading


def concept_prunable(
    node: ConceptNode,
    now_tick: int,
    now: datetime | None = None,
    threshold: float = 0.02,
) -> bool:
    """Whether a (settled) concept is past recovery: FADING, below
    ``threshold`` and idle for ``PRUNE_MIN_IDLE_TICKS``.

    A pure predicate of the settled state: whether the physical deletion
    happened at a reflect or is applied lazily at the next mention, the
    concept is dead from the first instant this holds.
    """
    return (
        node.maturity == Maturity.FADING
        and node.confidence < threshold
        and node.elapsed_since_activation(now_tick, now) >= PRUNE_MIN_IDLE_TICKS
    )


def _relaxed_relation(
    edge: RelationEdge, now_tick: int, now: datetime
) -> tuple[float, float] | None:
    """(weight, confidence) after settling to ``now_tick``; None inside the grace.

    Same exact moving-floor relaxation as concepts (``relax_confidence``):
    the floor is taken at the decay reference point and keeps falling on the
    era scale while the weight follows it, so settling in any number of
    pieces gives the same weight (docs/paper, Proposition 4.2).
    """
    elapsed = edge.decay_elapsed(now_tick, now)
    if elapsed < DECAY_GRACE_TICKS:
        return None
    # More reinforced relations decay slower
    half_life = RELATION_BASE_HALF_LIFE * (1.0 + edge.reinforcement_count * 0.5)
    # Relax toward the probability-anchored floor (explicit edges) or
    # toward zero (auto-discovered edges).
    floor = relation_floor(
        edge, now_tick=edge.decay_reference_tick(), now=edge.decay_reference_time()
    )
    return (
        relax_confidence(edge.weight, floor, half_life, elapsed),
        relax_confidence(edge.confidence, floor, half_life, elapsed),
    )


def relation_dead(
    edge: RelationEdge, now_tick: int, now: datetime | None = None
) -> bool:
    """Whether a relation's settled weight is below the prune line.

    A pure predicate of the settled state, like ``concept_prunable``: an
    edge is dead from the first instant this holds, whether a reflect has
    physically removed it or not.  Events that would touch a dead edge
    (a restatement, a co-occurrence) treat it as absent
    (``RelationManager.reap_dead``).
    """
    return projected_relation_weight(edge, now_tick, now) < RELATION_PRUNE_THRESHOLD


def concept_expired(
    node: ConceptNode, now_tick: int, now: datetime | None = None
) -> bool:
    """Whether ``node`` would be prunable once settled to ``now_tick``.

    Like ``relation_dead`` this reads the settled state without mutating
    the node, so it answers the same whether or not decay has been applied
    since the last activation.  Recently used concepts return early.
    """
    now = now or wall_now()
    if node.elapsed_since_activation(now_tick, now) < PRUNE_MIN_IDLE_TICKS:
        return False
    probe = node.model_copy()
    settle_concept(probe, now_tick, now)
    return concept_prunable(probe, now_tick, now)


def projected_relation_weight(
    edge: RelationEdge, now_tick: int, now: datetime | None = None
) -> float:
    """The weight ``settle_relation`` would leave, without mutating the edge.

    Lets a reader ask "is this connection alive?" without depending on
    whether a reflect has physically settled or pruned it yet.
    """
    relaxed = _relaxed_relation(edge, now_tick, now or wall_now())
    return edge.weight if relaxed is None else relaxed[0]


def settle_relation(
    edge: RelationEdge, now_tick: int, now: datetime | None = None
) -> bool:
    """Apply the decay a relation owes up to ``now_tick``.

    Weight and structural confidence relax toward the relation floor
    (explicit edges) or toward zero (auto-discovered edges); semantic
    ``probability`` is never touched.  Idempotent like
    ``settle_concept``.  Returns True when the weight is below the prune
    threshold.
    """
    now = now or wall_now()
    relaxed = _relaxed_relation(edge, now_tick, now)
    if relaxed is None:
        return False
    edge.weight, edge.confidence = relaxed
    # ``probability`` (belief the typed relation is correct) is
    # deliberately left alone: the passage of time is not evidence
    # against a relation, only against its current salience.
    edge.last_decayed_tick = now_tick
    edge.last_decayed_at = now
    return edge.weight < RELATION_PRUNE_THRESHOLD


class DecayEngine:
    """Applies cognitive-time decay to concepts and relations.

    Implements the ``DecayPolicy`` Protocol from ``world0.core``.
    """

    def __init__(
        self,
        concepts: "ConceptStore",
        relations: "RelationStore",
        clock: CognitiveClock | None = None,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._clock = clock or CognitiveClock()

    def decay_concepts(self) -> list[str]:
        """Apply decay to all concept confidences.

        Returns list of concept ids that fell below the fading threshold.
        """
        newly_fading: list[str] = []
        now_tick = self._clock.tick
        now = wall_now()

        for node in self._concepts.all():
            before = (node.last_decayed_tick, node.last_decayed_at)
            if settle_concept(node, now_tick, now):
                newly_fading.append(node.id)
            if (node.last_decayed_tick, node.last_decayed_at) != before:
                self._concepts.mark_dirty(node.id)

        return newly_fading

    def decay_relations(self) -> list[str]:
        """Apply decay to all relation weights.

        Returns list of relation ids that fell below threshold.
        """
        weak_relations: list[str] = []
        now_tick = self._clock.tick
        now = wall_now()

        for edge in self._relations.all():
            before = (edge.last_decayed_tick, edge.last_decayed_at)
            if settle_relation(edge, now_tick, now):
                weak_relations.append(edge.id)
            if (edge.last_decayed_tick, edge.last_decayed_at) != before:
                self._relations.mark_dirty(edge.id)

        return weak_relations

    def prune_concepts(self, threshold: float = 0.02) -> list[str]:
        """Remove concepts that have decayed beyond recovery (batch).

        A concept is pruned when it is FADING, its confidence is below
        ``threshold`` *and* it has been idle for ``PRUNE_MIN_IDLE_TICKS``
        observations — fading marks it as noise, the grace period keeps
        deletion from racing a slow re-mention.
        """
        now_tick = self._clock.tick
        now = wall_now()
        to_prune = [
            n.id
            for n in self._concepts.all()
            if concept_prunable(n, now_tick, now, threshold)
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
