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

Optimization: items touched less than one tick ago are skipped entirely,
avoiding unnecessary floating-point work.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from world0.schemas.clock import CognitiveClock, wall_now
from world0.schemas.concept import SALIENCE_ERA_HL, ConceptNode, Maturity
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
# Calibration (docs/world0-cognitive-dynamics-analysis.md §5): with 0.2 a
# concept re-observed every 24 observations leaves ``embryonic`` after
# ~700 observations and reaches ``established`` after ~1400; one
# re-observed every 168 settles as ``developing`` (≈0.36); one re-observed
# every 720 stays alive on its evidence floor; a one-off mention fades
# after ~54 observations; a concept confirmed 30× and then abandoned needs
# ~26 000 observations to fade.
CONCEPT_EVIDENCE_HL_GAIN: float = 0.20
CONCEPT_EVIDENCE_HL_MAX_SCALE: float = 8.0
# Absolute ceiling on the effective half-life so heavily used core
# concepts still forget on a bounded scale instead of becoming immortal.
CONCEPT_MAX_HALF_LIFE: float = 8760.0

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
RELATION_FLOOR_SHARE: float = 0.1


def concept_half_life(node: ConceptNode) -> float:
    """Effective half-life in ticks: maturity base × evidence scale."""
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
    """Weight floor an explicit relation relaxes toward (0 for Hebbian edges)."""
    if not edge.is_explicit:
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
    if elapsed > 0 and EVIDENCE_FLOOR_ERA_HL > 0:
        floor *= math.pow(0.5, elapsed / EVIDENCE_FLOOR_ERA_HL)
    return floor


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
            if node.maturity == Maturity.FADING and node.confidence <= 0.0:
                continue

            elapsed = node.decay_elapsed(now_tick, now)

            # Skip recently activated (or recently decayed) concepts; the
            # un-applied interval is not lost — it is measured from the
            # unchanged reference point on the next call.
            if elapsed < DECAY_GRACE_TICKS:
                continue

            decay_factor = math.pow(0.5, elapsed / concept_half_life(node))
            floor = evidence_floor(node, now_tick=now_tick, now=now)
            if node.confidence > floor:
                node.confidence = max(
                    0.0, floor + (node.confidence - floor) * decay_factor
                )
            node.last_decayed_tick = now_tick
            node.last_decayed_at = now
            self._concepts.mark_dirty(node.id)

            if (
                node.confidence < FADING_THRESHOLD
                and node.maturity != Maturity.FADING
            ):
                node.maturity = Maturity.FADING
                newly_fading.append(node.id)

        return newly_fading

    def decay_relations(self) -> list[str]:
        """Apply decay to all relation weights.

        Returns list of relation ids that fell below threshold.
        """
        weak_relations: list[str] = []
        now_tick = self._clock.tick
        now = wall_now()

        for edge in self._relations.all():
            elapsed = edge.decay_elapsed(now_tick, now)

            # Skip recently reinforced (or recently decayed) relations
            if elapsed < DECAY_GRACE_TICKS:
                continue

            # More reinforced relations decay slower
            half_life = RELATION_BASE_HALF_LIFE * (
                1.0 + edge.reinforcement_count * 0.5
            )
            decay_factor = math.pow(0.5, elapsed / half_life)

            # Relax toward the probability-anchored floor (explicit edges)
            # or toward zero (auto-discovered edges).
            floor = relation_floor(edge, now_tick=now_tick, now=now)
            if edge.weight > floor:
                edge.weight = max(0.0, floor + (edge.weight - floor) * decay_factor)
            if edge.confidence > floor:
                edge.confidence = max(
                    0.0, floor + (edge.confidence - floor) * decay_factor
                )
            # ``probability`` (belief the typed relation is correct) is
            # deliberately left alone: the passage of time is not evidence
            # against a relation, only against its current salience.
            edge.last_decayed_tick = now_tick
            edge.last_decayed_at = now
            self._relations.mark_dirty(edge.id)

            if edge.weight < 0.02:
                weak_relations.append(edge.id)

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
            if n.maturity == Maturity.FADING
            and n.confidence < threshold
            and n.elapsed_since_activation(now_tick, now) >= PRUNE_MIN_IDLE_TICKS
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
