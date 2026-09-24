"""Time-based decay — unused concepts and relations fade over time.

Decay rates are maturity-dependent:
  - core concepts decay very slowly
  - embryonic concepts decay fast
  - relations decay inversely with reinforcement count

Three guarantees this engine makes (analysis and probe evidence in
``docs/world0-cognitive-dynamics-analysis.md``):

1. **Idempotent in wall-clock time.**  Decay is applied to the interval
   since the *last decay application* (or the last activation, whichever
   is later), never to the whole span since the last activation.  Calling
   ``reflect()`` ten times in an hour therefore leaves the same
   confidence as calling it once — decay tracks elapsed time, not reflect
   frequency.

2. **Evidence-anchored (concepts).**  Repeated confirmation buys
   persistence in two ways.  The effective half-life stretches with
   activation count (as it already did for relations), and confidence
   relaxes toward an *evidence floor* instead of toward zero — an
   Ornstein–Uhlenbeck-style mean reversion.  The floor itself forgets on
   an "era" time scale (months), so nothing is immortal, but a concept
   confirmed thirty times no longer evaporates in a week of silence the
   way a one-off mention does.  Disconfirmation lowers the floor through
   ``evidence_balance()``.

3. **Semantic probability is not time-decayed (relations).**  ``weight``
   and ``confidence`` are operational traversal strengths and do decay;
   ``probability`` is the belief that the typed relation is *correct*,
   which only evidence (reinforce / weaken / extraction) may change.

Optimization: items activated/reinforced within a grace period are
skipped entirely, avoiding unnecessary floating-point work.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from world0.schemas.concept import ConceptNode, Maturity

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore

# Half-life in hours per maturity level, for a concept backed by a single
# confirming observation.  ``concept_half_life`` stretches it with
# additional evidence.
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

# ── Evidence-scaled concept half-life ─────────────────────────────────
# Every confirming activation beyond the first stretches the maturity
# half-life by this fraction, capped at CONCEPT_EVIDENCE_HL_MAX_SCALE.
# Mirrors the ``1 + 0.5 × reinforcement_count`` rule relations already
# use, with a gentler slope because maturity stages already encode a
# large share of a concept's evidence history.
#
# Calibration (docs/world0-cognitive-dynamics-analysis.md §5): with 0.2 a
# concept used daily leaves ``embryonic`` after ~4 weeks and reaches
# ``established`` after ~8; one used weekly settles as ``developing``
# (≈0.36); one used monthly stays alive on its evidence floor; a one-off
# mention fades in ~2 days; a concept confirmed 30× and then abandoned
# needs ~3 years to fade.
CONCEPT_EVIDENCE_HL_GAIN: float = 0.20
CONCEPT_EVIDENCE_HL_MAX_SCALE: float = 8.0
# Absolute ceiling on the effective half-life so heavily used core
# concepts still forget on a human time scale (one year) instead of
# becoming immortal.
CONCEPT_MAX_HALF_LIFE: float = 8760.0

# ── Evidence floor (mean-reversion target) ────────────────────────────
#   floor = EVIDENCE_FLOOR_MAX × evidence_balance × (n / (n + K))²
#           × 0.5 ** (hours_since_activation / EVIDENCE_FLOOR_ERA_HL)
# The squared saturation keeps the floor below FADING_THRESHOLD for
# lightly evidenced concepts (n ≲ 6) so genuine noise still fades and is
# pruned, while a concept confirmed ~30 times holds ≈ 0.20 and one
# confirmed 100 times ≈ 0.29.  The era half-life makes the floor itself
# forget: a 30-confirmation concept left untouched drops below the fading
# threshold after roughly a year.
EVIDENCE_FLOOR_MAX: float = 0.35
EVIDENCE_FLOOR_K: float = 10.0
EVIDENCE_FLOOR_ERA_HL: float = 4380.0  # ≈ 6 months

# Confidence below which a concept is marked FADING.
FADING_THRESHOLD: float = 0.05


def concept_half_life(node: ConceptNode) -> float:
    """Effective half-life in hours: maturity base × evidence scale."""
    base = CONCEPT_HALF_LIFE.get(node.maturity, 168.0)
    extra_evidence = max(0, node.activation_count - 1)
    scale = min(
        CONCEPT_EVIDENCE_HL_MAX_SCALE,
        1.0 + CONCEPT_EVIDENCE_HL_GAIN * extra_evidence,
    )
    return min(CONCEPT_MAX_HALF_LIFE, base * scale)


def evidence_floor(node: ConceptNode, *, now: datetime | None = None) -> float:
    """Confidence level that accumulated evidence keeps a concept above.

    Zero for a never-activated concept; grows with confirmations, shrinks
    with disconfirmations, and forgets on the era time scale.
    """
    n = node.activation_count
    if n <= 0:
        return 0.0
    saturation = (n / (n + EVIDENCE_FLOOR_K)) ** 2
    floor = EVIDENCE_FLOOR_MAX * node.evidence_balance() * saturation
    hours = node.hours_since_activation(now)
    if hours > 0 and EVIDENCE_FLOOR_ERA_HL > 0:
        floor *= math.pow(0.5, hours / EVIDENCE_FLOOR_ERA_HL)
    return floor


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
        now = datetime.now(timezone.utc)

        for node in self._concepts.all():
            if node.maturity == Maturity.FADING and node.confidence <= 0.0:
                continue

            hours = (now - node.decay_reference_time()).total_seconds() / 3600.0

            # Skip recently activated (or recently decayed) concepts; the
            # un-applied interval is not lost — it is measured from the
            # unchanged reference time on the next call.
            if hours < DECAY_GRACE_HOURS:
                continue

            decay_factor = math.pow(0.5, hours / concept_half_life(node))
            floor = evidence_floor(node, now=now)
            if node.confidence > floor:
                node.confidence = max(
                    0.0, floor + (node.confidence - floor) * decay_factor
                )
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
        """Apply time decay to all relation weights.

        Returns list of relation ids that fell below threshold.
        """
        weak_relations: list[str] = []
        now = datetime.now(timezone.utc)

        for edge in self._relations.all():
            hours = (now - edge.decay_reference_time()).total_seconds() / 3600.0

            # Skip recently reinforced (or recently decayed) relations
            if hours < DECAY_GRACE_HOURS:
                continue

            # More reinforced relations decay slower
            half_life = RELATION_BASE_HALF_LIFE * (
                1.0 + edge.reinforcement_count * 0.5
            )
            decay_factor = math.pow(0.5, hours / half_life)

            edge.weight = max(0.0, edge.weight * decay_factor)
            edge.confidence = max(0.0, edge.confidence * decay_factor)
            # ``probability`` (belief the typed relation is correct) is
            # deliberately left alone: the passage of time is not evidence
            # against a relation, only against its current salience.
            edge.last_decayed_at = now
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
