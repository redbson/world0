"""Shared cognitive coefficients used by multiple Dynamics engines.

These constants describe relations between *types of edges* and *time
scales* — they are not owned by any single engine.  Putting them here
breaks the previous concrete dependency in which ``community.py`` and
``color_diffusion.py`` had to import from ``activation.py``: now each
engine depends only on ``coefficients.py`` and on the ``ConceptStore``
/ ``RelationStore`` Protocols.

If you tune one of these constants, expect it to ripple through
activation spread, community detection and color diffusion at once —
that is intentional, since they describe properties of the underlying
relation graph rather than of any individual algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from world0.schemas.relation import RelationType

# ── Relation type propagation coefficients ────────────────────────────
# Stronger axis relations propagate more activation.
# Positive attraction is the strongest excitatory channel, parallel
# resonance is a moderate co-attention channel, and negative repulsion is
# handled separately by the activation engine as inhibition.
# Activation can override these per-Perspective; community detection
# and color diffusion always use the defaults below.
RELATION_TYPE_FACTOR: dict[RelationType, float] = {
    RelationType.POSITIVE: 1.0,
    RelationType.PARALLEL: 0.75,
    RelationType.NEGATIVE: 0.60,
}


# ── Temporal relevance half-lives (hours) ────────────────────────────
# "Soft" half-lives for freshness weighting during read-only operations
# (activation propagation, coupling for community detection).  These
# are *separate* from the hard decay half-lives in DecayEngine — those
# actually mutate confidence, while these only modulate scoring.
CONCEPT_TEMPORAL_HL: float = 168.0   # 1 week for concept freshness
RELATION_TEMPORAL_HL: float = 72.0   # 3 days for relation freshness


# ── Activation engine coefficients ───────────────────────────────────
# Defined here (not in activation.py) so the config dataclasses below
# can reference them without an import cycle; activation.py re-exports
# them for backward compatibility.

# Negative-axis links spread negative activation on an independent
# inhibition channel, scaled by this factor.
CONTRASTS_INHIBITION_FACTOR: float = 0.6

# Multiplier when the current task matches a concept's or relation's
# history.  Matching is graded: an exact substring hit earns the full
# boost, partial token overlap earns a proportional fraction.
TASK_AFFINITY_BOOST: float = 1.5

# Minimum graded affinity required before any boost applies — below
# this, token overlap is treated as coincidence.
MIN_TASK_AFFINITY: float = 0.34

# Low-confidence nodes still allow propagation to pass through at this
# minimum readiness level, preventing "dead node" blockage.
PROPAGATION_FLOOR: float = 0.3

# Propagated score is at least this fraction of the seed score at each
# depth step, widening the cognitive horizon from ~1 hop to 3-4 hops.
PROPAGATION_MIN_RATIO: float = 0.03


# ── Projection engine coefficients ───────────────────────────────────

# MMR diversity weight. 0 = pure score ranking, 1 = pure diversity.
MMR_LAMBDA: float = 0.3

# Relevance discount in MMR for candidates with *no* task association.
TASK_AFFINITY_DISCOUNT: float = 0.6

# Temporal freshness weight in MMR relevance computation.
TEMPORAL_WEIGHT: float = 0.3

# Half-life used for temporal relevance in projection (hours).
PROJECTION_TEMPORAL_HL: float = 168.0  # 1 week


# ── Engine configuration objects ─────────────────────────────────────
# Frozen dataclasses whose defaults are the module constants above.
# Engines read from their config; passing a custom config tunes one
# engine instance without touching global state — the substrate for
# the sensitivity-sweep harness (scripts/sweep_projection_quality.py).


def _default_relation_type_factor() -> Mapping[RelationType, float]:
    return MappingProxyType(dict(RELATION_TYPE_FACTOR))


@dataclass(frozen=True)
class ActivationConfig:
    """Tunable coefficients for ``ActivationEngine``."""

    relation_type_factor: Mapping[RelationType, float] = field(
        default_factory=_default_relation_type_factor
    )
    inhibition_factor: float = CONTRASTS_INHIBITION_FACTOR
    task_affinity_boost: float = TASK_AFFINITY_BOOST
    min_task_affinity: float = MIN_TASK_AFFINITY
    propagation_floor: float = PROPAGATION_FLOOR
    propagation_min_ratio: float = PROPAGATION_MIN_RATIO
    concept_temporal_hl: float = CONCEPT_TEMPORAL_HL
    relation_temporal_hl: float = RELATION_TEMPORAL_HL


@dataclass(frozen=True)
class ProjectionConfig:
    """Tunable coefficients for ``ProjectionEngine``."""

    mmr_lambda: float = MMR_LAMBDA
    task_affinity_discount: float = TASK_AFFINITY_DISCOUNT
    temporal_weight: float = TEMPORAL_WEIGHT
    temporal_hl: float = PROJECTION_TEMPORAL_HL
    min_relation_signal: float = 0.0
    max_relations: int | None = None
