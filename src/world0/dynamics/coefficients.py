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

from world0.schemas.relation import RelationType

# ── Relation type propagation coefficients ────────────────────────────
# Stronger semantic relations propagate more activation.
# "related_to" (the generic fallback) gets the lowest weight.
# Activation can override these per-Perspective; community detection
# and color diffusion always use the defaults below.
RELATION_TYPE_FACTOR: dict[RelationType, float] = {
    RelationType.DEPENDS_ON: 1.0,
    RelationType.CONTAINS: 0.95,
    RelationType.PART_OF: 0.95,
    RelationType.SUPPORTS: 0.85,
    RelationType.ACTIVATES: 0.90,
    RelationType.PRECEDES: 0.80,
    RelationType.DERIVED_FROM: 0.80,
    RelationType.SIMILAR_TO: 0.70,
    RelationType.CONTRASTS: 0.60,
    RelationType.RELATED_TO: 0.50,
}


# ── Temporal relevance half-lives (hours) ────────────────────────────
# "Soft" half-lives for freshness weighting during read-only operations
# (activation propagation, coupling for community detection).  These
# are *separate* from the hard decay half-lives in DecayEngine — those
# actually mutate confidence, while these only modulate scoring.
CONCEPT_TEMPORAL_HL: float = 168.0   # 1 week for concept freshness
RELATION_TEMPORAL_HL: float = 72.0   # 3 days for relation freshness
