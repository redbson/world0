"""Network entropy of the World 0 concept graph.

This metric estimates how *concentrated* vs. *diffuse* conceptual
attention is across the explicit concept-relation network.  It answers:

- Is activation concentrated on a few deterministic relation paths?
- Is the graph too diffuse / noisy to project usefully?
- Which concepts spread mass over many weakly-distinguished neighbors?

It is **not** token entropy, embedding entropy, or document diversity —
it is computed purely from the typed concept-relation structure.

The unit of entropy is the *local relation distribution* of one concept:
all relations from a concept are turned into a probability distribution
over its neighbors (mass summed per neighbor), and Shannon entropy of
that distribution — normalized by ``log2(degree)`` — measures how evenly
the concept's belief is spread.  The world-level metric is the
node-importance-weighted mean of those normalized local entropies.

Design: ``docs/world-network-entropy-design.md``.  That document predates
the axis-based relation model, so ``effective_weight`` here is re-grounded
on the *current* model (``probability × axis factor × explicitness``)
rather than the obsolete per-type factor table.
"""

from __future__ import annotations

import math
from collections import defaultdict

from pydantic import BaseModel, Field

from world0.dynamics.coefficients import RELATION_TYPE_FACTOR
from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge

# ── Tunables ──────────────────────────────────────────────────────────
# Hebbian (auto-discovered) edges carry less explicit conceptual
# structure than declared relations, so they contribute less information
# mass — mirrors the explicitness asymmetry already used in reinforce().
_EXPLICIT_FACTOR = 1.0
_HEBBIAN_FACTOR = 0.75

# Node importance weights local entropy so a noisy embryonic concept
# perturbs the world score less than a stable core concept whose local
# uncertainty disproportionately shapes projections.
_MATURITY_FACTOR: dict[Maturity, float] = {
    Maturity.EMBRYONIC: 0.50,
    Maturity.DEVELOPING: 0.75,
    Maturity.ESTABLISHED: 1.00,
    Maturity.CORE: 1.20,
    Maturity.FADING: 0.35,
}
_CONFIDENCE_FLOOR = 0.05

# Interpretation bands for normalized local entropy (see design doc §Interpretation).
_LOW_ENTROPY_MAX = 0.20    # highly concentrated / under-branched
_HIGH_ENTROPY_MIN = 0.70   # highly diffuse; likely noisy unless intentional


class NetworkEntropy(BaseModel):
    """Read-only diagnostic over the concept-relation graph.

    ``avg_network_entropy`` is in ``[0, 1]``:

    - ``0.00–0.20`` highly concentrated / under-connected
    - ``0.20–0.45`` structured and focused
    - ``0.45–0.70`` diverse but still interpretable
    - ``0.70–1.00`` highly diffuse; likely noisy unless intentional

    High entropy is not automatically bad — a bridge concept legitimately
    connects several neighborhoods — so the score is reported alongside
    supporting counts and a relation-type entropy that flags whether
    generic ``parallel`` edges dominate.
    """

    avg_network_entropy: float = 0.0
    nodes_considered: int = 0
    isolated_nodes: int = 0
    high_entropy_nodes: int = 0
    low_entropy_nodes: int = 0
    # Entropy over the three relation axes (positive/negative/parallel).
    # High values with parallel-dominant mass indicate a graph drifting
    # toward generic, weakly-typed structure.
    relation_type_entropy: float = 0.0
    relation_type_mass: dict[str, float] = Field(default_factory=dict)


def _effective_weight(rel: RelationEdge) -> float:
    """Information mass a relation contributes to its neighbor.

    Grounded on the current relation model: semantic ``probability``
    (belief the typed relation is correct), scaled by the axis
    propagation factor and an explicitness factor.  A weak generic
    ``parallel`` Hebbian edge therefore contributes far less mass than a
    reinforced explicit ``positive`` edge.
    """
    axis_factor = RELATION_TYPE_FACTOR.get(rel.relation_type, 0.5)
    explicitness = _EXPLICIT_FACTOR if rel.is_explicit else _HEBBIAN_FACTOR
    return max(0.0, rel.probability) * axis_factor * explicitness


def _node_importance(concept: ConceptNode) -> float:
    return max(concept.confidence, _CONFIDENCE_FLOOR) * _MATURITY_FACTOR.get(
        concept.maturity, 1.0
    )


def _local_entropy(neighbor_mass: dict[str, float]) -> float | None:
    """Normalized Shannon entropy of one concept's neighbor distribution.

    Returns ``None`` when the concept has fewer than two neighbors with
    positive mass (no branching uncertainty), so callers can distinguish
    "isolated/deterministic" from "genuinely 0 entropy".
    """
    masses = [m for m in neighbor_mass.values() if m > 0.0]
    if len(masses) < 2:
        return None
    total = sum(masses)
    if total <= 0.0:
        return None
    entropy = 0.0
    for m in masses:
        p = m / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(len(masses))


def compute_network_entropy(
    concepts: list[ConceptNode],
    relations: list[RelationEdge],
) -> NetworkEntropy:
    """Compute the world-average normalized network entropy.

    Pure function: it reads the concept/relation lists and returns a
    :class:`NetworkEntropy`.  It never mutates either argument.
    """
    if not concepts or not relations:
        return NetworkEntropy()

    concept_ids = {c.id for c in concepts}

    # adjacency: concept_id -> neighbor_id -> summed effective mass
    adjacency: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    type_mass: dict[str, float] = defaultdict(float)

    for rel in relations:
        src, tgt = rel.source_id, rel.target_id
        # Defensive: ignore self-relations and dangling endpoints.
        if src == tgt:
            continue
        if src not in concept_ids or tgt not in concept_ids:
            continue
        mass = _effective_weight(rel)
        if mass <= 0.0:
            continue
        adjacency[src][tgt] += mass
        adjacency[tgt][src] += mass
        type_mass[rel.relation_type.value] += mass

    weighted_entropy = 0.0
    importance_total = 0.0
    nodes_considered = 0
    isolated_nodes = 0
    high_entropy_nodes = 0
    low_entropy_nodes = 0

    for concept in concepts:
        local = _local_entropy(adjacency.get(concept.id, {}))
        if local is None:
            isolated_nodes += 1
            continue
        nodes_considered += 1
        importance = _node_importance(concept)
        weighted_entropy += importance * local
        importance_total += importance
        if local >= _HIGH_ENTROPY_MIN:
            high_entropy_nodes += 1
        elif local <= _LOW_ENTROPY_MAX:
            low_entropy_nodes += 1

    avg = (
        weighted_entropy / importance_total if importance_total > 0.0 else 0.0
    )

    return NetworkEntropy(
        avg_network_entropy=avg,
        nodes_considered=nodes_considered,
        isolated_nodes=isolated_nodes,
        high_entropy_nodes=high_entropy_nodes,
        low_entropy_nodes=low_entropy_nodes,
        relation_type_entropy=_relation_type_entropy(type_mass),
        relation_type_mass=dict(type_mass),
    )


def _relation_type_entropy(type_mass: dict[str, float]) -> float:
    """Normalized entropy over relation-axis mass.

    Diagnoses whether the graph's relational structure is dominated by a
    single axis (low entropy) or balanced across positive/negative/
    parallel channels (high entropy).  Reported together with
    ``relation_type_mass`` so a high value driven purely by generic
    ``parallel`` edges can be told apart from healthy axis diversity.
    """
    masses = [m for m in type_mass.values() if m > 0.0]
    if len(masses) < 2:
        return 0.0
    total = sum(masses)
    entropy = 0.0
    for m in masses:
        p = m / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(len(masses))
