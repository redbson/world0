"""Deep, deterministic tests for the network-entropy diagnostic.

These complement ``tests/test_network_entropy.py`` by pinning down
analytic expected values that are derived by hand from the published
algorithm (``docs/world-network-entropy-design.md``) and the actual
implementation in ``src/world0/metrics/entropy.py``.

The metric is a pure structural property of the typed concept-relation
graph, so every test here is fully deterministic: no LLM, no storage, no
randomness.  Where a closed-form value is derivable we assert it with a
tight float tolerance; otherwise we assert ordering/bounds invariants.

Key facts about the implementation that these tests exercise:

* ``_effective_weight(r) = max(0, r.probability)
  * RELATION_TYPE_FACTOR[r.type] * explicitness`` where the axis factor
  is ``positive=1.0, parallel=0.75, negative=0.60`` and explicitness is
  ``1.0`` for explicit edges and ``0.75`` for Hebbian edges.
* Local entropy uses the *undirected* neighbor distribution (each edge
  contributes mass to both endpoints) normalized by ``log2(degree)``.
* Nodes with fewer than two positive-mass neighbors are "isolated" and
  excluded from ``nodes_considered`` / the weighted mean.
* The world score is the node-importance-weighted mean of normalized
  local entropies, importance = ``max(confidence, 0.05) * maturity``.
"""

from __future__ import annotations

import math

import pytest

from world0.dynamics.coefficients import RELATION_TYPE_FACTOR
from world0.metrics.entropy import (
    NetworkEntropy,
    _effective_weight,
    compute_network_entropy,
)
from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge, RelationType

# Implementation tunables mirrored here so the analytic expectations are
# self-documenting.  These are read-only references, never mutated.
_HEBBIAN_FACTOR = 0.75
_LOW_ENTROPY_MAX = 0.20
_HIGH_ENTROPY_MIN = 0.70
_CONFIDENCE_FLOOR = 0.05
_MATURITY = {
    Maturity.EMBRYONIC: 0.50,
    Maturity.DEVELOPING: 0.75,
    Maturity.ESTABLISHED: 1.00,
    Maturity.CORE: 1.20,
    Maturity.FADING: 0.35,
}


def _concept(name: str, **kw) -> ConceptNode:
    return ConceptNode(name=name, **kw)


def _edge(src: str, tgt: str, **kw) -> RelationEdge:
    """Build an edge with probability pinned to the requested value.

    The model validator backfills probability from the semantic profile
    when the field is left at its default, so we re-assign it afterward to
    keep entropy a pure function of declared structure.
    """
    edge = RelationEdge(source_id=src, target_id=tgt, **kw)
    if "probability" in kw:
        edge.probability = kw["probability"]
    return edge


def _norm_entropy(masses: list[float]) -> float:
    """Reference normalized Shannon entropy of a positive mass vector."""
    positive = [m for m in masses if m > 0.0]
    if len(positive) < 2:
        return 0.0
    total = sum(positive)
    h = -sum((m / total) * math.log2(m / total) for m in positive)
    return h / math.log2(len(positive))


# ─────────────────────────────────────────────────────────────────────
# Effective-weight unit semantics (the per-edge mass each test relies on)
# ─────────────────────────────────────────────────────────────────────
class TestEffectiveWeight:
    def test_axis_factors_match_coefficients(self):
        for axis, factor in RELATION_TYPE_FACTOR.items():
            e = _edge("x", "y", relation_type=axis, is_explicit=True,
                      probability=1.0)
            assert math.isclose(_effective_weight(e), factor, abs_tol=1e-12)

    def test_hebbian_discount_is_three_quarters(self):
        explicit = _edge("x", "y", relation_type=RelationType.POSITIVE,
                         is_explicit=True, probability=0.8)
        hebbian = _edge("x", "y", relation_type=RelationType.POSITIVE,
                        is_explicit=False, probability=0.8)
        assert math.isclose(_effective_weight(explicit), 0.8, abs_tol=1e-12)
        assert math.isclose(
            _effective_weight(hebbian), 0.8 * _HEBBIAN_FACTOR, abs_tol=1e-12
        )

    def test_zero_probability_yields_zero_mass(self):
        e = _edge("x", "y", relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.0)
        assert _effective_weight(e) == 0.0


# ─────────────────────────────────────────────────────────────────────
# Boundary / degenerate graphs with KNOWN entropy of exactly 0
# ─────────────────────────────────────────────────────────────────────
class TestDegenerateGraphs:
    def test_single_node_no_edges_is_zero(self):
        a = _concept("a")
        result = compute_network_entropy([a], [])
        assert isinstance(result, NetworkEntropy)
        assert result.avg_network_entropy == 0.0
        assert result.nodes_considered == 0
        assert result.isolated_nodes == 0  # short-circuit: no relations
        assert result.high_entropy_nodes == 0
        assert result.low_entropy_nodes == 0
        assert result.relation_type_entropy == 0.0
        assert result.relation_type_mass == {}

    def test_single_edge_both_endpoints_isolated(self):
        # a-b: each side has exactly one neighbor -> no branching.
        a, b = _concept("a"), _concept("b")
        edge = _edge(a.id, b.id, is_explicit=True, probability=0.9)
        result = compute_network_entropy([a, b], [edge])
        assert result.avg_network_entropy == 0.0
        assert result.nodes_considered == 0
        assert result.isolated_nodes == 2

    def test_chain_middle_node_has_two_uniform_neighbors(self):
        # a-b-c with equal masses on both edges: only b branches, and its
        # two neighbors are equal -> normalized local entropy 1.0.  a and c
        # are isolated (degree 1).
        a = _concept("a", maturity=Maturity.ESTABLISHED, confidence=1.0)
        b = _concept("b", maturity=Maturity.ESTABLISHED, confidence=1.0)
        c = _concept("c", maturity=Maturity.ESTABLISHED, confidence=1.0)
        edges = [
            _edge(a.id, b.id, is_explicit=True, probability=0.5),
            _edge(b.id, c.id, is_explicit=True, probability=0.5),
        ]
        result = compute_network_entropy([a, b, c], edges)
        assert result.nodes_considered == 1
        assert result.isolated_nodes == 2
        assert math.isclose(result.avg_network_entropy, 1.0, abs_tol=1e-12)
        assert result.high_entropy_nodes == 1

    def test_all_zero_probability_edges_yield_zero(self):
        # Edges carry no mass -> adjacency empty -> nothing considered.
        h = _concept("hub", confidence=1.0)
        leaves = [_concept(f"l{i}") for i in range(3)]
        edges = [
            _edge(h.id, leaf.id, is_explicit=True, probability=0.0)
            for leaf in leaves
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert result.avg_network_entropy == 0.0
        assert result.nodes_considered == 0


# ─────────────────────────────────────────────────────────────────────
# Star graphs: hub neighbor distribution with closed-form entropy
# ─────────────────────────────────────────────────────────────────────
class TestStarGraphs:
    @pytest.mark.parametrize("n_leaves", [2, 3, 4, 5, 8])
    def test_uniform_star_hub_is_normalized_max(self, n_leaves):
        # Equal-weight fan-out -> hub local entropy normalizes to exactly 1.0
        # regardless of leaf count.  All leaves are isolated (degree 1).
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        leaves = [_concept(f"leaf{i}") for i in range(n_leaves)]
        edges = [
            _edge(h.id, leaf.id, is_explicit=True, probability=0.6)
            for leaf in leaves
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert result.nodes_considered == 1
        assert result.isolated_nodes == n_leaves
        assert math.isclose(result.avg_network_entropy, 1.0, abs_tol=1e-12)
        assert result.high_entropy_nodes == 1
        assert result.low_entropy_nodes == 0

    def test_skewed_star_between_zero_and_uniform_max(self):
        # Masses 0.8 / 0.2 over two neighbors -> H_norm = 0.7219...
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        l0, l1 = _concept("l0"), _concept("l1")
        edges = [
            _edge(h.id, l0.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.8),
            _edge(h.id, l1.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.2),
        ]
        result = compute_network_entropy([h, l0, l1], edges)
        expected = _norm_entropy([0.8, 0.2])
        assert math.isclose(expected, 0.7219280948873623, abs_tol=1e-12)
        assert math.isclose(
            result.avg_network_entropy, expected, abs_tol=1e-12
        )
        assert 0.0 < result.avg_network_entropy < 1.0

    def test_three_neighbor_skew_closed_form(self):
        # Masses 0.6 / 0.3 / 0.1 over three neighbors (all positive
        # explicit so mass == probability).
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        leaves = [_concept(f"l{i}") for i in range(3)]
        probs = [0.6, 0.3, 0.1]
        edges = [
            _edge(h.id, leaves[i].id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=probs[i])
            for i in range(3)
        ]
        result = compute_network_entropy([h, *leaves], edges)
        expected = _norm_entropy(probs)
        assert math.isclose(
            result.avg_network_entropy, expected, abs_tol=1e-12
        )
        assert 0.0 < result.avg_network_entropy < 1.0

    def test_duplicate_edges_to_same_neighbor_are_summed(self):
        # Two edges hub->leaf0 plus one hub->leaf1: leaf0 mass is the sum.
        # With both leaf0 edges 0.3 and the leaf1 edge 0.6 the two summed
        # masses are equal (0.6 each) -> uniform -> entropy 1.0.
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        l0, l1 = _concept("l0"), _concept("l1")
        edges = [
            _edge(h.id, l0.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.3),
            _edge(h.id, l0.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.3),
            _edge(h.id, l1.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.6),
        ]
        result = compute_network_entropy([h, l0, l1], edges)
        assert result.nodes_considered == 1
        assert math.isclose(result.avg_network_entropy, 1.0, abs_tol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# k-regular / equal-weight structures and normalization bounds
# ─────────────────────────────────────────────────────────────────────
class TestRegularAndBounds:
    def test_triangle_every_node_uniform_degree_two(self):
        # Fully connected triangle, equal masses: each node sees two equal
        # neighbors -> every local entropy 1.0 -> world average 1.0.
        a = _concept("a", maturity=Maturity.ESTABLISHED, confidence=1.0)
        b = _concept("b", maturity=Maturity.ESTABLISHED, confidence=1.0)
        c = _concept("c", maturity=Maturity.ESTABLISHED, confidence=1.0)
        edges = [
            _edge(a.id, b.id, is_explicit=True, probability=0.5),
            _edge(b.id, c.id, is_explicit=True, probability=0.5),
            _edge(a.id, c.id, is_explicit=True, probability=0.5),
        ]
        result = compute_network_entropy([a, b, c], edges)
        assert result.nodes_considered == 3
        assert result.isolated_nodes == 0
        assert math.isclose(result.avg_network_entropy, 1.0, abs_tol=1e-12)
        assert result.high_entropy_nodes == 3

    @pytest.mark.parametrize(
        "probs",
        [
            [0.9, 0.1],
            [0.7, 0.2, 0.1],
            [0.5, 0.5, 0.5, 0.5],
            [0.99, 0.5, 0.01],
            [0.4, 0.3, 0.2, 0.1],
            [0.6, 0.6, 0.6, 0.6, 0.6],
        ],
    )
    def test_world_entropy_always_within_unit_interval(self, probs):
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        leaves = [_concept(f"l{i}") for i in range(len(probs))]
        edges = [
            _edge(h.id, leaves[i].id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=probs[i])
            for i in range(len(probs))
        ]
        result = compute_network_entropy([h, *leaves], edges)
        # Normalization targets [0, 1]; allow float-rounding slack at the
        # max edge (a perfectly uniform fan-out can land at 1.0 + 1e-16).
        assert -1e-9 <= result.avg_network_entropy <= 1.0 + 1e-9

    def test_uniform_strictly_above_skewed_same_degree(self):
        def world(probs):
            h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
            leaves = [_concept(f"l{i}") for i in range(len(probs))]
            edges = [
                _edge(h.id, leaves[i].id, relation_type=RelationType.POSITIVE,
                      is_explicit=True, probability=probs[i])
                for i in range(len(probs))
            ]
            return compute_network_entropy([h, *leaves], edges).avg_network_entropy

        uniform = world([0.5, 0.5, 0.5, 0.5])
        mild = world([0.6, 0.5, 0.5, 0.4])
        sharp = world([0.9, 0.05, 0.03, 0.02])
        assert math.isclose(uniform, 1.0, abs_tol=1e-12)
        assert uniform > mild > sharp > 0.0

    def test_band_classification_low_high_thresholds(self):
        # A near-deterministic hub (one dominant neighbor) lands in the
        # low band; a uniform hub lands in the high band.
        def hub_local(probs):
            h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
            leaves = [_concept(f"l{i}") for i in range(len(probs))]
            edges = [
                _edge(h.id, leaves[i].id, relation_type=RelationType.POSITIVE,
                      is_explicit=True, probability=probs[i])
                for i in range(len(probs))
            ]
            return compute_network_entropy([h, *leaves], edges)

        low = hub_local([0.98, 0.01, 0.01])
        assert low.avg_network_entropy <= _LOW_ENTROPY_MAX
        assert low.low_entropy_nodes == 1
        assert low.high_entropy_nodes == 0

        high = hub_local([0.5, 0.5, 0.5])
        assert high.avg_network_entropy >= _HIGH_ENTROPY_MIN
        assert high.high_entropy_nodes == 1
        assert high.low_entropy_nodes == 0


# ─────────────────────────────────────────────────────────────────────
# Relation-type (axis) entropy: direction + closed-form for even mixes
# ─────────────────────────────────────────────────────────────────────
class TestRelationTypeEntropy:
    def test_all_same_axis_is_zero(self):
        h = _concept("hub", confidence=1.0)
        leaves = [_concept(f"l{i}") for i in range(4)]
        edges = [
            _edge(h.id, leaf.id, relation_type=RelationType.PARALLEL,
                  is_explicit=True, probability=0.7)
            for leaf in leaves
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert result.relation_type_entropy == 0.0
        assert set(result.relation_type_mass) == {"parallel"}

    def test_two_axes_equal_mass_is_one(self):
        # Balance positive vs parallel masses: positive prob 0.75 * 1.0 and
        # parallel prob 1.0 * 0.75 are both 0.75 -> equal -> entropy 1.0.
        h = _concept("hub", confidence=1.0)
        a, b = _concept("a"), _concept("b")
        edges = [
            _edge(h.id, a.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.75),
            _edge(h.id, b.id, relation_type=RelationType.PARALLEL,
                  is_explicit=True, probability=1.0),
        ]
        result = compute_network_entropy([h, a, b], edges)
        assert set(result.relation_type_mass) == {"positive", "parallel"}
        pos = result.relation_type_mass["positive"]
        par = result.relation_type_mass["parallel"]
        assert math.isclose(pos, par, abs_tol=1e-12)
        assert math.isclose(result.relation_type_entropy, 1.0, abs_tol=1e-12)

    def test_three_axes_equal_mass_is_one(self):
        # Equalize masses across all three axes by compensating for the
        # axis factors (pos 1.0, neg 0.6, parallel 0.75): choose
        # probabilities so prob*factor == 0.6 for each.
        h = _concept("hub", confidence=1.0)
        leaves = [_concept(f"l{i}") for i in range(3)]
        axes = [RelationType.POSITIVE, RelationType.NEGATIVE,
                RelationType.PARALLEL]
        probs = [0.6, 1.0, 0.8]  # 0.6*1.0, 1.0*0.6, 0.8*0.75 = 0.6 each
        edges = [
            _edge(h.id, leaves[i].id, relation_type=axes[i],
                  is_explicit=True, probability=probs[i])
            for i in range(3)
        ]
        result = compute_network_entropy([h, *leaves], edges)
        masses = result.relation_type_mass
        assert set(masses) == {"positive", "negative", "parallel"}
        assert math.isclose(masses["positive"], 0.6, abs_tol=1e-12)
        assert math.isclose(masses["negative"], 0.6, abs_tol=1e-12)
        assert math.isclose(masses["parallel"], 0.6, abs_tol=1e-12)
        assert math.isclose(result.relation_type_entropy, 1.0, abs_tol=1e-12)

    def test_skewed_axis_mix_strictly_between_zero_and_one(self):
        # Three axes present but unbalanced mass -> 0 < entropy < 1, and
        # below the perfectly-even three-axis maximum.
        h = _concept("hub", confidence=1.0)
        leaves = [_concept(f"l{i}") for i in range(3)]
        axes = [RelationType.POSITIVE, RelationType.NEGATIVE,
                RelationType.PARALLEL]
        probs = [0.9, 0.1, 0.1]
        edges = [
            _edge(h.id, leaves[i].id, relation_type=axes[i],
                  is_explicit=True, probability=probs[i])
            for i in range(3)
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert 0.0 < result.relation_type_entropy < 1.0

    def test_even_mix_exceeds_dominated_mix(self):
        # Direction check: a balanced two-axis mix has strictly higher
        # axis entropy than a mix dominated by one axis.
        def axis_entropy(pos_count, par_count):
            h = _concept("hub", confidence=1.0)
            leaves = []
            edges = []
            idx = 0
            for _ in range(pos_count):
                leaf = _concept(f"p{idx}")
                idx += 1
                leaves.append(leaf)
                edges.append(_edge(h.id, leaf.id,
                                   relation_type=RelationType.POSITIVE,
                                   is_explicit=True, probability=0.75))
            for _ in range(par_count):
                leaf = _concept(f"q{idx}")
                idx += 1
                leaves.append(leaf)
                edges.append(_edge(h.id, leaf.id,
                                   relation_type=RelationType.PARALLEL,
                                   is_explicit=True, probability=1.0))
            return compute_network_entropy(
                [h, *leaves], edges
            ).relation_type_entropy

        balanced = axis_entropy(3, 3)
        dominated = axis_entropy(5, 1)
        assert balanced > dominated > 0.0


# ─────────────────────────────────────────────────────────────────────
# World-level aggregate is the importance-weighted mean of local entropy
# ─────────────────────────────────────────────────────────────────────
class TestImportanceWeightedAggregate:
    def test_aggregate_equals_hand_computed_weighted_mean(self):
        # Two independent hubs with different local entropy and different
        # importance.  Reconstruct the weighted mean by hand.
        #   Hub A: CORE, confidence 1.0 -> importance 1.0 * 1.20 = 1.20
        #          neighbors masses 0.8/0.2 -> local H = _norm_entropy.
        #   Hub B: EMBRYONIC, confidence 0.1 -> importance 0.1 * 0.50 = 0.05
        #          neighbors uniform -> local H = 1.0.
        a = _concept("A", maturity=Maturity.CORE, confidence=1.0)
        b = _concept("B", maturity=Maturity.EMBRYONIC, confidence=0.1)
        a0, a1 = _concept("a0"), _concept("a1")
        b0, b1 = _concept("b0"), _concept("b1")
        edges = [
            _edge(a.id, a0.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.8),
            _edge(a.id, a1.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.2),
            _edge(b.id, b0.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.6),
            _edge(b.id, b1.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.6),
        ]
        result = compute_network_entropy([a, b, a0, a1, b0, b1], edges)

        local_a = _norm_entropy([0.8, 0.2])
        local_b = 1.0
        imp_a = max(1.0, _CONFIDENCE_FLOOR) * _MATURITY[Maturity.CORE]
        imp_b = max(0.1, _CONFIDENCE_FLOOR) * _MATURITY[Maturity.EMBRYONIC]
        expected = (imp_a * local_a + imp_b * local_b) / (imp_a + imp_b)

        assert result.nodes_considered == 2
        assert math.isclose(
            result.avg_network_entropy, expected, abs_tol=1e-12
        )
        # The heavy core hub pulls the average toward its lower local value.
        unweighted = (local_a + local_b) / 2
        assert result.avg_network_entropy < unweighted

    def test_confidence_floor_keeps_low_confidence_node_contributing(self):
        # A single hub with confidence 0 still contributes because of the
        # 0.05 floor -> importance > 0 -> avg equals its local entropy.
        h = _concept("hub", maturity=Maturity.EMBRYONIC, confidence=0.0)
        leaves = [_concept(f"l{i}") for i in range(3)]
        edges = [
            _edge(h.id, leaf.id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.6)
            for leaf in leaves
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert result.nodes_considered == 1
        # Sole considered node -> weighted mean collapses to its local 1.0.
        assert math.isclose(result.avg_network_entropy, 1.0, abs_tol=1e-12)

    def test_high_entropy_concept_count_threshold(self):
        # Three independent hubs: two uniform (local 1.0 >= 0.70 -> high)
        # and one near-deterministic (local <= 0.20 -> low).
        hubs = [
            _concept("h0", maturity=Maturity.ESTABLISHED, confidence=1.0),
            _concept("h1", maturity=Maturity.ESTABLISHED, confidence=1.0),
            _concept("h2", maturity=Maturity.ESTABLISHED, confidence=1.0),
        ]
        concepts = list(hubs)
        edges = []
        # Two uniform hubs.
        for hub in hubs[:2]:
            for j in range(3):
                leaf = _concept(f"{hub.name}-u{j}")
                concepts.append(leaf)
                edges.append(_edge(hub.id, leaf.id,
                                   relation_type=RelationType.POSITIVE,
                                   is_explicit=True, probability=0.6))
        # One sharply skewed hub.
        for prob in (0.98, 0.01, 0.01):
            leaf = _concept(f"h2-s{prob}")
            concepts.append(leaf)
            edges.append(_edge(hubs[2].id, leaf.id,
                               relation_type=RelationType.POSITIVE,
                               is_explicit=True, probability=prob))
        result = compute_network_entropy(concepts, edges)
        assert result.nodes_considered == 3
        assert result.high_entropy_nodes == 2
        assert result.low_entropy_nodes == 1


# ─────────────────────────────────────────────────────────────────────
# Determinism and purity
# ─────────────────────────────────────────────────────────────────────
class TestDeterminism:
    def _build(self):
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=0.8)
        leaves = [_concept(f"l{i}") for i in range(4)]
        edges = [
            _edge(h.id, leaves[0].id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.7),
            _edge(h.id, leaves[1].id, relation_type=RelationType.NEGATIVE,
                  is_explicit=True, probability=0.5),
            _edge(h.id, leaves[2].id, relation_type=RelationType.PARALLEL,
                  is_explicit=False, probability=0.4),
            _edge(h.id, leaves[3].id, relation_type=RelationType.POSITIVE,
                  is_explicit=True, probability=0.3),
        ]
        return [h, *leaves], edges

    def test_repeated_runs_are_bit_identical(self):
        concepts, edges = self._build()
        runs = [
            compute_network_entropy(concepts, edges).model_dump()
            for _ in range(5)
        ]
        for run in runs[1:]:
            assert run == runs[0]

    def test_reordering_inputs_does_not_change_score(self):
        concepts, edges = self._build()
        base = compute_network_entropy(concepts, edges)
        shuffled = compute_network_entropy(
            list(reversed(concepts)), list(reversed(edges))
        )
        assert math.isclose(
            base.avg_network_entropy, shuffled.avg_network_entropy,
            abs_tol=1e-12,
        )
        assert math.isclose(
            base.relation_type_entropy, shuffled.relation_type_entropy,
            abs_tol=1e-12,
        )
        assert base.relation_type_mass == shuffled.relation_type_mass

    def test_function_does_not_mutate_inputs(self):
        concepts, edges = self._build()
        probs_before = [e.probability for e in edges]
        conf_before = [c.confidence for c in concepts]
        compute_network_entropy(concepts, edges)
        assert [e.probability for e in edges] == probs_before
        assert [c.confidence for c in concepts] == conf_before
