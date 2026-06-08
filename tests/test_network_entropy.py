"""Tests for the read-only network-entropy diagnostic.

Entropy here is a structural property of the typed concept-relation
graph, so these tests are fully deterministic — no LLM, no storage.
"""

from __future__ import annotations

import math

from world0.metrics.entropy import NetworkEntropy, compute_network_entropy
from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge, RelationType


def _concept(name: str, **kw) -> ConceptNode:
    return ConceptNode(name=name, **kw)


def _edge(src: str, tgt: str, **kw) -> RelationEdge:
    edge = RelationEdge(source_id=src, target_id=tgt, **kw)
    # Pin probability so entropy is a function of structure, not of the
    # semantic-profile defaults applied in the model validator.
    if "probability" in kw:
        edge.probability = kw["probability"]
    return edge


class TestEdgeCases:
    def test_empty_world_is_zero(self):
        result = compute_network_entropy([], [])
        assert isinstance(result, NetworkEntropy)
        assert result.avg_network_entropy == 0.0
        assert result.nodes_considered == 0

    def test_concepts_without_relations_is_zero(self):
        concepts = [_concept("a"), _concept("b")]
        result = compute_network_entropy(concepts, [])
        assert result.avg_network_entropy == 0.0
        assert result.nodes_considered == 0

    def test_single_neighbor_has_no_branching(self):
        # a-b: each node has exactly one neighbor -> no branching uncertainty.
        a, b = _concept("a"), _concept("b")
        edge = _edge(a.id, b.id, is_explicit=True, probability=0.8)
        result = compute_network_entropy([a, b], [edge])
        assert result.avg_network_entropy == 0.0
        assert result.nodes_considered == 0
        assert result.isolated_nodes == 2

    def test_self_relation_is_ignored(self):
        a = _concept("a")
        edge = _edge(a.id, a.id, is_explicit=True, probability=0.9)
        result = compute_network_entropy([a], [edge])
        assert result.avg_network_entropy == 0.0

    def test_dangling_endpoint_is_ignored(self):
        a, b = _concept("a"), _concept("b")
        # target points at a concept not in the list.
        edge = _edge(a.id, "ghost", is_explicit=True, probability=0.9)
        result = compute_network_entropy([a, b], [edge])
        assert result.avg_network_entropy == 0.0


class TestLocalEntropy:
    def test_uniform_hub_is_max_entropy(self):
        # Hub `h` connects to three leaves with equal mass -> normalized
        # local entropy of the hub is 1.0.  Leaves each have one neighbor
        # so they don't contribute.
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        leaves = [_concept(f"leaf{i}") for i in range(3)]
        edges = [
            _edge(h.id, leaf.id, is_explicit=True, probability=0.8)
            for leaf in leaves
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert result.nodes_considered == 1
        assert math.isclose(result.avg_network_entropy, 1.0, abs_tol=1e-9)
        assert result.high_entropy_nodes == 1

    def test_skewed_hub_has_lower_entropy(self):
        # One dominant neighbor + two weak ones -> entropy below the
        # uniform case.
        h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
        leaves = [_concept(f"leaf{i}") for i in range(3)]
        edges = [
            _edge(h.id, leaves[0].id, is_explicit=True, probability=0.9),
            _edge(h.id, leaves[1].id, is_explicit=False, probability=0.1),
            _edge(h.id, leaves[2].id, is_explicit=False, probability=0.1),
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert 0.0 < result.avg_network_entropy < 1.0

    def test_uniform_beats_skewed(self):
        def hub_entropy(probs, explicit):
            h = _concept("hub", maturity=Maturity.ESTABLISHED, confidence=1.0)
            leaves = [_concept(f"l{i}") for i in range(len(probs))]
            edges = [
                _edge(h.id, leaves[i].id, is_explicit=explicit, probability=p)
                for i, p in enumerate(probs)
            ]
            return compute_network_entropy(
                [h, *leaves], edges
            ).avg_network_entropy

        uniform = hub_entropy([0.6, 0.6, 0.6], True)
        skewed = hub_entropy([0.9, 0.1, 0.1], True)
        assert uniform > skewed


class TestRelationTypeEntropy:
    def test_single_axis_is_zero(self):
        h = _concept("hub", confidence=1.0)
        leaves = [_concept(f"leaf{i}") for i in range(3)]
        edges = [
            _edge(
                h.id,
                leaf.id,
                relation_type=RelationType.POSITIVE,
                is_explicit=True,
                probability=0.7,
            )
            for leaf in leaves
        ]
        result = compute_network_entropy([h, *leaves], edges)
        assert result.relation_type_entropy == 0.0
        assert set(result.relation_type_mass) == {"positive"}

    def test_balanced_axes_raise_type_entropy(self):
        h = _concept("hub", confidence=1.0)
        leaves = [_concept(f"leaf{i}") for i in range(3)]
        axes = [
            RelationType.POSITIVE,
            RelationType.NEGATIVE,
            RelationType.PARALLEL,
        ]
        edges = [
            _edge(
                h.id,
                leaves[i].id,
                relation_type=axes[i],
                is_explicit=True,
                probability=0.7,
            )
            for i in range(3)
        ]
        result = compute_network_entropy([h, *leaves], edges)
        # Three axes present -> meaningful (non-zero) type diversity.
        assert result.relation_type_entropy > 0.0
        assert len(result.relation_type_mass) == 3


class TestImportanceWeighting:
    def test_core_concept_dominates_world_average(self):
        # A focused core hub (low local entropy) and a diffuse embryonic
        # hub (high local entropy).  Core importance should pull the
        # world average toward the focused value.
        core = _concept("core", maturity=Maturity.CORE, confidence=1.0)
        emb = _concept("emb", maturity=Maturity.EMBRYONIC, confidence=0.1)
        core_leaves = [_concept(f"c{i}") for i in range(3)]
        emb_leaves = [_concept(f"e{i}") for i in range(3)]
        edges = [
            # Core hub: skewed -> low local entropy.
            _edge(core.id, core_leaves[0].id, is_explicit=True, probability=0.95),
            _edge(core.id, core_leaves[1].id, is_explicit=False, probability=0.05),
            _edge(core.id, core_leaves[2].id, is_explicit=False, probability=0.05),
            # Embryonic hub: uniform -> high local entropy.
            _edge(emb.id, emb_leaves[0].id, is_explicit=True, probability=0.6),
            _edge(emb.id, emb_leaves[1].id, is_explicit=True, probability=0.6),
            _edge(emb.id, emb_leaves[2].id, is_explicit=True, probability=0.6),
        ]
        concepts = [core, emb, *core_leaves, *emb_leaves]
        result = compute_network_entropy(concepts, edges)
        assert result.nodes_considered == 2
        # Weighted mean must sit below the unweighted mean because the
        # low-entropy node carries far more importance.
        assert result.avg_network_entropy < 0.5


def test_function_is_pure_no_mutation():
    a, b, c = _concept("a"), _concept("b"), _concept("c")
    edges = [
        _edge(a.id, b.id, is_explicit=True, probability=0.8),
        _edge(a.id, c.id, is_explicit=True, probability=0.8),
    ]
    probs_before = [e.probability for e in edges]
    conf_before = [c.confidence for c in (a, b, c)]
    compute_network_entropy([a, b, c], edges)
    assert [e.probability for e in edges] == probs_before
    assert [x.confidence for x in (a, b, c)] == conf_before
