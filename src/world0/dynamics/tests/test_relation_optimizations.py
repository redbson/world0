"""Relation-usage optimizations from docs/world0-relation-optimization.md.

Three mechanisms, each tested against Protocol fakes only:

1. Seed specificity — rarer seeds (fewer distinct evidence sources)
   keep full activation; ubiquitous seeds are blended down.
2. Fan dilution — high-fan hubs spread less activation per edge.
3. SimilarityLinker — explicit ``similarity_kernel`` edges between
   near-duplicate concepts, never between single-token signatures.
"""

from __future__ import annotations

from datetime import datetime, timezone

from world0.core import SimilarityLinkerP
from world0.core.test_doubles import (
    FakeConceptStore,
    FakeRelationStore,
    make_concept,
    make_edge,
)
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.coefficients import ActivationConfig
from world0.dynamics.hebbian import COOCCURRENCE_THRESHOLD, HebbianEngine
from world0.dynamics.refinement import RelationRefiner
from world0.dynamics.similarity import (
    SIMILARITY_EDGE_THRESHOLD,
    SimilarityLinker,
)
from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationType


# ── Seed specificity ─────────────────────────────────────────────────


def _with_sources(node: ConceptNode, n: int) -> ConceptNode:
    for i in range(n):
        node.record_source_ref(source_id=f"src-{i}", source=f"doc {i}")
    return node


def test_rare_seed_outscores_ubiquitous_seed() -> None:
    rare = _with_sources(make_concept("rare", confidence=0.8), 1)
    common = _with_sources(make_concept("common", confidence=0.8), 8)
    cs = FakeConceptStore(seed=[rare, common])
    rs = FakeRelationStore()

    engine = ActivationEngine(cs, rs)
    scores = engine.activate([rare.id, common.id], record=False)

    assert scores[rare.id] > scores[common.id]


def test_seed_specificity_neutral_for_single_seed() -> None:
    common = _with_sources(make_concept("common", confidence=0.8), 8)
    cs = FakeConceptStore(seed=[common])
    rs = FakeRelationStore()

    engine = ActivationEngine(cs, rs)
    scores = engine.activate([common.id], record=False)

    assert scores[common.id] == common.confidence


def test_seed_specificity_disabled_by_config() -> None:
    rare = _with_sources(make_concept("rare", confidence=0.8), 1)
    common = _with_sources(make_concept("common", confidence=0.8), 8)
    cs = FakeConceptStore(seed=[rare, common])
    rs = FakeRelationStore()

    engine = ActivationEngine(
        cs, rs, config=ActivationConfig(seed_specificity_weight=0.0)
    )
    scores = engine.activate([rare.id, common.id], record=False)

    assert scores[rare.id] == scores[common.id]


# ── Fan dilution ─────────────────────────────────────────────────────


def _star(center_name: str, spokes: int):
    """A hub with ``spokes`` identical edges; returns (nodes, edges)."""
    hub = make_concept(center_name, confidence=0.8)
    nodes = [hub]
    edges = []
    for i in range(spokes):
        leaf = make_concept(f"{center_name}-leaf-{i}", confidence=0.8)
        nodes.append(leaf)
        edges.append(make_edge(hub.id, leaf.id, weight=0.6))
    return hub, nodes, edges


def test_high_fan_hub_spreads_less_per_edge() -> None:
    small_hub, small_nodes, small_edges = _star("small", 2)
    big_hub, big_nodes, big_edges = _star("big", 12)
    cs = FakeConceptStore(seed=small_nodes + big_nodes)
    rs = FakeRelationStore(seed=small_edges + big_edges)

    engine = ActivationEngine(cs, rs)
    small_scores = engine.activate([small_hub.id], record=False)
    big_scores = engine.activate([big_hub.id], record=False)

    small_leaf = small_nodes[1].id
    big_leaf = big_nodes[1].id
    assert small_scores[small_leaf] > big_scores[big_leaf]


def test_fan_dilution_inactive_below_threshold() -> None:
    hub, nodes, edges = _star("hub", 5)  # below default threshold of 7
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=edges)

    now = datetime.now(timezone.utc)
    diluted = ActivationEngine(cs, rs)
    plain = ActivationEngine(
        cs, rs, config=ActivationConfig(fan_dilution_strength=0.0)
    )
    leaf = nodes[1].id
    assert (
        diluted.activate([hub.id], record=False, now=now)[leaf]
        == plain.activate([hub.id], record=False, now=now)[leaf]
    )


def test_fan_dilution_never_flips_inhibitory() -> None:
    # ACT-R's smax - ln(fan) goes negative past fan ≈ 7; our smooth
    # dilution must keep every propagated score strictly positive.
    hub, nodes, edges = _star("mega", 40)
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=edges)

    scores = ActivationEngine(cs, rs).activate([hub.id], record=False)
    leaves = [n.id for n in nodes[1:]]
    assert all(scores.get(leaf, 0.0) > 0.0 for leaf in leaves)


# ── SimilarityLinker ─────────────────────────────────────────────────


class SimilarityConceptStore(FakeConceptStore):
    """Fake store with a canned ``find_similar`` response."""

    def __init__(self, *, seed=None, similar=None) -> None:
        super().__init__(seed=seed)
        self._similar = similar or []

    def find_similar(self, text, *, domain="", min_similarity=0.3, limit=5):
        return [
            (node, sim) for node, sim in self._similar if sim >= min_similarity
        ][:limit]


def test_linker_satisfies_protocol() -> None:
    cs = FakeConceptStore()
    rs = FakeRelationStore()
    assert isinstance(SimilarityLinker(cs, rs), SimilarityLinkerP)


def test_linker_creates_similarity_kernel_edge() -> None:
    a = make_concept("activation engine core", confidence=0.5)
    b = make_concept("spreading activation engine", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 0.62)])
    rs = FakeRelationStore()

    created = SimilarityLinker(cs, rs).link([a.id], provenance="t")

    assert len(created) == 1
    edge = rs.get(created[0])
    assert edge is not None
    assert edge.semantic_relation == "similarity_kernel"
    assert edge.relation_type == RelationType.PARALLEL
    assert edge.is_explicit is False
    # Similarity score is encoded as the edge weight.
    assert abs(edge.weight - 0.62) < 1e-9


def test_linker_skips_already_connected_pair() -> None:
    a = make_concept("graph projection view", confidence=0.5)
    b = make_concept("graph projection engine", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 0.9)])
    rs = FakeRelationStore(seed=[make_edge(a.id, b.id, weight=0.4)])

    assert SimilarityLinker(cs, rs).link([a.id]) == []


def test_linker_skips_single_token_signatures() -> None:
    # chain_2 / chain_3 both tokenize to {"chain"} — identical
    # signatures, but no compositional evidence of near-duplication.
    a = make_concept("chain_2", confidence=0.5)
    b = make_concept("chain_3", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 1.0)])
    rs = FakeRelationStore()

    assert SimilarityLinker(cs, rs).link([a.id]) == []


def test_linker_threshold_filters_weak_similarity() -> None:
    a = make_concept("vector database index", confidence=0.5)
    b = make_concept("vector clock protocol", confidence=0.5)
    weak = SIMILARITY_EDGE_THRESHOLD - 0.1
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, weak)])
    rs = FakeRelationStore()

    assert SimilarityLinker(cs, rs).link([a.id]) == []


# ── SimilarityLinker, LLM-judged mode ────────────────────────────────


class FakeLLM:
    """Canned ``complete_json`` provider recording its calls."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def test_llm_judge_links_pair_below_lexical_threshold() -> None:
    # "vector database" vs "vector store": Jaccard ≈ 0.33, invisible to
    # the lexical rule — the whole point of the judge.
    a = make_concept("vector database", confidence=0.5)
    b = make_concept("vector store", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 0.33)])
    rs = FakeRelationStore()
    llm = FakeLLM(
        '{"links": [{"index": 0, "relation": "approximate_equivalence",'
        ' "confidence": 0.9}]}'
    )

    created = SimilarityLinker(cs, rs, llm=llm).link([a.id])

    assert len(created) == 1
    edge = rs.get(created[0])
    assert edge.semantic_relation == "approximate_equivalence"
    assert edge.relation_type == RelationType.PARALLEL
    assert edge.is_explicit is False
    # Confidence 0.9 capped at the implicit-evidence ceiling.
    assert abs(edge.weight - 0.7) < 1e-9
    # The judge saw both concepts.
    assert "vector store" in llm.calls[0][1]


def test_llm_judge_rejects_lexical_lookalikes() -> None:
    # Perfect token overlap, different concepts — the judge says no
    # and no edge is created despite similarity 1.0.
    a = make_concept("payment gateway v2", confidence=0.5)
    b = make_concept("payment gateway v3", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 1.0)])
    rs = FakeRelationStore()
    llm = FakeLLM('{"links": []}')

    assert SimilarityLinker(cs, rs, llm=llm).link([a.id]) == []
    assert len(rs.all()) == 0


def test_llm_failure_falls_back_to_lexical_rule() -> None:
    a = make_concept("graph projection view", confidence=0.5)
    b = make_concept("graph projection engine", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 0.8)])
    rs = FakeRelationStore()

    class BrokenLLM:
        def complete_json(self, system: str, user: str) -> str:
            raise RuntimeError("provider down")

    created = SimilarityLinker(cs, rs, llm=BrokenLLM()).link([a.id])

    assert len(created) == 1
    edge = rs.get(created[0])
    assert edge.semantic_relation == "similarity_kernel"
    assert abs(edge.weight - 0.7) < 1e-9  # 0.8 capped


def test_llm_unknown_relation_collapses_to_similarity_kernel() -> None:
    a = make_concept("retry policy handler", confidence=0.5)
    b = make_concept("retry handler module", confidence=0.5)
    cs = SimilarityConceptStore(seed=[a, b], similar=[(b, 0.4)])
    rs = FakeRelationStore()
    llm = FakeLLM(
        '{"links": [{"index": 0, "relation": "same_thing", "confidence": 0.6}]}'
    )

    created = SimilarityLinker(cs, rs, llm=llm).link([a.id])

    assert len(created) == 1
    assert rs.get(created[0]).semantic_relation == "similarity_kernel"


# ── Hebbian, LLM-typed mode ──────────────────────────────────────────


def _crossed_pair(hebbian: HebbianEngine, id_a: str, id_b: str) -> list[str]:
    """Drive a pair over the co-occurrence threshold; return created ids."""
    created: list[str] = []
    for _ in range(COOCCURRENCE_THRESHOLD):
        created.extend(hebbian.learn([id_a, id_b], provenance="t"))
    return created


def test_hebbian_llm_types_discovered_relation_with_direction() -> None:
    a = make_concept("unit testing", confidence=0.5)
    b = make_concept("code quality", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()
    llm = FakeLLM(
        '{"relation": "enables", "direction": "a_to_b", "confidence": 0.8}'
    )
    h = HebbianEngine(rs, cs, llm=llm)

    created = _crossed_pair(h, a.id, b.id)

    assert len(created) == 1
    edge = rs.get(created[0])
    assert edge.semantic_relation == "enables"
    assert edge.relation_type == RelationType.POSITIVE
    assert edge.is_explicit is False
    # Pair order follows the learn() argument order, so "concept_a"
    # is the first id passed and a_to_b keeps that orientation.
    assert (edge.source_id, edge.target_id) == (a.id, b.id)


def test_hebbian_llm_direction_b_to_a_flips_endpoints() -> None:
    a = make_concept("alpha system", confidence=0.5)
    b = make_concept("beta system", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()
    llm = FakeLLM(
        '{"relation": "dependence", "direction": "b_to_a", "confidence": 0.7}'
    )
    h = HebbianEngine(rs, cs, llm=llm)

    created = _crossed_pair(h, a.id, b.id)

    edge = rs.get(created[0])
    assert (edge.source_id, edge.target_id) == (b.id, a.id)


def test_hebbian_llm_none_verdict_creates_no_relation() -> None:
    a = make_concept("coincidental one", confidence=0.5)
    b = make_concept("coincidental two", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()
    llm = FakeLLM('{"relation": "none", "direction": "a_to_b", "confidence": 0.9}')
    h = HebbianEngine(rs, cs, llm=llm)

    assert _crossed_pair(h, a.id, b.id) == []
    assert len(rs.all()) == 0
    # Counter cleared — the pair can re-accumulate and be re-judged.
    assert h.pending_pairs == 0


def test_hebbian_llm_failure_falls_back_to_untyped_parallel() -> None:
    a = make_concept("left part", confidence=0.5)
    b = make_concept("right part", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()

    class BrokenLLM:
        def complete_json(self, system: str, user: str) -> str:
            raise RuntimeError("provider down")

    h = HebbianEngine(rs, cs, llm=BrokenLLM())
    created = _crossed_pair(h, a.id, b.id)

    assert len(created) == 1
    edge = rs.get(created[0])
    assert edge.relation_type == RelationType.PARALLEL
    assert edge.semantic_relation == "generic_relation"


def test_hebbian_without_llm_keeps_legacy_untyped_behavior() -> None:
    a = make_concept("plain a", confidence=0.5)
    b = make_concept("plain b", confidence=0.5)
    rs = FakeRelationStore()
    h = HebbianEngine(rs)

    created = _crossed_pair(h, a.id, b.id)

    assert len(created) == 1
    assert rs.get(created[0]).relation_type == RelationType.PARALLEL


# ── RelationRefiner (reflect-time generic governance) ────────────────


def _generic_edge(rs: FakeRelationStore, a, b, reinforcement: int = 0):
    edge = make_edge(a.id, b.id, weight=0.4)
    edge.semantic_relation = "generic_relation"
    edge.reinforcement_count = reinforcement
    rs._edges[edge.id] = edge
    return edge


def test_refiner_retypes_generic_edge_in_place() -> None:
    a = make_concept("unit testing", confidence=0.5)
    b = make_concept("code quality", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()
    edge = _generic_edge(rs, a, b)
    llm = FakeLLM(
        '{"relation": "enables", "direction": "a_to_b", "confidence": 0.8}'
    )

    retyped, co_attention = RelationRefiner(cs, rs, llm).refine()

    assert retyped == ["unit testing → enables → code quality"]
    assert co_attention == []
    refreshed = rs.get(edge.id)
    assert refreshed.semantic_relation == "enables"
    assert refreshed.relation_type == RelationType.POSITIVE
    # Operational evidence preserved — refinement names, not re-scores.
    assert refreshed.weight == 0.4


def test_refiner_flips_direction_on_b_to_a() -> None:
    a = make_concept("service alpha", confidence=0.5)
    b = make_concept("service beta", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()
    edge = _generic_edge(rs, a, b)
    llm = FakeLLM(
        '{"relation": "dependence", "direction": "b_to_a", "confidence": 0.7}'
    )

    retyped, _ = RelationRefiner(cs, rs, llm).refine()

    refreshed = rs.get(edge.id)
    assert (refreshed.source_id, refreshed.target_id) == (b.id, a.id)
    assert retyped == ["service beta → dependence → service alpha"]


def test_refiner_none_verdict_demotes_to_co_attention() -> None:
    a = make_concept("noise one", confidence=0.5)
    b = make_concept("noise two", confidence=0.5)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore()
    edge = _generic_edge(rs, a, b)
    llm = FakeLLM('{"relation": "none", "direction": "a_to_b", "confidence": 0.9}')

    retyped, co_attention = RelationRefiner(cs, rs, llm).refine()

    assert retyped == []
    assert co_attention == ["noise one ↔ noise two"]
    refreshed = rs.get(edge.id)
    assert refreshed.refinement_state == "co_attention_only"
    assert refreshed.semantic_relation == "generic_relation"
    # Already-demoted edges are not re-judged next cycle.
    retyped2, co2 = RelationRefiner(cs, rs, llm).refine()
    assert retyped2 == [] and co2 == []


def test_refiner_budget_prefers_most_reinforced() -> None:
    cs_nodes = [make_concept(f"n{i}", confidence=0.5) for i in range(4)]
    cs = FakeConceptStore(seed=cs_nodes)
    rs = FakeRelationStore()
    weak = _generic_edge(rs, cs_nodes[0], cs_nodes[1], reinforcement=1)
    strong = _generic_edge(rs, cs_nodes[2], cs_nodes[3], reinforcement=9)
    llm = FakeLLM(
        '{"relation": "overlap", "direction": "a_to_b", "confidence": 0.6}'
    )

    RelationRefiner(cs, rs, llm, budget=1).refine()

    assert rs.get(strong.id).semantic_relation == "overlap"
    assert rs.get(weak.id).semantic_relation == "generic_relation"
