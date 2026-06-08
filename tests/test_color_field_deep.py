"""Deep, deterministic tests for color-field dynamics + community detection.

These tests drive the engines directly through the in-memory Protocol
fakes (``world0.core.test_doubles``) so the numeric behaviour is fully
deterministic and isolated from JSON I/O.  They complement the
World-level integration tests in ``tests/test_color_field_dynamics.py``
and ``tests/test_domain_color_diffusion.py`` by exercising the
component-level invariants of:

- ``ColorDiffusionEngine``: seed_and_diffuse / fade_step / settle /
  seed_from_communities
- ``ConceptNode.color_purity`` / ``ConceptNode.is_bridge``
- ``CommunityDetector`` label propagation
- ``CommunityManager`` stability lifecycle, color-source promotion,
  snapshot round-trip and pruning

Where exact constants are implementation detail we assert invariants
and inequalities; only stable, semantically meaningful values
(purity = 0.5 for an even mix, seed = DOMAIN_SEED_STRENGTH, etc.) are
pinned exactly.
"""

from __future__ import annotations

import pytest

from world0.communities.manager import (
    MIN_COLOR_SOURCE_STABILITY,
    CommunityManager,
)
from world0.core.test_doubles import (
    FakeConceptStore,
    FakeRelationStore,
    make_concept,
    make_edge,
)
from world0.dynamics.color_diffusion import (
    COMPONENT_EVAPORATE_THRESHOLD,
    DOMAIN_SEED_STRENGTH,
    ColorDiffusionEngine,
    normalize_domain_label,
)
from world0.dynamics.community import CommunityDetector
from world0.schemas.community import Community, signature_id
from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationType


# ── helpers ──────────────────────────────────────────────────────────


def _clique_edges(names: list[str], *, weight: float = 0.9) -> list:
    """Fully-connected positive edges over ``names``."""
    return [
        make_edge(a, b, relation_type=RelationType.POSITIVE, weight=weight)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]


def _engine(
    nodes: list[ConceptNode], edges: list
) -> tuple[ColorDiffusionEngine, FakeConceptStore, FakeRelationStore]:
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=edges)
    return ColorDiffusionEngine(cs, rs), cs, rs


def _detector(
    nodes: list[ConceptNode], edges: list
) -> tuple[CommunityDetector, FakeConceptStore, FakeRelationStore]:
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=edges)
    return CommunityDetector(cs, rs), cs, rs


# ════════════════════════════════════════════════════════════════════
# seed_and_diffuse
# ════════════════════════════════════════════════════════════════════


def test_seed_gives_domain_color_to_seeded_node() -> None:
    a = make_concept("a", concept_id="a")
    eng, cs, _ = _engine([a], [])

    eng.seed_and_diffuse(["a"], domain_label="backend")

    assert cs.get("a").domain_profile.get("backend") == pytest.approx(
        DOMAIN_SEED_STRENGTH
    )
    assert cs.get("a").domain == "backend"


def test_color_spreads_to_neighbor_with_attenuation() -> None:
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    eng, cs, _ = _engine(
        [a, b],
        [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)],
    )

    eng.seed_and_diffuse(["a"], domain_label="backend")

    seed = cs.get("a").domain_strength("backend")
    spread = cs.get("b").domain_strength("backend")
    # Neighbor receives the color, strictly attenuated below the seed.
    assert spread > 0.0
    assert spread < seed


def test_isolated_node_unaffected_by_diffusion() -> None:
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    isolated = make_concept("iso", concept_id="iso")
    eng, cs, _ = _engine(
        [a, b, isolated],
        [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)],
    )

    eng.seed_and_diffuse(["a"], domain_label="backend")

    # No edge touches the isolated node → no color load at all.
    assert cs.get("iso").domain_profile == {}


def test_multiple_domains_coexist_in_profile() -> None:
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    c = make_concept("c", concept_id="c")
    # b sits between two seeds, so it accumulates both domain colors.
    eng, cs, _ = _engine(
        [a, b, c],
        [
            make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.95),
            make_edge("c", "b", relation_type=RelationType.POSITIVE, weight=0.95),
        ],
    )

    eng.seed_and_diffuse(["a"], domain_label="backend")
    eng.seed_and_diffuse(["c"], domain_label="ml")

    profile = cs.get("b").domain_profile
    assert profile.get("backend", 0.0) > 0.0
    assert profile.get("ml", 0.0) > 0.0
    assert len(profile) >= 2


def test_seed_and_diffuse_ignores_generic_and_empty_labels() -> None:
    a = make_concept("a", concept_id="a")
    eng, cs, _ = _engine([a], [])

    eng.seed_and_diffuse(["a"], domain_label="")
    eng.seed_and_diffuse(["a"], domain_label="knowledge intake")  # generic

    assert cs.get("a").domain_profile == {}


def test_seed_and_diffuse_skips_unknown_ids() -> None:
    a = make_concept("a", concept_id="a")
    eng, cs, _ = _engine([a], [])

    # No KeyError / crash on an id that is not in the store.
    eng.seed_and_diffuse(["ghost"], domain_label="backend")
    assert cs.get("a").domain_profile == {}


# ════════════════════════════════════════════════════════════════════
# ConceptNode.color_purity
# ════════════════════════════════════════════════════════════════════


def test_color_purity_no_profile_is_one() -> None:
    assert ConceptNode(name="x").color_purity() == 1.0


def test_color_purity_single_color_is_one() -> None:
    n = ConceptNode(name="x", domain_profile={"a": 0.42})
    assert n.color_purity() == 1.0


def test_color_purity_even_mix_is_half() -> None:
    n = ConceptNode(name="x", domain_profile={"a": 0.4, "b": 0.4})
    assert n.color_purity() == pytest.approx(0.5)


def test_color_purity_matches_max_over_total() -> None:
    profile = {"a": 0.6, "b": 0.3, "c": 0.1}
    n = ConceptNode(name="x", domain_profile=profile)
    expected = max(profile.values()) / sum(profile.values())
    assert n.color_purity() == pytest.approx(expected)


def test_color_purity_zero_total_is_one() -> None:
    # Degenerate all-zero profile is treated as trivially pure.
    n = ConceptNode(name="x", domain_profile={"a": 0.0, "b": 0.0})
    assert n.color_purity() == 1.0


# ════════════════════════════════════════════════════════════════════
# ConceptNode.is_bridge
# ════════════════════════════════════════════════════════════════════


def test_is_bridge_false_without_profile() -> None:
    assert ConceptNode(name="x").is_bridge() is False


def test_is_bridge_false_for_single_color() -> None:
    n = ConceptNode(name="x", domain_profile={"a": 0.7})
    assert n.is_bridge() is False


def test_is_bridge_true_for_two_comparable_colors() -> None:
    # second/top = 0.4/0.5 = 0.8 >= 0.55, and 0.4 >= min_second.
    n = ConceptNode(name="x", domain_profile={"a": 0.5, "b": 0.4})
    assert n.is_bridge() is True


def test_is_bridge_false_when_second_below_ratio() -> None:
    # second/top = 0.2/0.5 = 0.4 < 0.55.
    n = ConceptNode(name="x", domain_profile={"a": 0.5, "b": 0.2})
    assert n.is_bridge() is False


def test_is_bridge_false_when_second_near_zero() -> None:
    # second 0.02 < min_second floor → noise, not a bridge.
    n = ConceptNode(name="x", domain_profile={"a": 0.5, "b": 0.02})
    assert n.is_bridge() is False


def test_is_bridge_threshold_parameters_honored() -> None:
    n = ConceptNode(name="x", domain_profile={"a": 0.5, "b": 0.3})
    # ratio 0.6: above default 0.55 → bridge; raise min_ratio above 0.6 → not.
    assert n.is_bridge() is True
    assert n.is_bridge(min_ratio=0.7) is False
    # A higher absolute floor for the second color also disqualifies it.
    assert n.is_bridge(min_second=0.31) is False


# ════════════════════════════════════════════════════════════════════
# fade_step
# ════════════════════════════════════════════════════════════════════


def test_fade_drops_unsupported_color() -> None:
    n = make_concept("x", concept_id="x")
    n.domain_profile = {"ghost": 0.5}
    eng, cs, _ = _engine([n], [])  # no neighbors → no support

    before = cs.get("x").domain_strength("ghost")
    touched = eng.fade_step()
    after = cs.get("x").domain_strength("ghost")

    assert touched == 1
    assert after < before


def test_fade_repeated_eventually_evaporates_component() -> None:
    n = make_concept("x", concept_id="x")
    # Pin just above the evaporation threshold so the discard path runs
    # deterministically within a bounded number of fade steps.
    n.domain_profile = {"trace": COMPONENT_EVAPORATE_THRESHOLD * 1.05}
    eng, cs, _ = _engine([n], [])

    for _ in range(50):
        if "trace" not in cs.get("x").domain_profile:
            break
        eng.fade_step()

    assert "trace" not in cs.get("x").domain_profile


def test_fade_preserves_neighborhood_supported_color() -> None:
    # Node 'a' carries two equally-strong colors; the neighbor only
    # supports 'kept'.  The supported color must fade less than the
    # unsupported one, since fade relaxes each component toward what the
    # neighborhood supplies (the per-node median is the normaliser).
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    a.domain_profile = {"kept": 0.5, "ghost": 0.5}
    b.domain_profile = {"kept": 0.6}
    eng, cs, _ = _engine(
        [a, b],
        [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)],
    )

    eng.fade_step()

    profile = cs.get("a").domain_profile
    assert profile["kept"] > profile["ghost"]
    # The unsupported color decayed below where it started.
    assert profile["ghost"] < 0.5


def test_fade_no_op_on_empty_profiles() -> None:
    a = make_concept("a", concept_id="a")
    eng, _, _ = _engine([a], [])
    assert eng.fade_step() == 0


# ════════════════════════════════════════════════════════════════════
# settle
# ════════════════════════════════════════════════════════════════════


def test_settle_spreads_color_to_neighbors() -> None:
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    a.domain_profile = {"backend": 0.5}
    eng, cs, _ = _engine(
        [a, b],
        [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)],
    )

    eng.settle(steps=1)

    assert cs.get("b").domain_strength("backend") > 0.0
    # Blend keeps every component within the [0, 1] normalization range.
    assert 0.0 <= cs.get("b").domain_strength("backend") <= 1.0


def test_settle_no_sources_is_harmless() -> None:
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    eng, cs, _ = _engine(
        [a, b],
        [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)],
    )

    eng.settle(steps=2)

    assert cs.get("a").domain_profile == {}
    assert cs.get("b").domain_profile == {}


# ════════════════════════════════════════════════════════════════════
# CommunityDetector — label propagation
# ════════════════════════════════════════════════════════════════════


def test_detector_finds_two_cliques() -> None:
    left = ["l1", "l2", "l3", "l4"]
    right = ["r1", "r2", "r3", "r4"]
    nodes = [make_concept(n, concept_id=n) for n in left + right]
    det, _, _ = _detector(nodes, _clique_edges(left) + _clique_edges(right))

    coms = det.detect()

    assert len(coms) == 2
    member_sets = [set(c.member_ids) for c in coms]
    assert set(left) in member_sets
    assert set(right) in member_sets
    # Every community has at least one core member.
    assert all(c.core_ids for c in coms)


def test_detector_clique_id_is_signature_of_members() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    det, _, _ = _detector(nodes, _clique_edges(names))

    coms = det.detect()

    assert len(coms) == 1
    assert coms[0].id == signature_id(names)


def test_detector_ignores_subthreshold_cluster() -> None:
    # Two-node pair is below MIN_COMMUNITY_SIZE (3) and must be dropped,
    # even though a third node keeps the graph at >= min_size overall.
    names = ["a", "b", "c"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    edges = [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)]
    det, _, _ = _detector(nodes, edges)

    coms = det.detect()

    # The connected pair (size 2) is noise; the singleton 'c' too.
    assert all(len(c.member_ids) >= 3 for c in coms)
    assert {"a", "b"} not in [set(c.member_ids) for c in coms]


def test_detector_empty_graph_returns_nothing() -> None:
    det, _, _ = _detector([], [])
    assert det.detect() == []


def test_detector_singletons_form_no_community() -> None:
    nodes = [make_concept(n, concept_id=n) for n in ["x", "y", "z", "w"]]
    det, _, _ = _detector(nodes, [])  # nodes but no relations
    assert det.detect() == []


def test_detector_is_deterministic_across_runs() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    det, _, _ = _detector(nodes, _clique_edges(names))

    first = det.detect()
    second = det.detect()

    assert [c.id for c in first] == [c.id for c in second]
    assert [sorted(c.member_ids) for c in first] == [
        sorted(c.member_ids) for c in second
    ]


# ════════════════════════════════════════════════════════════════════
# CommunityManager — stability lifecycle
# ════════════════════════════════════════════════════════════════════


def test_stability_increments_across_detect_cycles() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    det, _, _ = _detector(nodes, _clique_edges(names))
    mgr = CommunityManager(det)

    mgr.detect_and_update()
    first = max(c.stability for c in mgr.all())
    mgr.detect_and_update()
    second = max(c.stability for c in mgr.all())

    assert first == 1
    assert second == first + 1


def test_community_becomes_color_source_only_after_threshold() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    det, _, _ = _detector(nodes, _clique_edges(names))
    mgr = CommunityManager(det)

    result1 = mgr.detect_and_update()
    # First cycle: registered as candidate, stability 1 < threshold.
    assert MIN_COLOR_SOURCE_STABILITY == 2
    assert mgr.color_sources() == []
    assert result1.new  # brand-new this cycle
    assert result1.color_sources == []

    result2 = mgr.detect_and_update()
    # Second cycle: re-detected, stability crosses threshold → source.
    assert mgr.color_sources()
    assert result2.new == []  # nothing brand-new
    assert result2.matched  # the existing community was reinforced
    assert result2.color_sources


def test_snapshot_round_trip_preserves_communities_and_stability() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    det, _, _ = _detector(nodes, _clique_edges(names))
    mgr = CommunityManager(det)
    mgr.detect_and_update()
    mgr.detect_and_update()

    snap = mgr.snapshot()
    restored = CommunityManager.from_snapshot(snap, det)

    original = {c.id: c.stability for c in mgr.all()}
    rebuilt = {c.id: c.stability for c in restored.all()}
    assert rebuilt == original
    assert len(original) == 1
    # Color-source eligibility survives the round-trip.
    assert {c.id for c in restored.color_sources()} == {
        c.id for c in mgr.color_sources()
    }


def test_from_snapshot_none_is_empty_manager() -> None:
    det, _, _ = _detector([], [])
    mgr = CommunityManager.from_snapshot(None, det)
    assert mgr.all() == []
    assert mgr.color_sources() == []


def test_unseen_community_decays_and_is_pruned() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=_clique_edges(names))
    det = CommunityDetector(cs, rs)
    mgr = CommunityManager(det)

    mgr.detect_and_update()
    mgr.detect_and_update()
    assert max(c.stability for c in mgr.all()) == 2

    # The cluster disappears from the graph entirely.
    rs._edges.clear()

    r1 = mgr.detect_and_update()
    assert r1.pruned == []  # stability 2 → 1, not yet zero
    assert max(c.stability for c in mgr.all()) == 1

    r2 = mgr.detect_and_update()
    assert r2.pruned  # stability hit zero → dropped
    assert mgr.all() == []


def test_initial_communities_passed_to_constructor() -> None:
    com = Community(
        id="com_seed",
        member_ids=["a", "b", "c"],
        core_ids=["a"],
        stability=MIN_COLOR_SOURCE_STABILITY,
    )
    det, _, _ = _detector([], [])
    mgr = CommunityManager(det, initial=[com])

    assert [c.id for c in mgr.all()] == ["com_seed"]
    assert [c.id for c in mgr.color_sources()] == ["com_seed"]


# ════════════════════════════════════════════════════════════════════
# seed_from_communities
# ════════════════════════════════════════════════════════════════════


def test_seed_from_communities_dyes_members() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=_clique_edges(names))
    det = CommunityDetector(cs, rs)
    mgr = CommunityManager(det)
    mgr.detect_and_update()
    mgr.detect_and_update()

    sources = mgr.color_sources()
    assert sources
    eng = ColorDiffusionEngine(cs, rs)

    touched = eng.seed_from_communities(sources)

    label = sources[0].id
    assert touched == len(names)
    # Community id is a valid color label (survives normalization).
    assert normalize_domain_label(label) == label
    assert any(cs.get(n).domain_strength(label) > 0.0 for n in names)


def test_seed_from_communities_ignores_unstable_sources() -> None:
    names = ["a1", "a2", "a3", "a4"]
    nodes = [make_concept(n, concept_id=n) for n in names]
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=_clique_edges(names))
    det = CommunityDetector(cs, rs)
    mgr = CommunityManager(det)
    mgr.detect_and_update()  # only one cycle → stability 1, not a source

    eng = ColorDiffusionEngine(cs, rs)
    # Pass the raw (not-yet-stable) communities directly.
    touched = eng.seed_from_communities(mgr.all())

    assert touched == 0
    assert all(cs.get(n).domain_profile == {} for n in names)


def test_seed_from_communities_empty_list() -> None:
    eng, _, _ = _engine([], [])
    assert eng.seed_from_communities([]) == 0


def test_seed_from_communities_core_outranks_ring() -> None:
    # A 6-member community: core members get a stronger seed than the
    # wider ring (before diffusion blends them together).
    names = [f"m{i}" for i in range(6)]
    nodes = [make_concept(n, concept_id=n) for n in names]
    cs = FakeConceptStore(seed=nodes)
    rs = FakeRelationStore(seed=_clique_edges(names))
    det = CommunityDetector(cs, rs)
    mgr = CommunityManager(det)
    mgr.detect_and_update()
    mgr.detect_and_update()

    source = mgr.color_sources()[0]
    label = source.id
    core_ids = set(source.core_ids)
    ring_ids = [n for n in names if n not in core_ids]
    assert core_ids and ring_ids

    eng = ColorDiffusionEngine(cs, rs)
    eng.seed_from_communities([source], diffuse=False)

    core_strength = max(cs.get(cid).domain_strength(label) for cid in core_ids)
    ring_strength = max(cs.get(rid).domain_strength(label) for rid in ring_ids)
    assert core_strength > ring_strength


# ════════════════════════════════════════════════════════════════════
# Edge cases — empty / single node / single edge worlds
# ════════════════════════════════════════════════════════════════════


def test_empty_world_all_engines_quiet() -> None:
    eng, _, _ = _engine([], [])
    eng.seed_and_diffuse([], domain_label="backend")
    assert eng.fade_step() == 0
    eng.settle()

    det, _, _ = _detector([], [])
    assert det.detect() == []


def test_single_node_no_community_but_seedable() -> None:
    a = make_concept("a", concept_id="a")
    eng, cs, _ = _engine([a], [])
    eng.seed_and_diffuse(["a"], domain_label="backend")
    assert cs.get("a").domain_strength("backend") == pytest.approx(
        DOMAIN_SEED_STRENGTH
    )

    det, _, _ = _detector([a], [])
    assert det.detect() == []


def test_single_edge_diffuses_but_forms_no_community() -> None:
    a = make_concept("a", concept_id="a")
    b = make_concept("b", concept_id="b")
    edges = [make_edge("a", "b", relation_type=RelationType.POSITIVE, weight=0.9)]
    eng, cs, _ = _engine([a, b], edges)

    eng.seed_and_diffuse(["a"], domain_label="backend")
    assert cs.get("b").domain_strength("backend") > 0.0

    det, _, _ = _detector([a, b], edges)
    # Two nodes are below MIN_COMMUNITY_SIZE.
    assert det.detect() == []
