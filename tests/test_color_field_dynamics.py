"""Tests for the community-driven color-field mechanism (doc §29 Stage A).

Covers the loop: detect → reconcile → seed → fade.  The tests do not
inspect specific community ids (those are signature-derived and would
churn with member changes); instead they assert on the observable
properties documented in the spec — emergence, stability accumulation,
bridge identification, per-component fade, and noise rejection.
"""

from __future__ import annotations

from world0 import Observation, World
from world0.dynamics.color_diffusion import COMPONENT_EVAPORATE_THRESHOLD


def _ingest_clique(world: World, names: list[str], *, task: str) -> None:
    """Ingest a fully-connected clique under one task."""
    relations = [
        (a, b, "related_to") for i, a in enumerate(names) for b in names[i + 1 :]
    ]
    world.ingest(
        Observation(
            concepts=names,
            relations=relations,
            domain=task,
            task=task,
            source="bench",
        )
    )


def test_dense_cluster_emerges_as_community(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    _ingest_clique(world, ["alpha", "beta", "gamma", "delta"], task="cluster_a")

    result = world.reflect()

    assert len(result.new_communities) >= 1
    communities = world._communities.all()
    assert any(len(c.member_ids) >= 3 for c in communities)


def test_singleton_does_not_form_community(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    world.ingest(
        Observation(concepts=["lonely"], task="solo", source="bench")
    )
    world.ingest(
        Observation(concepts=["another"], task="solo", source="bench")
    )

    result = world.reflect()
    assert result.new_communities == []
    assert world._communities.all() == []


def test_community_stability_grows_across_reflect_cycles(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    members = ["one", "two", "three", "four"]

    _ingest_clique(world, members, task="persistent")
    world.reflect()
    after_first = world._communities.all()
    assert after_first, "first reflect should produce a community"
    initial_stability = max(c.stability for c in after_first)

    # Re-observe the same clique — community must be re-detected and
    # stability incremented, not registered as a new candidate.
    _ingest_clique(world, members, task="persistent")
    second = world.reflect()
    assert second.new_communities == []  # nothing brand-new

    after_second = world._communities.all()
    assert max(c.stability for c in after_second) > initial_stability
    assert any(
        c.is_color_source() for c in after_second
    ), "after two cycles a stable community should be a color source"


def test_stable_community_seeds_color_into_members(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    members = ["red1", "red2", "red3", "red4"]

    _ingest_clique(world, members, task="reds")
    world.reflect()
    _ingest_clique(world, members, task="reds")
    world.reflect()  # community now stable → color source

    sources = world._communities.color_sources()
    assert sources, "expected at least one stable color source"
    color_label = sources[0].id

    seeded = [
        world.concepts.resolve(name) for name in members
    ]
    seeded = [n for n in seeded if n is not None]
    assert any(
        n.domain_strength(color_label) > 0.0 for n in seeded
    ), "stable community should have injected its color into members"


def test_bridge_concept_identified_between_two_communities(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    left = ["L1", "L2", "L3", "L4"]
    right = ["R1", "R2", "R3", "R4"]

    _ingest_clique(world, left, task="left_domain")
    _ingest_clique(world, right, task="right_domain")
    # Bridge node belongs partially to both.
    world.ingest(
        Observation(
            concepts=["bridge"],
            relations=[
                ("bridge", "L1", "related_to"),
                ("bridge", "R1", "related_to"),
            ],
            domain="left_domain",
            task="left_domain",
            source="bench",
        )
    )
    world.ingest(
        Observation(
            concepts=["bridge"],
            relations=[
                ("bridge", "R2", "related_to"),
            ],
            domain="right_domain",
            task="right_domain",
            source="bench",
        )
    )

    # Two reflect cycles for both communities to stabilise + seed.
    world.reflect()
    _ingest_clique(world, left, task="left_domain")
    _ingest_clique(world, right, task="right_domain")
    world.reflect()

    bridge = world.concepts.resolve("bridge")
    assert bridge is not None
    # Bridge picked up colour from at least both task labels via diffusion.
    assert len(bridge.domain_profile) >= 2
    assert bridge.color_purity() < 1.0


def test_fade_drops_components_without_neighborhood_support(tmp_path):
    world = World(store_path=tmp_path / ".world0")

    # Seed a node with a colour that has no graph support.
    world.ingest(
        Observation(
            concepts=["isolated"], domain="ghost", task="ghost", source="bench"
        )
    )
    isolated = world.concepts.resolve("isolated")
    assert isolated is not None
    assert isolated.domain_strength("ghost") > 0.0
    initial = isolated.domain_strength("ghost")

    # Several reflect cycles with no further support for the "ghost"
    # color must drain the component.  Fade applies a baseline floor
    # when neighbourhood evidence is missing, so it cannot stay at the
    # same level forever.
    for _ in range(8):
        world.reflect()

    isolated = world.concepts.resolve("isolated")
    assert isolated is not None
    after = isolated.domain_strength("ghost")
    assert after < initial, (
        f"expected component to fade without support; "
        f"initial={initial}, after={after}"
    )


def test_evaporated_component_is_removed_from_profile(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    world.ingest(
        Observation(
            concepts=["fader"], domain="trace", task="trace", source="bench"
        )
    )
    node = world.concepts.resolve("fader")
    assert node is not None
    # Manually pin the component just above the evaporation threshold so
    # we can exercise the discard path deterministically.
    node.domain_profile = {"trace": COMPONENT_EVAPORATE_THRESHOLD * 1.05}
    world.concepts.mark_dirty(node.id)

    for _ in range(20):
        world.reflect()

    node = world.concepts.resolve("fader")
    assert node is not None
    assert "trace" not in node.domain_profile


def test_status_reports_color_field_diagnostics(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    _ingest_clique(world, ["s1", "s2", "s3", "s4"], task="status_check")
    world.reflect()
    _ingest_clique(world, ["s1", "s2", "s3", "s4"], task="status_check")
    world.reflect()

    status = world.status()
    assert status.total_communities >= 1
    assert status.stable_communities >= 1
    assert 0.0 <= status.avg_color_purity <= 1.0


def test_communities_persist_across_world_reload(tmp_path):
    store = tmp_path / ".world0"
    world = World(store_path=store)
    members = ["p1", "p2", "p3", "p4"]
    _ingest_clique(world, members, task="persisted")
    world.reflect()
    _ingest_clique(world, members, task="persisted")
    world.reflect()

    snapshot_count = len(world._communities.all())
    stable_before = sum(
        1 for c in world._communities.all() if c.is_color_source()
    )
    assert snapshot_count > 0
    assert stable_before > 0

    reopened = World(store_path=store)
    assert len(reopened._communities.all()) == snapshot_count
    assert (
        sum(1 for c in reopened._communities.all() if c.is_color_source())
        == stable_before
    )
