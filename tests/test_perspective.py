"""Tests: Perspective as a first-class context for projection.

The same concept-world must produce different projections under
different perspectives — that is the operational meaning of
"context-sensitive, not globally static".
"""

from __future__ import annotations

import pytest

from world0 import Observation, Perspective, World


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    # Shared substrate: hub, a dependency, an analogy.
    for _ in range(6):
        w.ingest(Observation(
            concepts=["model serving", "GPU inference", "Triton"],
            relations=[
                ("model serving", "GPU inference", "positive"),
                ("model serving", "Triton", "parallel"),
            ],
            task="bootstrap",
            source="t",
        ))
    return w


class TestRelationTypeWeights:
    def test_debug_perspective_amplifies_dependencies(self, world):
        debug = Perspective(
            name="debug",
            relation_type_weights={"positive": 1.5, "parallel": 0.1},
        )
        analogy = Perspective(
            name="analogy",
            relation_type_weights={"positive": 0.1, "parallel": 1.5},
        )

        proj_debug = world.project(["model serving"], perspective=debug)
        proj_analogy = world.project(["model serving"], perspective=analogy)

        gpu = world.concepts.resolve("GPU inference")
        triton = world.concepts.resolve("Triton")

        gpu_under_debug = proj_debug.activation_scores.get(gpu.id, 0.0)
        triton_under_debug = proj_debug.activation_scores.get(triton.id, 0.0)
        gpu_under_analogy = proj_analogy.activation_scores.get(gpu.id, 0.0)
        triton_under_analogy = proj_analogy.activation_scores.get(
            triton.id, 0.0
        )

        assert gpu_under_debug > triton_under_debug
        assert triton_under_analogy > gpu_under_analogy
        # And the ordering must actually invert between perspectives —
        # otherwise the weights are not doing real work.
        assert gpu_under_debug > gpu_under_analogy
        assert triton_under_analogy > triton_under_debug


class TestDomainAffinity:
    def test_active_domain_boosts_seed_score(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        w.ingest(Observation(
            concepts=["latency"],
            domain="infra",
            source="t",
        ))
        w.ingest(Observation(
            concepts=["latency"],
            domain="infra",
            source="t",
        ))

        # With domain in focus
        focus = Perspective(name="infra", active_domains=["infra"])
        with_focus = w.project(["latency"], perspective=focus)

        # Without
        neutral = Perspective(name="neutral")
        without_focus = w.project(["latency"], perspective=neutral)

        node = w.concepts.resolve("latency")
        s_with = with_focus.activation_scores.get(node.id, 0.0)
        s_without = without_focus.activation_scores.get(node.id, 0.0)
        assert s_with >= s_without
        # Boost is 1.3x by default; it must be strictly greater when
        # the concept has a real domain-profile entry for that domain.
        if node.domain_profile.get("infra", 0.0) > 0:
            assert s_with > s_without


class TestPerspectiveBackcompat:
    def test_bare_task_string_still_works(self, world):
        """Passing task='...' without a perspective keeps old behavior."""
        proj = world.project(["model serving"], task="bootstrap")
        assert proj.task == "bootstrap"
        assert len(proj.concepts) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Semantic-level weights, direction on directed edges only, profiles
# ═══════════════════════════════════════════════════════════════════════


def _score(projection, world, name: str) -> float:
    node = world.concepts.resolve(name)
    return projection.activation_scores.get(node.id, 0.0) if node else 0.0


@pytest.fixture
def structured_world(tmp_path):
    """hub depends_on Dep; hub contains Part; hub similar_to Twin;
    Upstream depends_on hub.  All explicit, no Hebbian edges needed."""
    w = World(store_path=tmp_path / ".world0")
    for _ in range(6):
        w.ingest(Observation(
            concepts=["hub", "Dep", "Part", "Twin", "Upstream"],
            relations=[
                ("hub", "Dep", "depends_on"),
                ("hub", "Part", "contains"),
                ("hub", "Twin", "similar_to"),
                ("Upstream", "hub", "depends_on"),
            ],
            source="t",
        ))
    return w


class TestSemanticRelationWeights:
    def test_semantic_key_distinguishes_relations_on_the_same_axis(self, structured_world):
        """dependence and inclusion are both positive-axis; a perspective
        must be able to tell them apart."""
        w = structured_world
        dep_view = Perspective(relation_type_weights={"dependence": 1.4, "inclusion": 0.3})
        tax_view = Perspective(relation_type_weights={"inclusion": 1.4, "dependence": 0.3})
        p_dep = w.project(["hub"], perspective=dep_view)
        p_tax = w.project(["hub"], perspective=tax_view)
        assert _score(p_dep, w, "Dep") > _score(p_dep, w, "Part")
        assert _score(p_tax, w, "Part") > _score(p_tax, w, "Dep")

    def test_alias_key_matches_canonical_name(self, structured_world):
        w = structured_world
        by_alias = w.project(["hub"], perspective=Perspective(relation_type_weights={"depends_on": 0.2}))
        by_name = w.project(["hub"], perspective=Perspective(relation_type_weights={"dependence": 0.2}))
        assert _score(by_alias, w, "Dep") == pytest.approx(_score(by_name, w, "Dep"))

    def test_semantic_key_wins_over_axis_key(self, structured_world):
        w = structured_world
        mixed = Perspective(relation_type_weights={"positive": 0.2, "dependence": 1.4})
        p = w.project(["hub"], perspective=mixed)
        assert _score(p, w, "Dep") > _score(p, w, "Part")

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown relation label"):
            Perspective(relation_type_weights={"co_occurs": 0.1})


class TestDirectionAppliesToDirectedEdgesOnly:
    def test_parallel_edge_ignores_direction_weights(self, structured_world):
        w = structured_world
        neutral = w.project(["hub"], perspective=Perspective())
        forward_only = w.project(
            ["hub"], perspective=Perspective(direction_weights={"forward": 1.0, "backward": 0.1})
        )
        backward_only = w.project(
            ["hub"], perspective=Perspective(direction_weights={"forward": 0.1, "backward": 1.0})
        )
        # Twin is reached over a symmetric similar_to edge: unaffected.
        assert _score(forward_only, w, "Twin") == pytest.approx(_score(neutral, w, "Twin"))
        assert _score(backward_only, w, "Twin") == pytest.approx(_score(neutral, w, "Twin"))
        # Dep (hub → Dep) drops under backward-only; Upstream (Upstream → hub)
        # drops under forward-only.
        assert _score(backward_only, w, "Dep") < _score(neutral, w, "Dep")
        assert _score(forward_only, w, "Upstream") < _score(neutral, w, "Upstream")

    def test_is_directed_follows_axis(self):
        from world0.schemas.relation import RelationEdge
        assert RelationEdge(source_id="a", target_id="b", semantic_relation="dependence").is_directed
        assert RelationEdge(source_id="a", target_id="b", semantic_relation="exclusion").is_directed
        assert not RelationEdge(source_id="a", target_id="b", semantic_relation="overlap").is_directed


class TestPerspectiveProfiles:
    def test_profiles_are_listed_and_resolvable(self):
        from world0.perspectives import PROFILES, get_perspective, perspective_names
        assert {"default", "dependency_map", "impact_map", "taxonomy", "analogy", "contrast"} <= set(perspective_names())
        for name in perspective_names():
            assert get_perspective(name).name == name
        assert get_perspective("Taxonomy", task="place it").task == "place it"
        assert PROFILES["taxonomy"].task == ""  # the registry entry is untouched
        with pytest.raises(KeyError):
            get_perspective("nonsense")

    def test_dependency_and_impact_maps_read_the_same_edges_in_opposite_directions(self, structured_world):
        w = structured_world
        dep = w.project(["hub"], perspective="dependency_map")
        imp = w.project(["hub"], perspective="impact_map")
        assert _score(dep, w, "Dep") > _score(dep, w, "Upstream")
        assert _score(imp, w, "Upstream") > _score(imp, w, "Dep")

    def test_taxonomy_and_analogy_foreground_different_neighbors(self, structured_world):
        w = structured_world
        tax = w.project(["hub"], perspective="taxonomy")
        ana = w.project(["hub"], perspective="analogy")
        assert _score(tax, w, "Part") > _score(tax, w, "Twin")
        assert _score(ana, w, "Twin") > _score(ana, w, "Part")

    def test_named_profile_keeps_bare_task(self, structured_world):
        p = structured_world.project(["hub"], perspective="taxonomy", task="curation")
        assert p.task == "curation"
