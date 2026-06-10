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


class TestSemanticRelationWeights:
    @pytest.fixture
    def typed_world(self, tmp_path):
        """Two POSITIVE relations with different semantic types.

        Axis-level weights cannot separate them — only semantic-level
        weights can, which is exactly what this class proves.
        """
        w = World(store_path=tmp_path / ".world0")
        for _ in range(6):
            w.ingest(Observation(
                concepts=["service", "database", "cache"],
                relations=[
                    ("service", "database", "depends_on"),  # → dependence
                    ("service", "cache", "supports"),       # → enables
                ],
                task="bootstrap",
                source="t",
            ))
        return w

    def test_semantic_override_separates_same_axis_relations(
        self, typed_world
    ):
        w = typed_world
        dep_lens = Perspective(
            name="dep",
            semantic_relation_weights={"dependence": 1.5, "enables": 0.1},
        )
        ena_lens = Perspective(
            name="ena",
            semantic_relation_weights={"dependence": 0.1, "enables": 1.5},
        )

        proj_dep = w.project(["service"], perspective=dep_lens)
        proj_ena = w.project(["service"], perspective=ena_lens)

        db = w.concepts.resolve("database")
        cache = w.concepts.resolve("cache")

        db_dep = proj_dep.activation_scores.get(db.id, 0.0)
        cache_dep = proj_dep.activation_scores.get(cache.id, 0.0)
        db_ena = proj_ena.activation_scores.get(db.id, 0.0)
        cache_ena = proj_ena.activation_scores.get(cache.id, 0.0)

        # Ordering must invert between the two lenses even though both
        # relations sit on the same POSITIVE axis.
        assert db_dep > cache_dep
        assert cache_ena > db_ena

    def test_semantic_weight_replaces_axis_weight(self):
        p = Perspective(
            relation_type_weights={"positive": 2.0},
            semantic_relation_weights={"dependence": 0.5},
        )
        # Semantic override wins for its relation...
        assert p.weight_for_relation("dependence", "positive", 1.0) == 0.5
        # ...axis override applies to other relations on that axis.
        assert p.weight_for_relation("enables", "positive", 1.0) == 2.0
        # ...global default when neither matches.
        assert p.weight_for_relation("overlap", "parallel", 0.75) == 0.75

    def test_unknown_semantic_keys_are_dropped(self):
        p = Perspective(
            semantic_relation_weights={
                "depends_on": 1.4,        # legacy alias → dependence
                "not_a_relation": 9.9,    # typo → dropped
                "overlap": 0.5,           # canonical → kept
            },
        )
        assert p.semantic_relation_weights == {
            "dependence": 1.4,
            "overlap": 0.5,
        }

    def test_generic_relation_key_is_allowed_explicitly(self):
        p = Perspective(
            semantic_relation_weights={"generic_relation": 0.2},
        )
        assert p.semantic_relation_weights == {"generic_relation": 0.2}

    def test_space_form_alias_is_kept(self):
        # "related to" is the natural-language form of the related_to
        # alias; it must survive (not be dropped as a typo).
        p = Perspective(
            semantic_relation_weights={"related to": 0.3},
        )
        assert p.semantic_relation_weights == {"generic_relation": 0.3}

    def test_space_form_canonical_relation_is_kept(self):
        p = Perspective(
            semantic_relation_weights={"similarity kernel": 1.4},
        )
        assert p.semantic_relation_weights == {"similarity_kernel": 1.4}


class TestPerspectiveProfiles:
    def test_project_accepts_profile_name(self, world):
        proj = world.project(["model serving"], perspective="debug")
        assert len(proj.concepts) >= 1

    def test_unknown_profile_name_raises_with_choices(self, world):
        with pytest.raises(KeyError, match="debug"):
            world.project(["model serving"], perspective="nope")

    def test_builtin_profiles_resolve(self, world):
        for name in ("default", "debug", "design", "research"):
            assert world.perspectives.get(name) is not None
            assert name in world.perspectives.names()

    def test_custom_profile_roundtrip(self, tmp_path):
        w1 = World(store_path=tmp_path / ".world0")
        custom = Perspective(
            name="oncall",
            active_domains=["infra"],
            semantic_relation_weights={"dependence": 1.6},
            render_style="compact",
        )
        w1.perspectives.put(custom)

        # A fresh World over the same store must see the profile.
        w2 = World(store_path=tmp_path / ".world0")
        loaded = w2.perspectives.get("oncall")
        assert loaded is not None
        assert loaded.active_domains == ["infra"]
        assert loaded.semantic_relation_weights == {"dependence": 1.6}
        assert loaded.render_style == "compact"

        assert w2.perspectives.remove("oncall") is True
        assert w2.perspectives.get("oncall") is None
        # Built-ins survive removal attempts.
        assert w2.perspectives.remove("debug") is False
        assert w2.perspectives.get("debug") is not None

    def test_custom_profile_shadows_builtin(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        w.perspectives.put(
            Perspective(name="debug", render_style="detailed")
        )
        assert w.perspectives.get("debug").render_style == "detailed"
