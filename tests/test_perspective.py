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
                ("model serving", "GPU inference", "depends_on"),
                ("model serving", "Triton", "similar_to"),
            ],
            task="bootstrap",
            source="t",
        ))
    return w


class TestRelationTypeWeights:
    def test_debug_perspective_amplifies_dependencies(self, world):
        debug = Perspective(
            name="debug",
            relation_type_weights={"depends_on": 1.5, "similar_to": 0.1},
        )
        analogy = Perspective(
            name="analogy",
            relation_type_weights={"depends_on": 0.1, "similar_to": 1.5},
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
