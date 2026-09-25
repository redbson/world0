"""Projection stability — small, irrelevant perturbations must not move a
projection (AGENTS.md: "Include tests for ambiguity, boundary cases, and
projection stability").

Probed on the cognitive benchmark world (analysis doc §7.10): twelve
unrelated observations, reflect() after 0–100 idle ticks and twenty
alternating unrelated mentions all leave the ranked projection byte-for-
byte identical; a single in-domain re-mention only reorders an exact tie.
These tests pin that down so a future change to activation, decay or MMR
cannot make the projection jittery without failing here.
"""

from __future__ import annotations

import pytest

from tests._cognitive_benchmark import (
    build_cognitive_benchmark_world,
    ranked_projection_names,
)
from world0 import Observation, World


def _ranked(world: World) -> list[str]:
    return ranked_projection_names(
        world.project(["model serving"], task="ml training", max_concepts=6, max_depth=4)
    )


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


@pytest.fixture
def world(tmp_path):
    return build_cognitive_benchmark_world(World(store_path=tmp_path / ".world0"))


class TestProjectionStability:
    def test_unrelated_observations_do_not_move_the_projection(self, world):
        base = _ranked(world)
        for _ in range(12):
            world.ingest(
                Observation(
                    concepts=["weather", "cooking"],
                    relations=[("weather", "cooking", "influences")],
                    source="s",
                )
            )
            assert _ranked(world) == base

    def test_reflect_cadence_does_not_move_the_projection(self, world):
        base = _ranked(world)
        for idle in (0, 1, 5, 20, 100):
            world.clock.advance(idle)
            world.reflect()
            assert _ranked(world) == base

    def test_alternating_unrelated_mentions_do_not_oscillate(self, world):
        prev = _ranked(world)
        changes = 0
        for k in range(20):
            world.ingest(Observation(concepts=["A" if k % 2 else "B"], source="s"))
            current = _ranked(world)
            changes += current != prev
            prev = current
        assert changes == 0

    def test_single_in_domain_re_mention_changes_at_most_one_slot(self, world):
        base = _ranked(world)
        for name in ("PyTorch", "optimizer", "gradient descent", "monitoring"):
            world.ingest(Observation(concepts=[name], source="s"))
            current = _ranked(world)
            # Same seed keeps the same top slot; the set may swap at most one
            # concept and the shared prefix stays.
            assert current[0] == base[0]
            assert _jaccard(current, base) >= 5 / 7
            assert current[:2] == base[:2]
