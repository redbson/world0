"""Seeds are the Agent's explicit focus and are always in their projection.

Probe (analysis doc §7.13): before the seed-first rule, ``deployment``
was missing from ``project(["PyTorch", "deployment"], task="ml training")``
at every size, and six seeds at ``max_concepts=3`` kept non-seed
neighbours over seeds — MMR treated seeds as ordinary candidates and
traded them for "diversity" or task fit.
"""

from __future__ import annotations

import pytest

from tests._cognitive_benchmark import build_cognitive_benchmark_world
from world0 import Observation, World


def _names(projection) -> list[str]:
    return [c.name for c in projection.concepts]


@pytest.fixture
def world(tmp_path):
    return build_cognitive_benchmark_world(World(store_path=tmp_path / ".world0"))


class TestSeedsAlwaysProjected:
    def test_cross_domain_seeds_survive_a_task(self, world):
        for n in (3, 4, 6):
            got = _names(world.project(["PyTorch", "deployment"], task="ml training", max_concepts=n, max_depth=2))
            assert {"PyTorch", "deployment"} <= set(got)
            assert len(got) == n

    def test_seeds_come_first_and_are_capped_by_score(self, world):
        seeds = ["model serving", "PyTorch", "training pipeline", "neural network", "gradient descent", "optimizer"]
        p = world.project(seeds, task="ml training", max_concepts=3, max_depth=2)
        got = _names(p)
        assert len(got) == 3 and set(got) <= set(seeds)
        scores = {c.name: p.activation_scores[c.id] for c in p.concepts}
        top3 = sorted(seeds, key=lambda s: -world.concepts.resolve(s).confidence)[:3]
        assert set(got) == set(top3) or sorted(scores.values(), reverse=True) == list(scores.values())

    def test_clique_seed_and_weak_seed_both_present(self, tmp_path):
        w = World(store_path=tmp_path / ".w")
        for _ in range(10):
            w.ingest(Observation(concepts=["a", "b", "c", "d", "e"], source="s"))
        w.ingest(Observation(concepts=["z", "y"], relations=[("z", "y", "depends_on")], source="s"))
        got = _names(w.project(["a", "z"], max_concepts=4))
        assert got[:2] == ["a", "z"] or got[:2] == ["z", "a"]
        assert len(got) == 4

    def test_seeds_precede_neighbours_in_order(self, world):
        got = _names(world.project(["PyTorch", "training pipeline"], task="ml training", max_concepts=6, max_depth=2))
        assert set(got[:2]) == {"PyTorch", "training pipeline"}
