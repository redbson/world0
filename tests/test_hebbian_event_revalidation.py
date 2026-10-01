"""Hebbian revalidation happens at the event, so the edge set is a function of the stream.

Round 29 (analysis doc §7.32; paper Theorem 3.7).  A generic co-occurrence
edge's association changes only when one of its endpoints is mentioned, so
``HebbianEngine.learn()`` re-judges exactly the generic edges incident to
the concepts it just counted.  Reflect's whole-store ``revalidate()`` is a
backstop that finds nothing after an ordinary stream, and the set of
generic edges at any point no longer depends on when (or whether) reflect
ran — the last reflect-only mutation named in the paper's residuals.
"""

from __future__ import annotations

import random

import pytest

from world0 import Observation, World

POOL = [f"c{i}" for i in range(40)]


def _generic_pairs(world: World) -> set[frozenset[str]]:
    return {
        frozenset((world.concepts.get(r.source_id).name, world.concepts.get(r.target_id).name))
        for r in world.relations.all()
        if r.semantic_relation == "generic_relation" and not r.is_explicit
    }


def _stream(seed: int, n: int = 300) -> list[Observation]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.3:  # a topic: concepts that really belong together
            concepts = rng.sample(POOL[:6], 4)
        else:
            concepts = rng.sample(POOL, 5)
        out.append(Observation(concepts=concepts, source="s"))
    return out


class TestEventTimeRevalidation:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_the_generic_edge_set_does_not_depend_on_the_reflect_cadence(self, tmp_path, seed):
        finals = {}
        for every in (1, 50, None):
            w = World(store_path=tmp_path / f"w{every}")
            for i, obs in enumerate(_stream(seed), start=1):
                w.ingest(obs)
                if every and i % every == 0:
                    w.reflect(light=True)
            finals[every] = _generic_pairs(w)
        assert finals[1] == finals[50] == finals[None]
        assert finals[None], "the stream must leave some well-associated edges"

    def test_reflect_has_nothing_left_to_revalidate(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        for obs in _stream(4):
            w.ingest(obs)
        result = w.reflect(light=True)
        assert result.stale_relations == []

    def test_the_edge_goes_at_the_event_that_drops_its_association(self, tmp_path):
        """x and y are linked on two shared mentions; as they then appear apart
        the association collapses and the edge is removed by an ingest, which
        reports it — not by a later reflect."""
        w = World(store_path=tmp_path / ".world0")
        w.ingest(Observation(concepts=["x", "y"], source="s"))
        w.ingest(Observation(concepts=["x", "y"], source="s"))
        assert _generic_pairs(w) == {frozenset({"x", "y"})}
        reported = None
        for i in range(40):
            r = w.ingest(Observation(concepts=["x", f"m{i}"], source="s"))
            if r.stale_relations:
                reported = (i, r.stale_relations)
                break
        assert reported is not None, "the edge should be removed during the stream"
        assert reported[1] == ["x ↔ y"]
        assert _generic_pairs(w) == set()

    def test_a_well_associated_edge_is_not_touched_by_other_mentions(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        for _ in range(15):
            w.ingest(Observation(concepts=["a", "b"], source="s"))
        for i in range(10):  # a appears elsewhere a little: association stays well above the cut
            w.ingest(Observation(concepts=["a", f"z{i}"], source="s"))
        assert frozenset({"a", "b"}) in _generic_pairs(w)
