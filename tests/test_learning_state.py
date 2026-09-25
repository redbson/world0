"""Learning-state persistence — Hebbian counters live in a separate store
record and are written on an amortised schedule at scale.

Profile (analysis doc §7.15): in a 2 000-concept world, serialising the
pending pairs and mention statistics into ``state.json`` on every
observation was 76 % of ingest cost (per-observation cost grew from
22 ms to 44 ms).  Now a small world stays restart-exact (written every
observation), a large one writes the record at most every
``LEARNING_PERSIST_EVERY`` observations plus at ``reflect()`` and
``close()``, and the per-observation state write carries only the clock
and reflect metadata.
"""

from __future__ import annotations

import json

import pytest

from world0 import Observation, World
from world0.store.json_store import JsonStore
from world0.store.sqlite_store import SqliteStore
from world0.world.facade import LEARNING_EAGER_LIMIT, LEARNING_PERSIST_EVERY


class _Counting:
    """Wrap a store and count learning-state writes."""

    def __init__(self, inner):
        self._inner = inner
        self.learning_writes = 0
        self.state_writes = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def save_learning_state(self, state):
        self.learning_writes += 1
        self._inner.save_learning_state(state)

    def save_state(self, state):
        self.state_writes += 1
        self._inner.save_state(state)


def _grow_past_eager_limit(world: World) -> None:
    """Ingest non-repeating 12-concept observations (30 pending pairs each,
    never converting to edges) until the learning record is "large"."""
    k = 0
    while (
        world._hebbian.pending_pairs + world._hebbian.tracked_concepts
        <= LEARNING_EAGER_LIMIT
    ):
        world.ingest(Observation(concepts=[f"g{k + j}" for j in range(12)], source="s"))
        k += 12
        assert k < 5000, "eager limit never exceeded"


def _wrap(world: World) -> _Counting:
    counting = _Counting(world._store)
    world._store = counting
    return counting


class TestSmallWorldIsRestartExact:
    def test_pending_pair_survives_restart_without_close(self, tmp_path):
        root = tmp_path / "w"
        first = World(store_path=root)
        first.ingest(Observation(concepts=["x", "y"], source="s"))
        second = World(store_path=root)
        assert second._hebbian.pending_pairs == 1
        assert second._hebbian.mentions(second.concepts.resolve("x").id) == 1

    def test_learning_record_is_separate_from_state(self, tmp_path):
        root = tmp_path / "w"
        w = World(store_path=root)
        w.ingest(Observation(concepts=["x", "y"], source="s"))
        state = json.loads((root / "state.json").read_text())
        learning = json.loads((root / "learning.json").read_text())
        assert "hebbian_pending" not in state and "hebbian_stats" not in state
        assert learning["hebbian_pending"] and learning["hebbian_stats"]["observations"] == 1

    def test_legacy_state_with_inline_counters_is_migrated(self, tmp_path):
        root = tmp_path / "w"
        store = JsonStore(root)
        store.save_state(
            {
                "tick": 3,
                "hebbian_pending": {"a|b": 1},
                "hebbian_stats": {"observations": 3, "mentions": {"a": 2, "b": 1}},
            }
        )
        w = World(store_path=root)
        assert w._hebbian.pending_pairs == 1
        assert w._hebbian.mentions("a") == 2
        w.ingest(Observation(concepts=["c"], source="s"))
        state = json.loads((root / "state.json").read_text())
        assert "hebbian_pending" not in state
        assert w._store.load_learning_state()["hebbian_stats"]["mentions"]["a"] == 2


class TestLargeWorldIsAmortised:
    @pytest.fixture
    def large(self, tmp_path):
        w = World(store_path=tmp_path / "w.sqlite")
        _grow_past_eager_limit(w)
        return w

    def test_learning_writes_are_amortised(self, large):
        counting = _wrap(large)
        for i in range(3 * LEARNING_PERSIST_EVERY):
            large.ingest(Observation(concepts=[f"z{i}", f"z{i + 1}"], source="s"))
        assert counting.state_writes == 3 * LEARNING_PERSIST_EVERY  # clock every time
        assert 2 <= counting.learning_writes <= 4

    def test_reflect_and_close_force_the_record(self, large):
        counting = _wrap(large)
        large.ingest(Observation(concepts=["p", "q"], source="s"))
        before = counting.learning_writes
        large.reflect(light=True)
        assert counting.learning_writes == before + 1
        large.ingest(Observation(concepts=["p", "q"], source="s"))
        large.close()
        assert counting.learning_writes == before + 2

    def test_close_makes_restart_exact(self, tmp_path):
        path = tmp_path / "w.sqlite"
        with World(store_path=path) as w:
            _grow_past_eager_limit(w)
            w.ingest(Observation(concepts=["fresh1", "fresh2"], source="s"))
            snapshot = w._hebbian.snapshot()
            stats = w._hebbian.stats_snapshot()
        again = World(store_path=path)
        assert again._hebbian.snapshot() == snapshot
        assert again._hebbian.stats_snapshot() == stats


class TestSqliteLearningRecord:
    def test_round_trip(self, tmp_path):
        store = SqliteStore(tmp_path / "w.sqlite")
        assert store.load_learning_state() == {}
        store.save_learning_state({"hebbian_pending": {"a|b": 2}})
        assert store.load_learning_state() == {"hebbian_pending": {"a|b": 2}}
        assert store.load_state() == {}  # separate record
        store.close()
