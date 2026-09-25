"""SQLite storage backend — Store contract, World round-trips, and a
flush-cost comparison against the JSON-per-file backend."""

from __future__ import annotations

import time

import pytest

from world0 import Observation, World
from world0.core import StorageBackend
from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationEdge
from world0.schemas.source import SourceRecord
from world0.store.json_store import JsonStore
from world0.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "world.sqlite")
    yield s
    s.close()


class TestStoreContract:
    def test_satisfies_storage_backend_protocol(self, store):
        assert isinstance(store, StorageBackend)

    def test_concept_round_trip_and_delete(self, store):
        node = ConceptNode(name="Python", description="language", task_profile={"t": 2})
        store.save_concept(node)
        loaded = store.load_concept(node.id)
        assert loaded is not None and loaded.name == "Python"
        assert loaded.task_profile == {"t": 2}
        node.description = "updated"
        store.save_concept(node)  # upsert
        assert store.load_concept(node.id).description == "updated"
        assert [c.id for c in store.load_all_concepts()] == [node.id]
        store.delete_concept(node.id)
        assert store.load_concept(node.id) is None

    def test_batches_are_atomic_upserts(self, store):
        nodes = [ConceptNode(name=f"c{i}") for i in range(20)]
        store.save_concepts_batch(nodes)
        assert len(store.load_all_concepts()) == 20
        nodes[0].name = "renamed"
        store.save_concepts_batch(nodes[:5])
        assert len(store.load_all_concepts()) == 20
        assert store.load_concept(nodes[0].id).name == "renamed"
        store.delete_concepts_batch([n.id for n in nodes[:10]])
        assert len(store.load_all_concepts()) == 10

    def test_relation_source_state_round_trip(self, store):
        edge = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence")
        store.save_relation(edge)
        assert store.load_relation(edge.id).semantic_relation == "dependence"
        store.save_relations_batch([edge])
        assert len(store.load_all_relations()) == 1
        store.delete_relations_batch([edge.id])
        assert store.load_all_relations() == []

        src = SourceRecord(id="src1", raw_text="hello world", content_hash="h", source="s")
        store.save_source(src)
        assert store.load_source("src1").raw_text == "hello world"
        assert [s.id for s in store.load_all_sources()] == ["src1"]
        assert store.load_source("nope") is None

        assert store.load_state() == {}
        store.save_state({"tick": 7, "hebbian_pending": {"a|b": 1}})
        assert store.load_state() == {"tick": 7, "hebbian_pending": {"a|b": 1}}
        store.save_state({"tick": 8})
        assert store.load_state() == {"tick": 8}

    def test_reopening_the_file_sees_the_data(self, tmp_path):
        path = tmp_path / "w.db"
        first = SqliteStore(path)
        first.save_concept(ConceptNode(name="persisted"))
        first.close()
        second = SqliteStore(path)
        assert [c.name for c in second.load_all_concepts()] == ["persisted"]
        second.close()


class TestWorldOnSqlite:
    def test_backend_selection(self, tmp_path):
        assert isinstance(World(store_path=tmp_path / "a.sqlite")._store, SqliteStore)
        assert isinstance(World(store_path=tmp_path / "b.db")._store, SqliteStore)
        assert isinstance(World(store_path=tmp_path / "c")._store, JsonStore)
        assert isinstance(World(store_path=tmp_path / "d", backend="sqlite")._store, SqliteStore)
        assert isinstance(World(store_path=tmp_path / "e.db", backend="json")._store, JsonStore)
        with pytest.raises(ValueError):
            World(store_path=tmp_path / "f", backend="parquet")

    def test_full_cycle_survives_restart(self, tmp_path):
        path = tmp_path / "world.sqlite"
        w = World(store_path=path)
        for _ in range(5):
            w.ingest(
                Observation(
                    concepts=["FastAPI", "Python", "PostgreSQL"],
                    relations=[("FastAPI", "Python", "depends_on")],
                    task="backend",
                    source="s",
                )
            )
        w.ingest(Observation(concepts=["FastAPI", "PostgreSQL"], source="s"))  # hebbian pending
        w.clock.advance(10)
        w.reflect()
        before = w.project(["FastAPI"], task="backend").render()
        tick = w.clock.tick
        pending = w._hebbian.snapshot()

        again = World(store_path=path)
        assert again.clock.tick == tick
        assert again._hebbian.snapshot() == pending
        assert again.status().total_concepts == 3
        assert again.project(["FastAPI"], task="backend").render() == before
        assert again.status().last_reflect is not None

    def test_prune_deletes_rows(self, tmp_path):
        w = World(store_path=tmp_path / "w.sqlite")
        w.ingest(Observation(concepts=["ephemeral"], source="s"))
        assert len(w._store.load_all_concepts()) == 1
        w.clock.advance(500)
        first = w.reflect()  # decays below the prune threshold → fading → pruned
        w.clock.advance(500)
        second = w.reflect()
        assert first.pruned_concepts or second.pruned_concepts
        assert w._store.load_all_concepts() == []


class TestFlushCost:
    def test_sqlite_flush_is_not_slower_than_json_at_scale(self, tmp_path, capsys):
        def run(world: World) -> float:
            names = [f"c{i}" for i in range(40)]
            start = time.perf_counter()
            for i in range(60):
                world.ingest(
                    Observation(
                        concepts=names,
                        relations=[(names[i % 40], names[(i + 1) % 40], "depends_on")],
                        task="t",
                        source="s",
                    )
                )
            return time.perf_counter() - start

        json_time = run(World(store_path=tmp_path / "json"))
        sqlite_time = run(World(store_path=tmp_path / "w.sqlite"))
        print(f"\nflush cost, 60 ingests × 40 concepts: json={json_time:.2f}s sqlite={sqlite_time:.2f}s")
        # Same work, one transaction per flush: must not be materially slower.
        assert sqlite_time < json_time * 1.5
