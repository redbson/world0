"""Tests: persistence — write → reload → state fully recovered."""

import pytest

from world0 import Observation, World


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / ".world0"


class TestPersistence:
    def test_concepts_survive_reload(self, store_path):
        """Concepts persisted and reloaded correctly."""
        w1 = World(store_path=store_path)
        w1.ingest(Observation(
            concepts=["Python", "ML"],
            relations=[("Python", "ML", "supports")],
            source="test",
        ))
        py_id = w1.concepts.resolve("Python").id
        py_conf = w1.concepts.resolve("Python").confidence

        # New World instance, same store
        w2 = World(store_path=store_path)
        py = w2.concepts.resolve("Python")
        assert py is not None
        assert py.id == py_id
        assert py.confidence == pytest.approx(py_conf)

    def test_relations_survive_reload(self, store_path):
        w1 = World(store_path=store_path)
        w1.ingest(Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "depends_on")],
            source="test",
        ))
        assert len(w1.relations) == 1

        w2 = World(store_path=store_path)
        assert len(w2.relations) == 1
        a = w2.concepts.resolve("A")
        b = w2.concepts.resolve("B")
        rels = w2.relations.find_any_between(a.id, b.id)
        assert len(rels) == 1
        assert rels[0].relation_type.value == "depends_on"

    def test_reinforcement_accumulates_across_sessions(self, store_path):
        """Multiple sessions reinforce the same concept."""
        # Session 1
        w1 = World(store_path=store_path)
        w1.ingest(Observation(concepts=["Python"], source="s1"))
        w1.ingest(Observation(concepts=["Python"], source="s1"))

        # Session 2
        w2 = World(store_path=store_path)
        w2.ingest(Observation(concepts=["Python"], source="s2"))
        w2.ingest(Observation(concepts=["Python"], source="s2"))

        py = w2.concepts.resolve("Python")
        assert py.activation_count == 4

    def test_reflect_state_persists(self, store_path):
        w1 = World(store_path=store_path)
        w1.ingest(Observation(concepts=["X"], source="test"))
        w1.reflect()

        w2 = World(store_path=store_path)
        assert w2.status().last_reflect is not None
