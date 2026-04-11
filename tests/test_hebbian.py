"""Tests: Hebbian learning — co-activation discovers and strengthens relations.

Hebbian relations require co-occurrence >= COOCCURRENCE_THRESHOLD (default 2)
before a RelationEdge is created. This prevents noise from single observations.
"""

import pytest

from world0 import Observation, World
from world0.dynamics.hebbian import COOCCURRENCE_THRESHOLD


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


class TestHebbianDiscovery:
    def test_single_cooccurrence_does_not_create_relation(self, world):
        """First co-occurrence only increments counter, no relation yet."""
        result = world.ingest(Observation(
            concepts=["Python", "Machine Learning"],
            source="test",
        ))
        assert len(result.hebbian_relations) == 0
        assert world._hebbian.pending_pairs == 1

    def test_cooccurrence_at_threshold_creates_relation(self, world):
        """After COOCCURRENCE_THRESHOLD co-occurrences, relation is created."""
        for i in range(COOCCURRENCE_THRESHOLD - 1):
            world.ingest(Observation(
                concepts=["Python", "Machine Learning"],
                source=f"s{i}",
            ))
        # One more should trigger creation
        result = world.ingest(Observation(
            concepts=["Python", "Machine Learning"],
            source="final",
        ))
        assert len(result.hebbian_relations) == 1
        assert "Python" in result.hebbian_relations[0]
        assert "Machine Learning" in result.hebbian_relations[0]

    def test_repeated_cooccurrence_strengthens(self, world):
        """After relation is created, further co-occurrence reinforces it."""
        # Create the relation first
        for i in range(COOCCURRENCE_THRESHOLD):
            world.ingest(Observation(concepts=["Python", "ML"], source=f"s{i}"))

        py = world.concepts.resolve("Python")
        ml = world.concepts.resolve("ML")
        rel = world.relations.find_any_between(py.id, ml.id)[0]
        initial_weight = rel.weight

        # Additional observation reinforces
        world.ingest(Observation(concepts=["Python", "ML"], source="extra"))
        rel = world.relations.find_any_between(py.id, ml.id)[0]
        assert rel.weight > initial_weight
        assert rel.reinforcement_count >= 1

    def test_three_concepts_create_three_relations(self, world):
        """Three co-occurring concepts → 3 pairwise relations after threshold."""
        for i in range(COOCCURRENCE_THRESHOLD):
            world.ingest(Observation(
                concepts=["A", "B", "C"],
                source=f"s{i}",
            ))
        a = world.concepts.resolve("A")
        b = world.concepts.resolve("B")
        c = world.concepts.resolve("C")
        assert len(world.relations.find_any_between(a.id, b.id)) >= 1
        assert len(world.relations.find_any_between(a.id, c.id)) >= 1
        assert len(world.relations.find_any_between(b.id, c.id)) >= 1

    def test_explicit_relation_not_duplicated_by_hebbian(self, world):
        """If an explicit relation exists, Hebbian reinforces it instead."""
        result = world.ingest(Observation(
            concepts=["Python", "ML"],
            relations=[("Python", "ML", "supports")],
            source="test",
        ))
        # The explicit relation was created, Hebbian should reinforce not duplicate
        py = world.concepts.resolve("Python")
        ml = world.concepts.resolve("ML")
        rels = world.relations.find_any_between(py.id, ml.id)
        assert len(rels) == 1  # only one relation, not two
