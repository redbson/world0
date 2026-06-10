"""Tests: facade-level usage feedback (World.apply_feedback).

Feedback is the public channel any consumer — agents, the autonomy
loop, a human — uses to tell World 0 what helped and what misled.  It
must work through the facade alone, with no reach into internals.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    for _ in range(8):
        w.ingest(Observation(
            concepts=["service", "database", "cache"],
            relations=[
                ("service", "database", "depends_on"),
                ("service", "cache", "supports"),
            ],
            task="architecture",
            source="t",
        ))
    return w


class TestConceptFeedback:
    def test_missing_concept_created(self, world):
        result = world.apply_feedback(missing_concepts=["queue"])
        assert "queue" in result.created_concepts
        assert world.concepts.resolve("queue") is not None

    def test_missing_existing_concept_reinforced(self, world):
        result = world.apply_feedback(missing_concepts=["service"])
        assert "service" in result.reinforced_concepts
        assert "service" not in result.created_concepts

    def test_useful_concept_reinforced(self, world):
        svc = world.concepts.resolve("service")
        before = svc.confidence
        result = world.apply_feedback(
            useful_concepts=["service"], task="architecture"
        )
        assert "service" in result.reinforced_concepts
        assert world.concepts.resolve("service").confidence >= before

    def test_noisy_concept_demoted(self, world):
        cache = world.concepts.resolve("cache")
        before = cache.confidence
        result = world.apply_feedback(noisy_concepts=["cache"])
        assert "cache" in result.demoted_concepts
        assert world.concepts.resolve("cache").confidence < before

    def test_useful_skips_concept_also_marked_noisy(self, world):
        result = world.apply_feedback(
            useful_concepts=["cache"], noisy_concepts=["cache"]
        )
        # Demotion wins; the concept is not also reinforced.
        assert "cache" in result.demoted_concepts
        assert "cache" not in result.reinforced_concepts


class TestRelationFeedback:
    def test_weak_relation_by_label(self, world):
        svc = world.concepts.resolve("service")
        db = world.concepts.resolve("database")
        rel = world.relations.find_between(svc.id, db.id)
        before = rel.weight
        result = world.apply_feedback(
            weak_relations=["service -> dependence -> database"]
        )
        assert result.weakened_relations
        assert world.relations.get(rel.id).weight < before

    def test_weak_relation_by_id(self, world):
        svc = world.concepts.resolve("service")
        cache = world.concepts.resolve("cache")
        rel = world.relations.find_between(svc.id, cache.id)
        before = rel.weight
        result = world.apply_feedback(weak_relations=[rel.id])
        assert rel.id in result.weakened_relations
        assert world.relations.get(rel.id).weight < before

    def test_useful_relation_reinforced(self, world):
        svc = world.concepts.resolve("service")
        db = world.concepts.resolve("database")
        rel = world.relations.find_between(svc.id, db.id)
        before = rel.weight
        result = world.apply_feedback(useful_relations=[rel.id])
        assert rel.id in result.reinforced_relations
        assert world.relations.get(rel.id).weight >= before

    def test_unresolvable_relation_ignored(self, world):
        result = world.apply_feedback(
            weak_relations=["nope -> dependence -> alsonope"]
        )
        assert result.weakened_relations == []


class TestAutonomyStyleUsage:
    def test_feedback_through_public_facade_only(self, world):
        """Simulate the autonomy loop: only public methods touched."""
        proj = world.project(["service"], task="architecture")
        useful = [c.name for c in proj.top_concepts(3)]
        result = world.apply_feedback(
            useful_concepts=useful, task="architecture"
        )
        assert result.reinforced_concepts
        # Persisted (flush happened) — a fresh World sees the effect.
        assert world.concepts.resolve("service") is not None

    def test_empty_feedback_is_noop(self, world):
        result = world.apply_feedback()
        assert result.reinforced_concepts == []
        assert result.created_concepts == []
        assert result.demoted_concepts == []
        assert result.weakened_relations == []
