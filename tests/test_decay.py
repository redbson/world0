"""Tests: time decay — unused concepts and relations fade."""

import pytest
from datetime import datetime, timezone, timedelta

from world0 import Observation, World
from world0.schemas.concept import Maturity


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


class TestConceptDecay:
    def test_recent_concept_barely_decays(self, world):
        """A just-created concept should not decay significantly."""
        world.ingest(Observation(concepts=["Fresh"], source="test"))
        node = world.concepts.resolve("Fresh")
        old_conf = node.confidence

        world._decay.decay_concepts()
        assert node.confidence == pytest.approx(old_conf, abs=0.01)

    def test_old_embryonic_decays_fast(self, world):
        """An embryonic concept untouched for 48h should decay heavily."""
        world.ingest(Observation(concepts=["Old"], source="test"))
        node = world.concepts.resolve("Old")
        # Simulate 48h without activation
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=48)

        world._decay.decay_concepts()
        # 48h with 24h half-life → ~25% remaining of initial ~0.21
        assert node.confidence < 0.06

    def test_core_concept_decays_slowly(self, world):
        """A core concept should retain confidence even after a week."""
        world.ingest(Observation(concepts=["Core"], source="test"))
        node = world.concepts.resolve("Core")
        node.maturity = Maturity.CORE
        node.confidence = 0.9
        # Simulate 1 week
        node.last_activated = datetime.now(timezone.utc) - timedelta(hours=168)

        world._decay.decay_concepts()
        # 168h with 2160h half-life → ~95% remaining
        assert node.confidence > 0.8


class TestRelationDecay:
    def test_unreinforced_relation_decays(self, world):
        """A relation with low reinforcement count should decay faster."""
        world.ingest(Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "supports")],
            source="test",
        ))
        a = world.concepts.resolve("A")
        b = world.concepts.resolve("B")
        rel = world.relations.find_any_between(a.id, b.id)[0]

        # Simulate 2 weeks (relation gets 1 reinforcement from hebbian)
        rel.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=336)
        world._decay.decay_relations()
        assert rel.weight < 0.05

    def test_heavily_reinforced_relation_persists(self, world):
        """A relation reinforced many times should decay slowly."""
        world.ingest(Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "supports")],
            source="test",
        ))
        a = world.concepts.resolve("A")
        b = world.concepts.resolve("B")
        rel = world.relations.find_any_between(a.id, b.id)[0]
        rel.reinforcement_count = 20
        rel.weight = 0.8

        # Simulate 1 week
        rel.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=168)
        world._decay.decay_relations()
        # 168h with half_life = 72*(1+20*0.5) = 792h → still most weight
        assert rel.weight > 0.5


class TestPruning:
    def test_fading_concept_pruned(self, world):
        world.ingest(Observation(concepts=["Garbage"], source="test"))
        node = world.concepts.resolve("Garbage")
        node.maturity = Maturity.FADING
        node.confidence = 0.001

        pruned = world._decay.prune_concepts()
        assert node.id in pruned
        assert world.concepts.resolve("Garbage") is None
