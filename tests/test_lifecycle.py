"""Tests: concept lifecycle — creation → reinforcement → promotion → decay."""

import pytest

from world0 import Observation, World
from world0.schemas.concept import Maturity


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


class TestConceptCreation:
    def test_new_concept_is_embryonic(self, world):
        world.ingest(Observation(concepts=["Python"], source="test"))
        node = world.concepts.resolve("Python")
        assert node is not None
        assert node.maturity == Maturity.EMBRYONIC
        assert node.confidence == pytest.approx(0.2, abs=0.05)

    def test_duplicate_ingest_reinforces(self, world):
        world.ingest(Observation(concepts=["Python"], source="s1"))
        world.ingest(Observation(concepts=["Python"], source="s2"))
        node = world.concepts.resolve("Python")
        assert node.activation_count == 2
        assert node.confidence > 0.1  # reinforced


class TestPromotion:
    def test_embryonic_to_developing(self, world):
        """After enough reinforcements, concept promotes to developing."""
        for i in range(5):
            world.ingest(Observation(concepts=["Python"], source=f"s{i}"))

        # Manually set confidence to meet threshold
        node = world.concepts.resolve("Python")
        node.confidence = 0.4
        promoted, _ = world._lifecycle.evaluate()
        node = world.concepts.resolve("Python")
        assert node.maturity == Maturity.DEVELOPING

    def test_developing_to_established(self, world):
        node, _ = world.concepts.get_or_create("Python")
        node.maturity = Maturity.DEVELOPING
        node.activation_count = 12
        node.confidence = 0.7
        promoted, _ = world._lifecycle.evaluate()
        assert node.id in promoted
        node = world.concepts.get(node.id)
        assert node.maturity == Maturity.ESTABLISHED

    def test_established_to_core(self, world):
        """Core requires both high activation and dense connections."""
        node, _ = world.concepts.get_or_create("Python")
        node.maturity = Maturity.ESTABLISHED
        node.activation_count = 35
        node.confidence = 0.9

        # Create 5+ connections
        for i in range(6):
            other, _ = world.concepts.get_or_create(f"Concept_{i}")
            world.relations.discover(node.id, other.id)

        promoted, _ = world._lifecycle.evaluate()
        assert node.id in promoted
        assert world.concepts.get(node.id).maturity == Maturity.CORE


class TestFadingAndRevival:
    def test_fading_concept_revives_on_activation(self, world):
        node, _ = world.concepts.get_or_create("OldConcept")
        node.maturity = Maturity.FADING
        node.confidence = 0.03

        # Re-activate
        world.ingest(Observation(concepts=["OldConcept"], source="revival"))
        node = world.concepts.resolve("OldConcept")
        assert node.maturity == Maturity.DEVELOPING
        assert node.confidence > 0.03
