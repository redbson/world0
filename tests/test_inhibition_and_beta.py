"""Tests: inhibitory spreading activation + Beta confidence updates."""

from __future__ import annotations

import pytest

from world0 import Observation, World


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


class TestInhibition:
    def test_contrast_partner_is_visible_but_not_a_path(self, world):
        """A stated contrast shows its partner (round 28) and spreads no further through it."""
        for _ in range(6):  # one pair per observation: no co-occurrence edge focus–beyond
            world.ingest(Observation(concepts=["focus", "distractor"],
                                     relations=[("focus", "distractor", "contrasts")], task="t", source="bench"))
            world.ingest(Observation(concepts=["distractor", "beyond"],
                                     relations=[("distractor", "beyond", "depends_on")], task="t", source="bench"))
        proj = world.project(["focus"], task="t", max_depth=3)
        distractor = world.concepts.resolve("distractor")
        beyond = world.concepts.resolve("beyond")
        focus = world.concepts.resolve("focus")
        assert 0 < proj.activation_scores[distractor.id] < proj.activation_scores[focus.id]
        assert beyond.id not in proj.activation_scores

    def test_excitation_overpowers_weak_inhibition(self, world):
        """A strong supporting path must still win over contrast."""
        for _ in range(8):
            world.ingest(Observation(
                concepts=["root", "target"],
                relations=[("root", "target", "depends_on")],
                task="t",
                source="bench",
            ))
        # Add a single contrast — not enough to erase the dependency
        world.ingest(Observation(
            concepts=["root", "target"],
            relations=[("root", "target", "contrasts")],
            task="t",
            source="bench",
        ))
        proj = world.project(["root"], task="t")
        target = world.concepts.resolve("target")
        assert target.id in proj.activation_scores
        assert proj.activation_scores[target.id] > 0


class TestBetaConfidence:
    def test_weaken_lowers_confidence(self, world):
        world.ingest(Observation(concepts=["hypothesis"], source="t"))
        node = world.concepts.resolve("hypothesis")
        before = node.confidence
        world.weaken("hypothesis", source="counterexample")
        after = node.confidence
        assert after < before
        assert node.disconfirmation_count == 1

    def test_evidence_balance_reflects_mix(self, world):
        world.ingest(Observation(concepts=["claim"], source="t"))
        # One confirmation already (ingest reinforces). Add two contradictions.
        world.weaken("claim")
        world.weaken("claim")
        node = world.concepts.resolve("claim")
        balance = node.evidence_balance()
        # With alpha=1+1=2 and beta=1+2=3 → ~0.4
        assert 0.3 < balance < 0.5

    def test_observation_weakened_field_wires_through(self, world):
        world.ingest(Observation(concepts=["drift"], source="t"))
        world.ingest(Observation(
            concepts=[],
            weakened=["drift"],
            source="review",
        ))
        node = world.concepts.resolve("drift")
        assert node.disconfirmation_count == 1

    def test_contradicted_relation_weakens_edge(self, world):
        world.ingest(Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "depends_on")],
            source="t",
        ))
        # Pump the edge so there's something to push down
        for _ in range(3):
            world.ingest(Observation(
                concepts=["A", "B"],
                relations=[("A", "B", "depends_on")],
                source="t",
            ))
        rel = world.relations.find_any_between(
            world.concepts.resolve("A").id,
            world.concepts.resolve("B").id,
        )[0]
        before = rel.weight

        result = world.ingest(Observation(
            concepts=[],
            contradicted_relations=[("A", "B", "depends_on")],
            source="review",
        ))
        after = rel.weight
        assert after < before
        assert result.weakened_relations
        assert rel.disconfirmation_count == 1


class TestBetaPosterior:
    def test_posterior_moves_toward_1_with_reinforcement(self, world):
        world.ingest(Observation(concepts=["strong"], source="t"))
        for _ in range(15):
            world.ingest(Observation(concepts=["strong"], source="t"))
        node = world.concepts.resolve("strong")
        alpha, beta = node.beta_posterior()
        assert alpha > beta
        assert node.evidence_balance() > 0.8

    def test_posterior_moves_toward_0_with_disconfirmation(self, world):
        world.ingest(Observation(concepts=["weak"], source="t"))
        for _ in range(8):
            world.weaken("weak")
        node = world.concepts.resolve("weak")
        alpha, beta = node.beta_posterior()
        assert beta > alpha
        assert node.evidence_balance() < 0.3
