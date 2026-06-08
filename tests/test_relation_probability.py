"""Tests for semantic relation score mapping."""

from __future__ import annotations

import pytest

from world0 import Observation, RelationPrior, World


def test_relation_scores_come_from_semantic_label_not_extraction_metadata(tmp_path):
    world = World(tmp_path / ".world0")

    world.ingest(
        Observation(
            concepts=["retrieval", "generation"],
            relations=[("retrieval", "generation", "supports")],
            extraction_metadata={
                "relations": [
                    {
                        "source": "retrieval",
                        "target": "generation",
                        "type": "supports",
                        "probability": 0.82,
                        "confidence": 0.91,
                    }
                ]
            },
        )
    )

    retrieval = world.concepts.resolve("retrieval")
    generation = world.concepts.resolve("generation")
    edge = world.relations.find_between(
        retrieval.id, generation.id, None
    )

    assert edge is not None
    assert edge.semantic_relation == "enables"
    assert edge.relation_type.value == "positive"
    assert edge.structural_strength == pytest.approx(0.82)
    assert edge.propagation_strength == pytest.approx(0.76)
    assert edge.probability == pytest.approx(0.76)
    assert 0.0 <= edge.weight <= 1.0
    assert 0.0 <= edge.confidence <= 1.0


def test_relation_prior_can_still_adjust_world_belief(
    tmp_path,
):
    world = World(tmp_path / ".world0")
    base = Observation(
        concepts=["a", "b"],
        relations=[("a", "b", "depends_on")],
        extraction_metadata={
            "relations": [
                {
                        "source": "a",
                        "target": "b",
                        "type": "depends_on",
                        "probability": 0.8,
                    }
                ]
            },
    )
    world.ingest(base)

    world.ingest(
        Observation(
            concepts=["a", "b"],
            relations=[("a", "b", "depends_on")],
            relation_priors=[
                RelationPrior(
                    source="a",
                    target="b",
                    relation_type="depends_on",
                    probability=0.2,
                    strength=1.0,
                )
            ],
            extraction_metadata={
                "relations": [
                    {
                        "source": "a",
                        "target": "b",
                        "type": "depends_on",
                        "probability": 0.5,
                    }
                ]
            },
        )
    )

    a = world.concepts.resolve("a")
    b = world.concepts.resolve("b")
    edge = world.relations.find_between(a.id, b.id, None)

    assert edge is not None
    assert edge.semantic_relation == "dependence"
    assert edge.structural_strength == pytest.approx(0.78)
    assert edge.propagation_strength == pytest.approx(0.70)
    assert 0.2 < edge.probability < 0.70
    assert edge.probability == pytest.approx(0.575, abs=0.0001)
