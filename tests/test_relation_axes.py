"""Tests for the three-axis relation model."""

import pytest

from world0.schemas.relation import (
    RelationEdge,
    RelationType,
    normalize_relation_type,
    normalize_semantic_relation,
    semantic_relation_spec,
)
from world0.schemas.types import Observation
from world0.world import World


def test_relation_type_iteration_exposes_only_axes():
    assert [rt.value for rt in RelationType] == [
        "positive",
        "negative",
        "parallel",
    ]


def test_legacy_relation_labels_normalize_to_axes():
    assert normalize_relation_type("supports") is RelationType.POSITIVE
    assert normalize_relation_type("depends_on") is RelationType.POSITIVE
    assert normalize_relation_type("contrasts") is RelationType.NEGATIVE
    assert normalize_relation_type("similar_to") is RelationType.PARALLEL
    assert normalize_relation_type("related_to") is RelationType.PARALLEL
    assert normalize_relation_type("unknown") is RelationType.PARALLEL


def test_relation_edge_coerces_legacy_saved_value():
    edge = RelationEdge(
        source_id="a",
        target_id="b",
        relation_type="contrasts",
    )

    assert edge.relation_type is RelationType.NEGATIVE
    assert edge.relation_type.value == "negative"
    assert edge.semantic_relation == "conflict"
    assert edge.structural_strength == pytest.approx(0.84)
    assert edge.propagation_strength == pytest.approx(0.10)


def test_semantic_relation_labels_map_to_axes_and_scores():
    assert normalize_semantic_relation("supports") == "enables"
    assert normalize_semantic_relation("deep conceptual overlap") == "overlap"

    disjoint = semantic_relation_spec("disjointness")
    assert disjoint.axis is RelationType.NEGATIVE
    assert disjoint.structural_strength == pytest.approx(0.95)
    assert disjoint.propagation_strength == pytest.approx(0.05)


def test_world_ingest_stores_axis_links(tmp_path):
    world = World(store_path=tmp_path)
    world.ingest(Observation(
        concepts=["trust", "co-creation", "conflict", "overlap"],
        relations=[
            ("trust", "co-creation", "positive"),
            ("trust", "conflict", "negative"),
            ("co-creation", "overlap", "parallel"),
        ],
    ))

    relation_types = {r.relation_type.value for r in world.relations.all()}
    semantic_relations = {r.semantic_relation for r in world.relations.all()}

    assert relation_types == {"positive", "negative", "parallel"}
    assert semantic_relations == {"mutual_reinforcement", "conflict", "generic_relation"}
