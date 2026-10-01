"""A store written by 0.2.0 keeps what it said when 0.3.0 opens it.

0.2.0 stored the relation label itself in ``relation_type`` (``depends_on``,
``part_of``, ``contrasts`` ...) and had no ``semantic_relation``.  Opening
such an edge must keep the label's semantics (not collapse it to the
axis default) and store a label phrased from the other end in the
canonical direction, exactly as a freshly stated claim would be.
"""

from __future__ import annotations

import json

from world0.schemas.relation import RelationEdge, RelationType


def _legacy(relation_type: str, src: str = "a", tgt: str = "b") -> dict:
    """An edge as 0.2.0 wrote it (no semantic_relation, label in relation_type)."""
    return {
        "id": "e1", "source_id": src, "target_id": tgt, "relation_type": relation_type,
        "weight": 0.5, "is_explicit": True, "confidence": 0.5, "reinforcement_count": 2,
        "last_reinforced": "2026-01-01T00:00:00+00:00", "discovered_at": "2026-01-01T00:00:00+00:00",
        "provenance": "agent", "task_history": ["t"],
    }


class TestLegacyLabels:
    def test_depends_on_keeps_its_semantics_and_direction(self):
        e = RelationEdge.model_validate_json(json.dumps(_legacy("depends_on", "api", "db")))
        assert e.semantic_relation == "dependence"
        assert e.relation_type is RelationType.POSITIVE
        assert (e.source_id, e.target_id) == ("api", "db")

    def test_part_of_becomes_inclusion_from_the_whole(self):
        e = RelationEdge.model_validate_json(json.dumps(_legacy("part_of", "wheel", "car")))
        assert e.semantic_relation == "inclusion"
        assert (e.source_id, e.target_id) == ("car", "wheel")  # "car contains wheel"

    def test_precedes_becomes_dependence_of_the_later_step(self):
        e = RelationEdge.model_validate_json(json.dumps(_legacy("precedes", "build", "deploy")))
        assert e.semantic_relation == "dependence"
        assert (e.source_id, e.target_id) == ("deploy", "build")  # "deploy depends on build"

    def test_contrasts_is_contrast_not_conflict(self):
        e = RelationEdge.model_validate_json(json.dumps(_legacy("contrasts", "api", "cache")))
        assert e.semantic_relation == "contrast"
        assert e.relation_type is RelationType.NEGATIVE

    def test_related_to_and_hebbian_edges_are_generic(self):
        d = _legacy("related_to"); d["is_explicit"] = False
        e = RelationEdge.model_validate_json(json.dumps(d))
        assert e.semantic_relation == "generic_relation"
        assert e.relation_type is RelationType.PARALLEL

    def test_axis_names_and_current_edges_are_untouched(self):
        e = RelationEdge.model_validate_json(json.dumps(_legacy("positive", "x", "y")))
        assert e.semantic_relation == "mutual_reinforcement"  # the axis default, as before
        assert (e.source_id, e.target_id) == ("x", "y")
        cur = RelationEdge(source_id="a", target_id="b", relation_type="part_of", semantic_relation="inclusion")
        assert (cur.source_id, cur.target_id) == ("a", "b")  # a current edge is stored oriented already
        rt = RelationEdge.model_validate(RelationEdge(source_id="a", target_id="b", semantic_relation="dependence").model_dump())
        assert rt.semantic_relation == "dependence" and (rt.source_id, rt.target_id) == ("a", "b")
