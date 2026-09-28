"""Relation claims — direction and contradiction (analysis doc §7.18).

Probes before the fix:

- **direction**: with ``X depends_on Y`` stated ten times, stating
  ``Y depends_on X`` matched the existing edge (lookup ignored
  orientation) and *confirmed* ``X → Y``; the reverse claim vanished;
- **contradiction**: ten ``A enables B`` then ten ``A conflict B`` left
  both edges confident (enables probability 0.85 unchanged) — the
  opposite claim was only honoured through ``contradicted_relations``.

Now directed claims are matched in their stated orientation (symmetric
semantics such as ``conflict`` in either), and an explicit claim on the
negative axis disconfirms an explicit positive / parallel claim about the
same pair and vice versa, so contradictory beliefs compete.
"""

from __future__ import annotations

from world0 import Observation, World
from world0.schemas.relation import (
    SYMMETRIC_SEMANTIC_RELATIONS,
    RelationEdge,
    RelationType,
)


def _state(world: World, src: str, tgt: str, rel: str, times: int = 1):
    result = None
    for _ in range(times):
        result = world.ingest(Observation(concepts=[src, tgt], relations=[(src, tgt, rel)], source="s"))
    return result


def _edges(world: World, a: str, b: str) -> list[RelationEdge]:
    return world.relations.find_any_between(world.concepts.resolve(a).id, world.concepts.resolve(b).id)


def _explicit(world: World, a: str, b: str, semantic: str) -> list[RelationEdge]:
    return [e for e in _edges(world, a, b) if e.is_explicit and e.semantic_relation == semantic]


class TestDirection:
    def test_reverse_directed_claim_is_a_separate_edge(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "X", "Y", "depends_on", 10)
        forward = _explicit(w, "X", "Y", "dependence")[0]
        p_before = forward.probability
        _state(w, "Y", "X", "depends_on")
        deps = _explicit(w, "X", "Y", "dependence")
        assert len(deps) == 2
        x = w.concepts.resolve("X").id
        forward = next(e for e in deps if e.source_id == x)
        assert forward.probability == p_before  # not confirmed by the reverse claim

    def test_same_direction_still_confirms(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "X", "Y", "depends_on")
        p0 = _explicit(w, "X", "Y", "dependence")[0].probability
        _state(w, "X", "Y", "depends_on")
        edges = _explicit(w, "X", "Y", "dependence")
        assert len(edges) == 1 and edges[0].probability > p0

    def test_symmetric_semantic_matches_either_orientation(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "A", "B", "conflict")
        p0 = _explicit(w, "A", "B", "conflict")[0].probability
        _state(w, "B", "A", "conflict")
        edges = _explicit(w, "A", "B", "conflict")
        assert len(edges) == 1 and edges[0].probability > p0

    def test_parallel_matches_either_orientation(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "A", "B", "similar_to")
        _state(w, "B", "A", "similar_to")
        assert len(_explicit(w, "A", "B", "similarity_kernel")) == 1

    def test_contradicting_reverse_direction_leaves_forward_edge(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "X", "Y", "depends_on", 3)
        edge = _explicit(w, "X", "Y", "dependence")[0]
        p0 = edge.probability
        w.ingest(Observation(concepts=["X", "Y"], contradicted_relations=[("Y", "X", "depends_on")], source="s"))
        assert _explicit(w, "X", "Y", "dependence")[0].probability == p0

    def test_edge_helpers(self):
        dep = RelationEdge(source_id="a", target_id="b", relation_type=RelationType.POSITIVE, semantic_relation="dependence")
        con = RelationEdge(source_id="a", target_id="b", relation_type=RelationType.NEGATIVE, semantic_relation="conflict")
        assert dep.is_directed and not con.is_directed
        assert dep.connects("a", "b", directed=True) and not dep.connects("b", "a", directed=True)
        assert dep.connects("b", "a") and con.connects("b", "a", directed=True)
        assert "conflict" in SYMMETRIC_SEMANTIC_RELATIONS and "dependence" not in SYMMETRIC_SEMANTIC_RELATIONS


class TestContradiction:
    def test_opposite_claim_disconfirms_and_is_reported(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "A", "B", "enables", 10)
        assert _explicit(w, "A", "B", "enables")[0].probability > 0.8
        result = _state(w, "A", "B", "conflict")
        assert "A → enables → B" in result.weakened_relations
        _state(w, "A", "B", "conflict", 9)
        assert _explicit(w, "A", "B", "enables")[0].probability < 0.5

    def test_both_directions_of_opposition(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "A", "B", "conflict", 5)
        p0 = _explicit(w, "A", "B", "conflict")[0].probability
        _state(w, "B", "A", "enables")
        assert _explicit(w, "A", "B", "conflict")[0].probability < p0

    def test_dominant_claim_wins(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        for i in range(8):
            _state(w, "A", "B", "conflict" if i % 4 == 3 else "enables")
        enables = _explicit(w, "A", "B", "enables")[0].probability
        conflict = _explicit(w, "A", "B", "conflict")[0].probability
        assert enables > conflict

    def test_compatible_claims_do_not_weaken(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _state(w, "A", "B", "enables", 3)
        p0 = _explicit(w, "A", "B", "enables")[0].probability
        result = _state(w, "A", "B", "similar_to")  # parallel, but no negative claim
        result2 = _state(w, "B", "A", "depends_on")  # positive vs positive
        assert _explicit(w, "A", "B", "enables")[0].probability == p0
        assert not result.weakened_relations and not result2.weakened_relations

    def test_cooccurrence_and_generic_edges_untouched(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        for _ in range(4):
            w.ingest(Observation(concepts=["A", "B"], source="s"))
        _state(w, "A", "B", "related_to", 2)
        _state(w, "A", "B", "conflict", 3)
        for e in _edges(w, "A", "B"):
            if e.relation_type != RelationType.NEGATIVE:
                assert e.disconfirmation_count == 0
