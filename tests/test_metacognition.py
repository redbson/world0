"""Metacognitive monitoring in projections (docs/mc, indicator HOT-2).

Baseline probe: a projection listed ``pytorch enables gpu cluster`` and
``pytorch conflict gpu cluster`` side by side without their beliefs and
without marking the pair as disputed; a concept mentioned once looked as
solid as one confirmed fifty times.  ``Projection.epistemic`` now carries
per-concept reliability and contested / leaning opposing claims, and the
render shows the belief of every explicit claim.
"""

from __future__ import annotations

from world0 import Observation, World
from world0.projection.metacognition import (
    CONTEST_MARGIN,
    TENTATIVE_EVIDENCE,
    WELL_EVIDENCED,
    assess,
)
from world0.schemas.relation import RelationEdge, RelationType


def _mention(world: World, name: str, times: int) -> None:
    for _ in range(times):
        world.ingest(Observation(concepts=[name, "hub"], relations=[("hub", name, "related_to")], source="s"))


def _edge(src, tgt, axis, semantic, belief, explicit=True):
    return RelationEdge(source_id=src, target_id=tgt, relation_type=axis,
                        semantic_relation=semantic, probability=belief, is_explicit=explicit)


class TestReliability:
    def test_levels_follow_evidence(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _mention(w, "once", 1)
        _mention(w, "several", 5)
        _mention(w, "often", 20)
        p = w.project(["hub"], max_concepts=10)
        by_name = {c.name: p.epistemic.reliability[c.id] for c in p.concepts}
        assert by_name["once"] == "tentative"
        assert by_name["several"] == "moderate"
        assert by_name["often"] == "well_evidenced"
        assert 0.0 < TENTATIVE_EVIDENCE < WELL_EVIDENCED < 1.0

    def test_thin_evidence_is_rendered(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        _mention(w, "often", 20)
        _mention(w, "once", 1)
        text = w.project(["hub"], max_concepts=10).render(style="full")
        assert "### Epistemic Status" in text
        assert "Thin evidence" in text and "once" in text.split("Thin evidence")[1]


class TestContestedClaims:
    def test_close_beliefs_are_contested(self):
        rels = [_edge("a", "b", RelationType.POSITIVE, "enables", 0.55),
                _edge("a", "b", RelationType.NEGATIVE, "conflict", 0.45)]
        status = assess([], rels)
        assert len(status.contested) == 1
        claim = status.contested[0]
        assert claim.status == "contested" and claim.margin < CONTEST_MARGIN
        assert claim.leading == rels[0].id
        assert [c[1] for c in claim.claims] == ["enables", "conflict"]

    def test_clear_lead_is_leaning(self):
        rels = [_edge("a", "b", RelationType.POSITIVE, "enables", 0.35),
                _edge("b", "a", RelationType.NEGATIVE, "conflict", 0.80)]
        claim = assess([], rels).contested[0]
        assert claim.status == "leaning" and claim.leading == rels[1].id

    def test_compatible_claims_are_not_contested(self):
        rels = [_edge("a", "b", RelationType.POSITIVE, "enables", 0.6),
                _edge("a", "b", RelationType.POSITIVE, "dependence", 0.6),
                _edge("a", "b", RelationType.PARALLEL, "generic_relation", 0.2)]
        assert assess([], rels).contested == []

    def test_cooccurrence_is_not_a_claim(self):
        rels = [_edge("a", "b", RelationType.NEGATIVE, "conflict", 0.6),
                _edge("a", "b", RelationType.PARALLEL, "similarity_kernel", 0.6, explicit=False)]
        assert assess([], rels).contested == []

    def test_world_projection_reports_leaning_claim(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        for _ in range(6):
            w.ingest(Observation(concepts=["pytorch", "gpu cluster"], relations=[("pytorch", "gpu cluster", "enables")], source="s"))
        for _ in range(4):
            w.ingest(Observation(concepts=["pytorch", "gpu cluster"], relations=[("pytorch", "gpu cluster", "conflict")], source="s"))
        p = w.project(["pytorch"])
        assert len(p.epistemic.contested) == 1
        text = p.render(style="full")
        assert "Leaning: pytorch enables gpu cluster" in text or "Contested:" in text


class TestRender:
    def test_belief_for_claims_cooccurrence_for_hebbian(self, tmp_path):
        w = World(store_path=tmp_path / "w")
        for _ in range(4):
            w.ingest(Observation(concepts=["a", "b", "c"], relations=[("a", "b", "depends_on")], source="s"))
        text = w.project(["a"]).render(style="full")
        line_ab = next(ln for ln in text.splitlines() if "→ dependence" in ln)
        assert "belief:" in line_ab
        assert any("co-occurrence" in ln for ln in text.splitlines() if "generic_relation" in ln)

    def test_empty_projection_has_empty_status(self, tmp_path):
        p = World(store_path=tmp_path / "w").project(["nothing"])
        assert p.epistemic.reliability == {} and p.epistemic.contested == []
        assert "Epistemic Status" not in p.render()
