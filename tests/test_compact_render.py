"""The compact render — the default prompt form of a projection (round 24).

LongRun: real readers answered 0.90 of the questions from a compact list of
typed claims and 0.57 from the full render at the same token budget; the
full render spends its tokens on ids, maturity, strengths and counts.  The
compact form must still carry everything a reader needs not to be misled:
which claims hold, which were withdrawn, which belong to another task and
which are contested or thin.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.schemas.relation import RELATION_PHRASES, SEMANTIC_RELATION_SPECS, relation_phrase
from world0.schemas.types import ConceptCandidate


def _claim_lines(text: str) -> list[str]:
    """Bullets before the first ``Label:`` line: the current claims."""
    out = []
    for line in text.splitlines():
        if line.endswith(":") and not line.startswith("- "):
            break
        if line.startswith("- "):
            out.append(line[2:])
    return out


def _section(text: str, label: str) -> list[str]:
    lines = text.splitlines()
    if label not in lines:
        return []
    out = []
    for line in lines[lines.index(label) + 1:]:
        if not line.startswith("- "):
            break
        out.append(line[2:])
    return out


class TestCompactRender:
    def test_claims_read_as_sentences_with_belief_strongest_first(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        w.ingest(Observation(concepts=["api", "cache"], relations=[("api", "cache", "depends_on")]))
        p = w.project(["api"], max_concepts=5)
        claims = _claim_lines(p.render())
        assert claims[0].startswith("api depends on db (belief ")  # stated four times
        assert claims[1].startswith("api depends on cache (belief ")
        beliefs = [float(c.rsplit("belief ", 1)[1].rstrip(")")) for c in claims]
        assert beliefs == sorted(beliefs, reverse=True)
        assert all(c.id not in p.render() for c in p.concepts)  # no ids

    def test_it_is_the_default_and_shorter_than_the_full_view(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["a", "b", "c"], relations=[("a", "b", "depends_on")]))
        p = w.project(["a"], max_concepts=5)
        assert p.render() == p.render(style="compact")
        assert len(p.render()) < len(p.render(style="full")) / 2
        with pytest.raises(ValueError):
            p.render(style="verbose")

    def test_co_occurrence_asserts_nothing(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")]))
            w.ingest(Observation(concepts=["a", "z"]))
        p = w.project(["a"], max_concepts=5)
        assert any(not r.is_explicit for r in p.relations)  # a–z is in view as co-occurrence
        text = p.render()
        assert [c.split(" (")[0] for c in _claim_lines(text)] == ["a depends on b"]
        assert "Also relevant: z." in text

    def test_a_withdrawn_claim_is_listed_apart(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["api", "db"], relations=[("api", "db", "depends_on")]))
        w.ingest(Observation(
            concepts=["api", "cache"],
            relations=[("api", "cache", "depends_on")],
            retracted_relations=[("api", "db", "depends_on")],
        ))
        text = w.project(["api"], max_concepts=5).render()
        assert not any(c.startswith("api depends on db") for c in _claim_lines(text))
        assert _section(text, "No longer holds:") == ["api depends on db"]

    def test_claims_of_other_tasks_are_labelled_with_their_task(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(4):
            w.ingest(Observation(concepts=["pipeline", "etl"], relations=[("pipeline", "etl", "depends_on")],
                                 task="data engineering work"))
            w.ingest(Observation(concepts=["pipeline", "optimizer"], relations=[("pipeline", "optimizer", "depends_on")],
                                 task="model training work"))
        text = w.project(["pipeline"], task="data engineering work", max_concepts=6).render()
        assert text.splitlines()[1] == "Context for: data engineering work"
        assert [c.split(" (")[0] for c in _claim_lines(text)] == ["pipeline depends on etl"]
        assert _section(text, "Seen under other tasks:") == [
            "pipeline depends on optimizer (model training work)"
        ]

    def test_contested_and_thin_knowledge_is_flagged(self, tmp_path):
        w = World(store_path=tmp_path)
        for _ in range(3):
            w.ingest(Observation(concepts=["cache", "db"], relations=[("cache", "db", "enables")]))
            w.ingest(Observation(concepts=["cache", "db"], relations=[("cache", "db", "conflict")]))
        w.ingest(Observation(concepts=["cache", "once"], relations=[("cache", "once", "depends_on")]))
        p = w.project(["cache", "db"], max_concepts=5)
        loose = _section(p.render(), "Hold loosely:")
        assert any(line.startswith(("contested: ", "leaning: ")) and "cache enables db" in line
                   and "cache conflicts with db" in line for line in loose)
        assert any(line.startswith("thin evidence: ") and "once" in line for line in loose)

    def test_homonyms_are_told_apart(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concept_candidates=[
            ConceptCandidate(uid="f", name="Apple", sense="fruit"),
            ConceptCandidate(uid="t", name="Apple", sense="technology company"),
            ConceptCandidate(uid="o", name="orchard"),
        ], relations=[("f", "o", "part_of")]))
        ids = [c.id for c in w.concepts.all() if c.name == "Apple"]
        text = w.project(ids, max_concepts=5).render()
        assert "Apple (fruit) belongs to orchard" in text
        assert "Apple (technology company)" in text

    def test_definitions_come_from_the_concept_cards(self, tmp_path):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concept_candidates=[
            ConceptCandidate(uid="a", name="idempotency", description="repeating an operation changes nothing"),
            ConceptCandidate(uid="b", name="retry"),
        ], relations=[("b", "a", "depends_on")]))
        text = w.project(["retry"], max_concepts=4).render()
        assert _section(text, "Definitions:") == ["idempotency: repeating an operation changes nothing"]

    def test_an_empty_view_says_so(self, tmp_path):
        p = World(store_path=tmp_path).project(["nothing"], task="triage")
        assert p.render().splitlines() == ["## Cognitive Context", "Context for: triage", "No concepts in view."]


class TestRelationPhrases:
    def test_every_semantic_relation_has_a_phrase(self):
        assert set(RELATION_PHRASES) == set(SEMANTIC_RELATION_SPECS)

    @pytest.mark.parametrize("label, sentence", [
        ("depends_on", "x depends on y"),
        ("contains", "x contains y"),
        ("inclusion", "x contains y"),
        ("part_of", "x belongs to y"),
        ("conflict", "x conflicts with y"),
        ("related_to", "x is related to y"),
    ])
    def test_a_claim_reads_in_the_direction_it_was_stated(self, tmp_path, label, sentence):
        w = World(store_path=tmp_path)
        w.ingest(Observation(concepts=["x", "y"], relations=[("x", "y", label)]))
        # Both seeded: a negative claim inhibits, so it does not activate its partner.
        claims = _claim_lines(w.project(["x", "y"], max_concepts=3).render())
        assert [c.split(" (belief")[0] for c in claims] == [sentence]
        assert relation_phrase(label) in sentence
