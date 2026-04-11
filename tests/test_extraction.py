"""Tests for ConceptExtractor — parsing logic and robustness.

These tests use a FakeLLM to verify extraction without API calls.
"""

import json

import pytest

from world0.extraction.extractor import ConceptExtractor
from world0.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    """Returns a pre-configured response for testing."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete_json(self, system: str, user: str) -> str:
        return self._response


# ── Parsing correctness ──────────────────────────────────────────────


class TestExtractionParsing:
    def test_parses_well_formed_json(self):
        response = json.dumps({
            "concepts": [
                {"name": "docker", "description": "Container runtime"},
                {"name": "kubernetes", "description": "Container orchestrator"},
            ],
            "relations": [
                {"source": "kubernetes", "target": "docker", "type": "depends_on"},
            ],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text", task="test", source="unit")

        assert obs.concepts == ["docker", "kubernetes"]
        assert obs.descriptions == {
            "docker": "Container runtime",
            "kubernetes": "Container orchestrator",
        }
        assert obs.relations == [("kubernetes", "docker", "depends_on")]
        assert obs.task == "test"
        assert obs.source == "unit"

    def test_parses_json_in_markdown_fences(self):
        response = '```json\n{"concepts": [{"name": "python"}], "relations": []}\n```'
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.concepts == ["python"]

    def test_parses_json_with_surrounding_text(self):
        response = 'Here is the result:\n{"concepts": [{"name": "redis"}], "relations": []}\nDone.'
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.concepts == ["redis"]

    def test_handles_concepts_as_plain_strings(self):
        response = json.dumps({
            "concepts": ["docker", "kubernetes"],
            "relations": [],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.concepts == ["docker", "kubernetes"]

    def test_empty_text_returns_empty_observation(self):
        extractor = ConceptExtractor(FakeLLM("should not be called"))
        obs = extractor.extract("", task="t", source="s")
        assert obs.concepts == []
        assert obs.relations == []
        assert obs.task == "t"

    def test_garbage_response_returns_empty_observation(self):
        extractor = ConceptExtractor(FakeLLM("this is not json at all"))
        obs = extractor.extract("some text")
        assert obs.concepts == []
        assert obs.relations == []


class TestExtractionValidation:
    def test_invalid_relation_type_falls_back_to_related_to(self):
        response = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "a", "target": "b", "type": "invented_type"}],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.relations == [("a", "b", "related_to")]

    def test_relation_with_unknown_concept_is_dropped(self):
        response = json.dumps({
            "concepts": [{"name": "a"}],
            "relations": [{"source": "a", "target": "nonexistent", "type": "depends_on"}],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.relations == []

    def test_relation_with_empty_source_is_dropped(self):
        response = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "", "target": "b", "type": "depends_on"}],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.relations == []

    def test_duplicate_concepts_preserved(self):
        """Extractor doesn't deduplicate — that's World.ingest()'s job."""
        response = json.dumps({
            "concepts": [{"name": "docker"}, {"name": "docker"}],
            "relations": [],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert len(obs.concepts) == 2

    def test_all_valid_relation_types_accepted(self):
        from world0.schemas.relation import RelationType

        concepts = [{"name": f"c{i}"} for i in range(20)]
        relations = []
        for i, rt in enumerate(RelationType):
            relations.append({
                "source": f"c{i * 2}",
                "target": f"c{i * 2 + 1}",
                "type": rt.value,
            })

        response = json.dumps({"concepts": concepts, "relations": relations})
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")

        extracted_types = {r[2] for r in obs.relations}
        assert extracted_types == {rt.value for rt in RelationType}
