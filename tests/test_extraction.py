"""Tests for ConceptExtractor — parsing logic and robustness.

These tests use a FakeLLM to verify extraction without API calls.
"""

import json

import pytest

from world0.extraction.extractor import ConceptExtractor
from world0.llm.base import LLMProvider
from world0.schemas.types import RelationPrior


class FakeLLM(LLMProvider):
    """Returns a pre-configured response for testing."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.system = ""
        self.user = ""

    def complete_json(self, system: str, user: str) -> str:
        self.system = system
        self.user = user
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
        assert obs.relations == [("kubernetes", "docker", "dependence")]
        assert obs.task == "test"
        assert obs.source == "unit"
        assert "## Task Context" in extractor._provider.user
        assert "test" in extractor._provider.user
        assert "unit" in extractor._provider.user

    def test_parses_enhanced_schema_with_metadata(self):
        response = json.dumps({
            "domain": "ai systems",
            "concepts": [
                {
                    "name": "retrieval augmented generation",
                    "description": "Retrieval-conditioned generation pattern",
                    "kind": "core",
                    "salience": 0.91,
                    "confidence": 0.86,
                    "evidence": "RAG grounds answers in retrieved context.",
                    "aliases": ["RAG"],
                },
                {
                    "name": "vector search",
                    "description": "Similarity search over vector embeddings",
                    "kind": "supporting",
                    "salience": 0.72,
                    "confidence": 0.8,
                    "evidence": "Vector search retrieves candidate passages.",
                },
            ],
            "relations": [
                {
                    "source": "RAG",
                    "target": "vector search",
                    "type": "depends_on",
                    "confidence": 0.82,
                    "evidence": "RAG uses vector search to retrieve context.",
                    "rationale": "RAG requires retrieval before generation.",
                }
            ],
            "weakened": ["old keyword search"],
            "contradicted_relations": [
                {
                    "source": "RAG",
                    "target": "vector search",
                    "type": "contrasts",
                }
            ],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text", task="answer grounding", source="unit")

        assert obs.domain == "ai systems"
        assert obs.concepts == [
            "retrieval augmented generation",
            "vector search",
        ]
        assert obs.relations == [
            ("retrieval augmented generation", "vector search", "dependence")
        ]
        assert obs.contradicted_relations == [
            ("retrieval augmented generation", "vector search", "conflict")
        ]
        assert obs.weakened == ["old keyword search"]
        concept_meta = obs.extraction_metadata["concepts"][
            "retrieval augmented generation"
        ]
        assert concept_meta["kind"] == "core"
        assert concept_meta["aliases"] == ["RAG"]
        assert concept_meta["salience"] == 0.91
        rel_meta = obs.extraction_metadata["relations"][0]
        assert "probability" not in rel_meta
        assert "confidence" not in rel_meta
        assert "requires retrieval" in rel_meta["rationale"]

    def test_presets_are_included_in_prompt_and_observation(self):
        response = json.dumps({
            "concepts": [{"uid": "c1", "name": "a"}, {"uid": "c2", "name": "b"}],
            "relations": [
                {
                    "source": "c1",
                    "target": "c2",
                    "type": "supports",
                    "probability": 0.7,
                }
            ],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract(
            "text",
            preset_relations=[
                RelationPrior(
                    source="a",
                    target="b",
                    relation_type="supports",
                    probability=0.4,
                )
            ],
        )

        assert "## Preset Relations" in extractor._provider.user
        assert '"type": "enables"' in extractor._provider.user
        assert '"probability": 0.4' not in extractor._provider.user
        assert obs.relation_priors[0].probability == 0.4
        assert obs.relations == [("c1", "c2", "enables")]
        assert "probability" not in obs.extraction_metadata["relations"][0]

    def test_parses_local_concept_uids_for_ambiguous_labels(self):
        response = json.dumps({
            "domain": "ambiguous labels",
            "concepts": [
                {
                    "uid": "c1",
                    "name": "Apple",
                    "sense": "technology company",
                    "kind": "entity",
                    "description": "Consumer technology company",
                },
                {
                    "uid": "c2",
                    "name": "apple",
                    "sense": "fruit",
                    "kind": "entity",
                    "description": "Edible fruit",
                },
            ],
            "relations": [
                {
                    "source": "c1",
                    "target": "c2",
                    "type": "contrasts",
                    "rationale": "Same label, different concept sense.",
                }
            ],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("Apple is not the apple you eat.")

        assert [c.uid for c in obs.concept_candidates] == ["c1", "c2"]
        assert obs.concept_candidates[0].sense == "technology company"
        assert obs.concept_candidates[1].sense == "fruit"
        assert obs.relations == [("c1", "c2", "conflict")]

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
        assert obs.extraction_metadata["parse_warnings"]


class TestExtractionValidation:
    def test_invalid_relation_type_falls_back_to_generic_relation(self):
        response = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "a", "target": "b", "type": "invented_type"}],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.relations == [("a", "b", "generic_relation")]

    def test_relation_with_unknown_concept_is_dropped(self):
        response = json.dumps({
            "concepts": [{"name": "a"}],
            "relations": [{"source": "a", "target": "nonexistent", "type": "depends_on"}],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.relations == []
        assert obs.extraction_metadata["dropped_relations"][0]["target"] == "nonexistent"

    def test_relation_endpoint_can_match_alias(self):
        response = json.dumps({
            "concepts": [
                {"name": "retrieval augmented generation", "aliases": ["RAG"]},
                {"name": "grounded answer"},
            ],
            "relations": [
                {"source": "RAG", "target": "grounded answer", "type": "supports"},
            ],
        })
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")
        assert obs.relations == [
            ("retrieval augmented generation", "grounded answer", "enables")
        ]

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
        from world0.schemas.relation import semantic_relation_names

        names = semantic_relation_names()
        concepts = [{"name": f"c{i}"} for i in range(len(names) * 2)]
        relations = []
        for i, relation_name in enumerate(names):
            relations.append({
                "source": f"c{i * 2}",
                "target": f"c{i * 2 + 1}",
                "type": relation_name,
            })

        response = json.dumps({"concepts": concepts, "relations": relations})
        extractor = ConceptExtractor(FakeLLM(response))
        obs = extractor.extract("text")

        extracted_types = {r[2] for r in obs.relations}
        assert extracted_types == set(names)
