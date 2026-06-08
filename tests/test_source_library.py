"""Tests for raw source library and concept source cards."""

import json

from world0 import World
from world0.llm.base import LLMProvider
from world0.sources import SourceLibrary
from world0.store.json_store import JsonStore


class SourceLLM(LLMProvider):
    def complete_json(self, system: str, user: str) -> str:
        return json.dumps({
            "domain": "rag systems",
            "concepts": [
                {
                    "name": "retrieval augmented generation",
                    "description": "Generation grounded in retrieved context",
                    "evidence": "RAG grounds answers in retrieved context.",
                },
                {
                    "name": "vector search",
                    "description": "Similarity search for retrieval",
                    "evidence": "Vector search finds relevant chunks.",
                },
            ],
            "relations": [
                {
                    "source": "retrieval augmented generation",
                    "target": "vector search",
                    "type": "depends_on",
                    "evidence": "RAG uses vector search.",
                    "rationale": "RAG requires retrieval.",
                }
            ],
        })


def test_source_library_records_and_searches_raw_text(tmp_path):
    store = JsonStore(tmp_path)
    library = SourceLibrary(store)

    record = library.record_raw(
        "RAG uses vector search to retrieve context.",
        task="agent grounding",
        source="note-1",
    )

    assert store.load_source(record.id).raw_text.startswith("RAG uses")
    assert (tmp_path / "sources" / f"{record.id}.json").exists()
    assert (tmp_path / "source_index.json").exists()
    assert library.search("vector context")[0].id == record.id


def test_ingest_text_persists_raw_source_and_concept_refs(tmp_path):
    raw = "RAG uses vector search to retrieve context."
    world = World(store_path=tmp_path, llm=SourceLLM())

    result = world.ingest_text(raw, task="agent grounding", source="note-1")

    assert "retrieval augmented generation" in result.new_concepts
    sources = world.sources.all()
    assert len(sources) == 1
    source = sources[0]
    assert source.raw_text == raw
    assert source.concepts == [
        "retrieval augmented generation",
        "vector search",
    ]
    assert source.relation_count == 1

    concept = world.concepts.resolve("retrieval augmented generation")
    assert concept is not None
    assert len(concept.source_refs) == 1
    ref = concept.source_refs[0]
    assert ref.source_id == source.id
    assert ref.source == "note-1"
    assert ref.task == "agent grounding"
    assert "RAG grounds answers" in ref.excerpt

    reloaded = World(store_path=tmp_path)
    reloaded_source = reloaded.sources.get(source.id)
    assert reloaded_source is not None
    assert reloaded_source.raw_text == raw
    reloaded_concept = reloaded.concepts.resolve("retrieval augmented generation")
    assert reloaded_concept.source_refs[0].source_id == source.id
