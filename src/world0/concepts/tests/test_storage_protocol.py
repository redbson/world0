"""Concepts brick — testable against the StorageBackend Protocol alone.

These tests prove that ``ConceptManager`` does not need ``JsonStore``
or any concrete persistence layer.  Any object satisfying
``world0.core.StorageBackend`` works.
"""

from __future__ import annotations

from world0.concepts.api import Concepts
from world0.core import StorageBackend
from world0.core.test_doubles import FakeStorageBackend
from world0.schemas.concept import ConceptNode


def test_fake_storage_backend_satisfies_protocol() -> None:
    backend = FakeStorageBackend()
    assert isinstance(backend, StorageBackend)


def test_concepts_works_with_fake_backend() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)

    node, is_new = cm.get_or_create("Python", origin="test", task="t1")
    assert is_new
    assert node.name == "Python"

    cm.flush()
    # Flush should have persisted via save_concepts_batch.
    saved = [name for name, _ in backend.calls if name == "save_concepts_batch"]
    assert saved, "expected ConceptManager.flush() to call save_concepts_batch"


def test_concepts_resolve_after_reload_from_backend() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)
    cm.get_or_create("Postgres", origin="t", task="t")
    cm.flush()

    fresh = Concepts(backend)
    fresh.load()
    found = fresh.resolve("postgres")
    assert found is not None
    assert found.name == "Postgres"


def test_semantic_identity_allows_same_label_different_senses() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)

    company, company_new = cm.get_or_create(
        "Apple",
        kind="entity",
        sense="technology company",
        domain="technology",
    )
    fruit, fruit_new = cm.get_or_create(
        "Apple",
        kind="entity",
        sense="fruit",
        domain="food",
    )

    assert company_new
    assert fruit_new
    assert company.id != fruit.id
    assert cm.resolve("Apple") is None
    assert cm.get(company.id) is company
    assert cm.get(fruit.id) is fruit


def test_salience_kind_does_not_change_semantic_identity() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)

    first, is_new = cm.get_or_create(
        "RAG",
        kind="core",
        sense="retrieval augmented generation architecture",
        domain="ai",
    )
    second, is_new_again = cm.get_or_create(
        "RAG",
        kind="supporting",
        sense="retrieval augmented generation architecture",
        domain="ai",
    )

    assert is_new
    assert not is_new_again
    assert second.id == first.id


def test_synonym_tokens_collapse_when_semantic_boundary_matches() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)

    full, is_new = cm.get_or_create(
        "retrieval augmented generation",
        kind="entity",
        sense="retrieval augmented generation architecture",
        domain="ai",
        description="Architecture that grounds generation in retrieved context.",
    )
    acronym, is_new_again = cm.get_or_create(
        "RAG",
        kind="entity",
        sense="retrieval augmented generation architecture",
        domain="ai",
        description="Architecture that grounds generation in retrieved context.",
        aliases=["retrieval augmented generation"],
    )

    assert is_new
    assert not is_new_again
    assert acronym.id == full.id
    assert cm.resolve("RAG") is full
    assert "RAG" in full.aliases


def test_generic_sense_does_not_collapse_different_tokens() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)

    apple, apple_new = cm.get_or_create(
        "apple",
        kind="entity",
        sense="fruit",
        domain="food",
        description="A fruit from an apple tree.",
    )
    orange, orange_new = cm.get_or_create(
        "orange",
        kind="entity",
        sense="fruit",
        domain="food",
        description="A citrus fruit.",
    )

    assert apple_new
    assert orange_new
    assert apple.id != orange.id


def test_concept_representation_uses_token_feature_uid() -> None:
    node = ConceptNode(
        id="abc123",
        name="Apple",
        sense="Technology Company",
        kind="entity",
        domain="technology",
    )

    assert node.representation() == "apple.technology-company.abc123"


def test_merge_via_facade_uses_only_protocol_surface() -> None:
    from world0.core.test_doubles import FakeRelationStore

    backend = FakeStorageBackend()
    cm = Concepts(backend)
    cm.get_or_create("PostgreSQL", origin="t", task="t")
    cm.get_or_create("Postgres", origin="t", task="t")

    keeper = cm.resolve("PostgreSQL")
    absorbed = cm.resolve("Postgres")
    assert keeper and absorbed

    rels = FakeRelationStore()
    result = cm.merge(keeper.id, absorbed.id, relations=rels)
    assert result is not None
    assert cm.resolve("Postgres") is keeper  # alias migrated


def test_merge_transfers_token_and_source_refs() -> None:
    backend = FakeStorageBackend()
    cm = Concepts(backend)
    keeper, _ = cm.get_or_create("retrieval augmented generation", origin="s1")
    absorbed, _ = cm.get_or_create("RAG", origin="s2")
    keeper.record_source_ref(source_id="raw-1", source="s1", task="t")
    keeper.record_token_ref(token="retrieval augmented generation", source_id="raw-1", source="s1", task="t")
    absorbed.record_source_ref(source_id="raw-2", source="s2", task="t")
    absorbed.record_token_ref(token="RAG", source_id="raw-2", source="s2", task="t")

    merged = cm.merge(keeper.id, absorbed.id)

    assert merged is keeper
    assert any(ref.source_id == "raw-2" for ref in keeper.source_refs)
    rag_refs = [ref for ref in keeper.token_refs if ref.token == "RAG"]
    assert rag_refs
    assert rag_refs[0].source_id == "raw-2"
