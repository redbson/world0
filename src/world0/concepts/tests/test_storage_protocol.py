"""Concepts brick — testable against the StorageBackend Protocol alone.

These tests prove that ``ConceptManager`` does not need ``JsonStore``
or any concrete persistence layer.  Any object satisfying
``world0.core.StorageBackend`` works.
"""

from __future__ import annotations

from world0.concepts.api import Concepts
from world0.core import StorageBackend
from world0.core.test_doubles import FakeStorageBackend


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
