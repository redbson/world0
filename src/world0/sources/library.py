"""Source library — raw material captured before extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.schemas.source import SourceRecord, source_tokens

if TYPE_CHECKING:
    from world0.core import StorageBackend
    from world0.schemas.types import Observation


class SourceLibrary:
    """Stores and indexes raw source material separately from concepts."""

    def __init__(self, store: StorageBackend) -> None:
        self._store = store

    def record_raw(
        self,
        raw_text: str,
        *,
        task: str = "",
        source: str = "",
    ) -> SourceRecord:
        record = SourceRecord.from_raw(raw_text, task=task, source=source)
        existing = self._store.load_source(record.id)
        if existing:
            return existing
        self._store.save_source(record)
        return record

    def attach_observation(
        self,
        source_id: str,
        observation: Observation,
    ) -> SourceRecord | None:
        record = self._store.load_source(source_id)
        if record is None:
            return None
        record.attach_extraction(
            concepts=observation.concepts,
            relation_count=len(observation.relations),
            domain=observation.domain,
        )
        self._store.save_source(record)
        return record

    def get(self, source_id: str) -> SourceRecord | None:
        return self._store.load_source(source_id)

    def all(self) -> list[SourceRecord]:
        return self._store.load_all_sources()

    def search(self, query: str, *, limit: int = 10) -> list[SourceRecord]:
        query_tokens = set(source_tokens(query))
        if not query_tokens:
            return []
        scored: list[tuple[int, SourceRecord]] = []
        for record in self._store.load_all_sources():
            score = len(query_tokens & set(record.tokens))
            concept_text = " ".join(record.concepts).lower()
            score += sum(1 for token in query_tokens if token in concept_text)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [record for _, record in scored[:limit]]
