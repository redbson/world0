"""SQLite-backed persistence for World 0.

One file, one transaction per flush.  ``JsonStore`` writes every dirty
concept and relation as its own file and rewrites the file in full, which
is fine for a few thousand records and a problem for a world that is
consolidated after every observation.  This backend keeps the same
contract (``store.base.Store`` / ``core.StorageBackend``) but stores each
record as a JSON payload row keyed by id, so a flush of N dirty records is
N upserts inside a single transaction, and loading the world is one
sequential scan per table.

Schema::

    concepts (id TEXT PRIMARY KEY, payload TEXT NOT NULL)
    relations(id TEXT PRIMARY KEY, payload TEXT NOT NULL)
    sources  (id TEXT PRIMARY KEY, payload TEXT NOT NULL)
    state    (id TEXT PRIMARY KEY, payload TEXT NOT NULL)    -- id = 'world'

Payloads are the same pydantic JSON the file backend writes, so a world
can be moved between backends by re-saving every record.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationEdge
from world0.schemas.source import SourceRecord
from world0.store.base import Store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts  (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relations (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources   (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state     (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
"""

_STATE_KEY = "world"


class SqliteStore(Store):
    """Single-file SQLite persistence implementing the ``Store`` contract."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._conn.close()

    # ── generic helpers ───────────────────────────────────────────────

    def _upsert(self, table: str, rows: Iterable[tuple[str, str]]) -> None:
        with self._conn:
            self._conn.executemany(
                f"INSERT INTO {table} (id, payload) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                list(rows),
            )

    def _delete(self, table: str, ids: Iterable[str]) -> None:
        with self._conn:
            self._conn.executemany(
                f"DELETE FROM {table} WHERE id = ?", [(i,) for i in ids]
            )

    def _load_one(self, table: str, record_id: str) -> str | None:
        row = self._conn.execute(
            f"SELECT payload FROM {table} WHERE id = ?", (record_id,)
        ).fetchone()
        return row[0] if row else None

    def _load_all(self, table: str) -> list[str]:
        return [
            row[0]
            for row in self._conn.execute(
                f"SELECT payload FROM {table} ORDER BY id"
            )
        ]

    # ── concepts ──────────────────────────────────────────────────────

    def save_concept(self, concept: ConceptNode) -> None:
        self._upsert("concepts", [(concept.id, concept.model_dump_json())])

    def load_concept(self, concept_id: str) -> ConceptNode | None:
        payload = self._load_one("concepts", concept_id)
        return ConceptNode.model_validate_json(payload) if payload else None

    def load_all_concepts(self) -> list[ConceptNode]:
        return [ConceptNode.model_validate_json(p) for p in self._load_all("concepts")]

    def delete_concept(self, concept_id: str) -> None:
        self._delete("concepts", [concept_id])

    def save_concepts_batch(self, concepts: list[ConceptNode]) -> None:
        self._upsert("concepts", ((c.id, c.model_dump_json()) for c in concepts))

    def delete_concepts_batch(self, concept_ids: list[str]) -> None:
        self._delete("concepts", concept_ids)

    # ── relations ─────────────────────────────────────────────────────

    def save_relation(self, relation: RelationEdge) -> None:
        self._upsert("relations", [(relation.id, relation.model_dump_json())])

    def load_relation(self, relation_id: str) -> RelationEdge | None:
        payload = self._load_one("relations", relation_id)
        return RelationEdge.model_validate_json(payload) if payload else None

    def load_all_relations(self) -> list[RelationEdge]:
        return [RelationEdge.model_validate_json(p) for p in self._load_all("relations")]

    def delete_relation(self, relation_id: str) -> None:
        self._delete("relations", [relation_id])

    def save_relations_batch(self, relations: list[RelationEdge]) -> None:
        self._upsert("relations", ((r.id, r.model_dump_json()) for r in relations))

    def delete_relations_batch(self, relation_ids: list[str]) -> None:
        self._delete("relations", relation_ids)

    # ── sources ───────────────────────────────────────────────────────

    def save_source(self, source: SourceRecord) -> None:
        self._upsert("sources", [(source.id, source.model_dump_json())])

    def load_source(self, source_id: str) -> SourceRecord | None:
        payload = self._load_one("sources", source_id)
        return SourceRecord.model_validate_json(payload) if payload else None

    def load_all_sources(self) -> list[SourceRecord]:
        return [SourceRecord.model_validate_json(p) for p in self._load_all("sources")]

    # ── state ─────────────────────────────────────────────────────────

    def save_state(self, state: dict) -> None:
        self._upsert(
            "state", [(_STATE_KEY, json.dumps(state, default=str))]
        )

    def load_state(self) -> dict:
        payload = self._load_one("state", _STATE_KEY)
        return json.loads(payload) if payload else {}
