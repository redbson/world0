"""Abstract store interface for World 0 persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationEdge
from world0.schemas.source import SourceRecord


class Store(ABC):
    """Persistence backend for the cognitive world.

    Abstracted so the implementation can be swapped (JSON → SQLite → etc.)
    without changing the cognitive logic.
    """

    # ── concepts ──────────────────────────────────────────────────────

    @abstractmethod
    def save_concept(self, concept: ConceptNode) -> None: ...

    @abstractmethod
    def load_concept(self, concept_id: str) -> ConceptNode | None: ...

    @abstractmethod
    def load_all_concepts(self) -> list[ConceptNode]: ...

    @abstractmethod
    def delete_concept(self, concept_id: str) -> None: ...

    def save_concepts_batch(self, concepts: list[ConceptNode]) -> None:
        """Save multiple concepts. Default: loop over save_concept."""
        for c in concepts:
            self.save_concept(c)

    def delete_concepts_batch(self, concept_ids: list[str]) -> None:
        """Delete multiple concepts. Default: loop over delete_concept."""
        for cid in concept_ids:
            self.delete_concept(cid)

    # ── relations ─────────────────────────────────────────────────────

    @abstractmethod
    def save_relation(self, relation: RelationEdge) -> None: ...

    @abstractmethod
    def load_relation(self, relation_id: str) -> RelationEdge | None: ...

    @abstractmethod
    def load_all_relations(self) -> list[RelationEdge]: ...

    @abstractmethod
    def delete_relation(self, relation_id: str) -> None: ...

    def save_relations_batch(self, relations: list[RelationEdge]) -> None:
        """Save multiple relations. Default: loop over save_relation."""
        for r in relations:
            self.save_relation(r)

    def delete_relations_batch(self, relation_ids: list[str]) -> None:
        """Delete multiple relations. Default: loop over delete_relation."""
        for rid in relation_ids:
            self.delete_relation(rid)

    # ── sources ────────────────────────────────────────────────────────

    def save_source(self, source: SourceRecord) -> None:
        raise NotImplementedError

    def load_source(self, source_id: str) -> SourceRecord | None:
        raise NotImplementedError

    def load_all_sources(self) -> list[SourceRecord]:
        raise NotImplementedError

    # ── state ─────────────────────────────────────────────────────────

    @abstractmethod
    def save_state(self, state: dict) -> None: ...

    @abstractmethod
    def load_state(self) -> dict: ...
