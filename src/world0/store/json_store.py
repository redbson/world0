"""JSON file-based persistence for World 0."""

from __future__ import annotations

import json
from pathlib import Path

from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationEdge
from world0.store.base import Store


class JsonStore(Store):
    """Stores concepts and relations as individual JSON files.

    Layout:
        {root}/concepts/{id}.json
        {root}/relations/{id}.json
        {root}/state.json
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._concepts_dir = self._root / "concepts"
        self._relations_dir = self._root / "relations"
        self._state_path = self._root / "state.json"

        self._concepts_dir.mkdir(parents=True, exist_ok=True)
        self._relations_dir.mkdir(parents=True, exist_ok=True)

    # ── concepts ──────────────────────────────────────────────────────

    def save_concept(self, concept: ConceptNode) -> None:
        path = self._concepts_dir / f"{concept.id}.json"
        path.write_text(concept.model_dump_json(indent=2), encoding="utf-8")

    def load_concept(self, concept_id: str) -> ConceptNode | None:
        path = self._concepts_dir / f"{concept_id}.json"
        if not path.exists():
            return None
        return ConceptNode.model_validate_json(path.read_text(encoding="utf-8"))

    def load_all_concepts(self) -> list[ConceptNode]:
        concepts = []
        for path in self._concepts_dir.glob("*.json"):
            concepts.append(
                ConceptNode.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return concepts

    def delete_concept(self, concept_id: str) -> None:
        path = self._concepts_dir / f"{concept_id}.json"
        if path.exists():
            path.unlink()

    def save_concepts_batch(self, concepts: list[ConceptNode]) -> None:
        for c in concepts:
            path = self._concepts_dir / f"{c.id}.json"
            path.write_text(c.model_dump_json(indent=2), encoding="utf-8")

    def delete_concepts_batch(self, concept_ids: list[str]) -> None:
        for cid in concept_ids:
            path = self._concepts_dir / f"{cid}.json"
            if path.exists():
                path.unlink()

    # ── relations ─────────────────────────────────────────────────────

    def save_relation(self, relation: RelationEdge) -> None:
        path = self._relations_dir / f"{relation.id}.json"
        path.write_text(relation.model_dump_json(indent=2), encoding="utf-8")

    def load_relation(self, relation_id: str) -> RelationEdge | None:
        path = self._relations_dir / f"{relation_id}.json"
        if not path.exists():
            return None
        return RelationEdge.model_validate_json(path.read_text(encoding="utf-8"))

    def load_all_relations(self) -> list[RelationEdge]:
        relations = []
        for path in self._relations_dir.glob("*.json"):
            relations.append(
                RelationEdge.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return relations

    def delete_relation(self, relation_id: str) -> None:
        path = self._relations_dir / f"{relation_id}.json"
        if path.exists():
            path.unlink()

    def save_relations_batch(self, relations: list[RelationEdge]) -> None:
        for r in relations:
            path = self._relations_dir / f"{r.id}.json"
            path.write_text(r.model_dump_json(indent=2), encoding="utf-8")

    def delete_relations_batch(self, relation_ids: list[str]) -> None:
        for rid in relation_ids:
            path = self._relations_dir / f"{rid}.json"
            if path.exists():
                path.unlink()

    # ── state ─────────────────────────────────────────────────────────

    def save_state(self, state: dict) -> None:
        self._state_path.write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )

    def load_state(self) -> dict:
        if not self._state_path.exists():
            return {}
        return json.loads(self._state_path.read_text(encoding="utf-8"))
