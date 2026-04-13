"""Relation lifecycle management — discover, reinforce, decay, refine."""

from __future__ import annotations

from world0.schemas.relation import RelationEdge, RelationType
from world0.store.base import Store


class RelationManager:
    """Manages the lifecycle of relations in the cognitive world.

    Relations are discovered through the Agent's work. They start weak
    and strengthen through repeated observation.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._relations: dict[str, RelationEdge] = {}
        self._by_concept: dict[str, list[str]] = {}  # concept_id → [relation_ids]
        self._dirty: set[str] = set()  # relation ids with unsaved changes

    def load(self) -> None:
        """Load all relations from persistent store."""
        self._relations.clear()
        self._by_concept.clear()
        for edge in self._store.load_all_relations():
            self._relations[edge.id] = edge
            self._index(edge)

    def save_all(self) -> None:
        """Persist all relations to store (batch)."""
        self._store.save_relations_batch(list(self._relations.values()))
        self._dirty.clear()

    def flush(self) -> None:
        """Persist only dirty (modified) relations to store."""
        if not self._dirty:
            return
        dirty_rels = [
            self._relations[rid] for rid in self._dirty if rid in self._relations
        ]
        self._store.save_relations_batch(dirty_rels)
        self._dirty.clear()

    def mark_dirty(self, relation_id: str) -> None:
        """Mark a relation as having unsaved changes."""
        self._dirty.add(relation_id)

    # ── discovery ─────────────────────────────────────────────────────

    def discover(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType = RelationType.RELATED_TO,
        *,
        provenance: str = "",
        is_explicit: bool = True,
    ) -> tuple[RelationEdge, bool]:
        """Find existing relation or discover a new one.

        Args:
            is_explicit: True for Agent-declared relations, False for
                Hebbian (auto-discovered) relations. Explicit relations
                can reach weight 1.0; Hebbian relations cap at 0.7.

        Returns (relation, is_new).
        """
        if source_id == target_id:
            raise ValueError("Cannot create a self-relation")

        existing = self.find_between(source_id, target_id, relation_type)
        if existing:
            return existing, False

        # Explicit relations start stronger than Hebbian (auto-discovered)
        init_weight = 0.3 if is_explicit else 0.15
        init_confidence = 0.3 if is_explicit else 0.15

        edge = RelationEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=init_weight,
            confidence=init_confidence,
            provenance=provenance,
            task_history=[provenance] if provenance else [],
            is_explicit=is_explicit,
        )
        self._relations[edge.id] = edge
        self._index(edge)
        self._dirty.add(edge.id)
        return edge, True

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, relation_id: str) -> RelationEdge | None:
        return self._relations.get(relation_id)

    def all(self) -> list[RelationEdge]:
        return list(self._relations.values())

    def for_concept(self, concept_id: str) -> list[RelationEdge]:
        rids = self._by_concept.get(concept_id, [])
        return [self._relations[rid] for rid in rids if rid in self._relations]

    def neighbors(self, concept_id: str) -> list[str]:
        result: list[str] = []
        for rel in self.for_concept(concept_id):
            other = rel.other_end(concept_id)
            if other:
                result.append(other)
        return result

    def find_between(
        self,
        id_a: str,
        id_b: str,
        relation_type: RelationType | None = None,
    ) -> RelationEdge | None:
        """Find a specific relation between two concepts."""
        for rel in self.for_concept(id_a):
            if not rel.involves(id_b):
                continue
            if relation_type is None or rel.relation_type == relation_type:
                return rel
        return None

    def find_any_between(self, id_a: str, id_b: str) -> list[RelationEdge]:
        return [r for r in self.for_concept(id_a) if r.involves(id_b)]

    # ── reinforcement ─────────────────────────────────────────────────

    def reinforce(self, relation_id: str, provenance: str = "") -> RelationEdge | None:
        edge = self._relations.get(relation_id)
        if not edge:
            return None
        edge.reinforce(provenance=provenance)
        self._dirty.add(edge.id)
        return edge

    def refine_type(self, relation_id: str, new_type: RelationType) -> None:
        """Agent refines a RELATED_TO relation to a more specific type."""
        edge = self._relations.get(relation_id)
        if edge:
            edge.relation_type = new_type
            self._dirty.add(edge.id)

    def adjust_strength(
        self,
        relation_id: str,
        *,
        weight_delta: float = 0.0,
        confidence_delta: float = 0.0,
    ) -> RelationEdge | None:
        """Apply bounded strength adjustments to a relation."""
        edge = self._relations.get(relation_id)
        if not edge:
            return None
        edge.weight = min(1.0, max(0.01, edge.weight + weight_delta))
        edge.confidence = min(1.0, max(0.01, edge.confidence + confidence_delta))
        self._dirty.add(edge.id)
        return edge

    # ── removal ───────────────────────────────────────────────────────

    def remove(self, relation_id: str) -> bool:
        edge = self._relations.pop(relation_id, None)
        if edge:
            self._unindex(edge)
            self._store.delete_relation(relation_id)
            return True
        return False

    def remove_for_concept(self, concept_id: str) -> int:
        rels = self.for_concept(concept_id)
        for r in rels:
            self._relations.pop(r.id, None)
            self._unindex(r)
            self._store.delete_relation(r.id)
        return len(rels)

    # ── internals ─────────────────────────────────────────────────────

    def _index(self, edge: RelationEdge) -> None:
        self._by_concept.setdefault(edge.source_id, []).append(edge.id)
        self._by_concept.setdefault(edge.target_id, []).append(edge.id)

    def _unindex(self, edge: RelationEdge) -> None:
        for cid in (edge.source_id, edge.target_id):
            ids = self._by_concept.get(cid, [])
            if edge.id in ids:
                ids.remove(edge.id)

    def __len__(self) -> int:
        return len(self._relations)
