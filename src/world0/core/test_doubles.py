"""Reusable Protocol-satisfying test doubles for World 0.

Each subsystem in World 0 depends only on Protocols from
``world0.core.interfaces``.  This module provides minimal in-memory
implementations of those Protocols so any Lego brick can be tested in
isolation — the brick under test sees nothing of its sibling
implementations.

These doubles are *not* test fixtures pinned to one test suite; they
are the canonical examples of what a Protocol-satisfying object looks
like.  Real production code should never import them at runtime.

Usage::

    from world0.core.test_doubles import FakeStorageBackend
    cm = ConceptManager(FakeStorageBackend())  # no JSON I/O, no temp dirs

What ships:
- ``FakeStorageBackend``  — satisfies ``StorageBackend``
- ``FakeConceptStore``    — satisfies ``ConceptStore`` (and Reader)
- ``FakeRelationStore``   — satisfies ``RelationStore`` (and Reader)
- ``FakeHebbianLearner``  — satisfies ``HebbianLearner``
- ``FakeColorField``      — satisfies ``ColorField``
- ``FakeDecayPolicy``     — satisfies ``DecayPolicy``
- ``FakeLifecyclePolicy`` — satisfies ``LifecyclePolicy``
- ``FakeWorldView``       — satisfies ``WorldView``

Every fake records the call list it has seen on a ``calls`` attribute,
making it trivial to assert "the pipeline invoked X with these args".
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge, RelationType

if TYPE_CHECKING:
    from world0.schemas.community import Community
    from world0.schemas.context import Perspective


# ── StorageBackend ───────────────────────────────────────────────────


class FakeStorageBackend:
    """In-memory ``StorageBackend`` — no filesystem, no JSON."""

    def __init__(self) -> None:
        self._concepts: dict[str, ConceptNode] = {}
        self._relations: dict[str, RelationEdge] = {}
        self._state: dict = {}
        self.calls: list[tuple[str, tuple]] = []

    def _record(self, name: str, *args: object) -> None:
        self.calls.append((name, args))

    # concepts
    def save_concept(self, concept: ConceptNode) -> None:
        self._concepts[concept.id] = concept.model_copy(deep=True)
        self._record("save_concept", concept.id)

    def load_concept(self, concept_id: str) -> ConceptNode | None:
        node = self._concepts.get(concept_id)
        return node.model_copy(deep=True) if node else None

    def load_all_concepts(self) -> list[ConceptNode]:
        return [c.model_copy(deep=True) for c in self._concepts.values()]

    def delete_concept(self, concept_id: str) -> None:
        self._concepts.pop(concept_id, None)
        self._record("delete_concept", concept_id)

    def save_concepts_batch(self, concepts: list[ConceptNode]) -> None:
        for c in concepts:
            self._concepts[c.id] = c.model_copy(deep=True)
        self._record("save_concepts_batch", tuple(c.id for c in concepts))

    def delete_concepts_batch(self, concept_ids: list[str]) -> None:
        for cid in concept_ids:
            self._concepts.pop(cid, None)
        self._record("delete_concepts_batch", tuple(concept_ids))

    # relations
    def save_relation(self, relation: RelationEdge) -> None:
        self._relations[relation.id] = relation.model_copy(deep=True)
        self._record("save_relation", relation.id)

    def load_relation(self, relation_id: str) -> RelationEdge | None:
        edge = self._relations.get(relation_id)
        return edge.model_copy(deep=True) if edge else None

    def load_all_relations(self) -> list[RelationEdge]:
        return [r.model_copy(deep=True) for r in self._relations.values()]

    def delete_relation(self, relation_id: str) -> None:
        self._relations.pop(relation_id, None)
        self._record("delete_relation", relation_id)

    def save_relations_batch(self, relations: list[RelationEdge]) -> None:
        for r in relations:
            self._relations[r.id] = r.model_copy(deep=True)
        self._record("save_relations_batch", tuple(r.id for r in relations))

    def delete_relations_batch(self, relation_ids: list[str]) -> None:
        for rid in relation_ids:
            self._relations.pop(rid, None)
        self._record("delete_relations_batch", tuple(relation_ids))

    # state
    def save_state(self, state: dict) -> None:
        self._state = dict(state)
        self._record("save_state", tuple(sorted(state.keys())))

    def load_state(self) -> dict:
        return dict(self._state)


# ── ConceptStore ─────────────────────────────────────────────────────


class FakeConceptStore:
    """In-memory ``ConceptStore``.  Sufficient for engine tests."""

    def __init__(self, *, seed: list[ConceptNode] | None = None) -> None:
        self._nodes: dict[str, ConceptNode] = {n.id: n for n in (seed or [])}
        self._dirty: set[str] = set()
        self.calls: list[tuple[str, tuple]] = []

    def _record(self, name: str, *args: object) -> None:
        self.calls.append((name, args))

    # reader
    def get(self, concept_id: str) -> ConceptNode | None:
        return self._nodes.get(concept_id)

    def resolve(self, name_or_id: str) -> ConceptNode | None:
        if name_or_id in self._nodes:
            return self._nodes[name_or_id]
        key = name_or_id.strip().lower()
        for node in self._nodes.values():
            if node.normalized_name() == key or key in [
                a.lower() for a in node.aliases
            ]:
                return node
        return None

    def all(self) -> list[ConceptNode]:
        return list(self._nodes.values())

    def by_maturity(self, maturity: Maturity) -> list[ConceptNode]:
        return [n for n in self._nodes.values() if n.maturity == maturity]

    def find_similar(
        self,
        text: str,
        *,
        domain: str = "",
        min_similarity: float = 0.3,
        limit: int = 5,
    ) -> list[tuple[ConceptNode, float]]:
        return []

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._nodes

    # writer
    def get_or_create(
        self,
        name: str,
        *,
        origin: str = "",
        task: str = "",
        description: str = "",
        domain: str = "",
        consolidate: bool = True,
    ) -> tuple[ConceptNode, bool]:
        existing = self.resolve(name)
        if existing:
            self._record("get_or_create.existing", name)
            return existing, False
        node = ConceptNode(
            name=name,
            description=description,
            domain=domain,
            origin=origin,
        )
        self._nodes[node.id] = node
        self._dirty.add(node.id)
        self._record("get_or_create.new", name)
        return node, True

    def reinforce(
        self, concept_id: str, *, source: str = "", task: str = ""
    ) -> ConceptNode | None:
        node = self._nodes.get(concept_id)
        if not node:
            return None
        node.activate(source=source, task=task)
        self._dirty.add(concept_id)
        self._record("reinforce", concept_id)
        return node

    def weaken(
        self, concept_id: str, *, source: str = "", task: str = ""
    ) -> ConceptNode | None:
        node = self._nodes.get(concept_id)
        if not node:
            return None
        node.weaken(source=source, task=task)
        self._dirty.add(concept_id)
        self._record("weaken", concept_id)
        return node

    def update_description(self, concept_id: str, description: str) -> None:
        node = self._nodes.get(concept_id)
        if node and description:
            node.description = description
            self._dirty.add(concept_id)

    def add_alias(self, concept_id: str, alias: str) -> bool:
        node = self._nodes.get(concept_id)
        if not node or alias.lower() in [a.lower() for a in node.aliases]:
            return False
        node.aliases.append(alias)
        self._dirty.add(concept_id)
        return True

    def set_aliases(self, concept_id: str, aliases: list[str]) -> None:
        node = self._nodes.get(concept_id)
        if node:
            node.aliases = list(aliases)
            self._dirty.add(concept_id)

    def update_maturity(self, concept_id: str, maturity: Maturity) -> None:
        node = self._nodes.get(concept_id)
        if node:
            node.maturity = maturity
            self._dirty.add(concept_id)

    def adjust_confidence(
        self, concept_id: str, delta: float
    ) -> ConceptNode | None:
        node = self._nodes.get(concept_id)
        if not node:
            return None
        node.confidence = max(0.0, min(1.0, node.confidence + delta))
        self._dirty.add(concept_id)
        return node

    def remove(self, concept_id: str) -> bool:
        if concept_id not in self._nodes:
            return False
        del self._nodes[concept_id]
        self._dirty.discard(concept_id)
        self._record("remove", concept_id)
        return True

    def merge(
        self,
        keeper_id: str,
        absorbed_id: str,
        relations=None,
    ) -> ConceptNode | None:
        keeper = self._nodes.get(keeper_id)
        absorbed = self._nodes.get(absorbed_id)
        if not keeper or not absorbed or keeper_id == absorbed_id:
            return None
        keeper.aliases.extend(
            a for a in [absorbed.name, *absorbed.aliases]
            if a.lower() != keeper.name.lower()
            and a.lower() not in [x.lower() for x in keeper.aliases]
        )
        del self._nodes[absorbed_id]
        if relations is not None:
            relations.migrate_concept(absorbed_id, keeper_id)
        self._dirty.add(keeper_id)
        self._record("merge", keeper_id, absorbed_id)
        return keeper

    def split(
        self,
        concept_id: str,
        new_name: str,
        *,
        aliases_to_move: list[str] | None = None,
        description: str = "",
        domain: str = "",
    ) -> ConceptNode | None:
        source = self._nodes.get(concept_id)
        if not source:
            return None
        new_node = ConceptNode(
            name=new_name,
            aliases=list(aliases_to_move or []),
            description=description,
            domain=domain or source.domain,
        )
        if aliases_to_move:
            keep = [a for a in source.aliases if a not in aliases_to_move]
            source.aliases = keep
        self._nodes[new_node.id] = new_node
        self._dirty.add(new_node.id)
        self._dirty.add(concept_id)
        self._record("split", concept_id, new_name)
        return new_node

    def mark_dirty(self, concept_id: str) -> None:
        self._dirty.add(concept_id)

    def flush(self) -> None:
        self._record("flush", tuple(sorted(self._dirty)))
        self._dirty.clear()

    def save_all(self) -> None:
        self._record("save_all", ())

    def load(self) -> None:
        self._record("load", ())

    def connection_count(self, concept_id: str, relations: list) -> int:
        return sum(1 for r in relations if r.involves(concept_id))


# ── RelationStore ────────────────────────────────────────────────────


class FakeRelationStore:
    """In-memory ``RelationStore``."""

    def __init__(
        self, *, seed: list[RelationEdge] | None = None
    ) -> None:
        self._edges: dict[str, RelationEdge] = {e.id: e for e in (seed or [])}
        self._dirty: set[str] = set()
        self.calls: list[tuple[str, tuple]] = []

    def _record(self, name: str, *args: object) -> None:
        self.calls.append((name, args))

    # reader
    def get(self, relation_id: str) -> RelationEdge | None:
        return self._edges.get(relation_id)

    def all(self) -> list[RelationEdge]:
        return list(self._edges.values())

    def for_concept(self, concept_id: str) -> list[RelationEdge]:
        return [e for e in self._edges.values() if e.involves(concept_id)]

    def neighbors(self, concept_id: str) -> list[str]:
        result: list[str] = []
        for e in self._edges.values():
            other = e.other_end(concept_id)
            if other:
                result.append(other)
        return result

    def find_between(
        self,
        id_a: str,
        id_b: str,
        relation_type: RelationType | None = None,
    ) -> RelationEdge | None:
        for e in self._edges.values():
            if {e.source_id, e.target_id} == {id_a, id_b}:
                if relation_type is None or e.relation_type == relation_type:
                    return e
        return None

    def find_any_between(
        self, id_a: str, id_b: str
    ) -> list[RelationEdge]:
        return [
            e for e in self._edges.values()
            if {e.source_id, e.target_id} == {id_a, id_b}
        ]

    def __len__(self) -> int:
        return len(self._edges)

    # writer
    def discover(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType = RelationType.RELATED_TO,
        *,
        provenance: str = "",
        is_explicit: bool = True,
    ) -> tuple[RelationEdge, bool]:
        existing = self.find_between(source_id, target_id, relation_type)
        if existing:
            return existing, False
        edge = RelationEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            is_explicit=is_explicit,
            provenance=provenance,
        )
        self._edges[edge.id] = edge
        self._dirty.add(edge.id)
        self._record("discover", source_id, target_id, relation_type.value)
        return edge, True

    def reinforce(
        self, relation_id: str, provenance: str = ""
    ) -> RelationEdge | None:
        edge = self._edges.get(relation_id)
        if not edge:
            return None
        edge.reinforce(provenance=provenance)
        self._dirty.add(relation_id)
        self._record("reinforce", relation_id)
        return edge

    def weaken(
        self, relation_id: str, provenance: str = ""
    ) -> RelationEdge | None:
        edge = self._edges.get(relation_id)
        if not edge:
            return None
        edge.weaken(provenance=provenance)
        self._dirty.add(relation_id)
        self._record("weaken", relation_id)
        return edge

    def refine_type(
        self, relation_id: str, new_type: RelationType
    ) -> None:
        edge = self._edges.get(relation_id)
        if edge:
            edge.relation_type = new_type
            self._dirty.add(relation_id)

    def adjust_strength(
        self,
        relation_id: str,
        *,
        weight_delta: float = 0.0,
        confidence_delta: float = 0.0,
    ) -> RelationEdge | None:
        edge = self._edges.get(relation_id)
        if not edge:
            return None
        edge.weight = max(0.0, min(1.0, edge.weight + weight_delta))
        edge.confidence = max(0.0, min(1.0, edge.confidence + confidence_delta))
        self._dirty.add(relation_id)
        return edge

    def remove(self, relation_id: str) -> bool:
        if relation_id not in self._edges:
            return False
        del self._edges[relation_id]
        self._dirty.discard(relation_id)
        self._record("remove", relation_id)
        return True

    def remove_for_concept(self, concept_id: str) -> int:
        ids = [e.id for e in self._edges.values() if e.involves(concept_id)]
        for rid in ids:
            del self._edges[rid]
            self._dirty.discard(rid)
        return len(ids)

    def migrate_concept(self, old_id: str, new_id: str) -> int:
        moved = 0
        for edge in self._edges.values():
            changed = False
            if edge.source_id == old_id:
                edge.source_id = new_id
                changed = True
            if edge.target_id == old_id:
                edge.target_id = new_id
                changed = True
            if changed:
                self._dirty.add(edge.id)
                moved += 1
        return moved

    def mark_dirty(self, relation_id: str) -> None:
        self._dirty.add(relation_id)

    def flush(self) -> None:
        self._record("flush", tuple(sorted(self._dirty)))
        self._dirty.clear()

    def save_all(self) -> None:
        self._record("save_all", ())

    def load(self) -> None:
        self._record("load", ())


# ── HebbianLearner / DecayPolicy / LifecyclePolicy / ColorField ─────


class FakeHebbianLearner:
    """Records ``learn`` calls; never creates relations on its own."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.next_new_relation_ids: list[str] = []

    def learn(
        self, concept_ids: list[str], *, provenance: str = ""
    ) -> list[str]:
        self.calls.append((list(concept_ids), provenance))
        out, self.next_new_relation_ids = self.next_new_relation_ids, []
        return out


class FakeDecayPolicy:
    """Returns canned id lists; lets pipelines exercise the wire-up."""

    def __init__(
        self,
        *,
        decayed_concepts: list[str] | None = None,
        decayed_relations: list[str] | None = None,
        pruned_concepts: list[str] | None = None,
        pruned_relations: list[str] | None = None,
    ) -> None:
        self._dc = decayed_concepts or []
        self._dr = decayed_relations or []
        self._pc = pruned_concepts or []
        self._pr = pruned_relations or []
        self.calls: list[str] = []

    def decay_concepts(self) -> list[str]:
        self.calls.append("decay_concepts")
        return list(self._dc)

    def decay_relations(self) -> list[str]:
        self.calls.append("decay_relations")
        return list(self._dr)

    def prune_concepts(self, threshold: float = 0.02) -> list[str]:
        self.calls.append("prune_concepts")
        return list(self._pc)

    def prune_relations(self, threshold: float = 0.02) -> list[str]:
        self.calls.append("prune_relations")
        return list(self._pr)


class FakeLifecyclePolicy:
    def __init__(
        self,
        *,
        promoted: list[str] | None = None,
        demoted: list[str] | None = None,
    ) -> None:
        self._promoted = promoted or []
        self._demoted = demoted or []
        self.calls = 0

    def evaluate(self) -> tuple[list[str], list[str]]:
        self.calls += 1
        return list(self._promoted), list(self._demoted)


class FakeColorField:
    def __init__(self) -> None:
        self.seed_and_diffuse_calls: list[tuple[list[str], str]] = []
        self.fade_calls = 0
        self.settle_calls = 0
        self.seed_from_communities_calls: list[int] = []

    def seed_and_diffuse(
        self,
        concept_ids: list[str],
        *,
        domain_label: str,
        steps: int = 2,
        rate: float = 0.4,
        decay: float = 0.5,
    ) -> None:
        self.seed_and_diffuse_calls.append((list(concept_ids), domain_label))

    def settle(self, steps: int = 1, rate: float = 0.2) -> None:
        self.settle_calls += 1

    def fade_step(
        self, *, dt: float = 1.0, tau: float = 1.0, evaporate: float = 0.0
    ) -> int:
        self.fade_calls += 1
        return 0

    def seed_from_communities(
        self, communities: list["Community"], *, diffuse: bool = True
    ) -> int:
        self.seed_from_communities_calls.append(len(communities))
        return 0


# ── WorldView ────────────────────────────────────────────────────────


class FakeWorldView:
    """Bundles a fake ConceptStore + RelationStore as a ``WorldView``."""

    def __init__(
        self,
        *,
        concepts: FakeConceptStore | None = None,
        relations: FakeRelationStore | None = None,
    ) -> None:
        self._concepts = concepts or FakeConceptStore()
        self._relations = relations or FakeRelationStore()

    @property
    def concepts(self) -> FakeConceptStore:
        return self._concepts

    @property
    def relations(self) -> FakeRelationStore:
        return self._relations


# ── Convenience factories ────────────────────────────────────────────


def make_concept(
    name: str = "x",
    *,
    domain: str = "",
    confidence: float = 0.5,
    maturity: Maturity = Maturity.EMBRYONIC,
    concept_id: str | None = None,
) -> ConceptNode:
    """Build a minimal ConceptNode for tests."""
    return ConceptNode(
        id=concept_id or uuid.uuid4().hex[:12],
        name=name,
        domain=domain,
        confidence=confidence,
        maturity=maturity,
    )


def make_edge(
    source_id: str,
    target_id: str,
    *,
    relation_type: RelationType = RelationType.RELATED_TO,
    weight: float = 0.4,
) -> RelationEdge:
    return RelationEdge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=weight,
    )


__all__ = [
    "FakeStorageBackend",
    "FakeConceptStore",
    "FakeRelationStore",
    "FakeHebbianLearner",
    "FakeDecayPolicy",
    "FakeLifecyclePolicy",
    "FakeColorField",
    "FakeWorldView",
    "make_concept",
    "make_edge",
]
