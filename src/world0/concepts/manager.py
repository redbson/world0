"""Concept lifecycle management — create, find, reinforce, merge, decay."""

from __future__ import annotations

from world0.schemas.concept import ConceptNode, Maturity
from world0.store.base import Store


class ConceptManager:
    """Manages the lifecycle of concepts in the cognitive world.

    Concepts are living units: they are created as embryonic, reinforced
    through repeated observation, promoted through maturity stages, and
    eventually fade if unused.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._concepts: dict[str, ConceptNode] = {}
        self._name_index: dict[str, str] = {}  # normalized name/alias → id
        self._dirty: set[str] = set()  # concept ids with unsaved changes

    def load(self) -> None:
        """Load all concepts from persistent store into memory."""
        self._concepts.clear()
        self._name_index.clear()
        for node in self._store.load_all_concepts():
            self._concepts[node.id] = node
            self._index_names(node)

    def save_all(self) -> None:
        """Persist all concepts to store (batch)."""
        self._store.save_concepts_batch(list(self._concepts.values()))
        self._dirty.clear()

    def flush(self) -> None:
        """Persist only dirty (modified) concepts to store."""
        if not self._dirty:
            return
        dirty_concepts = [
            self._concepts[cid] for cid in self._dirty if cid in self._concepts
        ]
        self._store.save_concepts_batch(dirty_concepts)
        self._dirty.clear()

    def mark_dirty(self, concept_id: str) -> None:
        """Mark a concept as having unsaved changes."""
        self._dirty.add(concept_id)

    # ── creation ──────────────────────────────────────────────────────

    def get_or_create(
        self,
        name: str,
        *,
        origin: str = "",
        task: str = "",
    ) -> tuple[ConceptNode, bool]:
        """Find existing concept by name/alias, or create as embryonic.

        Returns (concept, is_new).
        """
        existing = self.resolve(name)
        if existing:
            return existing, False

        node = ConceptNode(
            name=name,
            confidence=0.15,
            maturity=Maturity.EMBRYONIC,
            origin=origin,
        )
        self._concepts[node.id] = node
        self._index_names(node)
        self._dirty.add(node.id)
        return node, True

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, concept_id: str) -> ConceptNode | None:
        return self._concepts.get(concept_id)

    def resolve(self, name_or_id: str) -> ConceptNode | None:
        """Resolve by id, then by name/alias."""
        if name_or_id in self._concepts:
            return self._concepts[name_or_id]
        cid = self._name_index.get(name_or_id.strip().lower())
        if cid:
            return self._concepts.get(cid)
        return None

    def all(self) -> list[ConceptNode]:
        return list(self._concepts.values())

    def by_maturity(self, maturity: Maturity) -> list[ConceptNode]:
        return [c for c in self._concepts.values() if c.maturity == maturity]

    # ── reinforcement ─────────────────────────────────────────────────

    def reinforce(
        self, concept_id: str, *, source: str = "", task: str = ""
    ) -> ConceptNode | None:
        """Activate and reinforce a concept."""
        node = self._concepts.get(concept_id)
        if not node:
            return None
        node.activate(source=source, task=task)
        self._dirty.add(node.id)
        return node

    def update_description(self, concept_id: str, description: str) -> None:
        """Agent refines the description of a concept."""
        node = self._concepts.get(concept_id)
        if node and description:
            node.description = description
            self._dirty.add(node.id)

    def add_alias(self, concept_id: str, alias: str) -> bool:
        """Add an alias to a concept and update the name index.

        Returns True if the alias was added, False if it already existed
        or conflicts with another concept.
        """
        node = self._concepts.get(concept_id)
        if not node:
            return False

        normalized = alias.strip().lower()
        if not normalized:
            return False

        # Check for conflicts: alias must not resolve to a different concept
        existing_id = self._name_index.get(normalized)
        if existing_id and existing_id != concept_id:
            return False

        if normalized not in [a.strip().lower() for a in node.aliases]:
            node.aliases.append(alias.strip())
            self._name_index[normalized] = concept_id
            self._dirty.add(concept_id)

        return True

    def set_aliases(self, concept_id: str, aliases: list[str]) -> None:
        """Replace all aliases and rebuild the index for this concept."""
        node = self._concepts.get(concept_id)
        if not node:
            return

        # Remove old alias entries from index
        for old_name in node.all_names():
            if old_name != node.normalized_name():
                self._name_index.pop(old_name, None)

        node.aliases = [a.strip() for a in aliases if a.strip()]

        # Re-index all names
        self._index_names(node)
        self._dirty.add(concept_id)

    # ── mutation ──────────────────────────────────────────────────────

    def update_maturity(self, concept_id: str, maturity: Maturity) -> None:
        node = self._concepts.get(concept_id)
        if node:
            node.maturity = maturity
            self._dirty.add(concept_id)

    def remove(self, concept_id: str) -> bool:
        node = self._concepts.pop(concept_id, None)
        if node:
            for n in node.all_names():
                self._name_index.pop(n, None)
            self._store.delete_concept(concept_id)
            return True
        return False

    # ── internals ─────────────────────────────────────────────────────

    def _index_names(self, node: ConceptNode) -> None:
        for n in node.all_names():
            self._name_index[n] = node.id

    def connection_count(self, concept_id: str, relations: list) -> int:
        """Count how many relations involve this concept."""
        return sum(1 for r in relations if r.involves(concept_id))

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts
