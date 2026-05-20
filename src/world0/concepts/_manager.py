"""Core ConceptManager — storage + lifecycle (reinforce/weaken/aliases).

Heavy operations live in sibling modules:

- ``_indexes.py``       — name + token in-memory indexes
- ``_consolidation.py`` — signature-based candidate matching
- ``_identity_ops.py``  — merge & split mutations

This file owns the dict-of-concepts, the dirty set, and the
storage/flush pipeline.  Every other operation delegates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.concepts._consolidation import SignatureMatcher
from world0.concepts._identity_ops import merge_concepts, split_concept
from world0.concepts._indexes import NameIndex, TokenIndex
from world0.schemas.concept import ConceptNode, Maturity, tokenize_signature

if TYPE_CHECKING:
    from world0.core import RelationStore, StorageBackend


class ConceptManager:
    """Manages the lifecycle of concepts in the cognitive world.

    Concepts are living units: they are created as embryonic, reinforced
    through repeated observation, promoted through maturity stages, and
    eventually fade if unused.

    Implements the ``ConceptStore`` Protocol from ``world0.core``.
    """

    def __init__(self, store: StorageBackend) -> None:
        self._store = store
        self._concepts: dict[str, ConceptNode] = {}
        self._name_index = NameIndex()
        self._token_index = TokenIndex()
        self._dirty: set[str] = set()
        self._matcher = SignatureMatcher(self._token_index, self._concepts.get)

    # ── persistence ───────────────────────────────────────────────────

    def load(self) -> None:
        """Load all concepts from persistent store into memory."""
        self._concepts.clear()
        self._name_index.clear()
        self._token_index.clear()
        for node in self._store.load_all_concepts():
            self._concepts[node.id] = node
            self._name_index.index_node(node)
            self._token_index.index_node(node)

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
        self._dirty.add(concept_id)

    # ── creation ──────────────────────────────────────────────────────

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
        """Find existing concept by name/alias, or create as embryonic.

        When ``consolidate`` is True (default) and a description is
        supplied, the manager first looks for a concept whose signature
        overlaps strongly enough to be considered the same unit (see
        ``_consolidation.py``).  In that case ``name`` is attached as
        an alias rather than producing a duplicate node.

        Returns (concept, is_new).
        """
        existing = self.resolve(name)
        if existing:
            return existing, False

        # Auto-consolidation only fires when there is a genuine
        # semantic signature to match on — pure name-token overlap
        # between short labels is too weak as evidence to merge.
        if consolidate and description:
            probe_tokens = tokenize_signature(name) | tokenize_signature(
                description
            )
            if probe_tokens:
                candidate = self._matcher.find_best_match(
                    probe_tokens, domain=domain
                )
                if candidate is not None:
                    self.add_alias(candidate.id, name)
                    if description and not candidate.description:
                        candidate.description = description
                        self._dirty.add(candidate.id)
                    return candidate, False

        node = ConceptNode(
            name=name,
            confidence=0.15,
            maturity=Maturity.EMBRYONIC,
            origin=origin,
            description=description,
            domain=domain,
        )
        self._concepts[node.id] = node
        self._name_index.index_node(node)
        self._token_index.index_node(node)
        self._dirty.add(node.id)
        return node, True

    def find_similar(
        self,
        text: str,
        *,
        domain: str = "",
        min_similarity: float = 0.3,
        limit: int = 5,
    ) -> list[tuple[ConceptNode, float]]:
        """Return (concept, similarity) ranked by signature overlap."""
        return self._matcher.find_similar(
            tokenize_signature(text),
            domain=domain,
            min_similarity=min_similarity,
            limit=limit,
        )

    # ── lookup ────────────────────────────────────────────────────────

    def get(self, concept_id: str) -> ConceptNode | None:
        return self._concepts.get(concept_id)

    def resolve(self, name_or_id: str) -> ConceptNode | None:
        """Resolve by id, then by name/alias."""
        if name_or_id in self._concepts:
            return self._concepts[name_or_id]
        cid = self._name_index.get(name_or_id)
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

    def weaken(
        self, concept_id: str, *, source: str = "", task: str = ""
    ) -> ConceptNode | None:
        """Apply disconfirmation evidence to a concept."""
        node = self._concepts.get(concept_id)
        if not node:
            return None
        node.weaken(source=source, task=task)
        self._dirty.add(node.id)
        return node

    def update_description(self, concept_id: str, description: str) -> None:
        """Agent refines the description of a concept."""
        node = self._concepts.get(concept_id)
        if node and description:
            node.description = description
            self._dirty.add(node.id)
            self._token_index.index_node(node)

    def add_alias(self, concept_id: str, alias: str) -> bool:
        """Add an alias to a concept and update the name index.

        Returns True if the alias is now attached to ``concept_id``,
        False if it conflicts with another concept.
        """
        node = self._concepts.get(concept_id)
        if not node:
            return False

        normalized = alias.strip().lower()
        if not normalized:
            return False

        if not self._name_index.add_unique(alias, concept_id):
            return False

        if normalized not in [a.strip().lower() for a in node.aliases]:
            node.aliases.append(alias.strip())
            self._token_index.index_node(node)
            self._dirty.add(concept_id)

        return True

    def set_aliases(self, concept_id: str, aliases: list[str]) -> None:
        """Replace all aliases and rebuild the index for this concept."""
        node = self._concepts.get(concept_id)
        if not node:
            return

        # Drop old alias entries from the name index (keep canonical name).
        for old_name in node.all_names():
            if old_name != node.normalized_name():
                self._name_index.remove_if_owned(old_name, concept_id)

        node.aliases = [a.strip() for a in aliases if a.strip()]

        self._name_index.index_node(node)
        self._token_index.index_node(node)
        self._dirty.add(concept_id)

    # ── mutation ──────────────────────────────────────────────────────

    def update_maturity(self, concept_id: str, maturity: Maturity) -> None:
        node = self._concepts.get(concept_id)
        if node:
            node.maturity = maturity
            self._dirty.add(concept_id)

    def adjust_confidence(
        self, concept_id: str, delta: float
    ) -> ConceptNode | None:
        """Apply a bounded confidence adjustment to a concept."""
        node = self._concepts.get(concept_id)
        if not node:
            return None
        node.confidence = min(1.0, max(0.01, node.confidence + delta))
        self._dirty.add(concept_id)
        return node

    def remove(self, concept_id: str) -> bool:
        node = self._concepts.pop(concept_id, None)
        if not node:
            return False
        # Only clear index entries that still point at this concept.
        # `merge()` may have already re-mapped some of these names to
        # the keeper; those must survive the removal.
        for n in node.all_names():
            self._name_index.remove_if_owned(n, concept_id)
        self._token_index.unindex(concept_id)
        self._store.delete_concept(concept_id)
        return True

    # ── identity operations (delegate to _identity_ops) ──────────────

    def merge(
        self,
        keeper_id: str,
        absorbed_id: str,
        relations: RelationStore | None = None,
    ) -> ConceptNode | None:
        return merge_concepts(self, keeper_id, absorbed_id, relations)

    def split(
        self,
        concept_id: str,
        new_name: str,
        *,
        aliases_to_move: list[str] | None = None,
        description: str = "",
        domain: str = "",
    ) -> ConceptNode | None:
        return split_concept(
            self,
            concept_id,
            new_name,
            aliases_to_move=aliases_to_move,
            description=description,
            domain=domain,
        )

    # ── misc ──────────────────────────────────────────────────────────

    def connection_count(self, concept_id: str, relations: list) -> int:
        """Count how many relations involve this concept."""
        return sum(1 for r in relations if r.involves(concept_id))

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts
