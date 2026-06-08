"""In-memory indexes used by ConceptManager.

Two independent indexes:

- ``NameIndex`` — normalized name/alias → concept_id, the canonical
  resolution path for ``ConceptManager.resolve(name)``.
- ``TokenIndex`` — signature token → set of concept ids, the cheap
  candidate-shortlist for signature-based consolidation
  (see ``_consolidation.py``).

Both are pure data structures: no I/O, no concept-lifecycle knowledge,
no awareness of dirty tracking.  ConceptManager owns the orchestration.
"""

from __future__ import annotations


class NameIndex:
    """Normalized name/alias → concept ids.

    A lexical label is not a concept identity.  If a label maps to more
    than one concept UID, ``get`` returns ``None`` so callers do not
    accidentally collapse distinct senses such as Apple/company and
    apple/fruit.
    """

    def __init__(self) -> None:
        self._map: dict[str, set[str]] = {}

    def clear(self) -> None:
        self._map.clear()

    def get(self, name: str) -> str | None:
        ids = self._map.get(name.strip().lower())
        if not ids or len(ids) != 1:
            return None
        return next(iter(ids))

    def add(self, name: str, concept_id: str) -> None:
        """Add ``concept_id`` to the label's candidate set."""
        normalized = name.strip().lower()
        if normalized:
            self._map.setdefault(normalized, set()).add(concept_id)

    def add_unique(self, name: str, concept_id: str) -> bool:
        """Set the mapping only if it does not already point elsewhere.

        Returns True if the entry now points at ``concept_id`` (either
        freshly added or already so), False if another concept owns it.
        """
        normalized = name.strip().lower()
        if not normalized:
            return False
        existing = self._map.get(normalized, set())
        if existing and existing != {concept_id}:
            return False
        self._map.setdefault(normalized, set()).add(concept_id)
        return True

    def remove_if_owned(self, name: str, concept_id: str) -> bool:
        """Remove the entry only if it currently points at ``concept_id``."""
        normalized = name.strip().lower()
        ids = self._map.get(normalized)
        if ids and concept_id in ids:
            ids.discard(concept_id)
            if not ids:
                self._map.pop(normalized, None)
            return True
        return False

    def index_node(self, node) -> None:
        """Index every alias variant of ``node``."""
        for n in node.all_names():
            self._map.setdefault(n, set()).add(node.id)


class TokenIndex:
    """Signature token → set of concept ids that contain that token."""

    def __init__(self) -> None:
        self._map: dict[str, set[str]] = {}

    def clear(self) -> None:
        self._map.clear()

    def candidates(self, tokens: set[str]) -> set[str]:
        """Concept ids that share at least one token with ``tokens``."""
        result: set[str] = set()
        for tok in tokens:
            ids = self._map.get(tok)
            if ids:
                result.update(ids)
        return result

    def index_node(self, node) -> None:
        """Refresh entries for ``node`` from its current signature tokens."""
        self.unindex(node.id)
        for tok in node.signature_tokens():
            self._map.setdefault(tok, set()).add(node.id)

    def unindex(self, concept_id: str) -> None:
        empty: list[str] = []
        for tok, ids in self._map.items():
            ids.discard(concept_id)
            if not ids:
                empty.append(tok)
        for tok in empty:
            del self._map[tok]
