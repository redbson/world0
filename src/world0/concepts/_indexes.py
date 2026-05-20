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
    """Normalized name/alias → concept_id."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def clear(self) -> None:
        self._map.clear()

    def get(self, name: str) -> str | None:
        return self._map.get(name.strip().lower())

    def add(self, name: str, concept_id: str) -> None:
        """Set the mapping unconditionally (overwriting any prior entry)."""
        normalized = name.strip().lower()
        if normalized:
            self._map[normalized] = concept_id

    def add_unique(self, name: str, concept_id: str) -> bool:
        """Set the mapping only if it does not already point elsewhere.

        Returns True if the entry now points at ``concept_id`` (either
        freshly added or already so), False if another concept owns it.
        """
        normalized = name.strip().lower()
        if not normalized:
            return False
        existing = self._map.get(normalized)
        if existing and existing != concept_id:
            return False
        self._map[normalized] = concept_id
        return True

    def remove_if_owned(self, name: str, concept_id: str) -> bool:
        """Remove the entry only if it currently points at ``concept_id``."""
        normalized = name.strip().lower()
        if self._map.get(normalized) == concept_id:
            self._map.pop(normalized, None)
            return True
        return False

    def index_node(self, node) -> None:
        """Index every alias variant of ``node`` (overwrites prior entries)."""
        for n in node.all_names():
            self._map[n] = node.id


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
