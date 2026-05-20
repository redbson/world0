"""Identity operations exposed at the World facade level.

These are the user-facing convenience wrappers around the lower-level
``ConceptStore`` ops:

- ``merge(keeper, absorbed)`` — accepts names/ids, resolves, delegates
- ``split(source, new_name, ...)`` — same pattern
- ``weaken(concept, ...)`` — disconfirmation evidence
- ``find_similar(text, ...)`` — surface candidate matches

All callers are expected to flush the underlying stores after a
successful operation; this class does not own persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world0.core import ConceptStore, RelationStore


class IdentityOps:
    def __init__(
        self, *, concepts: ConceptStore, relations: RelationStore
    ) -> None:
        self._concepts = concepts
        self._relations = relations

    def merge(self, keeper: str, absorbed: str) -> bool:
        keeper_node = self._concepts.resolve(keeper)
        absorbed_node = self._concepts.resolve(absorbed)
        if not keeper_node or not absorbed_node:
            return False
        result = self._concepts.merge(
            keeper_node.id, absorbed_node.id, relations=self._relations
        )
        return result is not None

    def split(
        self,
        source: str,
        new_name: str,
        *,
        aliases_to_move: list[str] | None = None,
        description: str = "",
    ) -> str | None:
        source_node = self._concepts.resolve(source)
        if not source_node:
            return None
        new_node = self._concepts.split(
            source_node.id,
            new_name,
            aliases_to_move=aliases_to_move,
            description=description,
        )
        return new_node.id if new_node else None

    def weaken(
        self, concept: str, *, source: str = "", task: str = ""
    ) -> bool:
        node = self._concepts.resolve(concept)
        if not node:
            return False
        self._concepts.weaken(node.id, source=source, task=task)
        return True

    def find_similar(
        self,
        text: str,
        *,
        domain: str = "",
        min_similarity: float = 0.3,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        matches = self._concepts.find_similar(
            text,
            domain=domain,
            min_similarity=min_similarity,
            limit=limit,
        )
        return [(node.name, sim) for node, sim in matches]
