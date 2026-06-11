"""Relation-typing judge — the system LLM names how two concepts relate.

Shared by two consumers:

- ``HebbianEngine`` types co-occurrence pairs the moment they cross the
  discovery threshold (ingest time)
- ``RelationRefiner`` re-judges the existing stock of ``generic_relation``
  edges during ``reflect()`` (consolidation time)

Both go through the same ``relation.typing.system`` prompt and the same
verdict validation, so a relation gets the same semantics whether it is
typed at birth or refined later.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from world0.llm.parsing import extract_json
from world0.prompts import PromptRegistry
from world0.schemas.relation import SEMANTIC_RELATION_SPECS

if TYPE_CHECKING:
    from world0.core import LLMProvider


class RelationTypingJudge:
    """Asks the system LLM which semantic relation holds between two concepts."""

    def __init__(
        self,
        llm: "LLMProvider",
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = prompt_registry or PromptRegistry()

    def judge(self, node_a, node_b) -> tuple[str, str, float] | None:
        """Return ``(semantic_relation | "none", direction, confidence)``.

        ``None`` when the response carries no usable verdict (empty or
        unknown label).  Raises on provider/parse failure so callers can
        fall back to their non-LLM behavior.
        """
        system = self._prompts.render("relation.typing.system")
        payload = {
            "concept_a": self._describe(node_a),
            "concept_b": self._describe(node_b),
        }
        raw = self._llm.complete_json(
            system, json.dumps(payload, ensure_ascii=False)
        )
        data = json.loads(extract_json(raw))
        relation = str(data.get("relation", "")).strip().lower()
        if not relation:
            return None
        if relation != "none" and relation not in SEMANTIC_RELATION_SPECS:
            # Unknown label from the model — no usable verdict.
            return None
        direction = str(data.get("direction", "a_to_b")).strip().lower()
        if direction not in ("a_to_b", "b_to_a"):
            direction = "a_to_b"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return relation, direction, min(1.0, max(0.0, confidence))

    @staticmethod
    def _describe(node) -> dict:
        return {
            "name": node.name,
            "aliases": list(node.aliases),
            "description": node.description,
            "domain": node.domain,
        }
