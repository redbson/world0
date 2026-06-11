"""Relation refinement — reflect-time governance of generic relations.

``generic_relation`` is technical debt, not a stable asset (Rule 3:
typed first, generic only as temporary fallback).  Ingest-time typing
(``HebbianEngine`` + ``RelationTypingJudge``) keeps *new* discovered
relations typed; this engine works through the existing **stock**: each
``reflect()`` cycle re-judges a budgeted batch of generic edges with
the same system LLM.

Outcomes per edge:

- a canonical semantic relation → re-typed in place (operational
  weight/confidence/history preserved; direction flipped when the judge
  says so)
- ``"none"`` / no usable verdict → marked ``co_attention_only``: the
  pair demonstrably co-occurs but carries no nameable structure, so the
  edge keeps propagating yet renders with reduced projection salience —
  frequent co-occurrence must not masquerade as semantic structure.

The budget bounds LLM cost per cycle; the strongest generic edges
(highest reinforcement count) are judged first since they pollute
projections the most (``generic_pressure``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.dynamics.relation_typing import RelationTypingJudge
from world0.prompts import PromptRegistry

if TYPE_CHECKING:
    from world0.core import ConceptStoreReader, LLMProvider, RelationStore

# Generic edges judged per reflect() cycle.
REFINEMENT_BUDGET: int = 10


class RelationRefiner:
    """Re-types the stock of generic relations during reflect cycles."""

    def __init__(
        self,
        concepts: "ConceptStoreReader",
        relations: "RelationStore",
        llm: "LLMProvider",
        prompt_registry: PromptRegistry | None = None,
        *,
        budget: int = REFINEMENT_BUDGET,
    ) -> None:
        self._concepts = concepts
        self._relations = relations
        self._judge = RelationTypingJudge(llm, prompt_registry)
        self._budget = budget

    def refine(self) -> tuple[list[str], list[str]]:
        """Judge up to ``budget`` generic edges.

        Returns ``(retyped_labels, co_attention_labels)`` where each
        label is a human-readable ``source → relation → target`` /
        ``source ↔ target`` string for the ReflectResult.
        """
        candidates = [
            edge
            for edge in self._relations.all()
            if edge.semantic_relation == "generic_relation"
            and edge.refinement_state != "co_attention_only"
        ]
        # Strongest first: heavily reinforced generic edges surface in
        # projections most, so refining them buys the most pressure relief.
        candidates.sort(key=lambda e: e.reinforcement_count, reverse=True)

        retyped: list[str] = []
        co_attention: list[str] = []
        for edge in candidates[: self._budget]:
            source = self._concepts.get(edge.source_id)
            target = self._concepts.get(edge.target_id)
            if source is None or target is None:
                continue
            try:
                verdict = self._judge.judge(source, target)
            except Exception:
                # Provider hiccup — leave the edge untouched; it will be
                # re-considered next cycle.
                continue
            if verdict is None or verdict[0] in ("none", "generic_relation"):
                # Co-occurs, but no nameable structure (or the judge had
                # no usable verdict) — demote projection salience instead
                # of letting reinforcement masquerade as semantics.
                self._relations.set_refinement_state(
                    edge.id, "co_attention_only"
                )
                co_attention.append(f"{source.name} ↔ {target.name}")
                continue
            relation, direction, _confidence = verdict
            self._relations.retype_semantic(
                edge.id,
                relation,
                flip_direction=(direction == "b_to_a"),
            )
            if direction == "b_to_a":
                source, target = target, source
            retyped.append(f"{source.name} → {relation} → {target.name}")
        return retyped, co_attention
