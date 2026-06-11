"""Hebbian learning — "concepts that fire together, wire together."

When multiple concepts co-occur in a single observation, their connections
are automatically strengthened (or created if they don't exist).
This is the primary mechanism by which relations are *discovered*.

Optimization: a co-occurrence threshold prevents O(n²) relation explosion.
Only concept pairs that have co-occurred >= COOCCURRENCE_THRESHOLD times
actually produce a RelationEdge. Below that, only a counter is incremented.

Relation typing: bare co-occurrence says *that* two concepts connect, not
*how*.  When the system LLM is configured, every pair crossing the
threshold is typed by the ``relation.typing.system`` prompt — the model
picks the most specific semantic relation from the canonical inventory
(with direction and confidence), or rejects coincidental pairs outright.
This keeps Rule 3 ("relations must be typed; generic only as fallback")
honest for discovered relations, not just declared ones.  Without an LLM,
or when the judge fails, the engine falls back to the original untyped
PARALLEL edge.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

from world0.dynamics.relation_typing import RelationTypingJudge
from world0.prompts import PromptRegistry
from world0.schemas.relation import RelationType

if TYPE_CHECKING:
    from world0.core import ConceptStoreReader, LLMProvider, RelationStore
    from world0.schemas.relation import RelationEdge

# Minimum co-occurrence count before a Hebbian relation is created.
# Prevents noise relations from a single shared observation.
COOCCURRENCE_THRESHOLD: int = 2

# Maximum concept pairs to process per learn() call.
# When an observation contains many concepts, only the first MAX_PAIRS
# pairs (sorted by ID for determinism) are considered.
MAX_PAIRS: int = 30


class HebbianEngine:
    """Implements Hebbian co-activation learning for relation discovery.

    Implements the ``HebbianLearner`` Protocol from ``world0.core``.
    With an LLM configured, discovered relations are semantically typed
    instead of landing as untyped PARALLEL edges.
    """

    def __init__(
        self,
        relations: "RelationStore",
        concepts: "ConceptStoreReader | None" = None,
        *,
        llm: "LLMProvider | None" = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._relations = relations
        self._concepts = concepts
        self._judge_engine = (
            RelationTypingJudge(llm, prompt_registry) if llm else None
        )
        # Tracks co-occurrence counts for pairs that don't yet have a relation.
        # Key: frozenset({id_a, id_b}), Value: count
        self._cooccurrence: dict[frozenset[str], int] = defaultdict(int)

    def learn(
        self,
        concept_ids: list[str],
        *,
        provenance: str = "",
    ) -> list[str]:
        """Process co-occurring concepts: strengthen or create relations.

        For every pair of concepts in the list:
          - If a relation exists → reinforce it
          - If no relation exists and co-occurrence < threshold → increment counter
          - If no relation exists and co-occurrence >= threshold → create a
            relation, typed by the system LLM when one is configured
            (untyped PARALLEL otherwise, or when the judge rejects/fails)

        Returns list of relation ids that were created (not reinforced).
        """
        new_relation_ids: list[str] = []

        pairs = list(combinations(concept_ids, 2))
        if len(pairs) > MAX_PAIRS:
            pairs = sorted(pairs)[:MAX_PAIRS]

        for id_a, id_b in pairs:
            existing = self._relations.find_any_between(id_a, id_b)
            if existing:
                for rel in existing:
                    self._relations.reinforce(rel.id, provenance=provenance)
            else:
                key = frozenset((id_a, id_b))
                self._cooccurrence[key] += 1
                if self._cooccurrence[key] >= COOCCURRENCE_THRESHOLD:
                    edge = self._discover_typed(id_a, id_b, provenance)
                    if edge is not None:
                        new_relation_ids.append(edge.id)
                    # Clear counter once the pair has been adjudicated —
                    # rejected pairs re-accumulate and get re-judged if
                    # they keep co-occurring.
                    del self._cooccurrence[key]

        return new_relation_ids

    # ── typed discovery ──────────────────────────────────────────────

    def _discover_typed(
        self, id_a: str, id_b: str, provenance: str
    ) -> "RelationEdge | None":
        """Create the relation for a threshold-crossing pair.

        LLM path: judge the semantic relation (or reject).  Fallback
        path: the original untyped PARALLEL edge.
        """
        if self._judge_engine is not None and self._concepts is not None:
            node_a = self._concepts.get(id_a)
            node_b = self._concepts.get(id_b)
            if node_a is not None and node_b is not None:
                try:
                    verdict = self._judge_engine.judge(node_a, node_b)
                except Exception:
                    verdict = None  # provider hiccup → untyped fallback
                if verdict is not None:
                    relation, direction, _confidence = verdict
                    if relation == "none":
                        return None
                    source_id, target_id = (
                        (id_a, id_b)
                        if direction != "b_to_a"
                        else (id_b, id_a)
                    )
                    edge, is_new = self._relations.discover(
                        source_id,
                        target_id,
                        semantic_relation=relation,
                        provenance=provenance or "hebbian:llm-typed",
                        is_explicit=False,
                    )
                    return edge if is_new else None
        edge, is_new = self._relations.discover(
            id_a,
            id_b,
            RelationType.PARALLEL,
            provenance=provenance,
            is_explicit=False,
        )
        return edge if is_new else None

    @property
    def pending_pairs(self) -> int:
        """Number of concept pairs awaiting threshold before relation creation."""
        return len(self._cooccurrence)
