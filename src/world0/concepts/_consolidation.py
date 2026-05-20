"""Signature-based concept consolidation.

Two flavours of the same primitive:

- ``find_best_match`` — used by ``ConceptManager.get_or_create`` to
  auto-attach a freshly submitted name as an alias of an existing
  concept whose signature overlaps strongly enough.  Hard threshold
  (``SIGNATURE_CONSOLIDATION_THRESHOLD``).
- ``find_similar`` — soft variant returning ranked candidates above a
  caller-chosen ``min_similarity``.  Used by agents that want to
  surface merge candidates rather than auto-apply them.

Both share the same Jaccard-with-domain-penalty scoring so the merge
suggestion you see in the agent's UI matches the consolidation
decision the manager would take silently.
"""

from __future__ import annotations

from typing import Callable

from world0.concepts._indexes import TokenIndex
from world0.schemas.concept import ConceptNode

# Minimum Jaccard signature similarity at which a newly submitted name
# is auto-attached as an alias to an existing concept (instead of a new
# node being created).  Pitched high enough to avoid merging sibling
# concepts such as FastAPI and Starlette but low enough to catch
# `PostgreSQL` / `postgres database` / `pg` when a description is given.
SIGNATURE_CONSOLIDATION_THRESHOLD: float = 0.6

# Soft penalty applied when the probe domain disagrees with a candidate's
# domain — keeps cross-domain accidental tokens from forcing a merge.
DOMAIN_MISMATCH_PENALTY: float = 0.3


ConceptGetter = Callable[[str], "ConceptNode | None"]


class SignatureMatcher:
    """Scores candidate concepts against a probe signature."""

    def __init__(self, tokens: TokenIndex, get_concept: ConceptGetter) -> None:
        self._tokens = tokens
        self._get = get_concept

    def find_best_match(
        self, probe_tokens: set[str], *, domain: str = ""
    ) -> ConceptNode | None:
        """Highest-scoring candidate above ``SIGNATURE_CONSOLIDATION_THRESHOLD``."""
        if not probe_tokens:
            return None
        best: ConceptNode | None = None
        best_sim = 0.0
        for node, sim in self._iter_scored(probe_tokens, domain=domain):
            if sim > best_sim:
                best_sim = sim
                best = node
        if best_sim >= SIGNATURE_CONSOLIDATION_THRESHOLD:
            return best
        return None

    def find_similar(
        self,
        probe_tokens: set[str],
        *,
        domain: str = "",
        min_similarity: float = 0.3,
        limit: int = 5,
    ) -> list[tuple[ConceptNode, float]]:
        """Ranked candidates, filtered by ``min_similarity`` and capped at ``limit``."""
        if not probe_tokens:
            return []
        scored = [
            (node, sim)
            for node, sim in self._iter_scored(probe_tokens, domain=domain)
            if sim >= min_similarity
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _iter_scored(
        self, probe_tokens: set[str], *, domain: str
    ):
        """Yield (node, similarity) for every candidate the token index returns."""
        domain_lower = domain.strip().lower()
        for cid in self._tokens.candidates(probe_tokens):
            node = self._get(cid)
            if not node:
                continue
            node_tokens = node.signature_tokens()
            if not node_tokens:
                continue
            sim = len(probe_tokens & node_tokens) / len(
                probe_tokens | node_tokens
            )
            node_domain = node.domain.strip().lower()
            if (
                domain_lower
                and node_domain
                and domain_lower != node_domain
            ):
                sim *= DOMAIN_MISMATCH_PENALTY
            yield node, sim
