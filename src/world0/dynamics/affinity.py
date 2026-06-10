"""Task affinity scoring — graded match between a task and history labels.

The legacy behavior was a boolean substring test: ``task in entry.task``
either fired the full TASK_AFFINITY_BOOST or did nothing.  That is
brittle — "debug latency spike" and "latency debugging" share every
meaningful token yet never substring-match.

This module grades the match instead:

- exact substring hit → 1.0 (every case the old code matched still
  scores maximum, so legacy behavior is preserved bit-for-bit)
- otherwise → token containment: the fraction of the task's signature
  tokens found in the history entry, maximized over all entries

Token extraction reuses ``tokenize_signature`` from the concept schema
so task matching and concept consolidation share one vocabulary (same
stopwords, same CJK handling) — no second tokenizer to drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from world0.schemas.concept import tokenize_signature

if TYPE_CHECKING:
    from world0.schemas.context import Perspective


def task_tokens(text: str) -> set[str]:
    """Signature token set for a task label (lowercased, stopword-free)."""
    return tokenize_signature(text)


def task_affinity(task: str, history: Iterable[str]) -> float:
    """Graded affinity in [0, 1] between ``task`` and history task labels.

    1.0 for an exact substring hit (legacy semantics); otherwise the best
    token-containment ratio across history entries; 0.0 when nothing
    overlaps or either side is empty.
    """
    task_lower = task.strip().lower()
    if not task_lower:
        return 0.0
    tokens = task_tokens(task_lower)
    best = 0.0
    for entry in history:
        entry_lower = entry.strip().lower()
        if not entry_lower:
            continue
        if task_lower in entry_lower:
            return 1.0
        if not tokens:
            continue
        entry_toks = task_tokens(entry_lower)
        if not entry_toks:
            continue
        ratio = len(tokens & entry_toks) / len(tokens)
        if ratio > best:
            best = ratio
    return best


def concept_in_perspective_domain(
    node, perspective: "Perspective | None"
) -> bool:
    """Is the concept's dominant domain in focus for this perspective?

    Shared by the activation engine (propagation boost) and the
    projection engine (MMR relevance boost) so the two stages agree on
    what "in focus" means.  Prefers the strongest entry in the diffused
    ``domain_profile``, falling back to the static ``domain`` label.
    """
    if perspective is None or not perspective.active_domains:
        return False
    if node.domain_profile:
        top_domain, _ = max(
            node.domain_profile.items(), key=lambda item: item[1]
        )
        if perspective.domain_match(top_domain):
            return True
    return perspective.domain_match(node.domain)
