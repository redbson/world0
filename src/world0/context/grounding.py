"""Lexical task grounding: which concepts does a task string name?

``name_coverage`` measures how much of a concept's name (or one of its
aliases) the task mentions, at the same word level the concept
signatures use; ``ground_task`` turns that into a per-candidate task
affinity for a projection:

- a concept whose name is fully covered by the task is an **anchor**
  (affinity 1.0);
- a partially covered name counts in proportion to its coverage once
  at least ``GROUNDING_MIN_COVERAGE`` of it is mentioned;
- a direct neighbour of an anchor inherits ``GROUNDING_NEIGHBOR_SHARE``
  — a task about ``kubernetes`` is also about what kubernetes is
  connected to, but less so than about kubernetes itself.

Names without signature tokens (CJK text, very short labels) fall back
to whole-string containment in the task, so ``"部署"`` grounds the task
``"模型部署流程"``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from world0.schemas.concept import normalize_task_label, tokenize_signature

if TYPE_CHECKING:
    from world0.core import ConceptStore

# Minimum fraction of a concept name the task must mention before the
# concept counts as (partially) named by it.
GROUNDING_MIN_COVERAGE: float = 0.5

# Task affinity inherited by a direct neighbour of an anchor concept.
GROUNDING_NEIGHBOR_SHARE: float = 0.5


def name_coverage(task: str, names: Iterable[str]) -> float:
    """Largest fraction of any of ``names`` that ``task`` mentions."""
    task_norm = normalize_task_label(task)
    if not task_norm:
        return 0.0
    task_tokens = tokenize_signature(task_norm)
    best = 0.0
    for name in names:
        name_norm = normalize_task_label(name)
        if not name_norm:
            continue
        tokens = tokenize_signature(name_norm)
        if tokens:
            coverage = len(tokens & task_tokens) / len(tokens)
        else:
            coverage = 1.0 if len(name_norm) >= 2 and name_norm in task_norm else 0.0
        if coverage > best:
            best = coverage
            if best >= 1.0:
                break
    return best


def ground_task(
    task: str,
    candidate_ids: Iterable[str],
    concepts: "ConceptStore",
    neighbors: Mapping[str, set[str]],
) -> dict[str, float]:
    """Grounded task affinity in ``[0, 1]`` for each candidate concept.

    ``neighbors`` maps a candidate id to the ids it is directly related
    to (the projection already builds this map for its redundancy term).
    Only candidates with a non-zero affinity appear in the result.
    """
    if not normalize_task_label(task):
        return {}
    ids = list(candidate_ids)
    direct: dict[str, float] = {}
    for cid in ids:
        node = concepts.get(cid)
        if node is None:
            continue
        coverage = name_coverage(task, node.all_names())
        if coverage >= GROUNDING_MIN_COVERAGE:
            direct[cid] = coverage
    anchors = {cid for cid, cov in direct.items() if cov >= 1.0}
    grounded = dict(direct)
    if anchors:
        for cid in ids:
            if grounded.get(cid, 0.0) >= GROUNDING_NEIGHBOR_SHARE:
                continue
            if neighbors.get(cid, set()) & anchors:
                grounded[cid] = GROUNDING_NEIGHBOR_SHARE
    return grounded
