"""Retrieval-quality metrics for projections and activation.

Structural, deterministic, dependency-free measures of how good a
projection is *as a retrieval result* — given a known gold set, how
much of it was surfaced, how cleanly, in what order.  These power the
sweep/eval harnesses in ``scripts/`` and back the benchmark tests.

These are not cognitive-quality judgments (that needs an LLM judge);
they are the objective floor any projection must clear.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. 0 = disjoint, 1 = identical."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def precision_recall(
    retrieved: set, relevant: set
) -> tuple[float, float]:
    """Return (precision, recall) against a ground-truth set."""
    if not retrieved:
        return (0.0, 0.0)
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant) if relevant else 0.0
    return precision, recall


def ndcg_at_k(
    ranked: Sequence[str], relevant: set, k: int | None = None
) -> float:
    """Normalized Discounted Cumulative Gain at rank ``k``.

    Binary relevance: items in ``relevant`` score 1, others 0.  1.0 means
    every relevant item ranked above every irrelevant one.
    """
    if not relevant:
        return 0.0
    if k is None:
        k = len(ranked)
    ranked = list(ranked)[:k]

    dcg = 0.0
    for i, name in enumerate(ranked):
        gain = 1.0 if name in relevant else 0.0
        dcg += gain / math.log2(i + 2)  # i + 2 because log2(1) = 0

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def selection_diversity(neighbor_sets: Iterable[set]) -> float:
    """Average pairwise Jaccard *distance* among neighbor sets.

    Higher = the selected concepts cover structurally distinct regions
    (the property MMR exists to protect).  Returns 1.0 for fewer than two
    sets (a singleton is trivially diverse).
    """
    sets = list(neighbor_sets)
    if len(sets) < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            total += 1.0 - jaccard(sets[i], sets[j])
            pairs += 1
    return total / pairs if pairs else 1.0
