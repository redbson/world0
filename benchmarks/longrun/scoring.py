"""Score a returned ``Context`` against a query's hidden gold."""

from __future__ import annotations

from benchmarks.longrun.systems import Context
from benchmarks.longrun.tokens import chars4, est_tokens
from benchmarks.longrun.worldgen import Claim, Query


def prf(shown: set, gold: set) -> tuple[float, float, float]:
    hit = len(shown & gold)
    p = hit / len(shown) if shown else 0.0
    r = hit / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def closure(entry: str, claims: set[Claim]) -> set[str]:
    deps: dict[str, list[str]] = {}
    for c in claims:
        if c.rel == "depends_on":
            deps.setdefault(c.src, []).append(c.tgt)
    seen, frontier = set(), [entry]
    while frontier:
        cur = frontier.pop()
        for t in deps.get(cur, ()):
            if t not in seen:
                seen.add(t)
                frontier.append(t)
    return seen


def stale_category(q: Query, ctx: Context) -> tuple[str, bool]:
    cur = ctx.claims & q.current_claims
    old = ctx.claims & q.stale_claims
    if cur and not old:
        return "current_only", True
    if cur and old:
        # Both are shown: the reader can only tell them apart if the context
        # carries beliefs and the current claim's belief is the higher one.
        resolved = bool(ctx.beliefs) and max(ctx.beliefs.get(c, 0.0) for c in cur) > max(
            ctx.beliefs.get(c, 0.0) for c in old)
        return "both_resolved" if resolved else "both", resolved
    if old:
        return "stale_only", False
    return "neither", False


HEADLINE = {"focus": "claim_f1", "chain": "answer_f1", "bridge": "claim_f1",
            "stale": "stale_correct", "detail": "detail_hit"}


def score(q: Query, ctx: Context) -> dict:
    p, r, f = prf(ctx.claims, q.gold_claims)
    _, _, cf = prf(ctx.concepts, q.gold_concepts)
    row = {
        "claim_p": p, "claim_r": r, "claim_f1": f, "concept_f1": cf,
        "wrong_rate": len(ctx.claims & q.wrong_claims) / len(ctx.claims) if ctx.claims else 0.0,
        "shown_claims": len(ctx.claims),
        "tokens": est_tokens(ctx.text), "chars4": chars4(ctx.text),
        "answer_f1": 0.0, "stale_cat": "", "stale_correct": 0.0, "detail_hit": 0.0,
    }
    if q.kind == "chain":
        _, _, row["answer_f1"] = prf(closure(q.entry[0], ctx.claims), q.gold_answer)
    if q.kind == "stale":
        cat, ok = stale_category(q, ctx)
        row["stale_cat"], row["stale_correct"] = cat, float(ok)
        row["stale_shown"] = float(bool(ctx.claims & q.stale_claims))
    if q.kind == "detail":
        row["detail_hit"] = float(q.gold_ticket in ctx.tickets)
    row["headline"] = row[HEADLINE[q.kind]]
    return row
