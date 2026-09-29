"""Score a returned ``Context`` against a query's hidden gold.

Headline per kind (documented in docs/eval):

* focus   - recall of the hop-ball gold (precision, F1 and radii 1 / 3 alongside);
* chain   - F1 of the dependency closure an ideal reader derives from the shown claims;
* bridge  - recall of the bridge concept's claims in the task's domain times sense purity
            (share of the bridge concept's shown claims that belong to that domain; radius-free);
* stale   - strict: the current claim is shown and the retracted one is not
            (the lenient belief-order credit is reported separately);
* detail  - the ticket is in the shown text (episodic; out of scope for World 0).
"""

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


def stale_category(q: Query, ctx: Context) -> tuple[str, bool, bool]:
    """(category, strict correct, lenient correct)."""
    cur = ctx.claims & q.current_claims
    old = ctx.claims & q.stale_claims
    if cur and not old:
        return "current_only", True, True
    if cur and old:
        resolved = bool(ctx.beliefs) and max(ctx.beliefs.get(c, 0.0) for c in cur) > max(
            ctx.beliefs.get(c, 0.0) for c in old)
        return ("both_resolved" if resolved else "both"), False, resolved
    if old:
        return "stale_only", False, False
    return "neither", False, False


HEADLINE = {"focus": "claim_r", "chain": "answer_f1", "bridge": "bridge_sense",
            "stale": "stale_correct", "detail": "detail_hit"}


def score(q: Query, ctx: Context) -> dict:
    p, r, f = prf(ctx.claims, q.gold_claims)
    _, _, cf = prf(ctx.concepts, q.gold_concepts)
    row = {
        "claim_p": p, "claim_r": r, "claim_f1": f, "concept_f1": cf,
        "wrong_rate": len(ctx.claims & q.wrong_claims) / len(ctx.claims) if ctx.claims else 0.0,
        "shown_claims": len(ctx.claims),
        "tokens": est_tokens(ctx.text), "chars4": chars4(ctx.text),
        "answer_f1": 0.0, "stale_cat": "", "stale_correct": 0.0, "stale_lenient": 0.0,
        "stale_shown": 0.0, "detail_hit": 0.0, "ticket_precision": 0.0, "bridge_sense": 0.0,
    }
    if q.kind == "bridge":
        right, wrong = len(ctx.claims & q.gold_claims), len(ctx.claims & q.wrong_claims)
        purity = right / (right + wrong) if right + wrong else 0.0
        row["bridge_purity"] = purity
        row["bridge_sense"] = r * purity
    if q.kind == "focus":
        for radius, gold in q.gold_by_radius.items():
            _, rr, ff = prf(ctx.claims, gold)
            row[f"claim_r_r{radius}"], row[f"claim_f1_r{radius}"] = rr, ff
    if q.kind == "chain":
        _, _, row["answer_f1"] = prf(closure(q.entry[0], ctx.claims), q.gold_answer)
    if q.kind == "stale":
        cat, strict, lenient = stale_category(q, ctx)
        row["stale_cat"], row["stale_correct"], row["stale_lenient"] = cat, float(strict), float(lenient)
        row["stale_shown"] = float(bool(ctx.claims & q.stale_claims))
    if q.kind == "detail":
        hit = q.gold_ticket in ctx.tickets
        row["detail_hit"] = float(hit)
        row["ticket_precision"] = (1.0 / len(ctx.tickets)) if hit else 0.0
    row["headline"] = row[HEADLINE[q.kind]]
    return row
