"""Aggregate LongRun rows into markdown tables.

The unit of replication is the *seed* (one hidden world, one event stream,
every system on it): queries within a run are correlated, so every statistic
is computed from per-seed means and comparisons between systems are paired by
seed.  Intervals are t-intervals over seeds; paired contrasts use an exact
sign-flip test with Holm correction over the comparator family.

    python -m benchmarks.longrun.report docs/eval/results > docs/eval/results/tables.md
"""

from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import sys
from collections import defaultdict

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 19: 2.09, 29: 2.05}
KINDS = ["focus", "chain", "bridge", "stale", "detail"]
IN_SCOPE = ["focus", "chain", "bridge", "stale"]
ORDER = ["none", "window", "full_32k", "full_64k", "full_context", "rag", "rag_recency", "summary_buffer", "summary_task",
         "factstore", "fact_task", "fact_task_s2", "kg_static", "kg_temporal", "state_doc",
         "world0", "world0_compact", "world0_tuned", "world0_tuned_s2",
         "world0_reflect", "world0_focus", "world0_notask", "world0_depth1", "world0_depth3"]
LABEL = {"rag_recency": "rag_recency (GA recipe)", "summary_buffer": "summary_buffer (flat digest)",
         "kg_static": "kg_static (no retraction)", "full_context": "full_context (≤128k)"}
BUDGETS = [150, 300, 600, 1200, 2400, 4800]


def load(path: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".jsonl"):
            with open(os.path.join(path, name)) as fh:
                rows += [json.loads(line) for line in fh]
    return rows


def tcrit(n: int) -> float:
    df = n - 1
    for k in sorted(T975):
        if df <= k:
            return T975[k]
    return 1.96


def ci(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    return (m, float("nan")) if n < 2 else (m, tcrit(n) * statistics.stdev(vals) / math.sqrt(n))


def fmt(m: float, h: float, digits: int = 2) -> str:
    if math.isnan(m):
        return "-"
    return f"{m:.{digits}f}" if math.isnan(h) else f"{m:.{digits}f}±{h:.{digits}f}"


def qrows(rows, **eq):
    return [r for r in rows if r["type"] == "query" and all(r.get(k) == v for k, v in eq.items())]


def bud(system: str, b: int) -> int:
    return 0 if system.startswith("full_") else b


def per_seed(rows, metric, kinds=IN_SCOPE) -> dict[int, float]:
    """Per-seed mean of ``metric``; each kind's mean first, then the mean over ``kinds``."""
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] == "query" and r["kind"] in kinds and metric in r:
            acc[r["seed"]][r["kind"]].append(r[metric])
    return {s: statistics.mean(statistics.mean(v) for v in ks.values()) for s, ks in acc.items() if ks}


def sign_flip_p(diffs: list[float]) -> float:
    n = len(diffs)
    obs = abs(sum(diffs))
    if n == 0 or obs == 0:
        return 1.0
    hits = sum(abs(sum(d * s for d, s in zip(diffs, signs))) >= obs - 1e-12
               for signs in itertools.product((1, -1), repeat=n))
    return hits / 2 ** n


def paired(rows_a, rows_b, metric="headline", kinds=IN_SCOPE):
    pa, pb = per_seed(rows_a, metric, kinds), per_seed(rows_b, metric, kinds)
    seeds = sorted(set(pa) & set(pb))
    d = [pa[s] - pb[s] for s in seeds]
    m, h = ci(d)
    return m, h, sign_flip_p(d), sum(x > 0 for x in d), len(d)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in body])


def systems_in(rows, only=None):
    present = {r["system"] for r in rows}
    return [s for s in ORDER if s in present and (only is None or s in only)]


def name(s):
    return LABEL.get(s, s)


def tokens_used(rows, system, b, **eq):
    v = [r["tokens"] for r in qrows(rows, system=system, budget=bud(system, b), **eq)]
    return statistics.mean(v) if v else float("nan")


def utility_cell(rows, system, b, **eq):
    sel = qrows(rows, system=system, budget=bud(system, b), **eq)
    return fmt(*ci(list(per_seed(sel, "headline").values())))


def main(path: str) -> None:
    rows = load(path)
    out: list[str] = []
    main_rows = [r for r in rows if r.get("study") == "main"]

    if main_rows:
        out.append("## Main study: H=1000, 10 seeds\n")
        for b in (300, 600, 1200):
            body = []
            for s in systems_in(main_rows, set(ORDER[:17])):
                cells = [name(s)]
                for k in KINDS:
                    sel = qrows(main_rows, system=s, budget=bud(s, b), kind=k)
                    cells.append(fmt(*ci(list(per_seed(sel, "headline", [k]).values()))))
                cells.append(utility_cell(main_rows, s, b))
                cells.append(f"{tokens_used(main_rows, s, b):.0f}")
                body.append(cells)
            out.append(f"### budget {b} tokens\n")
            out.append(table(["system"] + KINDS + ["utility (4 in-scope kinds)", "tokens used"], body))
            out.append("")
        out.append("*Headline per kind: focus = recall of the 2-hop gold, chain = closure F1, bridge = recall × sense purity, stale = strict (current claim shown, retracted claim not), detail = ticket shown (episodic; outside World 0's scope, excluded from utility). Mean ± 95 % t-interval over seeds.*\n")

        out.append("## Utility vs token budget (H=1000)\n")
        body = []
        for s in systems_in(main_rows, set(ORDER[:17])):
            body.append([name(s)] + [f"{utility_cell(main_rows, s, b)} ({tokens_used(main_rows, s, b):.0f})" for b in BUDGETS])
        out.append(table(["system"] + [f"{b}" for b in BUDGETS], body))
        out.append("\n*cell = utility (mean tokens actually returned). Structured systems return what is relevant, not what the budget allows.*\n")

        out.append("## Focus queries: precision / recall / F1 and gold radius (H=1000, budget 1200)\n")
        body = []
        for s in systems_in(main_rows, set(ORDER[:17])):
            sel = qrows(main_rows, system=s, budget=bud(s, 1200), kind="focus")
            cells = [name(s)]
            for m in ("claim_p", "claim_r", "claim_f1", "wrong_rate", "claim_r_r1", "claim_r_r2", "claim_r_r3"):
                cells.append(fmt(*ci(list(per_seed(sel, m, ["focus"]).values()))))
            body.append(cells)
        out.append(table(["system", "precision", "recall", "F1", "other-domain rate", "recall r=1", "r=2", "r=3"], body))

        out.append("\n## Revisions (stale queries): what is shown when a claim was retracted (H=1000, budget 1200)\n")
        cats = ["current_only", "both_resolved", "both", "stale_only", "neither"]
        body = []
        for s in systems_in(main_rows, set(ORDER[:17])):
            rs = qrows(main_rows, system=s, budget=bud(s, 1200), kind="stale")
            n = len(rs) or 1
            body.append([name(s)] + [f"{100 * sum(r['stale_cat'] == c for r in rs) / n:.0f}%" for c in cats]
                        + [fmt(*ci(list(per_seed(rs, "stale_lenient", ["stale"]).values()))), str(len(rs))])
        out.append(table(["system"] + cats + ["lenient credit", "n queries"], body))
        out.append("\n*strict credit = current_only. both_resolved = both claims shown, the current one with the higher displayed belief (only systems that show beliefs can earn it; it rests on small belief gaps).*\n")

        for b in (600, 1200):
            out.append(f"## Paired contrast: world0_tuned vs each system (H=1000, budget {b}, utility)\n")
            a_rows = qrows(main_rows, system="world0_tuned", budget=b)
            comps = [s for s in systems_in(main_rows) if s != "world0_tuned"]
            res = [paired(a_rows, qrows(main_rows, system=s, budget=bud(s, b))) for s in comps]
            adj = holm([r[2] for r in res])
            body = [[name(s), f"{d:+.2f}", f"±{h:.2f}", f"{p:.4f}", f"{pa:.4f}", f"{w}/{n}"]
                    for s, (d, h, p, w, n), pa in zip(comps, res, adj)]
            out.append(table(["comparator", "Δ utility (tuned − comparator)", "95 % t", "sign-flip p", "Holm p", "seeds won"], body))
            out.append("")

    sc = [r for r in rows if r.get("study") == "scale"] + [r for r in main_rows]
    if sc:
        out.append("## Scale: utility by horizon (budget 1200)\n")
        hs = sorted({r["horizon"] for r in sc if r["type"] == "query"})
        body = []
        for s in systems_in(sc, set(ORDER[:17])):
            body.append([name(s)] + [utility_cell([r for r in sc if r["horizon"] == h], s, 1200) for h in hs])
        out.append(table(["system"] + [f"H={h}" for h in hs], body))
        out.append("\n### Stale (strict) by horizon, budget 1200\n")
        body = []
        for s in systems_in(sc, set(ORDER[:17])):
            cells = [name(s)]
            for h in hs:
                sel = qrows([r for r in sc if r["horizon"] == h], system=s, budget=bud(s, 1200), kind="stale")
                cells.append(fmt(*ci(list(per_seed(sel, "stale_correct", ["stale"]).values()))))
            body.append(cells)
        out.append(table(["system"] + [f"H={h}" for h in hs], body))
        out.append("\n### Resources by horizon (state owners)\n")
        res = [r for r in sc if r["type"] == "resource"]
        body = []
        for s in systems_in(res, set(ORDER[:17])):
            cells = [name(s)]
            for h in hs:
                rr = [r for r in res if r["system"] == s and r["horizon"] == h]
                cells.append(f"{statistics.mean(r['observe_ms_per_event'] for r in rr):.2f} ms, {statistics.mean(r.get('bytes', 0) for r in rr) / 1024:.0f} KB" if rr and "bytes" in rr[0] else "-")
            body.append(cells)
        out.append(table(["system (observe ms/event, stored KB)"] + [f"H={h}" for h in hs], body))
        r0 = [r for r in res if r["system"] == "none"]
        if r0:
            out.append("\nStream and state size (tokens): " + "; ".join(
                f"H={h}: stream {statistics.mean(r['stream_tokens'] for r in r0 if r['horizon'] == h):.0f}, "
                f"known state {statistics.mean(r['state_tokens'] for r in r0 if r['horizon'] == h):.0f}" for h in hs))

    bw = [r for r in rows if r.get("study") == "bigworld"]
    if bw:
        out.append("\n## Big growing world (40 domains × 40 concepts, H=6000): utility by budget\n")
        body = []
        for s in systems_in(bw, set(ORDER[:17])):
            body.append([name(s)] + [f"{utility_cell(bw, s, b)} ({tokens_used(bw, s, b):.0f})" for b in BUDGETS])
        out.append(table(["system"] + [f"{b}" for b in BUDGETS], body))
        r0 = [r for r in bw if r["type"] == "resource" and r["system"] == "none"]
        if r0:
            out.append(f"\nKnown state {statistics.mean(r['state_tokens'] for r in r0):.0f} tokens; stream {statistics.mean(r['stream_tokens'] for r in r0):.0f} tokens.")
        out.append("\n### per kind, budget 1200\n")
        body = []
        for s in systems_in(bw, set(ORDER[:17])):
            cells = [name(s)]
            for k in KINDS:
                sel = qrows(bw, system=s, budget=bud(s, 1200), kind=k)
                cells.append(fmt(*ci(list(per_seed(sel, "headline", [k]).values()))))
            body.append(cells)
        out.append(table(["system"] + KINDS, body))

    for study, param, label in (("extraction", "extract_p", "extractor error level p"),
                                ("taskmode", "task_mode", "task label at query time"),
                                ("chatter", "noise_rate", "chatter rate"),
                                ("verbosity", "verbosity", "filler tokens per event")):
        st = [r for r in rows if r.get("study") == study]
        if not st:
            continue
        st = st + [dict(r, **{param: {"extract_p": 0.0, "task_mode": "exact", "noise_rate": 0.25, "verbosity": 30}[param]})
                   for r in main_rows]
        vals = sorted({r[param] for r in st if r["type"] == "query"}, key=str)
        out.append(f"\n## Sensitivity: {label} (H=1000, budget 1200, utility)\n")
        body = []
        for s in systems_in(st, set(ORDER[:17])):
            body.append([name(s)] + [utility_cell([r for r in st if r[param] == v], s, 1200) for v in vals])
        out.append(table(["system"] + [f"{param}={v}" for v in vals], body))

    wl = [r for r in rows if r.get("study") == "window_limit"]
    if wl:
        out.append("\n## When the history is longer than the window (H=6000, ideal reader over what fits)\n")
        body = []
        for s in systems_in(wl):
            cells = [name(s)]
            for k in KINDS:
                sel = qrows(wl, system=s, budget=bud(s, 1200), kind=k)
                cells.append(fmt(*ci(list(per_seed(sel, "headline", [k]).values()))))
            cells.append(utility_cell(wl, s, 1200))
            cells.append(f"{tokens_used(wl, s, 1200):.0f}")
            body.append(cells)
        out.append(table(["system"] + KINDS + ["utility", "tokens used"], body))
        r0 = [r for r in wl if r["type"] == "resource"]
        if r0:
            out.append(f"\nFull stream: {statistics.mean(r['stream_tokens'] for r in r0):.0f} tokens.")

    ab = [r for r in rows if r.get("study") == "ablation"]
    if ab:
        out.append("\n## World 0 variants (H=1000, budget 1200)\n")
        body = []
        for s in systems_in(ab):
            if not s.startswith("world0"):
                continue
            cells = [s, utility_cell(ab, s, 1200)]
            for k in KINDS:
                sel = qrows(ab, system=s, budget=1200, kind=k)
                cells.append(fmt(*ci(list(per_seed(sel, "headline", [k]).values()))))
            body.append(cells)
        out.append(table(["variant", "utility"] + KINDS, body))

    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/eval/results")
