"""Aggregate LongRun rows into markdown tables.

The unit of replication is the *seed* (one hidden world, one event stream, all
systems on it): queries within a run are correlated, so every statistic is
computed from per-seed means, and comparisons between systems are paired by
seed.  Intervals are t-intervals over seeds.

    python -m benchmarks.longrun.report docs/eval/results > docs/eval/results/tables.md
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}
KINDS = ["focus", "chain", "bridge", "stale", "detail"]
ORDER = ["none", "window", "full_context", "rag", "rag_recency", "summary_buffer",
         "factstore", "kg_static", "kg_temporal", "world0", "world0_compact",
         "world0_reflect", "world0_focus", "world0_notask", "world0_depth1"]


def load(path: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".jsonl"):
            with open(os.path.join(path, name)) as fh:
                rows += [json.loads(line) for line in fh]
    return rows


def ci(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = statistics.mean(vals)
    if n < 2:
        return m, float("nan")
    return m, T975.get(n - 1, 1.96) * statistics.stdev(vals) / math.sqrt(n)


def fmt(m: float, h: float, digits: int = 2) -> str:
    if math.isnan(m):
        return "-"
    return f"{m:.{digits}f}" if math.isnan(h) else f"{m:.{digits}f}±{h:.{digits}f}"


def per_seed(rows, metric, where, by_kind_equal=True) -> dict[int, float]:
    """Per-seed mean of ``metric``; kinds weighted equally (each kind's mean, then mean over kinds)."""
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["type"] == "query" and where(r) and metric in r:
            acc[r["seed"]][r["kind"] if by_kind_equal else "all"].append(r[metric])
    return {s: statistics.mean(statistics.mean(v) for v in kinds.values()) for s, kinds in acc.items() if kinds}


def paired(rows, metric, a, b, where, boot=2000) -> tuple[float, float, float, int]:
    """Mean paired (a - b) over seeds, bootstrap 95 % interval, and seeds where a > b."""
    pa = per_seed(rows, metric, lambda r: where(r) and r["system"] == a)
    pb = per_seed(rows, metric, lambda r: where(r) and r["system"] == b)
    seeds = sorted(set(pa) & set(pb))
    diffs = [pa[s] - pb[s] for s in seeds]
    if not diffs:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(boot))
    return statistics.mean(diffs), means[int(0.025 * boot)], means[int(0.975 * boot)], sum(d > 0 for d in diffs)


def table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def systems_in(rows) -> list[str]:
    present = {r["system"] for r in rows}
    return [s for s in ORDER if s in present]


def study_rows(rows, study):
    return [r for r in rows if r.get("study") == study]


def main(path: str) -> None:
    rows = load(path)
    out: list[str] = []

    # 1. scale
    sc = study_rows(rows, "scale")
    if sc:
        out.append("## Scale: overall utility by horizon (budget 600 tokens)\n")
        hs = sorted({r["horizon"] for r in sc})
        body = []
        for s in systems_in(sc):
            cells = [s]
            for h in hs:
                b = 0 if s == "full_context" else 600
                m, half = ci(list(per_seed(sc, "headline", lambda r, s=s, h=h, b=b: r["system"] == s and r["horizon"] == h and r["budget"] == b).values()))
                cells.append(fmt(m, half))
            body.append(cells)
        out.append(table(["system"] + [f"H={h}" for h in hs], body))
        out.append("\n*utility = mean over the five query kinds of the kind's headline metric (focus/bridge: claim F1, chain: answer F1, stale: correct-current rate, detail: ticket hit rate); mean ± 95 % t-interval over seeds; full_context is unbounded.*\n")

        out.append("## Scale: per-kind headline at H=1000, by budget\n")
        for b in (150, 300, 600, 1200):
            body = []
            for s in systems_in(sc):
                bb = 0 if s == "full_context" else b
                cells = [s]
                for k in KINDS:
                    vals = list(per_seed([r for r in sc if r["kind"] == k], "headline",
                                         lambda r, s=s, bb=bb: r["system"] == s and r["horizon"] == 1000 and r["budget"] == bb,
                                         by_kind_equal=False).values())
                    cells.append(fmt(*ci(vals)))
                body.append(cells)
            out.append(f"### budget {b}\n")
            out.append(table(["system"] + KINDS, body))
            out.append("")

        out.append("## Focus queries: precision / recall at H=1000, budget 600\n")
        body = []
        for s in systems_in(sc):
            bb = 0 if s == "full_context" else 600
            cells = [s]
            for m in ("claim_p", "claim_r", "claim_f1", "wrong_rate", "tokens"):
                vals = list(per_seed([r for r in sc if r["kind"] == "focus"], m,
                                     lambda r, s=s, bb=bb: r["system"] == s and r["horizon"] == 1000 and r["budget"] == bb,
                                     by_kind_equal=False).values())
                cells.append(fmt(*ci(vals), digits=2 if m != "tokens" else 0))
            body.append(cells)
        out.append(table(["system", "precision", "recall", "F1", "wrong-sense rate", "tokens shown"], body))

        out.append("\n## Revision handling (stale queries), H=1000, budget 600\n")
        cats = ["current_only", "both_resolved", "both", "stale_only", "neither"]
        body = []
        for s in systems_in(sc):
            bb = 0 if s == "full_context" else 600
            rs = [r for r in sc if r["kind"] == "stale" and r["system"] == s and r["horizon"] == 1000 and r["budget"] == bb]
            n = len(rs) or 1
            body.append([s] + [f"{100 * sum(r['stale_cat'] == c for r in rs) / n:.0f}%" for c in cats] + [str(len(rs))])
        out.append(table(["system"] + cats + ["n"], body))

        out.append("\n## Recall of old information: headline vs age of the gold (H=3000, budget 600)\n")
        bins = [(0, 100), (100, 300), (300, 1000), (1000, 10**9)]
        body = []
        for s in systems_in(sc):
            bb = 0 if s == "full_context" else 600
            cells = [s]
            for lo, hi in bins:
                vals = [r["headline"] for r in sc if r["system"] == s and r["horizon"] == 3000 and r["budget"] == bb
                        and lo <= r["age"] < hi and r["kind"] in ("focus", "chain", "bridge")]
                cells.append(f"{statistics.mean(vals):.2f} (n={len(vals)})" if vals else "-")
            body.append(cells)
        out.append(table(["system"] + [f"age {lo}-{hi if hi < 10**9 else '∞'}" for lo, hi in bins], body))

        out.append("\n## Resources vs horizon\n")
        res = [r for r in sc if r["type"] == "resource"]
        hs2 = sorted({r["horizon"] for r in res})
        body = []
        for s in systems_in(res):
            cells = [s]
            for h in hs2:
                rr = [r for r in res if r["system"] == s and r["horizon"] == h]
                if rr:
                    cells.append(f"{statistics.mean(r['observe_ms_per_event'] for r in rr):.2f} ms / {statistics.mean(r.get('bytes', 0) for r in rr) / 1024:.0f} KB")
                else:
                    cells.append("-")
            body.append(cells)
        out.append(table(["system (observe ms/event / stored KB)"] + [f"H={h}" for h in hs2], body))
        lat = []
        for s in systems_in(sc):
            for h in hs2:
                v = [r["latency_ms"] for r in sc if r["type"] == "query" and r["system"] == s and r["horizon"] == h and r["budget"] == (0 if s == "full_context" else 600)]
                if v:
                    lat.append([s, str(h), f"{statistics.median(v):.2f}", f"{sorted(v)[int(0.95 * len(v)) - 1]:.2f}"])
        out.append("\n### Query latency (ms)\n")
        out.append(table(["system", "H", "median", "p95"], lat))

        out.append("\n## Paired comparison: world0_compact vs each baseline (H=1000, budget 600; utility over seeds)\n")
        body = []
        for s in systems_in(sc):
            if s in ("world0_compact",):
                continue
            bb = 600
            d, lo, hi, wins = paired(sc, "headline", "world0_compact", s,
                                     lambda r, s=s: r["horizon"] == 1000 and (r["budget"] == 600 or (s == "full_context" and r["budget"] == 0)))
            body.append([s, f"{d:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{wins}/6"])
        out.append(table(["baseline", "Δ utility (compact − baseline)", "bootstrap 95 %", "seeds won"], body))

    # 2. sensitivity studies
    for study, param, label in (("noise", "noise_rate", "chatter rate"), ("paraphrase", "task_paraphrase", "task paraphrase probability"),
                                ("verbosity", "verbosity", "filler tokens per event")):
        st = study_rows(rows, study)
        if not st:
            continue
        vals = sorted({r[param] for r in st})
        out.append(f"\n## Sensitivity to {label} (H=1000, budget 600, utility)\n")
        body = []
        for s in systems_in(st):
            cells = [s]
            for v in vals:
                bb = 0 if s == "full_context" else 600
                cells.append(fmt(*ci(list(per_seed(st, "headline", lambda r, s=s, v=v, bb=bb: r["system"] == s and r[param] == v and r["budget"] == bb).values()))))
            body.append(cells)
        out.append(table(["system"] + [f"{label}={v}" for v in vals], body))

    ab = study_rows(rows, "ablation")
    if ab:
        out.append("\n## World 0 variants (H=1000, budget 600, utility and per-kind)\n")
        body = []
        for s in systems_in(ab):
            if not s.startswith("world0"):
                continue
            cells = [s, fmt(*ci(list(per_seed(ab, "headline", lambda r, s=s: r["system"] == s and r["budget"] == 600).values())))]
            for k in KINDS:
                cells.append(fmt(*ci(list(per_seed([r for r in ab if r["kind"] == k], "headline",
                                                   lambda r, s=s: r["system"] == s and r["budget"] == 600, by_kind_equal=False).values()))))
            body.append(cells)
        out.append(table(["variant", "utility"] + KINDS, body))

    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/eval/results")
