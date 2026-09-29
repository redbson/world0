"""Run LongRun studies:  python -m benchmarks.longrun.run --study main --out docs/eval/results"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from benchmarks.longrun.scoring import score
from benchmarks.longrun.systems import make_system
from benchmarks.longrun.worldgen import GenConfig, Stream

BUDGETS = [150, 300, 600, 1200, 2400, 4800]
CORE = ["none", "window", "full_context", "rag", "rag_recency", "summary_buffer", "summary_task",
        "factstore", "fact_task", "kg_static", "kg_temporal", "state_doc",
        "world0", "world0_compact", "world0_tuned"]
ABLATION = CORE + ["world0_reflect", "world0_focus", "world0_notask", "world0_depth1", "world0_depth3"]
TUNE = ["w0-d1-r0", "w0-d2-r0", "w0-d3-r0", "w0-d1-r25", "w0-d2-r25", "w0-d3-r25"]
DEV_SEEDS = range(100, 105)


def studies(seeds: int) -> dict[str, list[dict]]:
    def jobs(systems, seed_list=None, **kw):
        return [dict(kw, seed=s, systems=systems, jitter=True) for s in (seed_list or range(seeds))]

    out: dict[str, list[dict]] = {
        "main": jobs(CORE, horizon=1000),
        "scale": [j for h in (300, 3000, 6000) for j in jobs(
            CORE, list(range(min(seeds, 6 if h < 6000 else 4))), horizon=h, query_every=max(10, h // 150))],
        "bigworld": jobs(CORE, list(range(min(seeds, 4))), horizon=6000, n_domains=40,
                         concepts_per_domain=40, growth=True, query_every=40),
        "extraction": [j for p in (0.05, 0.1, 0.2, 0.3) for j in jobs(CORE, list(range(min(seeds, 6))), horizon=1000, extract_p=p)],
        "taskmode": [j for m in ("none", "wrong") for j in jobs(CORE, list(range(min(seeds, 6))), horizon=1000, task_mode=m)],
        "chatter": [j for n in (0.0, 0.5, 0.75) for j in jobs(CORE, list(range(min(seeds, 6))), horizon=1000, noise_rate=n)],
        "verbosity": [j for v in (0, 120, 400) for j in jobs(CORE, list(range(min(seeds, 6))), horizon=1000, verbosity=v)],
        "ablation": jobs(ABLATION, list(range(min(seeds, 6))), horizon=1000),
        "tune": jobs(TUNE, list(DEV_SEEDS), horizon=1000),
        "smoke": [dict(horizon=200, seed=0, systems=CORE)],
    }
    return out


def link(known: set[str], text: str) -> list[str]:
    low = text.lower()
    out = []
    for name in sorted(known, key=len, reverse=True):
        if name.lower() in low and not any(name.lower() in o.lower() for o in out):
            out.append(name)
    return out


def _jitter(job: dict) -> dict:
    """Seed-derived generator hyper-parameters, so no conclusion rests on one setting."""
    rng = random.Random(job["seed"] * 7919 + 13)
    draw = {"popularity_skew": rng.uniform(0.6, 1.0), "focus_run_mean": rng.randint(15, 40),
            "dormant_domains": rng.randint(1, 3), "detail_rate": rng.uniform(0.1, 0.2)}
    return {**draw, **job}


def run_one(job: dict) -> list[dict]:
    job = dict(job)
    names = job.pop("systems")
    study = job.pop("study", "")
    timing = job.pop("timing", False)
    if job.pop("jitter", False):
        job = _jitter(job)
    cfg = GenConfig(**job)
    stream = Stream(cfg)
    systems: dict = {}
    for n in names:
        systems[n] = make_system(n, systems)
        if timing and hasattr(systems[n], "use_cache"):
            systems[n].use_cache = False
    observe_s = {n: 0.0 for n in names}
    known: set[str] = set()
    rows: list[dict] = []
    key = {"horizon": cfg.horizon, "noise_rate": cfg.noise_rate, "task_mode": cfg.task_mode,
           "verbosity": cfg.verbosity, "extract_p": cfg.extract_p, "n_domains": cfg.n_domains,
           "growth": cfg.growth, "seed": cfg.seed}
    stream_tokens = 0
    for ev, qs in stream.events():
        known |= set(ev.concepts)
        stream_tokens += ev.tokens
        for n, s in systems.items():
            t0 = time.perf_counter()
            s.observe(ev)
            observe_s[n] += time.perf_counter() - t0
        for q in qs:
            linked = link(known, q.text)
            for n, s in systems.items():
                for b in ([0] if s.unbounded else BUDGETS):
                    t0 = time.perf_counter()
                    ctx = s.query(q, linked, b or 10**9)
                    lat = (time.perf_counter() - t0) * 1000
                    if not s.unbounded and ctx.tokens > b:
                        raise AssertionError(f"{n} exceeded budget: {ctx.tokens} > {b}")
                    r = score(q, ctx)
                    r.update(type="query", study=study, system=n, budget=b, kind=q.kind, step=q.step,
                             age=q.age, gap=q.gap, gold_n=len(q.gold_claims), latency_ms=lat, **key)
                    rows.append(r)
    for n, s in systems.items():
        rows.append({"type": "resource", "study": study, "system": n, "state_tokens": stream.state_tokens(),
                     "stream_tokens": stream_tokens,
                     "observe_ms_per_event": 1000 * observe_s[n] / cfg.horizon, **s.stats(), **key})
    for s in systems.values():
        s.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=list(studies(1)))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="docs/eval/results")
    ap.add_argument("--timing", action="store_true", help="disable World 0's per-query projection cache")
    args = ap.parse_args()
    jobs = [dict(j, study=args.study, timing=args.timing) for j in studies(args.seeds)[args.study]]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.study}.jsonl")
    t0 = time.time()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "benchmarks"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", False
    n = 0
    with open(path, "w") as fh, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows in ex.map(run_one, jobs):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                n += 1
            fh.flush()
            print(f"  done {n} rows  ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out, f"{args.study}.meta.json"), "w") as fh:
        json.dump({"study": args.study, "jobs": len(jobs), "seeds": args.seeds, "budgets": BUDGETS,
                   "commit": commit, "dirty_src_or_benchmarks": dirty, "seconds": round(time.time() - t0, 1),
                   "python": sys.version.split()[0], "cpus": os.cpu_count(), "workers": args.workers}, fh, indent=1)


if __name__ == "__main__":
    main()
