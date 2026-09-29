"""Run LongRun studies:  python -m benchmarks.longrun.run --study scale --out docs/eval/results"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from benchmarks.longrun.scoring import score
from benchmarks.longrun.systems import ALL_SYSTEMS, World0Compact, World0NoTask, World0Shallow
from benchmarks.longrun.worldgen import GenConfig, Stream

BUDGETS = [150, 300, 600, 1200]
CORE = ["none", "window", "full_context", "rag", "rag_recency", "summary_buffer",
        "factstore", "kg_static", "kg_temporal", "world0", "world0_compact"]
ABLATION = CORE + ["world0_reflect", "world0_focus", "world0_notask", "world0_depth1"]
SHARED = {"world0_compact": World0Compact, "world0_notask": World0NoTask, "world0_depth1": World0Shallow}


def studies(seeds: int) -> dict[str, list[dict]]:
    def cfg(**kw):
        return dict(kw)

    out: dict[str, list[dict]] = {
        "scale": [cfg(horizon=h, seed=s, systems=CORE) for h in (300, 1000, 3000) for s in range(seeds)]
                 + [cfg(horizon=6000, seed=s, systems=CORE) for s in range(min(seeds, 3))],
        "noise": [cfg(horizon=1000, noise_rate=n, seed=s, systems=CORE) for n in (0.0, 0.25, 0.5, 0.75) for s in range(seeds)],
        "paraphrase": [cfg(horizon=1000, task_paraphrase=t, seed=s, systems=CORE) for t in (0.0, 0.5, 1.0) for s in range(seeds)],
        "verbosity": [cfg(horizon=1000, verbosity=v, seed=s, systems=CORE) for v in (0, 30, 120, 400) for s in range(seeds)],
        "ablation": [cfg(horizon=1000, seed=s, systems=ABLATION) for s in range(seeds)],
        "smoke": [cfg(horizon=200, seed=0, systems=CORE)],
    }
    return out


def link(known: set[str], text: str) -> list[str]:
    low = text.lower()
    out = []
    for name in sorted(known, key=len, reverse=True):
        if name.lower() in low and not any(name.lower() in o.lower() for o in out):
            out.append(name)
    return out


def run_one(job: dict) -> list[dict]:
    job = dict(job)
    names = job.pop("systems")
    study = job.pop("study", "")
    cfg = GenConfig(**job)
    stream = Stream(cfg)
    systems = {}
    for n in names:
        cls = ALL_SYSTEMS[n]
        systems[n] = cls(shared=systems["world0"]) if n in SHARED else cls()
    observe_s = {n: 0.0 for n in names}
    known: set[str] = set()
    rows: list[dict] = []
    key = {k: v for k, v in asdict(cfg).items() if k in
           ("horizon", "noise_rate", "task_paraphrase", "verbosity", "seed")}
    for ev, qs in stream.events():
        known |= set(ev.concepts)
        for n, s in systems.items():
            t0 = time.perf_counter()
            s.observe(ev)
            observe_s[n] += time.perf_counter() - t0
        for q in qs:
            linked = link(known, q.text)
            for n, s in systems.items():
                budgets = [0] if s.unbounded else BUDGETS
                for b in budgets:
                    t0 = time.perf_counter()
                    ctx = s.query(q, linked, b or 10**9)
                    lat = (time.perf_counter() - t0) * 1000
                    if not s.unbounded and ctx.tokens > b:
                        raise AssertionError(f"{n} exceeded budget: {ctx.tokens} > {b}")
                    r = score(q, ctx)
                    r.update(type="query", study=study, system=n, budget=b, kind=q.kind,
                             step=q.step, age=q.age, gap=q.gap, gold_n=len(q.gold_claims),
                             latency_ms=lat, **key)
                    rows.append(r)
    for n, s in systems.items():
        rows.append({"type": "resource", "study": study, "system": n,
                     "observe_ms_per_event": 1000 * observe_s[n] / cfg.horizon,
                     **s.stats(), **key})
    for s in systems.values():
        s.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=list(studies(1)))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="docs/eval/results")
    args = ap.parse_args()
    jobs = [dict(j, study=args.study) for j in studies(args.seeds)[args.study]]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.study}.jsonl")
    t0 = time.time()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
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
                   "commit": commit, "seconds": round(time.time() - t0, 1), "python": sys.version.split()[0]}, fh, indent=1)


if __name__ == "__main__":
    main()
