"""LLM-reader stage: does a real reader answer better from each system's context?

``prepare`` replays seeded streams, builds for every (query, condition) a file
holding the memory a system produced plus the question, under an opaque file
name; the gold stays in a manifest kept elsewhere.  A workflow then has one
reader agent per file (no other input), and ``grade`` scores the answers.

    python -m benchmarks.longrun.readers prepare --out <dir> --gold <manifest.json>
    python -m benchmarks.longrun.readers grade --gold <manifest.json> --answers <answers.json>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

from benchmarks.longrun.run import link
from benchmarks.longrun.systems import make_system
from benchmarks.longrun.worldgen import GenConfig, Query, Stream

# (condition name, system, budget)  budget 0 = unbounded / none
CONDITIONS = [
    ("none", "none", 0),
    ("full_context", "full_context", 0),
    ("window@1200", "window", 1200),
    ("rag@600", "rag", 600),
    ("rag@1200", "rag", 1200),
    ("summary_task@1200", "summary_task", 1200),
    ("factstore@1200", "factstore", 1200),
    ("fact_task@1200", "fact_task", 1200),
    ("kg_temporal@1200", "kg_temporal", 1200),
    ("state_doc@1200", "state_doc", 1200),
    ("world0@600", "world0", 600),
    ("world0@1200", "world0", 1200),
    ("world0_compact@600", "world0_compact", 600),
    ("world0_compact@1200", "world0_compact", 1200),
]
CAPS = {"focus": 8, "chain": 8, "bridge": 6, "stale": 6, "detail": 5}   # queries kept per kind per seed
SALT = "longrun-readers-v1"
INSTRUCTIONS = (
    "Answer ONLY from the MEMORY above. All names in this project are invented, so you cannot guess them; "
    "if the memory does not contain the answer, return an empty list. Names must be copied exactly as they "
    "appear in the memory. Return the answer as a list of names (for a ticket question, a one-element list with "
    "the ticket id). Do not use any tool other than reading this file, and do not read any other file."
)


def fnv(s: str) -> int:
    h = 2166136261
    for b in s.encode():
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def fname(qid: int, cond: str, salt: str = SALT) -> str:
    return f"m{fnv(f'{salt}|{qid}|{cond}'):08x}.txt"


def question(q: Query) -> tuple[str, set[str]]:
    """Reader-facing question and the gold answer set (concept names)."""
    prefix = f"{q.task_text}: " if q.task_text else ""
    if q.kind == "focus":
        e = q.entry
        return (f"{prefix}investigate {e[0]} and {e[1]}. List every concept that is directly or indirectly related to "
                f"them in this project area (up to two relationship steps away), not counting {e[0]} and {e[1]} themselves.",
                q.gold_concepts - set(e))
    if q.kind == "chain":
        x = q.entry[0]
        return (f"{prefix}what does {x} ultimately rely on? List every concept it depends on, directly or through other concepts.",
                set(q.gold_answer))
    if q.kind == "bridge":
        b = q.entry[0]
        return (f"{prefix}how does {b} fit here? List every concept that {b} is directly related to in this project area.",
                q.gold_concepts - {b})
    if q.kind == "stale":
        a = q.entry[0]
        return (f"{prefix}what does {a} depend on right now? List the concepts it directly depends on "
                f"(ignore anything that has since been corrected).",
                {c.tgt for c in q.gold_claims if c.src == a and c.rel == "depends_on"})
    claim = next(iter(q.gold_claims))
    return (f"{prefix}which ticket was raised when we noted that {claim.sentence()}? Give the ticket id.", {q.gold_ticket})


def prepare(out: str, gold_path: str, seeds: list[int], horizon: int, every: int,
            conds: list[str] | None = None, salt: str = SALT, caps: dict[str, int] | None = None) -> None:
    os.makedirs(out, exist_ok=True)
    caps = caps or CAPS
    conditions = [c for c in CONDITIONS if conds is None or c[0] in conds]
    manifest, qid = [], 0
    for seed in seeds:
        cfg = GenConfig(seed=seed, horizon=horizon, query_every=every)
        # First pass (generator only): choose which queries to keep, per kind, at random.
        by_kind: dict[str, list[int]] = defaultdict(list)
        for _, qs in Stream(cfg).events():
            for q in qs:
                by_kind[q.kind].append(q.step)
        rng = random.Random(seed)
        keep = {(k, st) for k, steps in by_kind.items() for st in rng.sample(steps, min(caps[k], len(steps)))}
        stream = Stream(cfg)
        systems: dict = {}
        for _, sysname, _ in conditions:
            if sysname not in systems:
                systems[sysname] = make_system(sysname, systems)
        known: set[str] = set()
        for ev, qs in stream.events():
            known |= set(ev.concepts)
            for s in systems.values():
                s.observe(ev)
            for q in qs:
                if (q.kind, q.step) not in keep:
                    continue
                text, gold = question(q)
                linked = link(known, q.text)
                entry = {"qid": qid, "seed": seed, "step": q.step, "kind": q.kind, "gold": sorted(gold),
                         "stale": sorted(c.tgt for c in q.stale_claims), "tokens": {}}
                for cond, sysname, budget in conditions:
                    ctx = systems[sysname].query(q, linked, budget or 10**9)
                    memory = ctx.text.strip() or "(nothing)"
                    body = f"MEMORY:\n{memory}\n\nQUESTION:\n{text}\n\nINSTRUCTIONS:\n{INSTRUCTIONS}\n"
                    with open(os.path.join(out, fname(qid, cond, salt)), "w") as fh:
                        fh.write(body)
                    entry["tokens"][cond] = ctx.tokens
                manifest.append(entry)
                qid += 1
        for s in systems.values():
            s.close()
    with open(gold_path, "w") as fh:
        json.dump({"salt": salt, "conditions": [c for c, _, _ in conditions], "queries": manifest}, fh)
    print(f"{qid} queries x {len(conditions)} conditions -> {out}")


def f1(pred: set[str], gold: set[str]) -> float:
    hit = len(pred & gold)
    if not pred and not gold:
        return 1.0
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def grade(gold_path: str, answers_path: str) -> None:
    man = json.load(open(gold_path))
    raw = json.load(open(answers_path))
    answers = {(a["q"], a["c"]): a["answer"] for a in raw if a}
    by = defaultdict(list)
    stale_named = defaultdict(list)
    for q in man["queries"]:
        for cond in man["conditions"]:
            ans = answers.get((q["qid"], cond))
            if ans is None:
                continue
            pred = {x.strip().lower() for x in ans}
            gold = {x.lower() for x in q["gold"]}
            by[(cond, q["kind"])].append(f1(pred, gold))
            if q["kind"] == "stale":
                stale_named[cond].append(float(any(x.lower() in pred for x in q["stale"])))
    kinds = ["focus", "chain", "bridge", "stale", "detail"]
    print("| condition | " + " | ".join(kinds) + " | four-kind mean | stale named | n |")
    print("|---|" + "---|" * (len(kinds) + 3))
    for cond in man["conditions"]:
        cells, four = [], []
        for k in kinds:
            v = by.get((cond, k), [])
            cells.append(f"{statistics.mean(v):.2f}" if v else "-")
            if v and k != "detail":
                four.append(statistics.mean(v))
        sn = stale_named.get(cond, [])
        n = sum(len(by.get((cond, k), [])) for k in kinds)
        print(f"| {cond} | " + " | ".join(cells) + f" | {statistics.mean(four):.2f} | "
              + (f"{100 * statistics.mean(sn):.0f}%" if sn else "-") + f" | {n} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--every", type=int, default=8)
    p.add_argument("--conds", nargs="+", default=None)
    p.add_argument("--salt", default=SALT)
    p.add_argument("--cap", type=int, default=None, help="queries per kind per seed (default: the CAPS table)")
    g = sub.add_parser("grade")
    g.add_argument("--gold", required=True)
    g.add_argument("--answers", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        caps = {k: a.cap for k in CAPS} if a.cap else None
        prepare(a.out, a.gold, a.seeds, a.horizon, a.every, a.conds, a.salt, caps)
    else:
        grade(a.gold, a.answers)


if __name__ == "__main__":
    main()
