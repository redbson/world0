"""Choose World 0's depth / reflect cadence on DEV seeds (never test seeds).

    python -m benchmarks.longrun.run --study tune --out docs/eval/results-dev
    python -m benchmarks.longrun.tune docs/eval/results-dev
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict

from benchmarks.longrun.systems import TUNED_PATH

KINDS = ("focus", "chain", "bridge", "stale")
BUDGET_SET = (300, 600, 1200)


def main(path: str) -> None:
    rows = [json.loads(line) for line in open(os.path.join(path, "tune.jsonl"))]
    acc: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["type"] == "query" and r["kind"] in KINDS and r["budget"] in BUDGET_SET:
            acc[r["system"]][r["seed"]][r["kind"]].append(r["headline"])
    score = {}
    for system, seeds in acc.items():
        per_seed = [statistics.mean(statistics.mean(v) for v in kinds.values()) for kinds in seeds.values()]
        score[system] = statistics.mean(per_seed)
    for system, s in sorted(score.items(), key=lambda kv: -kv[1]):
        print(f"{system:14s} utility4 {s:.3f}")
    best = max(score, key=score.get)
    depth, reflect = best[4:].split("-r")
    cfg = {"depth": int(depth), "reflect_every": int(reflect) or None, "chosen_on": "dev seeds 100-104",
           "utility4_by_variant": {k: round(v, 4) for k, v in score.items()}}
    with open(TUNED_PATH, "w") as fh:
        json.dump(cfg, fh, indent=1)
    print("wrote", TUNED_PATH, cfg["depth"], cfg["reflect_every"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/eval/results-dev")
