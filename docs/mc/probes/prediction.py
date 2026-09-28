"""Round 18 probe: prediction error on ingest (PP-1).

Run from the repository root:  python docs/mc/probes/prediction.py
"""

from __future__ import annotations

import random
import statistics
import tempfile
import time

from world0 import Observation, World

DATA = ["pipeline", "etl job", "warehouse", "schema", "batch window"]
ML = ["pipeline", "training run", "checkpoint", "gpu", "loss curve"]


def fresh() -> World:
    return World(store_path=tempfile.mkdtemp() + "/w")


def main() -> None:
    rng = random.Random(7)

    print("1. routine vs departures, in a world of two stable clusters")
    w = fresh()
    for _ in range(20):
        w.ingest(Observation(concepts=DATA, source="s"))
        w.ingest(Observation(concepts=["kubernetes", "helm chart", "rollout"], source="s"))
    cases = {
        "routine (full DATA cluster)": DATA,
        "routine subset (3 of 5)": DATA[:3],
        "etl job alone": ["etl job"],
        "cross-cluster (etl job + rollout)": ["etl job", "rollout"],
    }
    for label, concepts in cases.items():
        pe = w.ingest(Observation(concepts=concepts, source="s")).prediction
        print(f"   {label:34} surprise={pe.surprise:.2f} missing={pe.missing_ratio:.2f} "
              f"novelty={pe.novelty:.2f}  novel={pe.novel_pairs[:2]} missing={[m[1] for m in pe.missing][:4]}")

    print("\n2. concept drift: 'pipeline' used with DATA for 150 observations, then with ML")
    w = fresh()
    series = []
    for t in range(300):
        cluster = DATA if t < 150 else ML
        pick = rng.sample(cluster[1:], 3)
        series.append(w.ingest(Observation(concepts=["pipeline", *pick], source="s")).prediction.surprise)
    windows = [statistics.mean(series[i:i + 25]) for i in range(0, 300, 25)]
    print("   mean surprise per 25 observations:", [round(x, 2) for x in windows])

    print("\n3. structured vs random world (mean surprise over the last 200 of 400 observations)")
    for kind in ("structured", "random"):
        w = fresh()
        topics = [[f"t{t}c{i}" for i in range(6)] for t in range(10)]
        pool = [c for t in topics for c in t]
        s = []
        for step in range(400):
            obs = rng.sample(rng.choice(topics), 4) if kind == "structured" else rng.sample(pool, 4)
            s.append(w.ingest(Observation(concepts=obs, source="s")).prediction.surprise)
        print(f"   {kind:10}: {statistics.mean(s[200:]):.2f}")

    print("\n4. cost: 3 000 observations into 100 topics x 20 concepts")
    w = World(store_path=tempfile.mkdtemp() + "/w.sqlite")
    t0 = time.perf_counter()
    for step in range(3000):
        t = step % 100
        w.ingest(Observation(concepts=[f"t{t} c{(step // 100 + j) % 20}" for j in range(6)], source="s"))
    print(f"   {time.perf_counter() - t0:.1f} s total, {(time.perf_counter() - t0) / 3:.2f} ms/observation")


if __name__ == "__main__":
    main()
