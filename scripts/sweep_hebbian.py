"""Sweep ``HEBBIAN_MIN_ASSOCIATION`` (analysis doc §7.11).

Three worlds per threshold:

* **random** — 60 concepts, 400 observations of 6 drawn uniformly: every
  co-occurrence is chance, so generic edges are sprawl.
* **topics** — the same 60 concepts in 6 topics of 10; 80 % of the
  observations draw 6 concepts from one topic, 20 % mix two topics.
  Within-topic edges are signal, cross-topic edges are noise.
* the cognitive benchmark (ML / Ops precision and recall) — must not move.

Usage:
    python scripts/sweep_hebbian.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import world0.dynamics.hebbian as hebbian_mod  # noqa: E402
from _cognitive_benchmark import (  # noqa: E402
    ML_RELEVANT,
    OPS_RELEVANT,
    build_cognitive_benchmark_world,
    precision_recall,
    projection_names,
)
from world0 import Observation, World  # noqa: E402

POOL = [f"c{i}" for i in range(60)]
TOPICS = [POOL[i * 10 : (i + 1) * 10] for i in range(6)]
TOPIC_OF = {name: i for i, topic in enumerate(TOPICS) for name in topic}


def random_world(seed: int = 7) -> World:
    rng = random.Random(seed)
    w = World(store_path=tempfile.mkdtemp())
    for _ in range(400):
        w.ingest(Observation(concepts=rng.sample(POOL, 6), source="s"))
    return w


def topic_world(seed: int = 7) -> World:
    rng = random.Random(seed)
    w = World(store_path=tempfile.mkdtemp())
    for _ in range(400):
        if rng.random() < 0.8:
            concepts = rng.sample(rng.choice(TOPICS), 6)
        else:
            a, b = rng.sample(TOPICS, 2)
            concepts = rng.sample(a, 3) + rng.sample(b, 3)
        w.ingest(Observation(concepts=concepts, source="s"))
    return w


def generic_edges(w: World):
    within = cross = 0
    for r in w.relations.all():
        if r.semantic_relation != "generic_relation":
            continue
        a = w.concepts.get(r.source_id).name
        b = w.concepts.get(r.target_id).name
        if TOPIC_OF[a] == TOPIC_OF[b]:
            within += 1
        else:
            cross += 1
    return within, cross


def benchmark():
    w = build_cognitive_benchmark_world(World(store_path=tempfile.mkdtemp()))
    ml = projection_names(
        w.project(["model serving"], task="ml training", max_concepts=6, max_depth=4)
    )
    ops = projection_names(
        w.project(["model serving"], task="ops reliability", max_concepts=6, max_depth=4)
    )
    hebbian = sum(1 for r in w.relations.all() if not r.is_explicit)
    return precision_recall(ml, ML_RELEVANT), precision_recall(ops, OPS_RELEVANT), hebbian


def main() -> None:
    total_pairs = 60 * 59 // 2
    within_pairs = 6 * (10 * 9 // 2)
    print(f"pairs: {total_pairs} total, {within_pairs} within-topic")
    for theta in (0.0, 0.1, 0.2, 0.3, 0.4):
        hebbian_mod.HEBBIAN_MIN_ASSOCIATION = theta
        rw, rc = generic_edges(random_world())
        tw, tc = generic_edges(topic_world())
        (mlp, mlr), (opp, opr), hebb = benchmark()
        print(
            f"θ={theta:.1f}  random: {rw + rc:4d} generic edges ({(rw + rc) / total_pairs:.0%} of pairs)"
            f"  topics: within {tw:3d}/{within_pairs} ({tw / within_pairs:.0%}), cross {tc:3d}"
            f"  benchmark: ML {mlp:.2f}/{mlr:.2f} Ops {opp:.2f}/{opr:.2f} hebbian edges {hebb}"
        )


if __name__ == "__main__":
    main()
