"""Sweep ``SALIENCE_EVIDENCE_SHARE`` (§7.1 of the dynamics analysis).

Two measurements per share:

* the cognitive benchmark (ML / Ops precision and recall) — must not move;
* a slot-limited contest: a dependency confirmed fifty times, then dormant
  for Δ observations, competes with six fresh one-off mentions for four
  projection slots under the same seed.

The first row disables the evidence term entirely (old behaviour: time
charged twice against a dormant neighbor); the second keeps only the
evidence-aware readiness; the rest add the persistence floor.

Usage:
    python scripts/sweep_salience.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import world0.schemas.concept as concept_mod  # noqa: E402
from _cognitive_benchmark import (  # noqa: E402
    ML_RELEVANT,
    OPS_RELEVANT,
    build_cognitive_benchmark_world,
    precision_recall,
    projection_names,
    ranked_projection_names,
)
from world0 import Observation, World  # noqa: E402
from world0.schemas.concept import ConceptNode  # noqa: E402


def contest(dormant: int, rookies: int = 6, slots: int = 4):
    w = World(store_path=tempfile.mkdtemp())
    for _ in range(50):
        w.ingest(
            Observation(
                concepts=["Seed", "Veteran"],
                relations=[("Seed", "Veteran", "depends_on")],
                source="s",
            )
        )
        w.clock.advance(23)
    w.clock.advance(dormant)
    w.reflect(light=True)
    for i in range(rookies):
        w.ingest(
            Observation(
                concepts=["Seed", f"Rookie{i}"],
                relations=[("Seed", f"Rookie{i}", "depends_on")],
                source="s",
            )
        )
    ranked = ranked_projection_names(w.project(["Seed"], max_concepts=slots))
    seed = w.concepts.resolve("Seed").id
    scores = w._activation.activate([seed], record=False)
    veteran = scores.get(w.concepts.resolve("Veteran").id, 0.0)
    rookie = max(scores.get(w.concepts.resolve(f"Rookie{i}").id, 0.0) for i in range(rookies))
    rank = ranked.index("Veteran") + 1 if "Veteran" in ranked else None
    return veteran, rookie, rank


def benchmark():
    w = build_cognitive_benchmark_world(World(store_path=tempfile.mkdtemp()))
    ml = projection_names(
        w.project(["model serving"], task="ml training", max_concepts=6, max_depth=4)
    )
    ops = projection_names(
        w.project(["model serving"], task="ops reliability", max_concepts=6, max_depth=4)
    )
    return precision_recall(ml, ML_RELEVANT), precision_recall(ops, OPS_RELEVANT)


def main() -> None:
    real_evidence = ConceptNode.evidence
    rows = [
        ("baseline (no evidence term)", 0.0, lambda self, saturation_k=10.0: 0.0),
        ("readiness only", 0.0, real_evidence),
    ] + [(f"share={s:.1f}", s, real_evidence) for s in (0.3, 0.5, 0.7, 1.0)]
    try:
        for label, share, evidence in rows:
            concept_mod.SALIENCE_EVIDENCE_SHARE = share
            ConceptNode.evidence = evidence
            (mlp, mlr), (opp, opr) = benchmark()
            print(f"{label:28s} ML p/r={mlp:.2f}/{mlr:.2f}  Ops p/r={opp:.2f}/{opr:.2f}")
            for dormant in (500, 1000, 3000, 8000):
                veteran, rookie, rank = contest(dormant)
                print(
                    f"    Δ{dormant:<5d} veteran={veteran:.4f} rookie={rookie:.4f} "
                    f"ratio={veteran / max(rookie, 1e-9):.2f} rank@4={rank}"
                )
    finally:
        ConceptNode.evidence = real_evidence


if __name__ == "__main__":
    main()
