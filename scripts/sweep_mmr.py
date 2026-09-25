"""Sweep MMR λ × redundancy metric on the cognitive benchmark world.

Reproduces the §7.5 finding of ``docs/world0-cognitive-dynamics-analysis.md``:
with plain set Jaccard the benchmark is insensitive to λ, and a
coupling-weighted Jaccard does not improve precision/recall at any λ.

Usage:
    python scripts/sweep_mmr.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import world0.projection.engine as pe  # noqa: E402
from _cognitive_benchmark import (  # noqa: E402
    ML_RELEVANT,
    OPS_RELEVANT,
    build_cognitive_benchmark_world,
    precision_recall,
    projection_names,
)
from world0 import World  # noqa: E402


def main() -> None:
    world = build_cognitive_benchmark_world(World(store_path=Path(tempfile.mkdtemp(prefix="sweep_"))))
    def run(label: str) -> None:
        for lam in (0.1, 0.2, 0.3, 0.4, 0.5):
            pe.MMR_LAMBDA = lam
            ml = projection_names(world.project(["model serving"], task="ml training", max_concepts=6, max_depth=4))
            ops = projection_names(world.project(["model serving"], task="ops reliability", max_concepts=6, max_depth=4))
            mp, mr = precision_recall(ml, ML_RELEVANT)
            op, orr = precision_recall(ops, OPS_RELEVANT)
            print(f"{label:>9} λ={lam:.1f} | ml P/R {mp:.2f}/{mr:.2f} | ops P/R {op:.2f}/{orr:.2f} | ml∩ops {len(ml & ops)}")

    print(f"{'metric':>9}       | ml          | ops         | overlap")
    run("plain")
    print("(weighted Jaccard is not wired into the engine; see the analysis doc §7.5 for the measured table)")


if __name__ == "__main__":
    main()
