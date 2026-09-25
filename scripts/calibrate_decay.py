"""Calibrate evidence-anchored concept decay across re-observation cadences.

Simulates a concept that is re-observed every ``gap`` observations for
``total`` observations of cognitive time (World 0 counts time in ingests,
not hours), running decay + lifecycle after each activation.  Prints, per
``CONCEPT_EVIDENCE_HL_GAIN`` value, the final maturity/confidence, the day
each maturity stage was first reached, how long a one-off concept takes to
fade, and how a heavily confirmed concept declines once abandoned.

Usage:
    python scripts/calibrate_decay.py            # sweeps 0.1 0.2 0.3
    python scripts/calibrate_decay.py 0.15 0.25  # custom gains

The resulting table is reproduced in
``docs/world0-cognitive-dynamics-analysis.md`` §5.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import world0.dynamics.decay as decay_mod
from world0 import Observation, World
from world0.schemas.concept import Maturity


def advance(world: World, ticks: int) -> None:
    """Let ``ticks`` observations pass without touching any concept."""
    world.clock.advance(int(ticks))


def tick(world: World) -> None:
    world._decay.decay_concepts()
    world._decay.decay_relations()
    world._lifecycle.evaluate()


def simulate(gap_ticks: int, total_ticks: int) -> dict:
    world = World(store_path=Path(tempfile.mkdtemp(prefix="cal_")))
    steps = int(total_ticks / gap_ticks)
    first_reached: dict[str, float] = {}
    fading_episodes = 0
    previous: Maturity | None = None
    for i in range(steps):
        world.ingest(
            Observation(
                concepts=["c", "anchor"],
                relations=[("c", "anchor", "depends_on")],
                task="routine",
                source="s",
            )
        )
        advance(world, gap_ticks - 1)  # the ingest itself is one tick
        tick(world)
        node = world.concepts.resolve("c")
        first_reached.setdefault(node.maturity.value, world.clock.tick)
        if node.maturity == Maturity.FADING and previous != Maturity.FADING:
            fading_episodes += 1
        previous = node.maturity
    node = world.concepts.resolve("c")
    return {
        "final": node.maturity.value,
        "conf": round(node.confidence, 3),
        "n": node.activation_count,
        "first": first_reached,
        "fading_episodes": fading_episodes,
    }


def one_shot_fade_ticks() -> int:
    world = World(store_path=Path(tempfile.mkdtemp(prefix="cal1_")))
    world.ingest(Observation(concepts=["once"], source="s"))
    node = world.concepts.resolve("once")
    ticks = 0
    while node.maturity != Maturity.FADING and ticks < 720:
        advance(world, 6)
        ticks += 6
        world._decay.decay_concepts()
    return ticks


def abandon_decline(activations: int) -> list[tuple[int, float, str]]:
    world = World(store_path=Path(tempfile.mkdtemp(prefix="cal2_")))
    for _ in range(activations):
        world.ingest(
            Observation(
                concepts=["heavy", "a2"],
                relations=[("heavy", "a2", "depends_on")],
                task="t",
                source="s",
            )
        )
    node = world.concepts.resolve("heavy")
    node.maturity = Maturity.ESTABLISHED
    node.confidence = 0.8
    out: list[tuple[int, float, str]] = []
    total = 0
    for step_ticks in (720, 1440, 2160, 4440, 8760, 8760):
        advance(world, step_ticks)
        total += step_ticks
        world._decay.decay_concepts()
        out.append((total, round(node.confidence, 3), node.maturity.value))
    return out


def main(argv: list[str]) -> None:
    gains = [float(g) for g in argv] or [0.1, 0.2, 0.3]
    for gain in gains:
        decay_mod.CONCEPT_EVIDENCE_HL_GAIN = gain
        print(f"=== CONCEPT_EVIDENCE_HL_GAIN = {gain} ===")
        for gap, total in ((24, 2160), (72, 4320), (168, 8760), (720, 17520)):
            r = simulate(gap, total)
            print(
                f"  every {gap:>4} obs × {total:>5} obs → final={r['final']:<11} "
                f"conf={r['conf']:<6} n={r['n']:<4} first_reached(tick)={r['first']} "
                f"fading_episodes={r['fading_episodes']}"
            )
        print(f"  one-shot concept fades after ≈ {one_shot_fade_ticks()} observations")
        print(f"  30× then abandoned (established, 0.8) [(ticks, conf, maturity)]: {abandon_decline(30)}")


if __name__ == "__main__":
    main(sys.argv[1:])
