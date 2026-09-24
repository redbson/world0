"""Calibrate evidence-anchored concept decay across usage cadences.

Simulates a concept that is activated every ``gap`` hours for ``days`` days,
advancing *all* reference timestamps (including ``last_decayed_at``) between
activations and running decay + lifecycle after each one.  Prints, per
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
from datetime import timedelta
from pathlib import Path

import world0.dynamics.decay as decay_mod
from world0 import Observation, World
from world0.schemas.concept import Maturity


def advance(world: World, hours: float) -> None:
    """Move every dynamics timestamp ``hours`` into the past."""
    delta = timedelta(hours=hours)
    for node in world.concepts.all():
        node.last_activated -= delta
        if node.last_decayed_at:
            node.last_decayed_at -= delta
    for edge in world.relations.all():
        edge.last_reinforced -= delta
        if edge.last_decayed_at:
            edge.last_decayed_at -= delta


def tick(world: World) -> None:
    world._decay.decay_concepts()
    world._decay.decay_relations()
    world._lifecycle.evaluate()


def simulate(gap_hours: float, days: int) -> dict:
    world = World(store_path=Path(tempfile.mkdtemp(prefix="cal_")))
    steps = int(days * 24 / gap_hours)
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
        advance(world, gap_hours)
        tick(world)
        node = world.concepts.resolve("c")
        day = round((i + 1) * gap_hours / 24, 1)
        first_reached.setdefault(node.maturity.value, day)
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


def one_shot_fade_hours() -> float:
    world = World(store_path=Path(tempfile.mkdtemp(prefix="cal1_")))
    world.ingest(Observation(concepts=["once"], source="s"))
    node = world.concepts.resolve("once")
    hours = 0.0
    while node.maturity != Maturity.FADING and hours < 24 * 30:
        advance(world, 6)
        hours += 6
        world._decay.decay_concepts()
    return hours


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
    for step_days in (30, 60, 90, 185, 365, 365):
        advance(world, step_days * 24)
        total += step_days
        world._decay.decay_concepts()
        out.append((total, round(node.confidence, 3), node.maturity.value))
    return out


def main(argv: list[str]) -> None:
    gains = [float(g) for g in argv] or [0.1, 0.2, 0.3]
    for gain in gains:
        decay_mod.CONCEPT_EVIDENCE_HL_GAIN = gain
        print(f"=== CONCEPT_EVIDENCE_HL_GAIN = {gain} ===")
        for gap, days in ((24, 90), (72, 180), (168, 365), (720, 730)):
            r = simulate(gap, days)
            print(
                f"  every {gap:>4}h × {days:>3}d → final={r['final']:<11} "
                f"conf={r['conf']:<6} n={r['n']:<4} first_reached={r['first']} "
                f"fading_episodes={r['fading_episodes']}"
            )
        print(f"  one-shot concept fades after ≈ {one_shot_fade_hours():.0f} h")
        print(f"  30× then abandoned (established, 0.8): {abandon_decline(30)}")


if __name__ == "__main__":
    main(sys.argv[1:])
