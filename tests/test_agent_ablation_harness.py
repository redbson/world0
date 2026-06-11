"""Smoke test for the agent-context ablation harness.

Asserts mechanism-deterministic facts about the fixture, not
performance rankings (which may legitimately shift as coefficients
tune): every variant produces metrics, the projection covers the
expected concepts, and the violation metric separates silent dumps
from warning-guarded projections.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_agent_ablation import (  # noqa: E402
    FIXTURE_TASKS,
    VARIANTS,
    build_fixture_world,
    run_ablation,
)


def test_ablation_harness_end_to_end(tmp_path):
    world, log = build_fixture_world(tmp_path / ".world0")
    results = run_ablation(world, log, FIXTURE_TASKS)

    # Every variant scored, with all metric keys.
    assert set(results) == set(VARIANTS)
    for metrics in results.values():
        assert {"tokens", "coverage", "violation_rate", "efficiency"} <= set(
            metrics
        )

    # The floor is a floor.
    assert results["none"]["coverage"] == 0.0

    # The projection surfaces the expected concepts on the fixture.
    assert results["projection"]["coverage"] == 1.0

    # Discriminating fact: raw dumps include the constraint-forbidden
    # concept silently; the projection either suppresses it or guards
    # it behind a ⚠ counter-signal line.
    assert results["raw_history"]["violation_rate"] > 0.0
    assert results["projection"]["violation_rate"] == 0.0
    assert results["projection_counter"]["violation_rate"] == 0.0

    # The naive 1-hop neighborhood drags the forbidden neighbor in —
    # the failure mode the projection's inhibition exists to prevent.
    assert results["graph_neighbors"]["violation_rate"] > 0.0
