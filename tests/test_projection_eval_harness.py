"""Smoke test: the offline projection-eval harness runs and yields metrics.

Guards against bit-rot in scripts/eval_projection_matrix.py — it must
build its deterministic world, score every case, and produce the
expected metric keys without any LLM or network access.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "eval_projection_matrix.py",
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("eval_proj_matrix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the @dataclass can resolve string
    # annotations (PEP 563) against the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harness_runs_offline():
    harness = _load_harness()
    with tempfile.TemporaryDirectory() as tmp:
        world = harness.build_world(os.path.join(tmp, ".world0"))
        rows = [
            harness.evaluate_case(world, case, None)
            for case in harness.CASES
        ]

    assert len(rows) == len(harness.CASES)
    for row in rows:
        for key in harness._METRIC_KEYS:
            assert key in row
            assert isinstance(row[key], float)
        assert "inhibition_ok" in row


def test_typo_seed_resolves_via_fuzzy():
    """The fuzzy-seed case must actually recover its concepts."""
    harness = _load_harness()
    with tempfile.TemporaryDirectory() as tmp:
        world = harness.build_world(os.path.join(tmp, ".world0"))
        typo = next(c for c in harness.CASES if c.name == "typo_seed")
        row = harness.evaluate_case(world, typo, None)
    # A typo'd seed should still resolve and surface relevant concepts.
    assert row["seed_rate"] == 1.0
    assert row["r@8"] > 0.0
