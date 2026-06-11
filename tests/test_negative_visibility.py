"""Negative visibility policy — counter-signals in projection.

Negative relations inhibit propagation (always), but constraint-class
negatives must additionally *surface* as warnings: inhibition removes
the repelled concept from the view, and the agent should see why the
path is closed instead of never seeing the path.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.schemas.context import Perspective


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    w.ingest(Observation(
        concepts=[
            "sync replication",
            "low latency",
            "high throughput",
            "async replication",
        ],
        relations=[
            ("sync replication", "low latency", "violates_constraint"),
            ("sync replication", "high throughput", "enables"),
            ("sync replication", "async replication", "conflict"),
        ],
        task="design storage",
        source="t",
    ))
    for _ in range(3):
        w.ingest(Observation(
            concepts=["sync replication", "high throughput"], source="t"
        ))
    return w


class TestCounterSignals:
    def test_violates_constraint_exposed_by_default(self, world):
        p = world.project(["sync replication"], task="design storage")
        pairs = {
            (s.semantic_relation, s.target_name) for s in p.counter_signals
        }
        assert ("violates_constraint", "low latency") in pairs

    def test_exposed_signal_points_at_suppressed_concept(self, world):
        # The inhibited endpoint is NOT in the selected set — that is
        # the point of exposing the warning.
        p = world.project(["sync replication"], task="design storage")
        selected = {c.name for c in p.concepts}
        assert "low latency" not in selected
        assert any(
            s.target_name == "low latency" for s in p.counter_signals
        )

    def test_conflict_is_conditional_suppressed_by_default(self, world):
        p = world.project(["sync replication"], task="design storage")
        assert not any(
            s.semantic_relation == "conflict" for s in p.counter_signals
        )

    def test_perspective_opts_conditional_relation_in(self, world):
        debug = Perspective(
            name="debug", negative_visibility={"conflict": "expose"}
        )
        p = world.project(["sync replication"], perspective=debug)
        assert any(
            s.semantic_relation == "conflict" for s in p.counter_signals
        )

    def test_perspective_can_suppress_default_expose(self, world):
        quiet = Perspective(
            name="quiet",
            negative_visibility={"violates_constraint": "suppress"},
        )
        p = world.project(["sync replication"], perspective=quiet)
        assert p.counter_signals == []

    def test_render_includes_warning_section(self, world):
        p = world.project(["sync replication"], task="design storage")
        out = p.render()
        assert "### Constraint Warnings" in out
        assert "⚠" in out

    def test_render_unchanged_without_signals(self, world):
        quiet = Perspective(
            name="quiet",
            negative_visibility={"violates_constraint": "suppress"},
        )
        p = world.project(["sync replication"], perspective=quiet)
        assert "### Constraint Warnings" not in p.render()


class TestGenericPressure:
    def test_typed_world_has_zero_pressure(self, world):
        p = world.project(["sync replication"], task="design storage")
        assert p.generic_pressure == 0.0

    def test_generic_edges_raise_pressure(self, tmp_path):
        w = World(store_path=tmp_path / ".w2")
        # Two concepts related only generically.
        w.ingest(Observation(
            concepts=["thing one", "thing two"],
            relations=[("thing one", "thing two", "related to")],
            source="t",
        ))
        for _ in range(3):
            w.ingest(Observation(concepts=["thing one", "thing two"], source="t"))
        p = w.project(["thing one"])
        if p.relations:
            assert p.generic_pressure > 0.0
