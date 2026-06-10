"""Tests: ActivationConfig / ProjectionConfig and deterministic time.

Two contracts:
  1. The default config reproduces pre-refactor behavior exactly — the
     constants simply moved onto dataclasses.
  2. A custom config measurably changes output, and an injected ``now``
     makes temporal scoring reproducible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from world0 import Observation, World
from world0.dynamics.coefficients import (
    MIN_TASK_AFFINITY,
    MMR_LAMBDA,
    PROPAGATION_FLOOR,
    TASK_AFFINITY_BOOST,
    ActivationConfig,
    ProjectionConfig,
)


def _build(world: World) -> World:
    for _ in range(8):
        world.ingest(Observation(
            concepts=["service", "database", "cache", "queue"],
            relations=[
                ("service", "database", "depends_on"),
                ("service", "cache", "supports"),
                ("service", "queue", "supports"),
                ("database", "cache", "related_to"),
            ],
            task="architecture",
            source="t",
        ))
    return world


class TestDefaultsMatchConstants:
    def test_activation_config_defaults(self):
        cfg = ActivationConfig()
        assert cfg.task_affinity_boost == TASK_AFFINITY_BOOST
        assert cfg.min_task_affinity == MIN_TASK_AFFINITY
        assert cfg.propagation_floor == PROPAGATION_FLOOR

    def test_projection_config_defaults(self):
        cfg = ProjectionConfig()
        assert cfg.mmr_lambda == MMR_LAMBDA
        assert cfg.min_relation_signal == 0.0
        assert cfg.max_relations is None

    def test_explicit_default_config_matches_implicit(self, tmp_path):
        """World(config=default) behaves identically to World(no config).

        Concept ids are random per world, so compare the structure that
        the config governs: which concepts are selected and their scores
        keyed by name.
        """
        w_implicit = _build(World(store_path=tmp_path / "a"))
        w_explicit = _build(
            World(
                store_path=tmp_path / "b",
                activation_config=ActivationConfig(),
                projection_config=ProjectionConfig(),
            )
        )
        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        p1 = w_implicit.project(["service"], task="architecture", now=now)
        p2 = w_explicit.project(["service"], task="architecture", now=now)

        def by_name(proj):
            return {
                c.name: round(proj.activation_scores.get(c.id, 0.0), 9)
                for c in proj.concepts
            }

        assert by_name(p1) == by_name(p2)
        # Relation structure (by endpoint names + semantic type) matches.
        def rels(proj):
            names = {c.id: c.name for c in proj.concepts}
            return sorted(
                (names[r.source_id], names[r.target_id], r.semantic_relation)
                for r in proj.relations
            )

        assert rels(p1) == rels(p2)


class TestCustomConfigChangesOutput:
    def test_custom_min_relation_signal_filters(self, tmp_path):
        w = _build(
            World(
                store_path=tmp_path / ".world0",
                projection_config=ProjectionConfig(min_relation_signal=10.0),
            )
        )
        proj = w.project(["service", "database", "cache", "queue"])
        # Nothing clears an impossibly high signal bar.
        assert proj.relations == []

    def test_custom_max_relations_caps(self, tmp_path):
        w = _build(
            World(
                store_path=tmp_path / ".world0",
                projection_config=ProjectionConfig(max_relations=1),
            )
        )
        proj = w.project(["service", "database", "cache", "queue"])
        assert len(proj.relations) <= 1

    def test_propagation_floor_widens_horizon(self, tmp_path):
        # A higher min ratio keeps more distant concepts above threshold.
        tight = _build(
            World(
                store_path=tmp_path / "tight",
                activation_config=ActivationConfig(propagation_min_ratio=0.0),
            )
        )
        wide = _build(
            World(
                store_path=tmp_path / "wide",
                activation_config=ActivationConfig(propagation_min_ratio=0.5),
            )
        )
        now = datetime(2026, 6, 10, tzinfo=timezone.utc)
        p_tight = tight.project(["service"], max_depth=3, now=now)
        p_wide = wide.project(["service"], max_depth=3, now=now)
        assert len(p_wide.concepts) >= len(p_tight.concepts)


class TestDeterministicTime:
    def test_now_injection_is_reproducible(self, tmp_path):
        w = _build(World(store_path=tmp_path / ".world0"))
        now = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
        a = w.project(["service"], task="architecture", now=now)
        b = w.project(["service"], task="architecture", now=now)
        assert a.activation_scores == b.activation_scores

    def test_older_now_lowers_freshness(self, tmp_path):
        w = _build(World(store_path=tmp_path / ".world0"))
        svc = w.concepts.resolve("service")
        recent = svc.last_activated + timedelta(hours=1)
        ancient = svc.last_activated + timedelta(days=120)
        fresh = w.project(["service"], now=recent)
        stale = w.project(["service"], now=ancient)
        # Freshness only enters projection MMR scoring; the seed itself
        # keeps its confidence-based score, but distant concepts fade.
        assert sum(stale.activation_scores.values()) <= sum(
            fresh.activation_scores.values()
        ) + 1e-9
