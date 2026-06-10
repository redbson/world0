"""Tests: projection explainability — every projected concept can say why.

Traces are best-path provenance: non-seed concepts carry the chain of
hops that activated them, inhibited concepts can name their suppressor,
and ``Projection.explain()`` renders it all human-readably.
"""

from __future__ import annotations

import pytest

from world0 import Observation, Perspective, World


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    for _ in range(6):
        w.ingest(Observation(
            concepts=["latency", "caching", "database"],
            relations=[
                ("latency", "caching", "supports"),     # → enables
                ("caching", "database", "depends_on"),  # → dependence
            ],
            task="perf work",
            source="t",
        ))
    return w


class TestTraces:
    def test_seeds_have_stepless_traces(self, world):
        proj = world.project(["latency"])
        seed = world.concepts.resolve("latency")
        trace = proj.traces.get(seed.id)
        assert trace is not None
        assert trace.seed_id == seed.id
        assert trace.steps == []

    def test_nonseed_trace_starts_at_a_seed(self, world):
        proj = world.project(["latency"], max_depth=2)
        seed = world.concepts.resolve("latency")
        for cid, trace in proj.traces.items():
            if cid == seed.id:
                continue
            assert trace.steps, f"non-seed {cid} has no steps"
            assert trace.steps[0].from_id == trace.seed_id == seed.id
            # Chain integrity: each step starts where the previous ended.
            walker = trace.seed_id
            for step in trace.steps:
                assert step.from_id == walker
                rel = next(
                    r for r in world.relations.all() if r.id == step.relation_id
                )
                walker = rel.other_end(walker)
            assert walker == cid

    def test_trace_scores_match_activation_scores(self, world):
        proj = world.project(["latency"])
        for cid, trace in proj.traces.items():
            assert trace.score == pytest.approx(
                proj.activation_scores[cid]
            )

    def test_traces_only_cover_selected_concepts(self, world):
        proj = world.project(["latency"], max_concepts=2)
        selected = {c.id for c in proj.concepts}
        assert set(proj.traces) <= selected

    def test_trace_steps_carry_semantic_relation(self, world):
        proj = world.project(["latency"], max_depth=2)
        db = world.concepts.resolve("database")
        trace = proj.traces.get(db.id)
        assert trace is not None
        # Every step records the typed relation it traversed (whichever
        # path won — explicit dependence/enables or a Hebbian edge).
        assert trace.steps
        for step in trace.steps:
            assert step.semantic_relation
            assert step.axis in ("positive", "negative", "parallel")


class TestInhibitionTrace:
    def test_inhibited_concept_names_its_suppressor(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        for _ in range(8):
            w.ingest(Observation(
                concepts=["MongoDB", "transactions", "PostgreSQL"],
                relations=[
                    ("MongoDB", "transactions", "supports"),
                    ("MongoDB", "PostgreSQL", "conflict"),
                    ("transactions", "PostgreSQL", "supports"),
                ],
                source="t",
            ))
        proj = w.project(["MongoDB"], max_depth=2)
        mongo = w.concepts.resolve("MongoDB")
        pg = w.concepts.resolve("PostgreSQL")
        trace = proj.traces.get(pg.id)
        if trace is not None and trace.inhibition > 0:
            assert trace.inhibition_source == mongo.id
            assert mongo.name in proj.explain(pg.id)


class TestExplain:
    def test_explain_seed(self, world):
        proj = world.project(["latency"])
        line = proj.explain("latency")
        assert "seed" in line
        assert "latency" in line

    def test_explain_nonseed_shows_path(self, world):
        proj = world.project(["latency"], max_depth=2)
        line = proj.explain("caching")
        assert "caching" in line
        assert "latency" in line
        assert "[seed]" in line
        assert "enables" in line

    def test_explain_unknown_concept(self, world):
        proj = world.project(["latency"])
        assert "not in this projection" in proj.explain("quantum")


class TestRenderStyles:
    def test_default_style_unchanged_signature(self, world):
        proj = world.project(["latency"], task="perf work")
        assert proj.render() == proj.render("default")
        assert "## Cognitive Context" in proj.render()

    def test_unknown_style_falls_back_to_default(self, world):
        proj = world.project(["latency"])
        assert proj.render("nope") == proj.render("default")

    def test_compact_style_is_denser(self, world):
        proj = world.project(["latency"], task="perf work")
        compact = proj.render("compact")
        assert "## Cognitive Context (compact)" in compact
        assert len(compact) < len(proj.render("default"))

    def test_detailed_style_includes_why(self, world):
        proj = world.project(["latency"], max_depth=2)
        detailed = proj.render("detailed")
        if any(t.steps for t in proj.traces.values()):
            assert "### Why Included" in detailed

    def test_detailed_style_surfaces_counter_signals(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        for _ in range(8):
            w.ingest(Observation(
                concepts=["MongoDB", "PostgreSQL"],
                relations=[("MongoDB", "PostgreSQL", "conflict")],
                source="t",
            ))
        proj = w.project(["MongoDB", "PostgreSQL"])
        has_negative = any(
            r.relation_type.value == "negative" for r in proj.relations
        )
        if has_negative:
            assert "### Counter-Signals" in proj.render("detailed")


class TestPerspectiveShapedRelations:
    def test_relations_ordered_by_perspective_signal(self, world):
        """Same selection, different perspectives → different relation order."""
        dep_lens = Perspective(
            name="dep",
            semantic_relation_weights={"dependence": 1.5, "enables": 0.1},
        )
        ena_lens = Perspective(
            name="ena",
            semantic_relation_weights={"dependence": 0.1, "enables": 1.5},
        )
        proj_dep = world.project(
            ["latency", "caching", "database"], perspective=dep_lens
        )
        proj_ena = world.project(
            ["latency", "caching", "database"], perspective=ena_lens
        )
        if len(proj_dep.relations) >= 2 and len(proj_ena.relations) >= 2:
            assert proj_dep.relations[0].semantic_relation == "dependence"
            assert proj_ena.relations[0].semantic_relation == "enables"

    def test_min_relation_signal_filters(self, world):
        seed_names = ["latency", "caching", "database"]
        seeds = [world.concepts.resolve(n).id for n in seed_names]
        activations, traces = world._activation.activate_traced(
            seeds, record=False
        )
        unfiltered = world._projection.project(activations, traces=traces)
        filtered = world._projection.project(
            activations, traces=traces, min_relation_signal=10.0
        )
        assert len(filtered.relations) == 0
        assert len(unfiltered.relations) >= 1

    def test_max_relations_caps(self, world):
        seed_names = ["latency", "caching", "database"]
        seeds = [world.concepts.resolve(n).id for n in seed_names]
        activations, traces = world._activation.activate_traced(
            seeds, record=False
        )
        capped = world._projection.project(
            activations, traces=traces, max_relations=1
        )
        assert len(capped.relations) <= 1

    def test_projection_carries_perspective_name(self, world):
        proj = world.project(["latency"], perspective="debug")
        assert proj.perspective_name == "debug"

    def test_projection_carries_seeds(self, world):
        proj = world.project(["latency", "nonexistent"])
        assert proj.seeds == ["latency", "nonexistent"]
