"""Tests: projection — same world, different seeds/tasks → different views."""

import pytest

from world0 import ConceptCandidate, Observation, World


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")

    # Build a small cognitive world through observations
    w.ingest(Observation(
        concepts=["Python", "Machine Learning", "REST API"],
        relations=[
            ("Python", "Machine Learning", "supports"),
            ("Python", "REST API", "supports"),
        ],
        source="setup",
        task="initial",
    ))
    w.ingest(Observation(
        concepts=["Neural Network", "Machine Learning"],
        relations=[("Machine Learning", "Neural Network", "contains")],
        source="setup",
        task="initial",
    ))
    w.ingest(Observation(
        concepts=["Deployment", "REST API", "Monitoring"],
        relations=[
            ("Deployment", "REST API", "contains"),
            ("Deployment", "Monitoring", "depends_on"),
        ],
        source="setup",
        task="initial",
    ))

    # Reinforce concepts heavily to build up confidence
    for _ in range(15):
        w.ingest(Observation(concepts=["Python", "Machine Learning"], source="reinforcement"))
    for _ in range(15):
        w.ingest(Observation(concepts=["Deployment", "REST API", "Monitoring"], source="reinforcement"))
    for _ in range(10):
        w.ingest(Observation(concepts=["Neural Network"], source="reinforcement"))

    return w


class TestProjectionContent:
    def test_projection_contains_seeds(self, world):
        proj = world.project(["Python"])
        names = {c.name for c in proj.concepts}
        assert "Python" in names

    def test_projection_spreads_to_neighbors(self, world):
        proj = world.project(["Python"])
        names = {c.name for c in proj.concepts}
        # Python is connected to ML and REST API
        assert "Machine Learning" in names or "REST API" in names

    def test_projection_includes_internal_relations(self, world):
        proj = world.project(["Python", "Machine Learning"])
        concept_ids = {c.id for c in proj.concepts}
        for rel in proj.relations:
            assert rel.source_id in concept_ids
            assert rel.target_id in concept_ids


class TestProjectionDiffers:
    def test_different_seeds_different_projections(self, world):
        """Different seed concepts → different cognitive views."""
        proj_ml = world.project(["Machine Learning"], task="ML research")
        proj_ops = world.project(["Deployment"], task="DevOps")

        ml_names = {c.name for c in proj_ml.concepts}
        ops_names = {c.name for c in proj_ops.concepts}

        # ML projection should have Neural Network
        assert "Neural Network" in ml_names or "Python" in ml_names
        # Ops projection should have Monitoring
        assert "Monitoring" in ops_names or "REST API" in ops_names

        # They should not be identical
        assert ml_names != ops_names


class TestProjectionRender:
    def test_render_produces_markdown(self, world):
        proj = world.project(["Python", "Machine Learning"], task="ML work")
        rendered = proj.render()
        assert "## Cognitive Context" in rendered
        assert "Python" in rendered

    def test_render_includes_task(self, world):
        proj = world.project(["Python"], task="debug production")
        rendered = proj.render()
        assert "debug production" in rendered

    def test_empty_projection_renders(self, world):
        proj = world.project(["Nonexistent"])
        rendered = proj.render()
        assert "## Cognitive Context" in rendered

    def test_render_uses_concept_representation(self, world):
        world.ingest(Observation(
            concept_candidates=[
                ConceptCandidate(
                    uid="c1",
                    name="Apple",
                    sense="technology company",
                    kind="entity",
                    domain="technology",
                )
            ],
            task="identity",
        ))
        node = next(c for c in world.concepts.all() if c.name == "Apple")
        proj = world.project([node.id])
        rendered = proj.render()
        assert f"apple.technology-company.{node.id}" in rendered


class TestProjectionScoring:
    def test_seeds_have_highest_activation(self, world):
        proj = world.project(["Python"])
        py = world.concepts.resolve("Python")
        py_score = proj.activation_scores.get(py.id, 0)
        for c in proj.concepts:
            if c.id != py.id:
                assert proj.activation_scores.get(c.id, 0) <= py_score


class TestDefaultRenderSnapshot:
    """The default render shape is a stable contract for prompt injection.

    Renderer styles (compact/detailed) are additive — the default output
    structure must not drift when they evolve.
    """

    def test_default_render_structure(self, world):
        proj = world.project(["Python", "Machine Learning"], task="ML work")
        rendered = proj.render()
        lines = rendered.split("\n")

        # Exact header and section ordering.
        assert lines[0] == "## Cognitive Context"
        assert lines[1] == ""
        section_order = [
            ln for ln in lines if ln.startswith("### ")
        ]
        allowed = [
            "### Core Understanding",
            "### Active Concepts",
            "### Emerging Concepts",
            "### Key Relations",
            "### Task Context",
        ]
        assert section_order == [s for s in allowed if s in section_order]

        # Concept lines keep the historical format.
        concept_lines = [ln for ln in lines if ln.startswith("- **")]
        assert concept_lines, "no concept lines rendered"
        for ln in concept_lines:
            assert "confidence: " in ln

        # Relation lines keep the historical format.
        rel_lines = [
            ln for ln in lines
            if ln.startswith("- ") and "→" in ln and "structural" in ln
        ]
        for ln in rel_lines:
            assert "propagation: " in ln
            assert "reinforced " in ln

        # Task context footer.
        assert "Concepts activated for: ML work" in rendered

    def test_default_render_ends_with_trailing_newline_block(self, world):
        proj = world.project(["Python"], task="x")
        rendered = proj.render()
        # Historical shape: every section appends "", so render ends
        # with an empty line after the task context.
        assert rendered.endswith("Concepts activated for: x\n")
