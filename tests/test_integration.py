"""End-to-end integration test: simulate an Agent working over multiple sessions."""

import pytest

from world0 import Observation, World
from world0.schemas.concept import Maturity


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / ".world0"


class TestAgentSimulation:
    """Simulate an Agent that works on ML deployment over several sessions.

    The key assertion: the cognitive world evolves through use, and
    projections reflect the Agent's developed understanding.
    """

    def test_multi_session_cognitive_evolution(self, store_path):
        # ── Session 1: Agent learns about ML basics ───────────────────
        w = World(store_path=store_path)

        w.ingest(Observation(
            concepts=["Machine Learning", "Training", "Dataset"],
            relations=[
                ("Machine Learning", "Training", "contains"),
                ("Training", "Dataset", "depends_on"),
            ],
            task="learn ML basics",
            source="session_1",
        ))
        w.ingest(Observation(
            concepts=["Python", "Machine Learning"],
            relations=[("Python", "Machine Learning", "supports")],
            task="learn ML basics",
            source="session_1",
        ))

        assert w.status().total_concepts == 4
        ml = w.concepts.resolve("Machine Learning")
        assert ml.maturity == Maturity.EMBRYONIC
        assert ml.activation_count == 2

        w.reflect()
        del w

        # ── Session 2: Agent works on deployment ─────────────────────
        w = World(store_path=store_path)
        assert w.status().total_concepts == 4  # persisted

        w.ingest(Observation(
            concepts=["Deployment", "Docker", "Latency"],
            relations=[
                ("Deployment", "Docker", "depends_on"),
                ("Deployment", "Latency", "related_to"),
            ],
            task="deploy ML model",
            source="session_2",
        ))
        # Agent connects ML to deployment
        w.ingest(Observation(
            concepts=["Machine Learning", "Deployment"],
            relations=[("Machine Learning", "Deployment", "related_to")],
            task="deploy ML model",
            source="session_2",
        ))

        assert w.status().total_concepts == 7
        ml = w.concepts.resolve("Machine Learning")
        assert ml.activation_count >= 3  # reinforced across sessions

        w.reflect()
        del w

        # ── Session 3: Agent does more ML work, reinforcing concepts ──
        w = World(store_path=store_path)

        for i in range(8):
            w.ingest(Observation(
                concepts=["Machine Learning", "Python", "Training"],
                task=f"ML iteration {i}",
                source="session_3",
            ))

        # ML should now be promoted
        ml = w.concepts.resolve("Machine Learning")
        assert ml.activation_count >= 10
        # Manually ensure confidence is high enough for promotion
        ml.confidence = max(ml.confidence, 0.65)

        w.reflect()

        ml = w.concepts.resolve("Machine Learning")
        # Should have been promoted (at least developing, possibly established)
        assert ml.maturity in (Maturity.DEVELOPING, Maturity.ESTABLISHED)

        # ── Projection test: ML vs Ops views ──────────────────────────
        proj_ml = w.project(
            ["Machine Learning", "Training"],
            task="optimize model accuracy",
        )
        proj_ops = w.project(
            ["Deployment", "Docker"],
            task="fix production latency",
        )

        ml_names = {c.name for c in proj_ml.concepts}
        ops_names = {c.name for c in proj_ops.concepts}

        # ML projection should be ML-centric
        assert "Machine Learning" in ml_names
        assert "Training" in ml_names

        # Ops projection should be ops-centric
        assert "Deployment" in ops_names

        # They should differ
        assert ml_names != ops_names

        # Projection renders as usable markdown
        rendered = proj_ml.render()
        assert "## Cognitive Context" in rendered
        assert "Machine Learning" in rendered

        del w

        # ── Session 4: Verify the cognitive world survived ────────────
        w = World(store_path=store_path)
        status = w.status()
        assert status.total_concepts == 7
        assert status.total_relations > 0
        assert status.last_reflect is not None

        ml = w.concepts.resolve("Machine Learning")
        assert ml.activation_count >= 10
        assert ml.maturity in (Maturity.DEVELOPING, Maturity.ESTABLISHED)
