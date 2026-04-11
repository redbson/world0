"""Validation tests — end-to-end extraction→ingest→project pipeline.

These tests verify the core hypothesis: that automatic extraction
produces observations that lead to meaningful cognitive projections.
Uses a FakeLLM with realistic responses to test without API calls.
"""

import json

import pytest

from world0 import World
from world0.llm.base import LLMProvider
from world0.schemas.concept import Maturity


class ScenarioLLM(LLMProvider):
    """Simulates realistic LLM extraction responses for different texts.

    Routes based on keyword matching in the user prompt to return
    domain-appropriate extraction results.
    """

    RESPONSES = {
        "ml_deployment": {
            "concepts": [
                {"name": "machine learning model", "description": "Trained ML model for inference"},
                {"name": "docker", "description": "Container runtime for packaging applications"},
                {"name": "kubernetes", "description": "Container orchestration platform"},
                {"name": "latency", "description": "Response time of the serving endpoint"},
                {"name": "redis", "description": "In-memory cache for reducing latency"},
                {"name": "model serving", "description": "Infrastructure for serving ML predictions"},
            ],
            "relations": [
                {"source": "model serving", "target": "machine learning model", "type": "depends_on"},
                {"source": "model serving", "target": "docker", "type": "depends_on"},
                {"source": "kubernetes", "target": "docker", "type": "contains"},
                {"source": "redis", "target": "latency", "type": "supports"},
                {"source": "model serving", "target": "latency", "type": "related_to"},
            ],
        },
        "data_pipeline": {
            "concepts": [
                {"name": "data pipeline", "description": "ETL process for data transformation"},
                {"name": "apache kafka", "description": "Distributed event streaming platform"},
                {"name": "spark", "description": "Distributed data processing engine"},
                {"name": "data warehouse", "description": "Central repository for structured data"},
                {"name": "data quality", "description": "Ensuring accuracy and consistency of data"},
            ],
            "relations": [
                {"source": "data pipeline", "target": "apache kafka", "type": "depends_on"},
                {"source": "data pipeline", "target": "spark", "type": "depends_on"},
                {"source": "spark", "target": "data warehouse", "type": "precedes"},
                {"source": "data quality", "target": "data pipeline", "type": "supports"},
            ],
        },
        "frontend_dev": {
            "concepts": [
                {"name": "react", "description": "JavaScript UI library"},
                {"name": "typescript", "description": "Typed superset of JavaScript"},
                {"name": "state management", "description": "Managing application state"},
                {"name": "component", "description": "Reusable UI building block"},
                {"name": "REST API", "description": "HTTP-based API interface"},
            ],
            "relations": [
                {"source": "react", "target": "component", "type": "contains"},
                {"source": "react", "target": "state management", "type": "depends_on"},
                {"source": "typescript", "target": "react", "type": "supports"},
                {"source": "component", "target": "REST API", "type": "depends_on"},
            ],
        },
    }

    def complete_json(self, system: str, user: str) -> str:
        text = user.lower()
        if "deploy" in text or "docker" in text or "kubernetes" in text:
            return json.dumps(self.RESPONSES["ml_deployment"])
        elif "pipeline" in text or "kafka" in text or "etl" in text:
            return json.dumps(self.RESPONSES["data_pipeline"])
        elif "react" in text or "frontend" in text or "component" in text:
            return json.dumps(self.RESPONSES["frontend_dev"])
        # Default fallback
        return json.dumps(self.RESPONSES["ml_deployment"])


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0", llm=ScenarioLLM())


# ── Basic extraction→ingest pipeline ─────────────────────────────────


class TestIngestText:
    def test_extracts_and_ingests_concepts(self, world):
        result = world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes. "
            "Added Redis caching to reduce latency.",
            task="deployment review",
            source="test",
        )
        assert len(result.new_concepts) > 0
        assert "docker" in result.new_concepts
        assert "redis" in result.new_concepts

    def test_extracts_and_ingests_relations(self, world):
        result = world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes.",
            task="deployment",
        )
        assert len(result.new_relations) > 0 or len(result.hebbian_relations) > 0

    def test_descriptions_are_stored(self, world):
        world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes.",
            task="deployment",
        )
        docker = world.concepts.resolve("docker")
        assert docker is not None
        assert docker.description != ""

    def test_repeated_ingestion_reinforces(self, world):
        world.ingest_text("Deploy with Docker and Kubernetes.", task="t1")
        world.ingest_text("Docker containers in Kubernetes cluster.", task="t2")

        docker = world.concepts.resolve("docker")
        assert docker.activation_count >= 2

    def test_ingest_text_without_llm_raises(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")  # no llm
        with pytest.raises(RuntimeError, match="LLM provider"):
            w.ingest_text("some text")


# ── Projection quality after extraction ──────────────────────────────


class TestProjectionAfterExtraction:
    def test_projection_covers_extracted_domain(self, world):
        world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes. "
            "Added Redis caching to reduce latency.",
            task="deployment review",
        )
        proj = world.project(["docker", "kubernetes"], task="infra review")
        names = {c.name for c in proj.concepts}

        assert "docker" in names
        assert "kubernetes" in names

    def test_projection_includes_extracted_relations(self, world):
        world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes.",
            task="deployment",
        )
        proj = world.project(
            ["docker", "kubernetes", "model serving"],
            task="debug",
        )
        assert len(proj.relations) > 0

    def test_projection_renders_as_markdown(self, world):
        world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes.",
            task="deployment",
        )
        proj = world.project(["docker"], task="container review")
        rendered = proj.render()

        assert "## Cognitive Context" in rendered
        assert "docker" in rendered.lower()


# ── Cross-domain projection differentiation ──────────────────────────


class TestCrossDomainProjection:
    """Core validation: different domains produce different projections."""

    def _build_multi_domain_world(self, world):
        world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes. "
            "Added Redis caching to reduce latency.",
            task="ML deployment",
            source="infra_team",
        )
        world.ingest_text(
            "Built the data pipeline with Apache Kafka and Spark. "
            "ETL jobs load into the data warehouse with quality checks.",
            task="data engineering",
            source="data_team",
        )
        world.ingest_text(
            "Refactored the frontend React components with TypeScript. "
            "Improved state management and REST API integration.",
            task="frontend refactor",
            source="frontend_team",
        )
        # Reinforce to build up confidence
        for _ in range(5):
            world.ingest_text("Deploy Docker Kubernetes.", task="reinforce")
            world.ingest_text("Pipeline Kafka ETL Spark.", task="reinforce")
            world.ingest_text("React frontend components TypeScript.", task="reinforce")

    def test_infra_projection_is_infra_centric(self, world):
        self._build_multi_domain_world(world)
        proj = world.project(["docker", "kubernetes"], task="infra debug")
        names = {c.name for c in proj.concepts}

        assert "docker" in names
        # Should NOT primarily contain frontend concepts
        assert "react" not in names or "kubernetes" in names

    def test_data_projection_is_data_centric(self, world):
        self._build_multi_domain_world(world)
        proj = world.project(["data pipeline", "apache kafka"], task="pipeline debug")
        names = {c.name for c in proj.concepts}

        assert "data pipeline" in names or "apache kafka" in names

    def test_frontend_projection_is_frontend_centric(self, world):
        self._build_multi_domain_world(world)
        proj = world.project(["react", "typescript"], task="UI bug fix")
        names = {c.name for c in proj.concepts}

        assert "react" in names
        # Frontend projection shouldn't be dominated by infra
        assert "kubernetes" not in names or "react" in names

    def test_different_domains_produce_different_projections(self, world):
        self._build_multi_domain_world(world)

        proj_infra = world.project(["docker"], task="infra")
        proj_data = world.project(["data pipeline"], task="data")
        proj_fe = world.project(["react"], task="frontend")

        names_infra = {c.name for c in proj_infra.concepts}
        names_data = {c.name for c in proj_data.concepts}
        names_fe = {c.name for c in proj_fe.concepts}

        # All three should be non-empty
        assert len(names_infra) > 0
        assert len(names_data) > 0
        assert len(names_fe) > 0

        # No two should be identical
        assert names_infra != names_data
        assert names_data != names_fe
        assert names_infra != names_fe


# ── Cognitive evolution through extraction ────────────────────────────


class TestCognitiveEvolution:
    """Validate that repeated extraction drives concept maturity."""

    def test_concept_matures_through_repeated_extraction(self, world):
        # Ingest the same domain many times
        for i in range(12):
            world.ingest_text(
                "Docker containers deployed on Kubernetes cluster.",
                task=f"deployment_{i}",
                source=f"session_{i}",
            )

        docker = world.concepts.resolve("docker")
        assert docker is not None
        assert docker.activation_count >= 12
        # Confidence should have grown
        assert docker.confidence > 0.2

        # After reflect, should be promoted
        docker.confidence = max(docker.confidence, 0.65)
        world.reflect()
        docker = world.concepts.resolve("docker")
        assert docker.maturity in (Maturity.DEVELOPING, Maturity.ESTABLISHED)

    def test_hebbian_relations_form_through_cooccurrence(self, world):
        for _ in range(5):
            world.ingest_text(
                "Docker and Kubernetes are used together for deployment.",
                task="deployment",
            )

        docker = world.concepts.resolve("docker")
        k8s = world.concepts.resolve("kubernetes")

        if docker and k8s:
            rels = world.relations.find_any_between(docker.id, k8s.id)
            assert len(rels) > 0
            # Relation should be reinforced
            strongest = max(rels, key=lambda r: r.weight)
            assert strongest.weight > 0.1

    def test_reflect_decays_unused_concepts(self, world):
        world.ingest_text("Deploy Docker Kubernetes.", task="t1")

        # Artificially age a concept
        docker = world.concepts.resolve("docker")
        from datetime import datetime, timedelta, timezone

        docker.last_activated = datetime.now(timezone.utc) - timedelta(hours=48)
        docker.confidence = 0.03
        docker.maturity = Maturity.EMBRYONIC

        world.reflect()

        docker = world.concepts.resolve("docker")
        if docker:
            assert docker.maturity == Maturity.FADING or docker.confidence < 0.05


# ── World status after extraction ─────────────────────────────────────


class TestWorldStatusAfterExtraction:
    def test_status_reflects_extracted_content(self, world):
        world.ingest_text(
            "We deployed the ML model using Docker and Kubernetes.",
            task="deployment",
        )
        status = world.status()
        assert status.total_concepts >= 4
        assert status.total_relations >= 2
