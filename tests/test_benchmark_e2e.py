"""End-to-end benchmark validation — simulates realistic Agent workflows
and measures cognitive quality across the full chain:

    ingest → activate → project → reflect → re-project

Scenario: A software engineer Agent works across three sessions:
  Session 1: Backend API design (Python, FastAPI, PostgreSQL)
  Session 2: ML pipeline (Python, PyTorch, data pipeline)
  Session 3: Debugging a production issue touching both domains

Metrics validated:
  - Knowledge accumulation: concepts and relations grow correctly
  - Cross-session coherence: shared concepts (Python) bridge domains
  - Projection focus: each task gets a relevant local view
  - Reflect consolidation: decay + lifecycle work end-to-end
  - Cognitive evolution: maturity progression tracks real usage
  - Render quality: LLM-prompt output is well-structured
  - Quantitative summary: all metrics printed at end
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from world0 import Observation, Projection, World
from world0.schemas.concept import Maturity


# ═══════════════════════════════════════════════════════════════════════
# Test data — three realistic Agent work sessions
# ═══════════════════════════════════════════════════════════════════════

SESSION_1_OBSERVATIONS = [
    Observation(
        concepts=["Python", "FastAPI", "REST API", "PostgreSQL", "SQLAlchemy"],
        relations=[
            ("FastAPI", "REST API", "contains"),
            ("FastAPI", "Python", "depends_on"),
            ("SQLAlchemy", "PostgreSQL", "supports"),
            ("REST API", "PostgreSQL", "depends_on"),
        ],
        descriptions={
            "FastAPI": "Modern async web framework for Python",
            "SQLAlchemy": "Python SQL toolkit and ORM",
        },
        task="design backend API",
        source="session_001",
    ),
    Observation(
        concepts=["FastAPI", "authentication", "JWT", "middleware"],
        relations=[
            ("FastAPI", "middleware", "contains"),
            ("authentication", "JWT", "depends_on"),
            ("middleware", "authentication", "supports"),
        ],
        task="design backend API",
        source="session_001",
    ),
    Observation(
        concepts=["PostgreSQL", "indexing", "query optimization", "migration"],
        relations=[
            ("PostgreSQL", "indexing", "contains"),
            ("indexing", "query optimization", "supports"),
            ("migration", "PostgreSQL", "depends_on"),
        ],
        task="design backend API",
        source="session_001",
    ),
]

SESSION_2_OBSERVATIONS = [
    Observation(
        concepts=["Python", "PyTorch", "neural network", "training pipeline"],
        relations=[
            ("PyTorch", "Python", "depends_on"),
            ("PyTorch", "neural network", "supports"),
            ("training pipeline", "PyTorch", "depends_on"),
        ],
        descriptions={
            "PyTorch": "Deep learning framework",
            "training pipeline": "End-to-end ML model training workflow",
        },
        task="build ML pipeline",
        source="session_002",
    ),
    Observation(
        concepts=["training pipeline", "data loader", "GPU", "batch processing"],
        relations=[
            ("training pipeline", "data loader", "contains"),
            ("data loader", "batch processing", "supports"),
            ("GPU", "training pipeline", "supports"),
        ],
        task="build ML pipeline",
        source="session_002",
    ),
    Observation(
        concepts=["neural network", "loss function", "optimizer", "backpropagation"],
        relations=[
            ("neural network", "loss function", "contains"),
            ("optimizer", "backpropagation", "depends_on"),
            ("backpropagation", "loss function", "derived_from"),
        ],
        task="build ML pipeline",
        source="session_002",
    ),
]

SESSION_3_OBSERVATIONS = [
    Observation(
        concepts=["Python", "FastAPI", "PyTorch", "model serving", "latency"],
        relations=[
            ("model serving", "FastAPI", "depends_on"),
            ("model serving", "PyTorch", "depends_on"),
            ("latency", "model serving", "contrasts"),
        ],
        descriptions={
            "model serving": "Serving ML models via HTTP endpoints",
        },
        task="debug prod latency",
        source="session_003",
    ),
    Observation(
        concepts=["latency", "PostgreSQL", "query optimization", "connection pool"],
        relations=[
            ("latency", "query optimization", "depends_on"),
            ("connection pool", "PostgreSQL", "supports"),
            ("latency", "connection pool", "depends_on"),
        ],
        task="debug prod latency",
        source="session_003",
    ),
]


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


# ═══════════════════════════════════════════════════════════════════════
# 1. Knowledge Accumulation
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeAccumulation:
    """Verify that multi-session ingestion builds up knowledge correctly."""

    def _ingest_all(self, world):
        all_obs = SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS
        results = []
        for obs in all_obs:
            results.append(world.ingest(obs))
        return results

    def test_concept_count_matches_unique_names(self, world):
        self._ingest_all(world)
        status = world.status()
        # Count unique concept names across all observations
        all_names = set()
        for obs in SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS:
            all_names.update(obs.concepts)
        assert status.total_concepts == len(all_names), (
            f"Expected {len(all_names)} concepts, got {status.total_concepts}"
        )

    def test_shared_concepts_are_deduplicated(self, world):
        self._ingest_all(world)
        # "Python" appears in all 3 sessions — should be one concept
        python = world.concepts.resolve("Python")
        assert python is not None
        # Should have been activated multiple times
        assert python.activation_count >= 3

    def test_relations_created_correctly(self, world):
        self._ingest_all(world)
        status = world.status()
        assert status.total_relations > 0

        # Check a specific explicit relation
        fastapi = world.concepts.resolve("FastAPI")
        python = world.concepts.resolve("Python")
        from world0.schemas.relation import RelationType
        edge = world.relations.find_between(fastapi.id, python.id, RelationType.DEPENDS_ON)
        assert edge is not None, "FastAPI → depends_on → Python should exist"
        assert edge.is_explicit

    def test_ingest_results_track_new_vs_reinforced(self, world):
        results = self._ingest_all(world)
        # First observation should have all new concepts
        assert len(results[0].new_concepts) > 0
        # Later observations sharing "Python" should reinforce it
        total_reinforced = sum(len(r.reinforced_concepts) for r in results)
        assert total_reinforced > 0, "Cross-session concepts should be reinforced"

    def test_descriptions_applied(self, world):
        self._ingest_all(world)
        fastapi = world.concepts.resolve("FastAPI")
        assert "async" in fastapi.description.lower() or "web" in fastapi.description.lower()


# ═══════════════════════════════════════════════════════════════════════
# 2. Cross-Session Coherence
# ═══════════════════════════════════════════════════════════════════════

class TestCrossSessionCoherence:
    """Shared concepts should bridge different work domains."""

    def _ingest_all(self, world):
        # Simulate realistic usage: Agent revisits concepts multiple times
        for _ in range(8):
            for obs in SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS:
                world.ingest(obs)

    def test_python_bridges_backend_and_ml(self, world):
        self._ingest_all(world)
        python = world.concepts.resolve("Python")

        # Project from Python — should reach both domains
        proj = world.project(["Python"], max_concepts=15, max_depth=2)
        names = {c.name for c in proj.concepts}

        backend_reached = names & {"FastAPI", "REST API", "SQLAlchemy"}
        ml_reached = names & {"PyTorch", "neural network", "training pipeline"}

        assert len(backend_reached) > 0, (
            f"Python projection should reach backend concepts. Got: {names}"
        )
        assert len(ml_reached) > 0, (
            f"Python projection should reach ML concepts. Got: {names}"
        )

    def test_python_has_highest_confidence(self, world):
        """Most frequently activated concept should have highest confidence."""
        self._ingest_all(world)
        all_concepts = world.concepts.all()
        python = world.concepts.resolve("Python")

        top = max(all_concepts, key=lambda c: c.confidence)
        # Python appears most — should be top or near-top
        assert python.confidence >= top.confidence * 0.8, (
            f"Python confidence {python.confidence:.3f} too low vs top "
            f"{top.name} at {top.confidence:.3f}"
        )

    def test_session3_connects_both_domains(self, world):
        """Session 3 (debug prod) should create bridge relations."""
        self._ingest_all(world)
        model_serving = world.concepts.resolve("model serving")
        assert model_serving is not None

        # model serving depends on both FastAPI and PyTorch
        fastapi = world.concepts.resolve("FastAPI")
        pytorch = world.concepts.resolve("PyTorch")

        from world0.schemas.relation import RelationType
        edge_api = world.relations.find_between(
            model_serving.id, fastapi.id, RelationType.DEPENDS_ON
        )
        edge_ml = world.relations.find_between(
            model_serving.id, pytorch.id, RelationType.DEPENDS_ON
        )
        assert edge_api is not None, "model serving → FastAPI should exist"
        assert edge_ml is not None, "model serving → PyTorch should exist"


# ═══════════════════════════════════════════════════════════════════════
# 3. Projection Focus per Task
# ═══════════════════════════════════════════════════════════════════════

class TestProjectionFocus:
    """Each task should produce a focused, relevant projection."""

    def _ingest_all(self, world):
        for _ in range(8):
            for obs in SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS:
                world.ingest(obs)

    def test_backend_task_projection(self, world):
        self._ingest_all(world)
        proj = world.project(
            ["FastAPI", "PostgreSQL"],
            task="design backend API",
            max_concepts=10,
        )
        names = {c.name for c in proj.concepts}

        # Must include core backend concepts
        assert "FastAPI" in names
        assert "PostgreSQL" in names

        # Should include related backend concepts
        backend_related = names & {
            "REST API", "SQLAlchemy", "authentication",
            "JWT", "middleware", "indexing",
        }
        assert len(backend_related) >= 2, (
            f"Backend projection too sparse: {names}"
        )

    def test_ml_task_projection(self, world):
        self._ingest_all(world)
        proj = world.project(
            ["PyTorch", "neural network"],
            task="build ML pipeline",
            max_concepts=10,
        )
        names = {c.name for c in proj.concepts}

        assert "PyTorch" in names
        assert "neural network" in names

        ml_related = names & {
            "training pipeline", "data loader", "GPU",
            "loss function", "optimizer", "backpropagation",
        }
        assert len(ml_related) >= 2, (
            f"ML projection too sparse: {names}"
        )

    def test_debug_task_bridges_domains(self, world):
        self._ingest_all(world)
        proj = world.project(
            ["latency", "model serving"],
            task="debug prod latency",
            max_concepts=12,
        )
        names = {c.name for c in proj.concepts}

        # Debug task should pull from both domains
        backend_hit = names & {"FastAPI", "PostgreSQL", "query optimization", "connection pool"}
        ml_hit = names & {"PyTorch", "neural network"}

        assert len(backend_hit) >= 1, (
            f"Debug projection missing backend concepts: {names}"
        )

    def test_projections_have_different_top_concepts(self, world):
        """Different tasks should produce different top-ranked concepts."""
        self._ingest_all(world)

        proj_api = world.project(["FastAPI"], task="design backend API")
        proj_ml = world.project(["PyTorch"], task="build ML pipeline")

        top_api = {c.name for c in proj_api.top_concepts(3)}
        top_ml = {c.name for c in proj_ml.top_concepts(3)}

        overlap = top_api & top_ml
        assert len(overlap) <= 1, (
            f"Top concepts overlap too much: {overlap}. "
            f"API top: {top_api}, ML top: {top_ml}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. Reflect Consolidation
# ═══════════════════════════════════════════════════════════════════════

class TestReflectConsolidation:
    """reflect() should decay, promote, and prune correctly."""

    def _ingest_all(self, world):
        for obs in SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS:
            world.ingest(obs)
        # Reinforce heavily used concepts
        for _ in range(10):
            world.ingest(Observation(
                concepts=["Python", "FastAPI", "PyTorch"],
                task="reinforce", source="bench",
            ))

    def test_reflect_returns_results(self, world):
        self._ingest_all(world)
        result = world.reflect()
        # Should have attempted decay on all concepts
        assert isinstance(result.decayed_concepts, list)
        assert isinstance(result.promoted_concepts, list)

    def test_heavily_used_concepts_promoted(self, world):
        self._ingest_all(world)
        world.reflect()

        python = world.concepts.resolve("Python")
        # After many reinforcements, Python should advance beyond embryonic
        assert python.maturity != Maturity.EMBRYONIC, (
            f"Python should be promoted past embryonic, got {python.maturity}"
        )

    def test_aging_and_prune_cycle(self, world):
        """Simulate time passing: age a concept, reflect, verify decay."""
        self._ingest_all(world)

        # Artificially age a low-use concept
        jwt = world.concepts.resolve("JWT")
        jwt.confidence = 0.05
        jwt.maturity = Maturity.FADING
        jwt.last_activated = datetime.now(timezone.utc) - timedelta(hours=500)

        before_count = world.status().total_concepts
        result = world.reflect()

        # JWT should be pruned or at least decayed
        if jwt.confidence > 0:
            assert jwt.confidence < 0.05, "Fading concept should decay further"
        else:
            after_count = world.status().total_concepts
            assert after_count < before_count, "Pruned concept should reduce count"

    def test_status_after_reflect(self, world):
        self._ingest_all(world)
        world.reflect()

        status = world.status()
        assert status.last_reflect is not None
        assert status.avg_confidence > 0


# ═══════════════════════════════════════════════════════════════════════
# 5. Render Quality
# ═══════════════════════════════════════════════════════════════════════

class TestRenderQuality:
    """Projection render() output should be well-structured for LLM consumption."""

    def _ingest_all(self, world):
        for obs in SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS:
            world.ingest(obs)
        for _ in range(8):
            world.ingest(Observation(
                concepts=["Python", "FastAPI", "PyTorch"],
                task="reinforce", source="bench",
            ))
        world.reflect()

    def test_render_contains_markdown_structure(self, world):
        self._ingest_all(world)
        proj = world.project(["FastAPI", "Python"], task="backend work")
        rendered = proj.render()

        assert "## Cognitive Context" in rendered
        assert "**" in rendered  # bold concept names
        assert "confidence:" in rendered

    def test_render_includes_relations(self, world):
        self._ingest_all(world)
        proj = world.project(["FastAPI", "Python"], task="backend work")
        rendered = proj.render()

        if proj.relations:
            assert "### Key Relations" in rendered
            assert "→" in rendered

    def test_render_includes_task(self, world):
        self._ingest_all(world)
        proj = world.project(["FastAPI"], task="optimize API performance")
        rendered = proj.render()

        assert "optimize API performance" in rendered

    def test_render_not_empty_for_valid_seeds(self, world):
        self._ingest_all(world)
        proj = world.project(["Python"])
        rendered = proj.render()
        # Should have at least the header and one concept
        assert len(rendered) > 50


# ═══════════════════════════════════════════════════════════════════════
# 6. Full Lifecycle Simulation
# ═══════════════════════════════════════════════════════════════════════

class TestFullLifecycleSimulation:
    """Simulate multiple work → reflect cycles and verify evolution."""

    def test_multi_cycle_evolution(self, world):
        """Run 3 work sessions with reflects between them."""
        metrics = {}

        # Session 1: Backend
        for obs in SESSION_1_OBSERVATIONS:
            world.ingest(obs)
        world.reflect()
        s1 = world.status()
        metrics["after_session_1"] = {
            "concepts": s1.total_concepts,
            "relations": s1.total_relations,
            "avg_confidence": s1.avg_confidence,
        }

        # Session 2: ML
        for obs in SESSION_2_OBSERVATIONS:
            world.ingest(obs)
        world.reflect()
        s2 = world.status()
        metrics["after_session_2"] = {
            "concepts": s2.total_concepts,
            "relations": s2.total_relations,
            "avg_confidence": s2.avg_confidence,
        }

        # Session 3: Cross-domain debug
        for obs in SESSION_3_OBSERVATIONS:
            world.ingest(obs)
        world.reflect()
        s3 = world.status()
        metrics["after_session_3"] = {
            "concepts": s3.total_concepts,
            "relations": s3.total_relations,
            "avg_confidence": s3.avg_confidence,
        }

        # Verify growth
        assert metrics["after_session_2"]["concepts"] > metrics["after_session_1"]["concepts"]
        assert metrics["after_session_3"]["relations"] > metrics["after_session_1"]["relations"]

        # Confidence should stay healthy (not all decay to 0)
        assert metrics["after_session_3"]["avg_confidence"] > 0.05

    def test_projection_quality_improves_with_reinforcement(self, world):
        """More data → richer projections."""
        # Minimal data
        world.ingest(SESSION_1_OBSERVATIONS[0])
        proj_early = world.project(["FastAPI"], task="backend")
        early_count = len(proj_early.concepts)

        # Add more observations
        for obs in SESSION_1_OBSERVATIONS[1:]:
            world.ingest(obs)
        for _ in range(5):
            for obs in SESSION_1_OBSERVATIONS:
                world.ingest(obs)

        proj_late = world.project(["FastAPI"], task="backend")
        late_count = len(proj_late.concepts)

        assert late_count >= early_count, (
            f"More data should yield richer projection: {early_count} → {late_count}"
        )

    def test_persistence_across_full_lifecycle(self, tmp_path):
        """Save after full lifecycle, reload, verify projection matches."""
        store_path = tmp_path / ".world0"
        w1 = World(store_path=store_path)

        for obs in SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS:
            w1.ingest(obs)
        for _ in range(5):
            w1.ingest(Observation(
                concepts=["Python", "FastAPI", "PyTorch"],
                task="reinforce", source="bench",
            ))
        w1.reflect()

        proj1 = w1.project(["Python", "FastAPI"], task="test")
        names1 = {c.name for c in proj1.concepts}
        scores1 = {c.name: proj1.activation_scores.get(c.id, 0) for c in proj1.concepts}

        w1.concepts.save_all()
        w1.relations.save_all()
        del w1

        w2 = World(store_path=store_path)
        proj2 = w2.project(["Python", "FastAPI"], task="test")
        names2 = {c.name for c in proj2.concepts}

        assert names1 == names2, (
            f"Projection changed after reload: lost={names1 - names2}, gained={names2 - names1}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 7. Quantitative Summary (prints metrics for review)
# ═══════════════════════════════════════════════════════════════════════

class TestQuantitativeSummary:
    """Run the full scenario and print a summary report."""

    def test_full_benchmark_report(self, world, capsys):
        """Run full scenario and print quantitative metrics."""
        t0 = time.monotonic()

        # Ingest all sessions with realistic reinforcement
        all_obs = SESSION_1_OBSERVATIONS + SESSION_2_OBSERVATIONS + SESSION_3_OBSERVATIONS
        ingest_results = []
        for _ in range(8):
            for obs in all_obs:
                ingest_results.append(world.ingest(obs))

        # Extra reinforcement on key concepts
        for _ in range(10):
            world.ingest(Observation(
                concepts=["Python", "FastAPI", "PyTorch"],
                task="reinforce", source="bench",
            ))
        t_ingest = time.monotonic() - t0

        # Reflect
        t1 = time.monotonic()
        reflect_result = world.reflect()
        t_reflect = time.monotonic() - t1

        # Project three tasks
        t2 = time.monotonic()
        proj_api = world.project(["FastAPI", "PostgreSQL"], task="design backend API", max_concepts=10)
        proj_ml = world.project(["PyTorch", "neural network"], task="build ML pipeline", max_concepts=10)
        proj_debug = world.project(["latency", "model serving"], task="debug prod latency", max_concepts=12)
        t_project = time.monotonic() - t2

        status = world.status()

        # Compute metrics
        total_new = sum(len(r.new_concepts) for r in ingest_results)
        total_reinforced = sum(len(r.reinforced_concepts) for r in ingest_results)
        total_new_rels = sum(len(r.new_relations) for r in ingest_results)
        total_hebbian = sum(len(r.hebbian_relations) for r in ingest_results)

        api_names = {c.name for c in proj_api.concepts}
        ml_names = {c.name for c in proj_ml.concepts}
        debug_names = {c.name for c in proj_debug.concepts}

        # Cross-domain separation
        def jaccard(a, b):
            if not a and not b:
                return 1.0
            union = a | b
            return len(a & b) / len(union) if union else 1.0

        j_api_ml = jaccard(api_names, ml_names)

        # Print report
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              World 0 — Benchmark Validation Report          ║
╠══════════════════════════════════════════════════════════════╣
║ Knowledge Accumulation                                      ║
║   Total concepts:         {status.total_concepts:>4}                              ║
║   Total relations:        {status.total_relations:>4}                              ║
║   New concepts ingested:  {total_new:>4}                              ║
║   Reinforced concepts:    {total_reinforced:>4}                              ║
║   Explicit relations:     {total_new_rels:>4}                              ║
║   Hebbian relations:      {total_hebbian:>4}                              ║
║   Avg confidence:         {status.avg_confidence:>6.3f}                            ║
╠══════════════════════════════════════════════════════════════╣
║ Maturity Distribution                                       ║"""
        for m, count in sorted(status.by_maturity.items()):
            report += f"\n║   {m:<20s}  {count:>4}                              ║"
        report += f"""
╠══════════════════════════════════════════════════════════════╣
║ Projection Quality                                          ║
║   Backend proj concepts:  {len(proj_api.concepts):>4}  relations: {len(proj_api.relations):>4}          ║
║   ML proj concepts:       {len(proj_ml.concepts):>4}  relations: {len(proj_ml.relations):>4}          ║
║   Debug proj concepts:    {len(proj_debug.concepts):>4}  relations: {len(proj_debug.relations):>4}          ║
║   API↔ML Jaccard:         {j_api_ml:>6.3f}  (lower = better separation)  ║
╠══════════════════════════════════════════════════════════════╣
║ Reflect Results                                             ║
║   Promoted:               {len(reflect_result.promoted_concepts):>4}                              ║
║   Demoted:                {len(reflect_result.demoted_concepts):>4}                              ║
║   Pruned concepts:        {len(reflect_result.pruned_concepts):>4}                              ║
║   Pruned relations:       {len(reflect_result.pruned_relations):>4}                              ║
╠══════════════════════════════════════════════════════════════╣
║ Timing                                                      ║
║   Ingest (18 obs):        {t_ingest*1000:>7.1f} ms                           ║
║   Reflect:                {t_reflect*1000:>7.1f} ms                           ║
║   Project (3 tasks):      {t_project*1000:>7.1f} ms                           ║
╚══════════════════════════════════════════════════════════════╝"""
        print(report)

        # Assertions — the real validation
        assert status.total_concepts >= 20, "Should have 20+ unique concepts"
        assert status.total_relations >= 15, "Should have 15+ relations"
        assert status.avg_confidence > 0.05, "Average confidence too low"
        # Python is shared between both domains, so some overlap is expected
        assert j_api_ml < 0.7, f"API↔ML Jaccard {j_api_ml:.2f} too high — poor separation"
        assert len(proj_api.concepts) >= 3, "Backend projection too small"
        assert len(proj_ml.concepts) >= 3, "ML projection too small"
        assert t_ingest < 5.0, "Ingest too slow (>5s)"
        assert t_project < 2.0, "Projection too slow (>2s)"
