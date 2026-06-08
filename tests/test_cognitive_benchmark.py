"""Cognitive benchmark suite for World 0.

This suite measures whether the repository preserves the intended
concept-first cognitive behavior:

1. Concept integrity: identity stays stable and neighboring concepts stay distinct.
2. Relation quality: projections expose typed, explainable structure.
3. Context sensitivity: the same bridge concept yields different local views by task.
4. Activation locality: local neighborhoods outrank distant concepts.
5. Projection quality: views stay compact, connected, and domain-relevant.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World

from tests._cognitive_benchmark import (
    ML_RELEVANT,
    OPS_RELEVANT,
    build_activation_chain,
    build_cognitive_benchmark_world,
    connected_coverage,
    jaccard_distance,
    precision_recall,
    projection_names,
    projection_scores,
    ranked_projection_names,
    relation_triplets,
)


@pytest.fixture
def benchmark_world(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    return build_cognitive_benchmark_world(world)


@pytest.fixture
def chain_world(tmp_path):
    world = World(store_path=tmp_path / ".world0")
    chain_names = build_activation_chain(world)
    return world, chain_names


class TestConceptIntegrityBenchmark:
    def test_case_variants_collapse_to_one_identity(self, tmp_path):
        world = World(store_path=tmp_path / ".world0")

        world.ingest(
            Observation(
                concepts=["Python"],
                task="ops reliability",
                source="cognitive_benchmark",
            )
        )
        concept_count = world.status().total_concepts

        world.ingest(
            Observation(
                concepts=["python"],
                task="ml training",
                source="cognitive_benchmark",
            )
        )

        uppercase = world.concepts.resolve("Python")
        lowercase = world.concepts.resolve("python")

        assert uppercase is not None
        assert lowercase is not None
        assert uppercase.id == lowercase.id
        assert world.status().total_concepts == concept_count
        assert uppercase.activation_count >= 2

    def test_adjacent_concepts_remain_distinct_nodes(self, benchmark_world):
        deployment = benchmark_world.concepts.resolve("deployment")
        model_serving = benchmark_world.concepts.resolve("model serving")
        training_pipeline = benchmark_world.concepts.resolve("training pipeline")

        assert deployment is not None
        assert model_serving is not None
        assert training_pipeline is not None
        assert deployment.id != model_serving.id
        assert model_serving.id != training_pipeline.id


class TestRelationQualityBenchmark:
    def test_projection_preserves_typed_explicit_relations(self, benchmark_world):
        projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=5,
            max_depth=4,
        )

        triplets = relation_triplets(projection)
        typed_triplets = {
            triplet
            for triplet in triplets
            if triplet[1] in {"positive", "negative", "parallel"}
        }

        assert ("model serving", "positive", "FastAPI") in triplets
        assert ("model serving", "positive", "deployment") in triplets
        assert typed_triplets

    def test_relation_bridge_is_explainable_from_both_sides(self, benchmark_world):
        ml_projection = benchmark_world.project(
            ["model serving"],
            task="ml training",
            max_concepts=6,
            max_depth=4,
        )
        ops_projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=6,
            max_depth=4,
        )

        ml_triplets = relation_triplets(ml_projection)
        ops_triplets = relation_triplets(ops_projection)

        assert ("model serving", "positive", "PyTorch") in ml_triplets
        assert ("model serving", "positive", "FastAPI") in ops_triplets


class TestContextSensitivityBenchmark:
    def test_same_seed_shifts_domain_balance_by_task(self, benchmark_world):
        ml_projection = benchmark_world.project(
            ["model serving"],
            task="ml training",
            max_concepts=6,
            max_depth=4,
        )
        ops_projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=6,
            max_depth=4,
        )

        ml_names = projection_names(ml_projection)
        ops_names = projection_names(ops_projection)

        ml_precision, ml_recall = precision_recall(ml_names, ML_RELEVANT)
        ops_precision, ops_recall = precision_recall(ops_names, OPS_RELEVANT)

        assert ml_precision >= 0.65
        assert ml_recall >= 0.65
        assert ops_precision >= 0.80
        assert ops_recall >= 0.80

        ml_only_hits = len(ml_names & (ML_RELEVANT - OPS_RELEVANT))
        ops_only_hits = len(ops_names & (OPS_RELEVANT - ML_RELEVANT))
        ml_bleed = len(ml_names - ML_RELEVANT) / len(ml_names)
        ops_bleed = len(ops_names - OPS_RELEVANT) / len(ops_names)

        assert ml_only_hits >= 3
        assert ops_only_hits >= 3
        assert ml_bleed <= 0.35
        assert ops_bleed <= 0.20
        assert jaccard_distance(ml_names, ops_names) >= 0.45

    def test_same_world_produces_different_rank_order_under_task_context(
        self, benchmark_world
    ):
        ml_projection = benchmark_world.project(
            ["model serving"],
            task="ml training",
            max_concepts=6,
            max_depth=4,
        )
        ops_projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=6,
            max_depth=4,
        )

        ml_ranked = ranked_projection_names(ml_projection)
        ops_ranked = ranked_projection_names(ops_projection)

        assert ml_ranked != ops_ranked
        assert ml_ranked.index("PyTorch") < ml_ranked.index("deployment")
        assert ops_ranked.index("deployment") < ops_ranked.index("PyTorch")


class TestControlGroupBenchmark:
    def test_ml_context_outperforms_wrong_task_control(self, benchmark_world):
        task_projection = benchmark_world.project(
            ["model serving"],
            task="ml training",
            max_concepts=6,
            max_depth=4,
        )
        control_projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=6,
            max_depth=4,
        )

        task_names = projection_names(task_projection)
        control_names = projection_names(control_projection)

        task_precision, task_recall = precision_recall(task_names, ML_RELEVANT)
        control_precision, control_recall = precision_recall(
            control_names, ML_RELEVANT
        )

        assert task_precision > control_precision
        assert task_recall > control_recall
        assert "neural network" in task_names or "gradient descent" in task_names
        assert "autoscaling" in control_names or "latency" in control_names

    def test_ops_context_outperforms_wrong_task_control(self, benchmark_world):
        task_projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=6,
            max_depth=4,
        )
        control_projection = benchmark_world.project(
            ["model serving"],
            task="ml training",
            max_concepts=6,
            max_depth=4,
        )

        task_names = projection_names(task_projection)
        control_names = projection_names(control_projection)

        task_precision, task_recall = precision_recall(task_names, OPS_RELEVANT)
        control_precision, control_recall = precision_recall(
            control_names, OPS_RELEVANT
        )

        assert task_precision > control_precision
        assert task_recall > control_recall
        assert "autoscaling" in task_names or "latency" in task_names
        assert "neural network" in control_names or "gradient descent" in control_names


class TestActivationLocalityBenchmark:
    def test_chain_activation_decays_with_distance(self, chain_world):
        world, chain_names = chain_world
        projection = world.project(
            [chain_names[0]],
            task="activation corridor",
            max_concepts=len(chain_names),
            max_depth=len(chain_names),
            decay=0.5,
        )
        scores = projection_scores(world, projection)

        previous = float("inf")
        for name in chain_names:
            assert name in scores
            assert scores[name] <= previous
            previous = scores[name]

    def test_local_nodes_beat_bridge_and_cross_domain_nodes(self, benchmark_world):
        projection = benchmark_world.project(
            ["neural network"],
            task="ml training",
            max_concepts=8,
            max_depth=4,
        )
        scores = projection_scores(benchmark_world, projection)

        assert scores["gradient descent"] > scores["model serving"]
        assert scores["training pipeline"] > scores["model serving"]
        if "latency" in scores:
            assert scores["model serving"] > scores["latency"]


class TestProjectionQualityBenchmark:
    def test_projection_stays_compact_connected_and_relevant(self, benchmark_world):
        projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=5,
            max_depth=4,
        )

        names = projection_names(projection)
        precision, recall = precision_recall(names, OPS_RELEVANT)

        assert len(projection.concepts) <= 5
        assert connected_coverage(projection) >= 0.80
        assert precision >= 0.80
        assert recall >= 0.60

    def test_projection_is_local_not_global(self, benchmark_world):
        projection = benchmark_world.project(
            ["model serving"],
            task="ops reliability",
            max_concepts=5,
            max_depth=4,
        )
        names = projection_names(projection)

        assert "concept card" not in names
        assert "projection" not in names
        assert "relation typing" not in names
