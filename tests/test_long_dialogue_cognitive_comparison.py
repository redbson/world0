"""Long-dialogue cognitive comparison: World 0 vs ordinary memory.

The baseline system uses plain transcript memory only. It has no
concept graph, no typed relations, no context-sensitive activation,
and no projection step.

The comparison focuses on cognitive behavior under long mixed-domain
dialogue, not on language-model answer style.
"""

from __future__ import annotations

import pytest

from tests._memory_dialogue_baseline import (
    build_long_dialogue_systems,
    mentioned_concepts,
)

ML_LATENT_UPSTREAM = {"neural network", "gradient descent", "optimizer"}
WORLD0_DESIGN_CORE = {
    "relation typing",
    "context weighting",
    "activation",
    "projection",
}
WORLD0_DESIGN_NON_GOALS = {
    "memory system",
    "workflow scheduling",
    "vector search",
}
LONG_DIALOGUE_VOCABULARY = (
    ML_LATENT_UPSTREAM
    | {
        "model serving",
        "PyTorch",
        "training pipeline",
        "FastAPI",
        "deployment",
        "monitoring",
        "latency",
        "autoscaling",
    }
    | WORLD0_DESIGN_CORE
    | WORLD0_DESIGN_NON_GOALS
)


@pytest.fixture
def comparison_systems(tmp_path):
    return build_long_dialogue_systems(tmp_path)


class TestLongDialogueCognitiveComparison:
    def test_world0_recovers_latent_upstream_concepts_beyond_memory_recall(
        self, comparison_systems
    ):
        world, memory = comparison_systems

        world_projection = world.project(
            ["model serving"],
            task="ml training",
            max_concepts=6,
            max_depth=4,
        )
        world_names = {concept.name for concept in world_projection.concepts}
        memory_snippets = memory.recall(
            "what upstream assumptions matter for model serving",
            max_items=4,
        )
        memory_names = mentioned_concepts(memory_snippets, LONG_DIALOGUE_VOCABULARY)

        world_latent_hits = world_names & ML_LATENT_UPSTREAM
        memory_latent_hits = memory_names & ML_LATENT_UPSTREAM

        assert world_latent_hits
        assert len(world_latent_hits) > len(memory_latent_hits)
        assert "PyTorch" in world_names
        assert "PyTorch" in memory_names

    def test_world0_preserves_design_boundary_better_than_memory_recall(
        self, comparison_systems
    ):
        world, memory = comparison_systems

        world_projection = world.project(
            ["activation"],
            task="world0 design",
            max_concepts=5,
            max_depth=3,
        )
        world_names = {concept.name for concept in world_projection.concepts}
        memory_snippets = memory.recall(
            "how should the system shape a local conceptual view",
            max_items=3,
        )
        memory_names = mentioned_concepts(memory_snippets, LONG_DIALOGUE_VOCABULARY)

        assert len(world_names & WORLD0_DESIGN_CORE) >= 3
        assert not (world_names & WORLD0_DESIGN_NON_GOALS)
        assert memory_names & WORLD0_DESIGN_NON_GOALS
        assert len(world_names & WORLD0_DESIGN_NON_GOALS) < len(
            memory_names & WORLD0_DESIGN_NON_GOALS
        )
