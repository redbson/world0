"""Reusable fixtures and metrics for cognitive benchmark tests.

These helpers focus on World 0's intended cognitive chain:
concept -> relation -> context -> activation -> projection.
"""

from __future__ import annotations

from collections.abc import Iterable

from world0 import Observation, Projection, World

ML_RELEVANT = {
    "model serving",
    "PyTorch",
    "training pipeline",
    "neural network",
    "gradient descent",
    "optimizer",
}

OPS_RELEVANT = {
    "model serving",
    "FastAPI",
    "deployment",
    "monitoring",
    "latency",
    "autoscaling",
}

WORLD0_RELEVANT = {
    "concept card",
    "relation typing",
    "context weighting",
    "activation",
    "projection",
}


def build_cognitive_benchmark_world(world: World, rounds: int = 10) -> World:
    """Build a small benchmark world with two connected task domains."""
    observations = [
        Observation(
            concepts=[
                "model serving",
                "PyTorch",
                "training pipeline",
                "neural network",
                "gradient descent",
                "optimizer",
            ],
            relations=[
                ("model serving", "PyTorch", "depends_on"),
                ("PyTorch", "training pipeline", "supports"),
                ("training pipeline", "neural network", "contains"),
                ("neural network", "gradient descent", "depends_on"),
                ("optimizer", "gradient descent", "supports"),
            ],
            descriptions={
                "model serving": "Operational bridge from trained model to live system",
                "training pipeline": "A structured path for fitting and validating a model",
            },
            task="ml training",
            source="cognitive_benchmark",
        ),
        Observation(
            concepts=[
                "model serving",
                "FastAPI",
                "deployment",
                "monitoring",
                "latency",
                "autoscaling",
            ],
            relations=[
                ("model serving", "FastAPI", "depends_on"),
                ("model serving", "deployment", "depends_on"),
                ("deployment", "monitoring", "contains"),
                ("monitoring", "latency", "activates"),
                ("autoscaling", "latency", "supports"),
            ],
            descriptions={
                "deployment": "Moving a service into a live runtime environment",
                "monitoring": "Signals and instrumentation for service health",
            },
            task="ops reliability",
            source="cognitive_benchmark",
        ),
        Observation(
            concepts=[
                "concept card",
                "relation typing",
                "context weighting",
                "activation",
                "projection",
            ],
            relations=[
                ("concept card", "relation typing", "supports"),
                ("context weighting", "activation", "activates"),
                ("activation", "projection", "precedes"),
            ],
            descriptions={
                "concept card": "Editable source record for a concept",
                "projection": "A compact local conceptual view for downstream reasoning",
            },
            task="world0 design",
            source="cognitive_benchmark",
        ),
    ]

    for _ in range(rounds):
        for observation in observations:
            world.ingest(observation)

    return world


def build_activation_chain(world: World, length: int = 5) -> list[str]:
    """Build a simple path used to test activation locality."""
    names = [f"chain_{index}" for index in range(length)]
    for _ in range(8):
        for left, right in zip(names, names[1:]):
            world.ingest(
                Observation(
                    concepts=[left, right],
                    relations=[(left, right, "precedes")],
                    task="activation corridor",
                    source="cognitive_benchmark",
                )
            )
    return names


def projection_names(projection: Projection) -> set[str]:
    return {concept.name for concept in projection.concepts}


def ranked_projection_names(projection: Projection) -> list[str]:
    return [
        concept.name
        for concept in sorted(
            projection.concepts,
            key=lambda concept: projection.activation_scores.get(concept.id, 0.0),
            reverse=True,
        )
    ]


def projection_scores(world: World, projection: Projection) -> dict[str, float]:
    return {
        concept.name: projection.activation_scores.get(concept.id, 0.0)
        for concept in projection.concepts
        if world.concepts.resolve(concept.name)
    }


def relation_triplets(projection: Projection) -> set[tuple[str, str, str]]:
    id_to_name = {concept.id: concept.name for concept in projection.concepts}
    return {
        (
            id_to_name.get(relation.source_id, relation.source_id),
            relation.relation_type.value,
            id_to_name.get(relation.target_id, relation.target_id),
        )
        for relation in projection.relations
    }


def connected_coverage(projection: Projection) -> float:
    """Share of projected concepts that participate in at least one relation."""
    if not projection.concepts:
        return 1.0

    connected_ids: set[str] = set()
    for relation in projection.relations:
        connected_ids.add(relation.source_id)
        connected_ids.add(relation.target_id)

    return len(connected_ids) / len(projection.concepts)


def precision_recall(
    retrieved: Iterable[str], relevant: Iterable[str]
) -> tuple[float, float]:
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    if not retrieved_set:
        return 0.0, 0.0

    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set)
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def jaccard_distance(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))
