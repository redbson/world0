"""Tests for domain-color diffusion across the concept graph."""

from world0 import Observation, World


def test_seeded_concepts_receive_domain_color(tmp_path):
    world = World(store_path=tmp_path / ".world0")

    world.ingest(Observation(
        concepts=["FastAPI", "REST API"],
        relations=[("FastAPI", "REST API", "contains")],
        domain="backend",
        task="backend",
        source="bench",
    ))

    fastapi = world.concepts.resolve("FastAPI")
    assert fastapi is not None
    assert fastapi.domain == "backend"
    assert fastapi.domain_strength("backend") >= 0.5


def test_domain_color_diffuses_through_existing_relation(tmp_path):
    world = World(store_path=tmp_path / ".world0")

    world.ingest(Observation(
        concepts=["PostgreSQL"],
        source="bench",
    ))
    world.ingest(Observation(
        concepts=["FastAPI"],
        relations=[("FastAPI", "PostgreSQL", "depends_on")],
        domain="backend",
        task="backend",
        source="bench",
    ))

    postgres = world.concepts.resolve("PostgreSQL")
    assert postgres is not None
    assert postgres.domain_strength("backend") > 0.01
    assert postgres.domain == "backend"


def test_bridge_concept_accumulates_multiple_domain_colors(tmp_path):
    world = World(store_path=tmp_path / ".world0")

    world.ingest(Observation(concepts=["model serving"], source="bench"))
    world.ingest(Observation(concepts=["FastAPI"], source="bench"))
    world.ingest(Observation(concepts=["PyTorch"], source="bench"))

    world.ingest(Observation(
        concepts=["FastAPI"],
        relations=[("FastAPI", "model serving", "supports")],
        domain="backend",
        task="backend",
        source="bench",
    ))
    world.ingest(Observation(
        concepts=["PyTorch"],
        relations=[("PyTorch", "model serving", "supports")],
        domain="ml",
        task="ml",
        source="bench",
    ))

    bridge = world.concepts.resolve("model serving")
    assert bridge is not None
    assert bridge.domain_strength("backend") > 0.01
    assert bridge.domain_strength("ml") > 0.01
    assert len(bridge.domain_profile) >= 2
