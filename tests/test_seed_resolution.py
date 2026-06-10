"""Tests: robust seed resolution — typos, partials, ambiguity, context.

A mistyped or partial seed should still find its concept via the
existing signature machinery, and every projection must record *how*
each seed resolved so a missed seed is visible, not silent.
"""

from __future__ import annotations

import pytest

from world0 import ConceptCandidate, Observation, Perspective, World


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    for _ in range(8):
        w.ingest(Observation(
            concept_candidates=[
                ConceptCandidate(
                    name="PostgreSQL",
                    kind="database",
                    sense="relational database",
                    domain="infrastructure",
                    description="open source relational database system",
                    aliases=["Postgres"],
                ),
                ConceptCandidate(
                    name="vector database",
                    kind="database",
                    sense="similarity search store",
                    domain="infrastructure",
                    description="stores embeddings for nearest neighbor search",
                ),
            ],
            task="data layer",
            source="t",
        ))
    return w


class TestExactAndAlias:
    def test_exact_name_resolves(self, world):
        node, method = world.concepts.resolve_in_context("PostgreSQL")
        assert node is not None
        assert method == "exact"

    def test_alias_resolves_exact(self, world):
        node, method = world.concepts.resolve_in_context("Postgres")
        assert node is not None
        assert node.name == "PostgreSQL"
        assert method == "exact"


class TestFuzzy:
    def test_typo_resolves_fuzzy(self, world):
        node, method = world.concepts.resolve_in_context("postgresql databse")
        assert node is not None
        assert node.name == "PostgreSQL"
        assert method.startswith("fuzzy:")

    def test_partial_resolves_fuzzy(self, world):
        # Reordered / extra-word label that shares its real tokens with
        # the concept's name + description ("vector", "embeddings").
        node, method = world.concepts.resolve_in_context(
            "vector embeddings database"
        )
        assert node is not None
        assert node.name == "vector database"
        assert method.startswith("fuzzy:")

    def test_unrelated_stays_unresolved(self, world):
        node, method = world.concepts.resolve_in_context(
            "quantum chromodynamics"
        )
        assert node is None
        assert method == ""

    def test_fuzzy_disabled_returns_unresolved(self, world):
        node, method = world.concepts.resolve_in_context(
            "postgresql databse", fuzzy=False
        )
        assert node is None


class TestDomainDisambiguation:
    def test_ambiguous_label_resolves_by_domain(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        for _ in range(6):
            w.ingest(Observation(
                concept_candidates=[
                    ConceptCandidate(
                        name="Apple",
                        kind="entity",
                        sense="technology company",
                        domain="technology",
                        description="maker of iPhone and Mac",
                    ),
                    ConceptCandidate(
                        name="Apple",
                        kind="entity",
                        sense="fruit",
                        domain="food",
                        description="edible pomaceous fruit",
                    ),
                ],
                source="t",
            ))
        # Bare "Apple" is ambiguous → resolve() returns None.
        assert w.concepts.resolve("Apple") is None
        node, method = w.concepts.resolve_in_context(
            "Apple", active_domains=["technology"]
        )
        assert node is not None
        assert node.sense == "technology company"
        assert method == "disambiguated"


class TestProjectionSeedResolution:
    def test_seed_resolution_recorded(self, world):
        proj = world.project(["PostgreSQL", "postgresql databse", "nope"])
        assert proj.seed_resolution["PostgreSQL"] == "PostgreSQL"
        assert "fuzzy" in proj.seed_resolution["postgresql databse"]
        assert proj.seed_resolution["nope"] == "unresolved"

    def test_empty_projection_still_reports_resolution(self, world):
        proj = world.project(["nope", "alsonope"])
        assert proj.concepts == []
        assert proj.seed_resolution == {
            "nope": "unresolved",
            "alsonope": "unresolved",
        }

    def test_fuzzy_seeds_off_at_facade(self, world):
        proj = world.project(["postgresql databse"], fuzzy_seeds=False)
        assert proj.seed_resolution["postgresql databse"] == "unresolved"

    def test_perspective_domains_drive_disambiguation(self, tmp_path):
        w = World(store_path=tmp_path / ".world0")
        for _ in range(6):
            w.ingest(Observation(
                concept_candidates=[
                    ConceptCandidate(
                        name="Apple", kind="entity", sense="technology company",
                        domain="technology", description="maker of iPhone",
                    ),
                    ConceptCandidate(
                        name="Apple", kind="entity", sense="fruit",
                        domain="food", description="edible fruit",
                    ),
                ],
                source="t",
            ))
        lens = Perspective(name="tech", active_domains=["technology"])
        proj = w.project(["Apple"], perspective=lens)
        assert "disambiguated" in proj.seed_resolution["Apple"]
