"""Tests: signature-based concept identity, merge, split.

These cover the move from name-string identity to
signature-based identity: duplicates that arrive with
matching descriptions should be consolidated into one
node (via alias attachment), and merge/split should
carry evidence and relations across correctly.
"""

from __future__ import annotations

import pytest

from world0 import Observation, World
from world0.schemas.concept import tokenize_signature


@pytest.fixture
def world(tmp_path):
    return World(store_path=tmp_path / ".world0")


class TestSignatureTokenization:
    def test_stopwords_and_short_tokens_removed(self):
        tokens = tokenize_signature("The Modern Async Web Framework")
        assert "the" not in tokens
        assert "modern" in tokens
        assert "async" in tokens
        assert "framework" in tokens

    def test_punctuation_split(self):
        tokens = tokenize_signature("REST-API / gRPC-server")
        assert "rest" in tokens
        assert "api" in tokens
        assert "grpc" in tokens
        assert "server" in tokens


class TestAutoConsolidation:
    def test_duplicate_with_shared_description_is_aliased(self, world):
        """PostgreSQL and postgres database collapse onto the same node."""
        world.ingest(Observation(
            concepts=["PostgreSQL"],
            descriptions={"PostgreSQL": "open-source relational database system"},
            source="a",
        ))
        world.ingest(Observation(
            concepts=["postgres database"],
            descriptions={
                "postgres database": "open-source relational database system"
            },
            source="b",
        ))
        assert len(world.concepts.all()) == 1
        node = world.concepts.resolve("PostgreSQL")
        assert node is not None
        assert world.concepts.resolve("postgres database") is node

    def test_no_consolidation_without_description(self, world):
        """Ambiguous short names must never silently merge."""
        world.ingest(Observation(concepts=["chain_0", "chain_1"], source="t"))
        assert world.concepts.resolve("chain_0") is not None
        assert world.concepts.resolve("chain_1") is not None
        assert world.concepts.resolve("chain_0") is not world.concepts.resolve(
            "chain_1"
        )

    def test_different_domains_block_consolidation(self, world):
        """Same description words but different domains should not merge."""
        world.ingest(Observation(
            concepts=["python"],
            descriptions={"python": "scripting language"},
            domain="programming",
            source="a",
        ))
        world.ingest(Observation(
            concepts=["Python"],
            descriptions={"Python": "scripting language"},
            domain="biology",  # the snake
            source="b",
        ))
        # Case-insensitive name match *still* finds the first one via
        # the name index.  We care about the domain gate preventing a
        # brand-new-named duplicate with a different domain from merging.
        world.ingest(Observation(
            concepts=["python snake"],
            descriptions={"python snake": "scripting language"},
            domain="biology",
            source="c",
        ))
        # `python snake` must not absorb into the programming `python`
        programming = world.concepts.resolve("python")
        biology = world.concepts.resolve("python snake")
        assert programming is not biology


class TestFindSimilar:
    def test_find_similar_returns_ranked_candidates(self, world):
        world.ingest(Observation(
            concepts=["FastAPI"],
            descriptions={"FastAPI": "modern async web framework for python"},
            source="t",
        ))
        world.ingest(Observation(
            concepts=["Flask"],
            descriptions={"Flask": "minimal python web framework"},
            source="t",
        ))
        matches = world.find_similar(
            "web framework for python", min_similarity=0.2
        )
        names = [n for n, _ in matches]
        assert "FastAPI" in names
        assert "Flask" in names

    def test_find_similar_ranks_exact_token_match_highest(self, world):
        world.ingest(Observation(
            concepts=["RabbitMQ"],
            descriptions={"RabbitMQ": "message broker queue"},
            source="t",
        ))
        world.ingest(Observation(
            concepts=["Kafka"],
            descriptions={"Kafka": "distributed log streaming platform"},
            source="t",
        ))
        matches = world.find_similar(
            "message broker queue", min_similarity=0.2
        )
        assert matches[0][0] == "RabbitMQ"


class TestMerge:
    def test_merge_moves_aliases_and_relations(self, world):
        world.ingest(Observation(
            concepts=["PostgreSQL", "Kubernetes"],
            relations=[("Kubernetes", "PostgreSQL", "depends_on")],
            source="a",
        ))
        world.ingest(Observation(
            concepts=["postgres"],
            source="b",
        ))
        # Manual merge (no description ⇒ no auto-consolidation)
        ok = world.merge("PostgreSQL", "postgres")
        assert ok is True
        assert world.concepts.resolve("postgres") is world.concepts.resolve(
            "PostgreSQL"
        )
        # Relation must now hang off the kept node
        kept = world.concepts.resolve("PostgreSQL")
        k8s = world.concepts.resolve("Kubernetes")
        rels = world.relations.find_any_between(kept.id, k8s.id)
        assert len(rels) == 1

    def test_merge_sums_activation_counts(self, world):
        world.ingest(Observation(concepts=["A"], source="t"))
        world.ingest(Observation(concepts=["A"], source="t"))
        world.ingest(Observation(concepts=["A_copy"], source="t"))

        a_count = world.concepts.resolve("A").activation_count
        copy_count = world.concepts.resolve("A_copy").activation_count
        expected = a_count + copy_count

        world.merge("A", "A_copy")
        assert world.concepts.resolve("A").activation_count == expected

    def test_merge_folds_duplicate_edges(self, world):
        """Edges of the same type/direction must collapse into one."""
        world.ingest(Observation(
            concepts=["X", "Y", "Y_dup"],
            relations=[
                ("X", "Y", "depends_on"),
                ("X", "Y_dup", "depends_on"),
            ],
            source="t",
        ))
        x = world.concepts.resolve("X")
        y = world.concepts.resolve("Y")

        world.merge("Y", "Y_dup")
        rels = world.relations.find_any_between(x.id, y.id)
        typed = [r for r in rels if r.relation_type.value == "depends_on"]
        assert len(typed) == 1

    def test_merge_unknown_returns_false(self, world):
        world.ingest(Observation(concepts=["only_one"], source="t"))
        assert world.merge("only_one", "nonexistent") is False


class TestSplit:
    def test_split_detaches_alias(self, world):
        world.ingest(Observation(concepts=["db"], source="t"))
        world.concepts.add_alias(world.concepts.resolve("db").id, "database")
        new_id = world.split(
            "db",
            "mongo",
            aliases_to_move=["database"],
            description="document store",
        )
        assert new_id is not None
        new_node = world.concepts.get(new_id) if False else (
            world.concepts.resolve("mongo")
        )
        assert new_node is not None
        assert world.concepts.resolve("database") is new_node
        # Original node loses the alias
        assert world.concepts.resolve("db").name == "db"
        assert "database" not in [
            a.lower() for a in world.concepts.resolve("db").aliases
        ]

    def test_split_rejects_taken_name(self, world):
        world.ingest(Observation(concepts=["a", "b"], source="t"))
        assert world.split("a", "b") is None
