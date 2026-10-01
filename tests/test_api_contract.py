"""The unified API (docs/world0-api.md), 0.4 step: data shapes and reads.

The golden sequence of §8 is run through the Python entry point and the
structured output is asserted field by field; the same sequence will be
the contract test of the CLI / HTTP / MCP entry points when they exist.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import world0
from world0 import API_VERSION, Claim, ConceptCard, Observation, Statement, World
from world0.api import CLAIM_STATUSES, ConceptCardInput


@pytest.fixture
def world(tmp_path):
    w = World(store_path=tmp_path / ".world0")
    yield w
    w.close()


def _golden(w: World) -> None:
    w.ingest(Observation(
        concepts=["api", "db", "cache"],
        statements=[Statement("api", "depends_on", "db"), Statement("api", "conflict", "cache")],
        task="backend", source="design review", source_id="review-2026-10-01#3",
    ))
    w.ingest(Observation(withdrawals=[Statement("api", "conflict", "cache")], task="backend"))


class TestObservationInput:
    def test_statements_fold_into_pipeline_fields_and_read_back(self):
        obs = Observation(statements=[Statement("api", "depends_on", "db", belief=0.9, rationale="stated")],
                          withdrawals=[Statement("api", "conflict", "cache")],
                          denials=[Statement("db", "conflict", "api")], task="t")
        assert obs.relations == [("api", "db", "depends_on")]
        assert obs.relation_priors[0].probability == 0.9
        assert obs.retracted_relations == [("api", "cache", "conflict")]
        assert obs.contradicted_relations == [("db", "api", "conflict")]
        # a statement mentions its endpoints; a withdrawal / denial does not
        assert obs.concepts == ["api", "db"]
        assert [s.as_triple() for s in obs.statements] == [("api", "db", "depends_on")]
        assert obs.statements[0].belief == 0.9
        assert obs.withdrawals[0].relation == "conflict" and obs.withdrawals[0].target == "cache"
        assert obs.denials[0].source == "db"

    def test_statements_accept_dicts_and_tuples(self):
        obs = Observation(statements=[{"source": "a", "relation": "enables", "target": "b"}, ("c", "contains", "d")])
        assert obs.relations == [("a", "b", "enables"), ("c", "d", "contains")]

    def test_cards_become_candidates_and_plain_names_keep_one(self):
        obs = Observation(concepts=["db"], cards=[ConceptCardInput(name="api", sense="HTTP interface", kind="component",
                                                                   aliases=["API"], description="the public interface")])
        names = {c.name for c in obs.concept_candidates}
        assert names == {"api", "db"}  # ``db`` keeps a candidate although only named
        api = next(c for c in obs.concept_candidates if c.name == "api")
        assert (api.sense, api.kind, api.aliases, api.description) == ("HTTP interface", "component", ["API"], "the public interface")
        assert obs.concepts == ["db", "api"]

    def test_old_and_new_names_together_do_not_double_a_claim(self):
        obs = Observation(relations=[("api", "db", "depends_on")], statements=[Statement("api", "depends_on", "db")])
        assert obs.relations == [("api", "db", "depends_on")]

    def test_priors_read_back_by_pair_and_label(self):
        obs = Observation(statements=[Statement("a", "depends_on", "b", belief=0.9, rationale="r1"),
                                      Statement("a", "conflict", "b", belief=0.2, rationale="r2")])
        assert [(s.relation, s.belief, s.rationale) for s in obs.statements] == [("depends_on", 0.9, "r1"), ("conflict", 0.2, "r2")]

    def test_card_completes_a_candidate_of_the_same_name(self):
        obs = Observation(concept_candidates=[{"name": "api", "kind": "component"}],
                          cards=[ConceptCardInput(name="api", sense="HTTP interface", aliases=["API"])])
        cand = obs.concept_candidates[0]
        assert (cand.kind, cand.sense, cand.aliases) == ("component", "HTTP interface", ["API"])

    def test_malformed_statements_are_validation_errors(self):
        from pydantic import ValidationError
        for bad in (["a depends_on b"], [("a", "b")], [("a", "b", "c", "d")], "a depends_on b"):
            with pytest.raises(ValidationError):
                Observation(statements=bad)
        with pytest.raises(ValidationError):
            Observation(concepts="api", statements=[Statement("a", "enables", "b")])
        assert Observation(concepts=None, statements=[Statement("a", "enables", "b")]).concepts == ["a", "b"]

    def test_pipeline_field_names_still_work_unchanged(self):
        obs = Observation(concepts=["a", "b"], relations=[("a", "b", "depends_on")], retracted_relations=[("a", "b", "enables")])
        assert obs.statements[0].as_triple() == ("a", "b", "depends_on")
        assert obs.withdrawals[0].relation == "enables"


class TestGoldenSequence:
    def test_projection_structured_views(self, world):
        _golden(world)
        view = world.project(["api"], task="backend")
        assert view.api == API_VERSION
        assert view.seeds == ["api"] and view.task == "backend"
        assert [c.text for c in view.claims] == ["api depends on db"]
        c = view.claims[0]
        assert isinstance(c, Claim)
        assert (c.relation, c.axis, c.status, c.support, c.stated_under) == ("dependence", "positive", "current", 1, ["backend"])
        assert c.belief == pytest.approx(0.70, abs=0.01)
        assert c.source_id and c.target_id and c.source == "api" and c.target == "db"
        assert [(c.text, c.status) for c in view.no_longer_holds] == [("api conflicts with cache", "withdrawn")]
        assert view.other_tasks == [] and view.hold_loosely == []
        cards = {card.name: card for card in view.cards}
        assert set(cards) == {"api", "db"}  # the withdrawn partner is not in view
        assert all(isinstance(card, ConceptCard) for card in cards.values())
        assert cards["db"].tasks == ["backend"] and cards["db"].maturity == "embryonic"
        assert view.why[cards["api"].id] == "seed"
        assert view.why[cards["db"].id].startswith("via dependence from api")

    def test_structured_views_match_the_render(self, world):
        _golden(world)
        view = world.project(["api"], task="backend")
        text = view.render()
        for c in view.claims:
            assert f"- {c.text} (belief {c.belief:.2f})" in text
        assert "No longer holds:\n- api conflicts with cache" in text

    def test_model_dump_carries_the_views_and_version(self, world):
        _golden(world)
        data = json.loads(world.project(["api"], task="backend").model_dump_json())
        assert data["api"] == API_VERSION
        assert data["claims"][0]["text"] == "api depends on db"
        assert data["no_longer_holds"][0]["status"] == "withdrawn"
        assert {"cards", "hold_loosely", "other_tasks", "why", "seeds"} <= set(data)

    def test_card_claims_find(self, world):
        _golden(world)
        card = world.card("db")
        assert card is not None and card.name == "db" and card.sources == ["design review"]
        assert card.last_seen_tick == 1 and 0.0 < card.evidence < 1.0
        assert world.card("nobody") is None
        claims = world.claims("api")
        assert [(c.text, c.status) for c in claims] == [("api depends on db", "current"), ("api conflicts with cache", "withdrawn")]
        assert all(c.status in CLAIM_STATUSES for c in claims)
        assert world.claims("nobody") == []
        found = world.find("api")
        assert found and found[0][0].name == "api" and found[0][1] == pytest.approx(1.0)

    def test_claims_task_filter_leaves_neutral_and_matching(self, world):
        world.ingest(Observation(statements=[Statement("api", "depends_on", "db")], task="backend work"))
        world.ingest(Observation(statements=[Statement("api", "enables", "report")], task="analytics work"))
        world.ingest(Observation(statements=[Statement("api", "contains", "auth")]))  # no task: neutral
        texts = lambda claims: sorted(c.text for c in claims)
        assert texts(world.claims("api")) == ["api contains auth", "api depends on db", "api enables report"]
        assert texts(world.claims("api", task="backend work")) == ["api contains auth", "api depends on db"]
        assert texts(world.claims("api", task="analytics work")) == ["api contains auth", "api enables report"]

    def test_statement_sugar_is_ingest(self, world):
        r = world.state("db", "part_of", "platform", task="backend")
        assert r.api == API_VERSION
        assert r.new_relations == ["platform → inclusion → db"]  # part_of read from the part
        world.withdraw("db", "part_of", "platform", task="backend")
        assert [c.status for c in world.claims("platform")] == ["withdrawn"]
        before = world.card("db").confidence
        r2 = world.deny("db", "conflict", "cache", task="backend")  # nobody made this claim
        assert r2.weakened_concepts == [] and world.card("db").confidence == before

    def test_claims_reports_doubted_from_the_edge(self, world):
        world.state("api", "enables", "cache", task="t")
        for _ in range(6):  # each denial lowers belief a little; six take it below even odds
            world.deny("api", "enables", "cache", task="t")
        claim = next(c for c in world.claims("api") if c.target == "cache")
        assert claim.status == "doubted" and claim.belief < 0.5

    def test_other_tasks_and_hold_loosely_statuses(self, world):
        for _ in range(3):
            world.ingest(Observation(statements=[Statement("api", "depends_on", "db")], task="backend work"))
        world.ingest(Observation(statements=[Statement("api", "depends_on", "warehouse")], task="analytics work"))
        for _ in range(4):
            world.ingest(Observation(statements=[Statement("api", "enables", "db")], task="backend work"))
        world.ingest(Observation(statements=[Statement("api", "conflict", "db")], task="backend work"))
        view = world.project(["api"], task="backend work", max_depth=2)
        assert [c.text for c in view.other_tasks] == ["api depends on warehouse"]
        assert view.other_tasks[0].stated_under == ["analytics work"]
        loose = {c.text: c.status for c in view.hold_loosely}
        assert loose.get("api conflicts with db") == "outvoted"
        assert set(loose.values()) <= set(CLAIM_STATUSES)
        assert "Hold loosely:" in view.render() and "outvoted: api conflicts with db" in view.render()

    def test_result_versions(self, world):
        assert world.reflect().api == API_VERSION
        assert world.status().api == API_VERSION


class TestToCardToClaim:
    def test_record_conversions(self, world):
        _golden(world)
        node = world.concepts.resolve("api")
        card = node.to_card()
        assert (card.id, card.name, card.maturity) == (node.id, "api", node.maturity.value)
        edge = world.relations.for_concept(node.id)[0]
        claim = edge.to_claim("api", "x")
        assert claim.source == "api" and claim.target == "x" and claim.axis in ("positive", "negative", "parallel")
        assert claim.as_statement().source == "api"


class TestApiModuleIsAContract:
    def test_api_module_imports_nothing_from_world0(self):
        path = Path(world0.__file__).parent / "api.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        assert "world0" not in imported, imported
