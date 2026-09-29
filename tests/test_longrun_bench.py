"""The LongRun benchmark harness itself (benchmarks/longrun): a comparison is only
worth reading if the generator is deterministic, the gold sets only contain what
the Agent was told, every system respects the same token budget, and the scorer
counts what it claims to count."""

from __future__ import annotations


from benchmarks.longrun.run import BUDGETS, CORE, link, run_one
from benchmarks.longrun.scoring import closure, prf, score, stale_category
from benchmarks.longrun.parse import parse_compact, parse_shipped
from benchmarks.longrun.systems import Context, make_system
from benchmarks.longrun.tokens import est_tokens
from benchmarks.longrun.worldgen import Claim, GenConfig, Query, Stream


def _stream(**kw):
    return Stream(GenConfig(seed=3, horizon=300, query_every=10, **kw))


class TestGenerator:
    def test_stream_is_deterministic(self):
        a = [(e.text, [q.text for q in qs]) for e, qs in _stream().events()]
        b = [(e.text, [q.text for q in qs]) for e, qs in _stream().events()]
        assert a == b

    def test_gold_claims_were_stated_and_are_current(self):
        s = _stream()
        seen: list[tuple[Query, set[Claim]]] = []
        for _, qs in s.events():
            for q in qs:
                seen.append((q, set(s.stated_at)))
                current = {c for d in s.world.domains for c in s.world.truth[d]}
                assert q.gold_claims <= current | q.stale_claims, "gold must be current truth"
                assert q.gold_claims <= set(s.stated_at), "gold must have been stated"
                assert not (q.gold_claims & q.stale_claims)
        assert {q.kind for q, _ in seen} >= {"focus", "chain"}

    def test_revisions_retract_the_old_claim_and_state_the_new_one(self):
        s = _stream()
        evs = [e for e, _ in s.events()]
        assert s.revisions
        step, _, old, new = s.revisions[0]
        ev = evs[step]
        assert old in ev.retractions and new in ev.claims
        assert old not in s.world.truth[s.revisions[0][1]]

    def test_names_are_unique_pseudo_words(self):
        s = _stream()
        assert len(s.world.all_concepts) == len(set(s.world.all_concepts))


class TestScoring:
    def test_prf(self):
        assert prf({1, 2}, {2, 3}) == (0.5, 0.5, 0.5)
        assert prf(set(), {1}) == (0.0, 0.0, 0.0)

    def test_closure_follows_dependencies_only(self):
        cl = {Claim("a", "depends_on", "b"), Claim("b", "depends_on", "c"), Claim("a", "contains", "z")}
        assert closure("a", cl) == {"b", "c"}

    def test_stale_categories(self):
        q = Query(0, "stale", "d", "t", "t: q", ["a"],
                  stale_claims={Claim("a", "depends_on", "old")},
                  current_claims={Claim("a", "depends_on", "new")})
        old, new = next(iter(q.stale_claims)), next(iter(q.current_claims))
        assert stale_category(q, Context(claims={new}))[0] == "current_only"
        assert stale_category(q, Context(claims={old}))[0] == "stale_only"
        assert stale_category(q, Context(claims={old, new}))[0] == "both"
        assert stale_category(q, Context(claims={old, new}, beliefs={old: 0.4, new: 0.8}))[0] == "both_resolved"
        assert stale_category(q, Context())[0] == "neither"

    def test_detail_hit_needs_the_ticket(self):
        q = Query(0, "detail", "d", "t", "t: q", ["a"], gold_ticket="T1234")
        assert score(q, Context(tickets={"T1234"}))["detail_hit"] == 1.0
        assert score(q, Context(tickets={"T9999"}))["detail_hit"] == 0.0

    def test_token_estimate_charges_invented_names_more_than_common_words(self):
        assert est_tokens("depends on") < est_tokens("Zirokuth registry")


class TestSystems:
    def test_every_system_respects_the_budget_and_scores(self):
        rows = run_one({"horizon": 160, "seed": 5, "systems": CORE, "query_every": 20, "warmup": 60})
        queries = [r for r in rows if r["type"] == "query"]
        assert queries
        for r in queries:
            if r["system"] != "full_context":
                assert r["tokens"] <= r["budget"], r["system"]
            assert 0.0 <= r["headline"] <= 1.0
        assert {r["budget"] for r in queries if r["system"] != "full_context"} == set(BUDGETS)
        assert all(r["headline"] == 0.0 for r in queries if r["system"] == "none")

    def test_link_finds_the_longest_known_name(self):
        assert link({"Kex cache", "Kex"}, "why does Kex cache fail?") == ["Kex cache"]


class TestV2:
    def test_extraction_noise_leaves_the_text_stream_untouched(self):
        clean = [e.text for e, _ in Stream(GenConfig(seed=2, horizon=200)).events()]
        noisy = [e.text for e, _ in Stream(GenConfig(seed=2, horizon=200, extract_p=0.3)).events()]
        assert clean == noisy
        evs = [e for e, _ in Stream(GenConfig(seed=2, horizon=200, extract_p=0.3)).events() if e.kind == "task"]
        assert any(e.extracted.claims != e.claims for e in evs)

    def test_task_modes(self):
        none = Stream(GenConfig(seed=1, horizon=200, task_mode="none"))
        assert all(q.task_text == "" for _, qs in none.events() for q in qs)
        wrong = Stream(GenConfig(seed=1, horizon=200, task_mode="wrong"))
        assert all(q.task_text != wrong.task_label(q.domain) for _, qs in wrong.events() for q in qs)

    def test_strict_stale_credit_needs_the_retracted_claim_gone(self):
        q = Query(0, "stale", "d", "t", "t: q", ["a"], gold_claims={Claim("a", "depends_on", "new")},
                  stale_claims={Claim("a", "depends_on", "old")}, current_claims={Claim("a", "depends_on", "new")})
        both = Context(claims={Claim("a", "depends_on", "old"), Claim("a", "depends_on", "new")},
                       beliefs={Claim("a", "depends_on", "old"): 0.4, Claim("a", "depends_on", "new"): 0.8})
        row = score(q, both)
        assert row["stale_correct"] == 0.0 and row["stale_lenient"] == 1.0 and row["stale_shown"] == 1.0
        assert score(q, Context(claims={Claim("a", "depends_on", "new")}))["stale_correct"] == 1.0

    def test_renderers_round_trip_through_their_parsers(self):
        stream = Stream(GenConfig(seed=4, horizon=250, query_every=25))
        systems: dict = {}
        for name in ("world0", "world0_compact"):
            systems[name] = make_system(name, systems)
        known: set[str] = set()
        last = None
        for ev, qs in stream.events():
            known |= set(ev.concepts)
            for s in systems.values():
                s.observe(ev)
            last = qs[-1] if qs else last
        assert last is not None
        linked = link(known, last.text)
        for name, parse in (("world0", parse_shipped), ("world0_compact", parse_compact)):
            ctx = systems[name].query(last, linked, 1200)
            claims, concepts, _ = parse(ctx.text)
            assert claims == ctx.claims and concepts == ctx.concepts
        # scored claims must be visible in the text: every endpoint is named there
        ctx = systems["world0_compact"].query(last, linked, 1200)
        assert all(c.src in ctx.text and c.tgt in ctx.text for c in ctx.claims)

    def test_no_system_uses_the_hidden_domain(self):
        """Systems may see the task label but never ``ev.domain``."""
        import inspect

        import benchmarks.longrun.systems as m

        assert ".domain" not in inspect.getsource(m)


def test_report_core_tables_include_the_tuned_world0_system():
    from benchmarks.longrun import report

    assert {"world0", "world0_compact", "world0_tuned", "fact_task", "full_context"} <= report.MAIN
    assert not {"full_32k", "full_64k"} & report.MAIN
