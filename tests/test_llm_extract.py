"""LongRun with real LLM extraction (benchmarks/longrun/llm_extract.py, round 25).

The production prompt's raw answers are cached per (seed, horizon); the
production parser turns them into the benchmark's extractions.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.longrun import llm_extract
from benchmarks.longrun.worldgen import Claim, GenConfig, Stream


def _event_with_retraction(seed: int = 0, horizon: int = 600):
    s = Stream(GenConfig(seed=seed, horizon=horizon))
    for ev, _ in s.events():
        if ev.retractions:
            return s, ev
    raise AssertionError("no retraction in stream")


def test_a_raw_answer_becomes_the_benchmark_extraction():
    s, ev = _event_with_retraction()
    known = {llm_extract._key(c): c for c in s.world.all_concepts}
    r = ev.retractions[0]
    raw = json.dumps({
        "concepts": [{"uid": "c1", "name": r.src.upper()}, {"uid": "c2", "name": r.tgt},
                     {"uid": "c3", "name": "follow-up notes"}],
        "relations": [{"source": "c1", "target": "c2", "type": "depends_on"},
                      {"source": "c1", "target": "c3", "type": "part_of"}],
        "retracted_relations": [{"source": "c1", "target": "c2", "type": "depends_on"}],
    })
    x = llm_extract.to_extraction(raw, ev, known)
    assert Claim.make(r.src, "depends_on", r.tgt) in x.claims      # names mapped back to the world's
    assert Claim.make(r.src, "membership", "follow-up notes") in x.claims  # other labels kept as chosen
    assert x.retractions == [r]
    assert "follow-up notes" in x.concepts


def test_contradicted_relations_also_count_as_retractions():
    s, ev = _event_with_retraction()
    known = {llm_extract._key(c): c for c in s.world.all_concepts}
    r = ev.retractions[0]
    raw = json.dumps({"concepts": [{"uid": "a", "name": r.src}, {"uid": "b", "name": r.tgt}],
                      "contradicted_relations": [{"source": "a", "target": "b", "type": "depends_on"}]})
    assert llm_extract.to_extraction(raw, ev, known).retractions == [r]


def test_invalid_json_extracts_nothing():
    s, ev = _event_with_retraction()
    x = llm_extract.to_extraction("not json", ev, {})
    assert x.claims == [] and x.retractions == []


@pytest.mark.skipif(not __import__("os").path.exists(llm_extract.cache_path(0, 600)),
                    reason="no cached LLM extraction")
def test_the_cached_stream_replays_and_is_profiled():
    p = llm_extract.profile(0, 600)
    assert p["events"] == 600 and p["claims_gold"] > 0
    assert 0.0 <= p["drop"] <= 1.0 and p["kept"] + p["relabelled"] + p["reversed"] + p["dropped"] == p["claims_gold"]


def test_a_missing_cache_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_extract, "CACHE_DIR", str(tmp_path))
    llm_extract.load_cache.cache_clear()
    with pytest.raises(FileNotFoundError, match="no LLM extraction cache"):
        list(Stream(GenConfig(seed=5, horizon=50, extract_mode="llm")).events())
    llm_extract.load_cache.cache_clear()


class TestNaturalText:
    """``text_style="natural"``: the same stream in other words (round 26)."""

    def _pair(self, seed=1, horizon=300):
        a = list(Stream(GenConfig(seed=seed, horizon=horizon)).events())
        b = list(Stream(GenConfig(seed=seed, horizon=horizon, text_style="natural")).events())
        return a, b

    def test_only_the_words_change(self):
        a, b = self._pair()
        assert [(e.kind, e.claims, e.retractions, e.concepts, e.ticket) for e, _ in a] == \
               [(e.kind, e.claims, e.retractions, e.concepts, e.ticket) for e, _ in b]
        assert [[(q.kind, q.gold_claims, q.stale_claims) for q in qs] for _, qs in a] == \
               [[(q.kind, q.gold_claims, q.stale_claims) for q in qs] for _, qs in b]
        assert sum(ea.text != eb.text for (ea, _), (eb, _) in zip(a, b)) > 0.9 * len(a)

    def test_natural_text_is_deterministic(self):
        _, b = self._pair()
        _, c = self._pair()
        assert [e.text for e, _ in b] == [e.text for e, _ in c]

    def test_every_concept_and_ticket_is_named_in_the_text(self):
        _, b = self._pair()
        for ev, _ in b:
            low = ev.text.lower()
            assert all(c.lower() in low for c in ev.concepts), ev.text
            assert ev.ticket is None or ev.ticket in ev.text

    def test_the_wording_varies(self):
        _, b = self._pair(horizon=600)
        text = " ".join(e.text for e, _ in b)
        for phrase in ("depends on", "is a prerequisite for", "is part of", "It ", "do not conflict"):
            assert phrase in text, phrase

    def test_negated_distractors_name_no_true_claim(self, monkeypatch):
        from benchmarks.longrun import paraphrase

        seen = []
        orig = paraphrase.Realiser.task_text

        def spy(self, step, task, stated, retire, rest, ticket, filler, negated):
            seen.append((step, negated))
            return orig(self, step, task, stated, retire, rest, ticket, filler, negated)

        monkeypatch.setattr(paraphrase.Realiser, "task_text", spy)
        s = Stream(GenConfig(seed=1, horizon=600, text_style="natural"))
        evs = {ev.step: ev for ev, _ in s.events()}
        negs = [(st, n) for st, n in seen if n]
        assert len(negs) > 50
        for st, (a, b) in negs:
            d = evs[st].domain
            assert not any({c.src, c.tgt} == {a, b} for c in s.world.truth[d])
            assert not any({c.src, c.tgt} == {a, b} for c in evs[st].claims + evs[st].retractions)

    def test_an_article_or_case_does_not_make_another_concept(self):
        assert llm_extract._key("the Zakonax router") == llm_extract._key("zakonax ROUTER") == "zakonax router"
        assert llm_extract._key("theta router") == "theta router"

    def test_styles_are_cached_separately(self):
        assert llm_extract.cache_path(0, 600) != llm_extract.cache_path(0, 600, "natural")
        assert llm_extract.cache_path(0, 600).endswith("seed0_h600.json")  # the template cache keeps its name
