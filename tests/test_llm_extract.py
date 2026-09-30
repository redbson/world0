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
    assert 0.0 <= p["drop"] <= 1.0 and p["kept"] + p["relabelled"] + p["dropped"] == p["claims_gold"]


def test_a_missing_cache_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_extract, "CACHE_DIR", str(tmp_path))
    llm_extract.load_cache.cache_clear()
    with pytest.raises(FileNotFoundError, match="no LLM extraction cache"):
        list(Stream(GenConfig(seed=5, horizon=50, extract_mode="llm")).events())
    llm_extract.load_cache.cache_clear()
