"""Deep, deterministic tests for the raw-source provenance library.

Targets ``world0.sources.library.SourceLibrary`` and the
``world0.schemas.source.SourceRecord`` schema it persists.  All tests use a
real :class:`~world0.store.json_store.JsonStore` rooted at ``tmp_path`` so
behaviour is fully deterministic and exercises the on-disk round trip.

Coverage angles:
  * ``record_raw`` field/provenance/token construction and return value
  * dedup / idempotency keyed on (task, source, raw_text)
  * ``attach_observation`` concept + relation_count + domain updates
  * ``search`` token intersection, concept-text fallback, ranking, limit,
    case-insensitivity, and empty-query behaviour
  * persistence round trip via a fresh library over the same store
  * edge cases: empty/whitespace text, unicode/CJK, long text, dup tokens
"""

from __future__ import annotations

from world0.schemas.source import (
    SourceRecord,
    source_hash,
    source_id_for,
    source_tokens,
)
from world0.schemas.types import Observation
from world0.sources import SourceLibrary
from world0.store.json_store import JsonStore


def _library(tmp_path) -> SourceLibrary:
    return SourceLibrary(JsonStore(tmp_path))


# ─────────────────────────────────────────────────────────────────────
# record_raw — record construction and provenance
# ─────────────────────────────────────────────────────────────────────


def test_record_raw_returns_populated_record(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw(
        "Activation spreads across the concept graph.",
        task="study activation",
        source="notebook",
    )

    assert isinstance(rec, SourceRecord)
    assert rec.raw_text == "Activation spreads across the concept graph."
    assert rec.task == "study activation"
    assert rec.source == "notebook"
    # Untouched by extraction yet.
    assert rec.concepts == []
    assert rec.relation_count == 0
    assert rec.domain == ""


def test_record_raw_id_and_hash_are_deterministic(tmp_path):
    lib = _library(tmp_path)
    raw = "Projection produces a local conceptual view."
    rec = lib.record_raw(raw, task="t", source="s")

    assert rec.id == source_id_for(raw, task="t", source="s")
    assert len(rec.id) == 16
    assert rec.content_hash == source_hash(raw)
    # content_hash is over raw text only — independent of task/source.
    other = lib.record_raw(raw, task="other-task", source="other-src")
    assert other.content_hash == rec.content_hash
    assert other.id != rec.id


def test_record_raw_tokens_include_source_task_and_text(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("retrieval grounding", task="agent task", source="memo")

    assert rec.tokens == source_tokens("memo agent task retrieval grounding")
    assert "retrieval" in rec.tokens
    assert "grounding" in rec.tokens
    assert "agent" in rec.tokens
    assert "memo" in rec.tokens


def test_record_raw_persists_to_store(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("Context changes relevance.", task="ctx")

    on_disk = lib.get(rec.id)
    assert on_disk is not None
    assert on_disk.raw_text == "Context changes relevance."
    assert (tmp_path / "sources" / f"{rec.id}.json").exists()
    assert (tmp_path / "source_index.json").exists()


# ─────────────────────────────────────────────────────────────────────
# dedup / idempotency
# ─────────────────────────────────────────────────────────────────────


def test_record_raw_is_idempotent_for_identical_inputs(tmp_path):
    lib = _library(tmp_path)
    first = lib.record_raw("same text", task="a", source="b")
    second = lib.record_raw("same text", task="a", source="b")

    assert first.id == second.id
    # Re-recording returns the already-stored record, not a new one.
    assert len(lib.all()) == 1


def test_record_raw_reuse_preserves_attached_extraction(tmp_path):
    """A second record_raw must return the existing (enriched) record,
    not silently overwrite it with a blank one."""
    lib = _library(tmp_path)
    rec = lib.record_raw("enrich me", task="a", source="b")
    lib.attach_observation(
        rec.id,
        Observation(concepts=["alpha"], relations=[("alpha", "beta", "rel")],
                    domain="d"),
    )

    again = lib.record_raw("enrich me", task="a", source="b")
    assert again.concepts == ["alpha"]
    assert again.relation_count == 1
    assert again.domain == "d"


def test_record_raw_distinguishes_by_task_and_source(tmp_path):
    lib = _library(tmp_path)
    base = lib.record_raw("identical body", task="t1", source="s1")
    diff_task = lib.record_raw("identical body", task="t2", source="s1")
    diff_source = lib.record_raw("identical body", task="t1", source="s2")

    ids = {base.id, diff_task.id, diff_source.id}
    assert len(ids) == 3
    assert len(lib.all()) == 3


def test_record_raw_id_ignores_surrounding_task_source_whitespace(tmp_path):
    """source_id_for strips task/source — padded variants collapse to one id."""
    lib = _library(tmp_path)
    a = lib.record_raw("body", task="task", source="src")
    b = lib.record_raw("body", task="  task  ", source="  src  ")

    assert a.id == b.id
    assert len(lib.all()) == 1


# ─────────────────────────────────────────────────────────────────────
# attach_observation
# ─────────────────────────────────────────────────────────────────────


def test_attach_observation_updates_counts(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("raw", task="t")

    updated = lib.attach_observation(
        rec.id,
        Observation(
            concepts=["concept a", "concept b"],
            relations=[("concept a", "concept b", "depends_on")],
            domain="rag systems",
        ),
    )

    assert updated is not None
    assert updated.concepts == ["concept a", "concept b"]
    assert updated.relation_count == 1
    assert updated.domain == "rag systems"


def test_attach_observation_dedups_and_drops_empty_concepts(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("raw", task="t")

    updated = lib.attach_observation(
        rec.id,
        Observation(concepts=["x", "x", "", "y"], relations=[]),
    )

    assert updated.concepts == ["x", "y"]
    assert updated.relation_count == 0


def test_attach_observation_empty_domain_preserves_existing(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("raw", task="t")
    lib.attach_observation(rec.id, Observation(concepts=["a"], domain="first"))

    # A later observation with a blank domain must not wipe the existing one.
    again = lib.attach_observation(rec.id, Observation(concepts=["a"], domain=""))
    assert again.domain == "first"


def test_attach_observation_unknown_id_returns_none(tmp_path):
    lib = _library(tmp_path)
    assert lib.attach_observation("deadbeef", Observation(concepts=["a"])) is None


def test_attach_observation_persists_to_store(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("raw", task="t")
    lib.attach_observation(
        rec.id,
        Observation(concepts=["persisted"], relations=[("a", "b", "r")],
                    domain="dom"),
    )

    reloaded = lib.get(rec.id)
    assert reloaded.concepts == ["persisted"]
    assert reloaded.relation_count == 1
    assert reloaded.domain == "dom"


# ─────────────────────────────────────────────────────────────────────
# search
# ─────────────────────────────────────────────────────────────────────


def test_search_finds_record_by_content_token(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("Vector search retrieves relevant chunks.", task="t")

    hits = lib.search("vector chunks")
    assert hits and hits[0].id == rec.id


def test_search_is_case_insensitive(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("Activation Propagation Rules", task="t")

    assert lib.search("activation")[0].id == rec.id
    assert lib.search("ACTIVATION")[0].id == rec.id
    assert lib.search("AcTiVaTiOn")[0].id == rec.id


def test_search_no_match_returns_empty(tmp_path):
    lib = _library(tmp_path)
    lib.record_raw("retrieval augmented generation", task="t")

    assert lib.search("unrelated nonexistent terms") == []


def test_search_empty_or_short_query_returns_empty(tmp_path):
    lib = _library(tmp_path)
    lib.record_raw("anything here", task="t")

    # No tokens survive (empty, whitespace, single-char tokens).
    assert lib.search("") == []
    assert lib.search("   ") == []
    assert lib.search("a") == []


def test_search_matches_attached_concepts_text(tmp_path):
    """search also scores against attached concept text, not just tokens."""
    lib = _library(tmp_path)
    rec = lib.record_raw("opaque body without the concept word", task="t")
    lib.attach_observation(
        rec.id, Observation(concepts=["hierarchical clustering"])
    )

    hits = lib.search("hierarchical")
    assert hits and hits[0].id == rec.id


def test_search_ranks_higher_token_overlap_first(tmp_path):
    lib = _library(tmp_path)
    weak = lib.record_raw("vector indexes only", task="t1", source="s1")
    strong = lib.record_raw("vector search retrieval chunks", task="t2", source="s2")

    hits = lib.search("vector search retrieval chunks")
    assert [h.id for h in hits][:2] == [strong.id, weak.id]


def test_search_respects_limit(tmp_path):
    lib = _library(tmp_path)
    for i in range(5):
        lib.record_raw(f"shared keyword body variant {i}", task=f"t{i}")

    hits = lib.search("shared keyword body", limit=2)
    assert len(hits) == 2


# ─────────────────────────────────────────────────────────────────────
# persistence round trip
# ─────────────────────────────────────────────────────────────────────


def test_records_survive_fresh_library_over_same_store(tmp_path):
    first = _library(tmp_path)
    rec = first.record_raw("Perspective conditions the projection.", task="t",
                           source="s")
    first.attach_observation(
        rec.id, Observation(concepts=["perspective"], domain="cognition")
    )

    # Brand-new library + store object over the same directory.
    second = SourceLibrary(JsonStore(tmp_path))
    reloaded = second.get(rec.id)
    assert reloaded is not None
    assert reloaded.raw_text == "Perspective conditions the projection."
    assert reloaded.concepts == ["perspective"]
    assert reloaded.domain == "cognition"
    assert {r.id for r in second.all()} == {rec.id}
    # Search index also survives reload.
    assert second.search("perspective")[0].id == rec.id


def test_record_raw_dedup_holds_across_library_instances(tmp_path):
    first = _library(tmp_path)
    rec = first.record_raw("dedup across reload", task="t", source="s")

    second = SourceLibrary(JsonStore(tmp_path))
    again = second.record_raw("dedup across reload", task="t", source="s")
    assert again.id == rec.id
    assert len(second.all()) == 1


# ─────────────────────────────────────────────────────────────────────
# edge cases
# ─────────────────────────────────────────────────────────────────────


def test_empty_text_record_is_stored_and_unsearchable(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("", task="", source="")

    assert lib.get(rec.id) is not None
    assert rec.raw_text == ""
    assert rec.tokens == []  # nothing tokenizable
    # Empty content still hashes deterministically.
    assert rec.content_hash == source_hash("")


def test_whitespace_only_text_yields_no_tokens(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("   \n\t  ", task="", source="")

    assert rec.tokens == []
    assert lib.get(rec.id) is not None


def test_unicode_cjk_text_tokenizes_and_round_trips(tmp_path):
    lib = _library(tmp_path)
    raw = "概念 关系 激活 投影"
    rec = lib.record_raw(raw, task="认知", source="笔记")

    # \w with re.UNICODE keeps CJK word characters.
    assert "概念" in rec.tokens
    assert "认知" in rec.tokens

    reloaded = SourceLibrary(JsonStore(tmp_path)).get(rec.id)
    assert reloaded.raw_text == raw
    assert lib.search("概念")[0].id == rec.id


def test_unicode_search_is_lowercased_consistently(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("Café Naïve Façade", task="t")

    # Accented latin lowercases; tokens stored lowercase, query lowercased.
    assert "café" in rec.tokens
    assert lib.search("CAFÉ")[0].id == rec.id


def test_duplicate_tokens_are_collapsed(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("graph graph graph node node", task="")

    assert rec.tokens.count("graph") == 1
    assert rec.tokens.count("node") == 1
    assert source_tokens("graph graph node node") == ["graph", "node"]


def test_long_text_has_no_token_cap_and_round_trips(tmp_path):
    lib = _library(tmp_path)
    # Many distinct tokens — confirms there is no artificial token cap.
    words = " ".join(f"token{i}" for i in range(500))
    rec = lib.record_raw(words, task="bulk")

    # 500 distinct content tokens + the "bulk" task token.
    assert len(rec.tokens) == 501
    reloaded = SourceLibrary(JsonStore(tmp_path)).get(rec.id)
    assert reloaded.raw_text == words
    assert lib.search("token250")[0].id == rec.id


def test_single_char_tokens_are_filtered(tmp_path):
    lib = _library(tmp_path)
    rec = lib.record_raw("a b c de fg", task="")

    # len < 2 dropped; "de"/"fg" kept.
    assert "a" not in rec.tokens
    assert "b" not in rec.tokens
    assert "de" in rec.tokens
    assert "fg" in rec.tokens
