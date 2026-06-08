"""Deep, deterministic coverage of ``IngestResult`` field correctness.

This suite exercises the *result object* returned by ``IngestPipeline.run``
(and, for cross-checks, ``World.ingest`` with ``llm=None``) across every
ingest scenario that populates one of the seven ``IngestResult`` lists:

    new_concepts / reinforced_concepts / weakened_concepts
    new_relations / reinforced_relations / weakened_relations
    hebbian_relations

It deliberately goes *broader and deeper* than the neighbour suite
(``src/world0/world/tests/test_pipelines_with_fakes.py``) rather than
duplicating its three disconfirmation tests.  The focus is: which field
gets populated, with what label format, and that no value ever leaks into
the wrong field or is duplicated.

All tests are deterministic — the only stochastic component (Hebbian
learning) is driven through the ``FakeHebbianLearner`` queue.
"""

from __future__ import annotations

from world0.core.test_doubles import (
    FakeColorField,
    FakeConceptStore,
    FakeHebbianLearner,
    FakeRelationStore,
    make_concept,
    make_edge,
)
from world0.schemas.relation import RelationType
from world0.schemas.types import ConceptCandidate, IngestResult, Observation
from world0.world._ingest import IngestPipeline
from world0.world.facade import World


# ── helpers ──────────────────────────────────────────────────────────


def _pipeline(
    *,
    concepts: FakeConceptStore | None = None,
    relations: FakeRelationStore | None = None,
    hebbian: FakeHebbianLearner | None = None,
    color: FakeColorField | None = None,
) -> tuple[
    IngestPipeline,
    FakeConceptStore,
    FakeRelationStore,
    FakeHebbianLearner,
    FakeColorField,
]:
    cs = concepts or FakeConceptStore()
    rs = relations or FakeRelationStore()
    hb = hebbian or FakeHebbianLearner()
    cf = color or FakeColorField()
    pipeline = IngestPipeline(concepts=cs, relations=rs, hebbian=hb, color=cf)
    return pipeline, cs, rs, hb, cf


_RESULT_FIELDS = (
    "new_concepts",
    "reinforced_concepts",
    "weakened_concepts",
    "new_relations",
    "reinforced_relations",
    "weakened_relations",
    "hebbian_relations",
)


def _all_values(result: IngestResult) -> dict[str, list[str]]:
    return {f: list(getattr(result, f)) for f in _RESULT_FIELDS}


# ── new vs reinforced concepts ───────────────────────────────────────


def test_first_observation_is_new_only() -> None:
    pipeline, cs, *_ = _pipeline()
    result = pipeline.run(Observation(concepts=["Alpha"], task="t"))
    assert result.new_concepts == ["Alpha"]
    assert result.reinforced_concepts == []
    assert len(cs) == 1


def test_repeat_concept_is_reinforced_not_new() -> None:
    pipeline, *_ = _pipeline()
    pipeline.run(Observation(concepts=["Alpha"], task="t1"))
    result = pipeline.run(Observation(concepts=["Alpha"], task="t2"))
    assert result.new_concepts == []
    assert result.reinforced_concepts == ["Alpha"]


def test_mixed_batch_splits_new_and_reinforced() -> None:
    pipeline, cs, *_ = _pipeline()
    pipeline.run(Observation(concepts=["known"], task="t1"))
    result = pipeline.run(
        Observation(concepts=["known", "fresh"], task="t2")
    )
    assert result.new_concepts == ["fresh"]
    assert result.reinforced_concepts == ["known"]
    # Each concept lands in exactly one of the two buckets, never both.
    assert set(result.new_concepts).isdisjoint(result.reinforced_concepts)
    assert len(cs) == 2


def test_duplicate_name_within_one_batch_first_new_then_reinforced() -> None:
    # The same plain name twice in one observation: first occurrence
    # creates, second reinforces — order preserved.
    pipeline, cs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concept_candidates=[
                ConceptCandidate(uid="a", name="dup"),
                ConceptCandidate(uid="b", name="dup"),
            ],
            task="t",
        )
    )
    assert result.new_concepts == ["dup"]
    assert result.reinforced_concepts == ["dup"]
    assert len(cs) == 1


# ── concept candidates: sense / synonym handling ─────────────────────


def test_same_name_different_sense_yields_two_new_concepts() -> None:
    pipeline, cs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concept_candidates=[
                ConceptCandidate(
                    uid="c1",
                    name="Mercury",
                    kind="entity",
                    sense="planet",
                    domain="astronomy",
                ),
                ConceptCandidate(
                    uid="c2",
                    name="Mercury",
                    kind="entity",
                    sense="chemical element",
                    domain="chemistry",
                ),
            ],
            task="disambiguation",
        )
    )
    # Distinct senses → two distinct, both-new concepts.
    assert result.new_concepts == ["Mercury", "Mercury"]
    assert result.reinforced_concepts == []
    assert len(cs) == 2
    assert {n.sense for n in cs.all()} == {"planet", "chemical element"}


def test_synonym_alias_collapses_to_single_concept_no_duplicate_new() -> None:
    pipeline, cs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concept_candidates=[
                ConceptCandidate(
                    uid="c1",
                    name="large language model",
                    kind="entity",
                    sense="large language model",
                    domain="ai",
                ),
                ConceptCandidate(
                    uid="c2",
                    name="LLM",
                    kind="entity",
                    sense="large language model",
                    domain="ai",
                    aliases=["large language model"],
                ),
            ],
            task="synonym",
        )
    )
    # One canonical concept; the synonym reinforces rather than duplicating.
    assert len(cs.all()) == 1
    assert result.new_concepts == ["large language model"]
    assert result.reinforced_concepts == ["large language model"]
    # The canonical name never appears twice in new_concepts.
    assert result.new_concepts.count("large language model") == 1


# ── new vs reinforced relations + label format ───────────────────────


def test_first_relation_is_new_with_canonical_label() -> None:
    pipeline, _, rs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "positive")],
            task="t",
        )
    )
    # Label format: "src → semantic_relation → tgt" (normalized relation).
    assert result.new_relations == ["A → mutual_reinforcement → B"]
    assert result.reinforced_relations == []
    assert len(rs) == 1


def test_repeat_relation_is_reinforced_not_new() -> None:
    pipeline, *_ = _pipeline()
    pipeline.run(
        Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "membership")],
            task="t1",
        )
    )
    result = pipeline.run(
        Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "membership")],
            task="t2",
        )
    )
    assert result.new_relations == []
    assert result.reinforced_relations == ["A → membership → B"]


def test_relation_label_uses_normalized_semantic_relation() -> None:
    # "related_to" and "parallel" both normalize to generic_relation; the
    # label reflects the normalized form, not the raw input.
    pipeline, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["X", "Y"],
            relations=[("X", "Y", "related_to")],
            task="t",
        )
    )
    assert result.new_relations == ["X → generic_relation → Y"]


def test_relation_with_unresolved_endpoint_is_dropped() -> None:
    # "ghost" is never introduced as a concept and cannot be resolved, so
    # the relation is silently skipped — no entry in any relation field.
    pipeline, _, rs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["real"],
            relations=[("real", "ghost", "positive")],
            task="t",
        )
    )
    assert result.new_relations == []
    assert result.reinforced_relations == []
    assert len(rs) == 0


def test_two_distinct_relations_both_new() -> None:
    pipeline, _, rs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["A", "B", "C"],
            relations=[("A", "B", "positive"), ("B", "C", "membership")],
            task="t",
        )
    )
    assert result.new_relations == [
        "A → mutual_reinforcement → B",
        "B → membership → C",
    ]
    assert result.reinforced_relations == []
    assert len(rs) == 2


# ── weakened concepts ────────────────────────────────────────────────


def test_observation_weakened_populates_weakened_concepts() -> None:
    pipeline, cs, *_ = _pipeline()
    pipeline.run(Observation(concepts=["target"], task="t"))
    result = pipeline.run(Observation(weakened=["target"], task="t"))
    assert result.weakened_concepts == ["target"]
    # The concept actually recorded a disconfirmation event.
    node = cs.resolve("target")
    assert node is not None
    assert node.disconfirmation_count >= 1


def test_weakened_unknown_concept_is_noop() -> None:
    # A weakened concept that cannot be resolved produces no result entry.
    pipeline, *_ = _pipeline()
    result = pipeline.run(Observation(weakened=["nobody"], task="t"))
    assert result.weakened_concepts == []


# ── weakened relations (existing edge) ───────────────────────────────


def test_contradicted_relation_with_existing_edge_weakens_relation() -> None:
    pipeline, cs, rs, *_ = _pipeline()
    pipeline.run(
        Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "membership")],
            task="t",
        )
    )
    result = pipeline.run(
        Observation(
            concepts=["A", "B"],
            contradicted_relations=[("A", "B", "membership")],
            task="t",
        )
    )
    assert result.weakened_relations == ["A → membership → B"]
    # The edge path must NOT spill over into the concept fallback.
    assert result.weakened_concepts == []
    # The relation store recorded a weaken on the edge.
    assert any(name == "weaken" for name, _ in rs.calls)


# ── weakened concepts via contradiction WITHOUT an edge (the fix) ─────


def test_contradiction_without_edge_weakens_both_endpoints() -> None:
    pipeline, cs, rs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["MongoDB", "bottleneck"],
            contradicted_relations=[("MongoDB", "bottleneck", "membership")],
            task="profiling",
        )
    )
    assert sorted(result.weakened_concepts) == ["MongoDB", "bottleneck"]
    # No edge existed, so nothing lands in weakened_relations.
    assert result.weakened_relations == []
    # No edge was created as a side effect.
    assert len(rs) == 0
    # disconfirmation_count rose on both endpoints (the fix's effect).
    for node in cs.all():
        assert node.disconfirmation_count >= 1


def test_contradiction_without_edge_dedupes_shared_endpoints() -> None:
    # Two contradicted relations sharing an endpoint must not report that
    # endpoint twice in weakened_concepts.
    pipeline, cs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["hub", "x", "y"],
            contradicted_relations=[
                ("hub", "x", "membership"),
                ("hub", "y", "membership"),
            ],
            task="t",
        )
    )
    assert result.weakened_concepts.count("hub") == 1
    assert sorted(result.weakened_concepts) == ["hub", "x", "y"]


def test_contradiction_without_edge_does_not_double_count_within_one_pair() -> None:
    # Same name on both ends: only one concept exists, reported once.
    pipeline, cs, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["solo"],
            contradicted_relations=[("solo", "solo", "membership")],
            task="t",
        )
    )
    assert result.weakened_concepts == ["solo"]
    node = cs.resolve("solo")
    assert node is not None
    # Both weaken calls hit the same single node, so it disconfirms twice
    # but is reported exactly once.
    assert node.disconfirmation_count >= 2


# ── hebbian relations ────────────────────────────────────────────────


def test_hebbian_relations_populated_when_learner_emits_edge() -> None:
    # Seed an edge into the relation store and queue its id so the fake
    # Hebbian learner "discovers" it for two co-occurring concepts.
    a = make_concept("alpha", concept_id="id-a")
    b = make_concept("beta", concept_id="id-b")
    cs = FakeConceptStore(seed=[a, b])
    edge = make_edge("id-a", "id-b", relation_type=RelationType.PARALLEL)
    rs = FakeRelationStore(seed=[edge])
    hb = FakeHebbianLearner()
    hb.next_new_relation_ids = [edge.id]
    pipeline, *_ = _pipeline(concepts=cs, relations=rs, hebbian=hb)

    result = pipeline.run(
        Observation(concepts=["alpha", "beta"], task="t")
    )
    # Label format for hebbian edges: "a ↔ b".
    assert result.hebbian_relations == ["alpha ↔ beta"]


def test_hebbian_not_invoked_for_single_concept() -> None:
    cs = FakeConceptStore()
    hb = FakeHebbianLearner()
    hb.next_new_relation_ids = ["should-not-be-read"]
    pipeline, *_ = _pipeline(concepts=cs, hebbian=hb)
    result = pipeline.run(Observation(concepts=["lonely"], task="t"))
    # Single concept → hebbian step short-circuits, learner never called.
    assert hb.calls == []
    assert result.hebbian_relations == []
    # The queued id was not consumed.
    assert hb.next_new_relation_ids == ["should-not-be-read"]


def test_hebbian_skips_emitted_id_with_no_backing_edge() -> None:
    # Learner emits an id that has no edge in the store → skipped silently.
    cs = FakeConceptStore()
    hb = FakeHebbianLearner()
    hb.next_new_relation_ids = ["phantom-edge-id"]
    pipeline, *_ = _pipeline(concepts=cs, hebbian=hb)
    result = pipeline.run(Observation(concepts=["a", "b"], task="t"))
    assert hb.calls, "learner should be called for >=2 concepts"
    assert result.hebbian_relations == []


# ── no-cross-contamination / empties / idempotency ───────────────────


def test_empty_observation_yields_all_empty_result() -> None:
    pipeline, *_ = _pipeline()
    result = pipeline.run(Observation())
    for field, values in _all_values(result).items():
        assert values == [], f"{field} should be empty for empty observation"


def test_pure_new_concepts_do_not_touch_other_fields() -> None:
    pipeline, *_ = _pipeline()
    result = pipeline.run(Observation(concepts=["one", "two"], task="t"))
    values = _all_values(result)
    assert values["new_concepts"] == ["one", "two"]
    # Every other field stays empty — no accidental cross-population.
    for field in _RESULT_FIELDS:
        if field == "new_concepts":
            continue
        assert values[field] == [], f"{field} leaked"


def test_relation_endpoints_also_appear_as_new_concepts_not_relations() -> None:
    # Concepts and relations occupy separate fields; a relation's endpoints
    # are reported as concepts, the edge as a relation — never crossed.
    pipeline, *_ = _pipeline()
    result = pipeline.run(
        Observation(
            concepts=["A", "B"],
            relations=[("A", "B", "positive")],
            task="t",
        )
    )
    assert result.new_concepts == ["A", "B"]
    assert result.new_relations == ["A → mutual_reinforcement → B"]
    assert result.reinforced_concepts == []
    assert result.reinforced_relations == []


def test_label_ordering_and_idempotency_across_runs() -> None:
    # The same observation run twice on a fresh world produces identical
    # label sets (deterministic ordering); on a single world the second run
    # flips new → reinforced but keeps identical labels.
    obs = Observation(
        concepts=["A", "B", "C"],
        relations=[("A", "B", "positive"), ("B", "C", "membership")],
        task="t",
    )

    p1, *_ = _pipeline()
    r1 = p1.run(obs)

    p2, *_ = _pipeline()
    r2 = p2.run(obs)

    assert r1.new_concepts == r2.new_concepts
    assert r1.new_relations == r2.new_relations

    # Re-run on the SAME pipeline: identical labels, now reinforced.
    r1b = p1.run(obs)
    assert r1b.new_concepts == []
    assert r1b.reinforced_concepts == r1.new_concepts
    assert r1b.new_relations == []
    assert r1b.reinforced_relations == r1.new_relations


# ── cross-check via the World facade (llm=None) ──────────────────────


def test_world_facade_new_then_reinforced(tmp_path) -> None:
    world = World(store_path=tmp_path / "w", llm=None)
    r1 = world.ingest(Observation(concepts=["Caching"], task="t1"))
    assert r1.new_concepts == ["Caching"]
    assert r1.reinforced_concepts == []

    r2 = world.ingest(Observation(concepts=["Caching"], task="t2"))
    assert r2.new_concepts == []
    assert r2.reinforced_concepts == ["Caching"]


def test_world_facade_relation_new_then_reinforced(tmp_path) -> None:
    world = World(store_path=tmp_path / "w", llm=None)
    r1 = world.ingest(
        Observation(
            concepts=["Index", "Query"],
            relations=[("Index", "Query", "positive")],
            task="t",
        )
    )
    assert r1.new_relations == ["Index → mutual_reinforcement → Query"]

    r2 = world.ingest(
        Observation(
            concepts=["Index", "Query"],
            relations=[("Index", "Query", "positive")],
            task="t",
        )
    )
    assert r2.new_relations == []
    assert r2.reinforced_relations == ["Index → mutual_reinforcement → Query"]


def test_world_facade_contradiction_without_edge_weakens_endpoints(
    tmp_path,
) -> None:
    world = World(store_path=tmp_path / "w", llm=None)
    result = world.ingest(
        Observation(
            concepts=["Sharding", "Latency"],
            contradicted_relations=[("Sharding", "Latency", "membership")],
            task="t",
        )
    )
    assert sorted(result.weakened_concepts) == ["Latency", "Sharding"]
    assert result.weakened_relations == []
    # disconfirmation_count rose on both endpoints in the persisted world.
    for node in world.concepts.all():
        assert node.disconfirmation_count >= 1
