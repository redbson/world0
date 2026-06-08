"""Deep, deterministic unit tests for the three-axis relation model.

Covers schema-level helpers and RelationEdge dynamics in
``world0.schemas.relation`` without touching any store.  These tests
complement ``tests/test_relation_axes.py`` and
``tests/test_relation_probability.py`` (which exercise the World facade)
by drilling into the pure functions and edge-level math directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from world0.schemas.relation import (
    SEMANTIC_RELATION_SPECS,
    RelationEdge,
    RelationType,
    is_known_relation_type,
    normalize_relation_type,
    normalize_semantic_relation,
    relation_axis_descriptions,
    semantic_relation_names,
    semantic_relation_spec,
)


def _edge(**kwargs) -> RelationEdge:
    base = {"source_id": "a", "target_id": "b"}
    base.update(kwargs)
    return RelationEdge(**base)


# ---------------------------------------------------------------------------
# normalize_relation_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, axis",
    [
        # Legacy relation labels.
        ("supports", RelationType.POSITIVE),
        ("depends_on", RelationType.POSITIVE),
        ("contains", RelationType.POSITIVE),
        ("part_of", RelationType.POSITIVE),
        ("activates", RelationType.POSITIVE),
        ("precedes", RelationType.POSITIVE),
        ("derived_from", RelationType.POSITIVE),
        ("contrasts", RelationType.NEGATIVE),
        ("similar_to", RelationType.PARALLEL),
        ("related_to", RelationType.PARALLEL),
        # Axis words.
        ("positive", RelationType.POSITIVE),
        ("attraction", RelationType.POSITIVE),
        ("trust", RelationType.POSITIVE),
        ("co_creation", RelationType.POSITIVE),
        ("future_coupling", RelationType.POSITIVE),
        ("mutual_reinforcement", RelationType.POSITIVE),
        ("negative", RelationType.NEGATIVE),
        ("repulsion", RelationType.NEGATIVE),
        ("conflict", RelationType.NEGATIVE),
        ("incompatible_ontology", RelationType.NEGATIVE),
        ("instability", RelationType.NEGATIVE),
        ("adversarial_prediction", RelationType.NEGATIVE),
        ("parallel", RelationType.PARALLEL),
        ("resonance", RelationType.PARALLEL),
        ("mutual_understanding", RelationType.PARALLEL),
        ("deep_conceptual_overlap", RelationType.PARALLEL),
        ("recursive_co_modeling", RelationType.PARALLEL),
        ("persistent_attention_allocation", RelationType.PARALLEL),
    ],
)
def test_normalize_relation_type_known_labels(label, axis):
    assert normalize_relation_type(label) is axis


def test_normalize_relation_type_is_case_and_space_insensitive():
    assert normalize_relation_type("  SUPPORTS  ") is RelationType.POSITIVE
    assert normalize_relation_type("co creation") is RelationType.POSITIVE
    assert normalize_relation_type("Co-Creation") is RelationType.POSITIVE


@pytest.mark.parametrize("bad", ["unknown", "zzz", "", "   ", None])
def test_normalize_relation_type_unknown_and_empty_default_parallel(bad):
    assert normalize_relation_type(bad) is RelationType.PARALLEL


def test_normalize_relation_type_passthrough_for_enum():
    for axis in (RelationType.POSITIVE, RelationType.NEGATIVE, RelationType.PARALLEL):
        assert normalize_relation_type(axis) is axis
    # Enum aliases collapse onto their backing axis value.
    assert normalize_relation_type(RelationType.SUPPORTS) is RelationType.POSITIVE
    assert normalize_relation_type(RelationType.CONTRASTS) is RelationType.NEGATIVE
    assert normalize_relation_type(RelationType.SIMILAR_TO) is RelationType.PARALLEL


# ---------------------------------------------------------------------------
# normalize_semantic_relation + aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, canonical",
    [
        # Legacy relation labels.
        ("supports", "enables"),
        ("depends_on", "dependence"),
        ("contains", "inclusion"),
        ("part_of", "membership"),
        ("activates", "enables"),
        ("precedes", "dependence"),
        ("derived_from", "dependence"),
        ("contrasts", "conflict"),
        ("similar_to", "similarity_kernel"),
        ("related_to", "generic_relation"),
        # Axis words.
        ("positive", "mutual_reinforcement"),
        ("attraction", "mutual_reinforcement"),
        ("negative", "conflict"),
        ("repulsion", "conflict"),
        ("parallel", "generic_relation"),
        ("resonance", "overlap"),
        # Prior semantic labels.
        ("trust", "mutual_reinforcement"),
        ("mutual_understanding", "equivalence"),
        ("deep_conceptual_overlap", "overlap"),
        ("persistent_attention_allocation", "persistent_attention"),
        # Canonical names map to themselves.
        ("disjointness", "disjointness"),
        ("equivalence", "equivalence"),
    ],
)
def test_normalize_semantic_relation_maps_to_canonical(label, canonical):
    assert normalize_semantic_relation(label) == canonical


def test_normalize_semantic_relation_handles_spaces_and_case():
    assert normalize_semantic_relation("deep conceptual overlap") == "overlap"
    assert normalize_semantic_relation("  Supports ") == "enables"


@pytest.mark.parametrize("bad", ["", "   ", None, "definitely_not_a_relation"])
def test_normalize_semantic_relation_unknown_defaults_generic(bad):
    assert normalize_semantic_relation(bad) == "generic_relation"


# ---------------------------------------------------------------------------
# semantic_relation_spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, axis, structural, propagation",
    [
        ("membership", RelationType.POSITIVE, 0.94, 0.88),
        ("enables", RelationType.POSITIVE, 0.82, 0.76),
        ("dependence", RelationType.POSITIVE, 0.78, 0.70),
        ("disjointness", RelationType.NEGATIVE, 0.95, 0.05),
        ("conflict", RelationType.NEGATIVE, 0.84, 0.10),
        ("equivalence", RelationType.PARALLEL, 0.96, 0.92),
        ("overlap", RelationType.PARALLEL, 0.66, 0.60),
        ("generic_relation", RelationType.PARALLEL, 0.55, 0.45),
    ],
)
def test_semantic_relation_spec_axis_and_strengths(name, axis, structural, propagation):
    spec = semantic_relation_spec(name)
    assert spec.name == name
    assert spec.axis is axis
    assert spec.structural_strength == pytest.approx(structural)
    assert spec.propagation_strength == pytest.approx(propagation)


def test_semantic_relation_spec_resolves_aliases():
    # Goes through normalization first.
    assert semantic_relation_spec("supports") is SEMANTIC_RELATION_SPECS["enables"]
    assert semantic_relation_spec("unknown") is SEMANTIC_RELATION_SPECS["generic_relation"]


def test_negative_axis_relations_have_low_propagation():
    negatives = [
        spec
        for spec in SEMANTIC_RELATION_SPECS.values()
        if spec.axis is RelationType.NEGATIVE
    ]
    assert negatives  # sanity
    for spec in negatives:
        # Negative relations should not propagate activation strongly.
        assert spec.propagation_strength <= 0.15
        # Yet they remain structurally meaningful.
        assert spec.structural_strength >= 0.7


# ---------------------------------------------------------------------------
# semantic_relation_names / is_known_relation_type / axis descriptions
# ---------------------------------------------------------------------------


def test_semantic_relation_names_no_axis_returns_all_sorted():
    names = semantic_relation_names()
    assert names == sorted(SEMANTIC_RELATION_SPECS)
    assert len(names) == len(SEMANTIC_RELATION_SPECS)


def test_semantic_relation_names_filtered_by_axis():
    for axis in (RelationType.POSITIVE, RelationType.NEGATIVE, RelationType.PARALLEL):
        names = semantic_relation_names(axis)
        assert names == sorted(names)
        assert names  # each axis has at least one relation
        for name in names:
            assert SEMANTIC_RELATION_SPECS[name].axis is axis
    # Partition: the three axes together cover every spec exactly once.
    union = (
        set(semantic_relation_names(RelationType.POSITIVE))
        | set(semantic_relation_names(RelationType.NEGATIVE))
        | set(semantic_relation_names(RelationType.PARALLEL))
    )
    assert union == set(SEMANTIC_RELATION_SPECS)


def test_semantic_relation_names_accepts_axis_word():
    assert semantic_relation_names("negative") == semantic_relation_names(
        RelationType.NEGATIVE
    )


@pytest.mark.parametrize(
    "label, known",
    [
        ("supports", True),
        ("disjointness", True),
        ("positive", True),
        ("co creation", True),  # space-normalized legacy alias
        (RelationType.POSITIVE, True),
        ("", False),
        ("   ", False),
        (None, False),
        ("not_a_real_relation", False),
    ],
)
def test_is_known_relation_type(label, known):
    assert is_known_relation_type(label) is known


def test_relation_axis_descriptions_structure():
    desc = relation_axis_descriptions()
    assert set(desc.keys()) == {"positive", "negative", "parallel"}
    for axis_name, items in desc.items():
        assert isinstance(items, list)
        assert items, f"axis {axis_name} should list descriptive phrases"
        assert all(isinstance(item, str) for item in items)
    assert "conflict" in desc["negative"]
    assert "trust" in desc["positive"]


# ---------------------------------------------------------------------------
# RelationEdge validators + model_validator
# ---------------------------------------------------------------------------


def test_relation_type_validator_coerces_legacy_string():
    edge = _edge(relation_type="contrasts")
    assert edge.relation_type is RelationType.NEGATIVE


def test_semantic_relation_validator_coerces_legacy_string():
    edge = _edge(semantic_relation="supports")
    assert edge.semantic_relation == "enables"


def test_model_validator_infers_semantic_relation_from_axis():
    pos = _edge(relation_type="positive")
    assert pos.semantic_relation == "mutual_reinforcement"
    assert pos.relation_type is RelationType.POSITIVE

    neg = _edge(relation_type="negative")
    assert neg.semantic_relation == "conflict"
    assert neg.relation_type is RelationType.NEGATIVE

    par = _edge(relation_type="parallel")
    assert par.semantic_relation == "generic_relation"
    assert par.relation_type is RelationType.PARALLEL


def test_model_validator_applies_spec_strengths():
    edge = _edge(semantic_relation="disjointness")
    spec = SEMANTIC_RELATION_SPECS["disjointness"]
    assert edge.relation_type is RelationType.NEGATIVE
    assert edge.structural_strength == pytest.approx(spec.structural_strength)
    assert edge.propagation_strength == pytest.approx(spec.propagation_strength)


def test_model_validator_backfills_defaults_from_spec():
    # All of probability/weight/confidence at defaults -> driven by spec.
    edge = _edge(semantic_relation="enables")
    spec = SEMANTIC_RELATION_SPECS["enables"]
    assert edge.probability == pytest.approx(spec.propagation_strength)
    assert edge.weight == pytest.approx(spec.propagation_strength)
    assert edge.confidence == pytest.approx(spec.structural_strength)


def test_model_validator_does_not_backfill_when_values_explicit():
    edge = _edge(
        semantic_relation="enables",
        probability=0.7,
        weight=0.6,
        confidence=0.5,
    )
    assert edge.probability == pytest.approx(0.7)
    assert edge.weight == pytest.approx(0.6)
    assert edge.confidence == pytest.approx(0.5)


def test_explicit_semantic_relation_overrides_inferred_axis():
    # relation_type says positive but the explicit semantic relation is
    # a negative one; the spec axis wins.
    edge = _edge(relation_type="positive", semantic_relation="conflict")
    assert edge.semantic_relation == "conflict"
    assert edge.relation_type is RelationType.NEGATIVE


# ---------------------------------------------------------------------------
# reinforce
# ---------------------------------------------------------------------------


def test_reinforce_diminishing_returns_explicit():
    edge = _edge(semantic_relation="overlap", is_explicit=True)
    # overlap: weight starts at propagation 0.60, far from cap, so boosts
    # are visible and strictly diminishing.
    deltas = []
    prev = edge.weight
    for _ in range(3):
        edge.reinforce()
        deltas.append(edge.weight - prev)
        prev = edge.weight
    assert all(d > 0 for d in deltas)
    assert deltas[0] > deltas[1] > deltas[2]


def test_reinforce_explicit_can_approach_one():
    edge = _edge(semantic_relation="overlap", is_explicit=True)
    for _ in range(300):
        edge.reinforce()
    assert edge.weight == pytest.approx(1.0)
    assert edge.confidence == pytest.approx(1.0)


def test_reinforce_hebbian_capped_at_07():
    edge = _edge(semantic_relation="overlap", is_explicit=False)
    for _ in range(500):
        edge.reinforce()
    assert edge.weight == pytest.approx(0.7)
    assert edge.confidence == pytest.approx(0.7)
    assert edge.weight <= 0.7 + 1e-9
    assert edge.confidence <= 0.7 + 1e-9


def test_reinforce_updates_count_timestamp_and_history():
    edge = _edge(semantic_relation="overlap")
    before = edge.last_reinforced
    edge.reinforce("task-x")
    assert edge.reinforcement_count == 1
    assert edge.last_reinforced >= before
    assert edge.provenance == "task-x"
    assert edge.task_history == ["task-x"]
    # Repeated provenance is not duplicated.
    edge.reinforce("task-x")
    assert edge.reinforcement_count == 2
    assert edge.task_history == ["task-x"]
    # No provenance leaves history untouched.
    edge.reinforce()
    assert edge.reinforcement_count == 3
    assert edge.task_history == ["task-x"]


# ---------------------------------------------------------------------------
# weaken
# ---------------------------------------------------------------------------


def test_weaken_diminishing_penalty():
    edge = _edge(semantic_relation="enables")
    penalties = []
    prev = edge.confidence
    for _ in range(4):
        edge.weaken()
        penalties.append(prev - edge.confidence)
        prev = edge.confidence
    assert all(p > 0 for p in penalties)
    # Penalty magnitude shrinks with each disconfirmation.
    assert penalties[0] > penalties[1] > penalties[2] > penalties[3]


def test_weaken_respects_floor():
    edge = _edge(semantic_relation="enables")
    for _ in range(1000):
        edge.weaken()
    assert edge.weight == pytest.approx(0.01)
    assert edge.confidence == pytest.approx(0.01)


def test_weaken_probability_tracks_confidence_and_updates_metadata():
    edge = _edge(semantic_relation="enables")
    edge.weaken("disc-1")
    assert edge.probability == pytest.approx(edge.confidence)
    assert edge.disconfirmation_count == 1
    assert edge.last_weakened is not None
    assert edge.task_history == ["disc-1"]
    edge.weaken("disc-1")  # no duplicate
    assert edge.task_history == ["disc-1"]
    assert edge.disconfirmation_count == 2


# ---------------------------------------------------------------------------
# update_probability
# ---------------------------------------------------------------------------


def test_update_probability_evidence_blend_and_counts():
    edge = _edge(semantic_relation="enables")  # probability starts at 0.76
    assert edge.probability == pytest.approx(0.76)
    # total_strength = max(1, 2) = 2 ; total = 0.76 * 2 = 1.52
    # evidence 0.9 * 2.0 = 1.8 -> total 3.32 ; strength 4 -> 0.83
    edge.update_probability(evidence_probability=0.9)
    assert edge.probability == pytest.approx(0.83)
    assert edge.weight == pytest.approx(0.83)
    assert edge.confidence == pytest.approx(0.83)
    assert edge.probability_observation_count == 1
    # evidence >= 0.5 counts as reinforcement.
    assert edge.reinforcement_count == 1
    assert edge.disconfirmation_count == 0


def test_update_probability_low_evidence_is_disconfirmation():
    edge = _edge(semantic_relation="enables")
    edge.update_probability(evidence_probability=0.1)
    # 0.76*2=1.52 + 0.1*2=0.2 -> 1.72 / 4 = 0.43
    assert edge.probability == pytest.approx(0.43)
    assert edge.probability_observation_count == 1
    assert edge.reinforcement_count == 0
    assert edge.disconfirmation_count == 1


def test_update_probability_blends_prior_and_evidence():
    edge = _edge(semantic_relation="enables")
    edge.update_probability(
        prior_probability=0.2,
        prior_strength=1.0,
        evidence_probability=0.5,
        evidence_strength=2.0,
    )
    # 0.76*2=1.52 ; +0.2*1=0.2 -> 1.72 (ts 3) ; +0.5*2=1.0 -> 2.72 (ts 5)
    assert edge.probability == pytest.approx(2.72 / 5.0)
    # prior never bumps reinforce/disconfirm; only evidence does (0.5 -> reinforce)
    assert edge.reinforcement_count == 1
    assert edge.disconfirmation_count == 0


def test_update_probability_noop_without_prior_or_evidence():
    edge = _edge(semantic_relation="enables")
    p0 = edge.probability
    edge.update_probability()
    assert edge.probability == pytest.approx(p0)
    assert edge.probability_observation_count == 0
    assert edge.reinforcement_count == 0
    assert edge.disconfirmation_count == 0


def test_update_probability_clamps_out_of_range_evidence():
    edge = _edge(semantic_relation="enables")
    edge.update_probability(evidence_probability=5.0)  # clamped to 1.0
    # 1.52 + 1.0*2 = 3.52 / 4 = 0.88
    assert edge.probability == pytest.approx(0.88)
    assert 0.0 <= edge.probability <= 1.0


def test_update_probability_sets_provenance():
    edge = _edge(semantic_relation="enables")
    edge.update_probability(evidence_probability=0.9, provenance="run-7")
    assert edge.provenance == "run-7"
    assert edge.task_history == ["run-7"]


# ---------------------------------------------------------------------------
# ensure_probability
# ---------------------------------------------------------------------------


def test_ensure_probability_backfills_from_confidence():
    # Construct an edge that mimics a legacy record: default probability 0.3
    # but a non-default confidence and no observations.
    edge = RelationEdge.model_construct(
        source_id="a",
        target_id="b",
        relation_type=RelationType.PARALLEL,
        semantic_relation="generic_relation",
        structural_strength=0.55,
        propagation_strength=0.45,
        probability=0.3,
        probability_observation_count=0,
        weight=0.3,
        is_explicit=False,
        confidence=0.9,
        reinforcement_count=0,
        disconfirmation_count=0,
        last_reinforced=datetime.now(timezone.utc),
        last_weakened=None,
        discovered_at=datetime.now(timezone.utc),
        provenance="",
        task_history=[],
    )
    edge.ensure_probability()
    assert edge.probability == pytest.approx(0.9)


def test_ensure_probability_noop_when_confidence_is_default():
    edge = RelationEdge.model_construct(
        source_id="a",
        target_id="b",
        relation_type=RelationType.PARALLEL,
        semantic_relation="generic_relation",
        structural_strength=0.55,
        propagation_strength=0.45,
        probability=0.3,
        probability_observation_count=0,
        weight=0.3,
        is_explicit=False,
        confidence=0.3,
        reinforcement_count=0,
        disconfirmation_count=0,
        last_reinforced=datetime.now(timezone.utc),
        last_weakened=None,
        discovered_at=datetime.now(timezone.utc),
        provenance="",
        task_history=[],
    )
    edge.ensure_probability()
    assert edge.probability == pytest.approx(0.3)


def test_ensure_probability_noop_when_observed():
    edge = RelationEdge.model_construct(
        source_id="a",
        target_id="b",
        relation_type=RelationType.PARALLEL,
        semantic_relation="generic_relation",
        structural_strength=0.55,
        propagation_strength=0.45,
        probability=0.3,
        probability_observation_count=2,
        weight=0.3,
        is_explicit=False,
        confidence=0.9,
        reinforcement_count=0,
        disconfirmation_count=0,
        last_reinforced=datetime.now(timezone.utc),
        last_weakened=None,
        discovered_at=datetime.now(timezone.utc),
        provenance="",
        task_history=[],
    )
    edge.ensure_probability()
    assert edge.probability == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# beta_posterior + evidence_balance
# ---------------------------------------------------------------------------


def test_beta_posterior_defaults_uniform_prior():
    edge = _edge(semantic_relation="enables")
    alpha, beta = edge.beta_posterior()
    assert (alpha, beta) == (1.0, 1.0)


def test_beta_posterior_adds_counts():
    edge = _edge(semantic_relation="enables")
    edge.reinforcement_count = 3
    edge.disconfirmation_count = 1
    alpha, beta = edge.beta_posterior()
    assert alpha == pytest.approx(4.0)
    assert beta == pytest.approx(2.0)
    # Custom prior shifts both terms.
    alpha2, beta2 = edge.beta_posterior(prior_alpha=2.0, prior_beta=5.0)
    assert alpha2 == pytest.approx(5.0)
    assert beta2 == pytest.approx(6.0)


def test_evidence_balance_posterior_mean():
    edge = _edge(semantic_relation="enables")
    # No evidence -> uniform -> 0.5
    assert edge.evidence_balance() == pytest.approx(0.5)
    edge.reinforcement_count = 3
    edge.disconfirmation_count = 1
    # (1+3) / ((1+3)+(1+1)) = 4/6
    assert edge.evidence_balance() == pytest.approx(4.0 / 6.0)


def test_evidence_balance_moves_toward_one_with_reinforcement():
    edge = _edge(semantic_relation="enables")
    edge.reinforcement_count = 50
    edge.disconfirmation_count = 0
    assert edge.evidence_balance() > 0.95


# ---------------------------------------------------------------------------
# temporal_relevance
# ---------------------------------------------------------------------------


def test_temporal_relevance_is_one_when_fresh():
    edge = _edge(semantic_relation="enables")
    # Just reinforced (hours_since ~ 0): essentially full relevance.
    assert edge.temporal_relevance() == pytest.approx(1.0, abs=1e-3)


def test_temporal_relevance_half_at_one_half_life():
    edge = _edge(semantic_relation="enables")
    edge.reinforcement_count = 0  # effective half-life == base
    edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=72)
    assert edge.temporal_relevance(72.0) == pytest.approx(0.5, abs=1e-3)


def test_temporal_relevance_decays_monotonically():
    def rel(hours: float) -> float:
        edge = _edge(semantic_relation="enables")
        edge.reinforcement_count = 0
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=hours)
        return edge.temporal_relevance(72.0)

    values = [rel(h) for h in (1, 24, 72, 144, 240)]
    assert all(earlier >= later for earlier, later in zip(values, values[1:]))


def test_temporal_relevance_floor():
    edge = _edge(semantic_relation="enables")
    edge.reinforcement_count = 0
    edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=100_000)
    assert edge.temporal_relevance(72.0) == pytest.approx(0.15)


def test_temporal_relevance_longer_half_life_with_more_reinforcement():
    age = 72.0

    def rel(reinforcements: int) -> float:
        edge = _edge(semantic_relation="enables")
        edge.reinforcement_count = reinforcements
        edge.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=age)
        return edge.temporal_relevance(72.0)

    weak = rel(0)
    strong = rel(4)
    # More reinforced relations decay slower -> higher relevance at same age.
    assert strong > weak
    assert weak == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# involves / other_end
# ---------------------------------------------------------------------------


def test_involves():
    edge = _edge(source_id="x", target_id="y", semantic_relation="enables")
    assert edge.involves("x")
    assert edge.involves("y")
    assert not edge.involves("z")


def test_other_end():
    edge = _edge(source_id="x", target_id="y", semantic_relation="enables")
    assert edge.other_end("x") == "y"
    assert edge.other_end("y") == "x"
    assert edge.other_end("z") is None
