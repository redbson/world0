"""Dynamics bricks — testable against ConceptStore / RelationStore Protocols.

Each engine here gets *only* fakes from ``world0.core.test_doubles``.
This proves the engines do not depend on the concrete ``ConceptManager``
or ``RelationManager`` — only on the read/write Protocols those
managers happen to implement.
"""

from __future__ import annotations

from world0.core import (
    ActivationProvider,
    ConceptStore,
    DecayPolicy,
    HebbianLearner,
    LifecyclePolicy,
    RelationStore,
)
from world0.core.test_doubles import (
    FakeConceptStore,
    FakeRelationStore,
    make_concept,
    make_edge,
)
from world0.dynamics.activation import ActivationEngine
from world0.dynamics.decay import DecayEngine
from world0.dynamics.hebbian import COOCCURRENCE_THRESHOLD, HebbianEngine
from world0.dynamics.lifecycle import LifecycleEngine
from world0.schemas.relation import RelationType


# ── Fakes satisfy the Protocols ────────────────────────────────────


def test_fake_concept_store_satisfies_protocol() -> None:
    assert isinstance(FakeConceptStore(), ConceptStore)


def test_fake_relation_store_satisfies_protocol() -> None:
    assert isinstance(FakeRelationStore(), RelationStore)


# ── Engines satisfy their Protocols ────────────────────────────────


def test_engines_satisfy_their_protocols() -> None:
    cs = FakeConceptStore()
    rs = FakeRelationStore()
    assert isinstance(ActivationEngine(cs, rs), ActivationProvider)
    assert isinstance(HebbianEngine(rs), HebbianLearner)
    assert isinstance(DecayEngine(cs, rs), DecayPolicy)
    assert isinstance(LifecycleEngine(cs, rs), LifecyclePolicy)


# ── Hebbian: works with fake stores only ───────────────────────────


def test_hebbian_creates_relation_after_threshold_crossed() -> None:
    rs = FakeRelationStore()
    h = HebbianEngine(rs)

    a, b = "a-id", "b-id"

    new = h.learn([a, b], provenance="task-1")
    assert new == [], "first co-occurrence is below threshold"

    # Drive co-occurrence to threshold.
    for _ in range(COOCCURRENCE_THRESHOLD - 1):
        new = h.learn([a, b], provenance="task-1")
    assert len(new) == 1
    edge = rs.get(new[0])
    assert edge is not None
    assert edge.relation_type == RelationType.PARALLEL
    assert edge.is_explicit is False


def test_hebbian_reinforces_existing_relation() -> None:
    rs = FakeRelationStore()
    edge = make_edge("x", "y", weight=0.3)
    rs._edges[edge.id] = edge  # seed
    h = HebbianEngine(rs)

    h.learn(["x", "y"], provenance="t")
    refreshed = rs.get(edge.id)
    assert refreshed is not None
    assert refreshed.reinforcement_count == 1


# ── Activation: works with fake stores only ────────────────────────


def test_activation_spreads_through_fake_relation_store() -> None:
    a = make_concept("alpha", confidence=0.8)
    b = make_concept("beta", confidence=0.8)
    cs = FakeConceptStore(seed=[a, b])
    rs = FakeRelationStore(seed=[make_edge(a.id, b.id, weight=0.6)])

    engine = ActivationEngine(cs, rs)
    scores = engine.activate([a.id], max_depth=1, decay=0.5, record=False)

    assert scores[a.id] == max(scores.values())
    assert scores.get(b.id, 0.0) > 0.0


# ── Decay: only touches Protocol surface ───────────────────────────


def test_decay_returns_canned_state_through_fakes() -> None:
    cs = FakeConceptStore()
    rs = FakeRelationStore()
    d = DecayEngine(cs, rs)

    # No nodes → empty lists, no errors.
    assert d.decay_concepts() == []
    assert d.decay_relations() == []
    assert d.prune_concepts() == []
    assert d.prune_relations() == []
