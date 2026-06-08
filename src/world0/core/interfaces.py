"""Cross-module Protocols for World 0.

These contracts are the *only* thing subsystems may share.  Concrete
classes in ``concepts/``, ``relations/``, ``dynamics/*`` etc. are
expected to satisfy the relevant Protocol structurally — no inheritance
needed (PEP 544).

Read this file to understand the public surface every Lego brick must
provide; read the Protocol bodies to understand exactly which methods
are part of the contract (anything not listed here is an implementation
detail and may be changed freely).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from world0.schemas.community import Community
    from world0.schemas.concept import ConceptNode, Maturity
    from world0.schemas.context import Perspective
    from world0.schemas.relation import RelationEdge, RelationType
    from world0.schemas.source import SourceRecord
    from world0.schemas.types import Observation, Projection


# ── Persistence ───────────────────────────────────────────────────────


@runtime_checkable
class StorageBackend(Protocol):
    """Pluggable persistence for concepts, relations and world state.

    A concrete backend must persist ``ConceptNode`` and ``RelationEdge``
    instances individually (so single-item updates do not require
    reading every record back) and must support a free-form ``state``
    dict for cross-cycle coordination.
    """

    # concepts
    def save_concept(self, concept: ConceptNode) -> None: ...
    def load_concept(self, concept_id: str) -> ConceptNode | None: ...
    def load_all_concepts(self) -> list[ConceptNode]: ...
    def delete_concept(self, concept_id: str) -> None: ...
    def save_concepts_batch(self, concepts: list[ConceptNode]) -> None: ...
    def delete_concepts_batch(self, concept_ids: list[str]) -> None: ...

    # relations
    def save_relation(self, relation: RelationEdge) -> None: ...
    def load_relation(self, relation_id: str) -> RelationEdge | None: ...
    def load_all_relations(self) -> list[RelationEdge]: ...
    def delete_relation(self, relation_id: str) -> None: ...
    def save_relations_batch(self, relations: list[RelationEdge]) -> None: ...
    def delete_relations_batch(self, relation_ids: list[str]) -> None: ...

    # sources
    def save_source(self, source: SourceRecord) -> None: ...
    def load_source(self, source_id: str) -> SourceRecord | None: ...
    def load_all_sources(self) -> list[SourceRecord]: ...

    # state
    def save_state(self, state: dict) -> None: ...
    def load_state(self) -> dict: ...


# ── Concept storage ───────────────────────────────────────────────────


@runtime_checkable
class ConceptStoreReader(Protocol):
    """Read-only slice of ConceptStore — used by visualization, decay,
    activation, projection and any other consumer that must not mutate
    concept state.
    """

    def get(self, concept_id: str) -> ConceptNode | None: ...
    def resolve(self, name_or_id: str) -> ConceptNode | None: ...
    def all(self) -> list[ConceptNode]: ...
    def by_maturity(self, maturity: Maturity) -> list[ConceptNode]: ...
    def find_similar(
        self,
        text: str,
        *,
        domain: str = ...,
        min_similarity: float = ...,
        limit: int = ...,
    ) -> list[tuple[ConceptNode, float]]: ...
    def __len__(self) -> int: ...
    def __contains__(self, concept_id: str) -> bool: ...


@runtime_checkable
class ConceptStore(ConceptStoreReader, Protocol):
    """Full concept lifecycle — read + write.

    Implementations are typically thin wrappers around a ``StorageBackend``
    with in-memory indexes.  Anything mutating must mark the affected
    concept dirty so ``flush()`` can persist incrementally.
    """

    def get_or_create(
        self,
        name: str,
        *,
        origin: str = ...,
        task: str = ...,
        description: str = ...,
        kind: str = ...,
        sense: str = ...,
        domain: str = ...,
        aliases: list[str] | None = ...,
        identity_key: str = ...,
        consolidate: bool = ...,
    ) -> tuple[ConceptNode, bool]: ...

    def reinforce(
        self, concept_id: str, *, source: str = ..., task: str = ...
    ) -> ConceptNode | None: ...
    def weaken(
        self, concept_id: str, *, source: str = ..., task: str = ...
    ) -> ConceptNode | None: ...

    def update_description(self, concept_id: str, description: str) -> None: ...
    def add_alias(self, concept_id: str, alias: str) -> bool: ...
    def set_aliases(self, concept_id: str, aliases: list[str]) -> None: ...
    def update_maturity(self, concept_id: str, maturity: Maturity) -> None: ...
    def adjust_confidence(
        self, concept_id: str, delta: float
    ) -> ConceptNode | None: ...

    def remove(self, concept_id: str) -> bool: ...

    def merge(
        self,
        keeper_id: str,
        absorbed_id: str,
        relations: RelationStore | None = ...,
    ) -> ConceptNode | None: ...
    def split(
        self,
        concept_id: str,
        new_name: str,
        *,
        aliases_to_move: list[str] | None = ...,
        description: str = ...,
        domain: str = ...,
    ) -> ConceptNode | None: ...

    def mark_dirty(self, concept_id: str) -> None: ...
    def flush(self) -> None: ...
    def save_all(self) -> None: ...
    def load(self) -> None: ...

    def connection_count(self, concept_id: str, relations: list) -> int: ...


# ── Relation storage ─────────────────────────────────────────────────


@runtime_checkable
class RelationStoreReader(Protocol):
    """Read-only slice of RelationStore."""

    def get(self, relation_id: str) -> RelationEdge | None: ...
    def all(self) -> list[RelationEdge]: ...
    def for_concept(self, concept_id: str) -> list[RelationEdge]: ...
    def neighbors(self, concept_id: str) -> list[str]: ...
    def find_between(
        self,
        id_a: str,
        id_b: str,
        relation_type: RelationType | None = ...,
    ) -> RelationEdge | None: ...
    def find_any_between(
        self, id_a: str, id_b: str
    ) -> list[RelationEdge]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class RelationStore(RelationStoreReader, Protocol):
    """Full relation lifecycle."""

    def discover(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType = ...,
        *,
        semantic_relation: str = ...,
        provenance: str = ...,
        is_explicit: bool = ...,
        probability: float | None = ...,
        prior_probability: float | None = ...,
        prior_strength: float = ...,
        evidence_strength: float = ...,
    ) -> tuple[RelationEdge, bool]: ...

    def reinforce(
        self, relation_id: str, provenance: str = ...
    ) -> RelationEdge | None: ...
    def weaken(
        self, relation_id: str, provenance: str = ...
    ) -> RelationEdge | None: ...
    def refine_type(self, relation_id: str, new_type: RelationType) -> None: ...
    def adjust_strength(
        self,
        relation_id: str,
        *,
        weight_delta: float = ...,
        confidence_delta: float = ...,
    ) -> RelationEdge | None: ...

    def remove(self, relation_id: str) -> bool: ...
    def remove_for_concept(self, concept_id: str) -> int: ...
    def migrate_concept(self, old_id: str, new_id: str) -> int: ...

    def mark_dirty(self, relation_id: str) -> None: ...
    def flush(self) -> None: ...
    def save_all(self) -> None: ...
    def load(self) -> None: ...


# ── Cognitive Dynamics ───────────────────────────────────────────────
#
# Dynamics is *not* a single Protocol — different engines do different
# things (spread activation, learn relations, decay, evaluate maturity,
# detect communities, diffuse color).  Forcing them into one signature
# would be an abstraction lie.  Instead we declare one Protocol per
# engine *kind*, each expressing the minimal shape its consumers (the
# pipelines in ``world/``) actually need.


@runtime_checkable
class ActivationProvider(Protocol):
    """Spreading activation through the relation network."""

    def activate(
        self,
        seed_ids: list[str],
        *,
        max_depth: int = ...,
        decay: float = ...,
        min_activation: float = ...,
        source: str = ...,
        task: str = ...,
        record: bool = ...,
        perspective: Perspective | None = ...,
    ) -> dict[str, float]: ...


@runtime_checkable
class HebbianLearner(Protocol):
    """Co-activation → relation discovery / reinforcement."""

    def learn(
        self, concept_ids: list[str], *, provenance: str = ...
    ) -> list[str]: ...


@runtime_checkable
class DecayPolicy(Protocol):
    """Time-based decay + pruning."""

    def decay_concepts(self) -> list[str]: ...
    def decay_relations(self) -> list[str]: ...
    def prune_concepts(self, threshold: float = ...) -> list[str]: ...
    def prune_relations(self, threshold: float = ...) -> list[str]: ...


@runtime_checkable
class LifecyclePolicy(Protocol):
    """Maturity transitions for concepts."""

    def evaluate(self) -> tuple[list[str], list[str]]: ...


@runtime_checkable
class CommunityDetectorP(Protocol):
    """Stateless community detection over the current concept graph."""

    def detect(
        self, *, max_iters: int = ..., min_size: int = ...
    ) -> list[Community]: ...


@runtime_checkable
class ColorField(Protocol):
    """Domain-color seeding, diffusion and per-component fade."""

    def seed_and_diffuse(
        self,
        concept_ids: list[str],
        *,
        domain_label: str,
        steps: int = ...,
        rate: float = ...,
        decay: float = ...,
    ) -> None: ...

    def settle(self, steps: int = ..., rate: float = ...) -> None: ...

    def fade_step(
        self, *, dt: float = ..., tau: float = ..., evaporate: float = ...
    ) -> int: ...

    def seed_from_communities(
        self, communities: list[Community], *, diffuse: bool = ...
    ) -> int: ...


# ── Projection / Extraction ──────────────────────────────────────────


@runtime_checkable
class Projector(Protocol):
    """Build a Projection from raw activation scores."""

    def project(
        self,
        activations: dict[str, float],
        *,
        max_concepts: int = ...,
        min_activation: float = ...,
        task: str = ...,
    ) -> Projection: ...


@runtime_checkable
class Extractor(Protocol):
    """Turn raw text into a structured Observation."""

    def extract(
        self, text: str, *, task: str = ..., source: str = ...
    ) -> Observation: ...


# ── LLM provider (re-export of llm/base.py contract) ────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface for LLM calls — single-shot JSON completion.

    Mirrors ``world0.llm.base.LLMProvider`` (which remains an ABC for
    backwards compatibility with concrete provider implementations).
    Use this Protocol when you only consume an LLM and want to type
    against a structural contract instead of an ABC.
    """

    def complete_json(self, system: str, user: str) -> str: ...


# ── World view (read-only composite) ─────────────────────────────────


@runtime_checkable
class WorldView(Protocol):
    """Read-only composite view onto a World.

    Visualization, status reporting and any read-only consumer should
    depend on this Protocol instead of the concrete ``World`` class.
    Anything wanting to mutate state should depend on the relevant
    ``ConceptStore`` / ``RelationStore`` directly.
    """

    @property
    def concepts(self) -> ConceptStoreReader: ...

    @property
    def relations(self) -> RelationStoreReader: ...
