"""Input/output types for the World 0 Agent interface."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from world0.schemas.concept import ConceptNode
from world0.schemas.relation import RelationEdge


class ConceptCandidate(BaseModel):
    """A pre-ingest concept sense, not yet a stable concept node.

    ``uid`` is local to one observation/extraction response and is used
    by relations to point at the intended sense.  The persistent concept
    UID is assigned by ``ConceptManager`` after identity resolution.
    """

    uid: str = ""
    name: str
    kind: str = ""
    sense: str = ""
    domain: str = ""
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    salience: float | None = None
    confidence: float | None = None
    evidence: str = ""


class RelationPrior(BaseModel):
    """A preset relation probability used during extraction/ingest."""

    source: str
    target: str
    relation_type: str = "generic_relation"
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0)
    rationale: str = ""


class Observation(BaseModel):
    """What the Agent feeds into World 0 after working on a task.

    The Agent (an LLM) does the semantic extraction. World 0 does the
    structural and cognitive computation.

    In addition to the positive evidence channel (``concepts`` and
    ``relations``), an observation can also carry *negative* evidence:
    concepts that the Agent has reason to believe are wrong or
    irrelevant for the current task (``weakened``) and pairs of
    concepts whose asserted relation did not hold
    (``contradicted_relations``).  Both feed into Beta-style
    confidence updates.
    """

    concepts: list[str] = Field(default_factory=list)
    concept_candidates: list[ConceptCandidate] = Field(default_factory=list)
    relations: list[tuple[str, str, str]] = Field(default_factory=list)
    relation_priors: list[RelationPrior] = Field(default_factory=list)
    descriptions: dict[str, str] = Field(default_factory=dict)
    weakened: list[str] = Field(default_factory=list)
    contradicted_relations: list[tuple[str, str, str]] = Field(
        default_factory=list
    )
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
    domain: str = ""
    task: str = ""
    source: str = ""
    source_id: str = ""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IngestResult(BaseModel):
    """Result of ingesting an observation."""

    new_concepts: list[str] = Field(default_factory=list)
    reinforced_concepts: list[str] = Field(default_factory=list)
    weakened_concepts: list[str] = Field(default_factory=list)
    new_relations: list[str] = Field(default_factory=list)
    reinforced_relations: list[str] = Field(default_factory=list)
    weakened_relations: list[str] = Field(default_factory=list)
    hebbian_relations: list[str] = Field(default_factory=list)


class ActivationStep(BaseModel):
    """One hop in the best activation path that reached a concept."""

    from_id: str
    relation_id: str
    semantic_relation: str = ""
    axis: str = ""
    contribution: float = 0.0
    # True when PROPAGATION_MIN_RATIO lifted the raw score to the floor —
    # the concept is structurally connected but weakly driven.
    floored: bool = False


class ActivationTrace(BaseModel):
    """Why a concept appears in a projection: its best activation path.

    Best-path only, not a full provenance DAG — enough to answer
    "why am I seeing this" without recording every spread event.
    """

    seed_id: str
    score: float = 0.0
    inhibition: float = 0.0
    inhibition_source: str = ""
    steps: list[ActivationStep] = Field(default_factory=list)

    def extended_with(
        self, step: ActivationStep, score: float
    ) -> "ActivationTrace":
        """A new trace with one more hop appended (traces are immutable)."""
        return ActivationTrace(
            seed_id=self.seed_id,
            score=score,
            inhibition=self.inhibition,
            inhibition_source=self.inhibition_source,
            steps=[*self.steps, step],
        )


class Projection(BaseModel):
    """A local cognitive view — the operational output of World 0.

    This is what gets injected into the Agent's prompt to shape its reasoning.
    """

    concepts: list[ConceptNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    activation_scores: dict[str, float] = Field(default_factory=dict)
    task: str = ""
    # ── Provenance (additive; all default-empty for back-compat) ─────
    seeds: list[str] = Field(default_factory=list)
    # seed label → how it resolved ("exact", "fuzzy:0.52", "unresolved").
    seed_resolution: dict[str, str] = Field(default_factory=dict)
    perspective_name: str = ""
    # concept_id → best activation path that reached it.
    traces: dict[str, ActivationTrace] = Field(default_factory=dict)

    def top_concepts(self, n: int = 5) -> list[ConceptNode]:
        ranked = sorted(
            self.concepts,
            key=lambda c: self.activation_scores.get(c.id, 0.0),
            reverse=True,
        )
        return ranked[:n]

    def render(self, style: str = "default") -> str:
        """Render as LLM-prompt-ready markdown.

        ``style`` selects a renderer from ``world0.projection.render``
        ("default", "compact", "detailed"); unknown styles fall back to
        "default", whose output is byte-identical to the historical
        single-style render.
        """
        # Lazy import: schemas must stay importable without dragging in
        # the projection package (which itself imports schemas.types).
        from world0.projection.render import render_projection

        return render_projection(self, style=style)

    def explain(self, concept_ref: str) -> str:
        """Why is this concept in the projection?  Best-path rendering.

        ``concept_ref`` may be a concept id, name, or representation.
        Returns a one-line provenance string, or a fallback note when
        the concept is a seed / has no recorded trace.
        """
        node = None
        for c in self.concepts:
            if concept_ref in (c.id, c.name, c.representation()):
                node = c
                break
        if node is None:
            return f"{concept_ref}: not in this projection"

        names = {c.id: c.name for c in self.concepts}
        trace = self.traces.get(node.id)
        score = self.activation_scores.get(node.id, 0.0)
        if trace is None or not trace.steps:
            return f"{node.name}: seed concept (score {score:.2f})"

        # Walk the path backwards: target ←(relation)— source ... [seed]
        parts = [node.name]
        for step in reversed(trace.steps):
            origin = names.get(step.from_id, step.from_id)
            floored = ", floored" if step.floored else ""
            parts.append(
                f"←({step.semantic_relation}, {step.axis}, "
                f"×{step.contribution:.2f}{floored})— {origin}"
            )
        parts[-1] += " [seed]"
        line = " ".join(parts) + f" (score {score:.2f}"
        if trace.inhibition > 0:
            suppressor = names.get(
                trace.inhibition_source, trace.inhibition_source
            )
            line += f", suppressed {trace.inhibition:.2f} by {suppressor}"
        return line + ")"

    def _neighbor_names(self, concept_id: str) -> list[str]:
        names_map = {c.id: c.representation() for c in self.concepts}
        neighbors: list[str] = []
        for r in self.relations:
            other = r.other_end(concept_id)
            if other and other in names_map:
                neighbors.append(names_map[other])
        return neighbors


class FeedbackResult(BaseModel):
    """Outcome of applying usage feedback to the world.

    Feedback is how a consumer (an agent, the autonomy loop, a human)
    tells World 0 that a projection helped or misled: useful concepts
    get reinforced, missing ones created, noisy ones demoted, weak
    relations weakened — all through the public facade.
    """

    reinforced_concepts: list[str] = Field(default_factory=list)
    created_concepts: list[str] = Field(default_factory=list)
    demoted_concepts: list[str] = Field(default_factory=list)
    reinforced_relations: list[str] = Field(default_factory=list)
    weakened_relations: list[str] = Field(default_factory=list)


class ReflectResult(BaseModel):
    """Result of a reflect() cycle."""

    decayed_concepts: list[str] = Field(default_factory=list)
    promoted_concepts: list[str] = Field(default_factory=list)
    demoted_concepts: list[str] = Field(default_factory=list)
    pruned_concepts: list[str] = Field(default_factory=list)
    decayed_relations: list[str] = Field(default_factory=list)
    pruned_relations: list[str] = Field(default_factory=list)
    # Color-field dynamics (doc §29 Stage A observation layer).
    new_communities: list[str] = Field(default_factory=list)
    stable_communities: list[str] = Field(default_factory=list)
    pruned_communities: list[str] = Field(default_factory=list)
    color_sources: list[str] = Field(default_factory=list)


class WorldStatus(BaseModel):
    """Overview of the cognitive world's current state."""

    total_concepts: int = 0
    total_relations: int = 0
    by_maturity: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    last_reflect: datetime | None = None
    # Color-field diagnostics (doc §12.3).  Populated whenever
    # ``World.status()`` runs — independent of whether the caller
    # actually triggers a reflect cycle.
    total_communities: int = 0
    stable_communities: int = 0
    bridge_concepts: int = 0
    avg_color_purity: float = 1.0
    # Network-entropy diagnostics (docs/world-network-entropy-design.md).
    # Read-only structural signal: how diffuse vs. concentrated conceptual
    # attention is across the typed relation graph.  0 when the world has
    # no branching structure yet.
    avg_network_entropy: float = 0.0
    high_entropy_concepts: int = 0
    relation_type_entropy: float = 0.0
