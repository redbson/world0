"""Input/output types for the World 0 Agent interface."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from world0.schemas.concept import ConceptNode, Maturity
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


class ContestedClaim(BaseModel):
    """Opposing explicit claims about one concept pair in a projection.

    ``claims`` lists ``(relation_id, semantic_relation, belief)`` for every
    explicit claim about the pair; ``leading`` is the relation id of the
    most believed one.  ``status`` is ``"contested"`` when the leading
    belief exceeds the strongest opposing belief by less than the
    contest margin, ``"leaning"`` otherwise.
    """

    source_id: str
    target_id: str
    claims: list[tuple[str, str, float]] = Field(default_factory=list)
    leading: str = ""
    margin: float = 0.0
    status: str = "contested"


class EpistemicStatus(BaseModel):
    """Metacognitive annotation of a projection (analysis ``docs/mc``).

    Distinguishes what the world knows well from what it has barely seen
    or holds contradictory beliefs about — the "metacognitive monitoring"
    indicator (HOT-2) of consciousness science, used here as a functional
    signal for the Agent, not as a claim about experience.

    ``reliability`` maps concept id → ``"well_evidenced"``,
    ``"moderate"`` or ``"tentative"``.
    """

    reliability: dict[str, str] = Field(default_factory=dict)
    contested: list[ContestedClaim] = Field(default_factory=list)

    def tentative_ids(self) -> list[str]:
        return [cid for cid, level in self.reliability.items() if level == "tentative"]


class Projection(BaseModel):
    """A local cognitive view — the operational output of World 0.

    This is what gets injected into the Agent's prompt to shape its reasoning.
    """

    concepts: list[ConceptNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    activation_scores: dict[str, float] = Field(default_factory=dict)
    task: str = ""
    epistemic: EpistemicStatus = Field(default_factory=EpistemicStatus)

    def top_concepts(self, n: int = 5) -> list[ConceptNode]:
        ranked = sorted(
            self.concepts,
            key=lambda c: self.activation_scores.get(c.id, 0.0),
            reverse=True,
        )
        return ranked[:n]

    def render(self) -> str:
        """Render as LLM-prompt-ready markdown."""
        lines: list[str] = ["## Cognitive Context", ""]

        # Group by maturity
        core = []
        active = []
        emerging = []
        for c in sorted(
            self.concepts,
            key=lambda x: self.activation_scores.get(x.id, 0),
            reverse=True,
        ):
            score = self.activation_scores.get(c.id, 0)
            if c.maturity in (Maturity.CORE, Maturity.ESTABLISHED):
                core.append((c, score))
            elif c.maturity == Maturity.DEVELOPING:
                active.append((c, score))
            else:
                emerging.append((c, score))

        if core:
            lines.append("### Core Understanding")
            for c, s in core:
                desc = f": {c.description}" if c.description else ""
                neighbors = self._neighbor_names(c.id)
                linked = f" Linked to: {', '.join(neighbors)}." if neighbors else ""
                lines.append(
                    f"- **{c.representation()}** ({c.name}, {c.maturity.value}, "
                    f"confidence: {c.confidence:.2f}, evidence: {c.evidence():.2f})"
                    f"{desc}{linked}"
                )
            lines.append("")

        if active:
            lines.append("### Active Concepts")
            for c, s in active:
                desc = f": {c.description}" if c.description else ""
                lines.append(
                    f"- **{c.representation()}** ({c.name}, {c.maturity.value}, "
                    f"confidence: {c.confidence:.2f}, evidence: {c.evidence():.2f})"
                    f"{desc}"
                )
            lines.append("")

        if emerging:
            lines.append("### Emerging Concepts")
            for c, s in emerging:
                lines.append(
                    f"- **{c.representation()}** ({c.name}, {c.maturity.value}, "
                    f"confidence: {c.confidence:.2f}, evidence: {c.evidence():.2f})"
                )
            lines.append("")

        if self.relations:
            lines.append("### Key Relations")
            concept_names = {c.id: c.representation() for c in self.concepts}
            for r in sorted(self.relations, key=lambda x: x.weight, reverse=True)[:10]:
                src = concept_names.get(r.source_id, r.source_id)
                tgt = concept_names.get(r.target_id, r.target_id)
                lines.append(
                    f"- {src} → {r.semantic_relation} [{r.relation_type.value}] → {tgt} "
                    # Belief applies to claims; a co-occurrence edge
                    # asserts nothing beyond "seen together".
                    + (
                        f"(belief: {r.probability:.2f}, "
                        if r.is_explicit
                        else "(co-occurrence, "
                    )
                    + f"structural: {r.structural_strength:.2f}, "
                    f"propagation: {r.propagation_strength:.2f}, "
                    f"reinforced {r.reinforcement_count}×)"
                )
            lines.append("")

        epistemic_lines = self._render_epistemic()
        if epistemic_lines:
            lines.extend(epistemic_lines)

        if self.task:
            lines.append(f"### Task Context")
            lines.append(f"Concepts activated for: {self.task}")
            lines.append("")

        return "\n".join(lines)

    def _render_epistemic(self) -> list[str]:
        """What the Agent should hold loosely: contested and thin knowledge."""
        names = {c.id: c.name for c in self.concepts}
        rels = {r.id: r for r in self.relations}
        out: list[str] = []
        for claim in self.epistemic.contested:
            parts = []
            for rid, semantic, belief in claim.claims:
                rel = rels.get(rid)
                src = names.get(rel.source_id, "?") if rel else names.get(claim.source_id, "?")
                tgt = names.get(rel.target_id, "?") if rel else names.get(claim.target_id, "?")
                parts.append(f"{src} {semantic} {tgt} (belief {belief:.2f})")
            if claim.status == "contested":
                out.append(f"- Contested: {' vs '.join(parts)}")
            else:
                out.append(f"- Leaning: {' over '.join(parts)}")
        tentative = [names[cid] for cid in self.epistemic.tentative_ids() if cid in names]
        if tentative:
            out.append(f"- Thin evidence (seen once or twice): {', '.join(tentative)}")
        if not out:
            return []
        return ["### Epistemic Status", *out, ""]

    def _neighbor_names(self, concept_id: str) -> list[str]:
        names_map = {c.id: c.representation() for c in self.concepts}
        neighbors: list[str] = []
        for r in self.relations:
            other = r.other_end(concept_id)
            if other and other in names_map:
                neighbors.append(names_map[other])
        return neighbors


class ReflectResult(BaseModel):
    """Result of a reflect() cycle."""

    decayed_concepts: list[str] = Field(default_factory=list)
    promoted_concepts: list[str] = Field(default_factory=list)
    demoted_concepts: list[str] = Field(default_factory=list)
    pruned_concepts: list[str] = Field(default_factory=list)
    decayed_relations: list[str] = Field(default_factory=list)
    pruned_relations: list[str] = Field(default_factory=list)
    # Auto-discovered generic edges removed because their association no
    # longer passes the Hebbian gate (dynamics/hebbian.py ``revalidate``).
    stale_relations: list[str] = Field(default_factory=list)
    # Color-field dynamics (doc §29 Stage A observation layer).
    new_communities: list[str] = Field(default_factory=list)
    stable_communities: list[str] = Field(default_factory=list)
    pruned_communities: list[str] = Field(default_factory=list)
    color_sources: list[str] = Field(default_factory=list)


class WorldStatus(BaseModel):
    """Overview of the cognitive world's current state."""

    # Cognitive time: number of observations ingested so far (see
    # ``schemas/clock.py``).  Decay and freshness are measured in ticks.
    cognitive_tick: int = 0
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
