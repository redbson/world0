"""Input/output types for the World 0 Agent interface."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from world0.schemas.concept import ConceptNode, Maturity
from world0.schemas.relation import RelationEdge


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
    relations: list[tuple[str, str, str]] = Field(default_factory=list)
    descriptions: dict[str, str] = Field(default_factory=dict)
    weakened: list[str] = Field(default_factory=list)
    contradicted_relations: list[tuple[str, str, str]] = Field(
        default_factory=list
    )
    domain: str = ""
    task: str = ""
    source: str = ""
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


class Projection(BaseModel):
    """A local cognitive view — the operational output of World 0.

    This is what gets injected into the Agent's prompt to shape its reasoning.
    """

    concepts: list[ConceptNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    activation_scores: dict[str, float] = Field(default_factory=dict)
    task: str = ""

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
                    f"- **{c.name}** ({c.maturity.value}, "
                    f"confidence: {c.confidence:.2f}){desc}{linked}"
                )
            lines.append("")

        if active:
            lines.append("### Active Concepts")
            for c, s in active:
                desc = f": {c.description}" if c.description else ""
                lines.append(
                    f"- **{c.name}** ({c.maturity.value}, "
                    f"confidence: {c.confidence:.2f}){desc}"
                )
            lines.append("")

        if emerging:
            lines.append("### Emerging Concepts")
            for c, s in emerging:
                lines.append(
                    f"- **{c.name}** ({c.maturity.value}, "
                    f"confidence: {c.confidence:.2f})"
                )
            lines.append("")

        if self.relations:
            lines.append("### Key Relations")
            concept_names = {c.id: c.name for c in self.concepts}
            for r in sorted(self.relations, key=lambda x: x.weight, reverse=True)[:10]:
                src = concept_names.get(r.source_id, r.source_id)
                tgt = concept_names.get(r.target_id, r.target_id)
                lines.append(
                    f"- {src} → {r.relation_type.value} → {tgt} "
                    f"(strength: {r.weight:.2f}, reinforced {r.reinforcement_count}×)"
                )
            lines.append("")

        if self.task:
            lines.append(f"### Task Context")
            lines.append(f"Concepts activated for: {self.task}")
            lines.append("")

        return "\n".join(lines)

    def _neighbor_names(self, concept_id: str) -> list[str]:
        names_map = {c.id: c.name for c in self.concepts}
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
