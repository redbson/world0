"""Projection renderers — formatting only, no selection logic.

Three styles, dispatched by ``Projection.render(style=...)``:

- ``default``: the historical maturity-grouped markdown, byte-identical
  to the pre-styles single render (locked by a snapshot test)
- ``compact``: one line per concept, top relations only — for prompt
  injection where token budget matters (chat turns, neighborhoods)
- ``detailed``: default plus per-concept "why included" traces and a
  counter-signals section for negative-axis relations

Renderers never re-rank or filter concepts beyond display caps; which
concepts and relations appear is the engine's decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from world0.schemas.concept import Maturity
from world0.schemas.relation import RelationType

if TYPE_CHECKING:
    from world0.schemas.types import Projection


def render_default(p: "Projection") -> str:
    """The historical render — maturity-grouped markdown."""
    lines: list[str] = ["## Cognitive Context", ""]

    # Group by maturity
    core = []
    active = []
    emerging = []
    for c in sorted(
        p.concepts,
        key=lambda x: p.activation_scores.get(x.id, 0),
        reverse=True,
    ):
        score = p.activation_scores.get(c.id, 0)
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
            neighbors = p._neighbor_names(c.id)
            linked = f" Linked to: {', '.join(neighbors)}." if neighbors else ""
            lines.append(
                f"- **{c.representation()}** ({c.name}, {c.maturity.value}, "
                f"confidence: {c.confidence:.2f}){desc}{linked}"
            )
        lines.append("")

    if active:
        lines.append("### Active Concepts")
        for c, s in active:
            desc = f": {c.description}" if c.description else ""
            lines.append(
                f"- **{c.representation()}** ({c.name}, {c.maturity.value}, "
                f"confidence: {c.confidence:.2f}){desc}"
            )
        lines.append("")

    if emerging:
        lines.append("### Emerging Concepts")
        for c, s in emerging:
            lines.append(
                f"- **{c.representation()}** ({c.name}, {c.maturity.value}, "
                f"confidence: {c.confidence:.2f})"
            )
        lines.append("")

    if p.relations:
        lines.append("### Key Relations")
        concept_names = {c.id: c.representation() for c in p.concepts}
        for r in sorted(p.relations, key=lambda x: x.weight, reverse=True)[:10]:
            src = concept_names.get(r.source_id, r.source_id)
            tgt = concept_names.get(r.target_id, r.target_id)
            lines.append(
                f"- {src} → {r.semantic_relation} [{r.relation_type.value}] → {tgt} "
                f"(structural: {r.structural_strength:.2f}, "
                f"propagation: {r.propagation_strength:.2f}, "
                f"reinforced {r.reinforcement_count}×)"
            )
        lines.append("")

    if p.task:
        lines.append(f"### Task Context")
        lines.append(f"Concepts activated for: {p.task}")
        lines.append("")

    return "\n".join(lines)


def render_compact(p: "Projection") -> str:
    """Dense one-line-per-concept render for tight token budgets."""
    lines: list[str] = ["## Cognitive Context (compact)"]

    for c in sorted(
        p.concepts,
        key=lambda x: p.activation_scores.get(x.id, 0),
        reverse=True,
    ):
        score = p.activation_scores.get(c.id, 0.0)
        desc = f" — {c.description}" if c.description else ""
        lines.append(
            f"- {c.name} [{c.maturity.value}, {score:.2f}]{desc}"
        )

    if p.relations:
        lines.append("Relations:")
        concept_names = {c.id: c.name for c in p.concepts}
        for r in sorted(p.relations, key=lambda x: x.weight, reverse=True)[:5]:
            src = concept_names.get(r.source_id, r.source_id)
            tgt = concept_names.get(r.target_id, r.target_id)
            lines.append(f"- {src} —{r.semantic_relation}→ {tgt}")

    if p.task:
        lines.append(f"Task: {p.task}")

    return "\n".join(lines)


def render_detailed(p: "Projection") -> str:
    """Default render plus provenance traces and counter-signals."""
    out = render_default(p)
    extra: list[str] = []

    # Why-included lines for traced (non-seed) concepts, strongest first.
    traced = [
        c
        for c in sorted(
            p.concepts,
            key=lambda x: p.activation_scores.get(x.id, 0),
            reverse=True,
        )
        if p.traces.get(c.id) and p.traces[c.id].steps
    ]
    if traced:
        extra.append("### Why Included")
        for c in traced:
            extra.append(f"- {p.explain(c.id)}")
        extra.append("")

    # Negative-axis relations are counter-signals: things the world
    # believes pull *apart* — worth surfacing explicitly when reasoning.
    negatives = [
        r for r in p.relations if r.relation_type == RelationType.NEGATIVE
    ]
    if negatives:
        extra.append("### Counter-Signals")
        concept_names = {c.id: c.name for c in p.concepts}
        for r in sorted(negatives, key=lambda x: x.weight, reverse=True):
            src = concept_names.get(r.source_id, r.source_id)
            tgt = concept_names.get(r.target_id, r.target_id)
            extra.append(
                f"- {src} ✕ {tgt} ({r.semantic_relation}, "
                f"weight {r.weight:.2f})"
            )
        extra.append("")

    if not extra:
        return out
    return out + "\n".join(extra)


RENDERERS: dict[str, Callable[["Projection"], str]] = {
    "default": render_default,
    "compact": render_compact,
    "detailed": render_detailed,
}


def render_projection(p: "Projection", style: str = "default") -> str:
    """Dispatch to a named renderer; unknown styles fall back to default."""
    renderer = RENDERERS.get(style, render_default)
    return renderer(p)
