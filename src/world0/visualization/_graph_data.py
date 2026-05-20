"""Build the JSON payload consumed by the visualization template.

This module depends only on the ``WorldView`` Protocol — it does **not**
import the concrete ``World`` class.  Anything satisfying ``WorldView``
(real World, mock for tests, frozen snapshot wrapper) can be rendered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from world0.schemas.concept import ConceptNode

if TYPE_CHECKING:
    from world0.core import WorldView


def build_graph_data(view: WorldView) -> dict:
    """Extract nodes and edges from a WorldView into JSON-serialisable form."""
    concepts = view.concepts.all()
    relations = view.relations.all()

    id_to_name = {c.id: c.name for c in concepts}

    nodes = [_node_payload(c, view) for c in concepts]
    edges = [
        _edge_payload(r, id_to_name)
        for r in relations
        if r.source_id in id_to_name and r.target_id in id_to_name
    ]
    return {"nodes": nodes, "edges": edges}


def _node_payload(c: ConceptNode, view: WorldView) -> dict:
    connections = len(view.relations.for_concept(c.id))
    tasks = sorted({e.task for e in c.reinforcement_log if e.task})
    sources = sorted({e.source for e in c.reinforcement_log if e.source})
    dominant_domain = c.domain
    dominant_domain_strength = round(c.dominant_domain_strength(), 4)
    domain_color = ConceptNode.domain_color_for(dominant_domain)
    domain_profile = [
        {
            "domain": domain,
            "strength": round(strength, 4),
            "color": ConceptNode.domain_color_for(domain),
        }
        for domain, strength in c.sorted_domain_profile()[:6]
    ]

    return {
        "id": c.id,
        "name": c.name,
        "aliases": c.aliases,
        "description": c.description,
        "domain": dominant_domain,
        "domain_color": domain_color,
        "domain_strength": dominant_domain_strength,
        "domain_profile": domain_profile,
        "tags": c.tags,
        "confidence": round(c.confidence, 4),
        "maturity": c.maturity.value,
        "activation_count": c.activation_count,
        "connections": connections,
        "created_at": c.created_at.strftime("%Y-%m-%d %H:%M"),
        "last_activated": c.last_activated.strftime("%Y-%m-%d %H:%M"),
        "tasks": tasks[:10],
        "sources": sources[:10],
        "origin": c.origin,
    }


def _edge_payload(r, id_to_name: dict[str, str]) -> dict:
    return {
        "source": r.source_id,
        "target": r.target_id,
        "relation_type": r.relation_type.value,
        "weight": round(r.weight, 4),
        "confidence": round(r.confidence, 4),
        "is_explicit": r.is_explicit,
        "reinforcement_count": r.reinforcement_count,
        "source_name": id_to_name[r.source_id],
        "target_name": id_to_name[r.target_id],
    }
