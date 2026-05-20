"""Visualization brick — testable against the WorldView Protocol alone.

The renderer and graph-data builder should never reach for a concrete
``World``; they only need ``WorldView``.  These tests prove that by
feeding them ``FakeWorldView`` from ``world0.core.test_doubles``.
"""

from __future__ import annotations

import json

from world0.core import WorldView
from world0.core.test_doubles import (
    FakeConceptStore,
    FakeRelationStore,
    FakeWorldView,
    make_concept,
    make_edge,
)
from world0.visualization._graph_data import build_graph_data
from world0.visualization.renderer import render_html


def test_fake_world_view_satisfies_protocol() -> None:
    assert isinstance(FakeWorldView(), WorldView)


def test_build_graph_data_from_fake_view() -> None:
    a = make_concept("alpha", domain="ml")
    b = make_concept("beta", domain="ml")
    view = FakeWorldView(
        concepts=FakeConceptStore(seed=[a, b]),
        relations=FakeRelationStore(seed=[make_edge(a.id, b.id, weight=0.7)]),
    )

    payload = build_graph_data(view)

    assert {n["name"] for n in payload["nodes"]} == {"alpha", "beta"}
    assert len(payload["edges"]) == 1
    edge = payload["edges"][0]
    assert edge["source"] == a.id
    assert edge["target"] == b.id
    assert edge["source_name"] == "alpha"
    assert edge["target_name"] == "beta"


def test_build_graph_data_drops_edges_with_unknown_endpoints() -> None:
    a = make_concept("alpha")
    view = FakeWorldView(
        concepts=FakeConceptStore(seed=[a]),
        relations=FakeRelationStore(
            seed=[make_edge(a.id, "missing-id", weight=0.5)]
        ),
    )
    payload = build_graph_data(view)
    assert payload["edges"] == []


def test_render_html_injects_graph_payload() -> None:
    a = make_concept("alpha")
    b = make_concept("beta")
    view = FakeWorldView(
        concepts=FakeConceptStore(seed=[a, b]),
        relations=FakeRelationStore(seed=[make_edge(a.id, b.id)]),
    )

    html = render_html(view)

    assert "__GRAPH_DATA__" not in html, "template placeholder must be substituted"
    # The payload is embedded as JSON; round-trip it out of the HTML to check.
    start = html.find('{"nodes":')
    assert start != -1, "graph payload should be embedded as JSON"
    # Decode from the first `{` until JSON parser is happy.
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(html[start:])
    assert {n["name"] for n in payload["nodes"]} == {"alpha", "beta"}
    assert len(payload["edges"]) == 1


def test_render_html_on_empty_view() -> None:
    html = render_html(FakeWorldView())
    assert "__GRAPH_DATA__" not in html
    start = html.find('{"nodes":')
    assert start != -1
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(html[start:])
    assert payload == {"nodes": [], "edges": []}
