"""Visualization module for World 0 — interactive concept network explorer.

Public surface:

- ``render_html(view)`` → HTML string
- ``visualize(view, output, *, open_browser)`` → write & optionally open
- ``build_graph_data(view)`` → exposed for tests and alternative renderers

All entry points consume any object satisfying the ``WorldView``
Protocol (``world0.core.WorldView``) — no hard dependency on the
concrete ``World`` class.
"""

from world0.visualization._graph_data import build_graph_data
from world0.visualization.renderer import render_html, visualize

__all__ = ["build_graph_data", "render_html", "visualize"]
