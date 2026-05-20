"""Generate interactive HTML visualization of a World 0 concept network.

Thin entry point.  All real work lives elsewhere:

- HTML/CSS/JS is in ``template.html`` (loaded once at import time)
- Graph payload extraction is in ``_graph_data.py``
- The renderer depends only on the ``WorldView`` Protocol from
  ``world0.core``, so any compatible read-only view can be rendered.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from world0.visualization._graph_data import build_graph_data

if TYPE_CHECKING:
    from world0.core import WorldView


_TEMPLATE_PATH = Path(__file__).with_name("template.html")
_HTML_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")


def render_html(view: WorldView) -> str:
    """Return a self-contained HTML string visualising the WorldView."""
    graph_data = build_graph_data(view)
    return _HTML_TEMPLATE.replace(
        "__GRAPH_DATA__", json.dumps(graph_data, ensure_ascii=False)
    )


def visualize(
    view: WorldView,
    output: str | Path | None = None,
    *,
    open_browser: bool = True,
) -> Path:
    """Render the view to an HTML file and optionally open it in a browser.

    Args:
        view: Anything satisfying the ``WorldView`` Protocol.
        output: Output HTML path. Defaults to ``world0_viz.html`` in cwd.
        open_browser: Whether to open the result in the default browser.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output) if output else Path("world0_viz.html")
    output_path.write_text(render_html(view), encoding="utf-8")

    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")

    return output_path
