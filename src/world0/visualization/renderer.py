"""Generate interactive HTML visualization of a World 0 concept network.

Produces a self-contained HTML file (D3.js loaded from CDN) with:
  - Force-directed concept network graph
  - Concept card detail panel (click a node)
  - Color coding by maturity stage
  - Node size by confidence
  - Edge thickness by weight, labeled by relation type
  - Search / filter by name
  - Zoom and pan
"""

from __future__ import annotations

import json
import webbrowser
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world0.world import World


def _build_graph_data(world: World) -> dict:
    """Extract nodes and edges from a World instance into JSON-serializable form."""
    concepts = world.concepts.all()
    relations = world.relations.all()

    id_to_name = {c.id: c.name for c in concepts}

    nodes = []
    for c in concepts:
        connections = len(world.relations.for_concept(c.id))
        # Collect task history from reinforcement log
        tasks = sorted({e.task for e in c.reinforcement_log if e.task})
        sources = sorted({e.source for e in c.reinforcement_log if e.source})

        nodes.append({
            "id": c.id,
            "name": c.name,
            "aliases": c.aliases,
            "description": c.description,
            "domain": c.domain,
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
        })

    edges = []
    for r in relations:
        if r.source_id in id_to_name and r.target_id in id_to_name:
            edges.append({
                "source": r.source_id,
                "target": r.target_id,
                "relation_type": r.relation_type.value,
                "weight": round(r.weight, 4),
                "confidence": round(r.confidence, 4),
                "is_explicit": r.is_explicit,
                "reinforcement_count": r.reinforcement_count,
                "source_name": id_to_name[r.source_id],
                "target_name": id_to_name[r.target_id],
            })

    return {"nodes": nodes, "edges": edges}


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>World 0 — Concept Network</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: #0a0e17;
  color: #e0e6ed;
  overflow: hidden;
  height: 100vh;
}

/* ── Layout ── */
#container { display: flex; height: 100vh; }
#graph-panel { flex: 1; position: relative; }
#card-panel {
  width: 380px; min-width: 380px;
  background: #111827;
  border-left: 1px solid #1e293b;
  overflow-y: auto;
  transition: transform 0.3s;
}
#card-panel.collapsed { transform: translateX(380px); min-width: 0; width: 0; }

/* ── Top bar ── */
#top-bar {
  position: absolute; top: 0; left: 0; right: 0; z-index: 10;
  padding: 12px 20px;
  display: flex; align-items: center; gap: 16px;
  background: linear-gradient(180deg, rgba(10,14,23,0.95) 0%, rgba(10,14,23,0) 100%);
}
#top-bar h1 {
  font-size: 18px; font-weight: 600;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#search {
  padding: 6px 12px; border-radius: 6px;
  border: 1px solid #334155; background: #1e293b; color: #e0e6ed;
  font-size: 13px; width: 220px; outline: none;
}
#search:focus { border-color: #60a5fa; }
#stats { font-size: 12px; color: #64748b; margin-left: auto; }

/* ── Legend ── */
#legend {
  position: absolute; bottom: 16px; left: 20px; z-index: 10;
  display: flex; gap: 14px; font-size: 11px; color: #94a3b8;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }

/* ── SVG ── */
svg { width: 100%; height: 100%; }
.link { stroke-opacity: 0.5; }
.link-label { font-size: 9px; fill: #475569; pointer-events: none; }
.node-circle { cursor: pointer; stroke-width: 2px; transition: stroke-width 0.2s; }
.node-circle:hover { stroke-width: 4px; }
.node-label { font-size: 11px; fill: #cbd5e1; pointer-events: none; text-anchor: middle; }
.node-label.highlight { fill: #f1f5f9; font-weight: 600; }

/* ── Card panel ── */
.card-header {
  padding: 20px; border-bottom: 1px solid #1e293b;
  background: linear-gradient(135deg, #111827, #1a1f35);
}
.card-header h2 { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.card-header .maturity-badge {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
}
.card-section { padding: 16px 20px; border-bottom: 1px solid #1e293b0a; }
.card-section:last-child { border-bottom: none; }
.card-section h3 {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.8px; color: #64748b; margin-bottom: 8px;
}
.card-description { font-size: 13px; color: #94a3b8; line-height: 1.5; }

/* ── Stats grid ── */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-item {
  background: #1e293b; border-radius: 8px; padding: 10px 12px;
}
.stat-value { font-size: 20px; font-weight: 700; }
.stat-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Meter bar ── */
.meter { height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; margin-top: 4px; }
.meter-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

/* ── Tags / relations list ── */
.tag {
  display: inline-block; padding: 2px 8px; margin: 2px;
  border-radius: 4px; font-size: 11px; background: #1e293b; color: #94a3b8;
}
.relation-item {
  padding: 6px 0; font-size: 12px; display: flex; align-items: center; gap: 6px;
  border-bottom: 1px solid #1e293b0a;
}
.relation-item:last-child { border-bottom: none; }
.rel-arrow { color: #475569; }
.rel-type {
  padding: 1px 6px; border-radius: 3px;
  font-size: 10px; background: #1e293b; color: #60a5fa;
}
.rel-weight { font-size: 10px; color: #475569; margin-left: auto; }

/* ── Empty state ── */
.empty-card {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; color: #475569; text-align: center; padding: 40px;
}
.empty-card svg { width: 48px; height: 48px; margin-bottom: 16px; opacity: 0.3; }
.empty-card p { font-size: 13px; line-height: 1.6; }

/* ── Tooltip ── */
.tooltip {
  position: absolute; padding: 6px 10px; border-radius: 4px;
  background: #1e293b; border: 1px solid #334155; color: #e0e6ed;
  font-size: 11px; pointer-events: none; opacity: 0; transition: opacity 0.15s;
  z-index: 100; white-space: nowrap;
}
</style>
</head>
<body>
<div id="container">
  <div id="graph-panel">
    <div id="top-bar">
      <h1>World 0</h1>
      <input id="search" type="text" placeholder="搜索概念 / Search concepts..." />
      <span id="stats"></span>
    </div>
    <svg id="graph"></svg>
    <div id="legend"></div>
    <div class="tooltip" id="tooltip"></div>
  </div>
  <div id="card-panel">
    <div class="empty-card" id="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/>
      </svg>
      <p>点击任意概念节点<br>查看概念卡片详情<br><br>Click any concept node<br>to view its card</p>
    </div>
    <div id="card-content" style="display:none"></div>
  </div>
</div>

<script>
// ── Data ──
const DATA = __GRAPH_DATA__;

const MATURITY_COLORS = {
  embryonic:   { fill: "#f59e0b", stroke: "#d97706", bg: "#78350f", text: "#fbbf24" },
  developing:  { fill: "#3b82f6", stroke: "#2563eb", bg: "#1e3a5f", text: "#60a5fa" },
  established: { fill: "#10b981", stroke: "#059669", bg: "#064e3b", text: "#34d399" },
  core:        { fill: "#a78bfa", stroke: "#7c3aed", bg: "#4c1d95", text: "#c4b5fd" },
  fading:      { fill: "#6b7280", stroke: "#4b5563", bg: "#1f2937", text: "#9ca3af" },
};

const RELATION_COLORS = {
  depends_on: "#ef4444", contains: "#f59e0b", part_of: "#f59e0b",
  supports: "#10b981", activates: "#3b82f6", precedes: "#8b5cf6",
  derived_from: "#ec4899", similar_to: "#06b6d4", contrasts: "#f97316",
  related_to: "#6b7280",
};

// ── Stats ──
const statsEl = document.getElementById("stats");
const matCounts = {};
DATA.nodes.forEach(n => { matCounts[n.maturity] = (matCounts[n.maturity]||0) + 1; });
statsEl.textContent = `${DATA.nodes.length} concepts / ${DATA.edges.length} relations`;

// ── Legend ──
const legendEl = document.getElementById("legend");
Object.entries(MATURITY_COLORS).forEach(([m, c]) => {
  if (!matCounts[m]) return;
  const div = document.createElement("div");
  div.className = "legend-item";
  div.innerHTML = `<span class="legend-dot" style="background:${c.fill}"></span>${m} (${matCounts[m]})`;
  legendEl.appendChild(div);
});

// ── Graph setup ──
const svg = d3.select("#graph");
const width = document.getElementById("graph-panel").clientWidth;
const height = document.getElementById("graph-panel").clientHeight;

const g = svg.append("g");

// Zoom
const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

// Arrow markers per relation type
const defs = svg.append("defs");
Object.entries(RELATION_COLORS).forEach(([type, color]) => {
  defs.append("marker")
    .attr("id", `arrow-${type}`)
    .attr("viewBox", "0 -5 10 10").attr("refX", 22).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,-4L10,0L0,4").attr("fill", color).attr("opacity", 0.6);
});

// ── Simulation ──
const simulation = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.edges).id(d => d.id).distance(d => 120 / (d.weight + 0.1)))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(d => nodeRadius(d) + 8));

// ── Edges ──
const link = g.append("g").selectAll("line")
  .data(DATA.edges).join("line")
  .attr("class", "link")
  .attr("stroke", d => RELATION_COLORS[d.relation_type] || "#6b7280")
  .attr("stroke-width", d => Math.max(1, d.weight * 4))
  .attr("marker-end", d => `url(#arrow-${d.relation_type})`);

// Edge labels
const linkLabel = g.append("g").selectAll("text")
  .data(DATA.edges).join("text")
  .attr("class", "link-label")
  .text(d => d.relation_type);

// ── Nodes ──
function nodeRadius(d) { return 8 + d.confidence * 16 + Math.min(d.connections, 10) * 1.2; }

const node = g.append("g").selectAll("g")
  .data(DATA.nodes).join("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
  );

node.append("circle")
  .attr("class", "node-circle")
  .attr("r", nodeRadius)
  .attr("fill", d => MATURITY_COLORS[d.maturity]?.fill || "#6b7280")
  .attr("stroke", d => MATURITY_COLORS[d.maturity]?.stroke || "#4b5563")
  .on("click", (e, d) => showCard(d))
  .on("mouseover", (e, d) => showTooltip(e, d))
  .on("mouseout", hideTooltip);

node.append("text")
  .attr("class", "node-label")
  .attr("dy", d => nodeRadius(d) + 14)
  .text(d => d.name);

// ── Tick ──
simulation.on("tick", () => {
  link
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2);
  node.attr("transform", d => `translate(${d.x},${d.y})`);
});

// ── Tooltip ──
const tooltip = document.getElementById("tooltip");
function showTooltip(event, d) {
  tooltip.innerHTML = `<strong>${d.name}</strong> &nbsp;${d.maturity}<br>confidence: ${d.confidence} &nbsp; activations: ${d.activation_count}`;
  tooltip.style.left = (event.pageX + 12) + "px";
  tooltip.style.top = (event.pageY - 10) + "px";
  tooltip.style.opacity = 1;
}
function hideTooltip() { tooltip.style.opacity = 0; }

// ── Search ──
const searchEl = document.getElementById("search");
searchEl.addEventListener("input", () => {
  const q = searchEl.value.trim().toLowerCase();
  node.select("circle").attr("opacity", d =>
    !q || d.name.toLowerCase().includes(q) || d.aliases.some(a => a.toLowerCase().includes(q)) ? 1 : 0.12
  );
  node.select("text").attr("opacity", d =>
    !q || d.name.toLowerCase().includes(q) || d.aliases.some(a => a.toLowerCase().includes(q)) ? 1 : 0.08
  );
  link.attr("opacity", d => {
    if (!q) return 0.5;
    const sn = d.source.name?.toLowerCase() || "";
    const tn = d.target.name?.toLowerCase() || "";
    return sn.includes(q) || tn.includes(q) ? 0.7 : 0.05;
  });
});

// ── Card panel ──
let selectedNode = null;

function showCard(d) {
  selectedNode = d;
  document.getElementById("empty-state").style.display = "none";
  const el = document.getElementById("card-content");
  el.style.display = "block";

  const mc = MATURITY_COLORS[d.maturity] || MATURITY_COLORS.fading;

  // Gather relations for this node
  const rels = DATA.edges.filter(e =>
    (e.source.id || e.source) === d.id || (e.target.id || e.target) === d.id
  );

  const inbound = rels.filter(e => (e.target.id || e.target) === d.id);
  const outbound = rels.filter(e => (e.source.id || e.source) === d.id);

  function relHTML(list, direction) {
    return list.map(r => {
      const other = direction === "out"
        ? (r.target.name || r.target_name)
        : (r.source.name || r.source_name);
      const arrow = direction === "out" ? "→" : "←";
      const color = RELATION_COLORS[r.relation_type] || "#6b7280";
      return `<div class="relation-item">
        <span class="rel-arrow">${arrow}</span>
        <span class="rel-type" style="color:${color};border:1px solid ${color}33;background:${color}11">${r.relation_type}</span>
        <span>${other}</span>
        <span class="rel-weight">w=${r.weight} ×${r.reinforcement_count}</span>
      </div>`;
    }).join("");
  }

  el.innerHTML = `
    <div class="card-header">
      <h2>${d.name}</h2>
      <span class="maturity-badge" style="background:${mc.bg};color:${mc.text}">${d.maturity}</span>
      ${d.aliases.length ? `<div style="margin-top:6px;font-size:12px;color:#64748b">aka: ${d.aliases.join(", ")}</div>` : ""}
    </div>

    ${d.description ? `<div class="card-section"><h3>Description / 描述</h3><div class="card-description">${d.description}</div></div>` : ""}

    <div class="card-section">
      <h3>Cognitive Stats / 认知指标</h3>
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-value" style="color:${mc.text}">${d.confidence}</div>
          <div class="stat-label">Confidence / 置信度</div>
          <div class="meter"><div class="meter-fill" style="width:${d.confidence*100}%;background:${mc.fill}"></div></div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${d.activation_count}</div>
          <div class="stat-label">Activations / 激活次数</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${d.connections}</div>
          <div class="stat-label">Connections / 连接数</div>
        </div>
        <div class="stat-item">
          <div class="stat-value" style="font-size:14px">${d.last_activated}</div>
          <div class="stat-label">Last Active / 最后激活</div>
        </div>
      </div>
    </div>

    ${outbound.length ? `<div class="card-section"><h3>Outbound Relations / 出向关系 (${outbound.length})</h3>${relHTML(outbound, "out")}</div>` : ""}
    ${inbound.length ? `<div class="card-section"><h3>Inbound Relations / 入向关系 (${inbound.length})</h3>${relHTML(inbound, "in")}</div>` : ""}

    ${d.tasks.length ? `<div class="card-section"><h3>Associated Tasks / 关联任务</h3><div>${d.tasks.map(t => `<span class="tag">${t}</span>`).join("")}</div></div>` : ""}

    ${d.sources.length ? `<div class="card-section"><h3>Sources / 来源</h3><div>${d.sources.map(s => `<span class="tag">${s}</span>`).join("")}</div></div>` : ""}

    <div class="card-section" style="font-size:11px;color:#475569">
      <div>Origin: ${d.origin || "—"}</div>
      <div>Created: ${d.created_at}</div>
    </div>
  `;

  // Highlight in graph
  highlightNode(d);
}

function highlightNode(d) {
  const connectedIds = new Set();
  connectedIds.add(d.id);
  DATA.edges.forEach(e => {
    const sid = e.source.id || e.source;
    const tid = e.target.id || e.target;
    if (sid === d.id) connectedIds.add(tid);
    if (tid === d.id) connectedIds.add(sid);
  });

  node.select("circle").attr("opacity", n => connectedIds.has(n.id) ? 1 : 0.15);
  node.select("text")
    .attr("opacity", n => connectedIds.has(n.id) ? 1 : 0.1)
    .classed("highlight", n => n.id === d.id);
  link.attr("opacity", e => {
    const sid = e.source.id || e.source;
    const tid = e.target.id || e.target;
    return sid === d.id || tid === d.id ? 0.8 : 0.04;
  });
}

// Click on background to reset
svg.on("click", (e) => {
  if (e.target.tagName === "svg" || e.target.tagName === "SVG") {
    node.select("circle").attr("opacity", 1);
    node.select("text").attr("opacity", 1).classed("highlight", false);
    link.attr("opacity", 0.5);
  }
});

// ── Initial zoom to fit ──
setTimeout(() => {
  const bounds = g.node().getBBox();
  if (bounds.width > 0 && bounds.height > 0) {
    const scale = Math.min(width / (bounds.width + 100), height / (bounds.height + 100), 1.5);
    const tx = width / 2 - (bounds.x + bounds.width / 2) * scale;
    const ty = height / 2 - (bounds.y + bounds.height / 2) * scale;
    svg.transition().duration(800).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }
}, 2000);
</script>
</body>
</html>"""


def render_html(world: World) -> str:
    """Generate a self-contained HTML string visualizing the World's concept network."""
    graph_data = _build_graph_data(world)
    return _HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(graph_data, ensure_ascii=False))


def visualize(
    world: World,
    output: str | Path | None = None,
    *,
    open_browser: bool = True,
) -> Path:
    """Generate and optionally open an interactive concept network visualization.

    Args:
        world: The World instance to visualize.
        output: Output HTML file path. Defaults to "world0_viz.html" in cwd.
        open_browser: Whether to open the file in the default browser.

    Returns:
        Path to the generated HTML file.
    """
    if output is None:
        output = Path("world0_viz.html")
    else:
        output = Path(output)

    html = render_html(world)
    output.write_text(html, encoding="utf-8")

    if open_browser:
        webbrowser.open(f"file://{output.resolve()}")

    return output
