"""Build ``docs/research/brief.html`` from the daily notes in ``docs/research/notes/``.

    python docs/research/build_brief.py            # writes brief.html next to this file
    python docs/research/build_brief.py --stdout   # prints the HTML

The page is a self-contained brief: the latest round first (decision, probe
numbers, verification, next steps), then every round as a collapsible
entry, then every paper mentioned in any round with its relation to World 0
as a filterable table.  The data is embedded as JSON so the page needs no
server; the daily session rebuilds it and commits the result.

The notes follow the template in ``README.md`` §6; the parser is lenient
(headings by ``## ``, papers as ``- **title** … arXiv:id …``).
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES = HERE / "notes"
OUT = HERE / "brief.html"

SECTION_KEYS = [
    ("论文", "papers"), ("决策", "decision"), ("探针", "probe"), ("开发", "development"),
    ("验证", "verification"), ("提交", "commits"), ("未完成", "next"),
]
TAGS = ["相同", "相反", "可借用", "支持", "边界外", "仅标题", "其他"]


# ── markdown (the subset the notes use) → html ────────────────────────

def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w/])\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    text = re.sub(r"arXiv:(\d{4}\.\d{4,5})(v\d+)?",
                  lambda m: f'<a class="id" href="https://arxiv.org/abs/{m.group(1)}" target="_blank" rel="noopener">arXiv:{m.group(1)}</a>',
                  text)
    return text


def md_to_html(md: str) -> str:
    out: list[str] = []
    lines = md.strip("\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            if body:
                head, rest = body[0], body[1:]
                out.append('<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                           + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rest)
                           + "</tbody></table></div>")
            continue
        if re.match(r"^\s*[-*] ", line):
            items = []
            while i < len(lines) and (re.match(r"^\s*[-*] ", lines[i]) or (lines[i].startswith("  ") and items)):
                if re.match(r"^\s*[-*] ", lines[i]):
                    items.append(re.sub(r"^\s*[-*] ", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\. ", line):
            items = []
            while i < len(lines) and (re.match(r"^\s*\d+\. ", lines[i]) or (lines[i].startswith("  ") and items)):
                if re.match(r"^\s*\d+\. ", lines[i]):
                    items.append(re.sub(r"^\s*\d+\. ", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ol>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ol>")
            continue
        if line.startswith("#"):
            level = min(6, len(line) - len(line.lstrip("#")))
            out.append(f"<h{level + 2}>{_inline(line.lstrip('#').strip())}</h{level + 2}>")
            i += 1
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(\s*[-*] |\s*\d+\. |#|\|)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


# ── notes → data ──────────────────────────────────────────────────────

def parse_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    date = path.stem
    title = ""
    sections: dict[str, str] = {k: "" for _, k in SECTION_KEYS}
    current = None
    buf: list[str] = []

    def flush():
        if current is not None:
            sections[current] = (sections[current] + "\n" + "\n".join(buf)).strip("\n")

    for line in text.split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            buf = []
            head = line[3:]
            current = next((k for zh, k in SECTION_KEYS if head.startswith(zh)), None)
            continue
        buf.append(line)
    flush()
    papers = parse_papers(sections["papers"], date)
    return {
        "date": date,
        "title": title or date,
        "sections": sections,
        "html": {k: md_to_html(v) for k, v in sections.items()},
        "papers": papers,
        "decision_line": first_sentence(sections["decision"]),
        "verification_line": first_sentence(sections["verification"]),
        "numbers": probe_numbers(sections["probe"]),
    }


def first_sentence(md: str) -> str:
    """The first clause of a section, for the one-line headline of a round
    (code spans are kept as plain text; only Chinese full stops end it)."""
    text = re.sub(r"\s+", " ", md).strip()
    text = re.sub(r"\*\*|`", "", text)
    text = re.sub(r"^选：", "", text)
    m = re.match(r"^(.{0,160}?[。；])", text)
    return (m.group(1) if m else text[:160]).strip()


def probe_numbers(md: str) -> list[str]:
    """Bold sentences and table rows in the probe section: the figures worth a glance."""
    out = []
    for m in re.finditer(r"\*\*([^*]{4,120})\*\*", md):
        s = m.group(1).strip().rstrip("：:")
        looks_like_heading = s.endswith(("？", "?")) or s.startswith("探针")
        has_words = len(re.sub(r"[\d.,%\s]", "", s)) >= 2
        if re.search(r"\d", s) and has_words and not looks_like_heading and s not in out:
            out.append(s)
    # Sentences that state a conclusion with a number are the second source.
    for m in re.finditer(r"(?:结论|结果)[：:]\s*([^。]{6,160}。)", md):
        s = re.sub(r"\s+", " ", re.sub(r"\*\*|`", "", m.group(1))).strip()
        if re.search(r"\d", s) and s not in out:
            out.append(s)
    return out[:6]


def parse_papers(md: str, date: str) -> list[dict]:
    papers = []
    for raw in re.split(r"\n(?=- )", md.strip("\n")):
        line = " ".join(x.strip() for x in raw.split("\n")).strip()
        if not line.startswith("- "):
            continue
        line = line[2:]
        title_m = re.match(r"\*\*(.+?)\*\*", line)
        if not title_m:
            continue
        title = title_m.group(1).strip()
        ids = re.findall(r"arXiv:(\d{4}\.\d{4,5})", line)
        tag = "其他"
        rel = line.split("与 World 0", 1)[1] if "与 World 0" in line else ""
        for t in TAGS[:-1]:
            if t in rel or (t == "仅标题" and "仅标题" in line):
                tag = t
                break
        if tag == "其他" and "仅标题" in line:
            tag = "仅标题"
        rest = line[title_m.end():].lstrip(" —-–")
        if re.fullmatch(r"[（(]?arXiv:[\d.]+[)）]?", title):
            # a bullet that names only the id: the first clause is the title
            title = re.sub(r"\*\*|`", "", rest).split("。")[0][:80]
        papers.append({"date": date, "title": title, "arxiv": ids[0] if ids else "", "tag": tag,
                       "note": re.sub(r"\*\*", "", rest), "note_html": _inline(rest)})
    return papers


def load_rounds() -> list[dict]:
    notes = sorted(NOTES.glob("*.md"))
    return [parse_note(p) for p in notes if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem)]


# ── page ──────────────────────────────────────────────────────────────

TEMPLATE = r"""<title>World 0 研究简报</title>
<meta name="description" content="World 0 每日研究与开发的记录：论文、决策、探针数字、验证与下一步。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&display=swap">
<style>
/* Layout: a lab notebook. One reading column (68ch) for the brief and the
   rounds; the paper table alone may run wider inside its own scroller. */
:root {
  --bg: #f3f5f6;         /* slate-tinted paper */
  --surface: #ffffff;
  --fg: #16212a;
  --muted: #5d6b76;
  --line: #d6dde2;
  --accent: #0f766e;      /* deep teal: the one bold colour */
  --accent-soft: #d8efec;
  --tag-same: #2f7a3a;    --tag-same-bg: #e2f2e4;
  --tag-opp: #a8461f;     --tag-opp-bg: #f8e6dd;
  --tag-borrow: #1f5f99;  --tag-borrow-bg: #dfeaf7;
  --tag-support: #4a6f1c; --tag-support-bg: #eaf1dc;
  --tag-out: #6b7280;     --tag-out-bg: #e9ecef;
  --font-display: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --font-body: "Noto Sans SC", "IBM Plex Sans", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111719; --surface: #182024; --fg: #e6ebee; --muted: #9aa8b1; --line: #2b363c;
    --accent: #4fc3b6; --accent-soft: #163a37;
    --tag-same: #8fd79a; --tag-same-bg: #1d3a22; --tag-opp: #f0a37f; --tag-opp-bg: #46281a;
    --tag-borrow: #8fbdf0; --tag-borrow-bg: #1b3350; --tag-support: #bfd98a; --tag-support-bg: #2f3d18;
    --tag-out: #a3adb6; --tag-out-bg: #2a3238;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #111719; --surface: #182024; --fg: #e6ebee; --muted: #9aa8b1; --line: #2b363c;
  --accent: #4fc3b6; --accent-soft: #163a37;
  --tag-same: #8fd79a; --tag-same-bg: #1d3a22; --tag-opp: #f0a37f; --tag-opp-bg: #46281a;
  --tag-borrow: #8fbdf0; --tag-borrow-bg: #1b3350; --tag-support: #bfd98a; --tag-support-bg: #2f3d18;
  --tag-out: #a3adb6; --tag-out-bg: #2a3238;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 15px; line-height: 1.65;
       padding-inline: 16px; padding-block: 0 48px; }
.wrap { max-width: 72rem; margin: 0 auto; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:focus-visible, button:focus-visible, input:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
code, .mono { font-family: var(--font-mono); font-size: 0.92em; }
code { background: var(--accent-soft); padding: 0 .3em; border-radius: 3px; }
h1, h2, h3, h4 { font-family: var(--font-display); text-wrap: balance; line-height: 1.25; margin: 0; }
h1 { font-size: 1.9rem; font-weight: 600; letter-spacing: -0.01em; }
h2 { font-size: 1.25rem; font-weight: 600; margin-block: 2.2rem .9rem; display: flex; align-items: baseline; gap: .6rem; }
h2 .count { font: 500 .8rem var(--font-mono); color: var(--muted); }
h3 { font-size: 1.02rem; font-weight: 600; margin-block: 1.1rem .4rem; }
h4 { font-size: .95rem; font-weight: 600; margin-block: .9rem .3rem; }
.eyebrow { font: 500 .72rem var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
header { padding-block: 2.4rem 1.2rem; border-bottom: 1px solid var(--line); }
header .sub { color: var(--muted); margin-top: .5rem; max-width: 68ch; }
.stats { display: flex; flex-wrap: wrap; gap: 1.6rem 2.4rem; margin-top: 1.4rem; font-variant-numeric: tabular-nums; }
.stat .n { font: 500 1.6rem var(--font-mono); color: var(--fg); display: block; line-height: 1.1; }
.stat .l { color: var(--muted); font-size: .85rem; }
.brief { display: grid; grid-template-columns: minmax(0, 1fr); gap: 1rem; }
@media (min-width: 880px) { .brief { grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); } }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 1.1rem 1.25rem; min-width: 0; }
.card.lead { border-left: 3px solid var(--accent); }
.card p, .card li { max-width: 68ch; }
.numbers { display: flex; flex-direction: column; gap: .5rem; margin: 0; padding: 0; list-style: none; }
.numbers li { font-size: .92rem; padding-left: 1rem; position: relative; }
.numbers li::before { content: ""; position: absolute; left: 0; top: .62em; width: .4rem; height: .4rem; border-radius: 50%; background: var(--accent); }
details.round { border-top: 1px solid var(--line); padding-block: .6rem; }
details.round:last-of-type { border-bottom: 1px solid var(--line); }
details.round > summary { cursor: pointer; display: grid; grid-template-columns: auto minmax(0, 1fr); gap: .3rem 1rem; align-items: baseline; list-style: none; padding-block: .4rem; }
details.round > summary::-webkit-details-marker { display: none; }
details.round > summary .date { font: 500 .95rem var(--font-mono); color: var(--accent); white-space: nowrap; }
details.round > summary .headline { color: var(--fg); min-width: 0; overflow-wrap: anywhere; }
details.round > summary .headline .dim { color: var(--muted); }
details.round[open] > summary .headline { font-weight: 500; }
.sections { display: grid; gap: .2rem 1.5rem; grid-template-columns: minmax(0, 1fr); padding: .4rem 0 1rem; }
@media (min-width: 880px) { .sections { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } .sections .wide { grid-column: 1 / -1; } }
.section { min-width: 0; }
.section p, .section li { max-width: 68ch; }
ul, ol { padding-left: 1.3rem; margin-block: .3rem .6rem; }
li { margin-block: .25rem; }
p { margin-block: .4rem; }
.tablewrap { overflow-x: auto; margin-block: .6rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: .8rem; letter-spacing: .02em; white-space: nowrap; }
.tag { display: inline-block; font: 500 .72rem var(--font-mono); padding: .1rem .45rem; border-radius: 3px; white-space: nowrap; }
.tag[data-tag="相同"] { color: var(--tag-same); background: var(--tag-same-bg); }
.tag[data-tag="相反"] { color: var(--tag-opp); background: var(--tag-opp-bg); }
.tag[data-tag="可借用"] { color: var(--tag-borrow); background: var(--tag-borrow-bg); }
.tag[data-tag="支持"] { color: var(--tag-support); background: var(--tag-support-bg); }
.tag[data-tag="边界外"], .tag[data-tag="仅标题"], .tag[data-tag="其他"] { color: var(--tag-out); background: var(--tag-out-bg); }
.filters { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-block: .6rem 1rem; }
.filters button { font: 500 .78rem var(--font-mono); border: 1px solid var(--line); background: var(--surface); color: var(--fg); border-radius: 999px; padding: .25rem .7rem; cursor: pointer; }
.filters button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }
.filters input { font: inherit; font-size: .9rem; padding: .35rem .6rem; border: 1px solid var(--line); border-radius: 4px; background: var(--surface); color: var(--fg); min-width: 0; flex: 1 1 14rem; }
#papers td.t { min-width: 16rem; }
#papers td.d { white-space: nowrap; color: var(--muted); font-family: var(--font-mono); font-size: .8rem; }
#papers td.n { color: var(--muted); font-size: .86rem; }
.empty { color: var(--muted); font-style: italic; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; border-top: 1px solid var(--line); padding-top: 1rem; }
footer p { max-width: 68ch; }
@media (prefers-reduced-motion: no-preference) { details.round > summary { transition: color .15s; } }
</style>
<div class="wrap">
<header>
  <div class="eyebrow">World 0 · 每日研究与开发</div>
  <h1>World 0 研究简报</h1>
  <p class="sub">每天早上一轮：查最近一周的论文，选一项当天可完成的开发，先探针后改码，测试与命题验证通过后提交。
  这里是每一轮的记录与最新一轮的简报，由 <code>docs/research/build_brief.py</code> 从 <code>notes/*.md</code> 生成。</p>
  <div class="stats" id="stats"></div>
</header>

<h2>最新简报 <span class="count" id="latest-date"></span></h2>
<div class="brief" id="brief"></div>

<h2>全部轮次 <span class="count" id="rounds-count"></span></h2>
<div id="rounds"></div>

<h2>论文 <span class="count" id="papers-count"></span></h2>
<div class="filters" id="filters">
  <input id="paper-search" type="search" placeholder="搜索标题、arXiv id 或说明…" aria-label="搜索论文">
</div>
<div class="tablewrap"><table id="papers"><thead><tr><th>日期</th><th>论文</th><th>与 World 0</th><th>说明</th></tr></thead><tbody></tbody></table></div>

<footer>
  <p>协议：<code>docs/research/README.md</code>。笔记每日一份；本页在每轮结束时重建并随提交入库。生成于 <span class="mono" id="built"></span>。</p>
</footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  var data = JSON.parse(document.getElementById('data').textContent);
  var rounds = data.rounds.slice().sort(function (a, b) { return a.date < b.date ? 1 : -1; });
  var papers = [];
  rounds.forEach(function (r) { r.papers.forEach(function (p) { papers.push(p); }); });
  var byId = {};
  papers.forEach(function (p) { var k = p.arxiv || p.title; if (!byId[k]) byId[k] = p; });
  var uniq = Object.keys(byId).length;
  var devs = rounds.filter(function (r) { return r.sections.development.trim(); }).length;
  document.getElementById('built').textContent = data.built;
  document.getElementById('stats').innerHTML =
    stat(rounds.length, '轮次') + stat(uniq, '论文（去重）') + stat(devs, '开发项') + stat(data.tests || '—', '最近一轮测试');
  function stat(n, l) { return '<div class="stat"><span class="n">' + esc(String(n)) + '</span><span class="l">' + esc(l) + '</span></div>'; }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  var latest = rounds[0];
  if (latest) {
    document.getElementById('latest-date').textContent = latest.date;
    var nums = latest.numbers.length ? '<ul class="numbers">' + latest.numbers.map(function (n) { return '<li>' + esc(n) + '</li>'; }).join('') + '</ul>'
                                     : '<p class="empty">这一轮没有带数字的探针结论。</p>';
    document.getElementById('brief').innerHTML =
      '<div class="card lead"><div class="eyebrow">决策</div>' + (latest.html.decision || '<p class="empty">无</p>') +
      '<div class="eyebrow" style="margin-top:1rem">开发</div>' + (latest.html.development || '<p class="empty">无</p>') + '</div>' +
      '<div class="card"><div class="eyebrow">探针数字</div>' + nums +
      '<div class="eyebrow" style="margin-top:1rem">验证</div>' + (latest.html.verification || '<p class="empty">无</p>') +
      '<div class="eyebrow" style="margin-top:1rem">下一步</div>' + (latest.html.next || '<p class="empty">无</p>') + '</div>';
  } else {
    document.getElementById('brief').innerHTML = '<div class="card"><p class="empty">还没有笔记。第一轮运行后这里会出现当天的简报。</p></div>';
  }

  document.getElementById('rounds-count').textContent = rounds.length + ' 轮';
  document.getElementById('rounds').innerHTML = rounds.map(function (r, i) {
    var np = r.papers.length;
    var head = esc(r.decision_line || '（无决策记录）') + (np ? ' <span class="dim">· ' + np + ' 篇论文</span>' : '');
    function sec(label, key, wide) {
      var h = r.html[key];
      if (!h) return '';
      return '<div class="section' + (wide ? ' wide' : '') + '"><h4>' + label + '</h4>' + h + '</div>';
    }
    return '<details class="round"' + (i === 0 ? ' open' : '') + '><summary><span class="date">' + esc(r.date) + '</span><span class="headline">' + head + '</span></summary>' +
      '<div class="sections">' + sec('决策', 'decision') + sec('验证', 'verification') + sec('探针', 'probe', true) + sec('开发', 'development', true) +
      sec('论文', 'papers', true) + sec('提交', 'commits') + sec('未完成 / 下一步', 'next') + '</div></details>';
  }).join('');

  // papers table with tag filters and search
  var tags = ['相同', '相反', '可借用', '支持', '边界外', '仅标题', '其他'];
  var present = tags.filter(function (t) { return papers.some(function (p) { return p.tag === t; }); });
  var filters = document.getElementById('filters');
  var active = null;
  present.forEach(function (t) {
    var b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.setAttribute('aria-pressed', 'false'); b.id = 'f-' + t;
    b.addEventListener('click', function () { active = active === t ? null : t; render(); });
    filters.insertBefore(b, document.getElementById('paper-search'));
  });
  var search = document.getElementById('paper-search');
  search.addEventListener('input', render);
  function render() {
    var q = search.value.trim().toLowerCase();
    filters.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.textContent === active)); });
    var rows = papers.filter(function (p) { return (!active || p.tag === active) && (!q || (p.title + ' ' + p.arxiv + ' ' + p.note).toLowerCase().indexOf(q) >= 0); });
    document.getElementById('papers-count').textContent = rows.length + ' / ' + papers.length;
    document.querySelector('#papers tbody').innerHTML = rows.length ? rows.map(function (p) {
      var id = p.arxiv ? '<br><a class="id mono" href="https://arxiv.org/abs/' + esc(p.arxiv) + '" target="_blank" rel="noopener">arXiv:' + esc(p.arxiv) + '</a>' : '';
      return '<tr><td class="d">' + esc(p.date) + '</td><td class="t"><strong>' + esc(p.title) + '</strong>' + id + '</td>' +
        '<td><span class="tag" data-tag="' + esc(p.tag) + '">' + esc(p.tag) + '</span></td><td class="n">' + p.note_html + '</td></tr>';
    }).join('') : '<tr><td colspan="4" class="empty">没有匹配的论文。</td></tr>';
  }
  render();
})();
</script>
"""


def build() -> str:
    rounds = load_rounds()
    tests = ""
    if rounds:
        m = re.search(r"(\d[\d\s,]*)\s*passed", rounds[-1]["sections"]["verification"])
        if m:
            tests = m.group(1).replace(" ", "").replace(",", "") + " passed"
    data = {"built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "tests": tests,
            "rounds": [{k: v for k, v in r.items()} for r in rounds]}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def main(argv: list[str]) -> int:
    page = build()
    if "--stdout" in argv:
        sys.stdout.write(page)
        return 0
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parent.parent)} ({len(page) // 1024} KB, {len(load_rounds())} rounds)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
