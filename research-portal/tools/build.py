#!/usr/bin/env python3
"""World0 research portal builder.

Reads research-portal/data/*.json (source of truth), computes the daily
research delta, validates the records, and renders the static site.

    python3 tools/build.py --date YYYY-MM-DD

Only reports/<date>.html for the given date is (re)written; earlier daily
reports are never touched.  Run with -I (isolated mode) is fine; no deps.
"""
import argparse, datetime, html, json, os, re, sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
e = lambda s: html.escape(str(s), quote=True)

# ---------------------------------------------------------------- data ----
def load(name, default=None):
    p = os.path.join(D, name + ".json")
    if not os.path.exists(p):
        if default is None: raise SystemExit(f"missing data/{name}.json")
        return default
    with open(p, encoding="utf-8") as f: return json.load(f)

def save(name, obj):
    with open(os.path.join(D, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)

ID_RULES = {
    "papers": r"^(arxiv:\d{4}\.\d{4,5}|doi:.+)$", "findings": r"^F-\d{8}-\d{2}$",
    "evidence": r"^EV-\d{3,}$", "hypotheses": r"^H-\d{3,}$", "experiments": r"^E-\d{3,}$",
    "proposals": r"^AP-\d{3,}$", "reports": r"^R-\d{4}-\d{2}-\d{2}$",
}
REQ = {
    "papers": ["id","title","url","category","relevance","verification_level","first_seen","last_updated"],
    "findings": ["id","date","title","category","direction","importance","strength","paper","discovery","relevance","action"],
    "evidence": ["id","date","paper","kind","verification_level","statement","supports"],
    "hypotheses": ["id","hypothesis","motivation","status","evidence","next_experiment","history"],
    "experiments": ["id","hypothesis","description","status","history"],
    "proposals": ["id","component","area","title","change","benefit","complexity","risk","status","label","history"],
    "reports": ["id","date","file","title","status","summary","categories","importance"],
}
STRENGTH = {"CONFIRMED","SUPPORTED","EXPERIMENTAL","SPECULATIVE"}
HSTAT = {"PROPOSED","UNDER_TEST","SUPPORTED","REFUTED","INCONCLUSIVE"}
EVKIND = {"author_claim","experimental_observation","agent_interpretation","speculation"}

def validate(data):
    errs = []
    for name, rows in data.items():
        if name not in REQ: continue
        ids = [r.get("id") for r in rows]
        if len(ids) != len(set(ids)): errs.append(f"{name}: duplicate ids")
        for r in rows:
            for k in REQ[name]:
                if k not in r: errs.append(f"{name}/{r.get('id')}: missing {k}")
            if not re.match(ID_RULES[name], str(r.get("id"))): errs.append(f"{name}: bad id {r.get('id')}")
    papers = {p["id"] for p in data["papers"]}
    ev = {x["id"] for x in data["evidence"]}; hy = {h["id"] for h in data["hypotheses"]}
    ex = {x["id"] for x in data["experiments"]}; ap = {x["id"] for x in data["proposals"]}
    fi = {f["id"] for f in data["findings"]}
    lv = set(k for k in data["levels"] if k.startswith("V"))
    for p in data["papers"]:
        if p["verification_level"] not in lv: errs.append(f"paper {p['id']}: bad level")
    for x in data["evidence"]:
        if x["paper"] not in papers: errs.append(f"{x['id']}: unknown paper")
        if x["kind"] not in EVKIND: errs.append(f"{x['id']}: bad kind")
        for h in x["supports"]:
            if h not in hy: errs.append(f"{x['id']}: unknown hypothesis {h}")
    for f in data["findings"]:
        if f["paper"] not in papers: errs.append(f"{f['id']}: unknown paper")
        if f["strength"] not in STRENGTH: errs.append(f"{f['id']}: bad strength")
        for r in f.get("evidence", []):
            if r not in ev: errs.append(f"{f['id']}: unknown evidence {r}")
        # strength may not exceed what the verification levels allow
        levels = [data["pmap"][f["paper"]]["verification_level"]] + [data["emap"][r]["verification_level"] for r in f.get("evidence", []) if r in ev]
        srcs = {data["emap"][r]["paper"] for r in f.get("evidence", []) if r in ev} | {f["paper"]}
        if f["strength"] == "CONFIRMED" and not any(l >= "V3" for l in levels): errs.append(f"{f['id']}: CONFIRMED needs a V3+ source")
        if f["strength"] == "SUPPORTED" and len(srcs) < 2 and not any(l >= "V3" for l in levels): errs.append(f"{f['id']}: SUPPORTED needs 2 sources or V3")
    for h in data["hypotheses"]:
        if h["status"] not in HSTAT: errs.append(f"{h['id']}: bad status")
        if h["history"] and h["history"][-1]["status"] != h["status"]: errs.append(f"{h['id']}: status/history mismatch")
        for r in h["evidence"]:
            if r not in ev: errs.append(f"{h['id']}: unknown evidence {r}")
        if h["next_experiment"] and h["next_experiment"] not in ex: errs.append(f"{h['id']}: unknown experiment")
    for x in data["experiments"]:
        if x["hypothesis"] not in hy: errs.append(f"{x['id']}: unknown hypothesis")
    for x in data["proposals"]:
        if x["label"] not in STRENGTH: errs.append(f"{x['id']}: bad label")
        for r in x.get("findings", []):
            if r not in fi: errs.append(f"{x['id']}: unknown finding {r}")
    for r in data["reports"]:
        if not os.path.exists(os.path.join(ROOT, r["file"])) and r["date"] != data["date"]:
            errs.append(f"{r['id']}: report file missing")
    if errs:
        for x in errs: print("VALIDATION:", x, file=sys.stderr)
        raise SystemExit("validation failed")

# --------------------------------------------------------------- delta ----
def status_changes(rows, date):
    out = []
    for r in rows:
        h = r.get("history", [])
        for i, step in enumerate(h):
            if step.get("date") == date and i > 0 and step["status"] != h[i-1]["status"]:
                out.append(dict(id=r["id"], from_status=h[i-1]["status"], to_status=step["status"], reason=step.get("reason","")))
    return out

def compute_delta(data, date):
    def added(rows): return [r["id"] for r in rows if r.get("first_seen", r.get("date")) == date]
    def updated(rows): return [r["id"] for r in rows if r.get("last_updated") == date and r.get("first_seen", r.get("date")) != date]
    dates = sorted({r["date"] for r in data["reports"]})
    prev = [d for d in dates if d < date]
    return dict(
        date=date, previous_report=prev[-1] if prev else None,
        papers_added=added(data["papers"]), papers_updated=updated(data["papers"]),
        findings_new=[f["id"] for f in data["findings"] if f["date"] == date],
        evidence_new=[x["id"] for x in data["evidence"] if x["date"] == date],
        hypotheses_new=added(data["hypotheses"]), hypothesis_status_changes=status_changes(data["hypotheses"], date),
        experiments_new=added(data["experiments"]), experiment_status_changes=status_changes(data["experiments"], date),
        proposals_new=added(data["proposals"]), proposal_status_changes=status_changes(data["proposals"], date),
    )

# -------------------------------------------------------------- render ----
ZH = {}
def b(en, zh): return f'<span class="en">{en}</span><span class="zh">{zh}</span>'
def t(s):
    s = str(s); return b(e(s), e(ZH[s])) if s in ZH else e(s)
def tb(en):  # UI string with translation looked up in the dict
    return t(en)

LBL = {"V0":"TITLE_ONLY","V1":"SEARCH_SUMMARY","V2":"ABSTRACT_READ","V3":"FULL_TEXT_READ","V4":"INDEPENDENTLY_CHECKED"}
def vl(level): return f'<span class="vl {level}" title="{e(LBL.get(level,level))}">{e(level)} {t(LBL.get(level,level))}</span>'
def lbl(x): return f'<span class="lbl {e(x)}">{t(x)}</span>'
def chip(x, kind="cat"): return f'<span class="chip {kind}">{t(x)}</span>'

NAV_ITEMS = [("index.html","Today","今日"),("archive.html","Archive","归档"),("papers.html","Papers","论文"),("hypotheses.html","Hypotheses","假设"),("architecture.html","Architecture","架构")]
def page(title, body, depth, current=""):
    pre = "../" * depth
    nav = "".join(f'<a href="{pre}{h}"{" class=cur" if h == current else ""}>{b(en, zh)}</a>' for h, en, zh in NAV_ITEMS)
    return f"""<!doctype html><html lang="en" data-lang="both"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title><link rel="stylesheet" href="{pre}assets/css/style.css"></head><body>
<header class="top"><a class="brand" href="{pre}index.html">◆ WORLD0 RESEARCH INTELLIGENCE</a><nav>{nav}</nav><div class="langsw" role="group" aria-label="Language / 语言"><button data-l="en">EN</button><button data-l="zh">中文</button><button data-l="both">EN+中</button></div></header>
<main>{body}</main>
<footer>{b('Autonomous research output. Evidence labels describe literature support, not World0 implementation status. Verification levels: V0 title only · V1 search summary · V2 abstract · V3 full text · V4 independently checked.','自主研究产出。证据标签描述文献支持程度，并不代表 World0 已实现相应能力。验证等级：V0 仅标题 · V1 搜索摘要 · V2 摘要 · V3 全文 · V4 独立核查。')}</footer>
<script src="{pre}assets/js/app.js"></script></body></html>"""

def table(headers, rows, attrs=None):
    """headers: list of (en, zh); rows: list of list of html cells; attrs: per-row attr strings."""
    th = "".join(f"<th>{b(e(en), e(zh))}</th>" for en, zh in headers)
    body = ""
    for i, r in enumerate(rows):
        a = (attrs or [""] * len(rows))[i]
        body += f"<tr {a}>" + "".join(f'<td data-label="{e(headers[j][0])}">{c}</td>' for j, c in enumerate(r)) + "</tr>"
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'

def ul(items): return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"

def paper_link(pm, pid, depth):
    p = pm[pid]; pre = "../" * depth
    if p.get("argument"): return f'<a href="{pre}papers/{slug(pid)}.html">{e(p["title"])}</a>'
    return f'<a href="{e(p["url"])}" rel="noopener">{e(p["title"])}</a>'
def slug(pid): return re.sub(r"[^a-z0-9]+", "-", pid.lower()).strip("-")

def render_index(data, delta):
    pm, hm, em, apm = data["pmap"], data["hmap"], data["emap"], data["apmap"]
    date = data["date"]; rep = data["rmap"].get(date)
    st = data["state"]
    dep = st.get("last_successful_deployment")
    # --- delta strip
    sc = delta["hypothesis_status_changes"]
    def n(k): return len(delta[k])
    strip = "".join(f'<div><b>{v}</b><span>{t(k)}</span></div>' for k, v in [
        ("papers added", n("papers_added")), ("papers updated", n("papers_updated")), ("new findings", n("findings_new")),
        ("new evidence", n("evidence_new")), ("new hypotheses", n("hypotheses_new")), ("hypothesis status changes", len(sc)),
        ("new proposals", n("proposals_new"))])
    prev = delta["previous_report"]
    prevtxt = f'{t("since")} <a href="reports/{prev}.html">{prev}</a>' if prev else t("first research session; no previous report to diff against")
    # --- top discoveries
    todays = sorted([f for f in data["findings"] if f["date"] == date], key=lambda f: ({"HIGH":0,"MEDIUM":1,"LOW":2}[f["importance"]]))[:3]
    cards = "".join(f'''<article class="card find" data-cat="{e(f["category"])}"><div class="hd">{chip(f["category"])} {lbl(f["strength"])} <span class="imp {e(f["importance"])}">{t(f["importance"])}</span> {vl(pm[f["paper"]]["verification_level"])}</div>
<h3>{t(f["title"])}</h3><p><b>{t("Discovery")}:</b> {t(f["discovery"])}</p><p><b>World0:</b> {t(f["relevance"])}</p><p><b>{t("Action")}:</b> {t(f["action"])}</p>
<p class="src">{t("Source")}: {paper_link(pm, f["paper"], 0)} · <a href="reports/{date}.html#{e(f["id"])}">{t("analysis")} →</a></p></article>''' for f in todays)
    # --- World0 impact
    todays_ap = [p for p in data["proposals"] if p["id"] in delta["proposals_new"] or any(c["id"] == p["id"] for c in delta["proposal_status_changes"])]
    imp_rows = [[f'<code>{e(p["component"])}</code>', f'<a href="architecture.html#{e(p["id"])}">{t(p["title"])}</a>', lbl(p["label"]), t(p["complexity"]), t(p["status"])] for p in todays_ap]
    impact = table([("Component","组件"),("Proposal","提案"),("Evidence","证据"),("Complexity","复杂度"),("Status","状态")], imp_rows) if imp_rows else f'<p class="note">{t("No new architecture proposals today.")}</p>'
    # --- hypothesis movement
    rows = []
    for c in sc:
        h = hm[c["id"]]
        rows.append([f'<a href="hypotheses.html#{e(c["id"])}">{e(c["id"])}</a>', t(h["hypothesis"]), f'{t(c["from_status"])} → <b>{t(c["to_status"])}</b>', t(c["reason"])])
    for hid in delta["hypotheses_new"]:
        h = hm[hid]
        rows.append([f'<a href="hypotheses.html#{e(hid)}">{e(hid)}</a>', t(h["hypothesis"]), f'<b>{t("NEW")}</b> · {t(h["status"])}', t(h["history"][0].get("reason",""))])
    hyp = table([("ID","ID"),("Hypothesis","假设"),("Movement","变动"),("Why","原因")], rows) if rows else f'<p class="note">{t("No hypothesis gained or lost support today.")}</p>'
    # --- next
    nxt = ul([t(x) for x in (rep or {}).get("next_priorities", [])] + [f'<code>{e(x["id"])}</code> {t(x["description"])}' for x in data["experiments"] if x["status"] == "PENDING"][:3])
    # --- radar (today's papers only; full list on papers.html)
    radar_rows, radar_attrs = [], []
    for pid in delta["papers_added"] + delta["papers_updated"]:
        p = pm[pid]
        radar_rows.append([paper_link(pm, pid, 0), e(p.get("authors") or "—"), e(p["published"]), chip(p["category"]), f'{p["relevance"]:.2f}{" ★" if p["relevance"] >= 0.85 else ""}', vl(p["verification_level"]), f'<a href="{e(p["url"])}" rel="noopener">{e(p["id"])}</a>'])
        radar_attrs.append(f'data-cat="{e(p["category"])}"')
    radar = table([("Paper","论文"),("Authors","作者"),("Published","发表"),("Category","类别"),("Relevance","相关度"),("Verification","验证"),("Source","来源")], radar_rows, radar_attrs)
    recent = sorted(data["reports"], key=lambda r: r["date"], reverse=True)[:7]
    timeline = ul([f'<a href="{e(r["file"])}">{e(r["date"])}</a> · {t(r["title"])} · <span class="imp {e(r["importance"])}">{t(r["importance"])}</span> · {e(r["status"])}' for r in recent])
    weekly = sorted(os.listdir(os.path.join(ROOT, "weekly"))) if os.path.isdir(os.path.join(ROOT, "weekly")) else []
    weekly = [w for w in weekly if w.endswith(".html")]
    wk = f'<a href="weekly/{e(weekly[-1])}">{e(weekly[-1][:-5])}</a>' if weekly else t("none yet")
    def m(en, zh, v): return f'<div><dt>{b(en, zh)}</dt><dd>{v}</dd></div>'
    act = sum(1 for h in data["hypotheses"] if h["status"] in ("PROPOSED","UNDER_TEST"))
    pend = sum(1 for x in data["experiments"] if x["status"] == "PENDING")
    body = f"""<section class="hero"><h1>WORLD0 RESEARCH INTELLIGENCE</h1><p class="sub">Autonomous Cognitive Systems Research · 自主认知系统研究</p>
<dl class="meta">{m("Report date","报告日期",e(date))}{m("Research updated","研究更新时间",e(st.get("last_research_update","")))}{m("Verified deployment","已验证部署",e(dep) if dep else t("not yet verified"))}{m("Papers tracked","跟踪论文数",len(data["papers"]))}{m("Active hypotheses","活跃假设",act)}{m("Pending experiments","待做实验",pend)}{m("Weekly synthesis","周度综述",wk)}</dl></section>
<section id="delta"><h2>1 · {t("What changed today")}</h2><p class="note">{prevtxt} · <a href="data/deltas/{e(date)}.json">delta.json</a> · {t("Session status")}: <b>{e((rep or {}).get("status","?"))}</b></p><div class="strip">{strip}</div>
{('<p>' + t(rep["summary"]) + '</p>') if rep else ''}</section>
<section id="top"><h2>2 · {t("Most important discoveries")}</h2><div class="filters" data-target=".find"></div><div class="grid">{cards}</div></section>
<section id="impact"><h2>3 · {t("How this affects World0")}</h2>{impact}<p class="note"><a href="architecture.html">{t("Full architecture impact analysis")} →</a></p></section>
<section id="hyp"><h2>4 · {t("Hypotheses that moved")}</h2>{hyp}<p class="note"><a href="hypotheses.html">{t("Hypothesis lifecycle")} →</a></p></section>
<section id="next"><h2>5 · {t("What to investigate next")}</h2>{nxt}</section>
<section id="radar"><h2>6 · {t("Research radar")}</h2><div class="filters" data-target="#radar tbody tr"></div>{radar}<p class="note"><a href="papers.html">{t("All tracked papers")} →</a></p></section>
<section id="timeline"><h2>7 · {t("Recent reports")}</h2>{timeline}<p class="note"><a href="archive.html">{t("Full archive with filters")} →</a></p></section>"""
    return page("World0 Research Intelligence · 研究情报", body, 0, "index.html")

def render_archive(data):
    rows, attrs = [], []
    for r in sorted(data["reports"], key=lambda r: r["date"], reverse=True):
        rows.append([f'<a href="{e(r["file"])}">{e(r["date"])}</a>', t(r["title"]), " ".join(chip(c) for c in r["categories"]), " ".join(chip(d, "dir") for d in r.get("directions", [])), f'<span class="imp {e(r["importance"])}">{t(r["importance"])}</span>', e(r["status"]), f'<a href="data/deltas/{e(r["date"])}.json">Δ</a>' if os.path.exists(os.path.join(D, "deltas", r["date"] + ".json")) else "—"])
        attrs.append(f'data-cat="{e("|".join(r["categories"]))}" data-dir="{e("|".join(r.get("directions", [])))}" data-imp="{e(r["importance"])}" data-date="{e(r["date"])}"')
    cats = sorted({c for r in data["reports"] for c in r["categories"]}); dirs = sorted({d for r in data["reports"] for d in r.get("directions", [])})
    def sel(name, en, zh, opts): return f'<label>{b(en, zh)} <select data-filter="{name}"><option value="">all / 全部</option>' + "".join(f'<option value="{e(o)}">{e(o)}</option>' for o in opts) + "</select></label>"
    ctl = f'<div class="ctl">{sel("cat","Category","类别",cats)}{sel("dir","Direction","方向",dirs)}{sel("imp","Importance","重要性",["HIGH","MEDIUM","LOW"])}<label>{b("Date","日期")} <input type="text" data-filter="date" placeholder="2026-10"></label></div>'
    years = {}
    for r in data["reports"]: years.setdefault(r["date"][:7], []).append(r)
    months = ul([f'<a href="#m{e(k)}">{e(k)}</a> ({len(v)})' for k, v in sorted(years.items(), reverse=True)])
    weekly = sorted(f for f in os.listdir(os.path.join(ROOT, "weekly")) if f.endswith(".html")) if os.path.isdir(os.path.join(ROOT, "weekly")) else []
    wk = ul([f'<a href="weekly/{e(w)}">{e(w[:-5])}</a>' for w in reversed(weekly)]) if weekly else f'<p class="note">{t("none yet")}</p>'
    body = f"""<h1>{t("Research archive")}</h1><p class="note">{t("Every daily report is kept permanently. Filter by category, research direction, importance or date.")}</p>
{ctl}<section id="reports">{table([("Date","日期"),("Report","报告"),("Categories","类别"),("Directions","方向"),("Importance","重要性"),("Status","状态"),("Delta","增量")], rows, attrs)}</section>
<h2>{t("By month")}</h2>{months}<h2>{t("Weekly syntheses")}</h2>{wk}"""
    return page("World0 Research Archive · 归档", body, 0, "archive.html")

def render_papers(data):
    pm = data["pmap"]; rows, attrs = [], []
    for p in sorted(data["papers"], key=lambda p: (-p["relevance"], p["id"])):
        rows.append([paper_link(pm, p["id"], 0), e(p.get("authors") or "—"), e(p["published"]), chip(p["category"]), f'{p["relevance"]:.2f}{" ★" if p["relevance"] >= 0.85 else ""}', vl(p["verification_level"]), e(p["first_seen"]), f'<a href="{e(p["url"])}" rel="noopener">{e(p["id"])}</a>'])
        attrs.append(f'data-cat="{e(p["category"])}" data-imp="{e(p["verification_level"])}"')
    body = f"""<h1>{t("Tracked papers")}</h1><p class="note">{t("Deduplicated by arXiv ID / DOI. ★ marks high architectural relevance (≥ 0.85). Click a title for the structured analysis where one exists.")}</p>
<div class="filters" data-target="#plist tbody tr"></div><section id="plist">{table([("Paper","论文"),("Authors","作者"),("Published","发表"),("Category","类别"),("Relevance","相关度"),("Verification","验证"),("First seen","首次收录"),("Source","来源")], rows, attrs)}</section>"""
    return page("World0 Papers · 论文", body, 0, "papers.html")

def render_paper(data, p):
    pm, em = data["pmap"], data["emap"]; a = p["argument"]
    evs = [x for x in data["evidence"] if x["paper"] == p["id"]]
    fs = [f for f in data["findings"] if f["paper"] == p["id"]]
    aps = [x for x in data["proposals"] if any(f["id"] in x.get("findings", []) for f in fs)]
    def sec(en, zh, items, kind):
        return f'<h3>{b(en, zh)} <span class="kind">{t(kind)}</span></h3>' + ul([t(x) for x in items])
    body = f"""<p class="crumbs"><a href="../papers.html">{t("Papers")}</a> / {e(p["id"])}</p><h1>{e(p["title"])}</h1>
<dl class="meta"><div><dt>{t("Authors")}</dt><dd>{e(p["authors"]) if p.get("authors") else t("not recovered")}</dd></div><div><dt>{t("Published")}</dt><dd>{e(p["published"])}</dd></div><div><dt>{t("Source")}</dt><dd><a href="{e(p["url"])}" rel="noopener">{e(p["source"])} · {e(p["id"])}</a></dd></div><div><dt>DOI</dt><dd>{e(p.get("doi") or "—")}</dd></div><div><dt>{t("Category")}</dt><dd>{chip(p["category"])}</dd></div><div><dt>{t("Relevance")}</dt><dd>{p["relevance"]:.2f}</dd></div><div><dt>{t("Verification")}</dt><dd>{vl(p["verification_level"])}</dd></div><div><dt>{t("First seen")}</dt><dd>{e(p["first_seen"])} · {t("updated")} {e(p["last_updated"])}</dd></div></dl>
<p class="note">{t(data["rmap"][p["first_seen"]]["verification_note"]) if p["first_seen"] in data["rmap"] and data["rmap"][p["first_seen"]].get("verification_note") else ""}</p>
<section><h2>{t("Scientific argument")}</h2>{sec("Claims","主张",a["claims"],"author claim")}{sec("Premises / assumptions","前提 / 假设",a["premises"],"author claim")}{sec("Evidence","证据",a["evidence"],"experimental observation")}{sec("Counterpoints","反驳 / 质疑",a["counterpoints"],"agent interpretation")}{sec("Research-agent interpretation","研究代理解读",a["interpretation"],"agent interpretation")}</section>
<section><h2>{t("Core thesis")}</h2><p>{t(p["thesis"])}</p><h2>{t("Mechanism")}</h2><p>{t(p["mechanism"])}</p><h2>{t("Experimental evidence")}</h2><p>{t(p["evidence"])}</p><h2>{t("Limitations")}</h2><p>{t(p["limitations"])}</p><h2>{t("World0 implications")}</h2><p>{t(p["implications"])}</p></section>
<section><h2>{t("Evidence records")}</h2>{table([("ID","ID"),("Kind","类型"),("Level","等级"),("Statement","陈述"),("Supports","支持")], [[e(x["id"]), t(x["kind"]), vl(x["verification_level"]), t(x["statement"]), ", ".join(f'<a href="../hypotheses.html#{e(h)}">{e(h)}</a>' for h in x["supports"]) or "—"] for x in evs])}</section>
<section><h2>{t("Linked findings and proposals")}</h2>{ul([f'<a href="../reports/{e(f["date"])}.html#{e(f["id"])}">{e(f["id"])}</a> {t(f["title"])}' for f in fs] + [f'<a href="../architecture.html#{e(x["id"])}">{e(x["id"])}</a> {t(x["title"])}' for x in aps])}</section>"""
    return page(p["title"], body, 1, "papers.html")

def render_hypotheses(data):
    em, xm = data["emap"], data["xmap"]
    parts = []
    for h in data["hypotheses"]:
        hist = ul([f'<span class="when">{e(s["date"])}</span> {t(s["status"])} — {t(s.get("reason",""))}' for s in h["history"]])
        evs = table([("ID","ID"),("Kind","类型"),("Level","等级"),("Statement","陈述"),("Paper","论文")], [[e(x), t(em[x]["kind"]), vl(em[x]["verification_level"]), t(em[x]["statement"]), paper_link(data["pmap"], em[x]["paper"], 0)] for x in h["evidence"]]) if h["evidence"] else f'<p class="note">{t("no evidence recorded")}</p>'
        exps = [x for x in data["experiments"] if x["hypothesis"] == h["id"]]
        ex = ul([f'<code>{e(x["id"])}</code> {t(x["status"])} · {t(x["description"])}' for x in exps]) or f'<p class="note">{t("no experiment")}</p>'
        parts.append(f'''<article class="card" id="{e(h["id"])}" data-cat="{e(h["status"])}"><div class="hd"><code>{e(h["id"])}</code> {lbl(h["status"])} <span class="when">{t("since")} {e(h["first_seen"])}</span></div><h3>{t(h["hypothesis"])}</h3><p><b>{t("Motivation")}:</b> {t(h["motivation"])}</p>
<h4>{t("Lifecycle")}</h4>{hist}<h4>{t("Supporting evidence")}</h4>{evs}<h4>{t("Experiments")}</h4>{ex}</article>''')
    body = f"""<h1>{t("Research hypotheses")}</h1><p class="note">{t("Status: PROPOSED → UNDER_TEST → SUPPORTED / REFUTED / INCONCLUSIVE. Every status change is dated and justified.")}</p><div class="filters" data-target="main .card"></div>{"".join(parts)}"""
    return page("World0 Hypotheses · 假设", body, 0, "hypotheses.html")

def render_architecture(data):
    comps = ["Dynamic Ontology","Self-Model","Dream Mechanism","World Models","Cognitive Architecture"]
    latest = data["rmap"].get(data["date"], {})
    ins = {i["area"]: i for i in latest.get("insights", [])}
    overview = table([("Area","领域"),("Current evidence","当前证据"),("Note","说明"),("Open proposals","开放提案")], [[t(c), lbl(ins[c]["label"]) if c in ins else "—", t(ins[c]["note"]) if c in ins else "—", ", ".join(f'<a href="#{e(x["id"])}">{e(x["id"])}</a>' for x in data["proposals"] if x["area"] == c and x["status"] not in ("REJECTED","IMPLEMENTED")) or "—"] for c in comps])
    parts = []
    for x in data["proposals"]:
        fs = ", ".join(f'<a href="reports/{e(data["fmap"][f]["date"])}.html#{e(f)}">{e(f)}</a>' for f in x.get("findings", []))
        parts.append(f'''<article class="card" id="{e(x["id"])}" data-cat="{e(x["area"])}"><div class="hd"><code>{e(x["id"])}</code> {chip(x["area"])} {lbl(x["label"])} {t(x["status"])}</div><h3>{t(x["title"])}</h3><p><b>{t("Component")}:</b> <code>{e(x["component"])}</code></p><p><b>{t("Change")}:</b> {t(x["change"])}</p><p><b>{t("Expected benefit")}:</b> {t(x["benefit"])}</p><p><b>{t("Complexity")}:</b> {t(x["complexity"])} · <b>{t("Risk")}:</b> {t(x["risk"])}</p><p class="src">{t("Findings")}: {fs or "—"} · {t("Evidence")}: {e(", ".join(x.get("evidence", [])) or "—")} · {t("Experiments")}: {e(", ".join(x.get("experiments", [])) or "—")}</p>{ul([f'<span class="when">{e(s["date"])}</span> {t(s["status"])}' for s in x["history"]])}</article>''')
    body = f"""<h1>{t("World0 architecture impact")}</h1><p class="note">{t("Labels describe the literature evidence behind a proposal, not whether World0 has implemented it.")}</p>{overview}<h2>{t("Proposals")}</h2><div class="filters" data-target="main .card"></div>{"".join(parts)}"""
    return page("World0 Architecture Impact · 架构影响", body, 0, "architecture.html")

def render_report(data, delta):
    date = data["date"]; r = data["rmap"][date]; pm = data["pmap"]
    fs = [f for f in data["findings"] if f["date"] == date]
    fcards = "".join(f'<article class="card" id="{e(f["id"])}"><div class="hd">{chip(f["category"])} {lbl(f["strength"])} <span class="imp {e(f["importance"])}">{t(f["importance"])}</span></div><h3>{t(f["title"])}</h3><p>{t(f["discovery"])}</p><p><b>World0:</b> {t(f["relevance"])}</p><p><b>{t("Action")}:</b> {t(f["action"])}</p><p class="src">{paper_link(pm, f["paper"], 1)} · {t("evidence")}: {e(", ".join(f.get("evidence", [])))}</p></article>' for f in fs)
    radar = table([("Paper","论文"),("Authors","作者"),("Published","发表"),("Category","类别"),("Relevance","相关度"),("Verification","验证"),("Source","来源")], [[paper_link(pm, pid, 1), e(pm[pid].get("authors") or "—"), e(pm[pid]["published"]), chip(pm[pid]["category"]), f'{pm[pid]["relevance"]:.2f}', vl(pm[pid]["verification_level"]), f'<a href="{e(pm[pid]["url"])}" rel="noopener">{e(pid)}</a>'] for pid in delta["papers_added"] + delta["papers_updated"]])
    deep = ""
    for p in [pm[f["paper"]] for f in fs if pm[f["paper"]].get("argument")]:
        a = p["argument"]
        deep += f'<article class="card"><h3><a href="../papers/{slug(p["id"])}.html">{e(p["title"])}</a></h3><p>{vl(p["verification_level"])} · <a href="{e(p["url"])}" rel="noopener">{e(p["id"])}</a></p><h4>{t("Claims")}</h4>{ul([t(x) for x in a["claims"]])}<h4>{t("Evidence")}</h4>{ul([t(x) for x in a["evidence"]])}<h4>{t("Counterpoints")}</h4>{ul([t(x) for x in a["counterpoints"]])}<h4>{t("World0 implications")}</h4><p>{t(p["implications"])}</p></article>'
    ins = table([("Area","领域"),("Label","标签"),("Note","说明")], [[t(i["area"]), lbl(i["label"]), t(i["note"])] for i in r.get("insights", [])])
    hyps = table([("ID","ID"),("Hypothesis","假设"),("Status","状态"),("Evidence","证据"),("Next","下一步")], [[f'<a href="../hypotheses.html#{e(h["id"])}">{e(h["id"])}</a>', t(h["hypothesis"]), t(h["status"]), e(", ".join(h["evidence"])), e(h["next_experiment"])] for h in data["hypotheses"] if h["id"] in delta["hypotheses_new"] or any(c["id"] == h["id"] for c in delta["hypothesis_status_changes"])])
    aps = ul([f'<a href="../architecture.html#{e(x)}">{e(x)}</a> {t(data["apmap"][x]["title"])} — <code>{e(data["apmap"][x]["component"])}</code>' for x in delta["proposals_new"]]) or f'<p class="note">{t("none")}</p>'
    kupd = ul([t(f"{len(delta['papers_added'])} papers added"), t(f"{len(delta['evidence_new'])} evidence records added"), t(f"{len(delta['hypotheses_new'])} hypotheses proposed"), t(f"{len(delta['hypothesis_status_changes'])} hypothesis status changes"), t(f"{len(delta['proposals_new'])} architecture proposals added")])
    srcs = ul([f'<a href="{e(pm[pid]["url"])}" rel="noopener">{e(pm[pid]["title"])}</a> ({e(pid)})' for pid in delta["papers_added"] + delta["papers_updated"]])
    body = f"""<p class="crumbs"><a href="../archive.html">{t("Archive")}</a> / {e(date)}</p><h1>{t("Daily report")} {e(date)}</h1><p class="note">{t("Generated")} {e(data["now"])} · {t("Status")}: <b>{e(r["status"])}</b> · <a href="../data/deltas/{e(date)}.json">delta.json</a></p>
<h2>1 {t("Executive summary")}</h2><p>{t(r["summary"])}</p>{("<p class=note>" + t(r["verification_note"]) + "</p>") if r.get("verification_note") else ""}
<h2>2 {t("Key discoveries")}</h2>{fcards}<h2>3 {t("Research radar")}</h2>{radar}<h2>4 {t("Deep paper analysis")}</h2>{deep}
<h2>5 {t("World0 architecture implications")}</h2>{ins}{aps}<h2>6 {t("Hypotheses")}</h2>{hyps}
<h2>7 {t("Engineering recommendations")}</h2>{ul([t(x) for x in r.get("recommendations", [])])}<h2>8 {t("Research knowledge updates")}</h2>{kupd}
<h2>9 {t("Open questions")}</h2>{ul([t(x) for x in r.get("open_questions", [])])}<h2>10 {t("Next research priorities")}</h2>{ul([t(x) for x in r.get("next_priorities", [])])}<h2>{t("Sources")}</h2>{srcs}"""
    return page(f"World0 Report {date}", body, 1, "archive.html")

# ------------------------------------------------------------- checks ----
class _P(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ("a", "link", "script") and (d.get("href") or d.get("src")): self.links.append(d.get("href") or d.get("src"))

def check_html(written):
    bad = []
    for path in written:
        p = _P(); p.feed(open(path, encoding="utf-8").read())
        for l in p.links:
            if l.startswith(("http://", "https://", "#", "mailto:")): continue
            target = os.path.normpath(os.path.join(os.path.dirname(path), l.split("#")[0]))
            if not os.path.exists(target): bad.append(f"{os.path.relpath(path, ROOT)} -> {l}")
    if bad:
        for x in bad: print("LINK:", x, file=sys.stderr)
        raise SystemExit("broken internal links")

# --------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--deployed", help="mark this ISO timestamp as last verified deployment")
    a = ap.parse_args()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", a.date): raise SystemExit("bad --date")
    global ZH; ZH = load("i18n_zh", {})
    data = {k: load(k) for k in ["papers","findings","evidence","hypotheses","experiments","proposals","reports"]}
    data["levels"] = load("verification_levels"); data["state"] = load("research_state", {"runs": []})
    data["date"] = a.date; data["now"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data["pmap"] = {p["id"]: p for p in data["papers"]}; data["hmap"] = {h["id"]: h for h in data["hypotheses"]}
    data["emap"] = {x["id"]: x for x in data["evidence"]}; data["xmap"] = {x["id"]: x for x in data["experiments"]}
    data["apmap"] = {x["id"]: x for x in data["proposals"]}; data["fmap"] = {f["id"]: f for f in data["findings"]}
    data["rmap"] = {r["date"]: r for r in data["reports"]}
    if a.date not in data["rmap"]: raise SystemExit(f"no reports.json entry for {a.date}")
    validate(data)
    delta = compute_delta(data, a.date)
    os.makedirs(os.path.join(D, "deltas"), exist_ok=True)
    with open(os.path.join(D, "deltas", a.date + ".json"), "w", encoding="utf-8") as f: json.dump(delta, f, indent=1)
    st = data["state"]
    st["runs"] = [r for r in st.get("runs", []) if r["date"] != a.date] + [dict(date=a.date, generated=data["now"], status=data["rmap"][a.date]["status"], papers_added=len(delta["papers_added"]))]
    st["last_research_update"] = data["now"]
    if a.deployed: st["last_successful_deployment"] = a.deployed
    st.setdefault("last_successful_deployment", None)
    save("research_state", st)
    save("latest", dict(date=a.date, generated=data["now"], delta=delta, findings=[f for f in data["findings"] if f["date"] == a.date], insights=data["rmap"][a.date].get("insights", [])))
    written = []
    def w(rel, content):
        p = os.path.join(ROOT, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f: f.write(content)
        written.append(p)
    for p in data["papers"]:
        if p.get("argument"): w(f"papers/{slug(p['id'])}.html", render_paper(data, p))
    w(f"reports/{a.date}.html", render_report(data, delta))
    w("index.html", render_index(data, delta)); w("archive.html", render_archive(data)); w("papers.html", render_papers(data))
    w("hypotheses.html", render_hypotheses(data)); w("architecture.html", render_architecture(data))
    check_html(written + [os.path.join(ROOT, r["file"]) for r in data["reports"]])
    missing = sorted({s for s in _untranslated})
    print(f"built {len(written)} pages for {a.date}; delta: +{len(delta['papers_added'])} papers, {len(delta['findings_new'])} findings, {len(delta['hypothesis_status_changes'])} hypothesis changes; {len(missing)} untranslated strings")
    if os.environ.get("W0_SHOW_UNTRANSLATED"):
        for s in missing: print("  zh?", s)

_untranslated = set()
_t = t
def t(s):  # wrap to collect strings lacking a Chinese translation
    s = str(s)
    if s not in ZH and not re.fullmatch(r"[\d\W_]*|[A-Z0-9_\-]+|not recovered", s): _untranslated.add(s)
    return _t(s)

if __name__ == "__main__": main()
