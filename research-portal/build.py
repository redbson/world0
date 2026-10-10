#!/usr/bin/env python3
"""Build the World0 research portal from its data files.

    python3 build.py               # validate the data, write every generated file
    python3 build.py --check       # validate, and fail if a generated file is stale
    python3 build.py --no-history  # skip the comparison with the last commit
    python3 build.py --missing-zh  # list visible strings without a Chinese rendering

Inputs (written by the research routine; see README.md):
    data/papers.json         every paper ever recorded, one record per canonical id
    data/hypotheses.json     hypotheses with their status history
    data/experiments.json    experiments with their status history
    data/reports/DATE.json   one record per research session (immutable once committed)
    data/weekly/WEEK.json    weekly syntheses
    data/research_state.json publishing target and verified deployments
    data/i18n_zh.json        Chinese renderings of visible strings (optional)

Outputs (never edit by hand):
    index.html, reports/DATE.html, papers/ID.html, weekly/WEEK.html,
    data/latest.json, and the derived fields of data/research_state.json

Standard library only.  Every string taken from the data is HTML-escaped and
only http(s) URLs become links, so research text cannot inject markup.  The
output depends on the data alone (no clock), so ``--check`` is exact.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

AREAS = ("Dynamic Ontology", "Self-Model", "Dream Mechanism", "World Models", "Cognitive Architecture")
EVIDENCE = ("CONFIRMED", "SUPPORTED", "EXPERIMENTAL", "SPECULATIVE")
H_STATUS = ("PROPOSED", "UNDER_TEST", "SUPPORTED", "REFUTED", "INCONCLUSIVE")
H_ACTIVE = ("PROPOSED", "UNDER_TEST")
E_STATUS = ("PENDING", "RUNNING", "COMPLETED", "ABANDONED")
E_OPEN = ("PENDING", "RUNNING")
RUN_STATUS = ("COMPLETE", "PARTIAL", "FAILED")
IMPORTANCE = ("HIGH", "MEDIUM", "LOW")
HIGH_RELEVANCE = 0.8

SYMBOL = {
    "CONFIRMED": "■", "SUPPORTED": "◆", "EXPERIMENTAL": "▲", "SPECULATIVE": "○",
    "PROPOSED": "◇", "UNDER_TEST": "◈", "REFUTED": "✕", "INCONCLUSIVE": "◌",
    "PENDING": "□", "RUNNING": "◧", "COMPLETED": "■", "ABANDONED": "✕",
    "COMPLETE": "●", "PARTIAL": "◐", "FAILED": "✕",
    "HIGH": "▲", "MEDIUM": "◆", "LOW": "▽",
}

# Text sections of a paper analysis, with the provenance each one carries.
PAPER_SECTIONS = (
    ("thesis", "Core Thesis", "AUTHOR CLAIM"),
    ("foundations", "Theoretical Foundations", "AUTHOR CLAIM"),
    ("mechanism", "Proposed Mechanism", "AUTHOR CLAIM"),
    ("evidence", "Experimental Evidence", "REPORTED RESULTS"),
    ("limitations", "Limitations", "AGENT INTERPRETATION"),
    ("implications", "World0 Implications", "AGENT INTERPRETATION"),
    ("assessment", "Researcher's Assessment", "AGENT INTERPRETATION"),
)
IMPLICATION_KEYS = ("components", "improvements", "complexity", "benefits", "risks")
ASSESSMENT_KEYS = ("author_claims", "observations", "interpretation", "speculation")
KEY_LABEL = {
    "components": "Related World0 components", "improvements": "Potential improvements",
    "complexity": "Implementation complexity", "benefits": "Expected benefits",
    "risks": "Integration risks", "author_claims": "Author claims",
    "observations": "Experimental observations", "interpretation": "Agent interpretation",
    "speculation": "Speculative hypotheses",
}
TRACKED = {
    "paper": ("title", "authors", "published", "url", "category", "relevance", "verification",
              "thesis", "foundations", "mechanism", "evidence", "limitations", "implications", "assessment"),
    "hypothesis": ("hypothesis", "motivation", "evidence", "next_experiment"),
    "experiment": ("hypothesis", "description", "result", "artifact"),
}

PAPER_ID = re.compile(r"^(arxiv:\d{4}\.\d{4,5}|doi:10\.\d{4,9}/[^\s]+|title:[a-z0-9]+(?:-[a-z0-9]+)*)$")
PAPER_REF = re.compile(r"^(arxiv:\d{4}\.\d{4,5}|doi:10\.\d{4,9}/[^\s;,]+|title:[a-z0-9]+(?:-[a-z0-9]+)*)(.*)$", re.S)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")
WEEK = re.compile(r"^\d{4}-W\d{2}$")
PUBLISHED = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
HID = re.compile(r"^H-\d{3,}$")
EID = re.compile(r"^E-\d{3,}$")
FID = re.compile(r"^F-\d+$")
SHA = re.compile(r"^[0-9a-f]{7,40}$")


# --------------------------------------------------------------------------
# validation


class Problems(list):
    def need(self, cond: bool, where: str, msg: str) -> bool:
        if not cond:
            self.append(f"{where}: {msg}")
        return cond


def is_http(url: Any) -> bool:
    return isinstance(url, str) and re.match(r"^https?://[^\s<>\"']+$", url) is not None


def _text(p: Problems, obj: dict, key: str, where: str, required: bool = True) -> None:
    v = obj.get(key)
    if v is None and not required:
        return
    p.need(isinstance(v, str) and v.strip() != "", where, f"'{key}' must be a non-empty string")


def _strings(p: Problems, obj: dict, key: str, where: str) -> None:
    v = obj.get(key, [])
    p.need(isinstance(v, list) and all(isinstance(s, str) and s.strip() for s in v),
           where, f"'{key}' must be a list of non-empty strings")


def _links(p: Problems, obj: dict, key: str, where: str) -> None:
    v = obj.get(key, [])
    if not p.need(isinstance(v, list), where, f"'{key}' must be a list"):
        return
    for i, s in enumerate(v):
        w = f"{where} {key}[{i}]"
        if p.need(isinstance(s, dict), w, "must be an object {title, url}"):
            _text(p, s, "title", w)
            p.need(is_http(s.get("url")), w, "url must be an http(s) URL")


def _history(p: Problems, rec: dict, where: str, statuses: tuple[str, ...]) -> None:
    hist = rec.get("history")
    if not p.need(isinstance(hist, list) and hist, where, "'history' must be a non-empty list"):
        return
    last = ""
    for i, e in enumerate(hist):
        w = f"{where} history[{i}]"
        if not p.need(isinstance(e, dict), w, "must be an object"):
            return
        p.need(isinstance(e.get("date"), str) and DATE.match(e["date"]) is not None, w, "'date' must be YYYY-MM-DD")
        p.need(e.get("status") in statuses, w, f"'status' must be one of {', '.join(statuses)}")
        _text(p, e, "note", w, required=False)
        p.need(e.get("date", "") >= last, w, "dates must not go backwards")
        p.need(e.get("previous") is None or isinstance(e["previous"], dict), w, "'previous' must be an object")
        last = e.get("date", last)
    p.need(hist[-1].get("status") == rec.get("status"), where, "the last history entry must carry the current status")


def validate_papers(p: Problems, papers: Any) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not p.need(isinstance(papers, list), "papers.json", "must be a list"):
        return out
    for i, r in enumerate(papers):
        where = f"papers.json[{i}]"
        if not p.need(isinstance(r, dict), where, "must be an object"):
            continue
        pid = r.get("id")
        if not p.need(isinstance(pid, str) and PAPER_ID.match(pid) is not None, where,
                      "'id' must be arxiv:NNNN.NNNNN, doi:10.x/..., or title:lower-case-slug"):
            continue
        where = f"papers.json {pid}"
        p.need(pid not in out, where, "duplicate id (deduplicate by canonical identifier)")
        for key in ("title", "source", "category", "verification"):
            _text(p, r, key, where)
        p.need(r.get("authors") is None or (isinstance(r["authors"], str) and r["authors"].strip() != ""),
               where, "'authors' must be a non-empty string or null")
        p.need(isinstance(r.get("published"), str) and PUBLISHED.match(r["published"]) is not None,
               where, "'published' must be YYYY, YYYY-MM or YYYY-MM-DD")
        p.need(r.get("url") is None or is_http(r["url"]), where, "'url' must be an http(s) URL or null")
        rel = r.get("relevance")
        p.need(isinstance(rel, (int, float)) and not isinstance(rel, bool) and 0 <= rel <= 1,
               where, "'relevance' must be a number in [0, 1]")
        p.need(isinstance(r.get("deep"), bool), where, "'deep' must be true or false")
        p.need(isinstance(r.get("first_seen"), str) and DATE.match(r["first_seen"]) is not None,
               where, "'first_seen' must be YYYY-MM-DD")
        seen = r.get("seen", [r.get("first_seen")])
        if p.need(isinstance(seen, list) and seen and all(isinstance(d, str) and DATE.match(d) for d in seen),
                  where, "'seen' must be a non-empty list of dates"):
            p.need(min(seen) == r.get("first_seen"), where, "'first_seen' must be the earliest date in 'seen'")
            p.need(seen == sorted(set(seen)), where, "'seen' must be sorted and without repeats")
        for key in ("thesis", "foundations", "mechanism", "evidence", "limitations"):
            _text(p, r, key, where, required=False)
        for key, keys in (("implications", IMPLICATION_KEYS), ("assessment", ASSESSMENT_KEYS)):
            v = r.get(key)
            if isinstance(v, dict):
                p.need(set(v) <= set(keys) and all(isinstance(s, str) and s.strip() for s in v.values()),
                       where, f"'{key}' as an object takes string fields {', '.join(keys)}")
            else:
                _text(p, r, key, where, required=False)
        if r.get("deep"):
            p.need(bool(r.get("thesis")) and bool(r.get("mechanism")), where,
                   "a deep analysis needs at least 'thesis' and 'mechanism'")
        hist = r.get("history", [])
        if p.need(isinstance(hist, list), where, "'history' must be a list"):
            for j, e in enumerate(hist):
                w = f"{where} history[{j}]"
                if p.need(isinstance(e, dict), w, "must be an object"):
                    p.need(isinstance(e.get("date"), str) and DATE.match(e["date"]) is not None, w, "'date' must be YYYY-MM-DD")
                    _text(p, e, "note", w)
                    p.need(e.get("previous") is None or isinstance(e["previous"], dict), w, "'previous' must be an object")
        out[pid] = r
    return out


def validate_hypotheses(p: Problems, items: Any, papers: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not p.need(isinstance(items, list), "hypotheses.json", "must be a list"):
        return out
    for i, r in enumerate(items):
        where = f"hypotheses.json[{i}]"
        if not p.need(isinstance(r, dict) and isinstance(r.get("id"), str) and HID.match(r["id"]) is not None,
                      where, "must be an object with 'id' H-NNN"):
            continue
        where = f"hypotheses.json {r['id']}"
        p.need(r["id"] not in out, where, "duplicate id")
        _text(p, r, "hypothesis", where)
        _text(p, r, "motivation", where)
        p.need(r.get("status") in H_STATUS, where, f"'status' must be one of {', '.join(H_STATUS)}")
        _strings(p, r, "evidence", where)
        for ev in r.get("evidence", []) if isinstance(r.get("evidence"), list) else []:
            m = PAPER_REF.match(ev) if isinstance(ev, str) else None
            if m:
                p.need(m.group(1) in papers, where, f"evidence cites unknown paper {m.group(1)}")
        p.need(r.get("next_experiment") is None or (isinstance(r["next_experiment"], str) and EID.match(r["next_experiment"]) is not None),
               where, "'next_experiment' must be E-NNN or null")
        _history(p, r, where, H_STATUS)
        out[r["id"]] = r
    return out


def validate_experiments(p: Problems, items: Any, hyps: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not p.need(isinstance(items, list), "experiments.json", "must be a list"):
        return out
    for i, r in enumerate(items):
        where = f"experiments.json[{i}]"
        if not p.need(isinstance(r, dict) and isinstance(r.get("id"), str) and EID.match(r["id"]) is not None,
                      where, "must be an object with 'id' E-NNN"):
            continue
        where = f"experiments.json {r['id']}"
        p.need(r["id"] not in out, where, "duplicate id")
        p.need(r.get("hypothesis") in hyps, where, "'hypothesis' must name a known hypothesis")
        _text(p, r, "description", where)
        p.need(r.get("status") in E_STATUS, where, f"'status' must be one of {', '.join(E_STATUS)}")
        _text(p, r, "result", where, required=r.get("status") == "COMPLETED")
        _text(p, r, "artifact", where, required=False)
        _history(p, r, where, E_STATUS)
        out[r["id"]] = r
    for h in hyps.values():
        nxt = h.get("next_experiment")
        p.need(nxt is None or nxt in out, f"hypotheses.json {h['id']}", f"next_experiment {nxt} is not in experiments.json")
    return out


def validate_report(p: Problems, r: Any, name: str, papers: dict, hyps: dict, exps: dict) -> bool:
    where = f"reports/{name}"
    if not p.need(isinstance(r, dict), where, "must be an object"):
        return False
    ok = p.need(r.get("date") == name[:-5] and DATE.match(name[:-5]) is not None, where,
                "'date' must match the file name YYYY-MM-DD.json")
    p.need(isinstance(r.get("generated"), str) and STAMP.match(r["generated"]) is not None
           and r["generated"][:10] >= r.get("date", ""), where, "'generated' must be a UTC stamp YYYY-MM-DDTHH:MM:SSZ on or after 'date'")
    p.need(r.get("status") in RUN_STATUS, where, f"'status' must be one of {', '.join(RUN_STATUS)}")
    _text(p, r, "status_note", where, required=False)
    p.need(r.get("importance") in IMPORTANCE, where, f"'importance' must be one of {', '.join(IMPORTANCE)}")
    for key in ("direction", "summary"):
        _text(p, r, key, where)
    findings = r.get("findings")
    if p.need(isinstance(findings, list) and 1 <= len(findings) <= 5, where, "'findings' must list 1 to 5 findings"):
        fids = set()
        for i, f in enumerate(findings):
            w = f"{where} findings[{i}]"
            if not p.need(isinstance(f, dict) and isinstance(f.get("id"), str) and FID.match(f["id"]) is not None,
                          w, "must be an object with 'id' F-N"):
                continue
            p.need(f["id"] not in fids, w, "duplicate finding id")
            fids.add(f["id"])
            for key in ("title", "category", "discovery", "relevance", "action"):
                _text(p, f, key, w)
            p.need(f.get("strength") in EVIDENCE, w, f"'strength' must be one of {', '.join(EVIDENCE)}")
            p.need(f.get("paper") is None or f["paper"] in papers, w, "'paper' must be a known paper id or null")
    for key in ("radar", "deep"):
        v = r.get(key, [])
        if p.need(isinstance(v, list) and len(set(v)) == len(v), where, f"'{key}' must be a list of distinct paper ids"):
            for pid in v:
                p.need(pid in papers, where, f"'{key}' names unknown paper {pid}")
    for pid in r.get("deep", []) if isinstance(r.get("deep"), list) else []:
        p.need(papers.get(pid, {}).get("deep") is True, where, f"{pid} is listed under 'deep' but its record has deep=false")
    _text(p, r, "deep_note", where, required=False)
    ins = r.get("insights")
    if p.need(isinstance(ins, list), where, "'insights' must be a list"):
        areas = [x.get("area") for x in ins if isinstance(x, dict)]
        p.need(sorted(areas) == sorted(AREAS), where, f"'insights' must cover each area once: {', '.join(AREAS)}")
        for i, x in enumerate(ins):
            w = f"{where} insights[{i}]"
            if p.need(isinstance(x, dict), w, "must be an object"):
                p.need(x.get("label") in EVIDENCE, w, f"'label' must be one of {', '.join(EVIDENCE)}")
                _text(p, x, "note", w)
    for key, known in (("hypotheses", hyps), ("experiments", exps)):
        v = r.get(key, [])
        if p.need(isinstance(v, list), where, f"'{key}' must be a list"):
            for x in v:
                p.need(x in known, where, f"'{key}' names unknown id {x}")
    for key in ("recommendations", "knowledge_updates", "open_questions", "priorities", "queries", "editorial_notes"):
        _strings(p, r, key, where)
    for key in ("sources", "links"):
        _links(p, r, key, where)
    return ok


def validate_weekly(p: Problems, w: Any, name: str, report_dates: set[str]) -> bool:
    where = f"weekly/{name}"
    if not p.need(isinstance(w, dict), where, "must be an object"):
        return False
    ok = p.need(w.get("week") == name[:-5] and WEEK.match(name[:-5]) is not None, where,
                "'week' must match the file name YYYY-Www.json")
    p.need(isinstance(w.get("generated"), str) and STAMP.match(w["generated"]) is not None, where, "'generated' must be a UTC stamp")
    for key in ("from", "to"):
        p.need(isinstance(w.get(key), str) and DATE.match(w[key]) is not None, where, f"'{key}' must be YYYY-MM-DD")
    _text(p, w, "summary", where)
    for key in ("trends", "mechanisms", "contradictions", "evidence_changes", "opportunities", "gaps", "experiment_priorities"):
        _strings(p, w, key, where)
    v = w.get("reports", [])
    if p.need(isinstance(v, list), where, "'reports' must be a list of report dates"):
        for d in v:
            p.need(d in report_dates, where, f"no report for {d}")
    return ok


def validate_state(p: Problems, s: Any) -> None:
    where = "research_state.json"
    if not p.need(isinstance(s, dict), where, "must be an object"):
        return
    pub = s.get("publishing")
    if p.need(isinstance(pub, dict), where, "'publishing' must be an object"):
        p.need(isinstance(pub.get("repository"), str) and re.match(r"^[\w.-]+/[\w.-]+$", pub["repository"]) is not None,
               where, "publishing.repository must be owner/name")
        _text(p, pub, "branch", where)
        _text(p, pub, "folder", where)
        p.need(is_http(pub.get("site")), where, "publishing.site must be an http(s) URL")
    deps = s.get("deployments", [])
    if p.need(isinstance(deps, list), where, "'deployments' must be a list"):
        for i, d in enumerate(deps):
            w = f"{where} deployments[{i}]"
            if p.need(isinstance(d, dict), w, "must be an object"):
                p.need(isinstance(d.get("commit"), str) and SHA.match(d["commit"]) is not None, w, "'commit' must be a hex sha")
                p.need(isinstance(d.get("verified_at"), str) and STAMP.match(d["verified_at"]) is not None, w, "'verified_at' must be a UTC stamp")
                p.need(d.get("conclusion") in ("success", "failure"), w, "'conclusion' must be success or failure")
                p.need(d.get("run") is None or is_http(d["run"]), w, "'run' must be an http(s) URL or null")


# --------------------------------------------------------------------------
# history guard: records are appended to, never silently rewritten


def _git_head(rel: str) -> Any:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:./{rel}"],
                             capture_output=True, text=True, check=False)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


def _guard_records(p: Problems, kind: str, old: Any, new: dict[str, dict]) -> None:
    if not isinstance(old, list):
        return
    for o in old:
        if not isinstance(o, dict) or "id" not in o:
            continue
        where = f"{kind} {o['id']}"
        n = new.get(o["id"])
        if not p.need(n is not None, where, "was removed; records are never deleted (mark the status instead)"):
            continue
        oh, nh = o.get("history", []), n.get("history", [])
        if not p.need(nh[:len(oh)] == oh, where, "history was rewritten; only append entries"):
            continue
        added = nh[len(oh):]
        recorded: dict[str, Any] = {}
        for e in added:
            if isinstance(e, dict) and isinstance(e.get("previous"), dict):
                recorded.update(e["previous"])
        for key in TRACKED[kind]:
            if o.get(key) != n.get(key):
                p.need(key in recorded and recorded[key] == o.get(key), where,
                       f"'{key}' changed; append a history entry whose 'previous' holds the old value")
        if kind == "paper":
            p.need(set(o.get("seen", [o.get("first_seen")])) <= set(n.get("seen", [n.get("first_seen")])),
                   where, "dates were removed from 'seen'")
            if o.get("first_seen") != n.get("first_seen"):
                p.need(recorded.get("first_seen") == o.get("first_seen"), where,
                       "'first_seen' changed; append a history entry whose 'previous' holds the old value")


def guard_history(p: Problems, papers: dict, hyps: dict, exps: dict, reports: dict, weeklies: dict, state: dict) -> None:
    _guard_records(p, "paper", _git_head("data/papers.json"), papers)
    _guard_records(p, "hypothesis", _git_head("data/hypotheses.json"), hyps)
    _guard_records(p, "experiment", _git_head("data/experiments.json"), exps)
    old_state = _git_head("data/research_state.json") or {}
    for run in old_state.get("runs", []) if isinstance(old_state, dict) else []:
        d = run.get("date") if isinstance(run, dict) else None
        p.need(d in reports, f"run {d}", "its report record is missing; history is never deleted")
    for d, r in reports.items():
        old = _git_head(f"data/reports/{d}.json")
        if old is None:
            continue
        a = {k: v for k, v in old.items() if k != "editorial_notes"}
        b = {k: v for k, v in r.items() if k != "editorial_notes"}
        p.need(a == b, f"reports/{d}.json", "a committed report is immutable; add an 'editorial_notes' entry instead")
        on, nn = old.get("editorial_notes", []), r.get("editorial_notes", [])
        p.need(nn[:len(on)] == on, f"reports/{d}.json", "editorial notes were rewritten; only append")
    for wk, w in weeklies.items():
        old = _git_head(f"data/weekly/{wk}.json")
        p.need(old is None or old == w, f"weekly/{wk}.json", "a committed weekly synthesis is immutable")
    od = old_state.get("deployments", []) if isinstance(old_state, dict) else []
    p.need(state.get("deployments", [])[:len(od)] == od, "research_state.json", "deployments were rewritten; only append")


# --------------------------------------------------------------------------
# bilingual text
#
# Every visible string goes through L(): when data/i18n_zh.json (content) or
# UI_ZH (interface) has a Chinese rendering, the page carries both, as
# <span class="en"> and <span class="zh">; the language switch (EN / 中文 /
# both) only toggles which spans are shown.  Missing translations fall back
# to English; `build.py --missing-zh` lists them.

UI_ZH = {
    # masthead, footer
    "Findings": "发现", "Radar": "雷达", "Insights": "洞察", "Hypotheses": "假设", "Timeline": "时间线",
    "Latest report": "最新报告", "Autonomous Cognitive Systems Research": "自主认知系统研究",
    "Autonomous research output. Evidence labels describe the literature support for a finding, not the state of the World0 implementation.":
        "自主研究产出。证据标签描述文献对一项发现的支持程度，并不代表 World0 已实现相应能力。",
    "records up to": "记录截至",
    # section A
    "Current report": "当前报告", "Last research update": "最近研究更新", "Last verified deployment": "最近已验证部署",
    "Live Pages status": "实时 Pages 状态", "Papers analysed": "已分析论文", "Active hypotheses": "活跃假设",
    "Pending experiments": "待做实验", "Weekly synthesis": "周度综述", "deep": "深度",
    "not yet verified": "尚未验证", "needs JavaScript and api.github.com": "需要 JavaScript 与 api.github.com",
    "no report yet": "暂无报告", "none yet": "暂无",
    # sections B–F
    "Latest Key Findings": "最新关键发现", "Research Radar": "研究雷达", "Architecture Insights": "架构洞察",
    "Research Hypotheses": "研究假设", "Research Timeline": "研究时间线", "report": "报告",
    "No research session recorded yet.": "尚无研究记录。", "No insights yet.": "暂无洞察。",
    "Core discovery": "核心发现", "Relevance to World0": "与 World0 的关联", "Recommended action": "建议行动",
    "Detailed analysis →": "详细分析 →", "In report": "见报告", "Original source ↗": "原始来源 ↗",
    "papers": "论文", "relevance": "相关度", "relevance is the research agent's judgement": "相关度为研究代理的判断",
    "Search": "搜索", "Category": "类别", "Scope": "范围", "Sort": "排序", "All categories": "全部类别",
    "All records": "全部记录", "High relevance": "高相关", "Newest first": "最新优先", "Relevance": "相关度",
    "Publication date": "发表日期", "title, author, id": "标题、作者、编号",
    "Paper": "论文", "Authors": "作者", "Published": "发表", "First seen": "首次记录", "Source": "来源",
    "not recovered": "未获取", "not located": "未找到",
    "Labels classify the evidence behind each insight, not whether World0 implements the capability.":
        "标签描述每条洞察背后的证据强度，而非 World0 是否已实现该能力。",
    "unchanged": "未变", "was": "之前为",
    "active": "活跃", "resolved": "已结论", "No active hypothesis.": "暂无活跃假设。",
    "Resolved hypotheses": "已结论的假设", "Motivation": "动机", "Supporting evidence": "支持证据",
    "Next experiment": "下一个实验", "none": "无", "Experiments and status history": "实验与状态历史",
    "No experiment registered.": "暂无登记的实验。", "Result.": "结果。", "Artifact:": "产物：",
    "None recorded.": "暂无记录。",
    "Date": "日期", "Direction": "方向", "Importance": "重要性", "All months": "全部月份",
    "All directions": "全部方向", "Any importance": "任意重要性", "sessions": "次研究",
    "new": "新增", "hypotheses": "假设", "experiments": "实验",
    "Weekly syntheses": "周度综述", "No weekly synthesis yet.": "暂无周度综述。",
    # report page
    "DAILY REPORT": "每日报告", "Research Report": "研究报告", "Report date": "报告日期", "Generated": "生成时间",
    "Research status": "研究状态", "first report": "第一份报告", "All reports": "全部报告",
    "Executive Summary": "执行摘要", "Key Discoveries": "关键发现", "Deep Paper Analysis": "论文深度分析",
    "World0 Architecture Implications": "World0 架构启示", "New Research Hypotheses": "新研究假设",
    "Engineering Recommendations": "工程建议", "Research Knowledge Updates": "研究知识更新",
    "Open Questions": "开放问题", "Next Research Priorities": "后续研究优先事项", "Sources": "来源",
    "Related records": "相关记录", "Search queries": "搜索查询", "Editorial notes": "编辑说明", "Experiments": "实验",
    "Labels classify the evidence, not World0's implementation status.": "标签描述证据强度，而非 World0 的实现状态。",
    "No papers recorded in this session.": "本次研究未记录论文。",
    "No full-text deep analysis in this session.": "本次研究没有全文深度分析。",
    "No hypothesis registered or updated.": "没有登记或更新的假设。",
    "Verification:": "验证：", "paper page": "论文页", "Authors not recovered": "作者未获取", "None.": "无。",
    # paper page
    "Deep analysis.": "深度分析。",
    "Radar entry: recorded from search results, not a full-text deep analysis. Sections the source was not seen to state are marked as not recorded.":
        "雷达条目：根据搜索结果记录，并非全文深度分析。来源中未见到的内容标为“未记录”。",
    "Research Identity": "研究身份", "Analysis": "分析", "Connections": "关联", "Title": "标题",
    "Identifier": "标识符", "Original URL": "原始链接", "Verification": "验证方式", "Seen in reports": "出现于报告",
    "agent judgement": "研究代理判断", "no canonical identifier located": "未找到规范标识符",
    "Core Thesis": "核心论点", "Theoretical Foundations": "理论基础", "Proposed Mechanism": "提出的机制",
    "Experimental Evidence": "实验证据", "Limitations": "局限", "World0 Implications": "对 World0 的启示",
    "Researcher's Assessment": "研究者评估", "AUTHOR CLAIM": "作者主张", "REPORTED RESULTS": "报告的结果",
    "AGENT INTERPRETATION": "研究代理解读", "Related World0 components": "相关 World0 组件",
    "Potential improvements": "潜在改进", "Implementation complexity": "实现复杂度", "Expected benefits": "预期收益",
    "Integration risks": "集成风险", "Author claims": "作者主张", "Experimental observations": "实验观察",
    "Agent interpretation": "研究代理解读", "Speculative hypotheses": "推测性假设", "Not recorded.": "未记录。",
    "Not recovered. No result is reported here that the source was not seen to state.":
        "未获取。此处不记录任何未在来源中见到的结果。",
    "Finding": "发现", "Hypothesis": "假设", "No finding or hypothesis cites this paper yet.": "暂无发现或假设引用此论文。",
    "Record history": "记录历史",
    # weekly page
    "WEEKLY SYNTHESIS": "周度综述", "Week": "第", "Cumulative Understanding": "累积认识",
    "Major research trends": "主要研究趋势", "Recurring mechanisms": "反复出现的机制",
    "Contradictory findings": "相互矛盾的发现", "Changes in evidence strength": "证据强度变化",
    "High-value architectural opportunities": "高价值架构机会", "Research gaps": "研究空白",
    "Experiment priorities": "实验优先级", "Reports Covered": "涵盖的报告", "generated": "生成于",
}

JUDGEMENT = "relevance is the research agent's judgement"
ZH: dict[str, str] = {}
USED: set[str] = set()


def tr(s: str) -> str | None:
    USED.add(s)
    zh = ZH.get(s) or UI_ZH.get(s)
    return zh if zh and zh != s else None


def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def pair(en: str, zh: str | None) -> str:
    if not zh:
        return esc(en)
    return f'<span class="en">{esc(en)}</span><span class="zh" lang="zh-CN">{esc(zh)}</span>'


def L(s: Any) -> str:
    """A visible string, with its Chinese rendering when one is known."""
    if s is None or s == "":
        return ""
    return pair(str(s), tr(str(s)))


def plain(s: str) -> str:
    """Text where markup is impossible (<option>, placeholder): 'English / 中文'."""
    zh = tr(s)
    return esc(f"{s} / {zh}" if zh else s)


def slug(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", pid.replace(":", "-", 1))


def ext(url: Any, text: str, label: str | None = None) -> str:
    """An outbound link, or plain text when the URL is not http(s)."""
    inner = label if label is not None else esc(text)
    if not is_http(url):
        return inner
    return f'<a href="{esc(url)}" rel="noopener noreferrer external">{inner}</a>'


def paras(text: Any, cls: str = "") -> str:
    if not text:
        return ""
    c = f' class="{cls}"' if cls else ""
    en = [t.strip() for t in str(text).split("\n\n") if t.strip()]
    zh_all = tr(str(text))
    zh = [t.strip() for t in zh_all.split("\n\n") if t.strip()] if zh_all else []
    if zh and len(zh) == len(en):
        return "".join(f"<p{c}>{pair(a, b)}</p>" for a, b in zip(en, zh))
    out = "".join(f"<p{c}>{L(t)}</p>" for t in en)
    if zh:
        zc = f' class="zh{" " + cls if cls else ""}" lang="zh-CN"'
        out += "".join(f"<p{zc}>{esc(t)}</p>" for t in zh)
    return out


def chip(value: str, family: str) -> str:
    return f'<span class="chip {family}-{esc(value)}"><span aria-hidden="true">{SYMBOL.get(value, "◇")}</span> {esc(value.replace("_", " "))}</span>'


def tag(text: str) -> str:
    return f'<span class="tag">{L(text)}</span>'


def stamp(s: str | None) -> str:
    if not s:
        return '<span class="dim">—</span>'
    shown = s[:16].replace("T", " ") + " UTC" if len(s) >= 16 else s
    return f'<time datetime="{esc(s)}">{esc(shown)}</time>'


def rel_bar(v: float) -> str:
    pct = round(max(0.0, min(1.0, float(v))) * 100)
    star = ' <span class="star" title="High architectural relevance">★</span>' if v >= HIGH_RELEVANCE else ""
    return (f'<span class="rel"><span class="rel__bar" aria-hidden="true"><i style="width:{pct}%"></i></span>'
            f'<b>{v:.2f}</b>{star}</span>')


def ul(items: list[str], cls: str = "list") -> str:
    if not items:
        return f'<p class="dim">{L("None recorded.")}</p>'
    return f'<ul class="{cls}">' + "".join(f"<li>{L(s)}</li>" for s in items) + "</ul>"


def sec(idx: str, title: str, anchor: str, meta: str = "") -> str:
    m = f'<span class="sec__meta">{meta}</span>' if meta else ""
    return (f'<h2 class="sec" id="{anchor}"><span class="sec__idx">{esc(idx)}</span>'
            f'<span class="sec__title">{L(title)}</span><span class="sec__rule" aria-hidden="true"></span>{m}</h2>')


def ref_html(text: str, papers: dict, up: str) -> str:
    """Evidence text: a leading paper id becomes a link to its analysis page."""
    m = PAPER_REF.match(text)
    if not (m and m.group(1) in papers):
        return L(text)
    pid = m.group(1)
    zh = tr(text)
    zm = PAPER_REF.match(zh) if zh else None
    rest = pair(m.group(2), zm.group(2) if zm and zm.group(1) == pid else zh) if m.group(2) or zh else ""
    return f'<a href="{up}papers/{esc(slug(pid))}.html" class="pid">{esc(pid)}</a>{rest}'


FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600'
         '&amp;family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap">')
# Applies the saved language before first paint (the switch itself is in app.js).
LANG_BOOT = ('<script>try{var l=localStorage.getItem("w0-lang");'
             'if(l==="en"||l==="zh"||l==="both")document.documentElement.setAttribute("data-lang",l)}catch(e){}</script>')


def page(*, title: str, description: str, body: str, up: str, ctx: dict, attrs: str = "") -> str:
    latest = ctx["latest"]
    nav = (f'<nav class="topnav" aria-label="Portal">'
           f'<a href="{up}index.html#findings">{L("Findings")}</a><a href="{up}index.html#radar">{L("Radar")}</a>'
           f'<a href="{up}index.html#insights">{L("Insights")}</a><a href="{up}index.html#hypotheses">{L("Hypotheses")}</a>'
           f'<a href="{up}index.html#timeline">{L("Timeline")}</a>'
           + (f'<a href="{up}reports/{esc(latest)}.html" class="topnav__cta">{L("Latest report")}</a>' if latest else "")
           + "</nav>")
    langsw = ('<div class="langsw" role="group" aria-label="Language / 语言" data-enhance hidden>'
              '<button type="button" data-l="en">EN</button><button type="button" data-l="zh" lang="zh-CN">中文</button>'
              '<button type="button" data-l="both">EN+中文</button></div>')
    repo = ctx["state"]["publishing"]["repository"]
    return f"""<!doctype html>
<html lang="en" data-lang="both">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="description" content="{esc(description)}">
<title>{esc(title)}</title>
{LANG_BOOT}
{FONTS}
<link rel="stylesheet" href="{up}assets/css/style.css">
<script src="{up}assets/js/app.js" defer></script>
</head>
<body{attrs}>
<a class="skip" href="#main">Skip to content</a>
<header class="masthead">
<div class="wrap masthead__row">
<a class="brand" href="{up}index.html"><span class="brand__mark" aria-hidden="true">◆</span><span class="brand__name">WORLD0</span><span class="brand__sub">RESEARCH INTELLIGENCE</span></a>
<div class="masthead__tools">{nav}{langsw}</div>
</div>
</header>
<main id="main" class="wrap">
{body}
</main>
<footer class="foot">
<div class="wrap foot__row">
<p><span aria-hidden="true">◇</span> {L("Autonomous research output. Evidence labels describe the literature support for a finding, not the state of the World0 implementation.")}</p>
<p class="dim">Built from <code>data/*.json</code> by <code>build.py</code> · {L("records up to")} {stamp(ctx["updated"])} · <a href="https://github.com/{esc(repo)}" rel="noopener noreferrer external">github.com/{esc(repo)}</a></p>
</div>
</footer>
</body>
</html>
"""


# --------------------------------------------------------------------------
# pages


def finding_card(f: dict, date: str, papers: dict, up: str, in_report: bool) -> str:
    paper = papers.get(f.get("paper") or "")
    links = []
    if paper:
        links.append(f'<a href="{up}papers/{esc(slug(paper["id"]))}.html">{L("Detailed analysis →")}</a>')
    if not in_report:
        links.append(f'<a href="{up}reports/{esc(date)}.html#{esc(f["id"])}">{L("In report")} {esc(date)}</a>')
    elif paper:
        links.append(ext(paper.get("url"), "", L("Original source ↗")))
    anchor = f' id="{esc(f["id"])}"' if in_report else ""
    return f"""<article class="card finding"{anchor}>
<header class="card__head">{tag(f["category"])}{chip(f["strength"], "ev")}</header>
<h3 class="card__title"><span class="fid">{esc(f["id"])}</span> {L(f["title"])}</h3>
<dl class="kv">
<dt>{L("Core discovery")}</dt><dd>{L(f["discovery"])}</dd>
<dt>{L("Relevance to World0")}</dt><dd>{L(f["relevance"])}</dd>
<dt>{L("Recommended action")}</dt><dd>{L(f["action"])}</dd>
</dl>
<footer class="card__foot">{" ".join(links)}</footer>
</article>"""


RADAR_COLS = ("Paper", "Authors", "Published", "Category", "Relevance", "First seen", "Source")


def _cell(col: str, inner: str, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    zh = tr(col) or col
    return f'<td data-label="{esc(col)}" data-label-zh="{esc(zh)}"{c}>{inner}</td>'


def radar_table(ids: list[str], papers: dict, up: str, latest_ids: set[str], latest: str, controls: bool) -> str:
    cats = sorted({papers[i]["category"] for i in ids})
    rows = []
    for pid in ids:
        r = papers[pid]
        classes = []
        if r["relevance"] >= HIGH_RELEVANCE:
            classes.append("is-high")
        is_new = r["first_seen"] == latest
        if is_new:
            classes.append("is-new")
        text = " ".join(str(x) for x in (r["title"], r.get("authors") or "", pid, r["category"], tr(r["category"]) or "")).lower()
        authors = L(r["authors"]) if r.get("authors") else f'<span class="dim">{L("not recovered")}</span>'
        source = (ext(r.get("url"), f'{r["source"]} ↗') if r.get("url")
                  else f'<span class="dim">{L("not located")}</span>')
        new = ' <span class="badge">NEW</span>' if is_new else ""
        deep = ' <span class="badge badge--deep">DEEP</span>' if r.get("deep") else ""
        rows.append(
            f'<tr class="{" ".join(classes)}" data-cat="{esc(r["category"])}" data-rel="{r["relevance"]:.2f}" '
            f'data-pub="{esc(r["published"])}" data-seen="{esc(r["first_seen"])}" '
            f'data-latest="{1 if pid in latest_ids else 0}" data-text="{esc(text)}">'
            + _cell("Paper", f'<a href="{up}papers/{esc(slug(pid))}.html">{esc(r["title"])}</a>'
                             f'<span class="pid">{esc(pid)}</span>{new}{deep}', "radar__paper")
            + _cell("Authors", authors)
            + _cell("Published", esc(r["published"]), "mono")
            + _cell("Category", tag(r["category"]))
            + _cell("Relevance", rel_bar(r["relevance"]))
            + _cell("First seen", esc(r["first_seen"]), "mono")
            + _cell("Source", source) + "</tr>")
    ctl = ""
    if controls:
        opts = "".join(f'<option value="{esc(c)}">{plain(c)}</option>' for c in cats)
        ctl = f"""<div class="controls" data-enhance hidden>
<label><span>{L("Search")}</span><input type="search" id="radar-q" placeholder="{plain("title, author, id")}" autocomplete="off"></label>
<label><span>{L("Category")}</span><select id="radar-cat"><option value="">{plain("All categories")}</option>{opts}</select></label>
<label><span>{L("Scope")}</span><select id="radar-scope"><option value="">{plain("All records")}</option><option value="latest">{plain("Latest report")}</option><option value="high">{plain("High relevance")} ≥ {HIGH_RELEVANCE:.2f}</option></select></label>
<label><span>{L("Sort")}</span><select id="radar-sort"><option value="seen">{plain("Newest first")}</option><option value="rel">{plain("Relevance")}</option><option value="pub">{plain("Publication date")}</option></select></label>
<output id="radar-count" class="controls__count" aria-live="polite"></output>
</div>"""
    tid = ' id="radar-table"' if controls else ""
    head = "".join(f'<th scope="col">{L(c)}</th>' for c in RADAR_COLS)
    return f"""{ctl}
<div class="table-wrap">
<table class="radar"{tid}>
<thead><tr>{head}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</div>"""


def experiment_line(e: dict) -> str:
    result = f'<p class="exp__result"><b>{L("Result.")}</b> {L(e["result"])}</p>' if e.get("result") else ""
    art = f'<p class="dim exp__art">{L("Artifact:")} {L(e["artifact"])}</p>' if e.get("artifact") else ""
    return (f'<div class="exp" id="{esc(e["id"])}"><p class="exp__head"><span class="hid">{esc(e["id"])}</span> '
            f'{chip(e["status"], "es")}</p><p>{L(e["description"])}</p>{result}{art}</div>')


def hypothesis_card(h: dict, papers: dict, exps: dict, up: str, anchor: bool = True) -> str:
    ev = "".join(f"<li>{ref_html(x, papers, up)}</li>" for x in h.get("evidence", [])) or f'<li class="dim">{L("None recorded.")}</li>'
    related = [e for e in exps.values() if e["hypothesis"] == h["id"]]
    nxt = h.get("next_experiment")
    exp_html = "".join(experiment_line(e) for e in related) or f'<p class="dim">{L("No experiment registered.")}</p>'
    hist = "".join(
        f'<li><time datetime="{esc(e["date"])}">{esc(e["date"])}</time> {chip(e["status"], "hs")}'
        + (f' <span class="dim">{L(e["note"])}</span>' if e.get("note") else "") + "</li>"
        for e in h["history"])
    aid = f' id="{esc(h["id"])}"' if anchor else ""
    return f"""<article class="card hyp"{aid}>
<header class="card__head"><span class="hid">{esc(h["id"])}</span>{chip(h["status"], "hs")}</header>
<p class="hyp__text">{L(h["hypothesis"])}</p>
<dl class="kv">
<dt>{L("Motivation")}</dt><dd>{L(h["motivation"])}</dd>
<dt>{L("Supporting evidence")}</dt><dd><ul class="list list--tight">{ev}</ul></dd>
<dt>{L("Next experiment")}</dt><dd>{esc(nxt) if nxt else f'<span class="dim">{L("none")}</span>'}</dd>
</dl>
<details class="hyp__more"><summary>{L("Experiments and status history")}</summary>
{exp_html}
<ol class="hist">{hist}</ol>
</details>
</article>"""


def render_index(ctx: dict) -> str:
    papers, hyps, exps, reports = ctx["papers"], ctx["hyps"], ctx["exps"], ctx["reports"]
    state, latest = ctx["state"], ctx["latest"]
    up = ""
    rep = reports.get(latest)
    dates = sorted(reports)

    # A — header
    active = [h for h in hyps.values() if h["status"] in H_ACTIVE]
    open_exps = [e for e in exps.values() if e["status"] in E_OPEN]
    deep_n = sum(1 for p in papers.values() if p.get("deep"))
    dep = state.get("last_successful_deployment")
    dep_html = (f'{stamp(dep["verified_at"])} <span class="mono dim">{esc(dep["commit"][:7])}</span>'
                if dep else f'<span class="dim">{L("not yet verified")}</span>')
    cur = (f'<a href="reports/{esc(latest)}.html">{esc(latest)}</a> {chip(rep["status"], "st")}' if rep
           else f'<span class="dim">{L("no report yet")}</span>')
    weekly_latest = max(ctx["weeklies"]) if ctx["weeklies"] else None
    weekly_html = (f'<a href="weekly/{esc(weekly_latest)}.html">{esc(weekly_latest)}</a>' if weekly_latest
                   else f'<span class="dim">{L("none yet")}</span>')
    a = f"""<section class="hero" aria-labelledby="title">
<p class="eyebrow"><span aria-hidden="true">◆</span> WORLD0 / RESEARCH PORTAL</p>
<h1 id="title">WORLD0 RESEARCH INTELLIGENCE</h1>
<p class="subtitle">{L("Autonomous Cognitive Systems Research")}</p>
<dl class="meta">
<div class="meta__item"><dt>{L("Current report")}</dt><dd>{cur}</dd></div>
<div class="meta__item"><dt>{L("Last research update")}</dt><dd>{stamp(ctx["updated"])}</dd></div>
<div class="meta__item"><dt>{L("Last verified deployment")}</dt><dd>{dep_html}</dd></div>
<div class="meta__item"><dt>{L("Live Pages status")}</dt><dd data-live-deploy><span class="dim">{L("needs JavaScript and api.github.com")}</span></dd></div>
<div class="meta__item"><dt>{L("Papers analysed")}</dt><dd class="num">{len(papers)} <small>{deep_n} {L("deep")}</small></dd></div>
<div class="meta__item"><dt>{L("Active hypotheses")}</dt><dd class="num">{len(active)} <small>/ {len(hyps)}</small></dd></div>
<div class="meta__item"><dt>{L("Pending experiments")}</dt><dd class="num">{len(open_exps)} <small>/ {len(exps)}</small></dd></div>
<div class="meta__item"><dt>{L("Weekly synthesis")}</dt><dd>{weekly_html}</dd></div>
</dl>
</section>"""

    # B — findings
    if rep:
        cards = "\n".join(finding_card(f, latest, papers, up, in_report=False) for f in rep["findings"][:3])
        b = sec("B", "Latest Key Findings", "findings", f'{L("report")} {esc(latest)}') + f'\n<div class="grid grid--3">\n{cards}\n</div>'
    else:
        b = sec("B", "Latest Key Findings", "findings") + f'<p class="dim">{L("No research session recorded yet.")}</p>'

    # C — radar
    order = sorted(papers, key=lambda i: (papers[i]["first_seen"], papers[i]["relevance"], i), reverse=True)
    latest_ids = set(rep.get("radar", [])) if rep else set()
    c = (sec("C", "Research Radar", "radar",
             f'{len(papers)} {L("papers")} · <span class="star">★</span> {L("relevance")} ≥ {HIGH_RELEVANCE:.2f} · {L(JUDGEMENT)}')
         + radar_table(order, papers, up, latest_ids, latest or "", controls=True))

    # D — insights
    if rep:
        prev_date = dates[dates.index(latest) - 1] if dates.index(latest) > 0 else None
        prev = {x["area"]: x["label"] for x in reports[prev_date]["insights"]} if prev_date else {}
        cells = []
        by_area = {x["area"]: x for x in rep["insights"]}
        for area in AREAS:
            x = by_area[area]
            was = prev.get(area)
            trend = ""
            if was:
                moved = L("unchanged") if was == x["label"] else f'{L("was")} {esc(was)}'
                trend = f'<p class="trend"><span aria-hidden="true">↳</span> {esc(prev_date)}: {moved}</p>'
            cells.append(f'<article class="insight"><header class="insight__head"><h3>{L(area)}</h3>{chip(x["label"], "ev")}</header>'
                         f'<p>{L(x["note"])}</p>{trend}</article>')
        legend = " ".join(chip(lbl, "ev") for lbl in EVIDENCE)
        d = (sec("D", "Architecture Insights", "insights", f'{L("report")} {esc(latest)}')
             + f'\n<p class="note">{L("Labels classify the evidence behind each insight, not whether World0 implements the capability.")} {legend}</p>'
             + f'\n<div class="insights">{"".join(cells)}</div>')
    else:
        d = sec("D", "Architecture Insights", "insights") + f'<p class="dim">{L("No insights yet.")}</p>'

    # E — hypotheses
    act = sorted(active, key=lambda h: h["id"])
    done = sorted((h for h in hyps.values() if h["status"] not in H_ACTIVE), key=lambda h: h["id"])
    e = sec("E", "Research Hypotheses", "hypotheses", f'{len(act)} {L("active")} · {len(done)} {L("resolved")}')
    e += '\n<div class="grid grid--2">' + ("\n".join(hypothesis_card(h, papers, exps, up) for h in act)
                                          or f'<p class="dim">{L("No active hypothesis.")}</p>') + "</div>"
    if done:
        e += (f'\n<details class="resolved"><summary>{L("Resolved hypotheses")} ({len(done)})</summary>'
              f'<div class="grid grid--2">{"".join(hypothesis_card(h, papers, exps, up) for h in done)}</div></details>')

    # F — timeline
    runs = list(reversed(state["runs"]))
    months = sorted({r["date"][:7] for r in runs}, reverse=True)
    cats = sorted({c for r in runs for c in r["categories"]})
    dirs = sorted({r["direction"] for r in runs})
    items = []
    for r in runs:
        rr = reports[r["date"]]
        cat_tags = "".join(tag(c) for c in r["categories"])
        stats = " · ".join(f'{L(k)} {r[f]}' for k, f in (("papers", "papers_listed"), ("new", "papers_added"), ("deep", "deep_analyses"),
                                                         ("hypotheses", "hypotheses"), ("experiments", "experiments")))
        items.append(
            f'<li class="tl" data-month="{esc(r["date"][:7])}" data-cats="{esc("|" + "|".join(r["categories"]) + "|")}" '
            f'data-dir="{esc(r["direction"])}" data-imp="{esc(r["importance"])}">'
            f'<div class="tl__date"><a href="reports/{esc(r["date"])}.html">{esc(r["date"])}</a>{chip(r["status"], "st")}</div>'
            f'<div class="tl__body"><p class="tl__dir">{chip(r["importance"], "imp")} <span>{L(r["direction"])}</span></p>'
            f'{paras(rr["summary"])}'
            f'<p class="tl__stats mono">{stats}</p>'
            f'<p class="tags">{cat_tags}</p></div></li>')
    opt = lambda vals: "".join(f'<option value="{esc(v)}">{plain(v)}</option>' for v in vals)  # noqa: E731
    f_ctl = f"""<div class="controls" data-enhance hidden>
<label><span>{L("Date")}</span><select id="tl-month"><option value="">{plain("All months")}</option>{"".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in months)}</select></label>
<label><span>{L("Category")}</span><select id="tl-cat"><option value="">{plain("All categories")}</option>{opt(cats)}</select></label>
<label><span>{L("Direction")}</span><select id="tl-dir"><option value="">{plain("All directions")}</option>{opt(dirs)}</select></label>
<label><span>{L("Importance")}</span><select id="tl-imp"><option value="">{plain("Any importance")}</option>{"".join(f'<option value="{v}">{v}</option>' for v in IMPORTANCE)}</select></label>
<output id="tl-count" class="controls__count" aria-live="polite"></output>
</div>"""
    weekly = "".join(
        f'<li><a href="weekly/{esc(w)}.html">{esc(w)}</a> <span class="dim">{esc(ctx["weeklies"][w]["from"])} → {esc(ctx["weeklies"][w]["to"])}</span></li>'
        for w in sorted(ctx["weeklies"], reverse=True)) or f'<li class="dim">{L("No weekly synthesis yet.")}</li>'
    f = (sec("F", "Research Timeline", "timeline", f'{len(runs)} {L("sessions")}') + f_ctl
         + f'\n<ol class="timeline" id="timeline-list">{"".join(items)}</ol>'
         + f'\n<h3 class="sub">{L("Weekly syntheses")}</h3><ul class="list">{weekly}</ul>')

    body = "\n".join(f'<section class="section">{x}</section>' if i else x for i, x in enumerate((a, b, c, d, e, f)))
    pub = state["publishing"]
    attrs = f' data-repo="{esc(pub["repository"])}" data-branch="{esc(pub["branch"])}"'
    return page(title="World0 Research Intelligence", description="Autonomous cognitive systems research for the World0 project: findings, paper radar, architecture insights, hypotheses and history.",
                body=body, up=up, ctx=ctx, attrs=attrs)


def analysis_sections(r: dict, level: int = 3) -> str:
    out = []
    for key, title, prov in PAPER_SECTIONS:
        v = r.get(key)
        if isinstance(v, dict):
            inner = "".join(f"<dt>{L(KEY_LABEL[k])}</dt><dd>{L(v[k])}</dd>" for k in (IMPLICATION_KEYS + ASSESSMENT_KEYS) if k in v)
            content = f'<dl class="kv">{inner}</dl>'
        elif v:
            content = paras(v)
        else:
            missing = "Not recorded." if key != "evidence" else "Not recovered. No result is reported here that the source was not seen to state."
            content = f'<p class="dim">{L(missing)}</p>'
        out.append(f'<section class="analysis__part"><h{level}>{L(title)} <span class="prov">{L(prov)}</span></h{level}>{content}</section>')
    return "".join(out)


def identity(r: dict, ctx: dict, up: str) -> str:
    pid = r["id"]
    kind, _, value = pid.partition(":")
    ident = {"arxiv": esc(f"arXiv:{value}"), "doi": esc(f"DOI {value}"), "title": L("no canonical identifier located")}[kind]
    seen = ", ".join(f'<a href="{up}reports/{esc(d)}.html">{esc(d)}</a>' if d in ctx["reports"] else esc(d) for d in r.get("seen", [r["first_seen"]]))
    rows = [
        ("Title", esc(r["title"])),
        ("Authors", L(r["authors"]) if r.get("authors") else f'<span class="dim">{L("not recovered")}</span>'),
        ("Publication date", f'<span class="mono">{esc(r["published"])}</span>'),
        ("Source", L(r["source"])),
        ("Identifier", f'<span class="mono">{ident}</span>'),
        ("Original URL", ext(r.get("url"), r["url"]) if r.get("url") else f'<span class="dim">{L("not located")}</span>'),
        ("Category", tag(r["category"])),
        ("Relevance", rel_bar(r["relevance"]) + f' <span class="dim">{L("agent judgement")}</span>'),
        ("Verification", L(r["verification"])),
        ("Seen in reports", seen),
    ]
    return '<dl class="ident">' + "".join(f"<dt>{L(k)}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def render_paper(r: dict, ctx: dict) -> str:
    up = "../"
    pid = r["id"]
    cites = [h for h in ctx["hyps"].values() if any(PAPER_REF.match(x) and PAPER_REF.match(x).group(1) == pid for x in h.get("evidence", []))]
    finds = [(d, f) for d, rep in sorted(ctx["reports"].items()) for f in rep["findings"] if f.get("paper") == pid]
    rel = []
    for d, f in finds:
        rel.append(f'<li>{L("Finding")} <a href="{up}reports/{esc(d)}.html#{esc(f["id"])}">{esc(d)} {esc(f["id"])}</a> — {L(f["title"])}</li>')
    for h in cites:
        rel.append(f'<li>{L("Hypothesis")} <a href="{up}index.html#{esc(h["id"])}">{esc(h["id"])}</a> {chip(h["status"], "hs")} — {L(h["hypothesis"])}</li>')
    hist = "".join(f'<li><time>{esc(e["date"])}</time> {L(e["note"])}</li>' for e in r.get("history", []))
    depth = (f'<p class="note">{L("Deep analysis.")}</p>' if r.get("deep") else
             f'<p class="note">{L("Radar entry: recorded from search results, not a full-text deep analysis. Sections the source was not seen to state are marked as not recorded.")}</p>')
    none_rel = f'<li class="dim">{L("No finding or hypothesis cites this paper yet.")}</li>'
    body = f"""<p class="crumbs"><a href="{up}index.html#radar">{L("Research Radar")}</a> <span aria-hidden="true">/</span> <span class="mono">{esc(pid)}</span></p>
<article class="analysis">
<header class="page-head">
<p class="eyebrow">{tag(r["category"])} {'<span class="badge badge--deep">DEEP</span>' if r.get("deep") else ""}</p>
<h1>{esc(r["title"])}</h1>
{depth}
</header>
<section class="section">{sec("§", "Research Identity", "identity")}{identity(r, ctx, up)}</section>
<section class="section">{sec("§", "Analysis", "analysis")}<div class="analysis__grid">{analysis_sections(r)}</div></section>
<section class="section">{sec("§", "Connections", "connections")}<ul class="list">{"".join(rel) or none_rel}</ul>
{f'<h3 class="sub">{L("Record history")}</h3><ol class="hist hist--text">{hist}</ol>' if hist else ""}</section>
</article>"""
    return page(title=f"{r['title']} · World0 Research", description=f"World0 research analysis of {pid}: {r['title']}",
                body=body, up=up, ctx=ctx)


def render_report(r: dict, ctx: dict, prev: str | None, nxt: str | None) -> str:
    up = "../"
    papers, hyps, exps = ctx["papers"], ctx["hyps"], ctx["exps"]
    date = r["date"]
    run = next(x for x in ctx["state"]["runs"] if x["date"] == date)
    pager = ('<nav class="pager" aria-label="Reports">'
             + (f'<a href="{esc(prev)}.html" rel="prev">← {esc(prev)}</a>' if prev else f'<span class="dim">← {L("first report")}</span>')
             + f'<a href="{up}index.html#timeline">{L("All reports")}</a>'
             + (f'<a href="{esc(nxt)}.html" rel="next">{esc(nxt)} →</a>' if nxt else f'<span class="dim">{L("Latest report")} →</span>')
             + "</nav>")
    findings = "\n".join(finding_card(f, date, papers, up, in_report=True) for f in r["findings"])
    radar_ids = sorted(r.get("radar", []), key=lambda i: (-papers[i]["relevance"], i))
    radar = (radar_table(radar_ids, papers, up, set(radar_ids), date, controls=False) if radar_ids
             else f'<p class="dim">{L("No papers recorded in this session.")}</p>')
    deep = []
    for pid in r.get("deep", []):
        p = papers[pid]
        authors = L(p["authors"]) if p.get("authors") else L("Authors not recovered")
        deep.append(f'<article class="card analysis-card" id="{esc(slug(pid))}"><h3 class="card__title">{esc(p["title"])}</h3>'
                    f'<p class="dim">{authors} · <span class="mono">{esc(p["published"])}</span> · '
                    f'{ext(p.get("url"), pid)} · <a href="{up}papers/{esc(slug(pid))}.html">{L("paper page")}</a></p>'
                    f'<p class="note">{L("Verification:")} {L(p["verification"])}</p>'
                    f'<div class="analysis__grid">{analysis_sections(p, level=4)}</div></article>')
    deep_html = (paras(r.get("deep_note"), "note") + "".join(deep)) or f'<p class="dim">{L("No full-text deep analysis in this session.")}</p>'
    ins = "".join(f'<tr><th scope="row">{L(x["area"])}</th><td>{chip(x["label"], "ev")}</td><td>{L(x["note"])}</td></tr>'
                  for x in sorted(r["insights"], key=lambda x: AREAS.index(x["area"])))
    hyp_cards = "".join(hypothesis_card(hyps[h], papers, exps, up, anchor=False) for h in r.get("hypotheses", []))
    exp_html = "".join(experiment_line(exps[e]) for e in r.get("experiments", []))
    auto = pair(f'{run["papers_added"]} papers first recorded in this session; {run["papers_listed"]} on its radar; {run["deep_analyses"]} deep analyses.',
                f'本次首次记录论文 {run["papers_added"]} 篇；雷达列出 {run["papers_listed"]} 篇；深度分析 {run["deep_analyses"]} 篇。')
    updates = ul(r.get("knowledge_updates", [])).replace("</ul>", f"<li>{auto}</li></ul>") if r.get("knowledge_updates") else f'<ul class="list"><li>{auto}</li></ul>'
    sources = []
    for pid in r.get("radar", []):
        p = papers[pid]
        sources.append(f'<li>{ext(p.get("url"), p["title"])} <span class="pid">{esc(pid)}</span></li>')
    sources += [f'<li>{ext(s["url"], s["title"], L(s["title"]))}</li>' for s in r.get("sources", [])]
    links = "".join(f'<li>{ext(s["url"], s["title"], L(s["title"]))}</li>' for s in r.get("links", []))
    queries = ul(r.get("queries", []), "list mono-list") if r.get("queries") else ""
    notes = ul(r.get("editorial_notes", [])) if r.get("editorial_notes") else ""
    cats = "".join(tag(c) for c in run["categories"])
    none_li = f'<li class="dim">{L("None.")}</li>'
    body = f"""{pager}
<header class="page-head">
<p class="eyebrow"><span aria-hidden="true">◆</span> {L("DAILY REPORT")}</p>
<h1>{L("Research Report")} {esc(date)}</h1>
<dl class="meta meta--compact">
<div class="meta__item"><dt>{L("Report date")}</dt><dd class="mono">{esc(date)}</dd></div>
<div class="meta__item"><dt>{L("Generated")}</dt><dd>{stamp(r["generated"])}</dd></div>
<div class="meta__item"><dt>{L("Research status")}</dt><dd>{chip(r["status"], "st")}</dd></div>
<div class="meta__item"><dt>{L("Importance")}</dt><dd>{chip(r["importance"], "imp")}</dd></div>
<div class="meta__item meta__item--wide"><dt>{L("Direction")}</dt><dd>{L(r["direction"])}</dd></div>
</dl>
{paras(r.get("status_note"), "note")}
<p class="tags">{cats}</p>
</header>
<section class="section">{sec("1", "Executive Summary", "summary")}{paras(r["summary"])}</section>
<section class="section">{sec("2", "Key Discoveries", "discoveries")}<div class="grid grid--3">{findings}</div></section>
<section class="section">{sec("3", "Research Radar", "radar")}{radar}</section>
<section class="section">{sec("4", "Deep Paper Analysis", "deep")}{deep_html}</section>
<section class="section">{sec("5", "World0 Architecture Implications", "implications")}<p class="note">{L("Labels classify the evidence, not World0's implementation status.")}</p><div class="table-wrap"><table class="plain"><tbody>{ins}</tbody></table></div></section>
<section class="section">{sec("6", "New Research Hypotheses", "hypotheses")}{f'<div class="grid grid--2">{hyp_cards}</div>' if hyp_cards else f'<p class="dim">{L("No hypothesis registered or updated.")}</p>'}</section>
<section class="section">{sec("7", "Engineering Recommendations", "recommendations")}{ul(r.get("recommendations", []))}</section>
<section class="section">{sec("8", "Research Knowledge Updates", "updates")}{updates}{f'<h3 class="sub">{L("Experiments")}</h3>{exp_html}' if exp_html else ""}</section>
<section class="section">{sec("9", "Open Questions", "questions")}{ul(r.get("open_questions", []))}</section>
<section class="section">{sec("10", "Next Research Priorities", "priorities")}{ul(r.get("priorities", []))}</section>
<section class="section">{sec("§", "Sources", "sources")}<ul class="list">{"".join(sources) or none_li}</ul>
{f'<h3 class="sub">{L("Related records")}</h3><ul class="list">{links}</ul>' if links else ""}
{f'<h3 class="sub">{L("Search queries")}</h3>{queries}' if queries else ""}
{f'<h3 class="sub">{L("Editorial notes")}</h3>{notes}' if notes else ""}</section>
{pager}"""
    return page(title=f"Research Report {date} · World0", description=f"World0 research report for {date}: {r['direction']}",
                body=body, up=up, ctx=ctx)


def render_weekly(w: dict, ctx: dict) -> str:
    up = "../"
    reps = "".join(f'<li><a href="{up}reports/{esc(d)}.html">{esc(d)}</a> — {L(ctx["reports"][d]["direction"])}</li>' for d in w.get("reports", []))
    parts = [("Major research trends", "trends"), ("Recurring mechanisms", "mechanisms"), ("Contradictory findings", "contradictions"),
             ("Changes in evidence strength", "evidence_changes"), ("High-value architectural opportunities", "opportunities"),
             ("Research gaps", "gaps"), ("Experiment priorities", "experiment_priorities")]
    secs = "".join(f'<section class="section">{sec(str(i + 2), t, k)}{ul(w.get(k, []))}</section>' for i, (t, k) in enumerate(parts))
    none_li = f'<li class="dim">{L("None.")}</li>'
    body = f"""<p class="crumbs"><a href="{up}index.html#timeline">{L("Research Timeline")}</a> <span aria-hidden="true">/</span> {esc(w["week"])}</p>
<header class="page-head"><p class="eyebrow"><span aria-hidden="true">◆</span> {L("WEEKLY SYNTHESIS")}</p><h1>{esc(w["week"])}</h1>
<p class="dim"><span class="mono">{esc(w["from"])} → {esc(w["to"])}</span> · {L("generated")} {stamp(w["generated"])}</p></header>
<section class="section">{sec("1", "Cumulative Understanding", "summary")}{paras(w["summary"])}</section>
{secs}
<section class="section">{sec("§", "Reports Covered", "reports")}<ul class="list">{reps or none_li}</ul></section>"""
    return page(title=f"Weekly Synthesis {w['week']} · World0", description=f"World0 weekly research synthesis {w['week']}",
                body=body, up=up, ctx=ctx)


# --------------------------------------------------------------------------
# derived data


def derive_state(state: dict, reports: dict, papers: dict) -> dict:
    runs = []
    for d in sorted(reports):
        r = reports[d]
        cats = []
        for c in [f["category"] for f in r["findings"]] + [papers[p]["category"] for p in r.get("radar", [])]:
            if c not in cats:
                cats.append(c)
        runs.append({
            "date": d,
            "generated": r["generated"],
            "status": r["status"],
            "importance": r["importance"],
            "direction": r["direction"],
            "categories": cats,
            "papers_listed": len(r.get("radar", [])),
            "papers_added": sum(1 for p in papers.values() if p["first_seen"] == d),
            "deep_analyses": len(r.get("deep", [])),
            "hypotheses": len(r.get("hypotheses", [])),
            "experiments": len(r.get("experiments", [])),
            "report": f"reports/{d}.html",
        })
    ok = [d for d in state.get("deployments", []) if d.get("conclusion") == "success"]
    return {
        "schema": 1,
        "publishing": state["publishing"],
        "last_research_update": max((r["generated"] for r in reports.values()), default=None),
        "last_successful_deployment": ok[-1] if ok else None,
        "deployments": state.get("deployments", []),
        "runs": runs,
    }


def derive_latest(rep: dict | None) -> dict:
    if not rep:
        return {"date": None, "generated": None, "findings": [], "insights": []}
    return {
        "date": rep["date"],
        "generated": rep["generated"],
        "status": rep["status"],
        "report": f"reports/{rep['date']}.html",
        "findings": rep["findings"][:3],
        "insights": sorted(rep["insights"], key=lambda x: AREAS.index(x["area"])),
    }


class _Balance(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, t, attrs):
        if t not in self.VOID:
            self.stack.append(t)

    def handle_endtag(self, t):
        if not self.stack or self.stack[-1] != t:
            self.errors.append(f"unexpected </{t}> (open: {self.stack[-3:]})")
            if t in self.stack:
                while self.stack and self.stack.pop() != t:
                    pass
        else:
            self.stack.pop()


def check_html(name: str, text: str, p: Problems) -> None:
    b = _Balance()
    b.feed(text)
    b.close()
    p.need(not b.errors and not b.stack, name, f"unbalanced HTML: {(b.errors + [f'unclosed {b.stack}'])[0]}")


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


# --------------------------------------------------------------------------


def build(history: bool) -> tuple[dict[str, str], Problems]:
    p = Problems()
    raw = {}
    for name in ("papers", "hypotheses", "experiments", "research_state"):
        path = DATA / f"{name}.json"
        try:
            raw[name] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            p.append(f"{name}.json: missing")
            raw[name] = [] if name != "research_state" else {}
        except ValueError as exc:
            p.append(f"{name}.json: not valid JSON ({exc})")
            raw[name] = [] if name != "research_state" else {}
    zh_path = DATA / "i18n_zh.json"
    if zh_path.exists():
        try:
            zh = json.loads(zh_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            p.append(f"i18n_zh.json: not valid JSON ({exc})")
            zh = {}
        if p.need(isinstance(zh, dict) and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in zh.items()),
                  "i18n_zh.json", "must map English strings to non-empty Chinese strings"):
            ZH.clear()
            ZH.update(zh)
    papers = validate_papers(p, raw["papers"])
    hyps = validate_hypotheses(p, raw["hypotheses"], papers)
    exps = validate_experiments(p, raw["experiments"], hyps)
    validate_state(p, raw["research_state"])
    reports: dict[str, dict] = {}
    for path in sorted((DATA / "reports").glob("*.json")):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            p.append(f"reports/{path.name}: not valid JSON ({exc})")
            continue
        if validate_report(p, r, path.name, papers, hyps, exps):
            reports[r["date"]] = r
    weeklies: dict[str, dict] = {}
    for path in sorted((DATA / "weekly").glob("*.json")):
        try:
            w = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            p.append(f"weekly/{path.name}: not valid JSON ({exc})")
            continue
        if validate_weekly(p, w, path.name, set(reports)):
            weeklies[w["week"]] = w
    for pid, r in papers.items():
        for d in r.get("seen", [r["first_seen"]]):
            p.need(d in reports, f"papers.json {pid}", f"'seen' date {d} has no report")
    if p:
        return {}, p
    state = derive_state(raw["research_state"], reports, papers)
    if history:
        guard_history(p, papers, hyps, exps, reports, weeklies, state)
        if p:
            return {}, p
    latest = max(reports) if reports else None
    ctx = {"papers": papers, "hyps": hyps, "exps": exps, "reports": reports, "weeklies": weeklies,
           "state": state, "latest": latest, "updated": state["last_research_update"]}
    out: dict[str, str] = {"index.html": render_index(ctx)}
    dates = sorted(reports)
    for i, d in enumerate(dates):
        out[f"reports/{d}.html"] = render_report(reports[d], ctx, dates[i - 1] if i else None,
                                                 dates[i + 1] if i + 1 < len(dates) else None)
    for pid, r in papers.items():
        out[f"papers/{slug(pid)}.html"] = render_paper(r, ctx)
    for wk, w in weeklies.items():
        out[f"weekly/{wk}.html"] = render_weekly(w, ctx)
    for name, text in out.items():
        check_html(name, text, p)
    out["data/latest.json"] = dump(derive_latest(reports.get(latest) if latest else None))
    out["data/research_state.json"] = dump(state)
    return out, p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a generated file differs from what the data produces")
    ap.add_argument("--no-history", action="store_true", help="skip the comparison with the last commit")
    ap.add_argument("--missing-zh", action="store_true", help="list visible strings that have no Chinese rendering")
    args = ap.parse_args(argv)
    out, problems = build(history=not args.no_history)
    if args.missing_zh and not problems:
        missing = sorted(s for s in USED if re.search(r"[A-Za-z]{2}", s) and not (ZH.get(s) or UI_ZH.get(s))
                         and not PAPER_ID.match(s) and not is_http(s) and s not in ("arXiv",))
        print(json.dumps({s: "" for s in missing}, ensure_ascii=False, indent=1))
        return 0
    if problems:
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for msg in problems:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    stale = []
    for name, text in out.items():
        path = ROOT / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        stale.append(name)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    generated = {ROOT / n for n in out}
    orphans = [str(f.relative_to(ROOT)) for d in ("reports", "papers", "weekly") for f in (ROOT / d).glob("*.html") if f not in generated]
    for o in orphans:
        print(f"note: {o} is not produced by the data (left untouched)", file=sys.stderr)
    if args.check:
        if stale:
            print("stale generated files: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"ok: {len(out)} generated files are up to date")
        return 0
    print(f"ok: wrote {len(stale)} of {len(out)} generated files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
