# World0 Research Portal

Static research-intelligence dashboard for the World0 project (no backend, no runtime API keys).
Published by GitHub Pages at `https://redbson.github.io/world0/research-portal/`.

## Layout

| Path | Purpose |
|---|---|
| `index.html` | Daily overview: what changed, top discoveries, World0 impact, hypothesis movements, next steps |
| `archive.html` | Every daily report, filterable by category / direction / importance / date |
| `papers.html`, `papers/<id>.html` | Tracked papers; structured argument analysis for deep-read papers |
| `hypotheses.html` | Hypothesis lifecycle (dated status history, evidence, experiments) |
| `architecture.html` | World0 architecture impact: per-area evidence and concrete proposals |
| `reports/YYYY-MM-DD.html` | Permanent daily reports (never rewritten after their day) |
| `weekly/YYYY-Www.html` | Weekly syntheses |
| `data/` | Source of truth (JSON) — see below |
| `tools/build.py` | Renders the site from `data/`, computes the daily delta, validates everything |

## Data (stable identifiers)

| File | ID pattern | Notes |
|---|---|---|
| `papers.json` | `arxiv:NNNN.NNNNN` / `doi:…` | deduplicated by arXiv ID / DOI; `verification_level`, optional `argument` block |
| `findings.json` | `F-YYYYMMDD-NN` | one per key discovery; links paper, evidence, hypotheses, proposals |
| `evidence.json` | `EV-NNN` | `kind` ∈ author_claim / experimental_observation / agent_interpretation / speculation |
| `hypotheses.json` | `H-NNN` | `history[]` of dated status changes with reasons |
| `experiments.json` | `E-NNN` | `history[]` |
| `proposals.json` | `AP-NNN` | architecture proposals tied to World0 components |
| `reports.json` | `R-YYYY-MM-DD` | index of daily reports (summary, categories, directions, importance) |
| `deltas/YYYY-MM-DD.json` | — | generated: what changed that day vs. the previous state |
| `latest.json`, `research_state.json` | — | generated pointers / run log |
| `verification_levels.json` | V0–V4 | definitions and the strength rules the validator enforces |
| `i18n_zh.json` | — | Chinese translations keyed by the English string |

Verification levels: **V0** title only · **V1** search summary · **V2** abstract read · **V3** full text read · **V4** independently checked.
Evidence labels (CONFIRMED / SUPPORTED / EXPERIMENTAL / SPECULATIVE) describe literature support, never World0 implementation status; the validator rejects CONFIRMED without a V3+ source and SUPPORTED without two sources or V3.

## Daily workflow

1. Append the day's records to `data/*.json` (never delete or rewrite earlier records; change a status by appending to `history[]` and updating `last_updated`).
2. Add the day's entry to `reports.json`.
3. `python3 -I tools/build.py --date YYYY-MM-DD` — validates records and cross-references, writes `data/deltas/<date>.json`, `latest.json`, `reports/<date>.html` and regenerates the overview pages. Earlier `reports/*.html` are left untouched. `W0_SHOW_UNTRANSLATED=1` lists strings still missing a Chinese translation.
4. Commit only `research-portal/` (`research: daily intelligence YYYY-MM-DD`), push, then confirm the Pages build succeeded before recording a deployment (`--deployed <ISO timestamp>`).
