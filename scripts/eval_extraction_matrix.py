#!/usr/bin/env python
"""World 0 — model x prompt extraction-quality evaluation harness.

Runs the REAL extraction pipeline (ConceptExtractor parse + World.ingest)
over a fixed bilingual corpus for every (model, prompt) cell, scores the
resulting ConceptNodes with a quality rubric, and prints a comparison
table (optionally a Markdown report).

It is provider-agnostic and credential-aware: any backend whose API key /
endpoint is missing from the environment is skipped automatically, so the
script is safe to run anywhere (it simply evaluates whatever it can reach).

Backends & env vars
-------------------
- azure   : AZURE_OPENAI_ENDPOINT + (AZURE_OPENAI_KEY | AZURE_OPENAI_API_KEY)
            model label = Azure deployment name (e.g. gpt-5.4-nano, DeepSeek-V4-Pro, grok-4.3)
- openai  : OPENAI_API_KEY
- anthropic : ANTHROPIC_API_KEY | CLAUDE_API_KEY   (e.g. claude-sonnet-4-6)
- glm     : GLM_API_KEY  (Zhipu OpenAI-compatible endpoint)

Examples
--------
    python scripts/eval_extraction_matrix.py --list
    python scripts/eval_extraction_matrix.py --models gpt-5.4-nano,DeepSeek-V4-Pro --runs 3
    python scripts/eval_extraction_matrix.py --models all --prompts both --md report.md

Notes
-----
- claude-opus-4-8 rejects the `temperature` parameter; it is wired through a
  no-temperature adapter automatically.
- Same-name self-loop relations emitted by some models are sanitized before
  ingest (see docs/extraction-model-prompt-eval.md for the underlying bug).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Make `world0` importable when run from a source checkout.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from world0 import World  # noqa: E402
from world0.extraction.extractor import ConceptExtractor  # noqa: E402
from world0.llm.base import LLMProvider, LLMError  # noqa: E402
from world0.prompts import PromptRegistry  # noqa: E402
from world0.schemas.concept import ConceptNode  # noqa: E402

_CJK_RE = re.compile(r"[一-鿿]")
GENERIC_NOISE = frozenset({
    "system", "thing", "things", "stuff", "process", "processes", "data",
    "result", "results", "value", "values", "component", "components",
    "service", "item", "items", "part", "parts", "step", "way", "approach",
    "method",
})


def _norm(s: object) -> str:
    return re.sub(r"[_\W]+", " ", str(s).strip().lower()).strip()


# ── Corpus ────────────────────────────────────────────────────────────


@dataclass
class Case:
    id: str
    text: str
    task: str = ""


CORPUS: list[Case] = [
    Case("tech_arch", (
        "Our backend uses FastAPI as the web framework. FastAPI depends on "
        "Starlette for the ASGI layer and on Pydantic for request validation. "
        "The application stores its data in PostgreSQL, accessed through the "
        "SQLAlchemy ORM. We run everything behind an Nginx reverse proxy."
    ), "document the backend architecture"),
    Case("synonyms_acronyms", (
        "We added Retrieval Augmented Generation (RAG) to the assistant. RAG "
        "retrieves passages from a vector database before the language model "
        "generates an answer, which reduced hallucinations."
    ), "summarize the answer-grounding pipeline"),
    Case("relation_direction", (
        "Putting a Redis cache in front of the database dramatically reduced "
        "API latency. The cache absorbs most of the read traffic so the "
        "database is hit far less often."
    ), "explain the latency fix"),
    Case("generic_noise", (
        "The system processed the data and produced a result. Internally it "
        "used a method called gradient descent to minimize the loss function "
        "while training the neural network."
    ), "capture the training method"),
    Case("domain_disambiguation", (
        "Apple unveiled the Vision Pro headset, its first new computing "
        "platform in years. Separately, nutritionists note that an apple a "
        "day is a healthy fruit and a good source of fiber."
    ), "separate the two senses of 'apple'"),
    Case("contradiction", (
        "Earlier we assumed MongoDB was the performance bottleneck. After "
        "profiling, MongoDB is clearly not the bottleneck; the slow part is "
        "the image resizing step in the upload handler."
    ), "record the profiling correction"),
    Case("chinese", (
        "我们在推荐系统中引入了向量检索来召回候选物品，再用一个排序模型对候选集打分。"
        "向量检索依赖嵌入模型把物品编码成稠密向量，从而支持近似最近邻搜索。"
    ), "描述推荐系统召回链路"),
]


# ── Quality helpers ───────────────────────────────────────────────────


def _surfaces(w: World) -> list[str]:
    return [" ".join([c.name, *c.aliases]).lower() for c in w.concepts.all()]


def present(w: World, *alts: str) -> bool:
    surf = _surfaces(w)
    return any(any(a.lower() in s for a in alts) for s in surf)


def generic_noise(w: World) -> list[str]:
    return [c.name for c in w.concepts.all()
            if c.name.strip().lower() in GENERIC_NOISE]


def matching(w: World, *needles: str) -> list[ConceptNode]:
    out = []
    for c in w.concepts.all():
        hay = " ".join([c.name, *c.aliases, c.sense, c.description]).lower()
        if all(n.lower() in hay for n in needles):
            out.append(c)
    return out


def distinct_ids(*nodes) -> set[str]:
    return {n.id for n in nodes if n is not None}


def relation_between(w: World, a, b):
    edges = w.relations.find_any_between(a.id, b.id) if a and b else []
    return edges[0] if edges else None


# ── Prompts ───────────────────────────────────────────────────────────

DEFAULT_PROMPT = PromptRegistry().render("extraction.concepts_relations.system")
NAIVE_PROMPT = (
    "Extract concepts and relations from the text. "
    'Return JSON only: {"concepts": [{"name": "..."}], '
    '"relations": [["source", "target", "type"]]}. '
    "List every noun concept you can find."
)
PROMPTS = {"default": DEFAULT_PROMPT, "naive": NAIVE_PROMPT}


# ── Provider backends ─────────────────────────────────────────────────


def _anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")


def _azure_ready() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT")
                and (os.environ.get("AZURE_OPENAI_KEY")
                     or os.environ.get("AZURE_OPENAI_API_KEY")))


@dataclass
class ModelSpec:
    label: str
    backend: str   # azure | openai | anthropic | anthropic_notemp | glm
    model: str = ""   # backend model/deployment id (defaults to label)

    def __post_init__(self):
        if not self.model:
            self.model = self.label

    def available(self) -> bool:
        if self.backend == "azure":
            return _azure_ready()
        if self.backend in ("anthropic", "anthropic_notemp"):
            return bool(_anthropic_key())
        if self.backend == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        if self.backend == "glm":
            return bool(os.environ.get("GLM_API_KEY"))
        return False

    def build(self) -> LLMProvider:
        if self.backend == "azure":
            from world0.llm.azure_openai import AzureOpenAIProvider
            return AzureOpenAIProvider(model=self.model)
        if self.backend == "openai":
            from world0.llm.openai import OpenAIProvider
            return OpenAIProvider(model=self.model)
        if self.backend == "anthropic":
            from world0.llm.anthropic import AnthropicProvider
            return AnthropicProvider(model=self.model, api_key=_anthropic_key())
        if self.backend == "anthropic_notemp":
            return _AnthropicNoTemp(self.model)
        if self.backend == "glm":
            return _GLMProvider(self.model)
        raise ValueError(self.backend)


class _AnthropicNoTemp(LLMProvider):
    """Anthropic without the temperature param (required by opus-4.8)."""

    def __init__(self, model: str):
        from anthropic import Anthropic
        self._c = Anthropic(api_key=_anthropic_key())
        self._m = model

    def complete_json(self, system: str, user: str) -> str:
        try:
            r = self._c.messages.create(model=self._m, max_tokens=4096,
                                        system=system,
                                        messages=[{"role": "user", "content": user}])
            return r.content[0].text
        except Exception as e:  # noqa: BLE001
            raise LLMError(str(e))


class _GLMProvider(LLMProvider):
    """Zhipu GLM via its OpenAI-compatible endpoint."""

    def __init__(self, model: str):
        from openai import OpenAI
        self._c = OpenAI(api_key=os.environ.get("GLM_API_KEY"),
                         base_url="https://open.bigmodel.cn/api/paas/v4")
        self._m = model

    def complete_json(self, system: str, user: str) -> str:
        try:
            r = self._c.chat.completions.create(
                model=self._m, temperature=0.1,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            raise LLMError(str(e))


# Default model catalog (edit/extend freely; unavailable ones are skipped).
CATALOG: list[ModelSpec] = [
    ModelSpec("gpt-5.4", "azure"),
    ModelSpec("gpt-5.4-mini", "azure"),
    ModelSpec("gpt-5.4-nano", "azure"),
    ModelSpec("DeepSeek-V4-Pro", "azure"),
    ModelSpec("DeepSeek-V4-Flash", "azure"),
    ModelSpec("grok-4.3", "azure"),
    ModelSpec("glm-5.1", "glm"),
    ModelSpec("claude-opus-4-8", "anthropic_notemp"),
    ModelSpec("claude-sonnet-4-6", "anthropic"),
]
CATALOG_BY_LABEL = {m.label: m for m in CATALOG}

_parser = ConceptExtractor.__new__(ConceptExtractor)  # parse-only


# ── Evaluation ────────────────────────────────────────────────────────


def extract_one(spec: ModelSpec, prompt_name: str, case: Case):
    sys_prompt = PROMPTS[prompt_name]
    user = ConceptExtractor._build_user_prompt(case.text, task=case.task, source=case.id)
    for attempt in range(2):
        try:
            raw = spec.build().complete_json(sys_prompt, user)
            obs = _parser._parse_response(raw, task=case.task, source=case.id)
            return obs, None
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                return None, str(e)[:140]
            time.sleep(2)


def metrics_for(obs_by_case: dict) -> dict:
    acc = dict(concepts=0, noise=0, raw_rel=0, dropped=0, edges=0, generic=0, fail=0)
    flags: dict[str, bool] = {}
    for case in CORPUS:
        obs = obs_by_case.get(case.id)
        if obs is None:
            acc["fail"] += 1
            continue
        obs.relations = [r for r in obs.relations
                         if len(r) >= 2 and _norm(r[0]) != _norm(r[1])]
        obs.contradicted_relations = [r for r in obs.contradicted_relations
                                      if len(r) >= 2 and _norm(r[0]) != _norm(r[1])]
        with tempfile.TemporaryDirectory() as d:
            w = World(store_path=d)
            try:
                w.ingest(obs)
            except Exception:  # noqa: BLE001
                obs.relations = []
                obs.contradicted_relations = []
                w = World(store_path=d)
                w.ingest(obs)
            raw = obs.extraction_metadata.get("raw_counts", {})
            acc["concepts"] += len(w.concepts.all())
            acc["noise"] += len(generic_noise(w))
            acc["raw_rel"] += raw.get("relations", 0)
            acc["dropped"] += raw.get("dropped_relations", 0)
            rels = w.relations.all()
            acc["edges"] += len(rels)
            acc["generic"] += sum(1 for e in rels if e.semantic_relation == "generic_relation")
            if case.id == "synonyms_acronyms":
                flags["dedup"] = len(distinct_ids(
                    w.concepts.resolve("RAG"),
                    w.concepts.resolve("retrieval augmented generation"))) <= 1
            elif case.id == "domain_disambiguation":
                apple = {c.id for c in w.concepts.all()
                         if "apple" in " ".join([c.name, *c.aliases]).lower()}
                flags["domain"] = len(apple) >= 2
            elif case.id == "relation_direction":
                cache = (w.concepts.resolve("redis cache") or w.concepts.resolve("cache")
                         or (matching(w, "cache")[:1] or [None])[0])
                lat = w.concepts.resolve("latency") or (matching(w, "latency")[:1] or [None])[0]
                e = relation_between(w, cache, lat)
                bad = bool(e and e.source_id == lat.id and e.target_id == cache.id
                           and e.semantic_relation in {"dependence", "enables"})
                flags["direction"] = (e is not None) and not bad
            elif case.id == "contradiction":
                flags["contra"] = bool(obs.weakened or obs.contradicted_relations)
            elif case.id == "chinese":
                flags["zh"] = sum(1 for c in w.concepts.all() if _CJK_RE.search(c.name)) >= 2
    drop = acc["dropped"] / acc["raw_rel"] if acc["raw_rel"] else 0.0
    genr = acc["generic"] / acc["edges"] if acc["edges"] else 0.0
    bench = ["dedup", "domain", "direction", "contra", "zh"]
    passed = sum(1 for k in bench if flags.get(k))
    quality = passed + (acc["noise"] == 0) + (genr < 0.1) + (drop < 0.05) + (acc["fail"] == 0)
    return {"concepts": acc["concepts"], "noise": acc["noise"], "drop": drop,
            "generic": genr, "fail": acc["fail"], "quality": quality, **flags}


FLAGSET = {"dedup", "domain", "direction", "contra", "zh"}
ROW_ORDER = ["quality", "concepts", "noise", "drop", "generic",
             "dedup", "domain", "direction", "contra", "zh", "fail"]
ROW_LABEL = {"quality": "quality/9", "concepts": "concepts#", "noise": "noise",
             "drop": "drop_rate", "generic": "generic_rel", "dedup": "RAG dedup",
             "domain": "Apple split", "direction": "direction", "contra": "contradiction",
             "zh": "chinese", "fail": "failed"}


def aggregate(per_run: list[dict], runs: int) -> dict:
    def avg(k): return sum(x[k] for x in per_run) / len(per_run)
    def passes(k): return sum(1 for x in per_run if x.get(k))
    return {"quality": avg("quality"), "concepts": avg("concepts"),
            "noise": avg("noise"), "drop": avg("drop"), "generic": avg("generic"),
            "fail": sum(x["fail"] for x in per_run),
            "dedup": passes("dedup"), "domain": passes("domain"),
            "direction": passes("direction"), "contra": passes("contra"),
            "zh": passes("zh")}


def fmt(key: str, v, runs: int) -> str:
    if key in FLAGSET:
        return f"{v}/{runs}"
    if key in ("quality", "concepts", "noise"):
        return f"{v:.1f}"
    if key in ("drop", "generic"):
        return f"{v:.2f}"
    return str(v)


def render_markdown(cells: dict, cols: list[tuple[str, str]], runs: int) -> str:
    lines = ["# World 0 — extraction quality matrix",
             "",
             f"{runs} run(s) aggregated; flags = passes/{runs}, numbers = mean. "
             "Higher quality/RAG/Apple/direction/contradiction/chinese is better; "
             "lower noise/drop_rate/generic_rel is better.",
             ""]
    head = "| metric | " + " | ".join(f"{m}/{p}" for m, p in cols) + " |"
    sep = "|---|" + "|".join("---" for _ in cols) + "|"
    lines += [head, sep]
    for r in ROW_ORDER:
        row = f"| {ROW_LABEL[r]} | " + " | ".join(
            fmt(r, cells[c][r], runs) for c in cols) + " |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all",
                    help="comma-separated model labels, or 'all' (default)")
    ap.add_argument("--prompts", default="default", choices=["default", "naive", "both"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--md", default="", help="write a Markdown report to this path")
    ap.add_argument("--list", action="store_true", help="list catalog + availability and exit")
    args = ap.parse_args()

    if args.list:
        print("model".ljust(22), "backend".ljust(18), "available")
        for m in CATALOG:
            print(m.label.ljust(22), m.backend.ljust(18), "yes" if m.available() else "no")
        return 0

    if args.models == "all":
        specs = [m for m in CATALOG if m.available()]
    else:
        specs = []
        for lbl in [x.strip() for x in args.models.split(",") if x.strip()]:
            spec = CATALOG_BY_LABEL.get(lbl)
            if spec is None:
                print(f"unknown model {lbl!r} (see --list)", file=sys.stderr)
                continue
            specs.append(spec)
    runnable = [s for s in specs if s.available()]
    skipped = [s.label for s in specs if not s.available()]
    if skipped:
        print(f"skipping (no creds): {', '.join(skipped)}", file=sys.stderr)
    if not runnable:
        print("No runnable models (missing credentials). Nothing to do.", file=sys.stderr)
        return 0

    prompt_names = ["default", "naive"] if args.prompts == "both" else [args.prompts]
    cols = [(s.label, p) for s in runnable for p in prompt_names]

    tasks = [(s, p, run, c) for s in runnable for p in prompt_names
             for run in range(args.runs) for c in CORPUS]
    print(f"running {len(tasks)} real extractions "
          f"({args.runs} run(s) x {len(runnable)} model(s) x {len(prompt_names)} prompt(s)) ...",
          file=sys.stderr)
    raw_results: dict = {}
    t0 = time.time()

    def work(t):
        s, p, run, c = t
        obs, err = extract_one(s, p, c)
        return (s.label, p, run, c.id, obs, err)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for label, p, run, cid, obs, err in ex.map(work, tasks):
            raw_results.setdefault((label, p, run), {})[cid] = obs
            if obs is None:
                print(f"  ERR {label}/{p} r{run} {cid}: {err}", file=sys.stderr)
    print(f"done in {time.time()-t0:.0f}s", file=sys.stderr)

    cells = {}
    for (label, p) in cols:
        per_run = [metrics_for(raw_results[(label, p, r)]) for r in range(args.runs)]
        cells[(label, p)] = aggregate(per_run, args.runs)

    # Console table
    width = 13
    hdr = f"{'metric':<13}" + "".join(f"{(m+'/'+p)[:width-1]:>{width}}" for m, p in cols)
    print("\n" + "=" * len(hdr))
    print(f"CROSS-MODEL EXTRACTION QUALITY — {args.runs} run(s) aggregated")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in ROW_ORDER:
        print(f"{ROW_LABEL[r]:<13}" + "".join(
            f"{fmt(r, cells[c][r], args.runs):>{width}}" for c in cols))
    print("-" * len(hdr))
    rank = sorted(cols, key=lambda c: (-cells[c]["quality"], cells[c]["noise"]))
    print("ranking by mean quality/9:")
    for c in rank:
        print(f"  {cells[c]['quality']:.2f}  {c[0]}/{c[1]}")

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(render_markdown(cells, cols, args.runs))
        print(f"\nMarkdown report -> {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
