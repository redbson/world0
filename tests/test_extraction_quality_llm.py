"""LLM-backed extraction *quality* tests for World 0.

Unlike ``test_extraction.py`` (which uses a ``FakeLLM`` to lock down the
parser), this module exercises the **real** extraction → ingest →
``ConceptNode`` pipeline against a live LLM provider and asserts on the
*semantic quality* of the resulting concept nodes.

These tests are skipped automatically unless an LLM provider is
configured via environment variables:

- ``ANTHROPIC_API_KEY``  → uses :class:`AnthropicProvider`
- ``OPENAI_API_KEY``     → uses :class:`OpenAIProvider`
- ``W0_TEST_LLM_MODEL``  → optional model override

Run them explicitly with::

    ANTHROPIC_API_KEY=sk-... pytest tests/test_extraction_quality_llm.py -v

The assertions are written to be robust to LLM nondeterminism: they
check *quality properties* (a required concept exists, no generic-noise
nouns leaked in, a synonym did not duplicate, a sense split happened,
a relation points the right way) rather than exact extraction output.

The corpus and the assertion helpers are importable so an out-of-band
evaluation harness can drive the same cases through a scripted provider.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pytest

from world0 import World
from world0.schemas.concept import ConceptNode

# ── Provider selection ────────────────────────────────────────────────

_CJK_RE = re.compile(r"[一-鿿]")

# Generic nouns that should never become standalone concepts (they are
# only acceptable when domain-qualified, e.g. "recommendation system").
GENERIC_NOISE = frozenset({
    "system", "thing", "things", "stuff", "process", "processes",
    "data", "result", "results", "value", "values", "component",
    "components", "service", "item", "items", "part", "parts",
    "step", "way", "approach", "method",  # bare/contextless
})


def make_provider():
    """Build a real LLM provider from the environment, or return ``None``."""
    model = os.environ.get("W0_TEST_LLM_MODEL", "")
    if os.environ.get("ANTHROPIC_API_KEY"):
        from world0.llm.anthropic import AnthropicProvider

        return AnthropicProvider(model=model or "claude-sonnet-4-6")
    if os.environ.get("OPENAI_API_KEY"):
        from world0.llm.openai import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-4o-mini")
    return None


_PROVIDER = make_provider()

pytestmark = pytest.mark.skipif(
    _PROVIDER is None,
    reason="no LLM provider configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY)",
)


# ── Introspection helpers (importable, provider-agnostic) ─────────────


def concept_surfaces(world: World) -> list[str]:
    """Lowercased 'name + aliases' surface string for every concept."""
    return [
        " ".join([c.name, *c.aliases]).lower() for c in world.concepts.all()
    ]


def present(world: World, *alternatives: str) -> bool:
    """True if any alternative token appears in some concept's surface form."""
    surfaces = concept_surfaces(world)
    return any(
        any(alt.lower() in surface for alt in alternatives)
        for surface in surfaces
    )


def generic_noise_concepts(world: World) -> list[str]:
    """Concept names that are bare generic nouns (a quality smell)."""
    return [
        c.name
        for c in world.concepts.all()
        if c.name.strip().lower() in GENERIC_NOISE
    ]


def concepts_matching(world: World, *needles: str) -> list[ConceptNode]:
    """Concepts whose name/alias/sense/description contain *all* needles."""
    out: list[ConceptNode] = []
    for c in world.concepts.all():
        hay = " ".join([c.name, *c.aliases, c.sense, c.description]).lower()
        if all(n.lower() in hay for n in needles):
            out.append(c)
    return out


def distinct_ids(*nodes: ConceptNode | None) -> set[str]:
    return {n.id for n in nodes if n is not None}


def relation_between(world: World, a: ConceptNode, b: ConceptNode):
    """Return the first edge connecting two concepts (any direction)."""
    edges = world.relations.find_any_between(a.id, b.id)
    return edges[0] if edges else None


# ── Corpus ────────────────────────────────────────────────────────────


@dataclass
class Case:
    """One extraction-quality scenario."""

    id: str
    text: str
    task: str = ""
    domain: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


CORPUS: list[Case] = [
    Case(
        id="tech_arch",
        task="document the backend architecture",
        text=(
            "Our backend uses FastAPI as the web framework. FastAPI depends "
            "on Starlette for the ASGI layer and on Pydantic for request "
            "validation. The application stores its data in PostgreSQL, "
            "accessed through the SQLAlchemy ORM. We run everything behind "
            "an Nginx reverse proxy."
        ),
        notes="typed dependencies with correct direction; no generic nouns",
    ),
    Case(
        id="synonyms_acronyms",
        task="summarize the answer-grounding pipeline",
        text=(
            "We added Retrieval Augmented Generation (RAG) to the assistant. "
            "RAG retrieves passages from a vector database before the language "
            "model generates an answer, which reduced hallucinations."
        ),
        notes="RAG and 'retrieval augmented generation' must be ONE concept",
    ),
    Case(
        id="relation_direction",
        task="explain the latency fix",
        text=(
            "Putting a Redis cache in front of the database dramatically "
            "reduced API latency. The cache absorbs most of the read traffic "
            "so the database is hit far less often."
        ),
        notes="cache acts on latency/database, not the reverse",
    ),
    Case(
        id="generic_noise",
        task="capture the training method",
        text=(
            "The system processed the data and produced a result. Internally "
            "it used a method called gradient descent to minimize the loss "
            "function while training the neural network."
        ),
        notes="gradient descent / loss function survive; system/data/result do not",
    ),
    Case(
        id="domain_disambiguation",
        task="separate the two senses of 'apple'",
        text=(
            "Apple unveiled the Vision Pro headset, its first new computing "
            "platform in years. Separately, nutritionists note that an apple "
            "a day is a healthy fruit and a good source of fiber."
        ),
        notes="company-Apple and fruit-apple are distinct concept senses",
    ),
    Case(
        id="contradiction",
        task="record the profiling correction",
        text=(
            "Earlier we assumed MongoDB was the performance bottleneck. After "
            "profiling, MongoDB is clearly not the bottleneck; the slow part "
            "is the image resizing step in the upload handler."
        ),
        notes="should surface weakened concept or contradicted relation",
    ),
    Case(
        id="chinese",
        task="描述推荐系统召回链路",
        text=(
            "我们在推荐系统中引入了向量检索来召回候选物品，再用一个排序模型对"
            "候选集打分。向量检索依赖嵌入模型把物品编码成稠密向量，从而支持"
            "近似最近邻搜索。"
        ),
        notes="concept names must preserve Chinese (CJK) surface forms",
    ),
]

CASES_BY_ID = {c.id: c for c in CORPUS}


def fresh_world(tmp_path, provider) -> World:
    return World(store_path=str(tmp_path), llm=provider)


# ── Per-case quality tests ────────────────────────────────────────────


def test_tech_arch_dependencies_and_direction(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["tech_arch"]
    w.ingest_text(case.text, task=case.task, source=case.id)

    # Required architectural concepts are present.
    for needle in ("fastapi", "postgresql", "sqlalchemy"):
        assert present(w, needle), f"missing expected concept: {needle}"
    # At least one of the supporting deps.
    assert present(w, "starlette", "pydantic", "nginx")

    # No bare generic nouns leaked in.
    assert not generic_noise_concepts(w), generic_noise_concepts(w)

    # Dependency direction: FastAPI depends on Starlette/Pydantic, not vice
    # versa.  Find a dependency edge and check FastAPI is the dependent side.
    fastapi = w.concepts.resolve("fastapi")
    assert fastapi is not None
    dep = None
    for other_label in ("starlette", "pydantic"):
        other = w.concepts.resolve(other_label)
        if other is None:
            continue
        edge = relation_between(w, fastapi, other)
        if edge is not None:
            dep = (edge, other)
            break
    assert dep is not None, "no dependency edge found from FastAPI"
    edge, other = dep
    # The dependent (FastAPI) should be the source of a positive dependence,
    # i.e. the framework points at what it needs — not the library pointing
    # back at the framework.
    if edge.semantic_relation in {"dependence", "enables", "membership", "inclusion"}:
        assert edge.source_id == fastapi.id or edge.target_id == other.id


def test_synonym_acronym_does_not_duplicate(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["synonyms_acronyms"]
    w.ingest_text(case.text, task=case.task, source=case.id)

    rag = w.concepts.resolve("RAG")
    full = w.concepts.resolve("retrieval augmented generation")
    # Whatever the model named it, the two surface forms must NOT resolve to
    # two different concept nodes.
    ids = distinct_ids(rag, full)
    assert len(ids) <= 1, (
        "RAG and 'retrieval augmented generation' became duplicate concepts"
    )
    # And the concept must exist under at least one of the forms.
    assert rag is not None or full is not None
    # The supporting concept is present too.
    assert present(w, "vector database", "vector store", "vector")


def test_relation_direction_cache_latency(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["relation_direction"]
    w.ingest_text(case.text, task=case.task, source=case.id)

    assert present(w, "cache", "redis")
    assert present(w, "latency")

    cache = w.concepts.resolve("redis cache") or w.concepts.resolve("cache") \
        or (concepts_matching(w, "cache")[:1] or [None])[0]
    latency = w.concepts.resolve("latency") \
        or (concepts_matching(w, "latency")[:1] or [None])[0]
    assert cache is not None and latency is not None

    edge = relation_between(w, cache, latency)
    if edge is not None:
        # Latency must not be the *driver* of a dependence/enable toward the
        # cache — that would be a reversed relation.
        reversed_bad = (
            edge.source_id == latency.id
            and edge.target_id == cache.id
            and edge.semantic_relation in {"dependence", "enables"}
        )
        assert not reversed_bad, "relation direction is reversed (latency→cache)"


def test_generic_noise_is_filtered(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["generic_noise"]
    w.ingest_text(case.text, task=case.task, source=case.id)

    # The real concept survives.
    assert present(w, "gradient descent")
    # The noise does not.
    leaked = generic_noise_concepts(w)
    assert not leaked, f"generic noise leaked as concepts: {leaked}"


def test_domain_disambiguation_splits_senses(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["domain_disambiguation"]
    w.ingest_text(case.text, task=case.task, source=case.id)

    apple_nodes = [
        c for c in w.concepts.all()
        if "apple" in " ".join([c.name, *c.aliases]).lower()
    ]
    # The two senses (company, fruit) must not collapse into one node.
    assert len({c.id for c in apple_nodes}) >= 2, (
        "company-Apple and fruit-apple collapsed into a single concept"
    )


def test_contradiction_surfaces_negative_evidence(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["contradiction"]
    # Inspect the raw observation directly — the negative-evidence channels
    # live on the Observation before ingest folds them in.
    obs = w._extractor.extract(case.text, task=case.task, source=case.id)
    assert obs.weakened or obs.contradicted_relations, (
        "text explicitly rejects a belief but no weakened/contradicted "
        "evidence was extracted"
    )
    # The corrected cause is captured as a concept.
    result = w.ingest(obs)
    assert present(w, "image resizing", "resizing", "image")
    # The negative evidence was applied *somewhere* — as a reported weakened
    # concept/relation, or (when a contradicted relation has no existing edge)
    # as node-level disconfirmation on the endpoints.  Note: that last path is
    # not surfaced in IngestResult, so assert on the actual node/edge state.
    applied = bool(
        result.weakened_concepts
        or result.weakened_relations
        or obs.weakened
        or any(c.disconfirmation_count > 0 for c in w.concepts.all())
        or any(e.disconfirmation_count > 0 for e in w.relations.all())
    )
    assert applied, "contradiction extracted but no disconfirmation was applied"


def test_chinese_preserves_language(tmp_path):
    w = fresh_world(tmp_path, _PROVIDER)
    case = CASES_BY_ID["chinese"]
    w.ingest_text(case.text, task=case.task, source=case.id)

    names = [c.name for c in w.concepts.all()]
    cjk_names = [n for n in names if _CJK_RE.search(n)]
    assert len(cjk_names) >= 2, f"expected Chinese concept names, got {names}"
    # Key concepts present.
    assert present(w, "向量检索", "向量召回", "检索")
    assert present(w, "排序模型", "嵌入模型", "推荐系统")


def test_cross_text_identity_is_stable(tmp_path):
    """The same concept seen in two observations reinforces one node."""
    w = fresh_world(tmp_path, _PROVIDER)
    w.ingest_text(CASES_BY_ID["tech_arch"].text, task="arch", source="t1")
    fastapi = w.concepts.resolve("fastapi")
    assert fastapi is not None
    first_count = fastapi.activation_count
    before_total = len(w.concepts.all())

    w.ingest_text(
        "We upgraded FastAPI to the latest version and re-checked that "
        "Pydantic models still validate every request.",
        task="arch",
        source="t2",
    )
    fastapi_again = w.concepts.resolve("fastapi")
    assert fastapi_again is not None
    # Same identity, reinforced — not a duplicate.
    assert fastapi_again.id == fastapi.id
    assert fastapi_again.activation_count > first_count
    # FastAPI was not duplicated by the second observation.
    fastapi_like = [
        c for c in w.concepts.all()
        if c.name.strip().lower() == "fastapi"
    ]
    assert len(fastapi_like) == 1
    # The world grew sanely (no pathological duplication explosion).
    assert len(w.concepts.all()) >= before_total
