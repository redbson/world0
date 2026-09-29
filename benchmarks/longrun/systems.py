"""Memory strategies compared by LongRun, behind one interface.

Every system sees the same event stream through ``observe`` and, for a
query, must return a ``Context`` that fits ``budget`` tokens.  The harness
hands every system the same *linked entities* (the known concept names that
occur in the query text) and the query's task text, and counts context size
with ``tokens.est_tokens``.

Two kinds of input, both explicit:

* text systems (window, full context, RAG, summaries) read ``ev.text``; what
  an *ideal reader* extracts from the text they return is scored;
* structured systems (fact store, graphs, World 0) ingest ``ev.extracted``
  - what an extractor produced, perfect when ``extract_p = 0`` - and are
  scored on what their returned text says.

Nothing here has an LLM in the loop; ``docs/eval`` says what that leaves out.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from benchmarks.longrun.parse import parse_compact, parse_shipped
from benchmarks.longrun.tokens import est_tokens
from benchmarks.longrun.worldgen import Claim, Event, Query

TUNED_PATH = os.path.join(os.path.dirname(__file__), "tuned.json")


@dataclass
class Context:
    text: str = ""
    claims: set[Claim] = field(default_factory=set)
    concepts: set[str] = field(default_factory=set)
    tickets: set[str] = field(default_factory=set)
    beliefs: dict[Claim, float] = field(default_factory=dict)  # only where shown

    @property
    def tokens(self) -> int:
        return est_tokens(self.text)


def replay(events: list[Event]) -> tuple[set[Claim], set[str], set[str]]:
    """What an ideal reader concludes from events read in step order."""
    claims: set[Claim] = set()
    concepts: set[str] = set()
    tickets: set[str] = set()
    for ev in sorted(events, key=lambda e: e.step):
        claims |= set(ev.claims)
        claims -= set(ev.retractions)
        concepts |= set(ev.concepts)
        if ev.ticket:
            tickets.add(ev.ticket)
    return claims, concepts, tickets


def _from_events(events: list[Event]) -> Context:
    events = sorted(events, key=lambda e: e.step)
    claims, concepts, tickets = replay(events)
    return Context("\n".join(e.text for e in events), claims, concepts, tickets)


def _tokens(text: str) -> list[str]:
    for ch in ",.:()[];":
        text = text.replace(ch, " ")
    return text.lower().split()


def task_overlap(query_task: str, label: str) -> float:
    """Share of the query's task words that the label contains."""
    q = set(_tokens(query_task))
    return len(q & set(_tokens(label))) / len(q) if q else 0.0


# A task-tagged claim belongs to the query's task when the labels share at
# least this much of the query's *distinctive* task words.  Every system
# that uses task labels gets the same matcher World 0 uses
# (``TaskVocabulary``): with plain word overlap every benchmark task ("<domain>
# work") matched every other at 0.5 through the word "work", which silently
# disabled the task filters of fact_task / state_doc as well as World 0's.
TASK_MATCH = 0.5


class TaskMatcher:
    """Distinctiveness-weighted task matching over the labels a system has seen."""

    def __init__(self) -> None:
        from world0.schemas.concept import TaskVocabulary

        self.vocabulary = TaskVocabulary()
        self._seen: set[str] = set()

    def observe(self, label: str) -> None:
        if label and label not in self._seen:
            self._seen.add(label)
            self.vocabulary.add(label)

    def match(self, query_task: str, label: str) -> float:
        from world0.schemas.concept import task_match_score

        return task_match_score(query_task, label, self.vocabulary)


class System:
    name = "base"
    unbounded = False

    def observe(self, ev: Event) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def query(self, q: Query, linked: list[str], budget: int) -> Context:  # pragma: no cover
        raise NotImplementedError

    def stats(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def _fill(lines: list[tuple[str, Claim | None]], budget: int, header: str = "") -> Context:
    ctx, room = Context(), budget - (est_tokens(header) + 1 if header else 0)
    out = []
    for line, claim in lines:
        w = est_tokens(line) + 1
        if w > room:
            continue
        out.append(line)
        room -= w
        if claim is not None:
            ctx.claims.add(claim)
            ctx.concepts |= {claim.src, claim.tgt}
    ctx.text = (header + "\n" if header else "") + "\n".join(out)
    return ctx


# ── no memory ────────────────────────────────────────────────────────────
class NoMemory(System):
    name = "none"

    def observe(self, ev): pass
    def query(self, q, linked, budget): return Context()


# ── direct context: the last events that fit, or (up to a cap) everything ─
class Window(System):
    name = "window"

    def __init__(self) -> None:
        self.events: list[Event] = []

    def observe(self, ev): self.events.append(ev)

    def _last(self, budget: int) -> list[Event]:
        chosen, used = [], 0
        for ev in reversed(self.events):
            if used + ev.tokens + 1 > budget:
                break
            chosen.append(ev)
            used += ev.tokens + 1
        return chosen

    def query(self, q, linked, budget):
        return _from_events(self._last(budget))

    def stats(self):
        return {"items": len(self.events), "bytes": sum(len(e.text) for e in self.events)}


class FullContext(Window):
    """Everything the model's window (128k tokens) can hold, newest last."""

    name = "full_context"
    unbounded = True
    cap = 128_000

    def query(self, q, linked, budget):
        return _from_events(self._last(self.cap))


class FullContext32k(FullContext):
    """Everything a 32k-token window holds."""

    name = "full_32k"
    cap = 32_000


class FullContext64k(FullContext):
    name = "full_64k"
    cap = 64_000


# ── retrieval over past events (BM25) ────────────────────────────────────
class _BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.len: list[int] = []
        self.total = 0

    def add(self, text: str) -> int:
        doc = len(self.len)
        toks = _tokens(text)
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            self.post[t].append((doc, c))
        self.len.append(len(toks))
        self.total += len(toks)
        return doc

    def scores(self, query: str) -> dict[int, float]:
        n = len(self.len)
        avg = self.total / max(1, n)
        out: dict[int, float] = defaultdict(float)
        for t in set(_tokens(query)):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, c in plist:
                denom = c + self.k1 * (1 - self.b + self.b * self.len[doc] / avg)
                out[doc] += idf * c * (self.k1 + 1) / denom
        return out


class RAG(System):
    """Top-k past events by BM25 similarity to the query text."""

    name = "rag"

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.index = _BM25()

    def observe(self, ev):
        self.events.append(ev)
        self.index.add(ev.text)

    def _rank(self, q: Query) -> list[int]:
        sc = self.index.scores(q.text)
        return sorted(sc, key=lambda d: (-sc[d], -d))

    def query(self, q, linked, budget):
        chosen, used = [], 0
        for doc in self._rank(q):
            ev = self.events[doc]
            if used + ev.tokens + 1 <= budget:
                chosen.append(ev)
                used += ev.tokens + 1
        return _from_events(chosen)

    def stats(self):
        return {"items": len(self.events), "bytes": sum(len(e.text) for e in self.events)
                + 8 * sum(len(p) for p in self.index.post.values())}


class RAGRecency(RAG):
    """Generative-Agents recipe (relevance + recency + importance), not plain RAG."""

    name = "rag_recency"

    def _rank(self, q):
        sc = self.index.scores(q.text)
        if not sc:
            return list(range(len(self.events) - 1, -1, -1))
        cand = set(sorted(sc, key=lambda d: -sc[d])[:60]) | set(range(max(0, len(self.events) - 10), len(self.events)))
        now = len(self.events) - 1
        top = max(sc.values()) or 1.0

        def norm(vals):
            lo, hi = min(vals.values()), max(vals.values())
            return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in vals.items()}

        rel = norm({d: sc.get(d, 0.0) / top for d in cand})
        rec = norm({d: 0.99 ** (now - d) for d in cand})
        imp = norm({d: float(len(self.events[d].claims) + len(self.events[d].retractions)) for d in cand})
        tot = {d: rel[d] + rec[d] + imp[d] for d in cand}
        return sorted(tot, key=lambda d: (-tot[d], -d))


# ── summaries (extractive: an ideal summariser over the text) ────────────
class SummaryBuffer(System):
    """ConversationSummaryBuffer: recent-events window + one flat frequency digest."""

    name = "summary_buffer"

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.count: dict[Claim, int] = defaultdict(int)
        self.last: dict[Claim, int] = {}

    def observe(self, ev):
        self.events.append(ev)
        for c in ev.claims:
            self.count[c] += 1
            self.last[c] = ev.step
        for c in ev.retractions:
            self.count.pop(c, None)
            self.last.pop(c, None)

    def query(self, q, linked, budget):
        half, chosen, used = budget // 2, [], 0
        for ev in reversed(self.events):
            if used + ev.tokens + 1 > half:
                break
            chosen.append(ev)
            used += ev.tokens + 1
        ctx = _from_events(chosen)
        lines = [(f"{c.sentence()} (x{self.count[c]})", c)
                 for c in sorted(self.count, key=lambda c: (-self.count[c], -self.last[c], c))]
        digest = _fill(lines, budget - used - 2, "Summary of earlier work:")
        ctx.claims |= digest.claims
        ctx.concepts |= digest.concepts
        ctx.text = digest.text + "\n" + ctx.text
        return ctx

    def stats(self):
        return {"items": len(self.events) + len(self.count),
                "bytes": sum(len(e.text) for e in self.events)}


class SummaryTask(System):
    """One digest per task thread, the query's task first (a per-thread rolling summary)."""

    name = "summary_task"

    def __init__(self) -> None:
        self.count: dict[str, dict[Claim, int]] = defaultdict(lambda: defaultdict(int))
        self.last_task_step: dict[str, int] = {}
        self.matcher = TaskMatcher()

    def observe(self, ev):
        self.matcher.observe(ev.task)
        self.last_task_step[ev.task] = ev.step
        for c in ev.claims:
            self.count[ev.task][c] += 1
        for c in ev.retractions:
            for t in self.count:
                self.count[t].pop(c, None)

    def query(self, q, linked, budget):
        tasks = sorted(self.count, key=lambda t: (-self.matcher.match(q.task_text, t), -self.last_task_step[t]))
        lines: list[tuple[str, Claim | None]] = []
        for t in tasks:
            lines.append((f"[{t}]", None))
            lines += [(f"{c.sentence()} (x{n})", c) for c, n in
                      sorted(self.count[t].items(), key=lambda kv: (-kv[1], kv[0]))]
        return _fill(lines, budget)

    def stats(self):
        return {"items": sum(len(v) for v in self.count.values()), "bytes": sum(
            len(c.sentence()) for v in self.count.values() for c in v)}


# ── fact store and knowledge graphs (structured; ingest the extraction) ──
class FactStore(System):
    """Extracted facts as memory items (Mem0-like): entity match, one hop, no task."""

    name = "factstore"
    hops = 1
    use_task = False
    honor_retractions = True
    keep_details = True
    min_count = 1               # ignore facts stated fewer times than this (falls back if nothing remains)

    def __init__(self) -> None:
        self.count: dict[Claim, int] = defaultdict(int)
        self.last: dict[Claim, int] = {}
        self.tickets: dict[Claim, list[str]] = defaultdict(list)
        self.tasks: dict[Claim, Counter] = defaultdict(Counter)
        self.adj: dict[str, set[Claim]] = defaultdict(set)
        self.recent: list[Claim] = []
        self.matcher = TaskMatcher()

    def observe(self, ev):
        x = ev.extracted
        self.matcher.observe(ev.task)
        for c in x.claims:
            self.count[c] += 1
            self.last[c] = ev.step
            self.adj[c.src].add(c)
            self.adj[c.tgt].add(c)
            self.recent.append(c)
            if self.use_task:
                self.tasks[c][ev.task] += 1
            if x.ticket and self.keep_details and c == x.ticket_claim:
                self.tickets[c].append(x.ticket)
        if self.honor_retractions:
            for c in x.retractions:
                self.count.pop(c, None)
                self.last.pop(c, None)
                self.tickets.pop(c, None)
                self.tasks.pop(c, None)
                self.adj[c.src].discard(c)
                self.adj[c.tgt].discard(c)

    def _task_match(self, c: Claim, q: Query) -> float:
        if not (self.use_task and q.task_text):
            return 0.0
        return max((self.matcher.match(q.task_text, t) for t in self.tasks.get(c, ())), default=0.0)

    def _select(self, linked: list[str], q: Query) -> list[Claim]:
        seen = self._walk(linked, self.min_count)
        if not seen and self.min_count > 1:
            seen = self._walk(linked, 1)
        tm = {c: self._task_match(c, q) for c in seen}
        if self.use_task and any(v >= TASK_MATCH for v in tm.values()):
            seen = {c: h for c, h in seen.items() if tm[c] >= TASK_MATCH}
        return sorted(seen, key=lambda c: (seen[c], -tm[c], -self.count[c], -self.last[c], c))

    def _walk(self, linked: list[str], min_count: int) -> dict[Claim, int]:
        seen: dict[Claim, int] = {}
        frontier, visited = set(linked), set(linked)
        for hop in range(1, self.hops + 1):
            nxt = set()
            for x in frontier:
                for c in self.adj.get(x, ()):
                    if c in self.count and c not in seen and self.count[c] >= min_count:
                        seen[c] = hop
                        nxt |= {c.src, c.tgt}
            frontier = nxt - visited
            visited |= nxt
        return seen

    def _line(self, c: Claim) -> str:
        tk = f" (ticket {', '.join(self.tickets[c])})" if self.tickets.get(c) else ""
        return c.sentence() + tk

    def query(self, q, linked, budget):
        ranked = self._select(linked, q)
        if not ranked:
            ranked = [c for c in reversed(self.recent) if c in self.count][:200]
        lines = [(self._line(c), c) for c in ranked]
        ctx = _fill(lines, budget)
        for c in ctx.claims:
            ctx.tickets |= set(self.tickets.get(c, ()))
        return ctx

    def stats(self):
        return {"items": len(self.count), "bytes": sum(len(self._line(c)) for c in self.count)}


class FactTask(FactStore):
    """The same fact store with per-fact task tags, two hops, task filter."""

    name = "fact_task"
    hops = 2
    use_task = True


class FactTaskS2(FactTask):
    """fact_task that ignores facts stated only once (the evidence rule a fact store can apply)."""

    name = "fact_task_s2"
    min_count = 2


class KGTemporal(FactStore):
    """Graph with invalidation of retracted edges and provenance; two hops (Zep/GraphRAG-like)."""

    name = "kg_temporal"
    hops = 2


class KGStatic(FactStore):
    """Append-only graph, two hops: retractions are never applied."""

    name = "kg_static"
    hops = 2
    honor_retractions = False


class StateDoc(FactStore):
    """The whole current state as one document, this task's facts first."""

    name = "state_doc"
    use_task = True
    hops = 0

    def query(self, q, linked, budget):
        order = sorted(self.count, key=lambda c: (-self._task_match(c, q), -self.count[c], -self.last[c], c))
        ctx = _fill([(self._line(c), c) for c in order], budget)
        for c in ctx.claims:
            ctx.tickets |= set(self.tickets.get(c, ()))
        return ctx


# ── World 0 ─────────────────────────────────────────────────────────────
SIZES = (80, 60, 50, 40, 30, 24, 18, 14, 11, 9, 7, 5, 4, 3, 2, 1)


def _tuned() -> dict:
    try:
        with open(TUNED_PATH) as fh:
            return json.load(fh)
    except OSError:
        return {}


class World0(System):
    """World 0 as its facade ships (depth 2, no reflect) with ``Projection.render()``."""

    name = "world0"
    depth = 2
    reflect_every: int | None = None
    sustained = False
    use_task = True
    ingest_task = True
    compact = False
    use_cache = True
    min_support = 1             # compact render: only claims stated at least this often (fallback: all)

    def __init__(self, shared: "World0 | None" = None) -> None:
        from world0 import World

        self.shared = shared
        self._cache: dict = {}
        self._cache_step = -1
        if shared is not None:
            self.world, self.path = shared.world, shared.path
            return
        self._dir = tempfile.mkdtemp(prefix="longrun_w0_")
        self.path = os.path.join(self._dir, "w.sqlite")
        self.world = World(store_path=self.path, auto_reflect_every=self.reflect_every,
                           sustained_attention=self.sustained)

    def observe(self, ev):
        from world0 import Observation

        if self.shared is not None:
            return
        x = ev.extracted
        self.world.ingest(Observation(
            concepts=list(x.concepts),
            relations=[(c.src, c.tgt, c.rel) for c in x.claims],
            # "Correction: X no longer depends on Y" is a revision of the
            # world, not evidence the claim was wrong: withdraw it.
            retracted_relations=[(c.src, c.tgt, c.rel) for c in x.retractions],
            task=ev.task if self.ingest_task else "", source=f"step{ev.step}",
        ))

    def _render(self, proj) -> str:
        if not self.compact:
            return proj.render()
        names = {c.id: c.name for c in proj.concepts}
        lines, mentioned = [], set()
        rels = [r for r in proj.relations if r.is_explicit and _REL.get(r.semantic_relation)]
        strong = [r for r in rels if r.probability_observation_count + 1 >= self.min_support]
        for r in sorted(strong or rels, key=lambda r: (-r.probability, r.id)):
            cl = Claim.make(names[r.source_id], _REL[r.semantic_relation], names[r.target_id])
            lines.append(f"{cl.sentence()} (belief {r.probability:.2f})")
            mentioned |= {r.source_id, r.target_id}
        rest = [c.name for c in proj.concepts if c.id not in mentioned]
        if rest:
            lines.append("Also relevant: " + ", ".join(rest) + ".")
        return "\n".join(lines)

    def query(self, q, linked, budget):
        task = q.task_text if self.use_task else ""
        if q.step != self._cache_step or not self.use_cache:
            self._cache, self._cache_step = {}, q.step
        best = None
        for n in SIZES:
            key = (n, tuple(linked))
            if key not in self._cache:
                proj = self.world.project(linked, task=task, max_concepts=n, max_depth=self.depth)
                text = self._render(proj)
                self._cache[key] = (text, est_tokens(text))
            text, cost = self._cache[key]
            if cost <= budget:
                best = text
                break
        if best is None:
            return Context()
        claims, concepts, beliefs = (parse_compact if self.compact else parse_shipped)(best)
        return Context(best, claims, concepts, set(), beliefs)

    def stats(self):
        if self.shared is not None:
            return {"shares": "world0"}
        import sqlite3

        try:
            con = sqlite3.connect(self.path)
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()
        except sqlite3.Error:
            pass
        size = 0
        for suffix in ("", "-wal"):
            try:
                size += os.path.getsize(self.path + suffix)
            except OSError:
                pass
        return {"items": len(self.world.concepts.all()) + len(self.world.relations.all()), "bytes": size}

    def close(self):
        if self.shared is not None:
            return
        try:
            self.world.close()
        except Exception:
            pass


_REL = {"dependence": "depends_on", "inclusion": "contains", "conflict": "conflict", "enables": "enables"}


class World0Compact(World0):
    """The same projection rendered compactly (typed claims with belief).

    ``Projection.render()`` spends most tokens on concept ids, maturity and
    confidence annotations, co-occurrence edges and an attention section and
    prints at most ten relations; this renderer isolates how much of World 0's
    cost is the render and how much the projection.
    """

    name = "world0_compact"
    compact = True


class World0Tuned(World0Compact):
    """Compact render with depth / reflect chosen on dev seeds (tuned.json), never on test seeds."""

    name = "world0_tuned"

    def __init__(self, shared=None):
        cfg = _tuned()
        self.depth = int(cfg.get("depth", 2))
        self.reflect_every = cfg.get("reflect_every")
        if self.reflect_every:
            shared = None  # own world: reflect changes the state
        super().__init__(shared)


class World0TunedS2(World0Tuned):
    """world0_tuned that shows only relations stated at least twice (uses World 0's own evidence counts)."""

    name = "world0_tuned_s2"
    min_support = 2


class World0Reflect(World0Compact):
    name = "world0_reflect"
    reflect_every = 25


class World0Focus(World0Compact):
    name = "world0_focus"
    sustained = True


class World0NoTask(World0Compact):
    """No task label at ingest or at query."""

    name = "world0_notask"
    use_task = False
    ingest_task = False


class World0Depth1(World0Compact):
    name = "world0_depth1"
    depth = 1


class World0Depth3(World0Compact):
    name = "world0_depth3"
    depth = 3


class World0Custom(World0Compact):
    """Tuning harness: ``w0-d{depth}-r{reflect or 0}`` (compact render)."""

    def __init__(self, depth: int, reflect: int, shared=None):
        self.depth, self.reflect_every = depth, reflect or None
        self.name = f"w0-d{depth}-r{reflect}"
        super().__init__(shared if not reflect else None)


ALL_SYSTEMS: dict[str, type[System]] = {
    cls.name: cls for cls in (
        NoMemory, Window, FullContext, FullContext32k, FullContext64k, RAG, RAGRecency, SummaryBuffer, SummaryTask,
        FactStore, FactTask, FactTaskS2, KGStatic, KGTemporal, StateDoc,
        World0, World0Compact, World0Tuned, World0TunedS2, World0Reflect, World0Focus, World0NoTask,
        World0Depth1, World0Depth3,
    )
}
SHARES_WORLD0 = {"world0_compact", "world0_depth1", "world0_depth3", "world0_tuned"}


def make_system(name: str, instances: dict[str, System]) -> System:
    """Instantiate ``name``; variants that only differ at query time share the world0 state."""
    if name.startswith("w0-d"):
        d, r = name[4:].split("-r")
        return World0Custom(int(d), int(r), shared=instances.get("world0") if int(r) == 0 else None)
    cls = ALL_SYSTEMS[name]
    if name in SHARES_WORLD0 and "world0" in instances:
        return cls(shared=instances["world0"])
    return cls()
