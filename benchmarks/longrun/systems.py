"""Memory strategies compared by LongRun, behind one interface.

Every system sees the same event stream through ``observe`` and, for a
query, must return a ``Context`` that fits ``budget`` tokens.  The harness
hands every system the same *linked entities* (the known concept names that
occur in the query text), so nobody is disadvantaged by entity linking, and
counts context size with ``tokens.est_tokens``.

What each system stores is what a real deployment of that strategy stores;
the ``claims`` / ``tickets`` fields of a ``Context`` are what an ideal
reader could extract from the text the system produced (the LLM-reader
stage of the study checks that assumption with real readers).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

from benchmarks.longrun.tokens import est_tokens
from benchmarks.longrun.worldgen import Claim, Event, Query

SEM2REL = {"dependence": "depends_on", "inclusion": "contains",
           "conflict": "conflict", "enables": "enables"}


@dataclass
class Context:
    text: str = ""
    claims: set[Claim] = field(default_factory=set)
    concepts: set[str] = field(default_factory=set)
    tickets: set[str] = field(default_factory=set)
    beliefs: dict[Claim, float] = field(default_factory=dict)  # only where shown
    contested: set[frozenset[str]] = field(default_factory=set)

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
    out = []
    for w in text.lower().replace(",", " ").replace(".", " ").replace(":", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").split():
        out.append(w)
    return out


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


# ── no memory ────────────────────────────────────────────────────────────
class NoMemory(System):
    name = "none"

    def observe(self, ev): pass
    def query(self, q, linked, budget): return Context()


# ── direct context: the last events that fit, or everything ─────────────
class Window(System):
    name = "window"

    def __init__(self) -> None:
        self.events: list[Event] = []

    def observe(self, ev): self.events.append(ev)

    def query(self, q, linked, budget):
        chosen, used = [], 0
        for ev in reversed(self.events):
            if used + ev.tokens + 1 > budget:
                break
            chosen.append(ev)
            used += ev.tokens + 1
        return _from_events(chosen)

    def stats(self):
        return {"items": len(self.events), "bytes": sum(len(e.text) for e in self.events)}


class FullContext(Window):
    name = "full_context"
    unbounded = True

    def query(self, q, linked, budget):
        return _from_events(self.events)


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
    """Generative-Agents style retrieval: relevance + recency + importance."""

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


# ── summary buffer: recent window + frequency digest of older claims ─────
class SummaryBuffer(System):
    """ConversationSummaryBuffer with an extractive (frequency) summary.

    The summary is query-independent, as a rolling summary is; it keeps the
    most frequently stated claims, honours retractions, and gets the half of
    the budget that the recent-events window does not use.
    """

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
        half = budget // 2
        chosen, used = [], 0
        for ev in reversed(self.events):
            if used + ev.tokens + 1 > half:
                break
            chosen.append(ev)
            used += ev.tokens + 1
        ctx = _from_events(chosen)
        lines, room = [], budget - used - 2
        for c in sorted(self.count, key=lambda c: (-self.count[c], -self.last[c], c)):
            line = f"{c.sentence()} (x{self.count[c]})"
            w = est_tokens(line) + 1
            if w > room:
                break
            lines.append(line)
            room -= w
            ctx.claims.add(c)
            ctx.concepts |= {c.src, c.tgt}
        ctx.text = "Summary of earlier work:\n" + "\n".join(lines) + "\n" + ctx.text
        return ctx

    def stats(self):
        return {"items": len(self.events) + len(self.count),
                "bytes": sum(len(e.text) for e in self.events)}


# ── fact store (Mem0-like) and knowledge graphs ──────────────────────────
class FactStore(System):
    """Extracted facts as memory items; entity match; retractions delete."""

    name = "factstore"
    hops = 1
    honor_retractions = True
    keep_details = True

    def __init__(self) -> None:
        self.count: dict[Claim, int] = defaultdict(int)
        self.last: dict[Claim, int] = {}
        self.tickets: dict[Claim, list[str]] = defaultdict(list)
        self.adj: dict[str, set[Claim]] = defaultdict(set)
        self.recent: list[Claim] = []
        self.n_events = 0

    def observe(self, ev):
        self.n_events += 1
        for c in ev.claims:
            self.count[c] += 1
            self.last[c] = ev.step
            self.adj[c.src].add(c)
            self.adj[c.tgt].add(c)
            self.recent.append(c)
            if ev.ticket and self.keep_details and c == ev.ticket_claim:
                self.tickets[c].append(ev.ticket)
        if self.honor_retractions:
            for c in ev.retractions:
                self.count.pop(c, None)
                self.last.pop(c, None)
                self.tickets.pop(c, None)
                self.adj[c.src].discard(c)
                self.adj[c.tgt].discard(c)

    def _select(self, linked: list[str]) -> list[tuple[int, Claim]]:
        """(rank key, claim) pairs, best first."""
        seen: dict[Claim, int] = {}
        frontier = set(linked)
        visited = set(linked)
        for hop in range(1, self.hops + 1):
            nxt = set()
            for x in frontier:
                for c in self.adj.get(x, ()):
                    if c in self.count and c not in seen:
                        seen[c] = hop
                        nxt |= {c.src, c.tgt}
            frontier = nxt - visited
            visited |= nxt
        return sorted(((h, c) for c, h in seen.items()),
                      key=lambda hc: (hc[0], -self.count[hc[1]], -self.last[hc[1]], hc[1]))

    def _line(self, c: Claim) -> str:
        tk = f" (ticket {', '.join(self.tickets[c])})" if self.tickets.get(c) else ""
        return c.sentence() + tk

    def query(self, q, linked, budget):
        ranked = [c for _, c in self._select(linked)]
        if not ranked:
            ranked = [c for c in reversed(self.recent) if c in self.count][:200]
        ctx, room, lines = Context(), budget, []
        for c in ranked:
            line = self._line(c)
            w = est_tokens(line) + 1
            if w > room:
                continue
            lines.append(line)
            room -= w
            ctx.claims.add(c)
            ctx.concepts |= {c.src, c.tgt}
            ctx.tickets |= set(self.tickets.get(c, ()))
        ctx.text = "\n".join(lines)
        return ctx

    def stats(self):
        return {"items": len(self.count),
                "bytes": sum(len(self._line(c)) for c in self.count)}


class KGStatic(FactStore):
    """Append-only graph, two-hop expansion (GraphRAG-style static graph)."""

    name = "kg_static"
    hops = 2
    honor_retractions = False
    keep_details = False


class KGTemporal(FactStore):
    """Graph with invalidation of retracted edges, two-hop expansion."""

    name = "kg_temporal"
    hops = 2
    honor_retractions = True
    keep_details = False


# ── World 0 ─────────────────────────────────────────────────────────────
class World0(System):
    name = "world0"
    reflect_every: int | None = None
    sustained = False
    use_task = True
    depth = 3

    def __init__(self, shared: "World0 | None" = None) -> None:
        from world0 import World

        self.shared = shared
        self._cache: dict = {}
        self._cache_step = -1
        if shared is not None:
            # Same hidden state, different query behaviour or rendering.
            self.world, self.path, self.n = shared.world, shared.path, 0
            return
        self._dir = tempfile.mkdtemp(prefix="longrun_w0_")
        self.path = os.path.join(self._dir, "w.sqlite")
        self.world = World(store_path=self.path, auto_reflect_every=self.reflect_every,
                           sustained_attention=self.sustained)
        self.n = 0

    def observe(self, ev):
        from world0 import Observation

        if self.shared is not None:
            return
        self.n += 1
        self.world.ingest(Observation(
            concepts=list(ev.concepts),
            relations=[(c.src, c.tgt, c.rel) for c in ev.claims],
            contradicted_relations=[(c.src, c.tgt, c.rel) for c in ev.retractions],
            task=ev.task, source=f"step{ev.step}",
        ))

    def _claims(self, proj) -> tuple[set[Claim], dict[Claim, float], set[frozenset[str]]]:
        names = {c.id: c.name for c in proj.concepts}
        claims, beliefs = set(), {}
        for r in proj.relations:
            rel = SEM2REL.get(r.semantic_relation)
            if rel is None or not r.is_explicit:
                continue
            cl = Claim.make(names[r.source_id], rel, names[r.target_id])
            claims.add(cl)
            beliefs[cl] = max(beliefs.get(cl, 0.0), r.probability)
        contested = {frozenset((c.source_id, c.target_id)) for c in proj.epistemic.contested}
        return claims, beliefs, contested

    def _render(self, proj) -> str:
        return proj.render()

    def query(self, q, linked, budget):
        task = q.task_text if self.use_task else ""
        if q.step != self._cache_step:
            self._cache, self._cache_step = {}, q.step
        best = None
        for n in (40, 30, 24, 18, 14, 11, 9, 7, 5, 4, 3, 2, 1):
            key = (n, tuple(linked))
            if key not in self._cache:
                proj = self.world.project(linked, task=task, max_concepts=n, max_depth=self.depth)
                text = self._render(proj)
                self._cache[key] = (proj, text, est_tokens(text))
            proj, text, cost = self._cache[key]
            if cost <= budget:
                best = (proj, text)
                break
        if best is None:
            return Context()
        proj, text = best
        claims, beliefs, contested = self._claims(proj)
        return Context(text, claims, {c.name for c in proj.concepts}, set(), beliefs, contested)

    def stats(self):
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
        return {"items": len(self.world.concepts.all()) + len(self.world.relations.all()),
                "bytes": size}

    def close(self):
        if self.shared is not None:
            return
        try:
            self.world.close()
        except Exception:
            pass


class World0Compact(World0):
    """The same projection, rendered compactly (typed relations with belief).

    ``Projection.render()`` spends most of its tokens on concept ids, maturity
    and confidence annotations, co-occurrence edges and an attention section.
    This renderer is what a prompt integration that only wants the typed
    structure would write; it isolates how much of World 0's cost is the
    render and how much is the projection.
    """

    name = "world0_compact"

    def _render(self, proj) -> str:
        names = {c.id: c.name for c in proj.concepts}
        lines, mentioned = [], set()
        rels = [r for r in proj.relations
                if r.is_explicit and SEM2REL.get(r.semantic_relation)]
        for r in sorted(rels, key=lambda r: (-r.probability, r.id)):
            cl = Claim.make(names[r.source_id], SEM2REL[r.semantic_relation], names[r.target_id])
            lines.append(f"{cl.sentence()} (belief {r.probability:.2f})")
            mentioned |= {r.source_id, r.target_id}
        rest = [c.name for c in proj.concepts if c.id not in mentioned]
        if rest:
            lines.append("Also relevant: " + ", ".join(rest) + ".")
        return "\n".join(lines)


class World0Reflect(World0):
    name = "world0_reflect"
    reflect_every = 25


class World0Focus(World0):
    name = "world0_focus"
    sustained = True


class World0NoTask(World0):
    name = "world0_notask"
    use_task = False


class World0Shallow(World0):
    name = "world0_depth1"
    depth = 1


ALL_SYSTEMS: dict[str, type[System]] = {
    cls.name: cls for cls in (
        NoMemory, Window, FullContext, RAG, RAGRecency, SummaryBuffer,
        FactStore, KGStatic, KGTemporal, World0, World0Compact, World0Reflect, World0Focus,
        World0NoTask, World0Shallow,
    )
}
