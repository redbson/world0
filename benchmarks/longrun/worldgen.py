"""Synthetic long-horizon Agent stream with a hidden, evolving ground truth.

A hidden world of ``n_domains`` domains, each a small typed concept graph
(dependency layers, containment, conflicts), shares a few *bridge* concepts
between domains (the same name means different things in each).  An Agent
works on one domain at a time, in runs, and each event states a few
relations of the current truth plus some co-mentioned concepts; some events
are chatter (noise); at scheduled moments a relation is revised (the old
claim is retracted and a new one stated); some events carry a ticket id (an
episodic detail that is not a concept).

All names are invented (pseudo-words), so no system can guess a claim from
prior knowledge: whatever it answers, it answers from what it was given.

The generator also emits *queries* at checkpoints.  Every gold set is
restricted to what the Agent has actually been told so far (a relation never
stated cannot be recalled by any system) and to the current truth (a
retracted relation is stale, never gold).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

from benchmarks.longrun.tokens import FILLER_WORDS, est_tokens

SYMMETRIC = {"conflict"}
REL_PHRASE = {
    "depends_on": "depends on",
    "contains": "contains",
    "conflict": "conflicts with",
    "enables": "enables",
}
HEADS = [
    "service", "cache", "queue", "index", "gateway", "planner", "store",
    "router", "monitor", "loader", "encoder", "compiler", "registry",
    "scheduler", "validator", "resolver", "broker", "ledger", "sampler",
    "allocator",
]
_ONSETS = list("vkzrmtdlnsbghpfcwj")
_VOWELS = list("aeiou")
_CODAS = ["l", "n", "r", "x", "m", "th", "s", "k"]


class Claim(NamedTuple):
    src: str
    rel: str
    tgt: str

    @staticmethod
    def make(src: str, rel: str, tgt: str) -> "Claim":
        """Symmetric relations are stored with sorted endpoints."""
        if rel in SYMMETRIC and tgt < src:
            src, tgt = tgt, src
        return Claim(src, rel, tgt)

    def sentence(self) -> str:
        # A real extractor may choose a label outside the benchmark's four.
        return f"{self.src} {REL_PHRASE.get(self.rel, self.rel.replace('_', ' '))} {self.tgt}"


@dataclass
class GenConfig:
    seed: int = 0
    horizon: int = 1000
    n_domains: int = 8
    concepts_per_domain: int = 22
    n_bridge: int = 6
    layers: int = 5
    noise_rate: float = 0.25
    detail_rate: float = 0.15
    focus_run_mean: int = 25
    dormant_domains: int = 2       # active only in the first 20 % of the stream
    n_revisions: int | None = None  # default: max(4, horizon // 100)
    query_every: int = 10
    warmup: int = 80
    task_mode: str = "exact"        # how the query names its task: exact | none | wrong
    growth: bool = False            # domains appear over time instead of all existing from the start
    extract_p: float = 0.0          # extractor error level (drop p, wrong p/2, spurious p, missed retraction p)
    extract_mode: str = "sim"       # "sim": gold + injected errors (extract_p); "llm": cached real LLM output (llm_extract.py)
    llm_version: str = ""           # extract_mode="llm": "" = current prompt's cache, else an older one ("prompt_v1")
    text_style: str = "template"    # "template": one canonical sentence per claim; "natural": paraphrase.py (same stream, other words)
    allow_pair_collision: bool = False  # two typed claims on one ordered pair (revisions can still create one)
    popularity_skew: float = 0.8    # Zipf exponent over claims: rarely-stated claims exist
    verbosity: int = 30             # filler words per event (real transcripts are mostly not concepts)


@dataclass
class Extraction:
    """What an extractor emitted for an event (structured systems ingest this)."""

    concepts: list[str]
    claims: list["Claim"]
    retractions: list["Claim"]
    ticket: str | None
    ticket_claim: "Claim | None"
    # the extractor's "this is wrong" claims (a subset of ``retractions``,
    # which is all a baseline can act on); World 0 takes them as contradictions
    contradictions: list["Claim"] = field(default_factory=list)


@dataclass
class Event:
    step: int
    kind: str                       # "task" | "noise"
    domain: str | None
    task: str
    concepts: list[str]
    claims: list[Claim]
    retractions: list[Claim]
    ticket: str | None
    ticket_claim: Claim | None
    text: str
    extracted: Extraction | None = None

    @property
    def tokens(self) -> int:
        return est_tokens(self.text)


@dataclass
class Query:
    step: int
    kind: str                       # focus | chain | bridge | stale | detail
    domain: str
    task_text: str
    text: str
    entry: list[str]
    gold_claims: set[Claim] = field(default_factory=set)
    gold_concepts: set[str] = field(default_factory=set)
    gold_answer: set[str] = field(default_factory=set)    # chain: reachable concepts
    wrong_claims: set[Claim] = field(default_factory=set)  # other-sense / other-domain
    stale_claims: set[Claim] = field(default_factory=set)
    current_claims: set[Claim] = field(default_factory=set)  # stale: the revised truth
    gold_ticket: str | None = None
    gold_by_radius: dict[int, set[Claim]] = field(default_factory=dict)   # focus: hop-ball gold at radius 1..3
    age: int = 0                    # steps since the gold information was last stated
    gap: int = 0                    # steps since the domain was last worked on


def _pseudo_word(rng: random.Random) -> str:
    syll = rng.choice([2, 3])
    body = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(syll))
    return (body + rng.choice(_CODAS)).capitalize()


class _Namer:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def word(self) -> str:
        while True:
            w = _pseudo_word(self.rng)
            if w not in self.used:
                self.used.add(w)
                return w

    def concept(self) -> str:
        return f"{self.word()} {self.rng.choice(HEADS)}"


def _layer_sizes(n: int, layers: int) -> list[int]:
    profile = [3, 5, 6, 5, 3, 2, 2][:layers]
    total = sum(profile)
    sizes = [max(1, round(p * n / total)) for p in profile]
    while sum(sizes) > n:
        sizes[sizes.index(max(sizes))] -= 1
    while sum(sizes) < n:
        sizes[sizes.index(min(sizes))] += 1
    return sizes


class HiddenWorld:
    """The hidden truth: domains, their claims, bridges, revisions."""

    def __init__(self, cfg: GenConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        namer = _Namer(rng)
        self.domains = [namer.word() for _ in range(cfg.n_domains)]
        self.bridges = [namer.concept() for _ in range(cfg.n_bridge)]
        self.bridge_domains: dict[str, list[str]] = {}
        for b in self.bridges:
            self.bridge_domains[b] = rng.sample(self.domains, 2)
        self.layer_of: dict[tuple[str, str], int] = {}
        self.concepts: dict[str, list[str]] = {}
        self.truth: dict[str, set[Claim]] = {}
        self.popularity: dict[Claim, float] = {}
        for d in self.domains:
            self._build_domain(d, namer)
        self.all_concepts = sorted({c for cs in self.concepts.values() for c in cs})

    def _build_domain(self, d: str, namer: _Namer) -> None:
        cfg, rng = self.cfg, self.rng
        sizes = _layer_sizes(cfg.concepts_per_domain, cfg.layers)
        layers: list[list[str]] = [[] for _ in sizes]
        mine = [b for b in self.bridges if d in self.bridge_domains[b]]
        slots = [(li, k) for li, s in enumerate(sizes) for k in range(s)]
        rng.shuffle(slots)
        bridge_slots = {slot: b for slot, b in zip(slots[: len(mine)], mine)}
        for li, s in enumerate(sizes):
            for k in range(s):
                layers[li].append(bridge_slots.get((li, k)) or namer.concept())
        claims: set[Claim] = set()
        for li in range(len(layers) - 1):
            for c in layers[li]:
                for dep in rng.sample(layers[li + 1], min(rng.choice([1, 2, 2]), len(layers[li + 1]))):
                    claims.add(Claim.make(c, "depends_on", dep))
            for dep in layers[li + 1]:
                if not any(cl.tgt == dep for cl in claims):
                    claims.add(Claim.make(rng.choice(layers[li]), "depends_on", dep))
        def free(a: str, b: str) -> bool:
            return cfg.allow_pair_collision or not any({cl.src, cl.tgt} == {a, b} for cl in claims)

        for top in layers[0]:
            for sub in rng.sample(layers[1], min(2, len(layers[1]))):
                if free(top, sub):
                    claims.add(Claim.make(top, "contains", sub))
        for li in range(1, len(layers)):
            for _ in range(2):
                a, b = rng.sample(layers[li], 2) if len(layers[li]) >= 2 else (None, None)
                if a and free(a, b):
                    claims.add(Claim.make(a, "conflict", b))
        for _ in range(3):
            li = rng.randrange(len(layers) - 1)
            a, b = rng.choice(layers[li]), rng.choice(layers[li + 1])
            if free(a, b):
                claims.add(Claim.make(a, "enables", b))
        self.concepts[d] = [c for layer in layers for c in layer]
        for li, layer in enumerate(layers):
            for c in layer:
                self.layer_of[(d, c)] = li
        self.truth[d] = claims
        ordered = sorted(claims)
        rng.shuffle(ordered)
        for rank, cl in enumerate(ordered):
            self.popularity[cl] = 1.0 / (rank + 1) ** cfg.popularity_skew


class Stream:
    """Deterministic (seeded) event + query stream over a hidden world."""

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg
        self._known: dict[str, str] | None = None  # normalised → world concept name (llm extraction)
        self.rng = random.Random(cfg.seed)
        self.world = HiddenWorld(cfg, self.rng)
        self.stated_at: dict[Claim, list[int]] = {}
        self.tickets: dict[Claim, list[tuple[int, str]]] = {}
        self.last_active: dict[str, int] = {}
        self.revisions: list[tuple[int, str, Claim, Claim]] = []  # (step, domain, old, new)
        self.junk = 0
        self._ticket_ids: set[str] = set()
        self._revised: set[tuple[str, str]] = set()
        n = cfg.n_domains
        self.birth = {d: (int(i * 0.6 * cfg.horizon / n) if cfg.growth else 0)
                      for i, d in enumerate(self.world.domains)}
        # Domains 1..k are popular early and then go quiet (dormant).
        self.dormant = set(self.world.domains[1:1 + cfg.dormant_domains]) if cfg.dormant_domains else set()

    def state_tokens(self) -> int:
        """Tokens of the known current state (every told, still-true claim as a sentence)."""
        return sum(est_tokens(c.sentence()) + 1 for d in self.world.domains for c in self.known(d))

    # ── naming ────────────────────────────────────────────────────────
    def task_label(self, d: str) -> str:
        return f"{d} work"

    def task_text(self, d: str) -> str:
        mode = self.cfg.task_mode
        if mode == "none":
            return ""
        if mode == "wrong":
            others = [x for x in self.world.domains if x != d]
            return self.task_label(self.rng.choice(others))
        return self.task_label(d)

    # ── knowledge state ───────────────────────────────────────────────
    def known(self, d: str) -> set[Claim]:
        return {c for c in self.world.truth[d] if c in self.stated_at}

    def _closure(self, x: str, claims: set[Claim]) -> tuple[set[str], set[Claim]]:
        seen, used, frontier = set(), set(), [x]
        deps: dict[str, list[Claim]] = {}
        for c in claims:
            if c.rel == "depends_on":
                deps.setdefault(c.src, []).append(c)
        while frontier:
            cur = frontier.pop()
            for c in deps.get(cur, []):
                used.add(c)
                if c.tgt not in seen:
                    seen.add(c.tgt)
                    frontier.append(c.tgt)
        return seen, used

    def _neighborhood(self, entry: list[str], claims: set[Claim], hops: int = 2) -> set[Claim]:
        frontier, seen, out = set(entry), set(entry), set()
        for _ in range(hops):
            nxt = set()
            for c in claims:
                if c.src in frontier or c.tgt in frontier:
                    out.add(c)
                    nxt |= {c.src, c.tgt}
            frontier = nxt - seen
            seen |= nxt
        return out

    # ── event generation ──────────────────────────────────────────────
    def _pick_domain(self, step: int) -> str:
        cfg = self.cfg
        pool = [d for d in self.world.domains if self.birth[d] <= step]
        if step > 0.2 * cfg.horizon:
            pool = [d for d in pool if d not in self.dormant] or pool
        weights = [1.0 / (self.world.domains.index(d) + 1) ** 0.5 for d in pool]
        return self.rng.choices(pool, weights=weights)[0]

    def _filler(self) -> str:
        n = self.cfg.verbosity
        if n <= 0:
            return ""
        start = self.rng.randrange(len(FILLER_WORDS))
        words = [FILLER_WORDS[(start + i) % len(FILLER_WORDS)] for i in range(n)]
        return "Notes: " + " ".join(words) + "."

    def _task_event(self, step: int, d: str, retire: Claim | None, new: Claim | None) -> Event:
        rng = self.rng
        truth = sorted(self.world.truth[d])
        weights = [self.world.popularity[c] for c in truth]
        stated = [new] if new else [rng.choices(truth, weights=weights)[0]]
        ends = {stated[0].src, stated[0].tgt}
        near = [c for c in truth if c not in stated and (c.src in ends or c.tgt in ends)]
        stated += rng.sample(near, min(len(near), rng.choice([0, 1, 2])))
        concepts: list[str] = []
        for c in stated + ([retire] if retire else []):
            for x in (c.src, c.tgt):
                if x not in concepts:
                    concepts.append(x)
        extras = [x for x in self.world.concepts[d] if x not in concepts]
        for x in rng.sample(extras, min(len(extras), rng.choice([0, 1, 2]))):
            concepts.append(x)
        ticket = ticket_claim = None
        if rng.random() < self.cfg.detail_rate:
            while True:
                ticket = f"T{rng.randint(1000, 9999)}"
                if ticket not in self._ticket_ids:
                    self._ticket_ids.add(ticket)
                    break
            ticket_claim = stated[0]
        rest = [x for x in concepts if x not in {y for c in stated + ([retire] if retire else []) for y in (c.src, c.tgt)}]
        filler = self._filler()
        if self.cfg.text_style == "natural":
            text = self._natural_task_text(step, d, stated, retire, rest, ticket, filler)
        else:
            parts = [f"[step {step}] Working on {self.task_label(d)}:"]
            for i, c in enumerate(stated):
                tag = f" (ticket {ticket})" if ticket and i == 0 else ""
                parts.append(c.sentence() + tag + ".")
            if retire:
                parts.append(f"Correction: {retire.src} no longer {REL_PHRASE[retire.rel]} {retire.tgt}.")
            if rest:
                parts.append("Also touched: " + ", ".join(rest) + ".")
            if filler:
                parts.append(filler)
            text = " ".join(parts)
        return Event(step, "task", d, self.task_label(d), concepts, stated,
                     [retire] if retire else [], ticket, ticket_claim, text)

    def _natural_task_text(self, step: int, d: str, stated: list[Claim], retire: Claim | None,
                           rest: list[str], ticket: str | None, filler: str) -> str:
        """The same event in varied wording; draws only from the event's own rng."""
        from benchmarks.longrun.paraphrase import Realiser, event_rng

        rng = event_rng(self.cfg.seed, step)
        negated = None
        if rng.random() < 0.25:
            # a pair stated as *not* holding: it must never be a claim of the stream
            for _ in range(10):
                a, b = rng.sample(self.world.concepts[d], 2)
                if self._deniable(a, b, stated, retire):
                    negated = (a, b)
                    break
        return Realiser(rng).task_text(step, self.task_label(d), stated, retire, rest, ticket, filler, negated)

    def _deniable(self, a: str, b: str, stated: list[Claim], retire: Claim | None) -> bool:
        """No claim links a and b in any domain, now, before, in this event, or by a later revision.

        Bridge concepts are in several domains; a revision adds a dependency
        from a concept to one in the next layer (``_revision_candidate``), so
        such a pair could become true later and is never denied.
        """
        pair = {a, b}
        if any({c.src, c.tgt} == pair for c in [*stated, *([retire] if retire else [])]):
            return False
        if any({c.src, c.tgt} == pair for c in self.stated_at):
            return False
        last = self.cfg.layers - 2
        for d in self.world.domains:
            if any({c.src, c.tgt} == pair for c in self.world.truth[d]):
                return False
            la, lb = self.world.layer_of.get((d, a)), self.world.layer_of.get((d, b))
            if la is not None and lb is not None and (
                    (lb == la + 1 and la < last) or (la == lb + 1 and lb < last)):
                return False
        return True

    def _noise_event(self, step: int) -> Event:
        rng = self.rng
        pool = self.world.all_concepts
        concepts = rng.sample(pool, rng.choice([2, 3, 4]))
        if rng.random() < 0.5:
            self.junk += 1
            concepts.append(f"{_pseudo_word(rng)} scratch{self.junk}")
        filler = self._filler()
        if self.cfg.text_style == "natural":
            from benchmarks.longrun.paraphrase import Realiser, event_rng

            text = Realiser(event_rng(self.cfg.seed, step)).noise_text(step, concepts, filler)
        else:
            text = f"[step {step}] Chatter about " + ", ".join(concepts) + "."
            if filler:
                text += " " + filler
        return Event(step, "noise", None, "misc chatter", concepts, [], [], None, None, text)

    def events(self) -> Iterator[tuple[Event, list[Query]]]:
        cfg, rng = self.cfg, self.rng
        n_rev = cfg.n_revisions if cfg.n_revisions is not None else max(4, cfg.horizon // 100)
        rev_steps = {int(cfg.horizon * (0.3 + 0.5 * (i + 0.5) / n_rev)) for i in range(n_rev)}
        d, remaining = self._pick_domain(0), 0
        for step in range(cfg.horizon):
            if remaining <= 0 or (step > 0.2 * cfg.horizon and d in self.dormant):
                d = self._pick_domain(step)
                remaining = max(3, int(rng.expovariate(1.0 / cfg.focus_run_mean)))
            remaining -= 1
            retire = new = None
            if step in rev_steps:
                cand = self._revision_candidate(d)
                if cand:
                    retire, new = cand
            if retire is None and rng.random() < cfg.noise_rate:
                ev = self._noise_event(step)
            else:
                ev = self._task_event(step, d, retire, new)
                self.last_active[d] = step
            if retire and new:
                self.world.truth[d].discard(retire)
                self.world.truth[d].add(new)
                self.world.popularity[new] = self.world.popularity.get(retire, 0.2)
                self.revisions.append((step, d, retire, new))
            for c in ev.claims:
                self.stated_at.setdefault(c, []).append(step)
            if ev.ticket and ev.ticket_claim:
                self.tickets.setdefault(ev.ticket_claim, []).append((step, ev.ticket))
            ev.extracted = self._extract(ev)
            qs: list[Query] = []
            if step >= cfg.warmup and (step + 1) % cfg.query_every == 0:
                q = self._query(step)
                if q:
                    qs.append(q)
            yield ev, qs

    def _extract(self, ev: Event) -> Extraction:
        """The extractor's view of the event; error rate ``extract_p`` (own rng stream)."""
        if self.cfg.extract_mode == "llm":
            from benchmarks.longrun import llm_extract

            raw = llm_extract.load_cache(self.cfg.seed, self.cfg.horizon, self.cfg.text_style,
                                         self.cfg.llm_version)[str(ev.step)]
            if self._known is None:
                self._known = {llm_extract._key(c): c for c in self.world.all_concepts}
            return llm_extract.to_extraction(raw, ev, self._known)
        p = self.cfg.extract_p
        if p <= 0 or ev.kind != "task":
            return Extraction(list(ev.concepts), list(ev.claims), list(ev.retractions), ev.ticket, ev.ticket_claim)
        rng = random.Random(self.cfg.seed * 1_000_003 + ev.step)
        rels = ["depends_on", "contains", "conflict", "enables"]
        claims: list[Claim] = []
        for c in ev.claims:
            r = rng.random()
            if r < p:
                continue
            if r < p + p / 2:
                claims.append(Claim.make(c.src, rng.choice([x for x in rels if x != c.rel]), c.tgt))
            else:
                claims.append(c)
        if rng.random() < p and ev.domain:
            a, b = rng.sample(self.world.concepts[ev.domain], 2)
            claims.append(Claim.make(a, rng.choice(rels), b))
        retractions = [c for c in ev.retractions if rng.random() >= p]
        concepts = list(ev.concepts)
        for c in claims:
            for x in (c.src, c.tgt):
                if x not in concepts:
                    concepts.append(x)
        keep = ev.ticket_claim in claims
        return Extraction(concepts, claims, retractions, ev.ticket if keep else None,
                          ev.ticket_claim if keep else None)

    def _revision_candidate(self, d: str) -> tuple[Claim, Claim] | None:
        rng = self.rng
        cands = [c for c in sorted(self.world.truth[d])
                 if c.rel == "depends_on" and c in self.stated_at
                 and (d, c.src) not in self._revised
                 and self.world.layer_of[(d, c.src)] < self.cfg.layers - 2]
        if not cands:
            return None
        old = rng.choice(cands)
        li = self.world.layer_of[(d, old.tgt)]
        current = {c.tgt for c in self.world.truth[d] if c.src == old.src and c.rel == "depends_on"}
        pool = [x for x in self.world.concepts[d]
                if self.world.layer_of[(d, x)] == li and x not in current]
        if not pool:
            return None
        self._revised.add((d, old.src))
        return old, Claim.make(old.src, "depends_on", rng.choice(pool))

    # ── queries ───────────────────────────────────────────────────────
    def _query(self, step: int) -> Query | None:
        rng = self.rng
        kinds = ["focus", "chain", "bridge", "stale", "detail"]
        weights = [0.30, 0.25, 0.15, 0.15, 0.15]
        for _ in range(8):
            kind = rng.choices(kinds, weights=weights)[0]
            q = getattr(self, f"_q_{kind}")(step)
            if q:
                return q
        return self._q_focus(step)

    def _domain_for_query(self, step: int, min_known: int = 6) -> str | None:
        visited = [d for d in self.world.domains if len(self.known(d)) >= min_known]
        if not visited:
            return None
        current = max(visited, key=lambda d: self.last_active.get(d, -1))
        return current if rng_choice(self.rng, 0.5) else self.rng.choice(visited)

    def _base(self, step: int, kind: str, d: str, entry: list[str], question: str) -> Query:
        tt = self.task_text(d)
        return Query(step=step, kind=kind, domain=d, task_text=tt,
                     text=f"{tt}: {question}" if tt else question, entry=entry,
                     gap=step - self.last_active.get(d, -1))

    def _other(self, d: str) -> set[Claim]:
        return {c for e in self.world.domains if e != d for c in self.known(e)}

    def _age(self, claims: set[Claim], step: int) -> int:
        lasts = [max(self.stated_at[c]) for c in claims if c in self.stated_at]
        return step - (min(lasts) if lasts else step)

    def _q_focus(self, step: int) -> Query | None:
        d = self._domain_for_query(step)
        if not d:
            return None
        known = self.known(d)
        ends = sorted({x for c in known for x in (c.src, c.tgt)})
        entry = self.rng.sample(ends, 2)
        q = self._base(step, "focus", d, entry,
                       f"investigate {entry[0]} and {entry[1]}; what else is involved?")
        q.gold_by_radius = {r: self._neighborhood(entry, known, hops=r) for r in (1, 2, 3)}
        q.gold_claims = q.gold_by_radius[2]
        q.gold_concepts = {x for c in q.gold_claims for x in (c.src, c.tgt)} | set(entry)
        q.wrong_claims = self._other(d)
        q.age = self._age(q.gold_claims, step)
        return q

    def _q_chain(self, step: int) -> Query | None:
        d = self._domain_for_query(step)
        if not d:
            return None
        known = self.known(d)
        tops = [x for x in self.world.concepts[d]
                if self.world.layer_of[(d, x)] <= 1
                and len(self._closure(x, known)[0]) >= 3]
        if not tops:
            return None
        x = self.rng.choice(tops)
        reach, used = self._closure(x, known)
        q = self._base(step, "chain", d, [x], f"what does {x} ultimately rely on?")
        q.gold_claims, q.gold_answer = used, reach
        q.gold_concepts = reach | {x}
        q.wrong_claims = self._other(d)
        q.age = self._age(used, step)
        return q

    def _q_bridge(self, step: int) -> Query | None:
        options = []
        for b in self.world.bridges:
            for d in self.world.bridge_domains[b]:
                mine = {c for c in self.known(d) if b in (c.src, c.tgt)}
                other = {c for e in self.world.bridge_domains[b] if e != d
                         for c in self.known(e) if b in (c.src, c.tgt)}
                if len(mine) >= 2 and other:
                    options.append((b, d, mine, other))
        if not options:
            return None
        b, d, mine, other = self.rng.choice(options)
        q = self._base(step, "bridge", d, [b], f"how does {b} fit here?")
        q.gold_claims = mine
        q.gold_concepts = {x for c in mine for x in (c.src, c.tgt)}
        q.wrong_claims = other
        q.age = self._age(mine, step)
        return q

    def _q_stale(self, step: int) -> Query | None:
        done = [(s, d, o, n) for s, d, o, n in self.revisions if s <= step - 20]
        if not done:
            return None
        s, d, old, new = self.rng.choice(done)
        a = old.src
        current = {c for c in self.known(d) if c.src == a and c.rel == "depends_on"}
        q = self._base(step, "stale", d, [a], f"what does {a} depend on right now?")
        q.gold_claims = current
        q.gold_concepts = {x for c in current for x in (c.src, c.tgt)}
        q.stale_claims, q.current_claims = {old}, {new}
        q.age = self._age(current, step)
        return q

    def _q_detail(self, step: int) -> Query | None:
        options = []
        for claim, tk in self.tickets.items():
            if len(tk) == 1 and claim in self.world.truth[self._domain_of(claim)]:
                s, ticket = tk[0]
                if step - s >= 30:
                    options.append((claim, s, ticket))
        if not options:
            return None
        claim, s, ticket = self.rng.choice(options)
        d = self._domain_of(claim)
        q = self._base(step, "detail", d, [claim.src, claim.tgt],
                       f"which ticket was raised when we noted that {claim.sentence()}?")
        q.gold_claims = {claim}
        q.gold_concepts = {claim.src, claim.tgt}
        q.gold_ticket = ticket
        q.age = step - s
        return q

    def _domain_of(self, claim: Claim) -> str:
        for d in self.world.domains:
            if claim in self.world.truth[d]:
                return d
        for _, d, old, _ in self.revisions:
            if claim == old:
                return d
        raise KeyError(claim)


def rng_choice(rng: random.Random, p: float) -> bool:
    return rng.random() < p
