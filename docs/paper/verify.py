"""Numerical verification of the propositions in docs/paper/world0-formal.md.

Every number the paper quotes is computed here from the code itself (not
re-implemented formulas where the engine can be called directly), and every
closed form the paper proves is compared with the engine's behaviour.
Run from the repository root:

    python docs/paper/verify.py

Each check prints ``ok`` or raises AssertionError.
"""

from __future__ import annotations

import math
import random
import tempfile
from datetime import datetime, timezone

from world0 import Observation, World
from world0.context.focus import FOCUS_MIN_STRENGTH, FOCUS_RETENTION, Focus
from world0.dynamics.activation import (
    PROPAGATION_FLOOR_RANK_SPREAD,
    PROPAGATION_MIN_RATIO,
    RELATIVE_MIN_ACTIVATION,
    ActivationEngine,
    _accumulate,
)
from world0.dynamics.decay import (
    EVIDENCE_FLOOR_ERA_HL,
    FADING_THRESHOLD,
    PRUNE_MIN_IDLE_TICKS,
    RELATION_FLOOR_SHARE,
    DecayEngine,
    concept_half_life,
    evidence_floor,
    relax_confidence,
    settle_concept,
    settle_relation,
)
from world0.dynamics.lifecycle import (
    ACTIVATION_REDUCTION_STEP,
    BASE_CORE_CONNECTIONS,
    CORE_MIN_ACTIVATIONS,
    DENSE_DEVELOPING_ACTIVATIONS,
    DENSE_DEVELOPING_CONFIDENCE,
    DENSE_ESTABLISHED_ACTIVATIONS,
    DENSE_ESTABLISHED_CONFIDENCE,
    DENSE_ESTABLISHED_RECURRENCE,
    MIN_CORE_CONNECTIONS,
    RECURRENCE_FOR_DEVELOPING,
    RECURRENCE_FOR_ESTABLISHED,
    SPACED_DEVELOPING_EVIDENCE,
    SPACED_ESTABLISHED_BALANCE,
    SPACED_ESTABLISHED_EVIDENCE,
    LifecycleEngine,
    core_connections_required,
)
from world0.dynamics.hebbian import HEBBIAN_MIN_ASSOCIATION
from world0.projection.metacognition import TENTATIVE_EVIDENCE, WELL_EVIDENCED
from world0.schemas.clock import cognitive_elapsed
from world0.schemas.concept import RECURRENCE_CHAIN_GAP, RECURRENCE_WINDOW, ConceptNode, Maturity
from world0.schemas.relation import RelationEdge

E = EVIDENCE_FLOOR_ERA_HL
FIXED_NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def section(title: str) -> None:
    print(f"\n== {title}")


def ok(label: str, value: object = "") -> None:
    print(f"  ok  {label}{'  ' + str(value) if value != '' else ''}")


# ── §2 clock ────────────────────────────────────────────────────────────
def check_clock() -> None:
    section("§2 cognitive clock: elapsed time is additive over monotone instants")
    rng = random.Random(0)
    for _ in range(1000):
        t0, t1, t2 = sorted(rng.randint(0, 10_000) for _ in range(3))
        h = sorted(rng.uniform(0, 5000) for _ in range(3))
        w = [datetime.fromtimestamp(1.7e9 + x * 3600, tz=timezone.utc) for x in h]
        whole = cognitive_elapsed(t2, t0, w[2], w[0])
        parts = cognitive_elapsed(t2, t1, w[2], w[1]) + cognitive_elapsed(t1, t0, w[1], w[0])
        assert abs(whole - parts) < 1e-6
    ok("Prop 2.1 additivity (1000 random triples)")


# ── §3 concept dynamics ─────────────────────────────────────────────────
def _node(n: int, d: int = 0, **kw) -> ConceptNode:
    node = ConceptNode(name="x", **kw)
    node.activation_count = n
    node.disconfirmation_count = d
    node.last_activated = FIXED_NOW
    return node


def check_gain() -> None:
    section("§3.1 activation gain b(n) = 0.06/(1+0.08n)")
    node = ConceptNode(name="x")
    steps = 0
    while node.confidence < 0.6:
        node.activate(tick=steps)
        steps += 1
    approx = (1.08 * math.exp(0.45 / 0.75) - 1) / 0.08
    assert steps == 11, steps
    ok("activations from creation to confidence 0.6 (no decay)", f"{steps} (integral estimate {approx:.1f})")


def check_idempotency() -> None:
    section("§3.2 decay: exact semigroup (Prop 3.1) — the era correction of the frozen-floor scheme is gone")
    for n, h1, h2 in [(0, 30, 50), (40, 300, 500), (40, 2000, 3000), (8, 100, 50)]:
        results = []
        for split in (False, True):
            w = World(store_path=tempfile.mkdtemp())
            w.ingest(Observation(concepts=["x"]))
            node = w.concepts.resolve("x")
            node.activation_count = max(1, n)
            node.confidence = 0.9
            node.maturity = Maturity.DEVELOPING
            node.last_activated = FIXED_NOW
            dec = DecayEngine(w.concepts, w.relations, clock=w.clock)
            if split:
                w.clock.advance(h1)
                dec.decay_concepts()
                w.clock.advance(h2)
            else:
                w.clock.advance(h1 + h2)
            dec.decay_concepts()
            results.append(node.confidence)
        diff = results[1] - results[0]
        assert abs(diff) < 1e-9, (n, diff)
        # what the previous scheme (floor frozen at the interval end) would have left
        node = _node(max(1, n), maturity=Maturity.DEVELOPING)
        H = concept_half_life(node)
        r1, r2 = 2 ** (-h1 / H), 2 ** (-h2 / H)
        f = evidence_floor(node, now_tick=0, now=FIXED_NOW)
        f1, f2 = f * 2 ** (-h1 / E), f * 2 ** (-(h1 + h2) / E)
        old_gap = r2 * (1 - r1) * (f1 - f2)
        ok(f"n={n:<3} Δ1={h1:<5} Δ2={h2:<5} split−whole = {diff:+.2e}", f"(frozen-floor scheme: {old_gap:.6f})")
    rng = random.Random(3)
    worst = 0.0
    for _ in range(20_000):
        c, f = rng.uniform(0.01, 1), rng.uniform(0, 0.35)
        H = rng.choice([24, 168, 720, 2160, 8760, 4380.0000001])
        t1, t2 = rng.uniform(0, 3000), rng.uniform(0, 3000)
        whole = relax_confidence(c, f, H, t1 + t2)
        split = relax_confidence(relax_confidence(c, f, H, t1), f * 2 ** (-t1 / E), H, t2)
        worst = max(worst, abs(whole - split))
    assert worst < 1e-9
    ok("relax_confidence, 20 000 random (c, f, H, t1, t2) incl. c < f and H ≈ E", f"max |whole − split| = {worst:.1e}")


def _rk4_settle(node: ConceptNode, elapsed: float, dt: float = 0.02) -> float:
    """Independent reference: RK4 on c' = -λ (c - f(t))⁺ with the half-life
    switched to FADING's when c first drops below the threshold."""
    h_mature = concept_half_life(node)
    fading = node.model_copy()
    fading.maturity = Maturity.FADING
    h_fading = concept_half_life(fading)
    f0 = evidence_floor(node, now_tick=node.last_activated_tick, now=FIXED_NOW)
    mu = math.log(2) / E
    c, t, is_fading = node.confidence, 0.0, node.maturity == Maturity.FADING
    while t < elapsed - 1e-12:
        h = min(dt, elapsed - t)
        lam = math.log(2) / (h_fading if is_fading else h_mature)
        def rhs(tt: float, cc: float) -> float:
            return -lam * max(0.0, cc - f0 * math.exp(-mu * tt))
        k1 = rhs(t, c)
        k2 = rhs(t + h / 2, c + h / 2 * k1)
        k3 = rhs(t + h / 2, c + h / 2 * k2)
        k4 = rhs(t + h, c + h * k3)
        c += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not is_fading and c < FADING_THRESHOLD:
            is_fading = True
    return c


def check_fading_boundary() -> None:
    section("§3.2 the FADING boundary is exact: settling inside the crossing gap changes nothing")
    # (n, c0, maturity, crossing tick): the interval ends 30 ticks after the
    # crossing so the value still depends on where the half-life switched.
    cases = [(2, 0.2, Maturity.DEVELOPING, 437), (2, 0.06, Maturity.ESTABLISHED, 262), (2, 0.055, Maturity.CORE, 412)]
    for n, c0, maturity, cross in cases:
        end = cross + 30
        outs = []
        for cuts in ([], [cross // 2], [cross - 50, cross - 1, cross + 1, cross + 10], [1, 2, 3, cross - 20]):
            node = _node(n, maturity=maturity, confidence=c0)
            for cut in cuts + [end]:
                settle_concept(node, cut, FIXED_NOW)
            outs.append((node.maturity, node.confidence))
        confs = [c for _, c in outs]
        assert max(confs) - min(confs) < 1e-9 and len({m for m, _ in outs}) == 1, outs
        assert outs[0][0] == Maturity.FADING, "the case must actually cross the threshold"
        ref = _rk4_settle(_node(n, maturity=maturity, confidence=c0), float(end))
        assert abs(ref - confs[0]) < 1e-4, (ref, confs[0])
        early = _node(n, maturity=maturity, confidence=c0)
        settle_concept(early, cross - 5, FIXED_NOW)
        assert early.maturity != Maturity.FADING  # the crossing is inside the interval, not before it
        ok(f"n={n} {maturity.value:<11} c0={c0}", f"crosses at ~{cross} → fading {confs[0]:.6f}; RK4 reference {ref:.6f}; 4 settle schedules agree")


def check_schedule_independence() -> None:
    section("§3.2 Theorem 3.2: forgetting does not depend on the reflect cadence")
    def run(T: int, R: int | None, uses: int = 30) -> tuple[float, float]:
        w = World(store_path=tempfile.mkdtemp())
        for _ in range(uses):
            w.ingest(Observation(concepts=["c", "d"], relations=[("c", "d", "depends_on")]))
            for _ in range(T - 1):
                w.clock.advance(1)
                if R and w.clock.tick % R == 0:
                    w._decay.decay_concepts()
                    w._decay.decay_relations()
        w._decay.decay_concepts()
        w._decay.decay_relations()
        return w.concepts.resolve("c").confidence, w.relations.all()[0].weight
    for T in (24, 168, 720):
        res = {R: run(T, R) for R in (1, 50, 1000, None)}
        cs = [c for c, _ in res.values()]
        ws = [x for _, x in res.values()]
        # exact semigroup: only wall-clock drift (~1e-9) remains
        assert max(cs) - min(cs) < 1e-6 and max(ws) - min(ws) < 1e-6, res
        ok(f"use every {T:<4}", "reflect every 1 / 50 / 1000 / never → confidence "
           + " / ".join(f"{c:.6f}" for c in cs) + ", weight " + " / ".join(f"{x:.6f}" for x in ws))


def _stream(kind: str):
    """Observation streams (tick → Observation) for the maturity theorem."""
    def use(weaken=False, extra=()):
        return Observation(
            concepts=["c", "anchor", *extra],
            relations=[("c", "anchor", "depends_on")] + [("c", e, "supports") for e in extra],
            weakened=["c"] if weaken else [],
            task="routine",
            source="s",
        )
    if kind.startswith("cad"):
        T = int(kind[3:])
        return {k * T: use() for k in range(1, 31)}, 31 * T
    if kind == "ghost":
        # five one-shot partners expire; their edges outlive them in a world that never reflects
        ev = {1: Observation(
            concepts=["c"] + [f"p{i}" for i in range(5)],
            relations=[("c", f"p{i}", "depends_on") for i in range(5)],
            source="s",
        )}
        ev.update({1 + 48 * k: Observation(concepts=["c"], source="s") for k in range(1, 41)})
        return ev, 1 + 48 * 41
    if kind == "sparse-beyond-grace":
        T = int(PRUNE_MIN_IDLE_TICKS) + 80
        return {1 + T * k: Observation(concepts=["c"], source="s") for k in range(15)}, 1 + T * 14 + 5
    if kind == "burst":
        return {t: use() for t in range(1, 31)}, 5030
    if kind == "one-shot":
        return {1: use()}, 5000
    if kind == "disconf":
        return {k * 24: use(weaken=(k % 3 == 0)) for k in range(1, 31)}, 31 * 24
    if kind == "late-links":
        ev = {k * 24: use() for k in range(1, 41)}
        t0 = 40 * 24
        for i in range(5):
            ev[t0 + 300 + i] = Observation(
                concepts=[f"p{i}"], relations=[("c", f"p{i}", "supports")], source="s"
            )
        for k in range(1, 6):
            ev[t0 + 720 * k] = use()
        return ev, t0 + 720 * 6
    raise KeyError(kind)


def _drive(events, horizon: int, every: int | None):
    w = World(store_path=tempfile.mkdtemp())
    ticks = sorted(events)
    i = 0
    while w.clock.tick < horizon:
        now = w.clock.tick
        nxt_event = ticks[i] if i < len(ticks) else None
        nxt_reflect = (now // every + 1) * every if every else None
        stop = min(x for x in (nxt_event, nxt_reflect, horizon) if x is not None)
        if nxt_event == stop:
            w.clock.advance(stop - 1 - now)
            w.ingest(events[stop])
            i += 1
        else:
            w.clock.advance(stop - now)
        if every and stop % every == 0 and stop < horizon:
            w.reflect(light=True)
    w.reflect()
    return {c.name: (c.maturity.value, c.confidence, c.activation_count) for c in w.concepts.all()}


def check_maturity_schedule_independence() -> None:
    section("§3.5 Theorem 3.7: (existence, maturity, confidence) do not depend on the reflect cadence")
    for kind in ("cad24", "cad72", "cad168", "cad720", "burst", "one-shot", "disconf", "late-links",
                 "ghost", "sparse-beyond-grace"):
        events, horizon = _stream(kind)
        cadences = (1, 50, 1000, None)
        res = [_drive(events, horizon, r) for r in cadences]
        ref = res[-1]
        for r, state in zip(cadences, res):
            assert state.keys() == ref.keys(), (kind, r, sorted(state), sorted(ref))
            for name, (maturity, conf, n) in state.items():
                assert (maturity, n) == (ref[name][0], ref[name][2]), (kind, r, name, state[name], ref[name])
                assert abs(conf - ref[name][1]) < 1e-6, (kind, r, name, state[name], ref[name])
        if "c" not in ref:
            ok(f"{kind:<19}", "concept pruned under every cadence")
            continue
        spread = max(abs(st["c"][1] - ref["c"][1]) for st in res)
        ok(f"{kind:<19}", f"{ref['c'][0]:<11} n={ref['c'][2]:<3} spread over reflect every {cadences}: {spread:.1e}")


def check_noise_threshold() -> None:
    section("§3.3 evidence floor below FADING_THRESHOLD ⇔ n ≤ 6 (no disconfirmation)")
    rows = []
    for n in range(1, 11):
        f = evidence_floor(_node(n), now_tick=0, now=FIXED_NOW)
        rows.append((n, f))
        assert (f < FADING_THRESHOLD) == (n <= 6), (n, f)
    ok("floor(n)", ", ".join(f"{n}:{f:.4f}" for n, f in rows))


def check_one_off() -> None:
    section("§3.3 a one-off mention: fading and pruning times")
    w = World(store_path=tempfile.mkdtemp())
    w.ingest(Observation(concepts=["once"]))
    node = w.concepts.resolve("once")
    c1 = node.confidence
    f = evidence_floor(node, now_tick=w.clock.tick)
    t_fade = 24 * math.log2((c1 - f) / (FADING_THRESHOLD - f))
    t_02 = 24 * math.log2((c1 - f) / (0.02 - f))
    dec = DecayEngine(w.concepts, w.relations, clock=w.clock)
    faded_at = pruned_at = None
    for t in range(1, 800):
        w.clock.advance(1)
        dec.decay_concepts()
        if faded_at is None and node.maturity == Maturity.FADING:
            faded_at = t
        if dec.prune_concepts():
            pruned_at = t
            break
    assert faded_at is not None and abs(faded_at - t_fade) <= 1.5, (faded_at, t_fade)
    assert pruned_at == 720, pruned_at
    below_02 = None
    probe = ConceptNode(name="probe")
    probe.activate(tick=0)
    for t in range(1, 200):
        copy = probe.model_copy()
        settle_concept(copy, t, probe.last_activated)
        if copy.confidence < 0.02:
            below_02 = t
            break
    assert below_02 is not None and abs(below_02 - t_02) <= 1.0, (below_02, t_02)
    ok("initial confidence after one mention", f"{c1:.4f}")
    ok("marked FADING after", f"{faded_at} obs (closed form {t_fade:.1f})")
    ok("confidence < 0.02 after", f"{t_02:.1f} obs; deleted at {pruned_at} (grace 720)")


def _uses_to_established(T: int, reflect_every: int | None, cap: int = 60) -> tuple[int | None, int, int]:
    """(use at which ESTABLISHED, node re-creations, final n) for a concept used every T ticks."""
    w = World(store_path=tempfile.mkdtemp(), auto_reflect_every=None)
    ids: set[str] = set()
    recreated = -1
    established = None
    last_n = 0
    for use in range(1, cap + 1):
        w.ingest(Observation(concepts=["c"]))
        node = w.concepts.resolve("c")
        last_n = node.activation_count
        if node.id not in ids:
            ids.add(node.id)
            recreated += 1
        if established is None and node.maturity == Maturity.ESTABLISHED:
            established = use
            break
        for _ in range(T - 1):
            w.clock.advance(1)
            if reflect_every and w.clock.tick % reflect_every == 0:
                w.reflect(light=True)
    return established, recreated, last_n


def check_recurrence_bound() -> None:
    section("§3.3 Prop 3.5: a cadence within the prune grace reaches ESTABLISHED at use n*; beyond it the concept is noise")
    def e(n: int, d: int = 0) -> float:
        return (n + 1) / (n + d + 2) * n / (n + 10)
    n_star = next(n for n in range(10, 200) if e(n) >= SPACED_ESTABLISHED_EVIDENCE)
    ok("n* = min { n ≥ 10 : e(n, 0) ≥ 0.5 }  (WELL_EVIDENCED)", f"{n_star} uses; balance(n*) = {(n_star + 1) / (n_star + 2):.3f} ≥ {SPACED_ESTABLISHED_BALANCE}")
    assert n_star == 12
    for T in (24, 72, 168, 720):
        for reflect_every in (None, 50):
            used, recreated, _ = _uses_to_established(T, reflect_every)
            assert used == n_star and recreated == 0, (T, reflect_every, used, recreated)
        w = World(store_path=tempfile.mkdtemp(), auto_reflect_every=None)
        for _ in range(n_star):
            w.ingest(Observation(concepts=["c"]))
            w.clock.advance(T - 1)
        node = w.concepts.resolve("c")
        ok(f"cadence T={T}", f"ESTABLISHED at use {n_star} with and without reflect (tick {w.clock.tick - T + 1}, confidence {node.confidence:.3f}, recurrences {node.recurrence_count})")
    for T in (int(PRUNE_MIN_IDLE_TICKS) + 1, 800, 2000):
        outcomes = {r: _uses_to_established(T, r, cap=20) for r in (None, 50)}
        for r, (used, recreated, n) in outcomes.items():
            assert used is None and n == 1 and recreated >= 19, (T, r, outcomes)
        ok(f"cadence T={T} > prune grace {int(PRUNE_MIN_IDLE_TICKS)}", "never ESTABLISHED; the concept (n ≤ 6 keeps its floor under 0.05) is forgotten between uses — identical without reflect")
    # the recurrence chain of a concept that is *not* forgotten (n ≥ 7) breaks at exactly E/2
    for gap in (int(RECURRENCE_CHAIN_GAP) - 1, int(RECURRENCE_CHAIN_GAP)):
        node = ConceptNode(name="veteran")
        for k in range(8):
            node.activate(tick=1 + 24 * k)
        rho = node.recurrence_count
        node.activate(tick=node.last_activated_tick + gap)
        assert node.recurrence_count == (rho + 1 if gap < RECURRENCE_CHAIN_GAP else 1), (gap, node.recurrence_count)
    ok(f"chain gap E/2 = {RECURRENCE_CHAIN_GAP:.0f}", "a gap of E/2 − 1 keeps the recurrence chain (ρ+1); a gap of E/2 restarts it (ρ = 1)")


def check_evidence_thresholds() -> None:
    section("§3.5 / §9 evidence e(n) thresholds used by metacognition")
    ev = {n: _node(n).evidence() for n in range(0, 30)}
    tentative = max(n for n, e in ev.items() if e < TENTATIVE_EVIDENCE)
    well = min(n for n, e in ev.items() if e >= WELL_EVIDENCED)
    assert all(ev[n] < ev[n + 1] for n in range(29))
    ok("e(n) strictly increasing in n (d = 0)")
    assert tentative == 2 and well == 12, (tentative, well)
    ok("tentative ⇔ n ≤", f"{tentative}   (e(2)={ev[2]:.3f}, e(3)={ev[3]:.3f})")
    ok("well-evidenced ⇔ n ≥", f"{well}  (e(11)={ev[11]:.3f}, e(12)={ev[12]:.3f})")


def check_salience() -> None:
    section("§3.5 salience = max(freshness, 0.7·e·2^(−Δ/E)) ∈ [0.1, 1]")
    rng = random.Random(1)
    for _ in range(500):
        n = rng.randint(0, 200)
        node = _node(n)
        dt = rng.uniform(0, 20_000)
        s = node.salience(168.0, now_tick=int(dt), now=FIXED_NOW)
        fresh = max(0.1, 2 ** (-int(dt) / 168.0)) if int(dt) > 0 else 1.0
        pers = 0.7 * node.evidence() * 2 ** (-int(dt) / E)
        assert 0.1 <= s <= 1.0 and abs(s - max(fresh, pers)) < 1e-9
    veteran = _node(50)
    t = E * math.log2(0.7 * veteran.evidence() / 0.1)
    ok("closed form holds (500 random states)")
    ok("veteran n=50 stays above the freshness floor for", f"{t:.0f} idle observations")


# ── §4 relations ────────────────────────────────────────────────────────
def check_confirm() -> None:
    section("§4.1 confirm(): p_k = 1 − (1 − p0)·0.95^k")
    edge = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence", is_explicit=True)
    p0 = edge.probability
    for k in range(1, 21):
        edge.confirm()
        assert abs(edge.probability - (1 - (1 - p0) * 0.95 ** k)) < 1e-12
    ok("closed form (k = 1..20)", f"p0={p0:.2f} → p20={edge.probability:.4f}")


def check_relation_survival() -> None:
    section("§4.2 explicit relation survival ≥ E·log2(p/0.2·…) = E·log2(5p), both axes")
    survived: dict[tuple[str, int], int] = {}
    # ``depends_on`` is the reference relation; ``conflict`` and ``disjointness``
    # are negative-axis claims whose propagation strength is an inhibition gain
    # (0.10 / 0.05), not a belief: they are seeded from NEGATIVE_CLAIM_PRIOR and
    # must live exactly as long as the dependence (before: 450 / 350 vs 8100).
    for rel in ("depends_on", "conflict", "disjointness"):
        for restate in (0, 5):
            w = World(store_path=tempfile.mkdtemp())
            for _ in range(1 + restate):
                w.ingest(Observation(concepts=["a", "b"], relations=[("a", "b", rel)]))
            a, b = w.concepts.resolve("a"), w.concepts.resolve("b")
            edge = w.relations.find_any_between(a.id, b.id)[0]
            p = edge.probability
            gain = edge.propagation_strength
            bound = E * math.log2(p / (0.02 / RELATION_FLOOR_SHARE))
            dec = DecayEngine(w.concepts, w.relations, clock=w.clock)
            start = w.clock.tick
            pruned_at = None
            step = 50
            while pruned_at is None and w.clock.tick - start < 20_000:
                w.clock.advance(step)
                for node in (a, b):  # keep the concepts alive, relations idle
                    node.last_activated_tick = w.clock.tick
                    node.last_activated = datetime.now(timezone.utc)
                dec.decay_relations()
                if edge.id in dec.prune_relations():
                    pruned_at = w.clock.tick - start
            assert pruned_at is not None and pruned_at >= bound - step, (rel, pruned_at, bound)
            survived[(rel, restate)] = pruned_at
            ok(f"{rel:12s} restated x{restate} p={p:.3f} gain={gain:.2f}",
               f"bound {bound:.0f}, pruned after {pruned_at} idle observations")
    for rel in ("conflict", "disjointness"):
        for restate in (0, 5):
            assert survived[(rel, restate)] == survived[("depends_on", restate)], (rel, restate)
    ok("negative claims live exactly as long as a dependence",
       f"{survived[('conflict', 0)]} = {survived[('depends_on', 0)]} (stated once), "
       f"{survived[('conflict', 5)]} = {survived[('depends_on', 5)]} (x5)")


# ── §5 Hebbian ─────────────────────────────────────────────────────────
def check_jaccard() -> None:
    section("§5 Jaccard = 1/(1/P(b|a) + 1/P(a|b) − 1); asymptotic association")
    rng = random.Random(3)
    for _ in range(1000):
        c = rng.randint(1, 50)
        na, nb = c + rng.randint(0, 100), c + rng.randint(0, 100)
        j = c / (na + nb - c)
        assert abs(j - 1 / (na / c + nb / c - 1)) < 1e-12
    ok("identity (1000 random counts)")
    theta = HEBBIAN_MIN_ASSOCIATION
    ok("symmetric gate ⇔ P(b|a) ≥ 2θ/(1+θ) =", f"{2 * theta / (1 + theta):.4f}")
    q = 6 / 60
    ok("independent 6-of-60 mentions: J∞ = q/(2−q) =", f"{q / (2 - q):.4f}")
    rho = 3 / 5
    ok("topic of 6, 4 drawn: ρ = 3/5, J∞ = ρ/(2−ρ) =", f"{rho / (2 - rho):.4f}")
    # simulation of both regimes
    for kind in ("random", "topic"):
        rng = random.Random(5)
        w = World(store_path=tempfile.mkdtemp())
        pool = [f"c{i}" for i in range(60)]
        topics = [pool[i:i + 6] for i in range(0, 60, 6)]
        for _ in range(3000):
            obs = rng.sample(pool, 6) if kind == "random" else rng.sample(rng.choice(topics), 4)
            w._hebbian._observations += 1
            ids = []
            for name in obs:
                ids.append(name)
                w._hebbian._mentions[name] = w._hebbian._mentions.get(name, 0) + 1
            for i, x in enumerate(ids):
                for y in ids[i + 1:]:
                    key = "|".join(sorted((x, y)))
                    w._hebbian._linked[key] = w._hebbian._linked.get(key, 0) + 1
        h = w._hebbian
        js = []
        pairs = [(pool[0], pool[1]), (pool[2], pool[3]), (pool[4], pool[5])]
        for x, y in pairs:
            c = h._linked.get("|".join(sorted((x, y))), 0)
            js.append(c / (h.mentions(x) + h.mentions(y) - c))
        ok(f"simulated J ({kind}, 3000 obs)", ", ".join(f"{j:.3f}" for j in js))


def check_revalidation_count() -> None:
    section("§5.3 revalidation uses the exact co-occurrence count (F3)")
    w = World(store_path=tempfile.mkdtemp())
    for i in range(12):
        w.ingest(Observation(concepts=["x", f"n{i}"]))
        w.ingest(Observation(concepts=["y", f"m{i}"]))
    for _ in range(8):
        w.ingest(Observation(concepts=["x", "y"]))
    x, y = w.concepts.resolve("x"), w.concepts.resolve("y")
    edge = w.relations.find_any_between(x.id, y.id)[0]
    exact = w._hebbian.cooccurrences(x.id, y.id)
    lower = edge.reinforcement_count + 2
    assert exact > lower
    ok("delayed link", f"exact co-occurrences {exact}, old estimate r+2 = {lower}")


# ── §6 activation ──────────────────────────────────────────────────────
def check_noisy_or() -> None:
    section("§6.1 bounded noisy-OR ⊕_S")
    rng = random.Random(7)
    for _ in range(2000):
        S = rng.uniform(0.01, 1)
        a, b, c = (rng.uniform(0, S) for _ in range(3))
        ab = _accumulate(a, b, S)
        assert abs(ab - _accumulate(b, a, S)) < 1e-12
        assert abs(_accumulate(ab, c, S) - _accumulate(a, _accumulate(b, c, S), S)) < 1e-12
        assert max(a, b) - 1e-12 <= ab <= S + 1e-12
        if 0 < a < S and 0 < b < S:
            assert ab > max(a, b)
        big = _accumulate(0.0, rng.uniform(S, 3 * S), S)
        assert big <= S + 1e-12
    ok("commutative, associative, ≥ max, strictly > max inside (0,S), ≤ S (2000 draws)")


def check_floor() -> None:
    section("§6.2 rank-preserving floor φ")
    F = 1.0
    xs = [i / 1000 for i in range(1, 3000)]
    ys = [ActivationEngine._apply_floor(x, F) for x in xs]
    assert all(y2 > y1 for y1, y2 in zip(ys, ys[1:]))
    assert min(ys) >= (1 - PROPAGATION_FLOOR_RANK_SPREAD) * F
    lo = (1 - PROPAGATION_FLOOR_RANK_SPREAD) * PROPAGATION_MIN_RATIO
    ok("strictly increasing, image of (0,F) = (0.9F, F)")
    ok("every lifted signal ≥ 0.9·0.03·S =", f"{lo:.3f}·S > cut = {RELATIVE_MIN_ACTIVATION}·S")


def check_seed_dominance_and_horizon() -> None:
    section("§6.3 seed dominance and horizon completeness on random graphs")
    rng = random.Random(11)
    for trial in range(30):
        w = World(store_path=tempfile.mkdtemp())
        names = [f"v{i}" for i in range(25)]
        for _ in range(60):
            obs = rng.sample(names, 3)
            rel = rng.choice(["depends_on", "contains", "similar_to", "conflict", "related_to"])
            w.ingest(Observation(concepts=obs, relations=[(obs[0], obs[1], rel)]))
        w.clock.advance(rng.randint(0, 2000))
        seeds = [w.concepts.resolve(n).id for n in rng.sample(names, 2)]
        eng = ActivationEngine(w.concepts, w.relations, clock=w.clock)
        depth = rng.choice([1, 2, 3])
        act = eng.activate(seeds, max_depth=depth, decay=0.6, record=False)
        S = max(w.concepts.get(s).confidence for s in seeds)
        assert all(v <= S + 1e-9 for k, v in act.items() if k not in seeds)
        assert all(s in act for s in seeds)
        # horizon: every concept within `depth` non-negative hops is reached
        frontier, seen = set(seeds), set(seeds)
        for _ in range(depth):
            nxt = set()
            for cid in frontier:
                for rel in w.relations.for_concept(cid):
                    if rel.relation_type.value == "negative":
                        continue
                    o = rel.other_end(cid)
                    if o not in seen:
                        nxt.add(o)
            seen |= nxt
            frontier = nxt
        inhibited = {
            rel.other_end(c) for c in act for rel in w.relations.for_concept(c)
            if rel.relation_type.value == "negative"
        }
        missing = [c for c in seen if c not in act and c not in inhibited]
        assert not missing, missing
    ok("30 random worlds: no non-seed above S, every seed returned, full horizon reached")


# ── §7 projection ─────────────────────────────────────────────────────
def check_seed_first() -> None:
    section("§7.1 seeds-first holds for a faded seed (F1)")
    w = World(store_path=tempfile.mkdtemp())
    for _ in range(20):
        w.ingest(Observation(concepts=["strong", "friend"]))
    w.ingest(Observation(concepts=["weak"]))
    weak = w.concepts.resolve("weak")
    weak.confidence = 0.004
    weak.maturity = Maturity.FADING
    names = [c.name for c in w.project(["strong", "weak"]).concepts]
    assert "weak" in names
    ok("projection", names)


# ── §8 focus ───────────────────────────────────────────────────────────
def check_focus() -> None:
    section("§8 focus: capacity, lifetime, bounded bias")
    life = math.floor(math.log(FOCUS_MIN_STRENGTH) / math.log(FOCUS_RETENTION))
    f = Focus()
    f.update(["a"])
    stays = 0
    while "a" in f.items():
        f.update([])
        stays += 1 if "a" in f.items() else 0
    assert stays == life
    ok("an item ignited once survives exactly", f"{life} further updates")
    f = Focus()
    rng = random.Random(2)
    for _ in range(200):
        f.update(rng.sample([f"c{i}" for i in range(30)], rng.randint(0, 12)))
        assert len(f) <= 7
    ok("|F| ≤ 7 over 200 random updates")


# ── §9 prediction ──────────────────────────────────────────────────────
def check_prediction_calibration() -> None:
    section("§9 prediction error: calibration under the model's own hypothesis")
    rng = random.Random(4)
    for ps in ([0.6] * 4, [0.9, 0.8, 0.6], [1.0, 0.5], [0.7] * 10):
        M = sum(ps)
        V = sum(p * (1 - p) for p in ps)
        num, errs = [], []
        for _ in range(40_000):
            X = sum(p for p in ps if rng.random() >= p)   # absent with prob 1 − p
            num.append(X - V)
            errs.append(min(1.0, max(0.0, (X - V) / (M - V))))
        bound = 0.5 * math.sqrt(sum(p ** 3 * (1 - p) for p in ps)) / sum(p * p for p in ps)
        mean_num = sum(num) / len(num)
        mean_err = sum(errs) / len(errs)
        assert abs(mean_num) < 0.01 and mean_err <= bound + 0.01
        ok(f"p={ps}", f"E[X−V]≈{mean_num:+.4f}  E[e]≈{mean_err:.3f} ≤ bound {bound:.3f}")
    # the engine agrees with the closed form on a real world
    w = World(store_path=tempfile.mkdtemp())
    for _ in range(10):
        w.ingest(Observation(concepts=["a", "b", "c"]))
    r = w.ingest(Observation(concepts=["a"]))
    assert r.prediction.missing_ratio == 1.0
    ok("all certain companions absent → e = 1 (engine)")
    def lam(na: int, nb: int, n_obs: int) -> float:
        return 1 - math.exp(-na * nb / n_obs)

    ok("novelty weight 1 − e^(−λ): n=50/50 of N=100 →", f"{lam(50, 50, 100):.4f}; n=5/5 of N=1000 → {lam(5, 5, 1000):.4f}")


def check_prediction_per_companion() -> None:
    section("§9 one prediction per companion (F5)")
    w = World(store_path=tempfile.mkdtemp())
    for _ in range(10):
        w.ingest(Observation(concepts=["a", "b", "c"], relations=[("a", "b", "enables"), ("a", "b", "conflict")]))
    r = w.ingest(Observation(concepts=["a", "c"]))
    assert r.prediction.missing_ratio == 0.5, r.prediction
    ok("pair a–b with two opposing claims, b absent", f"missing ratio {r.prediction.missing_ratio} (was 0.6)")


def _gate_node(n: int, *, d: int = 0, rho: int = 1, c: float = 0.1, maturity: Maturity = Maturity.EMBRYONIC) -> ConceptNode:
    node = ConceptNode(name="g", maturity=maturity, confidence=c)
    node.activation_count = n
    node.disconfirmation_count = d
    node.recurrence_count = rho
    return node


def check_gate_table() -> None:
    section("§3.5 Definition 3.5: the promotion gates, at their boundaries")
    assert (DENSE_DEVELOPING_ACTIVATIONS, DENSE_DEVELOPING_CONFIDENCE, RECURRENCE_FOR_DEVELOPING, SPACED_DEVELOPING_EVIDENCE) == (3, 0.3, 3, 0.15)
    assert (DENSE_ESTABLISHED_ACTIVATIONS, DENSE_ESTABLISHED_RECURRENCE, DENSE_ESTABLISHED_CONFIDENCE,
            RECURRENCE_FOR_ESTABLISHED, SPACED_ESTABLISHED_EVIDENCE, SPACED_ESTABLISHED_BALANCE) == (10, 3, 0.6, 10, 0.5, 0.8)
    assert (CORE_MIN_ACTIVATIONS, BASE_CORE_CONNECTIONS, MIN_CORE_CONNECTIONS, ACTIVATION_REDUCTION_STEP) == (30, 5, 2, 20)
    assert TENTATIVE_EVIDENCE == SPACED_DEVELOPING_EVIDENCE and WELL_EVIDENCED == SPACED_ESTABLISHED_EVIDENCE
    ok("constants equal the table of Definition 3.5; the spaced gates' evidence lines are metacognition's")
    engine = LifecycleEngine(None, None)  # the two lower rungs read no stores
    emb, dev = Maturity.EMBRYONIC, Maturity.DEVELOPING
    cases = [
        (dict(n=3, c=0.3, rho=1, maturity=emb), Maturity.DEVELOPING, "embryonic dense: n ≥ 3 ∧ c ≥ 0.3"),
        (dict(n=3, c=0.299, rho=1, maturity=emb), None, "  … c just below 0.3, one window: no"),
        (dict(n=3, c=0.05, rho=3, maturity=emb), Maturity.DEVELOPING, "embryonic spaced: ρ ≥ 3 ∧ e ≥ 0.15 (e(3) = 0.185)"),
        (dict(n=2, c=0.05, rho=3, maturity=emb), None, "  … n = 2: e = 0.125 < 0.15, no"),
        (dict(n=10, c=0.6, rho=3, maturity=dev), Maturity.ESTABLISHED, "developing dense: n ≥ 10 ∧ ρ ≥ 3 ∧ c ≥ 0.6"),
        (dict(n=30, c=0.99, rho=2, maturity=dev), None, "  … a burst (ρ = 2) is never established by intensity"),
        (dict(n=10, c=0.6, rho=0, maturity=dev), Maturity.ESTABLISHED, "  … ρ = 0 (legacy record) keeps the old dense rule"),
        (dict(n=12, c=0.01, rho=10, maturity=dev), Maturity.ESTABLISHED, "developing spaced: ρ ≥ 10 ∧ e ≥ 0.5 ∧ β ≥ 0.8 (n = 12)"),
        (dict(n=11, c=0.01, rho=10, maturity=dev), None, "  … n = 11: e = 0.484 < 0.5, no"),
        (dict(n=40, d=12, c=0.01, rho=10, maturity=dev), None, "  … contested (β < 0.8), no"),
    ]
    for kwargs, expect, label in cases:
        got = engine._evaluate_one(_gate_node(**kwargs))
        assert got == expect, (label, got)
        ok(label, "" if expect is None else f"→ {expect.value}")
    for n, k in ((30, 5), (49, 5), (50, 4), (70, 3), (90, 2), (500, 2)):
        assert core_connections_required(n) == k, (n, core_connections_required(n))
    ok("core connections max(2, 5 − ⌊(n−30)/20⌋)", "n = 30/50/70/90 → 5/4/3/2")


def check_relation_settle_exact() -> None:
    section("§4.2 Prop 4.2': relation settling is an exact semigroup (as for concepts)")
    rng = random.Random(9)
    worst = 0.0
    for _ in range(3000):
        explicit = rng.random() < 0.7
        p, w0 = rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0)
        reinforced = rng.randint(0, 6)
        edges = []
        for _k in range(2):
            e = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence", is_explicit=explicit)
            e.probability, e.weight, e.confidence = p, w0, w0
            e.reinforcement_count = reinforced
            edges.append(e)
        total = rng.randint(30, 9000)
        cut = rng.randint(1, total - 1)
        settle_relation(edges[0], total)
        settle_relation(edges[1], cut)
        settle_relation(edges[1], total)
        worst = max(worst, abs(edges[0].weight - edges[1].weight), abs(edges[0].confidence - edges[1].confidence))
    assert worst < 1e-8, worst
    ok("3000 random (explicit/co-occurrence, p, w, r, split)", f"max |whole − split| = {worst:.1e}")
    # a belief below 0.2 puts the floor under the prune line: the claim is mortal, one above is not
    lives = {}
    for p in (0.1, 0.7):
        e = RelationEdge(source_id="a", target_id="b", semantic_relation="dependence", is_explicit=True)
        e.probability, e.weight, e.confidence = p, p, p
        t = 0
        while e.weight >= 0.02 and t < 20_000:
            t += 25
            settle_relation(e, t)
        lives[p] = t
    assert lives[0.1] < 1500 and lives[0.7] > 7000, lives
    ok("floor 0.1·p vs prune line 0.02", f"p = 0.1 dies after {lives[0.1]} ticks (noise); p = 0.7 after {lives[0.7]}")


def check_opposition() -> None:
    section("§4.3 opposition is symmetric (F2)")
    from world0.schemas.relation import SEMANTIC_RELATION_SPECS
    for a in SEMANTIC_RELATION_SPECS:
        for b in SEMANTIC_RELATION_SPECS:
            ea = RelationEdge(source_id="x", target_id="y", semantic_relation=a, is_explicit=True)
            eb = RelationEdge(source_id="x", target_id="y", semantic_relation=b, is_explicit=True)
            assert ea.opposes(eb.relation_type, eb.semantic_relation) == eb.opposes(ea.relation_type, ea.semantic_relation)
    ok(f"all {len(SEMANTIC_RELATION_SPECS) ** 2} ordered pairs of semantic relations")

    # A stated claim has the same standing on either axis: `dependence` and every
    # negative claim are seeded at 0.70, so swapping which side of a contested pair
    # is the negative one swaps the two beliefs exactly.  (Before, "enables x10 then
    # conflict x10" gave 0.45 vs 0.43 and its mirror 0.85 vs 0.03.)
    def beliefs(seq: str) -> tuple[float, float]:
        w = World(store_path=tempfile.mkdtemp())
        for ch in seq:
            rel = "depends_on" if ch == "E" else "conflict"
            w.ingest(Observation(concepts=["A", "B"], relations=[("A", "B", rel)]))
        ids = {w.concepts.resolve(n).id for n in "AB"}
        edges = [e for e in w.relations.all() if {e.source_id, e.target_id} == ids and e.is_explicit]
        pos = next(e for e in edges if e.relation_type.value != "negative")
        neg = next(e for e in edges if e.relation_type.value == "negative")
        return pos.probability, neg.probability

    swap = str.maketrans("EC", "CE")
    seqs = ["EC", "EEECCC", "E" * 10 + "C" * 10, "EC" * 10, "EEEC" * 5]
    for seq in seqs:
        e1, c1 = beliefs(seq)
        e2, c2 = beliefs(seq.translate(swap))
        assert abs(e1 - c2) < 1e-12 and abs(c1 - e2) < 1e-12, (seq, (e1, c1), (e2, c2))
    e, c = beliefs("E" * 10 + "C" * 10)
    ok(f"axis swap is an exact mirror for {len(seqs)} contested sequences",
       f"E10,C10: dependence {e:.3f} vs conflict {c:.3f} (the later block leads)")


def check_read_path() -> None:
    section("§3.2 Corollary 3.2': activation reads settled values, so a view between uses ignores reflect")
    outs = []
    for reflect in (False, True):
        w = World(store_path=tempfile.mkdtemp())
        for _ in range(6):
            w.ingest(Observation(concepts=["a", "b", "c"], relations=[("a", "b", "depends_on")]))
        w.clock.advance(300)
        if reflect:
            w.reflect(light=True)
        a = w.concepts.resolve("a")
        act = ActivationEngine(w.concepts, w.relations, clock=w.clock).activate([a.id], record=False)
        outs.append({w.concepts.get(k).name: v for k, v in act.items()})
    assert outs[0].keys() == outs[1].keys()
    assert all(abs(outs[0][k] - outs[1][k]) < 1e-9 for k in outs[0]), outs
    ok("same stream, same tick, reflect vs none", f"seed score {outs[0]['a']:.4f} / {outs[1]['a']:.4f}")


def check_task_vocabulary() -> None:
    section("§7.3 Prop 7.7: a word every task label carries does not match another task")
    from world0.schemas.concept import TaskVocabulary, task_match_score

    labels = [f"domain{i} work" for i in range(8)]
    vocab = TaskVocabulary()
    for label in labels:
        vocab.add(label)
    assert vocab.weight("work") == 0.0
    plain = task_match_score("domain1 work", "domain2 work")
    weighted = task_match_score("domain1 work", "domain2 work", vocab)
    assert plain == 0.5 and weighted == 0.0
    ok("'domain1 work' vs 'domain2 work'", f"unweighted {plain:.2f} → weighted {weighted:.2f}")
    w = World(store_path=tempfile.mkdtemp())
    for i in range(20):
        w.ingest(Observation(concepts=[f"x{i % 6}", f"y{i % 4}"], task=labels[i % 5]))
    ref = TaskVocabulary()
    for node in w.concepts.all():
        for label in node.task_profile:
            ref.add(label)
    assert w.concepts.task_vocabulary._labels == ref._labels
    ok("incremental vocabulary equals the one rebuilt from profiles", f"{len(ref)} labels")


def main() -> None:
    check_clock()
    check_gain()
    check_idempotency()
    check_fading_boundary()
    check_schedule_independence()
    check_maturity_schedule_independence()
    check_read_path()
    check_noise_threshold()
    check_one_off()
    check_recurrence_bound()
    check_evidence_thresholds()
    check_gate_table()
    check_salience()
    check_confirm()
    check_relation_settle_exact()
    check_relation_survival()
    check_opposition()
    check_jaccard()
    check_revalidation_count()
    check_noisy_or()
    check_floor()
    check_seed_dominance_and_horizon()
    check_seed_first()
    check_task_vocabulary()
    check_focus()
    check_prediction_calibration()
    check_prediction_per_companion()
    print("\nall propositions verified")


if __name__ == "__main__":
    main()
