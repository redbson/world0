"""Real LLM extraction for LongRun: the production prompt, cached raw outputs.

The default LongRun extractor reads the gold structure of each event and
injects controlled errors (``GenConfig.extract_p``).  This module replaces
it with what World 0 actually ships: every event's text goes through the
production extraction prompt (``extraction.concepts_relations.system``) to
an LLM, and the raw JSON is parsed by the production parser
(``ConceptExtractor._parse_response``).  Raw outputs are cached per
(seed, horizon), so a stream extracted once replays identically.

    # 1. write the prompts (no gold in the batch files)
    python -m benchmarks.longrun.llm_extract export --seed 0 --horizon 600 --out DIR
    # 2. an LLM answers DIR/batch_*.json into DIR/answers_*.json ({id: raw}),
    #    either through a provider ...
    python -m benchmarks.longrun.llm_extract answer --dir DIR --model gpt-5.4-nano
    #    ... or by any agent given only system.txt and one batch file
    # 3. collect into the cache, then measure the extractor against gold
    python -m benchmarks.longrun.llm_extract collect --seed 0 --horizon 600 --dir DIR
    python -m benchmarks.longrun.llm_extract profile --seed 0 --horizon 600

Runs then use ``GenConfig(extract_mode="llm")`` (``run.py --study llm``).
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
from collections import Counter

from benchmarks.longrun.worldgen import Claim, Event, Extraction, GenConfig, Stream

CACHE_DIR = os.path.join(os.path.dirname(__file__), "llm_cache")
# Canonical World 0 semantics → the benchmark's relation vocabulary; any
# other label the extractor chooses is kept as is (a wrongly typed claim).
_SEM2REL = {"dependence": "depends_on", "inclusion": "contains", "conflict": "conflict", "enables": "enables"}


def cache_path(seed: int, horizon: int) -> str:
    return os.path.join(CACHE_DIR, f"seed{seed}_h{horizon}.json")


@functools.lru_cache(maxsize=8)
def load_cache(seed: int, horizon: int) -> dict[str, str]:
    path = cache_path(seed, horizon)
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError as exc:
        raise FileNotFoundError(
            f"no LLM extraction cache for seed {seed}, horizon {horizon} ({path}); "
            "see benchmarks/longrun/llm_extract.py"
        ) from exc


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\W]+", " ", name.lower())).strip()


def _parser():
    from world0.extraction.extractor import ConceptExtractor

    return ConceptExtractor.__new__(ConceptExtractor)  # parsing needs no provider


def system_prompt() -> str:
    from world0.prompts import PromptRegistry

    return PromptRegistry().render("extraction.concepts_relations.system")


def user_prompt(ev: Event) -> str:
    from world0.extraction.extractor import ConceptExtractor

    return ConceptExtractor._build_user_prompt(ev.text, task=ev.task, source=f"step{ev.step}")


def to_extraction(raw: str, ev: Event, known: dict[str, str]) -> Extraction:
    """Parse one raw LLM answer with the production parser into a benchmark Extraction.

    ``known`` maps normalised names to the world's concept names, so an
    extracted "zakonax router" is the same concept as "Zakonax router";
    names the world does not have are kept (spurious concepts).  Withdrawn
    and contradicted relations both become retractions: the baselines have
    no other channel for "do not hold this now".
    """
    obs = _parser()._parse_response(raw, task=ev.task, source=f"step{ev.step}")
    by_uid = {c.uid: c.name for c in obs.concept_candidates if c.uid}

    def name(ref: str) -> str:
        n = by_uid.get(ref, ref)
        return known.get(_key(n), n)

    def claims(triples) -> list[Claim]:
        out = []
        for s, t, sem in triples:
            a, b = name(s), name(t)
            if a != b:
                c = Claim.make(a, _SEM2REL.get(sem, sem), b)
                if c not in out:
                    out.append(c)
        return out

    concepts = []
    for n in (name(c) for c in obs.concepts):
        if n not in concepts:
            concepts.append(n)
    got = claims(obs.relations)
    retractions = claims([*obs.retracted_relations, *obs.contradicted_relations])
    for c in got:
        for x in (c.src, c.tgt):
            if x not in concepts:
                concepts.append(x)
    keep = ev.ticket_claim is not None and ev.ticket_claim in got
    return Extraction(concepts, got, retractions, ev.ticket if keep else None, ev.ticket_claim if keep else None)


def _events(seed: int, horizon: int) -> list[Event]:
    return [ev for ev, _ in Stream(GenConfig(seed=seed, horizon=horizon)).events()]


def export(seed: int, horizon: int, out: str, batch: int) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "system.txt"), "w") as fh:
        fh.write(system_prompt())
    evs = _events(seed, horizon)
    for i in range(0, len(evs), batch):
        items = [{"id": str(ev.step), "user": user_prompt(ev)} for ev in evs[i:i + batch]]
        with open(os.path.join(out, f"batch_{i // batch:03d}.json"), "w") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=0)
    print(f"{len(evs)} events in {(len(evs) + batch - 1) // batch} batches → {out}")


def answer(directory: str, model: str) -> None:
    """Answer every batch through a World 0 LLM provider (needs its API key)."""
    from world0.agents.provider import create_provider

    provider = create_provider(model)
    system = open(os.path.join(directory, "system.txt")).read()
    for path in sorted(glob.glob(os.path.join(directory, "batch_*.json"))):
        out = path.replace("batch_", "answers_")
        done = json.load(open(out)) if os.path.exists(out) else {}
        for item in json.load(open(path)):
            if item["id"] not in done:
                done[item["id"]] = provider.complete_json(system, item["user"])
        with open(out, "w") as fh:
            json.dump(done, fh, ensure_ascii=False)


def collect(seed: int, horizon: int, directory: str) -> None:
    raw: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(directory, "answers_*.json"))):
        for k, v in json.load(open(path)).items():
            raw[str(k)] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    missing = [str(ev.step) for ev in _events(seed, horizon) if str(ev.step) not in raw]
    if missing:
        raise SystemExit(f"{len(missing)} events unanswered, e.g. {missing[:5]}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(seed, horizon), "w") as fh:
        json.dump(raw, fh, ensure_ascii=False, sort_keys=True)
    print(f"{len(raw)} answers → {cache_path(seed, horizon)}")


def profile(seed: int, horizon: int) -> dict:
    """The extractor measured against gold, in the simulator's error categories."""
    cfg = GenConfig(seed=seed, horizon=horizon, extract_mode="llm")
    s = Stream(cfg)
    n = Counter({k: 0 for k in ("kept", "relabelled", "dropped", "spurious", "retraction_kept",
                                 "retraction_missed", "retraction_spurious")})
    for ev, _ in s.events():
        x = ev.extracted
        gold = set(ev.claims)
        pairs = {frozenset((c.src, c.tgt)): c for c in gold}
        for c in gold:
            if c in x.claims:
                n["kept"] += 1
            elif any(frozenset((y.src, y.tgt)) == frozenset((c.src, c.tgt)) for y in x.claims):
                n["relabelled"] += 1
            else:
                n["dropped"] += 1
        for y in x.claims:
            if y not in gold and frozenset((y.src, y.tgt)) not in pairs:
                n["spurious"] += 1
        for r in ev.retractions:
            n["retraction_kept" if r in x.retractions else "retraction_missed"] += 1
        for r in x.retractions:
            if r not in ev.retractions:
                n["retraction_spurious"] += 1
        n["events"] += 1
        n["events_task" if ev.kind == "task" else "events_noise"] += 1
        n["concepts_extracted"] += len(x.concepts)
        n["concepts_unknown"] += sum(1 for c in x.concepts if c not in s.world.all_concepts)
    g = n["kept"] + n["relabelled"] + n["dropped"]
    r = n["retraction_kept"] + n["retraction_missed"]
    rates = {
        "claims_gold": g,
        "drop": n["dropped"] / max(g, 1),
        "relabel": n["relabelled"] / max(g, 1),
        "spurious_per_event": n["spurious"] / max(n["events"], 1),
        "retractions_gold": r,
        "missed_retraction": n["retraction_missed"] / max(r, 1),
        "spurious_retractions": n["retraction_spurious"],
        "unknown_concepts_per_event": n["concepts_unknown"] / max(n["events"], 1),
        **dict(n),
    }
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "collect", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--horizon", type=int, default=600)
        if name == "export":
            p.add_argument("--out", required=True)
            p.add_argument("--batch", type=int, default=50)
        if name == "collect":
            p.add_argument("--dir", required=True)
    p = sub.add_parser("answer")
    p.add_argument("--dir", required=True)
    p.add_argument("--model", default="gpt-5.4-nano")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.seed, a.horizon, a.out, a.batch)
    elif a.cmd == "answer":
        answer(a.dir, a.model)
    elif a.cmd == "collect":
        collect(a.seed, a.horizon, a.dir)
    else:
        print(json.dumps(profile(a.seed, a.horizon), indent=1))


if __name__ == "__main__":
    main()
