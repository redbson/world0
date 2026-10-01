"""``world0`` — the command-line entry point of the unified API (``docs/world0-api.md`` §5).

One subcommand per operation in ``world0.ops``; ``--json`` prints exactly
the JSON the HTTP surface returns, the default prints a human form
(``project`` prints the prompt-ready render).  The Agent shell keeps its
own ``pkm`` command.

    world0 --store .world0 project api db --task backend
    world0 ingest obs.json            # or: cat obs.json | world0 ingest -
    world0 state api depends_on db --task backend
    world0 card api --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from world0.api import API_VERSION
from world0.ops import ApiError, call
from world0.world import World


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="world0", description=f"World 0 unified API ({API_VERSION})")
    p.add_argument("--store", default=".world0", help="store path (default .world0)")
    p.add_argument("--backend", default="auto", choices=["auto", "json", "sqlite"])
    p.add_argument("--json", action="store_true", help="print the JSON result (same as the HTTP surface)")
    # ``--json`` is also accepted after the subcommand; SUPPRESS keeps the
    # subparser from clobbering the top-level value.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    top_sub = p.add_subparsers(dest="op", required=True)

    class _Sub:
        def add_parser(self, name, **kw):
            return top_sub.add_parser(name, parents=[common], **kw)

    sub = _Sub()

    s = sub.add_parser("ingest", help="write one observation from a JSON file ('-' for stdin)")
    s.add_argument("file")

    s = sub.add_parser("ingest-text", help="extract with the configured LLM, then ingest")
    s.add_argument("text")
    s.add_argument("--task", default="")
    s.add_argument("--source", default="")

    s = sub.add_parser("project", help="task-relevant local view around the seeds")
    s.add_argument("seeds", nargs="+")
    s.add_argument("--task", default="")
    s.add_argument("--perspective", default="")
    s.add_argument("--max-concepts", type=int, default=15)
    s.add_argument("--max-depth", type=int, default=2)

    s = sub.add_parser("card", help="one concept's card")
    s.add_argument("concept")

    s = sub.add_parser("claims", help="every claim about one concept")
    s.add_argument("concept")
    s.add_argument("--task", default="")

    s = sub.add_parser("find", help="concepts resembling a text")
    s.add_argument("q")
    s.add_argument("--limit", type=int, default=5)

    sub.add_parser("status", help="world status")

    s = sub.add_parser("reflect", help="consolidate")
    s.add_argument("--light", action="store_true")

    s = sub.add_parser("merge", help="two concepts are one")
    s.add_argument("keeper")
    s.add_argument("absorbed")

    s = sub.add_parser("split", help="one concept is two")
    s.add_argument("source")
    s.add_argument("new_name")
    s.add_argument("--alias", action="append", default=[], dest="aliases_to_move")
    s.add_argument("--description", default="")

    s = sub.add_parser("weaken", help="doubt about a concept itself")
    s.add_argument("concept")
    s.add_argument("--source", default="")
    s.add_argument("--task", default="")

    for verb, help_ in (("state", "one statement: <source> <relation> <target>"),
                        ("withdraw", "one withdrawal: the claim no longer holds"),
                        ("deny", "one denial: lowers the claim's belief")):
        s = sub.add_parser(verb, help=help_)
        s.add_argument("source")
        s.add_argument("relation")
        s.add_argument("target")
        s.add_argument("--task", default="")
        s.add_argument("--source-label", default="", dest="source_label")
        if verb == "state":
            s.add_argument("--belief", type=float, default=None)
    return p


def _params(ns: argparse.Namespace) -> dict[str, Any]:
    skip = {"store", "backend", "json", "op"}
    params = {k: v for k, v in vars(ns).items() if k not in skip and v is not None}
    if ns.op == "ingest":
        raw = sys.stdin.read() if ns.file == "-" else open(ns.file, encoding="utf-8").read()
        return json.loads(raw)
    return params


def _human(op: str, result: dict[str, Any]) -> str:
    if op == "project":
        return result["text"]
    if op == "card" or op in ("merge", "split", "weaken"):
        card = result.get("keeper") or result.get("new") or result.get("card") or result
        lines = [f"{card['name']} ({card['maturity']}, evidence {card['evidence']:.2f}, confidence {card['confidence']:.2f})"]
        if card.get("sense"):
            lines.append(f"  sense: {card['sense']}")
        if card.get("description"):
            lines.append(f"  {card['description']}")
        if card.get("aliases"):
            lines.append(f"  aliases: {', '.join(card['aliases'])}")
        if card.get("tasks"):
            lines.append(f"  tasks: {', '.join(card['tasks'])}")
        return "\n".join(lines)
    if op == "claims":
        return "\n".join(f"- {c['text']} (belief {c['belief']:.2f}, {c['status']})" for c in result["claims"]) or "(no claims)"
    if op == "find":
        return "\n".join(f"- {m['card']['name']} ({m['score']:.2f})" for m in result["matches"]) or "(no match)"
    if op == "status":
        keys = ("cognitive_tick", "total_concepts", "total_relations", "by_maturity")
        return "\n".join(f"{k}: {result[k]}" for k in keys if k in result)
    if op == "reflect":
        return ", ".join(f"{k} {len(v)}" for k, v in result.items() if isinstance(v, list)) or "nothing to do"
    # ingest-like results
    parts = [f"{k.replace('_', ' ')}: {', '.join(v)}" for k, v in result.items()
             if isinstance(v, list) and v and all(isinstance(x, str) for x in v)]
    return "\n".join(parts) or "nothing changed"


def main(argv: list[str] | None = None, *, stdout=None) -> int:
    out = stdout or sys.stdout
    ns = build_parser().parse_args(argv)
    op = ns.op.replace("-", "_")
    world = World(store_path=ns.store, backend=ns.backend)
    try:
        try:
            result = call(world, op, _params(ns))
        except ApiError as exc:
            if ns.json:
                print(json.dumps(exc.as_json(), ensure_ascii=False), file=out)
            else:
                print(f"error ({exc.code}): {exc.message}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False) if ns.json else _human(op, result), file=out)
        return 0
    finally:
        world.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
