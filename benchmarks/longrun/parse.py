"""Recover claims from what a system actually put in the prompt.

Scoring reads the rendered text, not the projection object behind it, so a
claim that the renderer dropped (the shipped ``Projection.render()`` prints
at most ten relations) is not credited.
"""

from __future__ import annotations

import re

from benchmarks.longrun.worldgen import REL_PHRASE, Claim

SEM2REL = {"dependence": "depends_on", "inclusion": "contains",
           "conflict": "conflict", "enables": "enables"}

_CONCEPT = re.compile(r"^- \*\*(\S+)\*\* \((.+?), (?:embryonic|developing|established|core|fading),")
_RELATION = re.compile(r"^- (\S+) → (\w+) \[(\w+)\] → (\S+) \((?:belief: ([0-9.]+)|co-occurrence)")
_PHRASES = "|".join(re.escape(p) for p in REL_PHRASE.values())
_SENTENCE = re.compile(rf"^(.+?) ({_PHRASES}) (.+?)(?: \(belief ([0-9.]+)\))?\.?$")
_PHRASE2REL = {v: k for k, v in REL_PHRASE.items()}


def parse_shipped(text: str) -> tuple[set[Claim], set[str], dict[Claim, float]]:
    """``Projection.render()``: concept lines name the ids used in the relation lines."""
    names: dict[str, str] = {}
    for line in text.splitlines():
        m = _CONCEPT.match(line)
        if m:
            names[m.group(1)] = m.group(2)
    claims: set[Claim] = set()
    beliefs: dict[Claim, float] = {}
    for line in text.splitlines():
        m = _RELATION.match(line)
        if not m:
            continue
        src, sem, _axis, tgt, belief = m.groups()
        rel = SEM2REL.get(sem)
        if rel and belief is not None and src in names and tgt in names:
            c = Claim.make(names[src], rel, names[tgt])
            claims.add(c)
            beliefs[c] = max(beliefs.get(c, 0.0), float(belief))
    return claims, set(names.values()), beliefs


def parse_compact(text: str) -> tuple[set[Claim], set[str], dict[Claim, float]]:
    """``Projection.render(style="compact")``: claim bullets before the first section.

    Withdrawn claims, claims from other tasks and the "Hold loosely"
    section come under their own ``Label:`` lines and are not credited as
    claims; ``Also relevant`` names the other concepts in view.
    """
    claims: set[Claim] = set()
    concepts: set[str] = set()
    beliefs: dict[Claim, float] = {}
    in_claims = True
    for line in text.splitlines():
        if line.startswith("Also relevant: "):
            concepts |= {x.strip() for x in line[len("Also relevant: "):].rstrip(".").split(",") if x.strip()}
            continue
        if line.endswith(":") and not line.startswith("- "):
            in_claims = False  # "Definitions:", "No longer holds:", …
            continue
        if not (in_claims and line.startswith("- ")):
            continue
        m = _SENTENCE.match(line[2:])
        if m:
            a, phrase, b, belief = m.groups()
            c = Claim.make(a, _PHRASE2REL[phrase], b)
            claims.add(c)
            concepts |= {a, b}
            if belief is not None:
                beliefs[c] = float(belief)
    return claims, concepts, beliefs
