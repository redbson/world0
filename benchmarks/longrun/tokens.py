"""One token-count estimate applied identically to every system.

Common English words (everything the event templates, filler and relation
phrases use) count as one token; any other alphabetic piece (the invented
concept names, hex ids) counts ``ceil(len / 3)``, digits ``ceil(len / 3)``
per run, punctuation one each.  It is a proxy for a BPE tokenizer, not a
tokenizer; ``chars / 4`` is reported alongside it as a sensitivity check.
"""

from __future__ import annotations

import math
import re

_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")

FILLER_WORDS = (
    "we also reviewed the earlier notes and agreed that the plan still looks "
    "reasonable given what was discussed before so the next step is to check "
    "again after lunch with the rest of the group because several people asked "
    "about timing while others wanted more detail on how things were going "
    "overall it seemed fine although a few questions remain open for now and "
    "should be revisited when there is time later this week"
).split()

COMMON = set(FILLER_WORDS) | {
    "step", "working", "on", "work", "depends", "contains", "conflicts", "with",
    "enables", "correction", "no", "longer", "also", "touched", "chatter",
    "about", "ticket", "summary", "of", "earlier", "x", "a", "the", "and", "to",
    "belief", "relevant", "what", "which", "was", "raised", "when", "we", "noted",
    "that", "does", "ultimately", "rely", "right", "now", "how", "fit", "here",
    "investigate", "else", "is", "involved", "depend", "look", "into", "area",
    "cognitive", "context", "key", "relations", "concepts", "emerging",
    "established", "core", "fading", "developing", "embryonic", "confidence",
    "evidence", "inclusion", "dependence", "conflict", "positive", "negative",
    "parallel", "structural", "propagation", "reinforced", "co", "occurrence",
    "generic", "relation", "why", "these", "via", "from", "used", "in", "this",
    "task", "before", "activated", "for", "next", "epistemic", "status",
    "contested", "leaning", "thin", "misc", "scratch",
} | {h for h in (
    "service cache queue index gateway planner store router monitor loader "
    "encoder compiler registry scheduler validator resolver broker ledger "
    "sampler allocator").split()}


def est_tokens(text: str) -> int:
    n = 0
    for m in _PIECE.findall(text):
        if m[0].isalpha():
            n += 1 if m.lower() in COMMON else max(1, math.ceil(len(m) / 3))
        elif m[0].isdigit():
            n += max(1, math.ceil(len(m) / 3))
        else:
            n += 1
    return n


def chars4(text: str) -> int:
    return math.ceil(len(text) / 4)
