"""Natural (non-templated) surface text for LongRun events.

The template text states every claim as "<A> depends on <B>." with the
relation's canonical phrase, so a real LLM extractor makes almost no errors
on it (round 25).  ``GenConfig(text_style="natural")`` writes the same events
the way people write work notes instead:

- each relation in many phrasings, active and passive ("B is a prerequisite
  for A", "C sits inside A", "A and D can't run together");
- a later claim about the same subject refers back with a pronoun ("It also
  bundles C");
- names in lower case or with an article ("the zakonax router");
- withdrawals in varied wording ("A stopped relying on B");
- negated distractors that state no claim ("someone asked whether A needs E —
  it doesn't");
- the ticket and the co-mentioned concepts in varied wording.

Only the words change: which events happen, what they state, and every gold
set are drawn from the generator's main random stream exactly as in the
template style, while the wording uses its own per-event stream.  So a
template run and a natural run of one seed are the same stream, and any
difference between them is the extractor's.
"""

from __future__ import annotations

import random

# Subject-first verb phrases ("<A> ..."): they can follow a pronoun.
_VP = {
    "depends_on": ["depends on {b}", "needs {b}", "relies on {b}", "can't start without {b}",
                   "requires {b}", "calls into {b} for everything it does"],
    "contains": ["contains {b}", "includes {b}", "has {b} inside it", "bundles {b}",
                 "is made up partly of {b}"],
    "conflict": ["conflicts with {b}", "clashes with {b}", "can't run alongside {b}",
                 "is incompatible with {b}"],
    "enables": ["enables {b}", "makes {b} possible", "unlocks {b}", "is what lets {b} work at all"],
}
# Object-first or symmetric forms: always spelled out in full.
_OTHER = {
    "depends_on": ["{b} is a prerequisite for {a}", "{b} has to be up before {a} can do anything",
                   "without {b}, {a} won't work"],
    "contains": ["{b} is part of {a}", "{b} sits inside {a}", "{b} is one of the pieces of {a}"],
    "conflict": ["{a} and {b} don't get along", "{a} and {b} are mutually exclusive",
                 "you can't have {a} and {b} running together"],
    "enables": ["{b} is made possible by {a}", "thanks to {a}, {b} becomes possible",
                "{b} is enabled by {a}"],
}
_WITHDRAW = {
    "depends_on": ["{a} no longer needs {b}", "we removed {a}'s dependency on {b}",
                   "{a} stopped relying on {b}", "turns out {a} doesn't depend on {b} anymore"],
    "contains": ["{b} was moved out of {a}", "{a} no longer includes {b}"],
    "conflict": ["{a} and {b} no longer clash", "the conflict between {a} and {b} is gone"],
    "enables": ["{a} no longer enables {b}", "{a} stopped making {b} possible"],
}
_WITHDRAW_LEAD = ["Update: ", "Correction: ", "Heads up, ", "Change of plan: ", ""]
_NEGATED = ["someone asked whether {a} needs {b}; it doesn't",
            "{a} does not depend on {b}, despite what the old diagram showed",
            "to be clear, {b} is not part of {a}",
            "we checked: {a} and {b} do not conflict"]
_LEADS = ["Working on {t}:", "Back on {t}.", "{t} session.", "Notes from {t}:", "Continuing {t}."]
_TICKET = [" (ticket {k})", ", tracked in {k},", " (see {k})", " — filed as {k} —"]  # after the subject
_TICKET_END = [" (ticket {k})", ", tracked in {k}", " (see {k})", " — filed as {k}"]
_TOUCHED = ["Also touched: {x}.", "{x} came up too.", "We also looked at {x}.",
            "Side note on {x}, nothing decided."]
_CHATTER = ["Chatter about {x}.", "Random chat mentioned {x}.", "Over lunch people brought up {x}.",
            "Someone forwarded a thread on {x}."]


def _join(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class Realiser:
    """Writes one event's text; ``rng`` is the event's own stream."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def name(self, n: str) -> str:
        r = self.rng.random()
        if r < 0.25:
            return n.lower()
        if r < 0.5:
            return "the " + n
        return n

    def claim(self, c, prev_src: str | None, ticket: str | None) -> tuple[str, str]:
        """One claim as a sentence; a pronoun when it continues the previous subject."""
        rng = self.rng
        if rng.random() < 0.55:
            tag = rng.choice(_TICKET).format(k=ticket) if ticket else ""
            vp = rng.choice(_VP[c.rel]).format(b=self.name(c.tgt))
            if prev_src == c.src and rng.random() < 0.7:
                subj = rng.choice(["It", "It also", "That one also"])
            else:
                subj = _cap(self.name(c.src))
            return f"{subj}{tag} {vp}.", c.src
        tag = rng.choice(_TICKET_END).format(k=ticket) if ticket else ""
        a, b = self.name(c.src), self.name(c.tgt)
        if c.rel == "conflict" and rng.random() < 0.5:
            a, b = b, a
        # the grammatical subject is not the source here: no pronoun may follow
        return _cap(rng.choice(_OTHER[c.rel]).format(a=a, b=b)) + f"{tag}.", None

    def task_text(self, step: int, task: str, stated: list, retire, rest: list[str],
                  ticket: str | None, filler: str, negated: tuple[str, str] | None) -> str:
        rng = self.rng
        parts = [f"[step {step}] " + rng.choice(_LEADS).format(t=task)]
        body: list[str] = []
        prev = None
        for i, c in enumerate(stated):
            s, prev = self.claim(c, prev, ticket if i == 0 else None)
            body.append(s)
        # other sentences go before or after the claims, never between a
        # claim and the pronoun that refers back to it
        if retire:
            w = rng.choice(_WITHDRAW[retire.rel]).format(a=self.name(retire.src), b=self.name(retire.tgt))
            line = _cap(rng.choice(_WITHDRAW_LEAD) + w) + "."
            body.insert(0 if rng.random() < 0.4 else len(body), line)
        if negated:
            a, b = negated
            line = _cap(rng.choice(_NEGATED).format(a=self.name(a), b=self.name(b))) + "."
            body.insert(0 if rng.random() < 0.5 else len(body), line)
        parts += body
        if rest:
            parts.append(_cap(rng.choice(_TOUCHED).format(x=_join([self.name(x) for x in rest]))))
        if filler:
            parts.append(filler)
        return " ".join(parts)

    def noise_text(self, step: int, concepts: list[str], filler: str) -> str:
        text = f"[step {step}] " + _cap(self.rng.choice(_CHATTER).format(x=_join([self.name(x) for x in concepts])))
        return text + (" " + filler if filler else "")


def event_rng(seed: int, step: int) -> random.Random:
    return random.Random(seed * 1_000_033 + step * 7 + 1)
