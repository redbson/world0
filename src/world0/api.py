"""World 0's interface data shapes (``docs/world0-api.md`` §4).  **Stable.**

These are the only shapes that cross the API boundary: what an Agent
writes (``Statement``, ``ConceptCardInput``) and what it reads back
(``ConceptCard``, ``Claim``).  The internal records (``ConceptNode``,
``RelationEdge``) carry the dynamics and may change freely; they produce
these shapes through ``to_card()`` / ``to_claim()``.

This module imports nothing from the rest of World 0 on purpose: the
shapes are a contract, not a view onto the implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Wire-format version carried by every result (``docs/world0-api.md`` §7).
#: Adding optional fields keeps the version; renaming, removing or
#: changing the meaning of a field bumps it.
API_VERSION = "world0/1"

CLAIM_STATUSES = ("current", "withdrawn", "doubted", "contested", "outvoted", "co_occurrence")


class Statement(BaseModel):
    """One stated relation, read as ``"<source> <relation> <target>"``.

    ``relation`` is any known label or alias (``depends_on``, ``part_of``,
    ``contrasts`` ...); direction follows the label's convention, so
    ``Statement("wheel", "part_of", "car")`` states that the car contains
    the wheel.  ``belief`` is an optional prior from the extractor.
    """

    source: str
    relation: str = "generic_relation"
    target: str
    belief: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = ""

    def __init__(self, source: str | None = None, relation: str | None = None,
                 target: str | None = None, **data) -> None:
        # Positional ``Statement("a", "depends_on", "b")`` as in the design.
        if source is not None:
            data["source"] = source
        if relation is not None:
            data["relation"] = relation
        if target is not None:
            data["target"] = target
        super().__init__(**data)

    def as_triple(self) -> tuple[str, str, str]:
        """``(source, target, relation)``: the order ``Observation.relations`` uses."""
        return (self.source, self.target, self.relation)


class ConceptCardInput(BaseModel):
    """What an Agent may say about a concept besides its name."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    sense: str = ""
    kind: str = ""
    domain: str = ""
    description: str = ""


class ConceptCard(BaseModel):
    """A concept as the Agent reads it: the editable, relation-bearing
    part, without the dynamics internals.

    ``evidence`` says how well the concept is attested (0–1, derived from
    confirmations and disconfirmations); ``confidence`` is its settled
    confidence now; ``tasks`` lists the tasks it has been activated under,
    most frequent first; ``last_seen_tick`` is cognitive time (the index
    of the last observation that mentioned it).
    """

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    sense: str = ""
    kind: str = ""
    domain: str = ""
    description: str = ""
    maturity: str = "embryonic"
    evidence: float = 0.0
    confidence: float = 0.0
    tasks: list[str] = Field(default_factory=list)
    last_seen_tick: int = 0
    first_seen_tick: int = 0
    sources: list[str] = Field(default_factory=list)
    # In long-term memory: well evidenced and recurring in spaced windows,
    # so it forgets on the slow curve (one halving per 35 040 observations).
    long_term: bool = False


class Claim(BaseModel):
    """One claim between two concepts as the Agent reads it.

    ``relation`` is the canonical semantic relation (``dependence``,
    ``inclusion``, ``contrast`` ...), ``axis`` the cognitive axis it lives
    on, ``belief`` its current belief, ``support`` the number of times it
    was stated (0 for a co-occurrence edge), ``stated_under`` the tasks it
    was stated under, ``text`` a one-line sentence, ``since_tick`` /
    ``until_tick`` the observation indices at which it was first stated and
    withdrawn.  ``status`` is one of
    ``CLAIM_STATUSES``: ``current`` (holds), ``withdrawn`` (no longer
    holds), ``doubted`` (argued below even odds), ``contested`` (an
    opposing claim is close), ``outvoted`` (an opposing claim or another
    label for the pair was stated far more often), ``co_occurrence`` (seen
    together; asserts nothing).
    """

    source: str
    source_id: str = ""
    relation: str
    target: str
    target_id: str = ""
    axis: str = "parallel"
    belief: float = 0.0
    support: int = 0
    status: str = "current"
    stated_under: list[str] = Field(default_factory=list)
    text: str = ""
    # Cognitive time (observation index) at which the claim was first
    # stated, and at which it was withdrawn (None while it holds).  System
    # time of the world, not the valid time of the fact: World 0 records
    # when it was told, not when the fact was true.
    since_tick: int = 0
    until_tick: int | None = None

    def as_statement(self) -> Statement:
        """The statement that would restate (or withdraw) this claim."""
        return Statement(self.source, self.relation, self.target)


__all__ = [
    "API_VERSION",
    "CLAIM_STATUSES",
    "Claim",
    "ConceptCard",
    "ConceptCardInput",
    "Statement",
]
