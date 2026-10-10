"""Perspective profiles — role-conditioned ways of viewing one concept-world.

A ``Perspective`` (``schemas/context.py``) is the data; this package holds
the ready-made profiles an Agent can name instead of assembling weights
by hand::

    world.project(["latency"], perspective="dependency_map")
    world.project(["latency"], perspective="impact_map", task="incident")

Each profile is a different *reading* of the same relations: the
dependency map follows what a concept relies on, the impact map who
relies on it, the taxonomy reads inclusion structure, the analogy view
favours resonance and the contrast view foregrounds tension.  Profiles
are deliberately few and explicit — a perspective is a documented
strategy, not a tuning knob.
"""

from __future__ import annotations

from world0.schemas.context import Perspective

__all__ = ["PROFILES", "get_perspective", "perspective_names"]


PROFILES: dict[str, Perspective] = {
    "default": Perspective(name="default"),
    # "What does this rely on?"  Follow dependence/enablement outward from
    # the seed; resonance and tension are background.
    "dependency_map": Perspective(
        name="dependency_map",
        role="engineer tracing what a concept relies on",
        relation_type_weights={
            "dependence": 1.4,
            "enables": 1.2,
            "positive": 0.8,
            "parallel": 0.4,
            "negative": 0.5,
        },
        direction_weights={"forward": 1.0, "backward": 0.4},
    ),
    # "What relies on this?"  The same relations read against their
    # direction — the blast radius of a change.
    "impact_map": Perspective(
        name="impact_map",
        role="engineer estimating the blast radius of a change",
        relation_type_weights={
            "dependence": 1.4,
            "enables": 1.2,
            "positive": 0.8,
            "parallel": 0.4,
            "negative": 0.5,
        },
        direction_weights={"forward": 0.4, "backward": 1.0},
    ),
    # "Where does this sit?"  Inclusion and membership structure first.
    "taxonomy": Perspective(
        name="taxonomy",
        role="curator placing a concept in its hierarchy",
        relation_type_weights={
            "inclusion": 1.4,
            "proper_inclusion": 1.4,
            "membership": 1.4,
            "positive": 0.6,
            "parallel": 0.3,
            "negative": 0.4,
        },
    ),
    # "What is this like?"  Resonance over structure.
    "analogy": Perspective(
        name="analogy",
        role="thinker looking for parallels and near-equivalents",
        relation_type_weights={"parallel": 1.4, "positive": 0.5, "negative": 0.4},
    ),
    # "What does this conflict with?"  Tension and exclusion foregrounded.
    "contrast": Perspective(
        name="contrast",
        role="reviewer looking for conflicts and exclusions",
        relation_type_weights={"negative": 1.5, "positive": 0.6, "parallel": 0.4},
    ),
}


def perspective_names() -> list[str]:
    """Names of the built-in profiles, in a stable order."""
    return list(PROFILES)


def get_perspective(name: str | Perspective, *, task: str = "") -> Perspective:
    """Resolve a profile by name (or pass a ``Perspective`` through).

    ``task`` is applied when the profile has no task of its own, so
    ``project(seeds, perspective="taxonomy", task="…")`` keeps task
    affinity.  Unknown names raise ``KeyError`` listing the profiles.
    """
    if isinstance(name, Perspective):
        base = name
    else:
        key = name.strip().lower()
        if key not in PROFILES:
            raise KeyError(
                f"unknown perspective profile {name!r}; "
                f"known: {', '.join(PROFILES)}"
            )
        base = PROFILES[key]
    if task and not base.task:
        return base.model_copy(update={"task": task})
    return base
