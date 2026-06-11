"""Perspective profiles — named, reusable lenses over the concept-world.

CLAUDE.md reserves the ``perspectives/`` module for "role-specific
projection strategies, perspective profiles".  A profile is just a
``Perspective`` with a name: built-in ones ship with the package,
custom ones persist in the world's ``state.json`` under a
``"perspectives"`` key (additive — older stores load unchanged).

Profiles overlay the global RELATION_TYPE_FACTOR and the hand-set
SEMANTIC_RELATION_SPECS strengths; they never rewrite them.
"""

from __future__ import annotations

from typing import Callable

from world0.schemas.context import Perspective


def builtin_profiles() -> dict[str, Perspective]:
    """The perspective presets that ship with World 0.

    Each preset biases *which relations carry activation* and *how the
    projection renders*, so the same world answers differently when
    debugged, designed against, or researched.
    """
    return {
        "default": Perspective(
            name="default",
            description="Neutral lens — global defaults, no overrides.",
        ),
        "debug": Perspective(
            name="debug",
            role="engineer",
            description=(
                "Causal lens: dependency and enablement chains dominate; "
                "conflicts stay loud; loose resonance is background noise."
            ),
            semantic_relation_weights={
                "dependence": 1.4,
                "enables": 1.3,
                "conflict": 1.2,
                "overlap": 0.5,
            },
            relation_type_weights={"parallel": 0.6},
            # Debugging needs conflicts and instability *visible*, not
            # just inhibited — opt the conditional relations in.
            negative_visibility={
                "conflict": "expose",
                "instability": "expose",
            },
            render_style="compact",
        ),
        "design": Perspective(
            name="design",
            role="architect",
            description=(
                "Structural lens: overlap, equivalence and co-creation "
                "surface analogies and composition opportunities."
            ),
            semantic_relation_weights={
                "overlap": 1.4,
                "equivalence": 1.3,
                "co_creation": 1.3,
                "dependence": 0.8,
            },
            render_style="default",
        ),
        "research": Perspective(
            name="research",
            role="researcher",
            description=(
                "Exploratory lens: similarity and overlap pull in adjacent "
                "territory; detailed rendering keeps the why visible."
            ),
            semantic_relation_weights={
                "similarity_kernel": 1.4,
                "overlap": 1.3,
                "persistent_attention": 1.2,
            },
            render_style="detailed",
        ),
    }


class PerspectiveRegistry:
    """Named perspectives: built-ins plus custom ones persisted in state.

    Custom profiles shadow built-ins of the same name.  The registry
    holds a reference to the world's mutable state dict and a save
    callback — it never touches the store directly, so the facade keeps
    owning the flush boundary.
    """

    _STATE_KEY = "perspectives"

    def __init__(
        self, state: dict, save: Callable[[], None]
    ) -> None:
        self._state = state
        self._save = save
        self._builtin = builtin_profiles()

    def get(self, name: str) -> Perspective | None:
        """Resolve a profile by name — custom first, then built-in."""
        raw = self._custom().get(name)
        if raw is not None:
            return Perspective.model_validate(raw)
        return self._builtin.get(name)

    def put(self, perspective: Perspective) -> None:
        """Persist a custom profile (overwrites same-named custom)."""
        custom = self._custom()
        custom[perspective.name] = perspective.model_dump(mode="json")
        self._state[self._STATE_KEY] = custom
        self._save()

    def remove(self, name: str) -> bool:
        """Drop a custom profile.  Built-ins cannot be removed."""
        custom = self._custom()
        if name not in custom:
            return False
        del custom[name]
        self._state[self._STATE_KEY] = custom
        self._save()
        return True

    def names(self) -> list[str]:
        """All resolvable profile names, built-in and custom."""
        return sorted(set(self._builtin) | set(self._custom()))

    def _custom(self) -> dict:
        value = self._state.get(self._STATE_KEY)
        return dict(value) if isinstance(value, dict) else {}
