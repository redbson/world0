"""Sustained focus — a limited-capacity workspace across projections.

Global workspace theory (``docs/mc/03-workspace.md``) describes a
limited-capacity workspace whose current contents shape what is attended
next (indicator GWT-4, state-dependent attention), entered through a
non-linear "ignition" (GWT-2).  Without it every projection started from
nothing: after two Ops-focused views, projecting the bridge concept
``deployment`` gave exactly the cold-start view.

``Focus`` is that workspace for World 0, kept deliberately small and
transient — an attention state, not memory:

- **ignition**: after a projection, a selected concept whose relevance
  reaches ``IGNITION_THRESHOLD`` of the view's strongest non-seed
  relevance (seeds always) enters at full strength, all-or-none;
- **capacity**: at most ``FOCUS_CAPACITY`` concepts, strongest kept;
- **retention**: every later projection multiplies existing strengths by
  ``FOCUS_RETENTION``; items below ``FOCUS_MIN_STRENGTH`` leave;
- **release**: a projection under a different explicit task clears it;
- **bias, never injection**: the next projection boosts only candidates
  its own seeds already reached (members by their strength, direct
  neighbours of members by ``FOCUS_NEIGHBOR_SHARE`` of it), so a view of
  unrelated seeds is untouched and focus cannot perseverate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from world0.schemas.concept import normalize_task_label

FOCUS_CAPACITY: int = 7
IGNITION_THRESHOLD: float = 0.5
FOCUS_RETENTION: float = 0.6
FOCUS_MIN_STRENGTH: float = 0.15
FOCUS_NEIGHBOR_SHARE: float = 0.5


class Focus:
    """The concepts currently held in the workspace, with strengths."""

    def __init__(self) -> None:
        self._items: dict[str, float] = {}
        self.task: str = ""

    def items(self) -> dict[str, float]:
        return dict(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self.task = ""

    def affinity(
        self,
        candidate_ids: Iterable[str],
        neighbors: Mapping[str, set[str]],
        *,
        exclude: Iterable[str] = (),
    ) -> dict[str, float]:
        """Focus affinity in ``[0, 1]`` for candidates of the next view.

        Concepts in ``exclude`` (the new view's seeds) pass no affinity to
        their neighbours: a seed that is also in focus says nothing about
        *which* of its neighbours the line of attention was about, and a
        bridge seed would otherwise spread focus into every cluster it
        touches.
        """
        if not self._items:
            return {}
        blocked = set(exclude)
        out: dict[str, float] = {}
        for cid in candidate_ids:
            score = self._items.get(cid, 0.0)
            for other in neighbors.get(cid, ()):
                if other in blocked:
                    continue
                strength = self._items.get(other)
                if strength:
                    score = max(score, FOCUS_NEIGHBOR_SHARE * strength)
            if score > 0.0:
                out[cid] = score
        return out

    def release_if_task_changed(self, task: str) -> bool:
        """Clear the workspace when an explicit new task begins."""
        new = normalize_task_label(task)
        if new and self.task and new != self.task:
            self.clear()
            return True
        return False

    def update(self, ignited: Iterable[str], task: str = "") -> None:
        """Advance one projection: retain, then ignite, then cap."""
        kept = {
            cid: strength * FOCUS_RETENTION
            for cid, strength in self._items.items()
            if strength * FOCUS_RETENTION >= FOCUS_MIN_STRENGTH
        }
        for cid in ignited:
            kept[cid] = 1.0
        ranked = sorted(kept.items(), key=lambda kv: (-kv[1], kv[0]))
        self._items = dict(ranked[:FOCUS_CAPACITY])
        label = normalize_task_label(task)
        if label:
            self.task = label
