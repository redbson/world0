"""Cognitive time for World 0.

World 0 measures time in **observations**, not seconds.  One ``ingest()``
advances the world's clock by one *tick*; decay, freshness and every other
temporal factor are functions of how many observations have passed since
a concept or relation was last touched.  An Agent that processes a thousand
observations in an hour therefore ages its concept-world a thousand times
faster than one that processes a single observation a day — time in the
concept-world is proportional to cognitive activity, and the calendar has
no say in how quickly an unused concept fades.

The wall clock is kept only as a *secondary drift term*: while the world is
idle (no ingests), real time still advances cognitive time at
``WALL_TICKS_PER_HOUR`` ticks per hour, so a world reopened after months of
silence has aged a little instead of being frozen exactly where it was
left.  At the default rate an idle day ≈ 2.4 ticks and an idle year ≈ 876
ticks — a fraction of what an active week of observations contributes.

Every timestamped record carries both coordinates (``*_tick`` and the
``datetime``); ``cognitive_elapsed`` combines them.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Ticks that one idle wall-clock hour is worth.  Small on purpose: the
# calendar is a tie-breaker for dormant worlds, not the primary clock.
WALL_TICKS_PER_HOUR: float = 0.1


def wall_now() -> datetime:
    return datetime.now(timezone.utc)


def cognitive_elapsed(
    now_tick: int,
    since_tick: int,
    now_time: datetime | None = None,
    since_time: datetime | None = None,
    *,
    drift: float = WALL_TICKS_PER_HOUR,
) -> float:
    """Cognitive time elapsed between two instants, in ticks.

    ``ticks elapsed + drift × wall-clock hours elapsed``.  Both components
    are clamped at zero so a record stamped "in the future" (clock reset,
    merged store) never yields negative elapsed time.
    """
    ticks = max(0, int(now_tick) - int(since_tick))
    hours = 0.0
    if now_time is not None and since_time is not None:
        hours = max(0.0, (now_time - since_time).total_seconds() / 3600.0)
    return ticks + drift * hours


class CognitiveClock:
    """Monotonic observation counter owned by one ``World``.

    The facade advances it once per ``ingest()`` and persists ``tick`` in
    the world state.  Engines read ``tick`` to stamp records and to compute
    elapsed cognitive time; tests and simulations may call ``advance(n)``
    to let ``n`` observations pass without touching any concept.
    """

    def __init__(self, tick: int = 0) -> None:
        self._tick = max(0, int(tick))

    @property
    def tick(self) -> int:
        return self._tick

    def advance(self, ticks: int = 1) -> int:
        """Move the clock forward by ``ticks`` observations; returns the new tick."""
        if ticks < 0:
            raise ValueError("cognitive time cannot move backwards")
        self._tick += int(ticks)
        return self._tick

    def elapsed(
        self,
        since_tick: int,
        since_time: datetime | None = None,
        *,
        now_time: datetime | None = None,
    ) -> float:
        """Cognitive time elapsed since ``(since_tick, since_time)``."""
        return cognitive_elapsed(
            self._tick,
            since_tick,
            now_time or (wall_now() if since_time is not None else None),
            since_time,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CognitiveClock(tick={self._tick})"
