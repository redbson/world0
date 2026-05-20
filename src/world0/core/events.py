"""Cross-module events.

Events are how Lego bricks talk to each other without importing each
other's concrete classes.  A subsystem that wants to react to
"a concept was reinforced" can subscribe to ``ConceptReinforced`` on the
``EventBus`` instead of being directly called by whoever did the
reinforcing.

This module deliberately ships only:

- a small set of lightweight, frozen event dataclasses;
- an ``EventBus`` Protocol so consumers can be tested with a mock bus;
- a no-op default (``NullEventBus``) and an in-process default
  (``InMemoryEventBus``) so callers always have something to publish to.

No engine in this codebase is *required* to use the bus today — the
contract is here so the wiring can be added incrementally per module
without breaking the existing direct-call pipelines.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar, runtime_checkable


# ── Event dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Event:
    """Marker base class for all events on the bus."""


@dataclass(frozen=True, slots=True)
class ConceptCreated(Event):
    concept_id: str
    name: str
    origin: str = ""
    task: str = ""


@dataclass(frozen=True, slots=True)
class ConceptReinforced(Event):
    concept_id: str
    source: str = ""
    task: str = ""


@dataclass(frozen=True, slots=True)
class ConceptWeakened(Event):
    concept_id: str
    source: str = ""
    task: str = ""


@dataclass(frozen=True, slots=True)
class RelationDiscovered(Event):
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    is_explicit: bool


@dataclass(frozen=True, slots=True)
class RelationReinforced(Event):
    relation_id: str
    provenance: str = ""


@dataclass(frozen=True, slots=True)
class RelationWeakened(Event):
    relation_id: str
    provenance: str = ""


# ── Bus ──────────────────────────────────────────────────────────────


E = TypeVar("E", bound=Event)
EventHandler = Callable[[E], None]


@runtime_checkable
class EventBus(Protocol):
    """Pub/sub Protocol for cross-module events."""

    def publish(self, event: Event) -> None: ...

    def subscribe(
        self, event_type: type[Event], handler: EventHandler
    ) -> None: ...


class NullEventBus:
    """Default bus that drops every event.

    Useful as a no-op fallback when no subscribers are configured —
    publishers can call ``publish`` unconditionally without paying for
    list traversal.
    """

    def publish(self, event: Event) -> None:  # noqa: D401 - Protocol
        return None

    def subscribe(
        self, event_type: type[Event], handler: EventHandler
    ) -> None:
        return None


class InMemoryEventBus:
    """Synchronous in-process bus.

    Handlers are invoked in subscription order on the publishing thread.
    A failing handler does not abort the publish loop — exceptions are
    swallowed per-handler so one bad subscriber cannot break the rest.
    Pluggable error handling is intentionally out of scope until a
    real consumer asks for it.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[EventHandler]] = defaultdict(
            list
        )

    def publish(self, event: Event) -> None:
        for event_type, handlers in self._handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception:  # noqa: BLE001 - isolation by design
                        continue

    def subscribe(
        self, event_type: type[Event], handler: EventHandler
    ) -> None:
        self._handlers[event_type].append(handler)
