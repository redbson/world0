"""World 0 cross-module contracts.

Everything in ``core`` is a Protocol or a value type that subsystems
(``concepts``, ``relations``, ``dynamics/*``, ``projection``,
``visualization`` …) depend on.  Subsystems must **not** import each
other's concrete classes; they may only depend on the Protocols here.

This is the rule that makes the project Lego-style: any subsystem can
be swapped for an alternative implementation as long as it satisfies
the relevant Protocol.
"""

from world0.core.events import (
    ConceptCreated,
    ConceptReinforced,
    ConceptWeakened,
    Event,
    EventBus,
    EventHandler,
    InMemoryEventBus,
    NullEventBus,
    RelationDiscovered,
    RelationReinforced,
    RelationWeakened,
)
from world0.core.interfaces import (
    ActivationProvider,
    ColorField,
    CommunityDetectorP,
    ConceptStore,
    ConceptStoreReader,
    DecayPolicy,
    Extractor,
    HebbianLearner,
    LifecyclePolicy,
    LLMProvider,
    Projector,
    RelationStore,
    RelationStoreReader,
    StorageBackend,
    WorldView,
)

__all__ = [
    # interfaces
    "ActivationProvider",
    "ColorField",
    "CommunityDetectorP",
    "ConceptStore",
    "ConceptStoreReader",
    "DecayPolicy",
    "Extractor",
    "HebbianLearner",
    "LLMProvider",
    "LifecyclePolicy",
    "Projector",
    "RelationStore",
    "RelationStoreReader",
    "StorageBackend",
    "WorldView",
    # events
    "ConceptCreated",
    "ConceptReinforced",
    "ConceptWeakened",
    "Event",
    "EventBus",
    "EventHandler",
    "InMemoryEventBus",
    "NullEventBus",
    "RelationDiscovered",
    "RelationReinforced",
    "RelationWeakened",
]
