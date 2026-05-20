"""``concepts`` — concept lifecycle management.

Public surface lives in ``api.py``.  ``manager.py`` is a backwards
compatibility shim for the old ``world0.concepts.manager`` import path.
"""

from world0.concepts.api import (
    Concepts,
    ConceptManager,
    SIGNATURE_CONSOLIDATION_THRESHOLD,
)

__all__ = [
    "Concepts",
    "ConceptManager",
    "SIGNATURE_CONSOLIDATION_THRESHOLD",
]
