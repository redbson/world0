"""Backwards-compatible re-export.

The implementation moved to ``_manager.py`` (and siblings) as part of
the Lego-style modularisation.  This file is kept so existing imports
continue to work:

    from world0.concepts.manager import ConceptManager
    from world0.concepts.manager import SIGNATURE_CONSOLIDATION_THRESHOLD

New code should import from ``world0.concepts.api`` instead.
"""

from world0.concepts._consolidation import SIGNATURE_CONSOLIDATION_THRESHOLD
from world0.concepts._manager import ConceptManager

__all__ = ["ConceptManager", "SIGNATURE_CONSOLIDATION_THRESHOLD"]
