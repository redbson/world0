"""Public API of the ``concepts`` Lego brick.

Other subsystems should import from here:

    from world0.concepts.api import Concepts
    from world0.concepts.api import SIGNATURE_CONSOLIDATION_THRESHOLD

The class is exported under both names — ``Concepts`` is the new
short, intent-revealing alias; ``ConceptManager`` is kept so existing
callers keep working without churn.

Internal modules (``_manager``, ``_indexes``, ``_consolidation``,
``_identity_ops``) are implementation details and may be reorganised
without notice.
"""

from world0.concepts._consolidation import (
    DOMAIN_MISMATCH_PENALTY,
    SIGNATURE_CONSOLIDATION_THRESHOLD,
)
from world0.concepts._manager import ConceptManager

# Public, intent-revealing alias.  The class is the same object —
# ``Concepts`` is the name to prefer in new code.
Concepts = ConceptManager

__all__ = [
    "Concepts",
    "ConceptManager",
    "SIGNATURE_CONSOLIDATION_THRESHOLD",
    "DOMAIN_MISMATCH_PENALTY",
]
