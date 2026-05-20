"""Cognitive dynamics — six independent Lego bricks.

Each engine satisfies a Protocol from ``world0.core``:

| Engine                     | Protocol            |
|----------------------------|---------------------|
| ``ActivationEngine``       | ActivationProvider  |
| ``HebbianEngine``          | HebbianLearner      |
| ``DecayEngine``            | DecayPolicy         |
| ``LifecycleEngine``        | LifecyclePolicy     |
| ``CommunityDetector``      | CommunityDetectorP  |
| ``ColorDiffusionEngine``   | ColorField          |

Engines depend only on ``ConceptStore`` / ``RelationStore`` Protocols
and on ``coefficients.py`` for shared cognitive constants — they do
**not** import each other.  Any subset can be swapped for an alternative
implementation as long as the Protocol contract is satisfied.
"""

from world0.dynamics.activation import ActivationEngine
from world0.dynamics.color_diffusion import ColorDiffusionEngine
from world0.dynamics.community import CommunityDetector
from world0.dynamics.decay import DecayEngine
from world0.dynamics.hebbian import HebbianEngine
from world0.dynamics.lifecycle import LifecycleEngine

__all__ = [
    "ActivationEngine",
    "ColorDiffusionEngine",
    "CommunityDetector",
    "DecayEngine",
    "HebbianEngine",
    "LifecycleEngine",
]
