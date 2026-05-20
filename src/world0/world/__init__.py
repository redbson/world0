"""World 0 facade package.

The public surface is the ``World`` class — a thin orchestrator that
wires Lego bricks together (stores, dynamics engines, pipelines) and
delegates the actual work to them.

Internal modules (``facade``, ``_ingest``, ``_reflect``, ``_identity``,
``_status``) are implementation details.  External callers should keep
importing ``from world0 import World`` or ``from world0.world import World``.

Pipeline classes are exposed for advanced/testing scenarios where you
want to drive a single stage against your own ConceptStore /
RelationStore mocks.
"""

from world0.world._identity import IdentityOps
from world0.world._ingest import IngestPipeline
from world0.world._reflect import ReflectPipeline
from world0.world._status import build_status
from world0.world.facade import World

__all__ = [
    "World",
    "IngestPipeline",
    "ReflectPipeline",
    "IdentityOps",
    "build_status",
]
