"""Context — task conditions that change which concepts matter now.

A task has two sources of meaning for World 0:

- **history**: the tasks a concept has actually served under
  (``ConceptNode.task_profile`` / ``task_affinity``), learned from
  observations that carry a ``task`` label;
- **grounding**: the concepts the task *names*.  A task string such as
  ``"kubernetes rollout"`` refers to the concepts ``kubernetes`` and
  ``rollout`` whether or not any observation was ever labelled with it.

History alone left a new task — or any task in a world built from
unlabelled observations — without effect (docs §7.17).  Grounding gives
such a task a structural foothold: the concepts it names, and their
direct neighbours, become task-relevant.

The third source is the **current focus** (``Focus``): what the previous
views held in a limited-capacity workspace biases the next one
(``docs/mc/03-workspace.md``).
"""

from world0.context.focus import (
    FOCUS_CAPACITY,
    IGNITION_THRESHOLD,
    Focus,
)
from world0.context.grounding import (
    GROUNDING_MIN_COVERAGE,
    GROUNDING_NEIGHBOR_SHARE,
    ground_task,
    name_coverage,
)

__all__ = [
    "FOCUS_CAPACITY",
    "IGNITION_THRESHOLD",
    "Focus",
    "GROUNDING_MIN_COVERAGE",
    "GROUNDING_NEIGHBOR_SHARE",
    "ground_task",
    "name_coverage",
]
