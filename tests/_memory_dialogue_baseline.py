"""Test-only baseline: a dialogue system with ordinary transcript memory.

This module intentionally does not use World 0 concepts, relations,
activation, or projection. It stores plain dialogue turns and recalls
them through lexical overlap plus a small recency bias.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from world0 import Observation, World

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "back",
    "by",
    "do",
    "does",
    "for",
    "from",
    "here",
    "how",
    "in",
    "into",
    "is",
    "it",
    "later",
    "need",
    "not",
    "of",
    "on",
    "or",
    "our",
    "same",
    "should",
    "still",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "with",
}


@dataclass(frozen=True)
class DialogueTurn:
    user: str
    assistant: str
    observation: Observation


@dataclass(frozen=True)
class MemorySnippet:
    index: int
    user: str
    assistant: str
    text: str
    score: float


class MemoryOnlyDialogueSystem:
    """Simple transcript memory baseline.

    Behavior:
    - stores turns in order
    - retrieves top-k turns via token overlap
    - applies a small recency bias

    This is a useful control system when comparing against World 0's
    concept/relation/context-sensitive projection behavior.
    """

    def __init__(self) -> None:
        self._turns: list[MemorySnippet] = []

    def ingest_turn(self, user: str, assistant: str) -> None:
        index = len(self._turns)
        text = f"{user}\n{assistant}".lower()
        self._turns.append(
            MemorySnippet(
                index=index,
                user=user,
                assistant=assistant,
                text=text,
                score=0.0,
            )
        )

    def recall(self, query: str, *, max_items: int = 4) -> list[MemorySnippet]:
        query_tokens = _tokens(query)
        scored: list[MemorySnippet] = []

        for turn in self._turns:
            turn_tokens = _tokens(turn.text)
            overlap = len(query_tokens & turn_tokens)
            phrase_bonus = (
                2.0
                if "model serving" in query.lower()
                and "model serving" in turn.text
                else 0.0
            )
            recency_bias = turn.index * 0.02
            score = overlap + phrase_bonus + recency_bias
            if score <= 0:
                continue
            scored.append(
                MemorySnippet(
                    index=turn.index,
                    user=turn.user,
                    assistant=turn.assistant,
                    text=turn.text,
                    score=score,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:max_items]


def build_long_dialogue_systems(tmp_path) -> tuple[World, MemoryOnlyDialogueSystem]:
    """Build a long mixed-domain dialogue in both systems."""
    world = World(store_path=tmp_path / ".world0")
    memory = MemoryOnlyDialogueSystem()

    for turn in LONG_DIALOGUE_TURNS:
        world.ingest(turn.observation)
        memory.ingest_turn(turn.user, turn.assistant)

    return world, memory


def mentioned_concepts(
    snippets: list[MemorySnippet], vocabulary: set[str]
) -> set[str]:
    text = "\n".join(snippet.text for snippet in snippets)
    return {
        concept
        for concept in vocabulary
        if concept.lower() in text
    }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_/-]+", text.lower())
        if token not in STOPWORDS
    }


LONG_DIALOGUE_TURNS = [
    DialogueTurn(
        user="Map the training backbone for the new ranking model.",
        assistant="PyTorch supports the training pipeline.",
        observation=Observation(
            concepts=["PyTorch", "training pipeline"],
            relations=[("PyTorch", "training pipeline", "supports")],
            task="ml training",
            source="dialogue_turn_01",
        ),
    ),
    DialogueTurn(
        user="What sits inside the training path?",
        assistant="The training pipeline contains the neural network.",
        observation=Observation(
            concepts=["training pipeline", "neural network"],
            relations=[("training pipeline", "neural network", "contains")],
            task="ml training",
            source="dialogue_turn_02",
        ),
    ),
    DialogueTurn(
        user="What drives optimization?",
        assistant="The neural network depends on gradient descent.",
        observation=Observation(
            concepts=["neural network", "gradient descent"],
            relations=[("neural network", "gradient descent", "depends_on")],
            task="ml training",
            source="dialogue_turn_03",
        ),
    ),
    DialogueTurn(
        user="How do we tune it?",
        assistant="The optimizer supports gradient descent.",
        observation=Observation(
            concepts=["optimizer", "gradient descent"],
            relations=[("optimizer", "gradient descent", "supports")],
            task="ml training",
            source="dialogue_turn_04",
        ),
    ),
    DialogueTurn(
        user="How does serving connect back upstream?",
        assistant="Model serving depends on PyTorch.",
        observation=Observation(
            concepts=["model serving", "PyTorch"],
            relations=[("model serving", "PyTorch", "depends_on")],
            task="ml training",
            source="dialogue_turn_05",
        ),
    ),
    DialogueTurn(
        user="Map the runtime edge.",
        assistant="Model serving depends on FastAPI.",
        observation=Observation(
            concepts=["model serving", "FastAPI"],
            relations=[("model serving", "FastAPI", "depends_on")],
            task="ops reliability",
            source="dialogue_turn_06",
        ),
    ),
    DialogueTurn(
        user="What carries the release path?",
        assistant="Model serving depends on deployment.",
        observation=Observation(
            concepts=["model serving", "deployment"],
            relations=[("model serving", "deployment", "depends_on")],
            task="ops reliability",
            source="dialogue_turn_07",
        ),
    ),
    DialogueTurn(
        user="What watches the live system?",
        assistant="Deployment contains monitoring.",
        observation=Observation(
            concepts=["deployment", "monitoring"],
            relations=[("deployment", "monitoring", "contains")],
            task="ops reliability",
            source="dialogue_turn_08",
        ),
    ),
    DialogueTurn(
        user="What opens the incident path?",
        assistant="Monitoring activates latency investigation.",
        observation=Observation(
            concepts=["monitoring", "latency"],
            relations=[("monitoring", "latency", "activates")],
            task="ops reliability",
            source="dialogue_turn_09",
        ),
    ),
    DialogueTurn(
        user="What helps control latency?",
        assistant="Autoscaling supports latency control.",
        observation=Observation(
            concepts=["autoscaling", "latency"],
            relations=[("autoscaling", "latency", "supports")],
            task="ops reliability",
            source="dialogue_turn_10",
        ),
    ),
    DialogueTurn(
        user="Should this become a memory system?",
        assistant="No. Keep projection separate from a memory system.",
        observation=Observation(
            concepts=["projection"],
            task="world0 design",
            source="dialogue_turn_11",
        ),
    ),
    DialogueTurn(
        user="Should we add workflow scheduling?",
        assistant="Workflow scheduling is a non-goal here.",
        observation=Observation(
            concepts=["workflow scheduling"],
            task="non-goal",
            source="dialogue_turn_12",
        ),
    ),
    DialogueTurn(
        user="What about vector search?",
        assistant="Vector search is not the same as the concept world.",
        observation=Observation(
            concepts=["vector search"],
            task="non-goal",
            source="dialogue_turn_13",
        ),
    ),
    DialogueTurn(
        user="Do we need note archives?",
        assistant="A note archive still does not replace relation typing.",
        observation=Observation(
            concepts=["relation typing"],
            task="world0 design",
            source="dialogue_turn_14",
        ),
    ),
    DialogueTurn(
        user="How should relevance shift when the task changes?",
        assistant="Context weighting should activate a different local view.",
        observation=Observation(
            concepts=["context weighting", "activation"],
            relations=[("context weighting", "activation", "activates")],
            task="world0 design",
            source="dialogue_turn_15",
        ),
    ),
    DialogueTurn(
        user="What is the final cognitive output?",
        assistant="Activation should precede projection.",
        observation=Observation(
            concepts=["activation", "projection"],
            relations=[("activation", "projection", "precedes")],
            task="world0 design",
            source="dialogue_turn_16",
        ),
    ),
]
