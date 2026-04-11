"""ConceptExtractor — LLM-powered extraction of concepts and relations from text.

This is the bridge between raw conversation/text and World 0's structured
Observation format. The LLM does the semantic heavy-lifting; World 0 handles
the cognitive computation.
"""

from __future__ import annotations

import json
import re

from world0.llm.base import LLMProvider
from world0.schemas.relation import RelationType
from world0.schemas.types import Observation

# All valid relation type values for prompt and validation
_VALID_RELATION_TYPES = {rt.value for rt in RelationType}

_SYSTEM_PROMPT = """\
You are a concept extraction engine for a cognitive system called World 0.

Your job is to extract **concepts** and **relations** from the given text.

## What is a concept?
A concept is a meaningful semantic unit — not a trivial word. Good concepts are:
- Domain terms (e.g., "machine learning", "REST API", "event sourcing")
- Processes or methods (e.g., "gradient descent", "blue-green deployment")
- Architectural components (e.g., "message queue", "load balancer")
- Roles or actors (e.g., "data engineer", "end user")
- Abstract principles (e.g., "separation of concerns", "eventual consistency")

Do NOT extract:
- Generic words ("system", "thing", "process" without context)
- Stopwords or filler
- Redundant near-duplicates (pick the most specific form)

## What is a relation?
A typed connection between two concepts. Available relation types:
- contains: A contains B as a component
- part_of: A is part of B
- depends_on: A depends on B
- supports: A supports or enables B
- contrasts: A is in contrast with B
- similar_to: A is similar to B
- activates: A triggers or activates B
- precedes: A comes before B in a sequence
- derived_from: A is derived from B
- related_to: generic fallback (use sparingly)

## Output format
Respond with ONLY a JSON object:
{
  "concepts": [
    {"name": "concept name", "description": "one-line description"}
  ],
  "relations": [
    {"source": "concept A", "target": "concept B", "type": "relation_type"}
  ]
}

Rules:
- Extract 3-15 concepts depending on text length and density.
- Extract meaningful relations — don't force connections that aren't there.
- Use the most specific relation type that applies.
- Concept names should be normalized: lowercase, concise, canonical form.
- Every concept in a relation must appear in the concepts list.
- Respond ONLY with the JSON object, no markdown fences, no explanation.\
"""


class ConceptExtractor:
    """Extracts concepts and relations from text using an LLM.

    Usage::

        from world0.llm import OpenAIProvider
        from world0.extraction import ConceptExtractor

        provider = OpenAIProvider(model="gpt-5-mini")
        extractor = ConceptExtractor(provider)

        observation = extractor.extract(
            "We deployed the ML model using Docker and Kubernetes. "
            "Latency dropped after we added Redis caching.",
            task="deployment review",
            source="session_5",
        )
        # observation is a ready-to-use Observation for World.ingest()
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(
        self,
        text: str,
        *,
        task: str = "",
        source: str = "",
    ) -> Observation:
        """Extract concepts and relations from text.

        Args:
            text: The raw text to extract from (conversation, document, etc.).
            task: Task context label for the resulting Observation.
            source: Source label for provenance tracking.

        Returns:
            An Observation ready to be passed to ``World.ingest()``.
        """
        if not text.strip():
            return Observation(task=task, source=source)

        raw = self._provider.complete_json(_SYSTEM_PROMPT, text)
        return self._parse_response(raw, task=task, source=source)

    def _parse_response(
        self, raw: str, *, task: str, source: str
    ) -> Observation:
        """Parse LLM JSON response into an Observation.

        Robust to common LLM output quirks: markdown fences, trailing text.
        """
        cleaned = self._extract_json(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # If parsing fails entirely, return empty observation
            return Observation(task=task, source=source)

        # Extract concept names and descriptions
        concept_names: list[str] = []
        descriptions: dict[str, str] = {}

        raw_concepts = data.get("concepts", [])
        if isinstance(raw_concepts, list):
            for item in raw_concepts:
                if isinstance(item, dict):
                    name = item.get("name", "").strip()
                    if name:
                        concept_names.append(name)
                        desc = item.get("description", "").strip()
                        if desc:
                            descriptions[name] = desc
                elif isinstance(item, str) and item.strip():
                    concept_names.append(item.strip())

        # Extract relations
        relations: list[tuple[str, str, str]] = []
        concept_set = {n.lower() for n in concept_names}

        raw_relations = data.get("relations", [])
        if isinstance(raw_relations, list):
            for item in raw_relations:
                if not isinstance(item, dict):
                    continue
                src = item.get("source", "").strip()
                tgt = item.get("target", "").strip()
                rel_type = item.get("type", "related_to").strip()

                if not src or not tgt:
                    continue
                # Validate both ends exist in concepts
                if src.lower() not in concept_set or tgt.lower() not in concept_set:
                    continue
                # Validate relation type
                if rel_type not in _VALID_RELATION_TYPES:
                    rel_type = "related_to"

                relations.append((src, tgt, rel_type))

        return Observation(
            concepts=concept_names,
            relations=relations,
            descriptions=descriptions,
            task=task,
            source=source,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try to find JSON in markdown code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Try to find bare JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text
