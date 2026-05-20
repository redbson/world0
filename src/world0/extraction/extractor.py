"""ConceptExtractor — LLM-powered extraction of concepts and relations from text.

This is the bridge between raw conversation/text and World 0's structured
Observation format. The LLM does the semantic heavy-lifting; World 0 handles
the cognitive computation.
"""

from __future__ import annotations

import json
import re

from world0.llm.base import LLMProvider
from world0.prompts import PromptRegistry
from world0.schemas.relation import RelationType
from world0.schemas.types import Observation

# All valid relation type values for prompt and validation
_VALID_RELATION_TYPES = {rt.value for rt in RelationType}


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

    def __init__(
        self,
        provider: LLMProvider,
        *,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._prompts = prompt_registry or PromptRegistry()

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

        system_prompt = self._prompts.render("extraction.concepts_relations.system")
        raw = self._provider.complete_json(system_prompt, text)
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
