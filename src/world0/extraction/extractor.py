"""ConceptExtractor — LLM-powered extraction of concepts and relations from text.

This is the bridge between raw conversation/text and World 0's structured
Observation format. The LLM does the semantic heavy-lifting; World 0 handles
the cognitive computation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from world0.llm.base import LLMProvider
from world0.prompts import PromptRegistry
from world0.schemas.relation import normalize_semantic_relation, semantic_relation_names
from world0.schemas.types import ConceptCandidate, Observation, RelationPrior

# All valid relation language labels for prompt and validation.
_VALID_RELATION_TYPES = set(semantic_relation_names())


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
        preset_relations: list[RelationPrior | dict[str, Any]] | None = None,
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
            return Observation(
                task=task,
                source=source,
                relation_priors=self._coerce_relation_priors(preset_relations),
            )

        system_prompt = self._prompts.render("extraction.concepts_relations.system")
        relation_priors = self._coerce_relation_priors(preset_relations)
        user_prompt = self._build_user_prompt(
            text,
            task=task,
            source=source,
            preset_relations=relation_priors,
        )
        raw = self._provider.complete_json(system_prompt, user_prompt)
        return self._parse_response(
            raw,
            task=task,
            source=source,
            relation_priors=relation_priors,
        )

    def _parse_response(
        self,
        raw: str,
        *,
        task: str,
        source: str,
        relation_priors: list[RelationPrior] | None = None,
    ) -> Observation:
        """Parse LLM JSON response into an Observation.

        Robust to common LLM output quirks: markdown fences, trailing text.
        """
        cleaned = self._extract_json(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # If parsing fails entirely, return empty observation
            return Observation(
                task=task,
                source=source,
                relation_priors=list(relation_priors or []),
                extraction_metadata={
                    "parse_warnings": ["response was not valid JSON"],
                    "raw_response": raw,
                },
            )

        concept_names: list[str] = []
        concept_candidates: list[ConceptCandidate] = []
        descriptions: dict[str, str] = {}
        aliases_by_name: dict[str, list[str]] = {}
        candidates_by_uid: dict[str, ConceptCandidate] = {}
        concept_metadata: dict[str, dict[str, Any]] = {}
        parse_warnings: list[str] = []

        raw_concepts = data.get("concepts", [])
        if isinstance(raw_concepts, list):
            for item in raw_concepts:
                if isinstance(item, dict):
                    name = item.get("name", "").strip()
                    if name:
                        concept_names.append(name)
                        uid = str(item.get("uid", "")).strip()
                        desc = item.get("description", "").strip()
                        kind = str(item.get("kind", "")).strip()
                        sense = str(item.get("sense", "")).strip()
                        domain = str(item.get("domain", "")).strip()
                        evidence = str(item.get("evidence", "")).strip()
                        salience = self._number_or_none(item.get("salience"))
                        confidence = self._number_or_none(item.get("confidence"))
                        aliases = self._string_list(item.get("aliases"))
                        candidate = ConceptCandidate(
                            uid=uid,
                            name=name,
                            kind=kind,
                            sense=sense,
                            domain=domain,
                            description=desc,
                            aliases=aliases,
                            salience=salience,
                            confidence=confidence,
                            evidence=evidence,
                        )
                        concept_candidates.append(candidate)
                        if uid:
                            candidates_by_uid[uid] = candidate
                        if desc:
                            descriptions[name] = desc
                        aliases_by_name[name] = aliases
                        concept_metadata[name] = {
                            "uid": uid,
                            "kind": kind,
                            "sense": sense,
                            "domain": domain,
                            "salience": salience,
                            "confidence": confidence,
                            "evidence": evidence,
                            "aliases": aliases,
                        }
                elif isinstance(item, str) and item.strip():
                    concept_names.append(item.strip())
                    concept_candidates.append(
                        ConceptCandidate(name=item.strip())
                    )

        canonical = self._canonical_name_map(
            concept_candidates,
            concept_names,
            aliases_by_name,
        )

        relations: list[tuple[str, str, str]] = []
        relation_metadata: list[dict[str, Any]] = []
        dropped_relations: list[dict[str, Any]] = []
        raw_relations = data.get("relations", [])
        if isinstance(raw_relations, list):
            for item in raw_relations:
                parsed = self._parse_relation_item(item)
                if parsed is None:
                    continue
                src, tgt, rel_type, meta = parsed

                if not src or not tgt:
                    continue
                resolved_src = self._resolve_endpoint(src, canonical, candidates_by_uid)
                resolved_tgt = self._resolve_endpoint(tgt, canonical, candidates_by_uid)
                if not resolved_src or not resolved_tgt:
                    dropped_relations.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "reason": "endpoint did not match any concept or alias",
                    })
                    continue
                rel_type = normalize_semantic_relation(rel_type)

                relations.append((resolved_src, resolved_tgt, rel_type))
                relation_metadata.append({
                    "source": resolved_src,
                    "target": resolved_tgt,
                    "type": rel_type,
                    **meta,
                })

        weakened = self._canonicalize_names(data.get("weakened"), canonical)
        contradicted_relations = self._parse_relation_list(
            data.get("contradicted_relations"),
            canonical,
            parse_warnings,
        )

        return Observation(
            concepts=concept_names,
            concept_candidates=concept_candidates,
            relations=relations,
            descriptions=descriptions,
            weakened=weakened,
            contradicted_relations=contradicted_relations,
            domain=str(data.get("domain", "")).strip(),
            task=task,
            source=source,
            relation_priors=list(relation_priors or []),
            extraction_metadata={
                "concepts": concept_metadata,
                "relations": relation_metadata,
                "dropped_relations": dropped_relations,
                "parse_warnings": parse_warnings,
                "raw_counts": {
                    "concepts": len(raw_concepts) if isinstance(raw_concepts, list) else 0,
                    "relations": len(raw_relations) if isinstance(raw_relations, list) else 0,
                    "accepted_relations": len(relations),
                    "dropped_relations": len(dropped_relations),
                },
            },
        )

    @staticmethod
    def _build_user_prompt(
        text: str,
        *,
        task: str,
        source: str,
        preset_relations: list[RelationPrior] | None = None,
    ) -> str:
        sections = [
            "## Task Context",
            task or "none",
            "",
            "## Source",
            source or "none",
            "",
            "## Extraction Goal",
            (
                "Extract concepts and relations that would help World 0 build "
                "a reusable cognitive projection for this task."
            ),
            "",
            "## Text",
            text,
        ]
        if preset_relations:
            sections.extend([
                "",
                "## Preset Relations",
                (
                    "Use these relation priors as candidate beliefs. "
                    "Re-evaluate them against the text and output only the "
                    "relation label when you accept or adjust one."
                ),
                json.dumps(
                    [
                        {
                            "source": prior.source,
                            "target": prior.target,
                            "type": normalize_semantic_relation(prior.relation_type),
                            "rationale": prior.rationale,
                        }
                        for prior in preset_relations
                    ],
                    ensure_ascii=False,
                ),
            ])
        return "\n".join(sections)

    @staticmethod
    def _coerce_relation_priors(
        value: list[RelationPrior | dict[str, Any]] | None,
    ) -> list[RelationPrior]:
        if not value:
            return []
        priors: list[RelationPrior] = []
        for item in value:
            if isinstance(item, RelationPrior):
                priors.append(item)
                continue
            if isinstance(item, dict):
                try:
                    data = dict(item)
                    if "type" in data and "relation_type" not in data:
                        data["relation_type"] = data["type"]
                    priors.append(RelationPrior(**data))
                except Exception:
                    continue
        return priors

    @classmethod
    def _canonical_name_map(
        cls,
        concept_candidates: list[ConceptCandidate],
        concept_names: list[str],
        aliases_by_name: dict[str, list[str]],
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for candidate in concept_candidates:
            if candidate.uid:
                mapping[candidate.uid] = candidate.uid
            name = candidate.name
            key = cls._normalize_key(name)
            if key and key not in mapping:
                mapping[key] = candidate.uid or name
            for alias in candidate.aliases:
                alias_key = cls._normalize_key(alias)
                if alias_key and alias_key not in mapping:
                    mapping[alias_key] = candidate.uid or name
        for name in concept_names:
            key = cls._normalize_key(name)
            if key and key not in mapping:
                mapping[key] = name
            for alias in aliases_by_name.get(name, []):
                alias_key = cls._normalize_key(alias)
                if alias_key and alias_key not in mapping:
                    mapping[alias_key] = name
        return mapping

    @classmethod
    def _resolve_endpoint(
        cls,
        value: str,
        canonical: dict[str, str],
        candidates_by_uid: dict[str, ConceptCandidate],
    ) -> str | None:
        direct = value.strip()
        if direct in candidates_by_uid:
            return direct
        return canonical.get(cls._normalize_key(direct))

    @staticmethod
    def _normalize_key(value: str) -> str:
        lowered = value.strip().lower()
        compact = re.sub(r"[_\W]+", " ", lowered, flags=re.UNICODE)
        return re.sub(r"\s+", " ", compact).strip()

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _number_or_none(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _canonicalize_names(
        self,
        value: Any,
        canonical: dict[str, str],
    ) -> list[str]:
        names = self._string_list(value)
        result: list[str] = []
        for name in names:
            resolved = canonical.get(self._normalize_key(name), name)
            if resolved not in result:
                result.append(resolved)
        return result

    def _parse_relation_list(
        self,
        value: Any,
        canonical: dict[str, str],
        parse_warnings: list[str],
    ) -> list[tuple[str, str, str]]:
        result: list[tuple[str, str, str]] = []
        if not isinstance(value, list):
            return result
        for item in value:
            parsed = self._parse_relation_item(item)
            if parsed is None:
                continue
            src, tgt, rel_type, _meta = parsed
            resolved_src = canonical.get(self._normalize_key(src))
            resolved_tgt = canonical.get(self._normalize_key(tgt))
            if not resolved_src or not resolved_tgt:
                parse_warnings.append(
                    f"contradicted relation endpoint not found: {src} -> {tgt}"
                )
                continue
            rel_type = normalize_semantic_relation(rel_type)
            result.append((resolved_src, resolved_tgt, rel_type))
        return result

    def _parse_relation_item(
        self,
        item: Any,
    ) -> tuple[str, str, str, dict[str, Any]] | None:
        if isinstance(item, dict):
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            rel_type = normalize_semantic_relation(item.get("type", "generic_relation"))
            return src, tgt, rel_type, {
                "evidence": str(item.get("evidence", "")).strip(),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            return (
                str(item[0]).strip(),
                str(item[1]).strip(),
                normalize_semantic_relation(str(item[2]).strip()),
                {},
            )
        return None

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
