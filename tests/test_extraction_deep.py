"""Deep, deterministic tests for ConceptExtractor parsing + ingest robustness.

These tests exercise the LLM-extraction parsing surface of World 0 without
ever touching a real model.  A local ``FakeLLM(LLMProvider)`` returns canned
JSON (mirroring the pattern in ``tests/test_extraction.py``), so every
assertion is fully deterministic.

Coverage angles:
- ``_extract_json`` robustness across fence/prose/junk/nesting/non-JSON shapes.
- Concept shapes: dicts, plain strings, mixed, blank/whitespace names,
  duplicates, and full metadata capture into ConceptCandidate + metadata.
- Relations: dict and 3-tuple forms, unknown-type fallback, empty endpoints,
  endpoint resolution by uid / canonical name / alias / normalized form,
  unresolved → dropped_relations, and raw_counts bookkeeping.
- weakened canonicalization + de-dup; contradicted_relations resolution and
  parse_warnings on unresolved endpoints.
- relation_priors via preset_relations (RelationPrior + dict coercion incl.
  ``type`` -> ``relation_type``), prompt inclusion, and probability not leaking
  into the prompt.
- ``_number_or_none`` / ``_string_list`` edge cases.
- End-to-end through ``World(store_path=tmp, llm=FakeLLM)``: synonym/alias dedup,
  sense-based splitting, and the no-noise-filter boundary.
"""

import json

import pytest

from world0.extraction.extractor import ConceptExtractor
from world0.llm.base import LLMProvider
from world0.schemas.types import ConceptCandidate, Observation, RelationPrior
from world0.world.facade import World


class FakeLLM(LLMProvider):
    """Returns a pre-configured response for testing (no network)."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.system = ""
        self.user = ""
        self.calls = 0

    def complete_json(self, system: str, user: str) -> str:
        self.system = system
        self.user = user
        self.calls += 1
        return self._response


def _extract(response: str, **kwargs) -> Observation:
    extractor = ConceptExtractor(FakeLLM(response))
    return extractor.extract("some text", **kwargs)


# ── _extract_json robustness ─────────────────────────────────────────


class TestExtractJsonRobustness:
    def test_bare_object(self):
        obj = '{"concepts": [{"name": "alpha"}], "relations": []}'
        obs = _extract(obj)
        assert obs.concepts == ["alpha"]
        assert obs.extraction_metadata["parse_warnings"] == []

    def test_fenced_with_json_language(self):
        resp = '```json\n{"concepts": [{"name": "beta"}], "relations": []}\n```'
        obs = _extract(resp)
        assert obs.concepts == ["beta"]

    def test_fenced_without_language(self):
        resp = '```\n{"concepts": [{"name": "gamma"}], "relations": []}\n```'
        obs = _extract(resp)
        assert obs.concepts == ["gamma"]

    def test_prose_surrounded_object(self):
        resp = (
            "Sure! Here is the structured extraction you asked for:\n"
            '{"concepts": [{"name": "delta"}], "relations": []}\n'
            "Let me know if you need anything else."
        )
        obs = _extract(resp)
        assert obs.concepts == ["delta"]

    def test_leading_and_trailing_junk(self):
        resp = '$$$ noise <<< {"concepts": [{"name": "epsilon"}]} >>> trailing'
        obs = _extract(resp)
        assert obs.concepts == ["epsilon"]

    def test_nested_braces_are_captured(self):
        # The bare-object regex is greedy ({.*}), so a nested object inside
        # the top-level JSON must be preserved intact.
        payload = {
            "concepts": [
                {"name": "zeta", "salience": 0.5},
                {"name": "eta"},
            ],
            "relations": [{"source": "zeta", "target": "eta", "type": "supports"}],
        }
        resp = "prefix " + json.dumps(payload) + " suffix"
        obs = _extract(resp)
        assert obs.concepts == ["zeta", "eta"]
        assert obs.relations == [("zeta", "eta", "enables")]

    def test_non_json_returns_empty_observation_with_warning(self):
        obs = _extract("there is absolutely no json here, just words")
        assert obs.concepts == []
        assert obs.relations == []
        assert obs.extraction_metadata["parse_warnings"] == [
            "response was not valid JSON"
        ]
        assert "raw_response" in obs.extraction_metadata

    def test_malformed_json_object_returns_warning(self):
        # Looks like an object (has braces) but is not valid JSON.
        obs = _extract('{"concepts": [oops not valid] }')
        assert obs.concepts == []
        assert obs.extraction_metadata["parse_warnings"] == [
            "response was not valid JSON"
        ]

    def test_fence_takes_precedence_over_bare_scan(self):
        # A fenced block should be preferred even if other braces precede it.
        resp = (
            "ignore {this: bad} blob\n"
            '```json\n{"concepts": [{"name": "theta"}]}\n```'
        )
        obs = _extract(resp)
        assert obs.concepts == ["theta"]


# ── Concept shapes ───────────────────────────────────────────────────


class TestConceptShapes:
    def test_list_of_dicts(self):
        resp = json.dumps({"concepts": [{"name": "a"}, {"name": "b"}]})
        obs = _extract(resp)
        assert obs.concepts == ["a", "b"]
        assert [c.name for c in obs.concept_candidates] == ["a", "b"]

    def test_list_of_plain_strings(self):
        resp = json.dumps({"concepts": ["x", "y", "z"]})
        obs = _extract(resp)
        assert obs.concepts == ["x", "y", "z"]
        assert all(isinstance(c, ConceptCandidate) for c in obs.concept_candidates)

    def test_mixed_dicts_and_strings(self):
        resp = json.dumps({"concepts": [{"name": "dict_one"}, "string_two"]})
        obs = _extract(resp)
        assert obs.concepts == ["dict_one", "string_two"]
        assert obs.concept_candidates[0].name == "dict_one"
        assert obs.concept_candidates[1].name == "string_two"

    def test_missing_name_dict_is_skipped(self):
        resp = json.dumps({"concepts": [{"description": "no name here"}, {"name": "kept"}]})
        obs = _extract(resp)
        assert obs.concepts == ["kept"]

    def test_whitespace_name_dict_is_skipped(self):
        resp = json.dumps({"concepts": [{"name": "   "}, {"name": "real"}]})
        obs = _extract(resp)
        assert obs.concepts == ["real"]

    def test_whitespace_only_string_is_skipped(self):
        resp = json.dumps({"concepts": ["   ", "valid"]})
        obs = _extract(resp)
        assert obs.concepts == ["valid"]

    def test_name_is_stripped(self):
        resp = json.dumps({"concepts": [{"name": "  padded  "}]})
        obs = _extract(resp)
        assert obs.concepts == ["padded"]
        assert obs.concept_candidates[0].name == "padded"

    def test_duplicates_preserved(self):
        resp = json.dumps({"concepts": [{"name": "dup"}, {"name": "dup"}, "dup"]})
        obs = _extract(resp)
        assert obs.concepts == ["dup", "dup", "dup"]
        assert len(obs.concept_candidates) == 3

    def test_full_metadata_captured_into_candidate_and_metadata(self):
        resp = json.dumps({
            "concepts": [
                {
                    "uid": "c1",
                    "name": "transformer",
                    "kind": "architecture",
                    "sense": "neural network model",
                    "domain": "deep learning",
                    "description": "Attention-based sequence model",
                    "aliases": ["attention model", "xfmr"],
                    "salience": 0.93,
                    "confidence": 0.88,
                    "evidence": "Transformers use self-attention.",
                }
            ]
        })
        obs = _extract(resp)
        cand = obs.concept_candidates[0]
        assert cand.uid == "c1"
        assert cand.name == "transformer"
        assert cand.kind == "architecture"
        assert cand.sense == "neural network model"
        assert cand.domain == "deep learning"
        assert cand.description == "Attention-based sequence model"
        assert cand.aliases == ["attention model", "xfmr"]
        assert cand.salience == 0.93
        assert cand.confidence == 0.88
        assert cand.evidence == "Transformers use self-attention."

        meta = obs.extraction_metadata["concepts"]["transformer"]
        assert meta["uid"] == "c1"
        assert meta["kind"] == "architecture"
        assert meta["sense"] == "neural network model"
        assert meta["domain"] == "deep learning"
        assert meta["salience"] == 0.93
        assert meta["confidence"] == 0.88
        assert meta["evidence"] == "Transformers use self-attention."
        assert meta["aliases"] == ["attention model", "xfmr"]

        assert obs.descriptions["transformer"] == "Attention-based sequence model"

    def test_description_omitted_when_blank(self):
        resp = json.dumps({"concepts": [{"name": "bare", "description": "   "}]})
        obs = _extract(resp)
        # blank description is stripped to empty -> not put in descriptions map
        assert "bare" not in obs.descriptions
        assert obs.concept_candidates[0].description == ""

    def test_plain_string_candidate_has_empty_metadata(self):
        resp = json.dumps({"concepts": ["lonely"]})
        obs = _extract(resp)
        cand = obs.concept_candidates[0]
        assert cand.uid == ""
        assert cand.aliases == []
        assert cand.salience is None
        assert cand.confidence is None
        # Plain strings don't populate per-name concept metadata.
        assert "lonely" not in obs.extraction_metadata["concepts"]


# ── Relations ────────────────────────────────────────────────────────


class TestRelations:
    def test_dict_form(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "a", "target": "b", "type": "depends_on"}],
        })
        obs = _extract(resp)
        assert obs.relations == [("a", "b", "dependence")]

    def test_tuple_form(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [["a", "b", "supports"]],
        })
        obs = _extract(resp)
        assert obs.relations == [("a", "b", "enables")]

    def test_tuple_form_unknown_type_falls_back_to_generic(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [["a", "b", "totally_made_up"]],
        })
        obs = _extract(resp)
        assert obs.relations == [("a", "b", "generic_relation")]

    def test_dict_unknown_type_falls_back_to_generic(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "a", "target": "b", "type": "frobnicates"}],
        })
        obs = _extract(resp)
        assert obs.relations == [("a", "b", "generic_relation")]

    def test_dict_missing_type_defaults_generic(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "a", "target": "b"}],
        })
        obs = _extract(resp)
        assert obs.relations == [("a", "b", "generic_relation")]

    def test_tuple_too_short_is_ignored(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [["a", "b"]],
        })
        obs = _extract(resp)
        assert obs.relations == []
        # Two-item lists aren't relations; not counted as dropped either.
        assert obs.extraction_metadata["dropped_relations"] == []

    def test_empty_source_dropped(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "", "target": "b", "type": "supports"}],
        })
        obs = _extract(resp)
        assert obs.relations == []
        # Empty endpoints are filtered before resolution, so not recorded
        # as resolution-failure drops.
        assert obs.extraction_metadata["dropped_relations"] == []

    def test_empty_target_dropped(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [{"source": "a", "target": "", "type": "supports"}],
        })
        obs = _extract(resp)
        assert obs.relations == []
        assert obs.extraction_metadata["dropped_relations"] == []

    def test_resolution_by_uid(self):
        resp = json.dumps({
            "concepts": [
                {"uid": "c1", "name": "Alpha"},
                {"uid": "c2", "name": "Beta"},
            ],
            "relations": [{"source": "c1", "target": "c2", "type": "supports"}],
        })
        obs = _extract(resp)
        assert obs.relations == [("c1", "c2", "enables")]

    def test_resolution_by_canonical_name(self):
        resp = json.dumps({
            "concepts": [{"name": "Alpha"}, {"name": "Beta"}],
            "relations": [{"source": "Alpha", "target": "Beta", "type": "supports"}],
        })
        obs = _extract(resp)
        assert obs.relations == [("Alpha", "Beta", "enables")]

    def test_resolution_by_alias(self):
        resp = json.dumps({
            "concepts": [
                {"name": "retrieval augmented generation", "aliases": ["RAG"]},
                {"name": "grounded answer"},
            ],
            "relations": [
                {"source": "RAG", "target": "grounded answer", "type": "supports"}
            ],
        })
        obs = _extract(resp)
        assert obs.relations == [
            ("retrieval augmented generation", "grounded answer", "enables")
        ]

    def test_resolution_by_normalized_case_and_punctuation(self):
        # Endpoint uses different case and punctuation than the concept name.
        resp = json.dumps({
            "concepts": [{"name": "Vector-Search"}, {"name": "embedding"}],
            "relations": [
                {"source": "vector search", "target": "EMBEDDING", "type": "depends_on"}
            ],
        })
        obs = _extract(resp)
        assert obs.relations == [("Vector-Search", "embedding", "dependence")]

    def test_unresolved_endpoint_recorded_with_reason(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}],
            "relations": [{"source": "a", "target": "ghost", "type": "supports"}],
        })
        obs = _extract(resp)
        assert obs.relations == []
        dropped = obs.extraction_metadata["dropped_relations"]
        assert len(dropped) == 1
        assert dropped[0]["source"] == "a"
        assert dropped[0]["target"] == "ghost"
        # rel_type is normalized at parse time before the drop is recorded.
        assert dropped[0]["type"] == "enables"
        assert dropped[0]["reason"] == "endpoint did not match any concept or alias"

    def test_relation_evidence_and_rationale_in_metadata(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "relations": [
                {
                    "source": "a",
                    "target": "b",
                    "type": "supports",
                    "evidence": "a supports b in text",
                    "rationale": "because reasons",
                }
            ],
        })
        obs = _extract(resp)
        rel_meta = obs.extraction_metadata["relations"][0]
        assert rel_meta["source"] == "a"
        assert rel_meta["target"] == "b"
        assert rel_meta["type"] == "enables"
        assert rel_meta["evidence"] == "a supports b in text"
        assert rel_meta["rationale"] == "because reasons"

    def test_raw_counts_correctness(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "relations": [
                {"source": "a", "target": "b", "type": "supports"},   # accepted
                {"source": "a", "target": "ghost", "type": "supports"},  # dropped
                {"source": "", "target": "c", "type": "supports"},  # empty -> filtered
            ],
        })
        obs = _extract(resp)
        counts = obs.extraction_metadata["raw_counts"]
        assert counts["concepts"] == 3
        assert counts["relations"] == 3
        assert counts["accepted_relations"] == 1
        assert counts["dropped_relations"] == 1

    def test_raw_counts_non_list_relations(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}],
            "relations": {"not": "a list"},
        })
        obs = _extract(resp)
        counts = obs.extraction_metadata["raw_counts"]
        assert counts["relations"] == 0
        assert counts["accepted_relations"] == 0
        assert obs.relations == []


# ── weakened + contradicted relations ────────────────────────────────


class TestWeakenedAndContradicted:
    def test_weakened_canonicalized_to_concept_name(self):
        resp = json.dumps({
            "concepts": [{"name": "Keyword Search"}],
            "weakened": ["keyword search"],
        })
        obs = _extract(resp)
        # Normalized key resolves the weakened token to the canonical name.
        assert obs.weakened == ["Keyword Search"]

    def test_weakened_via_alias_canonicalizes(self):
        resp = json.dumps({
            "concepts": [{"name": "term frequency", "aliases": ["TF"]}],
            "weakened": ["tf"],
        })
        obs = _extract(resp)
        assert obs.weakened == ["term frequency"]

    def test_weakened_unknown_kept_as_is(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}],
            "weakened": ["unknown thing"],
        })
        obs = _extract(resp)
        assert obs.weakened == ["unknown thing"]

    def test_weakened_deduplicated(self):
        # Two tokens that canonicalize to the same concept collapse to one.
        resp = json.dumps({
            "concepts": [{"name": "Keyword Search"}],
            "weakened": ["keyword search", "Keyword-Search", "KEYWORD  SEARCH"],
        })
        obs = _extract(resp)
        assert obs.weakened == ["Keyword Search"]

    def test_contradicted_relations_parsed_and_resolved(self):
        resp = json.dumps({
            "concepts": [
                {"name": "RAG", "aliases": ["retrieval augmented generation"]},
                {"name": "fine tuning"},
            ],
            "contradicted_relations": [
                {"source": "retrieval augmented generation", "target": "fine tuning",
                 "type": "contrasts"}
            ],
        })
        obs = _extract(resp)
        assert obs.contradicted_relations == [("RAG", "fine tuning", "conflict")]

    def test_contradicted_unresolved_endpoint_adds_parse_warning(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}],
            "contradicted_relations": [
                {"source": "a", "target": "phantom", "type": "contrasts"}
            ],
        })
        obs = _extract(resp)
        assert obs.contradicted_relations == []
        warnings = obs.extraction_metadata["parse_warnings"]
        assert any("contradicted relation endpoint not found" in w for w in warnings)
        assert any("phantom" in w for w in warnings)

    def test_contradicted_relations_tuple_form(self):
        resp = json.dumps({
            "concepts": [{"name": "a"}, {"name": "b"}],
            "contradicted_relations": [["a", "b", "conflict"]],
        })
        obs = _extract(resp)
        assert obs.contradicted_relations == [("a", "b", "conflict")]


# ── relation_priors / preset_relations ───────────────────────────────


class TestRelationPriors:
    def test_relation_prior_object_in_prompt_and_observation(self):
        resp = json.dumps({"concepts": [{"name": "a"}], "relations": []})
        extractor = ConceptExtractor(FakeLLM(resp))
        obs = extractor.extract(
            "text",
            preset_relations=[
                RelationPrior(
                    source="a",
                    target="b",
                    relation_type="supports",
                    probability=0.4,
                    rationale="prior belief",
                )
            ],
        )
        user = extractor._provider.user
        assert "## Preset Relations" in user
        # relation_type is normalized for the prompt.
        assert '"type": "enables"' in user
        assert '"rationale": "prior belief"' in user
        # Probability must NOT leak into the prompt.
        assert "probability" not in user
        assert "0.4" not in user
        # But it is preserved on the Observation.
        assert obs.relation_priors[0].probability == 0.4
        assert obs.relation_priors[0].source == "a"

    def test_dict_prior_coerced_with_relation_type(self):
        resp = json.dumps({"concepts": [{"name": "a"}]})
        extractor = ConceptExtractor(FakeLLM(resp))
        obs = extractor.extract(
            "text",
            preset_relations=[
                {"source": "a", "target": "b", "relation_type": "depends_on",
                 "probability": 0.7}
            ],
        )
        prior = obs.relation_priors[0]
        assert isinstance(prior, RelationPrior)
        assert prior.relation_type == "depends_on"
        assert prior.probability == 0.7

    def test_dict_prior_type_key_mapped_to_relation_type(self):
        resp = json.dumps({"concepts": [{"name": "a"}]})
        extractor = ConceptExtractor(FakeLLM(resp))
        obs = extractor.extract(
            "text",
            preset_relations=[{"source": "a", "target": "b", "type": "supports"}],
        )
        prior = obs.relation_priors[0]
        assert prior.relation_type == "supports"
        # In the prompt, "supports" is normalized to its canonical label.
        assert '"type": "enables"' in extractor._provider.user

    def test_invalid_dict_prior_skipped(self):
        # Missing required 'source'/'target' -> RelationPrior() raises -> skipped.
        resp = json.dumps({"concepts": [{"name": "a"}]})
        extractor = ConceptExtractor(FakeLLM(resp))
        obs = extractor.extract(
            "text",
            preset_relations=[
                {"target": "b", "type": "supports"},  # invalid (no source)
                {"source": "a", "target": "b", "type": "supports"},  # valid
            ],
        )
        assert len(obs.relation_priors) == 1
        assert obs.relation_priors[0].source == "a"

    def test_priors_preserved_on_empty_text(self):
        # Empty text short-circuits before the LLM call, but priors still ride.
        extractor = ConceptExtractor(FakeLLM("never called"))
        obs = extractor.extract(
            "   ",
            preset_relations=[RelationPrior(source="a", target="b")],
        )
        assert extractor._provider.calls == 0
        assert len(obs.relation_priors) == 1
        assert obs.relation_priors[0].source == "a"

    def test_no_preset_section_when_none(self):
        resp = json.dumps({"concepts": [{"name": "a"}]})
        extractor = ConceptExtractor(FakeLLM(resp))
        extractor.extract("text")
        assert "## Preset Relations" not in extractor._provider.user


# ── _number_or_none / _string_list edge cases ────────────────────────


class TestNumberOrNone:
    def test_none(self):
        assert ConceptExtractor._number_or_none(None) is None

    def test_empty_string(self):
        assert ConceptExtractor._number_or_none("") is None

    def test_int(self):
        assert ConceptExtractor._number_or_none(3) == 3.0

    def test_float(self):
        assert ConceptExtractor._number_or_none(0.5) == 0.5

    def test_numeric_string(self):
        assert ConceptExtractor._number_or_none("0.75") == 0.75

    def test_non_numeric_string(self):
        assert ConceptExtractor._number_or_none("high") is None

    def test_bool_is_coerced_to_float(self):
        # bool is a subclass of int; float(True) == 1.0 — documents behavior.
        assert ConceptExtractor._number_or_none(True) == 1.0

    def test_list_returns_none(self):
        assert ConceptExtractor._number_or_none([1, 2]) is None


class TestStringList:
    def test_non_list_returns_empty(self):
        assert ConceptExtractor._string_list("not a list") == []
        assert ConceptExtractor._string_list(None) == []
        assert ConceptExtractor._string_list(42) == []

    def test_strips_and_filters_blanks(self):
        assert ConceptExtractor._string_list(["  a ", "", "  ", "b"]) == ["a", "b"]

    def test_coerces_non_strings(self):
        assert ConceptExtractor._string_list([1, 2.5, "x"]) == ["1", "2.5", "x"]

    def test_empty_list(self):
        assert ConceptExtractor._string_list([]) == []


# ── End-to-end through World(store_path=tmp, llm=FakeLLM) ─────────────


def _world(tmp_path, response: str) -> World:
    return World(store_path=str(tmp_path / "w0"), llm=FakeLLM(response))


class TestEndToEnd:
    def test_synonym_alias_dedup_single_node(self, tmp_path):
        """Same concept declared once with an alias; the alias and canonical
        name both resolve to a single ConceptNode after ingest."""
        resp = json.dumps({
            "concepts": [
                {
                    "name": "retrieval augmented generation",
                    "kind": "technique",
                    "sense": "retrieval conditioned generation",
                    "description": "Grounds generation in retrieved context.",
                    "aliases": ["RAG"],
                }
            ],
            "relations": [],
        })
        world = _world(tmp_path, resp)
        world.ingest_text("RAG grounds answers.", task="t", source="s")

        by_canonical = world.concepts.resolve("retrieval augmented generation")
        by_alias = world.concepts.resolve("RAG")
        assert by_canonical is not None
        assert by_alias is not None
        assert by_canonical.id == by_alias.id

    def test_different_sense_same_name_creates_two_nodes(self, tmp_path):
        """Two concepts share a label but differ in sense -> two nodes."""
        resp = json.dumps({
            "concepts": [
                {
                    "uid": "c1",
                    "name": "Apple",
                    "kind": "entity",
                    "sense": "technology company",
                    "description": "Consumer technology company.",
                },
                {
                    "uid": "c2",
                    "name": "Apple",
                    "kind": "entity",
                    "sense": "fruit",
                    "description": "An edible fruit.",
                },
            ],
            "relations": [],
        })
        world = _world(tmp_path, resp)
        result = world.ingest_text("Apple vs apple.", task="t", source="s")
        # Two distinct senses -> two new concept nodes.
        assert len(result.new_concepts) == 2

        ids = {
            c.id for c in world.concepts.all()
            if c.name == "Apple"
        }
        assert len(ids) == 2

    def test_generic_noise_names_still_created(self, tmp_path):
        """Boundary: the extractor does NOT filter generic/noise concept
        names — World 0 creates them as-is.  This documents that filtering
        is intentionally out of the extractor's scope."""
        resp = json.dumps({
            "concepts": [
                {"name": "thing"},
                {"name": "stuff"},
                {"name": "it"},
            ],
            "relations": [],
        })
        world = _world(tmp_path, resp)
        result = world.ingest_text("Some text.", task="t", source="s")
        assert sorted(result.new_concepts) == ["it", "stuff", "thing"]
        for name in ("thing", "stuff", "it"):
            assert world.concepts.resolve(name) is not None

    def test_end_to_end_relation_persisted(self, tmp_path):
        resp = json.dumps({
            "concepts": [{"uid": "c1", "name": "cache"}, {"uid": "c2", "name": "latency"}],
            "relations": [{"source": "c1", "target": "c2", "type": "depends_on"}],
        })
        world = _world(tmp_path, resp)
        result = world.ingest_text("Cache affects latency.", task="t", source="s")
        assert result.new_relations  # at least the explicit relation
