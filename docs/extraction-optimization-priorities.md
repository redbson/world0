# World 0 Extraction Optimization Priorities

本文档定义 World 0 的 LLM 概念/关系抽取优化优先级。目标不是让抽取器“多抽”，而是让它更稳定地生成可用于激活、投影和长期演化的概念结构。

## 背景

当前抽取链路位于：

- `src/world0/extraction/extractor.py`
- `src/world0/prompts/defaults.py`
- `src/world0/world/facade.py`
- `src/world0/world/_ingest.py`

现状是：`ConceptExtractor` 把原始文本发送给 LLM，要求返回扁平 JSON：

```json
{
  "concepts": [{"name": "concept name", "description": "one-line description"}],
  "relations": [{"source": "concept A", "target": "concept B", "type": "relation_type"}]
}
```

该设计实现简单，但会导致浅层关键词、泛关系、重复概念、关系方向错误，以及无法审计抽取质量。

## 优化原则

1. **概念质量优先于数量**  
   抽取结果应该支持概念稳定、关系清晰和投影有用，而不是覆盖文本中的所有名词。

2. **证据优先于推测**  
   概念和关系应尽量绑定来源证据，避免 LLM 仅凭常识扩展。

3. **任务上下文优先于全文显著性**  
   同一段文本在不同任务下应激活不同概念，抽取器必须接收 task/context。

4. **兼容旧 schema，逐步增强**  
   先让 extractor 兼容新字段，再逐步让 ingest 和 projection 使用这些字段。

5. **用 golden tests 锁定质量**  
   prompt 优化必须有可重复验证的案例，不能只靠主观试用。

## 优先级总览

| 优先级 | 项目 | 价值 | 风险 | 建议顺序 |
| --- | --- | --- | --- | --- |
| P0 | 加强 extraction prompt 和输出 schema | 高 | 低 | 先做 |
| P0 | 增加 golden extraction tests | 高 | 低 | 同步做 |
| P1 | 将 task/source/domain 注入抽取输入 | 高 | 低 | 紧接 P0 |
| P1 | endpoint 对齐与 alias/fuzzy resolution | 高 | 中 | P0 后 |
| P1 | JSON repair/retry 和错误可观测性 | 中 | 低 | P0 后 |
| P2 | 引入 salience/confidence/evidence 元数据 | 高 | 中 | schema 稳定后 |
| P2 | existing-world-aware extraction | 高 | 中 | 元数据后 |
| P3 | 两阶段抽取：概念先行，关系随后 | 高 | 中高 | 质量基准稳定后 |
| P3 | 关系类型细化器/审计器 | 中 | 中 | 视测试结果 |

## P0: Prompt 与 Schema 升级

### 问题

当前 prompt 只定义了概念和关系，没有要求：

- 证据片段
- 概念重要性
- 概念类型
- 关系置信度
- 关系方向理由
- 是否应弱化旧概念
- 是否存在反证关系

LLM 因此倾向于输出“看起来像关键词”的内容，而不是 World 0 需要的概念结构。

### 建议 schema

保持兼容旧格式，同时允许新格式：

```json
{
  "domain": "domain label",
  "concepts": [
    {
      "name": "canonical concept name",
      "description": "one-sentence concept boundary",
      "kind": "core|supporting|background|entity|process|principle",
      "salience": 0.0,
      "confidence": 0.0,
      "evidence": "short source excerpt",
      "aliases": ["alternate name"]
    }
  ],
  "relations": [
    {
      "source": "concept A",
      "target": "concept B",
      "type": "depends_on",
      "confidence": 0.0,
      "evidence": "short source excerpt",
      "rationale": "why this type and direction are correct"
    }
  ],
  "weakened": ["concept name"],
  "contradicted_relations": [
    {"source": "concept A", "target": "concept B", "type": "supports"}
  ]
}
```

### Prompt rules to add

- Extract only concepts that help future task understanding.
- Prefer stable conceptual units over one-off facts.
- Do not extract generic nouns unless domain-qualified.
- Use `related_to` only when no more specific relation applies.
- Every relation must have evidence from the input.
- Relation direction must follow the semantic dependency or causal direction.
- Include aliases only for names present or strongly implied in the text.
- If a concept is only background, mark it as `background` and use low salience.

### Acceptance criteria

- Existing tests continue to pass.
- Extractor parses both old and new schema.
- Golden tests prove that generic terms are filtered or downranked.
- Relations include evidence and rationale in raw parsed metadata, even if ingest does not use all fields immediately.

## P0: Golden Extraction Tests

### Why

Without quality tests, prompt edits are unverifiable. The test suite currently checks JSON parsing and validation, but not semantic extraction quality.

### Required fixtures

Add 8-12 cases covering:

1. Technical architecture text  
   Should extract architecture concepts and typed dependencies.

2. Research paragraph  
   Should extract mechanisms, claims, uncertainty, and open concepts.

3. User dialogue  
   Should focus on task-relevant concepts, not conversational filler.

4. Synonyms and acronyms  
   Example: `RAG` and `retrieval augmented generation` should not become unrelated duplicate concepts.

5. Relation direction  
   Example: `Redis cache reduces latency` should not produce `latency supports redis cache`.

6. Generic noise  
   Terms like `system`, `process`, `thing`, `data`, `result` should not appear unless domain-qualified.

7. Contradiction/disconfirmation  
   Text that rejects an earlier relation should populate `contradicted_relations`.

8. Domain-sensitive extraction  
   Same term in different domains should preserve domain context.

### Acceptance criteria

- Tests are deterministic with fake LLM responses for parser behavior.
- At least one integration-style test checks an extracted observation through `World.ingest()`.
- Expected behavior is documented in test names, not hidden in comments.

## P1: Inject Task, Source, and Domain Into Extraction

### Problem

`ConceptExtractor.extract()` receives `task` and `source`, but currently sends only raw `text` to the LLM. This loses the reason for extraction.

### Design

Render user prompt as structured input:

```text
## Task Context
{task}

## Source
{source}

## Extraction Goal
Extract concepts and relations that would help World 0 build a reusable cognitive projection for this task.

## Text
{text}
```

If domain is available later, include it as a separate field.

### Acceptance criteria

- LLM sees task/source in extraction calls.
- Tests assert that the provider receives structured user prompt.
- Existing direct extraction behavior remains compatible.

## P1: Endpoint Alignment and Alias Resolution

### Problem

Relations are silently dropped when endpoints do not exactly match extracted concept names. Example:

```json
{
  "concepts": [{"name": "retrieval augmented generation", "aliases": ["RAG"]}],
  "relations": [{"source": "RAG", "target": "vector search", "type": "depends_on"}]
}
```

The current parser would likely drop the relation if `RAG` is not in the concept list as a canonical name.

### Design

Before dropping a relation:

1. Build canonical map from names and aliases.
2. Normalize case, whitespace, punctuation.
3. Try alias match.
4. Try simple token containment for high-confidence obvious cases.
5. Only then drop and record a parse warning.

### Acceptance criteria

- Alias endpoint relations survive parsing.
- Dropped relations are observable in debug metadata or parser warnings.
- No self-relations are created after normalization.

## P1: JSON Repair and Observability

### Problem

Invalid LLM JSON currently returns an empty `Observation`. This hides extraction failures.

### Design

- Add a single repair attempt when JSON parsing fails.
- If repair is unavailable, return empty observation but attach or log parse failure metadata.
- Track counts:
  - raw concepts returned
  - accepted concepts
  - raw relations returned
  - accepted relations
  - dropped relations
  - parse repair used

### Acceptance criteria

- Malformed JSON failure is detectable in tests.
- Silent empty extraction becomes rare and explainable.

## P2: Use Salience, Confidence, and Evidence

### Problem

All created concepts start with the same base confidence. All explicit relations start with the same base relation confidence. This treats weak background mentions and strong core claims equally.

### Design

Add extraction metadata in a way that does not overload the core concept model too early:

- In parser: preserve salience/confidence/evidence.
- In ingest: optionally map confidence into initial confidence.
- In relation manager: optionally map relation confidence into initial edge confidence.
- Evidence can initially stay in provenance/debug metadata before becoming first-class schema.

### Acceptance criteria

- Low-salience background concepts do not dominate activation.
- Strong evidence concepts start stronger than incidental mentions.
- Projection quality improves on golden cases.

## P2: Existing-World-Aware Extraction

### Problem

The extractor does not know existing concepts or aliases, so it may create duplicates.

### Design

Before extraction:

1. Use lightweight token search over existing concepts using text + task.
2. Pass top candidates to the LLM:

```text
Existing concepts to reuse when appropriate:
- retrieval augmented generation: ...
- vector search: ...
- context window: ...
```

3. Tell LLM to reuse canonical names when the same concept appears.

### Acceptance criteria

- Duplicate rate decreases in synonym/acronym tests.
- Existing concept activation increases instead of creating near-duplicates.
- Prompt remains bounded; do not dump the whole concept world.

## P3: Two-Stage Extraction

### Problem

Asking the LLM to identify concepts and relations in one pass makes relation quality depend on unstable concept boundaries.

### Design

Stage 1: Concept candidates

- Extract concepts with evidence, kind, salience, aliases.
- Filter/downrank.
- Canonicalize names.

Stage 2: Relations between accepted concepts

- Provide only accepted concept list.
- Ask for typed relations with direction, evidence, rationale.

### Acceptance criteria

- Relation endpoint mismatch drops sharply.
- `related_to` usage decreases.
- Direction tests pass more consistently.

## P3: Relation Type Auditor

### Problem

Relation type mistakes are hard to fix once ingested and reinforced.

### Design

Add optional relation audit pass:

- Input: accepted concepts + proposed relations + evidence.
- Output: corrected type/direction, or reject.
- Only use for high-value/long inputs or when relation confidence is low.

### Acceptance criteria

- Fewer incorrect `depends_on/supports/activates` edges.
- Audit cost is bounded and optional.

## Implementation Order

1. Update `extraction.concepts_relations.system`.
2. Extend parser to read new schema while preserving old schema behavior.
3. Add golden parser tests and integration tests.
4. Inject task/source into extraction user prompt.
5. Add alias endpoint resolution.
6. Add parser warning/metadata surface.
7. Map confidence/salience into ingest behavior.
8. Add existing-world candidates.
9. Consider two-stage extraction only after tests show single-pass limits.

## Non-Goals

- Do not turn extraction into a general document archive.
- Do not store raw source documents in World 0 core.
- Do not rely only on embeddings for identity.
- Do not make prompt templates concept cards.
- Do not create a workflow engine inside extraction.

## Success Metrics

Track these over fixed evaluation texts:

- accepted concepts per 1,000 tokens
- duplicate concept rate
- generic/noise concept rate
- relation endpoint drop rate
- `related_to` ratio
- relation direction accuracy
- projection usefulness in downstream `ask()`
- empty extraction rate

## Recommended First Patch

The first patch should be intentionally narrow:

1. Replace the default extraction prompt with the enhanced schema.
2. Add parser support for:
   - `domain`
   - concept `aliases`
   - concept `salience`
   - concept `confidence`
   - concept `evidence`
   - relation `confidence`
   - relation `evidence`
   - relation `rationale`
   - `weakened`
   - `contradicted_relations`
3. Keep `Observation` output compatible by still returning:
   - `concepts`
   - `relations`
   - `descriptions`
   - `domain`
   - `weakened`
   - `contradicted_relations`
4. Add parser tests for both old and new schema.

This gives immediate quality gains without forcing a large redesign of concept storage or projection.
