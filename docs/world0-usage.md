# World 0 使用文档 / World 0 Usage Guide

本文档面向三类使用者：

This document is written for three kinds of users:

- 把 `world0` 当 Python 库使用的开发者
- 通过 `pkm` / `pkm-web` / `pkm-gui` 跑 Agent 的使用者
- 想基于 Protocol 扩展或替换模块的集成者
- developers using `world0` as a Python library
- users driving the agent through `pkm`, `pkm-web`, or `pkm-gui`
- integrators who want to extend or replace modules via the Protocol contracts

---

## 1. World 0 是什么 / What World 0 Is

World 0 是一个面向 LLM Agent 的“概念世界”系统。它的核心不是存事实，而是把任务中的概念、关系、上下文和激活状态组织成一个可投影的局部认知视图。

World 0 is a concept-world system for LLM agents. Its goal is not to store facts, but to organize concepts, relations, context, and activation into a projectable local cognitive view for a task.

它解决的是这类问题：

It is designed to answer questions like:

- 当前任务里哪些概念最重要
- 这些概念之间是什么关系
- 现在应该向 Agent 注入怎样的局部概念视图
- Which concepts matter most for the current task?
- How are those concepts related?
- What local conceptual projection should be injected into the agent now?

明确不做：通用知识库、记忆系统、文档归档、工作流引擎。

Explicit non-goals: general knowledge base, memory system, document archive, workflow engine.

---

## 2. 模块结构 / Module Layout

项目按“乐高积木”原则组织 —— 所有跨模块调用只依赖 `core/` 里定义的 Protocol 契约，任何实现都可以被替换。

The project is organized Lego-style — every cross-module call depends only on the Protocol contracts in `core/`, and any implementation can be swapped out.

| 层 / Layer | 模块 / Module | 作用 / Responsibility |
| --- | --- | --- |
| 契约根 / Contract root | `core/` | 所有 Protocol（`StorageBackend`、`ConceptStore`、`RelationStore`、`ActivationProvider`、`HebbianLearner`、`DecayPolicy`、`LifecyclePolicy`、`ColorField`、`Projector`、`Extractor`、`LLMProvider`、`WorldView`…）、事件总线、测试替身 |
| 数据模型 / Schemas | `schemas/` | `ConceptNode`、`RelationEdge`、`Observation`、`Projection`、`IngestResult`、`ReflectResult`、`WorldStatus`、`Perspective`、`Community` |
| 存储 / Storage | `store/` | `JsonStore` — 满足 `StorageBackend` 的默认持久化 |
| 概念积木 / Concepts | `concepts/` | `Concepts`（= `ConceptManager`）：创建、解析、强化、别名与索引 |
| 关系积木 / Relations | `relations/` | `RelationManager`：发现、强化、遍历、类型细化 |
| 动力学 / Dynamics | `dynamics/` | 6 个引擎：`activation`、`hebbian`、`decay`、`lifecycle`、`community`、`color_diffusion` |
| 社区 / Communities | `communities/` | 标签传播社区检测 + 跨 reflect 周期状态 |
| 投影 / Projection | `projection/` | MMR 式局部概念视图生成 |
| 可视化 / Visualization | `visualization/` | `renderer` + `_graph_data` + `template.html` |
| 编排 / Orchestration | `world/` | `facade.World` + 三条流水线：`IngestPipeline`、`ReflectPipeline`、`IdentityOps` |
| 提取 / Extraction | `extraction/` | 从原始文本抽取 `Observation` |
| LLM 适配 / LLM | `llm/` | OpenAI、Anthropic、Azure OpenAI provider |
| 上层 Agent | `agents/` | `pkm`（CLI）、`pkm-web`（浏览器）、`pkm-gui`（macOS 原生）、MCP 服务端 |

每个积木都自带 `tests/` 子目录，用 `world0.core.test_doubles` 提供的 Protocol 假实现做单元测试 —— 不需要启动真实存储或其它模块。

Every brick ships a sibling `tests/` directory that drives it with Protocol fakes from `world0.core.test_doubles` — no real storage or sibling modules needed.

---

## 3. 安装 / Installation

基础：

Base:

```bash
pip install -e .
```

开发：

Development:

```bash
pip install -e ".[dev]"
```

浏览器 UI：

Browser UI:

```bash
pip install -e ".[web]"
```

macOS 原生窗口：

Native macOS window:

```bash
pip install -e ".[gui]"
```

LLM 提取：

LLM extraction:

```bash
pip install -e ".[openai]"      # OpenAI / Azure OpenAI
pip install -e ".[anthropic]"   # Anthropic
pip install -e ".[all]"         # 全部 / everything
```

Python 要求 ≥ 3.10。

Python ≥ 3.10 required.

---

## 4. 最小使用路径 / Minimal Workflow

World 0 的最小操作链：

The minimal loop:

1. `ingest` — 输入观察
2. `project` — 为任务生成局部投影
3. `reflect` — 任务结束后做巩固、衰减和修剪
1. `ingest` — feed an observation in
2. `project` — produce a task-local projection
3. `reflect` — consolidate, decay, and prune after the task

### 4.1 Python API

```python
from world0 import Observation, World

w = World(store_path=".world0")

w.ingest(Observation(
    concepts=["FastAPI", "PostgreSQL", "Latency"],
    relations=[
        ("FastAPI", "PostgreSQL", "depends_on"),
        ("Latency", "PostgreSQL", "related_to"),
    ],
    descriptions={
        "FastAPI": "Python async web framework",
        "Latency": "Request-response delay under load",
    },
    task="debug production API latency",
    source="incident_042",
))

projection = w.project(
    ["FastAPI", "Latency"],
    task="find the most relevant conceptual neighborhood",
    max_concepts=8,
    max_depth=2,
)

print(projection.render())   # Markdown — 直接拼进 prompt

reflect_result = w.reflect()
print(reflect_result.promoted_concepts)
```

### 4.2 `Observation` 字段 / Fields

| 字段 / Field | 类型 / Type | 说明 / Meaning |
| --- | --- | --- |
| `concepts` | `list[str]` | 本次任务涉及的概念名 |
| `relations` | `list[tuple[src, tgt, type]]` | 显式关系三元组；type 可用枚举值或字符串 |
| `descriptions` | `dict[str, str]` | 可选的概念描述 |
| `weakened` | `list[str]` | 负向证据：本次任务里被证伪/不相关的概念 |
| `contradicted_relations` | `list[tuple]` | 负向证据：未成立的关系 |
| `domain` | `str` | 领域标签，驱动色彩场扩散 |
| `task` | `str` | 任务上下文 |
| `source` | `str` | 来源标识（会话、工单、文档名等） |
| `timestamp` | `datetime` | 默认自动填充 |

正负双通道：`concepts` + `relations` 提供正向证据；`weakened` + `contradicted_relations` 提供负向证据，驱动 Beta 风格的置信度更新。

Positive + negative channels: `concepts`/`relations` give positive evidence; `weakened`/`contradicted_relations` give negative evidence and drive Beta-style confidence updates.

---

## 5. 核心 API / Core API

### `World(store_path=".world0", llm=None)`

构造时会自动：

On construction, the facade will:

- 打开 `JsonStore(store_path)` 并 `load()` 概念和关系
- 用默认实现装配 6 个动力学引擎、投影引擎、社区检测器
- 从 `state.json` 恢复社区快照（如果存在）
- open `JsonStore(store_path)` and `load()` concepts and relations
- wire up the 6 dynamics engines, projection engine, and community detector with the default implementations
- restore the community snapshot from `state.json` if present

传入 `llm=...` 才会启用 `ingest_text()`。

Passing `llm=...` is what enables `ingest_text()`.

### `World.ingest(observation) -> IngestResult`

- 创建或强化概念、关系（含负向削弱）
- 对共现概念做 Hebbian 学习（超过阈值时生成隐式关系）
- 更新概念的领域色彩场
- 由 facade 刷盘（流水线本身不落盘）
- creates or reinforces concepts and relations (with negative weakening)
- runs Hebbian learning on co-occurring concepts (generates implicit relations past threshold)
- updates per-concept domain color field
- facade owns the flush boundary — pipelines never persist

返回 `IngestResult`：`new_concepts`、`reinforced_concepts`、`weakened_concepts`、`new_relations`、`reinforced_relations`、`weakened_relations`、`hebbian_relations`。

### `World.ingest_text(text, *, task="", source="") -> IngestResult`

用配置的 LLM provider 把原始文本抽成 `Observation`，再交给 `ingest()`。未配 `llm` 时抛 `RuntimeError`。

Uses the configured LLM provider to extract an `Observation` from raw text, then delegates to `ingest()`. Raises `RuntimeError` if no `llm` was configured.

### `World.project(seeds, *, task="", perspective=None, max_concepts=15, max_depth=2, decay=0.5) -> Projection`

- 把 `seeds` 名解析成概念 id
- 扩散激活（`ActivationEngine`）
- 经 `ProjectionEngine`（MMR）过滤出局部视图
- 返回 `Projection`
- resolves seed names → concept ids
- runs spreading activation (`ActivationEngine`)
- filters via `ProjectionEngine` (MMR)
- returns a `Projection`

`Projection.render()` 产出可直接注入 prompt 的 Markdown：分区为 *Core Understanding / Active Concepts / Emerging Concepts / Key Relations / Task Context*。`Projection.top_concepts(n)` 返回按激活分排序的前 N 个概念。

`Projection.render()` produces prompt-ready Markdown grouped into *Core Understanding / Active Concepts / Emerging Concepts / Key Relations / Task Context*. `Projection.top_concepts(n)` returns the top-N concepts by activation score.

### `World.reflect() -> ReflectResult`

五阶段流水线：

Five-stage pipeline:

1. 概念衰减 + 关系衰减
2. 社区检测与更新
3. 色彩场 fade → 由社区重新播种 → settle
4. 生命周期评估（成熟度升降）
5. 关系修剪 → 概念修剪
1. decay concepts + decay relations
2. detect and update communities
3. color-field fade → reseed from communities → settle
4. lifecycle evaluation (maturity up/down)
5. prune relations → prune concepts

`ReflectResult` 包含：`decayed_*`、`promoted_concepts`、`demoted_concepts`、`pruned_*`、`new_communities`、`stable_communities`、`pruned_communities`、`color_sources`。

建议在任务结束或阶段切换时调用，不要每轮交互都 reflect。

Call it at task or phase boundaries — avoid reflecting after every single interaction.

### `World.status() -> WorldStatus`

即时返回：`total_concepts`、`total_relations`、`by_maturity`、`avg_confidence`、`last_reflect`、`total_communities`、`stable_communities`、`bridge_concepts`、`avg_color_purity`。不触发衰减。

Returns the cognitive world snapshot immediately without triggering decay.

### `World.visualize(output=None, *, open_browser=True) -> Path`

渲染交互式 HTML 概念网络。`output=None` 时落到当前目录的 `world0_viz.html`。

Renders an interactive HTML concept network. Defaults to `./world0_viz.html`.

### 身份操作 / Identity Operations

| API | 作用 / Does |
| --- | --- |
| `merge(keeper, absorbed)` | 把 `absorbed` 并入 `keeper`：别名迁移、关系迁移、节点删除 |
| `split(source, new_name, *, aliases_to_move=..., description="")` | 从 `source` 分裂出一个新概念，返回新 id |
| `weaken(concept, *, source="", task="")` | 单独降低某个概念的置信度 |
| `find_similar(text, *, domain="", min_similarity=0.3, limit=5)` | 用签名相似度找相近概念，返回 `[(concept_id, score), ...]` |

身份操作修改完会立即刷盘，和 `ingest`/`reflect` 一致。

Identity operations flush immediately, matching `ingest`/`reflect` semantics.

---

## 6. 关系类型 / Relation Types

`relations=[(src, tgt, type)]` 里 `type` 支持下列枚举（字符串/枚举值均可）：

```
contains · part_of · depends_on · supports · contrasts ·
similar_to · activates · precedes · derived_from · related_to
```

`related_to` 是默认/回退类型。Agent 后续可用 `RelationManager.refine_type()` 细化类型。

`related_to` is the default / fallback. Agents can refine types later via `RelationManager.refine_type()`.

---

## 7. LLM 提取 / LLM Extraction

```python
from world0 import World
from world0.llm import OpenAIProvider, AnthropicProvider, AzureOpenAIProvider

w = World(
    store_path=".world0",
    llm=OpenAIProvider(model="gpt-5.4"),
)

w.ingest_text(
    "We migrated the auth service from Express to FastAPI and now use PostgreSQL.",
    task="auth migration",
    source="migration_note",
)
```

运行时换 provider：

Swap providers at runtime:

```python
w.set_llm(AnthropicProvider(model="claude-sonnet-4-6"))
w.set_llm(None)   # 关闭 LLM，恢复到只接受结构化 Observation
```

环境变量：

Environment variables:

- OpenAI — `OPENAI_API_KEY`
- Anthropic — `ANTHROPIC_API_KEY`
- Azure OpenAI — `AZURE_OPENAI_API_KEY` 或 `AZURE_OPENAI_KEY`，以及 `AZURE_OPENAI_ENDPOINT`

---

## 8. CLI 使用 / CLI Usage

安装后可用 `pkm` 命令。默认存储目录是 `~/.pkm_world`，默认 provider 是 `anthropic`。

After install, `pkm` is available. Default store: `~/.pkm_world`. Default provider: `anthropic`.

### 8.1 全局参数 / Global Flags

```
--store <path>                # 存储目录
--provider {anthropic,openai,azure-openai,none}
--model   <name>              # 覆盖默认模型
```

### 8.2 子命令 / Subcommands

```bash
# 从文本学习（需 provider）
pkm --provider openai learn "Transformers use self-attention."

# 问答（基于当前概念世界做局部投影）
pkm --provider none ask "What do I know about FastAPI?"

# 探索单个概念
pkm --provider none explore "FastAPI"

# 手工建立关系
pkm --provider none connect "FastAPI" "PostgreSQL" --type depends_on

# 搜索概念
pkm --provider none search "latency"

# 公网搜索 + 可选抓取摘要
pkm --provider anthropic web-search "latest MCP patterns" \
    --limit 5 --fetch-pages --domains "modelcontextprotocol.io"

# 查看状态
pkm --provider none status

# 跑一次巩固
pkm --provider none reflect

# 生成可视化
pkm --provider none viz --output world0_viz.html
```

不带子命令启动则进入交互终端。

Running `pkm` without a subcommand enters the interactive terminal.

---

## 9. Web / GUI

浏览器版：

Browser:

```bash
pkm-web --provider none --host 127.0.0.1 --port 8420
```

macOS 原生窗口：

Native macOS window:

```bash
pkm-gui --provider none
```

两者都接受和 `pkm` 一样的 `--store` / `--provider` / `--model`。

Both accept the same `--store` / `--provider` / `--model` flags as `pkm`.

---

## 10. 持久化布局 / Persistence Layout

默认 JSON 布局：

Default JSON layout:

```
.world0/
├── concepts/         # 概念节点 / concept nodes
├── relations/        # 关系边 / relation edges
└── state.json        # 世界级状态：last_reflect、社区快照等
```

刷盘时机由 facade 决定：`ingest()`、`reflect()`、身份操作结束后各刷一次。流水线自身不碰磁盘 —— 这让它们可以在纯内存的 `FakeConceptStore` / `FakeRelationStore` 下单独测试。

Flush boundaries are owned by the facade: once each after `ingest()`, `reflect()`, and identity operations. Pipelines themselves never touch disk, which lets them be tested against in-memory `FakeConceptStore` / `FakeRelationStore` in isolation.

---

## 11. 扩展与替换 / Extension and Replacement

因为所有跨模块调用都走 `core/` Protocol，替换任一积木只需要实现对应 Protocol。

Because every cross-module call goes through a Protocol in `core/`, replacing any brick only requires implementing the Protocol.

示例：用你自己的存储后端替换 `JsonStore`。

Example — swap in your own storage backend:

```python
from world0.core import StorageBackend
from world0.concepts.api import Concepts

class MyBackend:
    # 只要你实现 StorageBackend 要求的所有方法即可
    def save_concept(self, concept): ...
    def load_concept(self, concept_id): ...
    def load_all_concepts(self): ...
    # ... 其余方法见 world0.core.interfaces.StorageBackend

backend: StorageBackend = MyBackend()     # runtime_checkable Protocol
concepts = Concepts(backend)              # 不需要改 ConceptManager
```

想在不接触真实存储的前提下测试自己的动力学引擎？用 `core.test_doubles`：

Want to test your own engine without touching real storage? Use `core.test_doubles`:

```python
from world0.core.test_doubles import (
    FakeConceptStore, FakeRelationStore, make_concept, make_edge,
)
from my_pkg import MyDecayEngine

a = make_concept("alpha", confidence=0.8)
b = make_concept("beta",  confidence=0.2)
cs = FakeConceptStore(seed=[a, b])
rs = FakeRelationStore(seed=[make_edge(a.id, b.id, weight=0.4)])

engine = MyDecayEngine(cs, rs)
assert engine.decay_concepts()   # 只走 Protocol 接口，完全离线
```

可用的假实现：`FakeStorageBackend`、`FakeConceptStore`、`FakeRelationStore`、`FakeHebbianLearner`、`FakeDecayPolicy`、`FakeLifecyclePolicy`、`FakeColorField`、`FakeWorldView`。每个都在 `calls` 属性上记录收到的调用，便于断言。

Available fakes: `FakeStorageBackend`, `FakeConceptStore`, `FakeRelationStore`, `FakeHebbianLearner`, `FakeDecayPolicy`, `FakeLifecyclePolicy`, `FakeColorField`, `FakeWorldView`. Each records received calls on a `.calls` attribute for easy assertions.

事件总线可选：

Optional event bus:

```python
from world0.core import InMemoryEventBus, ConceptCreated

bus = InMemoryEventBus()
bus.subscribe(ConceptCreated, lambda ev: print("new concept:", ev.name))
```

---

## 12. 推荐工作方式 / Recommended Working Styles

### 方式一：结构化摄入 / Structured ingestion

已经知道本轮任务有哪些概念和关系：

You already know the concepts and relations involved:

1. 用 `Observation` 明确正/负向证据
2. 调 `ingest()`
3. `project()` 出局部视图

### 方式二：文本摄入 / Text ingestion

有会议纪要、日志、对话或研究笔记：

Meeting notes, logs, chats, research notes:

1. 配 LLM provider
2. `ingest_text()`
3. `project()` 出任务相关的局部概念图

### 方式三：周期性巩固 / Periodic consolidation

长期运行的 Agent：

Long-running agents:

1. 任务过程中持续 `ingest`
2. 阶段结束调 `reflect`
3. `status()` 监控膨胀 / 退化

---

## 13. 常见问题 / FAQ

**`project()` 返回空？** 通常是：种子概念没 `ingest` 过、拼写不一致、或 `store_path` 指错。

**`project()` empty?** Usually: seeds never ingested, name mismatch, or wrong `store_path`.

**`ingest_text()` 报错？** 没传 `llm=...`、对应依赖未装、或 API key 未配。

**`ingest_text()` fails?** No `llm=...` passed, provider extra not installed, or API key missing.

**为什么 `reflect` 要显式调用？** 把“输入”和“巩固”拆开。这样每次输入不会立刻触发衰减和修剪，避免行为抖动，也更贴近批次式任务节奏。

**Why is `reflect` explicit?** It separates observation ingestion from consolidation — keeps every input from triggering decay and pruning, reduces jitter, matches batch-oriented task rhythms.

**能不能完全关掉色彩场和社区？** 可以。继承 `World` 后覆盖 `self._color_diffusion` / `self._communities` 为 no-op 实现（满足 `ColorField` / `CommunityDetectorP` 即可）。

**Can I disable color field and communities entirely?** Yes — subclass `World` and replace `self._color_diffusion` / `self._communities` with no-op implementations that satisfy `ColorField` / `CommunityDetectorP`.

---

## 14. 开发与验证 / Development

跑全量测试（包含每个积木的本地 `tests/`）：

Run the full suite (including each brick's local `tests/`):

```bash
pytest -q
```

只跑某个积木的 Protocol 级测试：

Only the Protocol-level tests of one brick:

```bash
pytest src/world0/concepts/tests -q
pytest src/world0/dynamics/tests -q
pytest src/world0/world/tests -q
pytest src/world0/visualization/tests -q
```

跑顶层集成测试：

Top-level integration tests:

```bash
pytest tests/ -q
```

进一步阅读：

Further reading:

- [README](../README.md)
- [DesignPhilosophy.md](../DesignPhilosophy.md)
- [DCTMTemporalDynamics.md](../DCTMTemporalDynamics.md)
- [world0-paper.md](world0-paper.md)
- [world0-color-field-dynamics.md](world0-color-field-dynamics.md)
- [TODO.md](../TODO.md)
