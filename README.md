# World 0

A cognitive concept-world system for LLM Agents.
<br>面向 LLM Agent 的认知概念世界系统。

World 0 organizes concepts and relations into context-sensitive, projectable structures that support **understanding** rather than storage. It gives an Agent a persistent cognitive layer: concepts are created, connected, activated, and projected into task-relevant local views that can be injected directly into prompts.

World 0 将概念和关系组织为上下文敏感、可投影的结构，服务于**理解**而非存储。它为 Agent 提供持久化的认知层：概念被创建、连接、激活，并投影为与任务相关的局部视图，可直接注入到 prompt 中。

## Why World 0 / 为什么需要 World 0

LLM Agents work on tasks, not databases. They need a system that can answer:

LLM Agent 面向任务工作，而非面向数据库。它们需要一个能回答以下问题的系统：

- What concepts matter for the task at hand? / 当前任务中哪些概念重要？
- How are they related? / 它们之间有什么关系？
- What local conceptual view should I inject into the next prompt? / 应该向下一个 prompt 注入怎样的局部概念视图？

World 0 is not a knowledge graph, not a memory system, not an ontology. It is a cognitive structure that turns accumulated observations into focused, task-relevant projections.

World 0 不是知识图谱，不是记忆系统，不是本体论。它是一个认知结构，将积累的观察转化为聚焦的、与任务相关的投影。

## Roadmap / 路线

The current agent development priorities are tracked in [`TODO.md`](TODO.md).

当前 Agent 的功能开发优先级记录在 [`TODO.md`](TODO.md)。

## Documentation / 文档

- [`docs/world0-usage.md`](docs/world0-usage.md) — operational usage guide for World 0 / World 0 操作与使用文档
- [`docs/world0-color-field-dynamics.md`](docs/world0-color-field-dynamics.md) — dynamics-first design for community-born color fields / 基于动力学的群落生色与褪色设计
- [`DesignPhilosophy.md`](DesignPhilosophy.md) — design rationale and framing / 设计哲学与边界
- [`TODO.md`](TODO.md) — current implementation priorities / 当前实现优先级

## Quickstart / 快速开始

The repository includes a launcher that can set up a local virtualenv and
run the CLI, browser UI, native GUI, tests, and one-shot agent commands:

本仓库包含一个启动器，可用于创建本地虚拟环境，并启动 CLI、浏览器界面、
原生 GUI、测试和一次性 Agent 命令：

```bash
./start_world0.sh setup
./start_world0.sh
./start_world0.sh web --provider none
./start_world0.sh ask "What concepts matter here?"
./start_world0.sh test -- -k projection
```

For a resident browser-based GUI service:

如需启动常驻的浏览器 GUI 服务：

```bash
./start_world0.sh service-start --provider none
./start_world0.sh service-status
./start_world0.sh service-logs -f
./start_world0.sh service-stop
```

You can still use the installed Python entry points directly:

你也可以继续直接使用 Python 安装后的入口：

```bash
pip install -e .
```

If you want the browser UI:

如果你希望使用浏览器界面：

```bash
pip install -e ".[web]"
pkm-web --provider none
```

If you want the native macOS window:

如果你希望使用 macOS 原生窗口：

```bash
pip install -e ".[gui]"
pkm-gui
```

If you want LLM-powered extraction or agentic mode, also install a provider extra:

如果你希望使用 LLM 提取或 agentic 模式，还需要安装 provider 扩展：

```bash
pip install -e ".[openai,web]"
# or / 或
pip install -e ".[anthropic,web]"
```

### Interface Modes / 交互入口

- `pkm` — terminal interface / 终端交互
- `pkm-web` — browser interface / 浏览器界面
- `pkm-gui` — native GUI / 原生 GUI
- `World(...)` — Python library API / Python 库接口

For a task-oriented walkthrough of the Python API, CLI, Web/UI launchers,
LLM provider setup, persistence layout, and recommended workflows, see
[`docs/world0-usage.md`](docs/world0-usage.md).

如果你需要面向任务的完整使用说明，包括 Python API、CLI、Web/UI 启动、
LLM provider 配置、持久化目录结构与推荐工作流，请查看
[`docs/world0-usage.md`](docs/world0-usage.md)。

### Research Workflow / 研究工作流

The browser UI now includes a dedicated `Research` mode. It can:

浏览器界面现在包含独立的 `Research` 模式。它可以：

- search the public web for a topic / 为主题搜索公开网页
- fetch and read candidate sources / 抓取并阅读候选来源
- distill findings into concept-first notes / 将结果提炼为概念优先的笔记
- optionally learn those findings into World 0 / 可选择把研究结果学习进 World 0
- project the updated concept-world back into an Agent-facing brief / 将更新后的概念世界重新投影为 Agent 可用简报

### Think With World 0 / 用 World 0 思考

World 0 is most useful when used as an operational chain:

World 0 最有价值的使用方式是一条操作链：

1. Submit an observation / 提交观察
2. Shape typed relations / 明确关系类型
3. Activate the relevant neighborhood / 激活相关概念邻域
4. Generate a local projection for the task / 为任务生成局部投影

It is not just “store some notes and query later”.

它不是“先存笔记，之后再检索”。

```python
from world0 import World, Observation

w = World(store_path=".world0")

# Agent submits observations from its work
# Agent 提交工作中的观察
w.ingest(Observation(
    concepts=["Python", "FastAPI", "REST API", "PostgreSQL"],
    relations=[
        ("FastAPI", "Python", "depends_on"),
        ("FastAPI", "REST API", "contains"),
        ("REST API", "PostgreSQL", "depends_on"),
    ],
    descriptions={"FastAPI": "Modern async web framework for Python"},
    task="design backend API",
    source="session_001",
))

# Agent requests a cognitive projection for a new task
# Agent 为新任务请求认知投影
proj = w.project(["FastAPI", "PostgreSQL"], task="optimize query performance")
print(proj.render())

# After task completion, consolidate
# 任务完成后，进行认知巩固
result = w.reflect()
```

### Example Workflow / 一个完整工作流

```python
from world0 import World, Observation

w = World(store_path=".world0")

# 1. Observe / 观察
w.ingest(Observation(
    concepts=["FastAPI", "SQLAlchemy", "PostgreSQL", "latency"],
    relations=[
        ("FastAPI", "SQLAlchemy", "depends_on"),
        ("SQLAlchemy", "PostgreSQL", "depends_on"),
        ("latency", "PostgreSQL", "related_to"),
    ],
    task="debug production API latency",
    source="incident_042",
))

# 2. Project / 投影
projection = w.project(
    ["FastAPI", "latency"],
    task="find the most relevant conceptual neighborhood for the incident",
)

print(projection.render())
```

The `render()` output is markdown, ready to inject into an Agent's system prompt:

`render()` 输出为 markdown 格式，可直接注入 Agent 的 system prompt：

```markdown
## Cognitive Context

### Core Understanding
- **FastAPI** (developing, confidence: 0.73): Modern async web framework for Python. Linked to: Python, REST API.
- **PostgreSQL** (developing, confidence: 0.68). Linked to: REST API, SQLAlchemy.

### Key Relations
- FastAPI → depends_on → Python (strength: 0.62, reinforced 8x)
- REST API → depends_on → PostgreSQL (strength: 0.55, reinforced 6x)

### Task Context
Concepts activated for: optimize query performance
```

## Core Concepts / 核心概念

| Concept / 概念 | Description / 说明 |
|---------|-------------|
| **Concept / 概念** | A semantic unit that can be linked, activated, and projected. Concepts have confidence, maturity, and activation history. / 可被链接、激活和投影的语义单元。概念具有置信度、成熟度和激活历史。 |
| **Relation / 关系** | A typed, weighted connection between concepts. Relations are discovered through observation and strengthened by repetition. / 概念之间有类型的、加权的连接。关系通过观察被发现，通过重复被强化。 |
| **Activation / 激活** | Spreading activation from seed concepts through the relation network, modulated by relation type, weight, and task affinity. / 从种子概念通过关系网络进行扩散激活，受关系类型、权重和任务亲和度调节。 |
| **Projection / 投影** | A task-relevant local view generated from the broader concept-world. This is the operational output. / 从更广泛的概念世界中生成的与任务相关的局部视图。这是系统的操作输出。 |
| **Reflect / 反思** | Cognitive consolidation: decay unused concepts, promote active ones, prune noise. / 认知巩固：衰减不用的概念，晋升活跃概念，修剪噪声。 |

## Agent Interface / Agent 接口

The `World` class exposes four methods. That's the entire API.

`World` 类暴露四个方法，这就是全部 API。

Read these methods as cognitive operations, not CRUD methods:

请把这些方法理解为认知操作，而不是 CRUD 方法：

- `ingest` = add observations into the concept-world / 将观察注入概念世界
- `project` = request a local task view / 请求任务级局部视图
- `reflect` = consolidate and prune / 巩固并修剪
- `ingest_text` = use an LLM to turn raw text into observations / 用 LLM 将原始文本转成观察

### `ingest(observation)` — Feed observations / 输入观察

```python
w.ingest(Observation(
    concepts=["Docker", "Kubernetes", "deployment"],
    relations=[("Kubernetes", "Docker", "depends_on")],
    task="container orchestration",
    source="session_003",
))
```

Concepts are created or reinforced. Relations are discovered or strengthened. Co-occurring concepts form Hebbian connections automatically.

概念被创建或强化。关系被发现或加强。共现的概念自动形成 Hebbian 连接。

### `ingest_text(text)` — LLM-powered extraction / LLM 驱动的提取

```python
from world0.llm import OpenAIProvider

w = World(store_path=".world0", llm=OpenAIProvider())
w.ingest_text(
    "We migrated the auth service from Express to FastAPI, "
    "using SQLAlchemy with PostgreSQL for the user store.",
    task="auth migration",
)
```

Requires an LLM provider (`OpenAIProvider` or `AnthropicProvider`). The LLM extracts concepts and relations from raw text, then feeds them into the standard ingest pipeline.

需要 LLM 提供者（`OpenAIProvider` 或 `AnthropicProvider`）。LLM 从原始文本中提取概念和关系，然后送入标准的 ingest 管线。

### `project(seeds, task=)` — Generate a projection / 生成投影

```python
proj = w.project(
    ["FastAPI", "PostgreSQL"],
    task="debug prod latency",
    max_concepts=10,
    max_depth=2,
)

# Inject into Agent prompt / 注入 Agent prompt
system_prompt = f"You are debugging a latency issue.\n\n{proj.render()}"

# Or inspect programmatically / 或以编程方式检查
for c in proj.top_concepts(5):
    print(c.name, c.maturity, c.confidence)
```

Projection uses spreading activation with task-affinity boosting and MMR (Maximal Marginal Relevance) selection for diversity.

投影使用扩散激活与任务亲和度加权，并通过 MMR（最大边际相关性）选择策略保证多样性。

### `reflect()` — Consolidate / 巩固

```python
result = w.reflect()
print(f"Promoted: {len(result.promoted_concepts)}")
print(f"Pruned:   {len(result.pruned_concepts)}")
```

Call after a task is complete. Decays unused concepts, promotes frequently activated ones through maturity stages, and prunes noise.

在任务完成后调用。衰减未使用的概念，将频繁激活的概念通过成熟度阶段晋升，修剪噪声。

## Concept Lifecycle / 概念生命周期

Concepts evolve through maturity stages based on activation frequency and confidence:

概念根据激活频率和置信度在成熟度阶段之间演化：

```
embryonic → developing → established → core
  (萌芽)     (发展中)      (已建立)     (核心)
                                         ↑
                              fading ─────┘ (revives on re-activation / 重新激活时复苏)
                              (衰退)
```

| Transition / 转换 | Requirements / 条件 |
|------------|-------------|
| embryonic → developing / 萌芽 → 发展中 | activation_count >= 3, confidence >= 0.3 |
| developing → established / 发展中 → 已建立 | activation_count >= 10, confidence >= 0.6 |
| established → core / 已建立 → 核心 | activation_count >= 30, connections >= 5 |
| any → fading / 任意 → 衰退 | confidence decays below 0.05 / 置信度衰减至 0.05 以下 |
| fading → developing / 衰退 → 发展中 | re-activated by an observation / 被观察重新激活 |

Decay rates are maturity-dependent: embryonic concepts fade in ~1 day, core concepts persist for ~3 months.

衰减速率取决于成熟度：萌芽概念约 1 天衰退，核心概念可持续约 3 个月。

## Relation Types / 关系类型

Relations are typed and influence activation propagation strength:

关系是有类型的，且影响激活传播强度：

| Type / 类型 | Propagation Factor / 传播系数 | Description / 说明 |
|------|-------------------|-------------|
| `depends_on` | 1.0 | Strong structural dependency / 强结构依赖 |
| `contains` | 0.95 | Part-whole containment / 整体-部分包含 |
| `part_of` | 0.95 | Inverse of contains / contains 的逆关系 |
| `activates` | 0.90 | Causal activation / 因果激活 |
| `supports` | 0.85 | Supportive association / 支持性关联 |
| `precedes` | 0.80 | Temporal/logical ordering / 时间/逻辑顺序 |
| `derived_from` | 0.80 | Origin relationship / 来源关系 |
| `similar_to` | 0.70 | Similarity / 相似 |
| `related_to` | 0.50 | Generic fallback / 通用回退 |
| `contrasts` | 0.40 | Opposition / contrast / 对立/对比 |

Hebbian relations (`related_to`, auto-discovered from co-occurrence) are capped at weight 0.7. Explicit relations declared by the Agent can reach 1.0.

Hebbian 关系（`related_to`，从共现中自动发现）权重上限为 0.7。Agent 显式声明的关系可达 1.0。

## Architecture / 架构

```
src/world0/
├── world/                # World facade + pipelines / 统一 Agent 接口与管线
│   ├── facade.py         # World — unified Agent interface / 统一接口
│   ├── _ingest.py        # ingest pipeline / 摄入管线
│   ├── _reflect.py       # reflect pipeline / 反思管线
│   └── _status.py        # world status / 世界状态
├── concepts/             # Concept lifecycle + identity / 概念生命周期与身份
│   ├── manager.py / api.py
│   ├── _consolidation.py # signature-based dedup / 基于签名的去重
│   └── _identity_ops.py  # merge / split / 合并与拆分
├── relations/manager.py  # Typed (3-axis) relation lifecycle / 三轴关系生命周期
├── dynamics/             # Cognitive dynamics / 认知动力学
│   ├── activation.py     # Spreading activation / 扩散激活
│   ├── decay.py · hebbian.py · lifecycle.py · coefficients.py
│   ├── color_diffusion.py# Community color field / 群落色场
│   └── community.py      # Community detection / 群落检测
├── communities/manager.py# Community persistence / 群落持久化
├── spaces/               # Isolated concept worlds / 隔离的概念世界
├── projection/engine.py  # MMR-based projection / 基于 MMR 的投影
├── extraction/extractor.py  # LLM-powered extraction / LLM 驱动提取
├── sources/library.py    # Raw source provenance / 原始来源溯源
├── metrics/entropy.py    # Network-entropy diagnostics / 网络熵诊断
├── models/config.py      # Per-operation model config / 分操作模型配置
├── prompts/              # Configurable prompt registry / 可配置 prompt 注册表
├── agents/               # Agent interfaces / Agent 接口
│   ├── pkm.py · cli.py · web.py · gui.py
│   ├── external.py       # Read-only claude/codex consultations / 只读外部咨询
│   ├── tools/ · mcp/ · research.py
├── llm/                  # base.py · openai.py · anthropic.py · azure_openai.py
├── schemas/              # concept · relation · types · context · community · space · source
├── store/                # base.py · json_store.py — pluggable persistence / 可插拔持久化
└── visualization/renderer.py  # Interactive HTML graph / 交互式 HTML 图
```

### Data Flow / 数据流

```
Observation ─→ ingest() ─→ concepts + relations + hebbian ─→ flush to disk
(观察)          (摄入)       (概念 + 关系 + Hebbian 学习)      (批量写盘)
                                          │
Seeds + Task ─→ project() ─→ activation ─→ MMR selection ─→ Projection
(种子 + 任务)    (投影)        (激活)         (MMR 选择)       (投影输出)
                                                                │
                                                         .render() → markdown
                                          │
              reflect() ─→ decay ─→ lifecycle ─→ prune ─→ flush to disk
              (反思)        (衰减)    (生命周期)    (修剪)    (批量写盘)
```

## Persistence / 持久化

World 0 persists to disk as individual JSON files:

World 0 以独立 JSON 文件的形式持久化到磁盘：

```
.world0/
├── concepts/
│   ├── a1b2c3d4e5f6.json
│   └── ...
├── relations/
│   ├── f6e5d4c3b2a1.json
│   └── ...
└── state.json
```

Writes use a dirty-flag mechanism: in-memory mutations are batched and flushed at `ingest()` and `reflect()` boundaries, not on every operation. The `Store` interface is abstract — swap `JsonStore` for a different backend without changing cognitive logic.

写入使用脏标记机制：内存中的变更被批量收集，在 `ingest()` 和 `reflect()` 边界处统一刷盘，而非每次操作都写入。`Store` 接口是抽象的——可以替换 `JsonStore` 为其他后端而不影响认知逻辑。

## Key Design Decisions / 关键设计决策

**Concept-first, not fact-first. / 概念优先，而非事实优先。** World 0 does not store facts. It stores concepts with confidence, maturity, and activation history. Facts live in the Agent's context; World 0 provides the conceptual scaffolding. / World 0 不存储事实。它存储带有置信度、成熟度和激活历史的概念。事实存在于 Agent 的上下文中；World 0 提供概念脚手架。

**Relations are first-class. / 关系是一等公民。** Not just `related_to` edges — relations are typed, weighted, reinforced, and decay independently. Relation type influences activation propagation strength. / 不只是 `related_to` 边——关系是有类型的、加权的、可强化的，且独立衰减。关系类型影响激活传播强度。

**Context changes relevance. / 上下文改变相关性。** The same concept-world produces different projections under different task contexts. Task affinity boosts concepts and relations associated with the current task by 1.5x. / 同一个概念世界在不同任务上下文下产生不同的投影。任务亲和度将与当前任务相关的概念和关系提升 1.5 倍。

**Projection is the output. / 投影是输出。** The system is only useful if it can turn a larger concept-world into a smaller, task-relevant view. Projection uses MMR selection to balance relevance against diversity. / 系统只有在能将更大的概念世界转化为更小的、与任务相关的视图时才有用。投影使用 MMR 选择来平衡相关性和多样性。

**Hebbian learning with threshold. / 带阈值的 Hebbian 学习。** Co-occurring concepts don't immediately form relations — they need to co-occur at least twice before a connection is created. This prevents noise from single observations. / 共现概念不会立即形成关系——需要至少共现两次才会创建连接。这防止了单次观察产生的噪声。

**Graceful decay. / 优雅衰减。** Unused concepts decay exponentially with maturity-dependent half-lives. Core concepts resist decay (3-month half-life); embryonic concepts fade in a day. This keeps the world clean without manual pruning. / 未使用的概念按指数衰减，半衰期取决于成熟度。核心概念抵抗衰减（3 个月半衰期）；萌芽概念在一天内衰退。这让概念世界保持整洁，无需手动修剪。

## Development / 开发

```bash
pip install -e ".[dev]"
pytest                        # 554 tests (546 passing, 8 skipped without an LLM key), ~15s / 554 个测试（546 通过，8 个在无 LLM key 时跳过），约 15 秒
pytest tests/test_benchmark.py -v   # cognitive quality benchmarks / 认知质量基准
pytest tests/test_benchmark_e2e.py -v -s   # end-to-end scenario / 端到端场景
ANTHROPIC_API_KEY=sk-... pytest tests/test_extraction_quality_llm.py -v   # real-LLM extraction quality / 真实 LLM 提取质量
```

### Test Coverage / 测试覆盖

| Suite / 测试套件 | Tests / 数量 | What it validates / 验证内容 |
|-------|-------|-------------------|
| `test_benchmark.py` | 43 | Activation precision, projection relevance, confidence dynamics, decay curves, Hebbian convergence, cross-domain separation, scale behavior, lifecycle thresholds, persistence fidelity, projection stability, task sensitivity, relation type differentiation, alias management / 激活精度、投影相关性、置信度动态、衰减曲线、Hebbian 收敛、跨域分离、规模行为、生命周期阈值、持久化保真、投影稳定性、任务敏感性、关系类型区分、别名管理 |
| `test_benchmark_e2e.py` | 24 | Multi-session Agent scenario: knowledge accumulation, cross-session coherence, projection focus, reflect consolidation, render quality, full lifecycle simulation, quantitative report / 多会话 Agent 场景：知识积累、跨会话一致性、投影聚焦、反思巩固、渲染质量、全生命周期模拟、量化报告 |
| `test_extraction_quality_llm.py` | 8 | Real-LLM extraction quality: synonym/acronym dedup, generic-noise filtering, relation direction, domain-sense split, contradiction handling, Chinese language preservation, cross-text identity (skipped without an LLM key) / 真实 LLM 提取质量：同义词/缩写去重、泛词噪声过滤、关系方向、领域义项拆分、矛盾处理、中文保持、跨文本身份（无 LLM key 时跳过） |
| Other tests / 其他测试 | ~479 | Unit/integration tests for concepts, relations, dynamics (incl. color-field & communities), spaces, sources, metrics, projection, extraction, agents (PKM/CLI/web/external), LLM providers, persistence / 概念、关系、动力学（含色场与群落）、空间、来源、指标、投影、提取、Agent（PKM/CLI/web/外部）、LLM 提供者、持久化的单元与集成测试 |

Total: 554 tests (546 passing, 8 skipped without an LLM provider). / 共 554 个测试（546 通过，8 个在无 LLM provider 时跳过）。

## Requirements / 依赖

- Python >= 3.10
- pydantic >= 2.0
- Optional / 可选: `openai >= 1.0` or `anthropic >= 0.30` for LLM-powered extraction / 用于 LLM 驱动的概念提取

## License / 许可证

MIT
