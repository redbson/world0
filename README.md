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

- [`docs/world0-core-concepts.md`](docs/world0-core-concepts.md) — authoritative design doc for the core concepts (Concept/Relation/Context/Activation/Projection/Perspective + dynamics, with exact formulas & coefficients) / 核心概念权威设计文档（含与代码一致的公式与系数）
- [`docs/world0-paper.md`](docs/world0-paper.md) — the World 0 paper: system model, dynamics, evaluation, related work / World 0 论文：系统模型、动力学、评测与相关工作
- [`docs/world0-usage.md`](docs/world0-usage.md) — operational usage guide for World 0 / World 0 操作与使用文档
- [`docs/world0-color-field-dynamics.md`](docs/world0-color-field-dynamics.md) — dynamics-first design for community-born color fields / 基于动力学的群落生色与褪色设计
- [`docs/extraction-model-prompt-eval.md`](docs/extraction-model-prompt-eval.md) — model × prompt extraction-quality evaluation (why gpt-5.4-nano is the default) / 模型×prompt 提取质量评测（为何默认 gpt-5.4-nano）
- [`docs/projection-eval-baseline.md`](docs/projection-eval-baseline.md) — offline projection-quality metrics (P@k / NDCG / noise / inhibition) / 离线投影质量指标
- [`docs/projection-sweep-baseline.md`](docs/projection-sweep-baseline.md) — coefficient sensitivity report for activation/projection tuning / 激活/投影系数敏感性报告
- [`DesignPhilosophy.md`](DesignPhilosophy.md) — design rationale and framing / 设计哲学与边界
- [`TODO.md`](TODO.md) — current implementation priorities / 当前实现优先级

> **Repository split / 仓库拆分.** World 0 is the **cognitive core library**. The PKM / Agent application layer — terminal CLI, browser UI, native GUI, MCP server, research, and the autonomous digital individual — now lives in its own project, [**`world0-pkm`**](../world0-pkm), which depends on this package. / World 0 是**认知核心库**。PKM / Agent 应用层（CLI、浏览器界面、原生 GUI、MCP、研究、自主数字个体）已迁出为独立项目 [**`world0-pkm`**](../world0-pkm)，它依赖本包。

## Quickstart / 快速开始

World 0 is a Python library. Install it (optionally with an LLM provider for `ingest_text`):

World 0 是一个 Python 库。安装它（如需 `ingest_text` 文本提取再加 LLM provider 扩展）：

```bash
pip install -e .                 # core only / 仅核心
pip install -e ".[openai]"       # + OpenAI for LLM extraction
pip install -e ".[anthropic]"    # + Anthropic
pip install -e ".[dev]"          # + test tooling / 测试工具
```

```python
from world0 import World, Observation

w = World(store_path=".world0")
w.ingest(Observation(
    concepts=["FastAPI", "PostgreSQL"],
    relations=[("FastAPI", "PostgreSQL", "depends_on")],
    task="design backend API",
))
print(w.project(["FastAPI"], task="optimize query performance").render())
```

For interactive interfaces (CLI / web / GUI) and agentic / autonomous modes, install the
[`world0-pkm`](../world0-pkm) project, which provides `pkm`, `pkm-web`, and `pkm-gui`.

如需交互界面（CLI / web / GUI）与 agentic / 自主模式，请安装 [`world0-pkm`](../world0-pkm) 项目，
它提供 `pkm`、`pkm-web`、`pkm-gui` 入口。

For a task-oriented walkthrough of the Python API, persistence layout, and recommended
workflows, see [`docs/world0-usage.md`](docs/world0-usage.md).

面向任务的完整使用说明（Python API、持久化目录、推荐工作流）见
[`docs/world0-usage.md`](docs/world0-usage.md)。

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

### Perspectives — role-conditioned lenses / 视角：角色化透镜

The **same** concept-world yields different projections under different perspectives. A perspective re-weights how individual *semantic relations* (not just the three axes) carry activation, biases in-focus domains, and chooses a render style — so the world answers differently when debugged, designed against, or researched.

**同一个**概念世界在不同视角下产出不同投影。视角重新加权单个*语义关系*（不止三轴）的激活传播、偏置在焦域，并选择渲染风格——于是世界在“调试 / 设计 / 研究”时给出不同回答。

```python
# Built-in profiles: default / debug / design / research
proj = w.project(["model serving"], perspective="debug")     # by name
print(w.perspectives.names())

# Or an inline / custom perspective
from world0 import Perspective
lens = Perspective(
    name="oncall",
    active_domains=["infra"],
    semantic_relation_weights={"dependence": 1.6, "overlap": 0.4},  # per-relation
    render_style="compact",
)
proj = w.project(["model serving"], perspective=lens)
w.perspectives.put(lens)        # persist a custom profile to the store
```

### Projection explainability / 投影可解释性

Every projected concept carries its **best activation path** — why it was reached from the seeds — and the projection renders in three styles.

每个投影概念都携带其**最佳激活路径**——它为何从种子被激活到——投影支持三种渲染风格。

```python
proj = w.project(["model serving"], max_depth=3)
print(proj.explain("kv cache"))
# kv cache ←(dependence, positive, ×0.42)— request batching ←(...)— model serving [seed] (score 0.31)

print(proj.render("compact"))    # dense, for tight token budgets
print(proj.render("detailed"))   # adds "Why included" traces + counter-signals
print(proj.render())             # "default" — unchanged maturity-grouped view

print(proj.seed_resolution)      # {"model servng": "model serving (fuzzy:0.67)"}  ← typo recovered
```

Seeds resolve robustly: exact name/alias → domain disambiguation → fuzzy token match, recorded transparently in `seed_resolution` so a guessed or missed seed is never silent.

种子稳健解析：精确名/别名 → 领域消歧 → 模糊 token 匹配，并透明记录在 `seed_resolution` 中，从而被猜测或未命中的种子绝不静默。

### `apply_feedback(...)` — Close the loop / 闭环反馈

Tell World 0 what helped and what misled: useful concepts get reinforced, missing ones created, noisy ones demoted, weak relations weakened — all through the public facade (the same API the autonomous individual uses).

告诉 World 0 什么有用、什么误导：有用概念被强化、缺失概念被创建、噪声概念被降权、弱关系被削弱——全部经由公开接口（与自主个体所用相同的 API）。

```python
w.apply_feedback(
    useful_concepts=["inference engine"],
    missing_concepts=["speculative decoding"],   # created + reinforced
    noisy_concepts=["serving flags"],            # demoted
    weak_relations=["model serving -> dependence -> serving env"],
    task="serving optimization",
)
```

### Evaluation harnesses / 评测工具

Two LLM-free, deterministic harnesses keep projection quality honest:

两个无需 LLM、确定性的工具用于持续校准投影质量：

```bash
# Offline retrieval metrics (P@k / NDCG / noise / typed-ratio / inhibition)
python scripts/eval_projection_matrix.py --md docs/projection-eval-baseline.md

# Coefficient sensitivity report (a tuning aid, not an auto-tuner)
python scripts/sweep_projection_quality.py \
    --param projection.mmr_lambda=0.0,0.3,0.6 \
    --param activation.propagation_min_ratio=0.03,0.1,0.3
```

Baselines live in [`docs/projection-eval-baseline.md`](docs/projection-eval-baseline.md) and [`docs/projection-sweep-baseline.md`](docs/projection-sweep-baseline.md).

### `reflect()` — Consolidate / 巩固

```python
result = w.reflect()
print(f"Promoted: {len(result.promoted_concepts)}")
print(f"Pruned:   {len(result.pruned_concepts)}")
```

Call after a task is complete. Decays unused concepts, promotes frequently activated ones through maturity stages, and prunes noise.

在任务完成后调用。衰减未使用的概念，将频繁激活的概念通过成熟度阶段晋升，修剪噪声。

## Digital Individual / 数字个体

The PKM layer can run as an autonomous **digital individual** that tends and grows its own concept-world — with a *self* (persona seed → emergent self), a *will* (four intrinsic drives), and a *life* (`perceive → deliberate → act → reflect`). This lives entirely in the agent layer and acts only through the public `World` facade, so the cognitive core is never modified and "World 0 is not a memory system" stays true.

PKM 层可作为一个自主的**数字个体**运行，照料并生长自己的概念世界——拥有*自我*（人格种子→涌现自我）、*意志*（四种内在驱力）与*生命*（`感知 → 抉择 → 行动 → 反思`）。它完全位于 agent 层，只通过公开的 `World` 接口行动，因此认知核心从不被修改。

The digital individual now lives in the [**`world0-pkm`**](../world0-pkm) project (`pkm/autonomy/`). See its README for usage. / 数字个体现位于 [**`world0-pkm`**](../world0-pkm) 项目（`pkm/autonomy/`），用法见其 README。

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

## Model Selection / 模型选择

The **recommended default extraction model is `gpt-5.4-nano`** (via Azure OpenAI). A real `model × prompt` evaluation across 9 models (3 runs each) found that on the production extraction prompt, structural quality is largely model-agnostic — `gpt-5.4-nano` reaches the top tier (synonym dedup, sense disambiguation, typed relations, contradiction handling, Chinese preservation all stable) at the lowest cost and latency. Full evidence: [`docs/extraction-model-prompt-eval.md`](docs/extraction-model-prompt-eval.md).

**推荐的默认提取模型是 `gpt-5.4-nano`**（经 Azure OpenAI）。一项覆盖 9 个模型、各 3 轮的真实 `模型×prompt` 评测发现：在生产提取 prompt 下，结构质量基本与模型无关，`gpt-5.4-nano` 以最低成本/延迟达到第一梯队。完整证据见 [`docs/extraction-model-prompt-eval.md`](docs/extraction-model-prompt-eval.md)。

When you construct a `World` with an LLM provider, pass the model you want directly. Good alternatives for extraction: `azure-openai/DeepSeek-V4-Pro` (strong Chinese, low cost) and `glm-5.1`. Avoid `claude-sonnet-4-6` for bilingual corpora — it does not reliably preserve Chinese concept names.

构造带 LLM provider 的 `World` 时直接传入所需模型。提取的其他优选：`azure-openai/DeepSeek-V4-Pro`（中文强、便宜）、`glm-5.1`。双语语料避免用 `claude-sonnet-4-6`——它不能稳定保留中文概念名。

> Per-operation model routing (`extraction` / `answer` / `query_extract`, the `pkm model …` CLI, and `models.json`) is part of the agent layer in [`world0-pkm`](../world0-pkm). / 分操作模型路由（`pkm model …` CLI 与 `models.json`）属于 [`world0-pkm`](../world0-pkm) 的 agent 层。

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
├── prompts/              # Configurable prompt registry / 可配置 prompt 注册表
├── llm/                  # base.py · openai.py · anthropic.py · azure_openai.py
├── schemas/              # concept · relation · types · context · community · space · source
├── store/                # base.py · json_store.py — pluggable persistence / 可插拔持久化
└── visualization/renderer.py  # Interactive HTML graph / 交互式 HTML 图
```

The PKM / Agent application layer (`pkm` package: CLI, web, GUI, MCP, tools, research, external consultations, `models/` model-routing, and `autonomy/` the digital individual) lives in the separate [`world0-pkm`](../world0-pkm) project, which depends on this package.

PKM / Agent 应用层（`pkm` 包：CLI、web、GUI、MCP、工具、研究、外部咨询、`models/` 模型路由、`autonomy/` 数字个体）位于独立的 [`world0-pkm`](../world0-pkm) 项目，它依赖本包。

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
pytest                        # 811 tests (803 passing, 8 skipped without an LLM key), ~8s / 811 个测试（803 通过，8 个在无 LLM key 时跳过），约 8 秒
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
| Other tests / 其他测试 | ~736 | Unit/integration tests for concepts, relations, dynamics (incl. color-field & communities), perspectives, projection (incl. explainability & render styles), seed resolution, coefficient config, context/task-affinity, usage feedback, spaces, sources, metrics, extraction, LLM providers, persistence / 概念、关系、动力学（含色场与群落）、视角、投影（含可解释性与渲染风格）、种子解析、系数配置、上下文/任务亲和、使用反馈、空间、来源、指标、提取、LLM 提供者、持久化的单元与集成测试 |

Total: 811 tests (803 passing, 8 skipped without an LLM provider). The PKM / Agent layer tests live in [`world0-pkm`](../world0-pkm). / 共 811 个测试（803 通过，8 个在无 LLM provider 时跳过）。PKM / Agent 层测试位于 [`world0-pkm`](../world0-pkm)。

## Requirements / 依赖

- Python >= 3.10
- pydantic >= 2.0
- Optional / 可选: `openai >= 1.0` or `anthropic >= 0.30` for LLM-powered extraction / 用于 LLM 驱动的概念提取

## License / 许可证

MIT
