# World 0 统一 API 设计：一套可以长期不变的使用接口

> 状态：设计稿（0.3.0 之后，面向 0.4 → 1.0）。本文定义 World 0 对外的**唯一一套动词与数据形状**，以及它们在
> Python、CLI、HTTP、MCP 四个入口上的同构映射、稳定性分级与演化规则。它不新增认知机制；它把已经存在的能力收口成
> 一个可以承诺多年不变的表面。实现路线在 §9。

---

## 0. 一句话

**World 0 对外只有一条链：`ingest`（写）→ `project`（读）→ `reflect`（巩固），外加对概念身份与主张的有限整理（curate）
和只读的检视（inspect）；所有入口说同一套词、交换同一套数据形状，内部记录永不直接出现在接口上。**

---

## 1. 现状诊断（0.3.0）

0.3.0 的能力是完整的，但"对外表面"是三十轮迭代中长出来的，有五处会让接口无法长期稳定：

| # | 问题 | 现状 |
|---|---|---|
| 1 | **词汇漂移**：同一件事在不同入口叫不同名字 | Python `World.ingest` / `project`；PKM 与 CLI / Web 叫 `learn` / `ask` / `explore` / `connect`；Web 路由 `/api/learn`、`/api/ask` |
| 2 | **内部记录泄漏到接口**：`Projection.concepts` 是 `ConceptNode`（30 多个字段：`reinforcement_log`、`last_decayed_tick`…），`Projection.relations` 是 `RelationEdge`（`weight` / `probability` / `confidence` / `structural_strength` 四个互相纠缠的数） | 任何内部字段改名都是破坏性变更；调用方被迫理解动力学细节 |
| 3 | **写入形状不统一**：关系以 `(src, tgt, type)` 三元组、`RelationPrior`、`contradicted_relations`、`retracted_relations` 四种形态进入；结果里概念有的用名字、有的用 id | 调用方要同时记住四种关系输入与两种句柄 |
| 4 | **Web API 把认知层与 Agent 壳混在一起**：`/api/*` 下 45 条路由里，认知操作不到 10 条，其余是会话、prompt、skill、MCP 客户端、space 管理 | 认知层的接口承诺被 Agent 功能的频繁变化拖着一起变 |
| 5 | **没有稳定性分级与线格式版本**：没有说哪些是承诺、哪些是临时；JSON 输出没有版本号 | 既不能放心依赖，也不能放心演进 |

此外 MCP 目前只有**客户端**（World 0 调别人的工具），没有**服务端**（别的 Agent 调 World 0）；`TODO.md` P2 第 11 项已列。

## 2. 设计原则

1. **动词来自链条，不来自实现。** AGENTS.md 的链是 概念 → 概念卡 → 关系 → 语境 → 激活 → 投影；接口动词只有写入观察、生成投影、
   巩固、整理身份、检视五类。激活、Hebbian、衰减、结算都不是接口动词——它们是 `project` 与 `ingest` 的内部实现。
2. **观察是唯一的知识写入。** 任何改变"世界知道什么"的操作都表达为一个 `Observation`（陈述、撤回、否证）。
   只有身份操作（合并 / 拆分）不是观察，因为它们改变的是"谁是谁"而不是"什么成立"。
3. **投影是唯一的知识读取。** 概念卡与主张列表也是投影的特例（以一个概念为种子的零跳 / 一跳视图），而不是另一套读取路径。
4. **名字是 Agent 的句柄，id 是整理的句柄。** 所有输入接受名字（经别名解析）；所有输出同时给出名字与 id；只有身份操作要求 id 以避免歧义。
5. **接口数据形状（DTO）与内部记录分离。** 接口上只出现 `ConceptCard`、`Claim`、`Projection`、`Observation`、`IngestResult`、
   `ReflectResult`、`WorldStatus` 七种形状；`ConceptNode`、`RelationEdge` 是内部记录，可以自由演化。
6. **四个入口同构。** Python 方法名 = CLI 子命令 = HTTP 路径段 = MCP 工具名；参数名与返回形状一致；每个入口只是编码不同。
7. **认知层与 Agent 壳分离。** 核心 API 只覆盖 World 0 本身；会话、prompt、模型选择、skill、MCP 客户端、space 属于 Agent 壳，
   走独立的命名空间，各自演化，不在核心承诺之内。
8. **边界不进接口。** 没有"存一段原文再取回"的操作（World 0 不是记忆），没有"按时间列出事件"的操作（不是日志），
   没有任务编排（不是工作流）。`ingest_text` 只是"经抽取后写入"的便捷方法，原文不可取回。
9. **可演化**：稳定性分级 + 线格式版本 + 弃用周期（§7）。

## 3. 动词表

五类、十二个动词。每个动词在四个入口上同名。

| 类 | 动词 | 作用 | 幂等 | 写 |
|---|---|---|---|---|
| 写入 | `ingest` | 写入一条观察（陈述 / 撤回 / 否证 / 概念卡信息） | 否（`source_id` 相同的观察可被调用方去重） | ✓ |
| 写入 | `ingest_text` | 经 LLM 抽取后 `ingest`；原文不保存 | 否 | ✓ |
| 读取 | `project` | 从种子出发生成任务相关的局部视图 | ✓ | — |
| 读取 | `card` | 一个概念的概念卡（零跳投影） | ✓ | — |
| 读取 | `claims` | 一个概念的全部当前 / 已撤回 / 争议主张（一跳投影，不做选择） | ✓ | — |
| 读取 | `find` | 按名字 / 别名 / 相似度找概念 | ✓ | — |
| 读取 | `status` | 世界状态摘要 | ✓ | — |
| 巩固 | `reflect` | 社群、色场、物理删除；`light=True` 只做轻量一遍 | ✓（状态是流的函数，reflect 不改变读出的值） | ✓ |
| 整理 | `merge` | 两个概念是同一个 | 否 | ✓ |
| 整理 | `split` | 一个概念是两个 | 否 | ✓ |
| 整理 | `weaken` | 对一个概念本身表示怀疑 | 否 | ✓ |
| 生命周期 | `close` | 刷盘并释放资源 | ✓ | ✓ |

不进入动词表的东西，以及理由：
- `retract` / `contradict` / `state`：都是 `ingest` 的一种观察（原则 2）。可以提供语法糖（§4.2），但语法糖必须构造 `Observation` 再调 `ingest`，
  不得另开写入路径。
- `visualize`：输出 HTML，是工具不是认知操作；归入 Agent 壳或 `world0.tools`。
- `set_llm`：是构造参数的运行时变体，归入 Agent 壳 / 配置。
- `learn` / `ask` / `explore` / `connect`：PKM 的自然语言包装。保留为 PKM 的方法名，但文档与 CLI 帮助里注明它们等于
  `ingest_text` / `project(...).render()` / `card` + `claims` / `ingest(Observation(relations=[…]))`；新的入口不再引入这组别名。

## 4. 数据形状（DTO）

接口上只出现下面七种 pydantic 模型。它们住在 `world0.api`（新模块），并从 `world0` 顶层导出。内部记录 `ConceptNode`、
`RelationEdge` 不从顶层导出，各带 `to_card()` / `to_claim()`。

### 4.1 `Observation`（写入，已存在，收口）

```python
class Observation(BaseModel):
    # 什么被提到了
    concepts: list[str]                       # 名字；经别名解析
    cards: list[ConceptCardInput]             # 可选：名字之外的卡片信息（sense、description、aliases、kind、domain）
    # 什么成立 / 不再成立 / 不成立
    statements: list[Statement]               # (source, relation, target[, belief, rationale])
    withdrawals: list[Statement]              # 曾成立、现在不再成立 → 退出视图
    denials: list[Statement]                  # 对某条主张的否认 → 只降低其信念；无对象的否认什么也不改变
    weakened: list[str]                       # 对概念本身的怀疑
    # 语境与来源
    task: str                                 # 主张的语境就是陈述它的任务
    source: str                               # 来源描述
    source_id: str                            # 调用方的幂等键
    timestamp: datetime
    extraction_metadata: dict                 # 抽取器的附注，World 0 不解释
```

与 0.3.0 字段的对应：`relations` + `relation_priors` → `statements`（一个 `Statement` 同时承载标签与可选先验，消除两种形态）；
`retracted_relations` → `withdrawals`；`contradicted_relations` → `denials`；`concept_candidates` + `descriptions` → `cards`。
0.3.0 的字段名在 0.4 继续接受（构造时转换并发出 `DeprecationWarning`），1.0 移除。

```python
class Statement(BaseModel):
    source: str                # 名字
    relation: str              # 标签：任何已知别名（depends_on / part_of / contrasts …）；按"source relation target"读
    target: str
    belief: float | None = None     # 可选先验（抽取器的概率）
    rationale: str = ""
```

方向约定不变：`Statement("wheel", "part_of", "car")` 存为 "car contains wheel"（§6 的别名表是接口承诺的一部分）。

### 4.2 语法糖（可选，构造 Observation）

```python
world.state("api", "depends_on", "db", task="backend")        # == ingest(Observation(statements=[…], task=…))
world.withdraw("api", "depends_on", "db", task="backend")     # == ingest(Observation(withdrawals=[…]))
world.deny("api", "conflict", "cache", task="backend")        # == ingest(Observation(denials=[…]))
```

每个都返回 `IngestResult`；它们只是 `ingest` 的拼写，不是新的写入路径。

### 4.3 `ConceptCard`（读取）

概念卡是概念的**接口形态**：Agent 能读、能编辑、能据以建立关系的那部分，不含动力学内部量。

```python
class ConceptCard(BaseModel):
    id: str
    name: str
    aliases: list[str]
    sense: str                 # 同名概念靠它区分
    kind: str
    domain: str
    description: str
    maturity: str              # embryonic | developing | established | core | fading
    evidence: float            # 0–1：被陈述 / 确认的充分程度（导出量）
    confidence: float          # 0–1：当前结算后的置信度
    tasks: list[str]           # 在哪些任务下被激活过（按频次降序）
    last_seen_tick: int        # 认知时间：最后一次被提及是第几个观察
    sources: list[str]         # 来源描述（去重）
```

`ConceptNode.to_card()` 产生它；`World.card(name)` 返回它或 `None`。

### 4.4 `Claim`（读取）

```python
class Claim(BaseModel):
    source: str; source_id: str
    relation: str              # 规范语义关系名（dependence / inclusion / contrast …）
    target: str; target_id: str
    axis: str                  # positive | negative | parallel
    belief: float              # 0–1
    support: int               # 被陈述的次数（共现边为 0）
    status: str                # current | withdrawn | contested | outvoted | co_occurrence
    stated_under: list[str]    # 陈述它的任务
    text: str                  # 一句话："api depends on db"
```

`RelationEdge.to_claim()` 产生它。`status` 把 0.3.0 渲染里的四个分节（当前 / No longer holds / Hold loosely / Seen under other tasks）
变成了结构化字段，调用方不必解析文本。

### 4.5 `Projection`（读取，已存在，收口）

```python
class Projection(BaseModel):
    api: str = "world0/1"            # 线格式版本
    seeds: list[str]
    task: str
    perspective: str
    concepts: list[ConceptCard]      # 视图里的概念（种子永远在）
    claims: list[Claim]              # 当前主张（status == current）
    no_longer_holds: list[Claim]
    other_tasks: list[Claim]         # 在别的任务下陈述的
    hold_loosely: list[Claim]        # 争议 / 被压过 / 证据薄
    activation: dict[str, float]     # id → 激活
    why: dict[str, str]              # id → 一句话（"reached via db / named in task / seed"）
    def render(self, style: str = "compact") -> str
```

与 0.3.0 的对应：`concepts: list[ConceptNode]` → `list[ConceptCard]`；`relations` / `retracted` / `other_contexts` → 按 `status` 分组的
`Claim` 列表；`epistemic` 折叠进 `Claim.status` 与 `ConceptCard.evidence`；`attention: dict[str, AttentionTrace]` 折叠为 `why`。
`render()` 的输出文本格式**不是**接口承诺（它会随可读性研究改），结构化字段才是。

### 4.6 `IngestResult`、`ReflectResult`、`WorldStatus`

保持 0.3.0 的字段，加上 `api` 版本字段；`IngestResult` 里的列表元素统一为 `{"id","name"}`（现在是名字字符串）
与 `"A → relation → B"` 文本两者并列（`claims_changed: list[Claim]`）。

## 5. 四个入口的同构映射

| 动词 | Python | CLI (`world0`) | HTTP | MCP 工具 |
|---|---|---|---|---|
| ingest | `world.ingest(obs)` | `world0 ingest --json obs.json` | `POST /v1/ingest` | `world0.ingest` |
| ingest_text | `world.ingest_text(text, task=…)` | `world0 ingest-text "…" --task …` | `POST /v1/ingest-text` | `world0.ingest_text` |
| project | `world.project(seeds, task=…)` | `world0 project api db --task backend` | `POST /v1/project` | `world0.project` |
| card | `world.card("api")` | `world0 card api` | `GET /v1/card/{name}` | `world0.card` |
| claims | `world.claims("api", task=…)` | `world0 claims api` | `GET /v1/claims/{name}` | `world0.claims` |
| find | `world.find("auth")` | `world0 find auth` | `GET /v1/find?q=` | `world0.find` |
| status | `world.status()` | `world0 status` | `GET /v1/status` | `world0.status` |
| reflect | `world.reflect(light=…)` | `world0 reflect [--light]` | `POST /v1/reflect` | `world0.reflect` |
| merge / split / weaken | `world.merge(keeper_id, absorbed_id)` … | `world0 merge <id> <id>` … | `POST /v1/merge` … | `world0.merge` … |

规则：
- **HTTP 在 `/v1/` 下**，请求与响应体就是 §4 的 DTO 的 JSON；错误体 `{"api":"world0/1","error":{"code","message"}}`，
  code 取 `not_found | invalid_observation | unknown_relation | identity_conflict | llm_unavailable`。
- **CLI 的 `--json`** 输出与 HTTP 响应体字节相同；默认输出是 `render()` 或人读表格。入口名改为 `world0`；`pkm` 保留为 Agent 壳的入口。
- **MCP 服务端**（`world0.agents.mcp.server`）把同一组动词暴露为工具，`inputSchema` 由 DTO 的 JSON Schema 生成，
  `project` 的结果以 `render()` 文本 + 结构化 JSON 两种 content 返回。
- **Agent 壳**的东西（会话、prompt、模型、skill、MCP 客户端、space、研究、web 搜索）在 `/agent/*`、`pkm …`、`pkm.*` 工具下，
  不在 `/v1/`，不受本文承诺约束。

## 6. 接口承诺的"语义常量"

下面这些不是实现细节，是接口的一部分；改动它们等于改大版本。

- **关系标签与别名**：`depends_on` / `derived_from` / `precedes` → `dependence`；`supports` / `activates` → `enables`；
  `contains` / `part_of` → `inclusion`（`part_of` 从整体读）；`contrasts` / `negative` / `repulsion` → `contrast`；`conflict` 就是冲突；
  `similar_to` → `similarity_kernel`；`related_to` → `generic_relation`；`precedes`、`part_of` 的方向反转。
- **主张身份** = (source, target, axis, label)。同一对上的另一个标签是另一条主张。
- **撤回 vs 否认**：撤回使主张退出视图；否认只降低信念；对不存在的主张的否认什么也不改变。
- **语境**：主张的语境是陈述它的任务；`project(task=…)` 把别的任务的主张放到 `other_tasks`；没有任务时不拆分。
- **种子永远在视图里**；`max_concepts` 是上限不是目标。
- **状态是观察流的函数**：任何读取不因 `reflect` 的时机而不同。
- **存储向后兼容**：旧版本写的存储在加载时迁移，不要求离线转换。

## 7. 稳定性分级与演化规则

三级，写进每个公开符号的 docstring 首行：

| 级 | 含义 | 范围 |
|---|---|---|
| **Stable** | 不做破坏性改动；移除要先弃用一个次版本（0.x 阶段按次版本计），并在 CHANGELOG `Migration` 节说明 | §3 动词、§4 DTO、§5 映射、§6 语义常量、`World(store_path, backend=)`、存储格式的可加载性 |
| **Provisional** | 可能改，改动在 CHANGELOG 列出，不保证弃用期 | `render()` 文本格式、`Perspective` 配置项、`WorldStatus` 的统计字段、`IngestResult.prediction`、`ReflectResult` 的社群 / 色场字段、`sustained_attention` |
| **Internal** | 随时改 | `ConceptNode`、`RelationEdge`、`dynamics/*`、`activation/*`、`projection/engine.py`、系数常量、`world0.agents.*` 的实现 |

线格式版本 `api: "world0/1"`：DTO 新增可选字段不升版本；字段改名 / 删除 / 语义改变升到 `world0/2`，服务端同时接受前一版至少一个次版本。

弃用方式：旧字段名在构造时转换 + `DeprecationWarning`（Python）；HTTP 旧路径 301 到新路径一个次版本；CLI 旧子命令打印提示并转发。

## 8. 一个完整的调用序列（任何入口都一样）

```python
from world0 import World, Observation, Statement

w = World(".world0")
w.ingest(Observation(
    concepts=["api", "db", "cache"],
    statements=[Statement("api", "depends_on", "db"), Statement("api", "conflict", "cache")],
    task="backend", source="design review", source_id="review-2026-10-01#3",
))
w.ingest(Observation(withdrawals=[Statement("api", "conflict", "cache")], task="backend"))

view = w.project(["api"], task="backend")
view.claims            # [Claim(api depends on db, belief 0.70, support 1, current)]
view.no_longer_holds   # [Claim(api conflicts with cache, withdrawn)]
view.render()          # 直接放进提示词的文本

w.card("db")           # ConceptCard(name="db", maturity="embryonic", evidence=…, tasks=["backend"])
w.claims("api")        # 全部主张，含 withdrawn
w.reflect()
w.close()
```

同一序列的 HTTP 形态是 `POST /v1/ingest` ×2、`POST /v1/project`、`GET /v1/card/db`、`GET /v1/claims/api`、`POST /v1/reflect`，请求体就是上面的
对象的 JSON；MCP 形态是同名工具。

## 9. 实现路线（不破坏 0.3.0）

| 版本 | 工作 | 兼容 |
|---|---|---|
| **0.4** | 新模块 `world0.api`：`Statement`、`ConceptCard`、`Claim`；`ConceptNode.to_card()`、`RelationEdge.to_claim()`；`World.card()` / `claims()` / `find()`（`find_similar` 转发）；`Observation` 接受 `statements` / `withdrawals` / `denials` / `cards`，旧字段名转换 + 警告；`Projection` 增加 `cards` / `claims` / `no_longer_holds` / `other_tasks` / `hold_loosely` / `why` 字段，旧字段保留；`api` 版本字段；稳定性分级写进 docstring | 完全兼容 |
| **0.5** | CLI 入口 `world0`（§5 子命令，`--json`）；HTTP `/v1/*`（独立于 `/api/*` 的 Agent 壳路由）；MCP 服务端 `world0.agents.mcp.server`（TODO P2-11）；`pkm` 的 `learn/ask/explore/connect` 文档化为别名 | 旧 `/api/*`、`pkm` 继续工作 |
| **0.6** | 移除 `Projection` 的 `ConceptNode` / `RelationEdge` 直出字段与 `Observation` 旧字段名（按 §7 弃用期）；`world0` 顶层不再导出内部记录 | 破坏性，在 CHANGELOG `Migration` 节说明 |
| **1.0** | 冻结 §3–§6；`api: "world0/1"` 成为长期承诺 | — |

每一步的验收：`tests/test_api_contract.py` 对四个入口做同一组黄金用例（§8 的序列），断言结构化输出逐字段相同；
`tests/test_layer_boundaries.py` 增加"`world0.api` 不导入 `agents`、`dynamics`、`projection.engine`"的 AST 检查。

## 10. 不做的事

- 不提供"取回原文 / 事件日志 / 按时间检索"的接口（记忆与日志是相邻系统）。
- 不提供"运行一个流程 / 定时任务"的接口（工作流）。
- 不把激活、衰减、Hebbian、结算暴露为动词或可调参数——它们是 `project` 与 `ingest` 的实现；需要研究它们的人用 `world0.dynamics`（Internal）。
- 不为向量检索开接口；`find` 用名字 / 别名 / 签名相似度。
- 不在核心 API 里放会话、prompt、模型选择：这些属于 Agent 壳。
