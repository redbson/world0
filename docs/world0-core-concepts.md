# World 0 核心概念设计文档

> **文档定位**：本文是 World 0 核心认知系统（`World` facade 及其下层）的权威设计文档。
> 所有公式、系数与阈值均与当前代码实现一一对应（标注了来源模块）。
> 阅读顺序建议：先读第 1–2 节建立总览，再按需深入各一等对象的章节。
>
> 相关文档：[`world0-architecture.md`](world0-architecture.md)（组件架构图）·
> [`world0-paper.md`](world0-paper.md)（论文）·
> [`world0-usage.md`](world0-usage.md)（使用指南）·
> [`../DesignPhilosophy.md`](../DesignPhilosophy.md)（设计哲学）

---

## 1. 系统定位与边界

World 0 是面向 LLM Agent 的**概念世界认知层**（cognitive concept-world layer）。它回答的问题是：

- 当前任务中哪些概念正在发挥作用？
- 这些概念之间的关键关系是什么？
- 应该向 Agent 注入怎样的**局部**概念视图？

它**不是**：

| 不是 | 区别 |
|------|------|
| 记忆系统 | 记忆回答"发生了什么"；World 0 回答"应如何理解"。情节性记忆（journal）属于 agent 层 |
| 知识图谱 | 知识图谱以实体与事实完备性为目标；World 0 以任务条件下的局部投影可用性为目标 |
| 向量检索 | 向量检索召回相似片段但缺乏显式结构；World 0 的输出是带类型关系的可解释概念视图 |
| 形式本体 | 本体追求逻辑推理完备；World 0 追求 LLM 可读写性与认知操作价值 |

核心立场（见 `DesignPhilosophy.md`）：**概念优先、关系优先、上下文敏感、为理解服务、LLM 原生、投影导向**。

---

## 2. 一等概念总览

World 0 建立在七个一等对象之上，它们构成一条**操作链**而非 CRUD 存储：

| 一等对象 | 实现载体 | 一句话定义 |
|---------|---------|-----------|
| **Concept** | `schemas/concept.py` `ConceptNode` | 可被链接、激活、强化、衰减、投影的最小语义单元 |
| **Concept Card** | `Observation` + `ConceptNode` 的描述/来源字段 | 概念的原始可编辑记录形态：名称、描述、别名、来源摘录 |
| **Relation** | `schemas/relation.py` `RelationEdge` | 概念间有类型、有方向、有权重、可演化的一等连接 |
| **Context** | `schemas/context.py` `Context` | 决定"此刻什么更重要"的条件变量：任务 + 视角 + 时钟 |
| **Activation** | `dynamics/activation.py` | 任务感知的双通道扩散激活：从种子选出局部认知区域 |
| **Projection** | `projection/engine.py` | 系统的操作输出：任务相关、低冗余、可解释的局部概念视图 |
| **Perspective** | `perspectives/` + `Context.perspective` | 角色化透镜：同一世界在不同视角下产出不同投影 |

操作链：

```
Observation ──ingest──▶ concepts + relations + Hebbian + 色场
                              │
Seeds + Task + Perspective ──project──▶ 种子解析 → 扩散激活 → MMR 选择 → Projection.render()
                              │
                        ──reflect──▶ 衰减 → 群落检测 → 色场再播种 → 生命周期 → 修剪
                              │
                        ──apply_feedback──▶ 强化有用 / 降权噪声 / 削弱弱关系（闭环）
```

---

## 3. Concept（概念）

### 3.1 设计意图

概念不是词项，也不是事实容器，而是一个**带认知状态的语义单元**：它有置信度（confidence）、成熟度（maturity）、激活历史与来源溯源。概念的"重要性"从不全局固定——它由上下文在激活与投影时动态决定。

### 3.2 数据模型（`ConceptNode`）

| 分组 | 字段 | 类型 / 默认值 | 语义 |
|------|------|--------------|------|
| 身份 | `id` | `uuid4().hex[:12]` | 持久唯一标识 |
| | `name` / `aliases` | str / `[]` | 主标签与别名集合 |
| | `identity_key` | `""` | 语义身份键（由 kind/sense/domain 构建） |
| | `kind` / `sense` / `domain` | `""` | 类别 / 义项消歧标记 / 领域标签 |
| | `domain_profile` | `dict[str, float]` | 领域→强度映射（**色场**，见 §9.3） |
| 描述 | `description` / `tags` | `""` / `[]` | Concept Card 的可编辑内容 |
| 认知状态 | `confidence` | `0.15` | 置信度 ∈ [0.01, 1.0]，初始萌芽值 |
| | `maturity` | `EMBRYONIC` | 五阶段生命周期（见 §3.5） |
| | `activation_count` / `disconfirmation_count` | 0 / 0 | 强化 / 证伪计数 |
| | `last_activated` / `last_weakened` | now / None | 时间锚点（驱动衰减与新鲜度） |
| 溯源 | `origin` / `reinforcement_log` | | 初始来源 / 激活事件日志（含 task 标签，驱动任务亲和） |
| | `source_refs` / `token_refs` | | 原始素材指针 / 表面形式观察 |

### 3.3 语义身份与归并

概念的同一性不靠字符串相等，而靠一套**级联归并机制**（`concepts/_manager.py`、`_consolidation.py`）：

1. **语义身份精确查找**：若 kind/sense 非空，按 `identity_key` 在身份索引中精确匹配；
2. **同义词匹配**：复合评分 ≥ **0.78** 才归并（词面重叠 + sense/描述 Jaccard；或共享 token ≥ 4 且 Jaccard ≥ 0.82）；
3. **名称/别名精确解析**；
4. **签名自动合并**：以 `name|description` 的签名 token 集（token 长度 ≥ 2、过滤停用词）计算 Jaccard 相似度，**≥ 0.6** 自动归并；
5. 全部未命中 → 新建 EMBRYONIC 概念（confidence = 0.15）。

两个保护机制：

- **跨域惩罚**：两概念同时声明了不同 `domain` 时，签名相似度 × **0.3**——同名不同域的概念（如 `Apple/公司` vs `Apple/水果`）不会被错误归并；
- **语义边界检查**：kind 冲突（忽略 core/supporting/background 这类角色性 kind）直接拒绝归并。

模糊查询（投影种子解析用）使用非对称包含度（containment）≥ **0.45**。

支持显式身份操作：`merge(keeper, absorbed)`（别名与关系重定向）与 `split(source, new_name)`（义项拆分）。

### 3.4 置信度动力学

置信度更新采用**递减收益**曲线——越成熟的概念越难被单次事件改变：

```
强化  activate():  Δ⁺ = 0.06 / (1 + 0.08·activation_count)，上限 1.0
                   若 maturity == FADING → 复活为 DEVELOPING
证伪  weaken():    Δ⁻ = 0.06 / (1 + 0.08·disconfirmation_count)，下限 0.01
```

下限 0.01 而非 0 是有意设计：被证伪的概念保留极弱的痕迹，等待修剪而不是即刻消失，从而可被审计。

### 3.5 生命周期（maturity 状态机）

```
embryonic ──act≥3, conf≥0.3──▶ developing ──act≥10, conf≥0.6──▶ established ──act≥30, conn≥N*──▶ core
     │                              │                                │
     └──────── conf < 0.05 ────────┴────────────────────────────────┘──▶ fading ──重新激活──▶ developing
```

- **N\*（动态连接阈值）**：`max(2, 5 − (activation_count − 30) // 20)`——激活极频繁的概念可以用较少连接证明其中枢地位（`dynamics/lifecycle.py`）；
- **衰减半衰期按成熟度分层**（`dynamics/decay.py`）：

  | maturity | 半衰期 | 直觉 |
  |----------|--------|------|
  | embryonic | 24 h | 未经验证的概念一天内褪去 |
  | developing | 168 h（1 周） | |
  | established | 720 h（1 月） | |
  | core | 2160 h（3 月） | 中枢概念抵抗遗忘 |
  | fading | 24 h | 加速退出 |

- 衰减公式：`conf' = conf · 0.5^(Δh / HL)`；conf < **0.05** 转入 FADING；FADING 且 conf < **0.02** 被修剪；最近 **1 h** 内激活过的概念跳过衰减（宽限期）。

这套机制使概念世界具有"生长—稳定—退化—再激活"的生命特征，无需人工清理。

---

## 4. Relation（关系）

### 4.1 三轴模型

关系类型收敛为**三条认知轴**（`schemas/relation.py` `RelationType`），轴决定激活传播的全局基调（`dynamics/coefficients.py` `RELATION_TYPE_FACTOR`）：

| 轴 | 传播系数 | 认知语义 |
|----|---------|---------|
| `POSITIVE` | 1.00 | 吸引、依赖、使能、共创、相互强化——激励通道 |
| `PARALLEL` | 0.75 | 等价、重叠、共鸣、持续共注意——共振通道 |
| `NEGATIVE` | 0.60（抑制系数） | 互斥、冲突、约束违反、对抗——**抑制通道**（见 §6.4） |

旧式关系类型（`depends_on`、`contains`、`similar_to`、`contrasts` 等）作为向后兼容别名自动映射到三轴。

### 4.2 语义关系谱

轴之下是 26 个**语义关系**（`SemanticRelationSpec`），每个携带两种独立强度：

- `structural_strength`——该关系作为结构事实的强度（用于呈现与解释）；
- `propagation_strength`——该关系传导激活的强度（用于动力学）。

两者分离是关键设计：**NEGATIVE 轴关系结构强度很高（0.76–0.95）但传播强度极低（0.05–0.12）**——"A 与 B 互斥"是强结构事实，但绝不应把激活从 A 传给 B。

| 轴 | 语义关系（structural / propagation） |
|----|--------------------------------------|
| POSITIVE | membership (.94/.88) · proper_inclusion (.93/.87) · inclusion (.92/.86) · functional_map (.90/.84) · co_creation (.88/.82) · mutual_reinforcement (.86/.82) · future_coupling (.84/.78) · enables (.82/.76) · dependence (.78/.70) |
| NEGATIVE | disjointness (.95/.05) · incompatible_ontology (.90/.06) · exclusion (.90/.08) · complement (.88/.10) · violates_constraint (.86/.08) · conflict (.84/.10) · instability (.78/.12) · adversarial_prediction (.76/.10) |
| PARALLEL | equivalence (.96/.92) · quotient_map (.93/.90) · recursive_co_modeling (.86/.78) · approximate_equivalence (.82/.74) · persistent_attention (.78/.70) · similarity_kernel (.70/.64) · overlap (.66/.60) · generic_relation (.55/.45) · co_membership (.50/.45) |

`generic_relation` 是显式的临时回退（CLAUDE.md Rule 3），评测中以 `typed_ratio` 跟踪其占比。

### 4.2b 分数语义消费边界（不变量）

关系上的五个分数各有**唯一的消费端**。混用它们会制造 score soup——既表示可信度、又表示传播量、又表示展示优先级的"万能权重"。下表是硬边界：

| 字段 | 主要消费端 | **不应承担的职责** |
|------|-----------|-------------------|
| `structural_strength` | 渲染 / 解释 / 反信号显著度（counter-signal salience） | 不直接进入激活传播 |
| `propagation_strength` | 激活基准（声明的可传导性） | 不用于语义真值排序 |
| `weight` | 操作强化、衰减、激活实际乘子、投影关系排序 | 不等同于关系类型的语义概率 |
| `confidence` | 关系可靠性、修剪阈值、反馈治理 | 不直接代表语义类型成立概率 |
| `probability` | 关系类型信念 / 证据融合（Beta 后验） | 不用于原始激活强度 |

一个典型佐证：反信号（`CounterSignal`）的显著度刻意取 `structural_strength` 而非 `weight`——"这条约束作为结构事实有多强"属于解释通道，与该边被强化了多少次无关。

### 4.2c 负向关系可见性策略

负向关系有两种不同的认知用途，由两套机制分别承担：

- **传播层抑制**（恒开）：激活引擎的 inhibition 通道，见 §6；
- **投影层反信号**（按语义关系配置）：`NEGATIVE_VISIBILITY_DEFAULTS`，被抑制出局的概念以 ⚠ 警告形式告知 Agent"为什么这条路被关了"。

| 语义关系 | 默认策略 | 理由 |
|---------|---------|------|
| `disjointness` / `exclusion` / `complement` / `adversarial_prediction` | suppress | 分类边界：挡住错误概念本身就是任务，浮出反而是噪声 |
| `violates_constraint` | **expose** | 约束告警：Agent 必须看见 |
| `conflict` / `instability` / `incompatible_ontology` | conditional | 默认压制；debug / design review / ontology repair 等视角经 `Perspective.negative_visibility` 选择性暴露 |

暴露的反信号进入 `Projection.counter_signals` 并渲染为 `### Constraint Warnings` 节（无反信号时默认渲染保持字节不变）。

### 4.3 数据模型（`RelationEdge`）

| 分组 | 字段 | 默认值 | 语义 |
|------|------|--------|------|
| 拓扑 | `source_id` / `target_id` | 必填 | 有向边 |
| 类型 | `relation_type` / `semantic_relation` | PARALLEL / `""` | 轴由语义关系自动推导 |
| 强度 | `structural_strength` / `propagation_strength` | 0.55 / 0.45 | 来自语义谱 |
| | `weight` / `confidence` | 0.3 / 0.3 | 操作强度（激活与衰减使用） |
| | `probability` | 0.3 | **语义置信度**（见 §4.6） |
| 演化 | `reinforcement_count` / `disconfirmation_count` / `probability_observation_count` | 0 | 证据计数 |
| 标志 | `is_explicit` | False | 显式声明 vs Hebbian 自发现 |
| 溯源 | `provenance` / `task_history` | | 来源 / 参与过的任务（驱动任务亲和） |

### 4.4 显式关系 vs Hebbian 关系

两类关系遵循不同的强化曲线与上限——**Agent 显式声明的结构知识天然高于统计共现**：

```
显式（is_explicit=True）：  Δ⁺ = 0.08 / (1 + 0.05·n)，weight/confidence 上限 1.0
Hebbian（自发现）：         Δ⁺ = 0.06 / (1 + 0.15·n)，上限 0.7
削弱（共同）：              Δ⁻ = 0.06 / (1 + 0.10·n)，下限 0.01；削弱后 probability := confidence
```

**Hebbian 共现学习**（`dynamics/hebbian.py`，发生在 ingest 时）：

- 同一观察中共现的概念对累积共现计数，达到阈值 **2** 才创建一条 PARALLEL 边——单次偶然共现不产生结构；
- 单次观察最多处理 **30** 个概念对（防 O(n²) 爆炸）；
- 已有关系（任意类型）则直接强化，不再新建。

### 4.5 关系衰减

关系半衰期随强化次数增长——**被反复确认的关系衰减更慢**：

```
HL_eff = 72 h · (1 + 0.5 · reinforcement_count)
weight' = weight · 0.5^(Δh / HL_eff)；weight < 0.02 被修剪
```

### 4.6 关系概率（语义置信度）

`probability` 回答的不是"这条边多强"，而是"**给定当前概念世界与抽取证据，这类关系成立的概率**"。更新采用证据融合（见 `docs/relation-probability-redesign.md`）：

```
s = 2 + reinforcement_count + disconfirmation_count + probability_observation_count
p' = (p·s + p_prior·1.0 + p_evidence·2.0) / (s + 3.0)
```

设计原则：旧信念不被单次抽取覆盖（s 随历史增长）；当前文本证据（×2.0）权重高于类型先验（×1.0）；重复观察提高稳定性。

矛盾处理：摄入与既有关系矛盾的关系时，若无既有边可削弱，则削弱**端点概念**并记录在 `IngestResult.weakened_concepts`——矛盾信号不会被静默丢弃。

---

## 5. Context（上下文）

### 5.1 Context 对象

```python
class Context:
    task: str = ""                    # 当前任务标签
    perspective: Perspective          # 视角（见 §8）
    now: datetime | None = None       # 可注入时钟（确定性评测）

    effective_task = perspective.task or task   # 视角任务优先
```

Context 是贯穿激活与投影的**条件变量**，不是静态标签。同一种子集合在不同 task 下产生不同投影，这是系统的核心承诺（CLAUDE.md Rule 4），并有专门测试与评测用例守护。

### 5.2 任务亲和度（`dynamics/affinity.py`）

任务与历史标签的匹配是**分级 token 包含**，而非字符串相等：

```
A(task, history) = 1.0                          若 task 是某条历史的子串
                 = max over history of |T_task ∩ T_entry| / |T_task|   否则
```

其中 T 为签名 token 集（长度 ≥ 2、滤停用词）。亲和度的使用受门槛约束：

- `MIN_TASK_AFFINITY = 0.34`——低于此视为巧合匹配，不予加成；
- 达标后加成为 `β_task = 1 + (1.5 − 1)·A = 1 + 0.5·A`（`TASK_AFFINITY_BOOST = 1.5`），即满配匹配最多放大 1.5 倍。

亲和度同时作用于**关系的 task_history** 和**概念的 reinforcement_log**，取两者较大值。

---

## 6. Activation（激活）

### 6.1 双通道扩散激活模型

`ActivationEngine`（`dynamics/activation.py`）从种子概念出发做 BFS 扩散，但维护**两条独立通道**：

- **激励通道**：POSITIVE 与 PARALLEL 边传播正激活；
- **抑制通道**：NEGATIVE 边累积抑制量，**不是弱激励，而是反激励**。

最终净激活：`net(v) = max(0, activation(v) − inhibition(v))`，低于 `min_activation` 的被丢弃。被强烈排斥的概念即使有弱激活路径也会从投影中消失——这是"对比/互斥"语义的正确动力学表达。

### 6.2 种子激活

```
a₀(s) = min(1.0, confidence_s · β_task · β_dom)
```

其中 `β_dom`：种子概念的主导领域落在视角的 `active_domains` 内时 × `domain_affinity_boost`（默认 1.3）。

### 6.3 传播公式（激励通道）

一跳传播的完整乘法链：

```
a_{d+1}(n) = a_d(c)                          当前节点激活值
           · w_cn                             边权 weight
           · ω_P(ρ_cn)                        视角对该语义关系的权重（级联，见 §8.1）
           · max(conf_n, 0.3)                 邻居 readiness（PROPAGATION_FLOOR=0.3 兜底）
           · δ^(d+1)                          深度衰减（引擎默认 δ=0.6；World.project 入口默认 0.5）
           · β_task                           任务亲和加成 ∈ [1.0, 1.5]
           · β_dom                            领域亲和加成 ∈ {1.0, 1.3}
           · φ_rel                            关系新鲜度 = max(0.15, 0.5^(Δh/HL_eff))
           · φ_node                           概念新鲜度 = max(0.1, 0.5^(Δh/168h))
```

要点：

- **传播不只是"图上走几步"**——它同时是任务感知（β_task）、视角感知（ω_P、β_dom）、时间感知（φ）、置信度感知（readiness）的局部认知扩散；
- **新鲜度是只读软衰减**：仅在传播计算时打折，不修改库中的置信度（硬衰减只发生在 reflect）；
- **传播地板**：传播值 > 0 但低于 `max_seed × 0.03`（`PROPAGATION_MIN_RATIO`）时被抬到该地板——激活边界上的概念仍能收到最小信号；
- **多源合并取最大值**，并记录最强路径（驱动可解释性，见 §7.4）。

### 6.4 抑制通道

```
inhibition(n) += raw · 0.6        （CONTRASTS_INHIBITION_FACTOR，对 NEGATIVE 边）
```

抑制独立累积、不再向外传播，最终在净激活处与激励对消。

---

## 7. Projection（投影）

投影是系统的**操作输出**（CLAUDE.md Rule 5）：把更大的概念世界变成一份"足够小、足够准、足够有结构"的任务视图。

### 7.1 种子解析级联

用户给出的种子可能拼错、有歧义、或只是近似名称。`resolve_in_context()`（`concepts/_manager.py`）执行四层级联：

```
1. exact          ID / 名称 / 别名精确匹配
2. disambiguated  多候选时，恰好一个落在视角活跃域 → 选它
3. fuzzy:<score>  签名包含度模糊匹配，阈值 0.45
4. unresolved     无解
```

每个种子的解析方式透明记录在 `Projection.seed_resolution`（如 `{"model servng": "model serving (fuzzy:0.67)"}`）——**被猜测或未命中的种子绝不静默**。

### 7.2 候选相关性

激活产出的每个候选概念计算相关性：

```
rel(c) = (a(c) / a_max)                     归一化激活
       · (0.6 + 0.4·A_c)                    任务亲和（TASK_AFFINITY_DISCOUNT=0.6：无任务关联打 6 折，不清零）
       · dom_c                              领域亲和 ∈ {1.0, 1.3}
       · (0.7 + 0.3·φ_c)                    时间新鲜度（TEMPORAL_WEIGHT=0.3，HL=168h）
```

### 7.3 MMR 选择

不按分数直接截断，而用 MMR（Maximal Marginal Relevance）贪心选择，平衡相关性与多样性：

```
MMR(c) = (1 − λ)·rel(c) − λ·max_{s∈已选} J(N(c), N(s))      λ = 0.3
```

冗余度定义为候选与已选概念**邻域集合的最大 Jaccard 相似度**——选中的概念应覆盖结构上不同的邻域，而不是同一邻域的多个成员。

### 7.4 可解释性

- **激活路径**：每个非种子投影概念携带 `ActivationTrace` 最佳路径，`Projection.explain(name)` 输出形如：
  `kv cache ←(dependence, positive, ×0.42)— request batching ←(…)— model serving [seed] (score 0.31)`
- **Counter-signals**：投影内所有 NEGATIVE 轴关系单独呈现（`A ✕ B (conflict, weight 0.62)`）——告诉 Agent"哪些概念在此任务中相互排斥"与"哪些概念相关"同样重要。

### 7.5 渲染风格（`projection/render.py`）

| 风格 | 结构 | 适用 |
|------|------|------|
| `default` | 按成熟度分组（核心理解 / 活跃 / 新兴）+ top-10 关系 | 标准 prompt 注入 |
| `compact` | 单行概念 + top-5 关系 | 紧 token 预算 |
| `detailed` | default + "Why included" 激活路径 + Counter-signals | 需要可审计推理时 |

渲染输出为 markdown，直接注入 Agent 的 system prompt。

---

## 8. Perspective（视角）

### 8.1 设计意图与权重级联

同一概念世界在"调试 / 设计 / 研究"视角下应当给出不同回答。Perspective 通过**三层权重级联**介入激活传播：

```
ω_P(ρ) = semantic_relation_weights[ρ]        若存在该语义关系的覆盖
       = relation_type_weights[axis(ρ)]      否则若存在轴级覆盖
       = RELATION_TYPE_FACTOR[axis(ρ)]       否则全局默认 {positive:1.0, parallel:0.75, negative:0.6}
```

字段全表：`name` · `role` · `task`（覆盖 Context.task）· `description` · `active_domains`（领域聚焦）· `relation_type_weights`（轴级）· `semantic_relation_weights`（**单个语义关系级**）· `domain_affinity_boost`（默认 1.3）· `render_style`。

### 8.2 内置 profile（`perspectives/profiles.py`）

| profile | role | 语义关系权重 | 轴权重 | 渲染 | 直觉 |
|---------|------|-------------|--------|------|------|
| `default` | — | — | — | default | 中性 |
| `debug` | engineer | dependence 1.4 · enables 1.3 · conflict 1.2 · overlap 0.5 | parallel 0.6 | compact | 沿依赖链追因果，压制松散相似 |
| `design` | architect | overlap 1.4 · equivalence 1.3 · co_creation 1.3 · dependence 0.8 | — | default | 找同构与组合机会，弱化实现依赖 |
| `research` | researcher | similarity_kernel 1.4 · overlap 1.3 · persistent_attention 1.2 | — | detailed | 放大类比与长期关注信号 |

### 8.3 持久化

`PerspectiveRegistry` 把自定义视角序列化进世界状态（`state.json`）；内置 profile 不可删除。`World.project(perspective=...)` 同时接受名称字符串与内联 `Perspective` 实例。

---

## 9. 巩固与演化（Reflect）

`reflect()` 是周期性认知巩固，五阶段管线（`world/_reflect.py`）：

```
衰减（decay） → 群落检测（communities） → 色场再播种（color reseed） → 生命周期评估（lifecycle） → 修剪（prune）
```

### 9.1 群落检测（`dynamics/community.py`）

- **确定性同步标签传播**（无随机性，可复现）：最多 8 轮迭代，按加权投票更新标签，字典序破平；
- 有效耦合考虑时间新鲜度：`K_ij = w_ij · f_axis · φ_rel · (φ_i + φ_j)/2`；
- 群落最小规模 **3**；内部耦合度最高的 **35%** 成员（至少 1 个）为 core；
- 群落跨 reflect 周期持久化，`stability` 计数器跟踪重现次数。

### 9.2 色场动力学（`dynamics/color_diffusion.py`）

色场（`domain_profile`）是**自组织的领域感知**：稳定群落自发"生色"，颜色沿关系扩散，失去结构支撑后自然褪去（完整设计见 `docs/world0-color-field-dynamics.md`）。统一演化方程：

```
dc^α/dt = −η^α L_t c^α  +  s^α(t)  −  μ^α(t) ⊙ c^α
            扩散项          源项         褪色项
```

实现要点：

- **生色门槛**：群落 `stability ≥ 2` 且规模 ≥ 3 才成为色源（结构先于颜色）；core 成员注入 0.65，普通成员 0.35；任务/领域显式注入 0.55；
- **饱和混色**：`c' = c + a·(1 − c)`——颜色叠加不超过 1.0；
- **扩散**：基础速率 0.18，每步衰减 0.62，标准 2 步；穿越群落边界自然减弱（按耦合强度调制）；
- **褪色**：每个颜色分量独立褪色，褪色率 `μ = max(0.02, deficit/4.0)`，deficit 为该分量相对邻域中位数的不足——**有邻域支撑的颜色保持，孤立颜色蒸发**（分量 < 0.015 删除）。

色场的消费端：视角的 `active_domains` 据此做领域消歧与领域加成（§7.1、§6.2），可视化据此着色。

### 9.3 防污染保证

色场的数学性质（正性、有界性、无源衰减）保证不会发生无限污染：颜色有源（稳定群落）有汇（褪色与蒸发），混色态是稳定边界层而非污染。

---

## 10. 诊断与评测

### 10.1 网络熵（`metrics/entropy.py`）

诊断概念注意力是聚焦还是弥散。每个概念的局部关系分布 Shannon 熵（归一化），按节点重要性加权汇总：

```
有效权重    m(r) = max(0, probability) · f_axis · e_r        e: 显式 1.0，Hebbian 0.75
局部熵      H_norm(c) = −Σ p_i log₂ p_i / log₂|N(c)|         （|N| < 2 不计）
节点重要性  imp(c) = max(conf, 0.05) · maturity_factor        {emb .5, dev .75, est 1.0, core 1.2, fading .35}
世界熵      Σ imp·H_norm / Σ imp
```

解释区间：0–0.20 高度集中/欠连接 · 0.20–0.45 结构化且聚焦 · 0.45–0.70 多样但可解释 · 0.70–1.00 弥散（疑似噪声）。

### 10.2 投影质量评测（`scripts/eval_projection_matrix.py`）

无 LLM、确定性的离线评测，指标：`seed_rate`（种子解析率）、`P@5/P@8`、`R@8`、`NDCG@8`、`noise_rate`（干扰概念混入率）、`typed_ratio`（非 generic 关系占比）、`inhibition`（应抑制概念是否泄漏）。

当前基线（`docs/projection-eval-baseline.md`，默认视角）：

| 用例 | seed_rate | P@5 | P@8 | R@8 | NDCG@8 | noise | typed | inhibition |
|------|-----------|-----|-----|-----|--------|-------|-------|------------|
| ml_from_serving | 1.000 | 0.600 | 0.750 | 1.000 | 0.915 | 0.250 | 0.389 | ok |
| ops_from_serving | 1.000 | 0.800 | 0.750 | 1.000 | 0.971 | 0.250 | 0.333 | ok |
| typo_seed（拼错种子） | 1.000 | 0.600 | 0.750 | 1.000 | 0.915 | 0.000 | 0.389 | ok |

系数敏感性由 `scripts/sweep_projection_quality.py` 报告（这是调参辅助而非自动调优器——调优保持人工、可解释）。

### 10.3 提取质量评测

模型 × prompt 评测（9 模型 × 3 轮，`docs/extraction-model-prompt-eval.md`）的核心结论：**prompt 是主导杠杆**——生产 prompt 下结构质量基本与模型无关，默认提取模型取最低成本的顶级梯队成员 `gpt-5.4-nano`（9.0/9，零噪声、中文保持稳定）。

---

## 11. 反馈闭环（`World.apply_feedback`）

投影的消费者（Agent 或人）可以把"什么有用、什么误导"反哺给世界，全部经由公开 facade：

| 反馈通道 | 操作 | 语义 |
|---------|------|------|
| `useful_concepts` | reinforce（+激活记录，含 task） | 投影中被证实有帮助的概念 |
| `missing_concepts` | get_or_create + reinforce | 投影遗漏但实际需要的概念 |
| `noisy_concepts` | confidence −0.05 | 此任务中误导性概念 |
| `useful_relations` | reinforce | 推理中被证实的关系 |
| `weak_relations` | weight/confidence −0.05 | 虚假或不相关的关系 |

关系引用支持 ID 或人类可读标签 `"src -> relation -> tgt"`（自动解析语义关系名或轴名）。noisy 与 useful 冲突时 noisy 优先（同一概念不会先降权再强化）。

这条通道使 World 0 的演化不只来自被动观察，还来自**使用结果**——投影质量随使用闭环收敛。

---

## 12. 设计不变量

任何改动都不应违反以下不变量（违反即架构漂移）：

1. **概念优先**：事实不入核心模型；概念携带认知状态，不携带文档；
2. **关系必须有类型**：`generic_relation` 只是显式的临时回退，且被 `typed_ratio` 指标持续监控；
3. **上下文改变相关性**：相同种子 + 不同任务 ⇒ 不同投影（有评测用例守护）；
4. **投影是唯一操作输出**：任何"扩大存储"的特性若不改善投影质量即为低优先级；
5. **NEGATIVE 是抑制不是弱激励**：结构强度与传播强度分离；
6. **显式 > 统计**：Hebbian 关系权重上限 0.7，永远低于显式声明的 1.0；
7. **结构先于颜色**：只有稳定群落才生色；颜色无源必褪；
8. **读路径不写库**：新鲜度软衰减只影响传播计算；硬衰减只发生在 reflect；
9. **不静默**：种子解析、矛盾削弱、修剪全部留痕（`seed_resolution`、`weakened_concepts`、reflect 报告）；
10. **核心与 agent 层边界**：自主个体（autonomy）只通过公开 `World` facade 行动；情节性记忆（journal）永不进入核心。

---

## 附录 A：系数总表（`dynamics/coefficients.py` 等）

| 系数 | 值 | 作用域 |
|------|-----|--------|
| `RELATION_TYPE_FACTOR` | positive 1.0 / parallel 0.75 / negative 0.60 | 轴级传播默认 |
| `CONTRASTS_INHIBITION_FACTOR` | 0.6 | 抑制通道强度 |
| `TASK_AFFINITY_BOOST` / `MIN_TASK_AFFINITY` | 1.5 / 0.34 | 任务加成 / 门槛 |
| `PROPAGATION_FLOOR` | 0.3 | 低置信邻居最小通过性 |
| `PROPAGATION_MIN_RATIO` | 0.03 | 传播地板（× max seed） |
| `CONCEPT_TEMPORAL_HL` / `RELATION_TEMPORAL_HL` | 168 h / 72 h | 只读新鲜度半衰期 |
| `MMR_LAMBDA` | 0.3 | 多样性权重 |
| `TASK_AFFINITY_DISCOUNT` | 0.6 | 投影相关性的无任务折扣 |
| `TEMPORAL_WEIGHT` / `PROJECTION_TEMPORAL_HL` | 0.3 / 168 h | 投影新鲜度 |
| 概念半衰期 | 24 / 168 / 720 / 2160 / 24 h | 按成熟度（硬衰减） |
| 关系基础半衰期 | 72 h × (1 + 0.5·n) | 硬衰减 |
| 修剪阈值 | 概念 0.02（且 FADING）/ 关系 0.02 | reflect |
| Hebbian | 共现阈值 2 / 单次最大 30 对 / 上限 0.7 | ingest |
| 归并阈值 | 签名 0.6 / 同义词 0.78 / 模糊 0.45 / 跨域 ×0.3 | 概念身份 |
| 生命周期 | dev: act≥3,conf≥0.3 · est: act≥10,conf≥0.6 · core: act≥30,conn≥max(2,5−(act−30)//20) | lifecycle |
| 色场 | 注入 0.55 / 群落 core 0.65、member 0.35 / 扩散 0.18×0.62^step×2 步 / 褪色 μ≥0.02, τ=4 / 蒸发 0.015 | color field |
| 群落 | 标签传播 ≤8 轮 / 最小规模 3 / core 35% / 色源 stability≥2 | communities |
| 熵 | 显式 1.0 / Hebbian 0.75；成熟度因子 .5/.75/1.0/1.2/.35 | metrics |
| `apply_feedback` | confidence_delta 0.05 | 反馈 |
