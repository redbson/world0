# World 0 关系（Relation）设计与使用优化报告

> 本文将一次多源、经对抗式核验的深度调研（state-of-the-art 文献）与 World 0 当前关系子系统的**实际代码实现**逐条对照，给出可落地的优化建议。
>
> 调研规模：5 个检索角度 → 19 篇来源 → 抽取 94 条主张 → 核验 25 条（**23 条确认 / 2 条证伪**）。每条建议都附带证据强度与对应的 `文件:行号` 锚点。

---

## 0. 结论速览（先看这里）

World 0 的几项核心设计选择经文献验证是**深思熟虑的取舍，而非缺陷**：

- **类型化的三轴模型（positive / parallel / negative）是合理的差异化设计。** 所有被调研的生产级系统（Zep/Graphiti、Mem0-g、A-MEM、HippoRAG）都采用开放式 / 无类型边；而文献明确指出：无类型边**无法**支撑"按边类型遍历（edge-type-aware traversal）"与"视角加权（perspective weighting）"——这恰恰是 World 0 的 `SEMANTIC_RELATION_SPECS` + `Perspective.weight_for_relation` 所提供的能力。**应当保留。**
- **structural_strength 与 propagation_strength 的拆分**直接得到 ACT-R 激活方程 `A_i = B_i + Σ W_j·S_{j,i}` 的支持（内在基础项 + 上下文驱动的扩散项）。

可落地的优化集中为 **5 类、6 项**改动，按"证据强度 / 实施成本"排序见文末优先级表。

---

## 1. 用概念稀有度（IDF 式）加权种子激活 —— 投入产出比最高、成本最低

**证据（高，3-0 通过）：** HippoRAG 定义节点特异度 `s_i = 1/|P_i|`（该概念被抽取自的来源数量的倒数），在传播前乘进每个种子的初始概率。消融该项后 R@5 由 72.9 降到 70.9。核心洞见：**给"种子"加偏置，而不是给"边"加偏置**——这样能保持边语义干净，且计算极廉价。［arXiv:2405.14831］

**对应代码：** `activate_traced()` 将所有查询概念以统一 `seed_score` 作为种子后扩散（[`activation.py:117`](../src/world0/dynamics/activation.py)），种子阶段完全没有稀有度。

**改动：** 用特异度项缩放每个种子的初始激活——例如取"该概念被抽取自多少来源 / 任务"的倒数（你已经在 `reinforcement_log` 任务与来源 provenance 中记录了这些信息）。稀有种子 = 更强、更具判别力的种子。本质上是对 `seed_score_max` 的一个乘子，**不需要改任何边**。

---

## 2. 倾向"带重启的传播"，而非"硬半径截断"

**证据（高，3-0 通过）：** HippoRAG 的决定性消融实验——以查询节点为种子的 PPR 显著优于"仅查询节点"（R@5 72.9 vs 56.2）；但**不带 PPR、直接加入 1 跳邻居反而比仅查询节点更差**（R@2 42.2 vs 50.7）。原文："Adding the neighborhood without PPR leads to worse performance."（无加权的固定跳数扩展，注入的噪声多于信号。）［arXiv:2405.14831］

**对应代码：** BFS 前沿扩展，`depth_factor = decay^(depth+1)`，外加 `PROPAGATION_MIN_RATIO=0.03` 的下限**拓宽**视野至 3-4 跳（[`activation.py:207`](../src/world0/dynamics/activation.py)、[`coefficients.py`](../src/world0/dynamics/coefficients.py)）。你属于正确的家族（带衰减的扩散，而非硬截断）——但**flooring 机制有再次注入 HippoRAG 所警告噪声的风险**：它把弱的远距离节点重新顶到下限。

**改动：** 对下限保持警惕。要么 (a) 让 `PROPAGATION_MIN_RATIO` 可按视角调参并把默认值调低；要么 (b) 转向"重启 / 归一化"模型——每次扩散守恒总激活质量，而非设下限。
**调研标记的开放问题：** PPR 假设转移概率非负，而你的**抑制通道可把激活压到负值**，因此负轴可能需要单独的（非 PPR）传播规则或有符号图（signed-graph）变体。**不要盲目用 PPR 替换双通道设计。**

---

## 3. 引入"时间失效（temporal invalidation）"——与抑制轴区分开

**证据（高，3-0 通过）：** SOTA 的矛盾处理**不是**负边、也不是删除。Zep 把被取代边的 `t_invalid` 设为新边的 `t_valid`（双时间维 bi-temporal，保留历史）；Mem0-g 通过离散的 ADD/UPDATE/DELETE/NOOP 由 LLM 将过时关系标记为失效。［arXiv:2501.13956、2504.19413］

**对应代码：** 矛盾通过 `weaken()` 处理——下调 `confidence`/`weight`，递增 `disconfirmation_count`（[`manager.py:393`](../src/world0/relations/manager.py)、[`_ingest.py:292`](../src/world0/world/_ingest.py)）。你的负轴（`conflict`、`disjointness` 等）建模的是**语义对立**（A 抑制 B）。

**调研揭示的关键区分：** *"A 曾经与 B 相关，但现在不再"*（时间矛盾）与 *"A 在语义上对立于 B"*（抑制）在架构上是两回事。你目前把两者都塞进 weaken + 负轴，会**丢失历史**——被削弱的边无法与被取代的边区分。

**针对本项目的特别提醒：** CLAUDE.md 明确划定了"不承担记忆 / 事实存储职责"的边界。因此这需要一次设计决策，而非照搬——*"这条概念关系不再成立"*属于合法的认知范畴；*"这条事实过期了"*属于记忆。若要给 `RelationEdge` 增加 `valid`/`invalidated_by` 生命周期标记，应**仅限于概念结构层面的变化**，并把抑制轴**保留给真正的语义对立**。

---

## 4. 把"声明权重"与"学习权重"融合进边权；并修复枢纽节点不稳定

此处有**两项**发现。

### (a) 枢纽节点不稳定（高，3-0 通过）

ACT-R 的扇出稀释 `S_ji = smax − ln(fan_ji)`，**当源概念出现在 >~7 个 chunk 时会翻转为抑制**（ln 7 = 1.95 > 默认 smax 2.0）。你的"类型 → `propagation_strength`"确定性映射没有度数项，因此**高扇出的枢纽概念会过度扩散**。［ACT-R CogSci-2014］

> **改动：** 在 `activate_traced` 中按源节点的扇出 / 度数稀释 `propagation_strength`（或对每个源做扩散归一化）。度数信息可由 `relations.for_concept(id)` 取得。

### (b) 声明 + 学习 的边权融合（高，3-0 通过）

GraphRAG 在创建时声明一个数值 `relationship_strength`，**同时**把重复实例计数聚合为学习权重。你的 `structural_strength`/`propagation_strength`（声明的，来自规格表）与 `reinforcement_count`（学习的，Hebbian）正是这两个通道——但调研指出一个**没有任何来源给出答案的开放问题**：如何组合两者？候选：相加、相乘，或**在不同阶段消费的独立通道**（声明权重 = 是否可遍历；学习权重 = 排序）。［arXiv:2404.16130］

> 最契合你架构的是"分阶段"方案：声明权重在激活阶段把关边是否被遍历；学习的 `reinforcement_count`/`weight` 在投影阶段为边排序（[`engine.py:235`](../src/world0/projection/engine.py)）。

**反模式（已证伪，0-3）：** **不要**把 ACT-R 当作你 Hebbian 共现学习的先例——ACT-R 的 `S_ji` 是带扇出稀释的*声明*值，并非学习而来。你正确的先例是 GraphRAG 的重复计数聚合，那是有证据支持的。

**另一项值得考虑（高，3-0 通过）：** ACT-R 的基础激活 `B_i = ln(Σ t_j^{−d})`（多痕迹幂律）比你当前的单时间戳指数衰减 `0.5^(hours/HL)` 更有原则性（[`decay.py:81`](../src/world0/dynamics/decay.py)、[`manager.py:485`](../src/world0/relations/manager.py)）。频繁被强化的关系应自动抵抗衰减；存在一种"优化版基础激活近似"可避免存储全部时间戳，因此对边也是可行的。

---

## 5. 两项投影阶段（projection-time）升级

### (a) 增加一类"相似 / 等价"边（高，3-0 通过）

HippoRAG 在余弦相似度 > 0.8 时自动加入"同义（synonymy）"边；消融后 R@5 由 72.9 降到 70.5（在实体密集数据上影响最大）。［arXiv:2405.14831］你的并行轴里已经有 `equivalence`/`similarity_kernel`/`approximate_equivalence`（[`relation.py:90`](../src/world0/schemas/relation.py)）——但它们仅在 LLM 显式声明时才创建。

> **改动：** 在 ingest 阶段从嵌入相似度**自动生成 `similarity_kernel` 边**，并使其**显式化**（符合你的 Rule 1：显式结构优先于潜在相似度）。这能在无需独立向量检索层的情况下，为"模式补全"桥接近重复概念。

### (b) 对边排序，而非只是遍历（高，3-0 通过）

Zep 对边做嵌入 + 重排；Mem0-g 对整条三元组做嵌入并按相似度排序。GraphRAG 在 token 预算下按**度数**选边——这是类型盲的，*正是你的类型化模型可以胜过的基线*。［arXiv:2501.13956、2404.16130］你的投影已经按 `weight × perspective_factor` 对关系排序并以 `max_relations` 截断（[`engine.py:235`](../src/world0/projection/engine.py)）——**这一点你已领先**。升级方向：让预算选择**按边类型 + 视角加权**（你能做到，GraphRAG 做不到），并考虑对关系的 rationale 文本做嵌入以用于打破平局。

---

## 6. 优先级建议与实施状态

| # | 改动 | 成本 | 证据 | 状态 |
|---|------|------|------|------|
| 1 | 稀有度加权种子 | **低** | 3-0 | ✅ **已实现** — `ActivationConfig.seed_specificity_weight`（默认 0.3），种子按相对稀有度归一化加权，单种子中性 |
| 4a | 扇出 / 度数稀释传播 | 低 | 3-0 | ✅ **已实现** — `FAN_DILUTION_THRESHOLD=7` + 平滑对数稀释，永不翻转为负（修复而非复刻 ACT-R 病灶） |
| 5a | 自动相似边（显式） | 中 | 3-0 | ✅ **已实现（两阶段）** — `dynamics/similarity.py` `SimilarityLinker`：签名匹配低阈值**召回**候选 + LLM `similarity.judge.system` **精判**语义（可选 `equivalence`/`approximate_equivalence`/`similarity_kernel`，A-MEM/Mem0 模式）。LLM 可链接零词干重叠对（"vector database" ≈ "vector store"、跨语言）并拒绝 token 仿冒对（`chain_2` vs `chain_3`）；无 LLM 时回退严格词法规则（Jaccard ≥ 0.5 + 单 token 禁链），判定失败也回退、ingest 永不中断。相似强度编码为边权，封顶 0.7 |
| 4b | 分阶段"声明 vs 学习"边权 | 中 | 3-0 | ⏳ 待做 — `activation.py` + `engine.py` |
| 3 | 时间失效生命周期 | 中-高 | 3-0 | ⏳ 待做 — **需先做"记忆 vs 认知"边界决策** |
| 2 | 重启 / 有符号传播 | **高** | 3-0 | ⏳ 待做 — **研究级开放问题**（PPR 假设非负转移） |

**实施备注（2026-06）：** 第 1、4a、5a 项已落地并配套 15 个专项测试
（`dynamics/tests/test_relation_optimizations.py`），全量套件 818 通过。
实现中发现并修正了一个调研未覆盖的精度陷阱：签名分词会丢弃数字
（`chain_2` → `{chain}`），若不设"最少 2 个签名 token"门槛，相似链接器
会在人造命名的兄弟概念间建满权重捷径——这正是 HippoRAG 用高阈值
（cosine 0.8）防御的同一类噪声。该陷阱随后推动 5a 升级为两阶段：
词法只负责召回，精度交给 LLM 判定（语义而非拼写），词法严格规则
降级为无 LLM / 判定失败时的回退路径。成本：每个有候选的新概念至多
增加一次 LLM 调用（`ingest_text` 本身已有一次抽取调用）。

**后续落地（同月）—— 全语义路径统一到系统 LLM：** Hebbian 共现关系
（最后一条非 LLM 的关系创建路径）升级为 LLM 判型：跨过共现阈值的概念对
由 `relation.typing.system` 从 26 类语义清单中选取最具体的关系（含方向与
置信度），巧合对返回 `"none"` 直接拒绝（计数器清零、持续共现会重新累积
并再判）。这落实了调研发现 #1 的建议（"LLM 提议标签 → 归一化映射到类型
体系"，Graphiti 的抽取灵活性 + World 0 的结构保留），也使 Rule 3
（generic 仅作回退）对"发现的关系"同样成立。配置 LLM 后，文本抽取、
相似判定、共现判型三条语义路径共用同一个系统选定模型；无 LLM 时
全部回退到原词法/统计行为。全量套件 823 通过（20 个专项测试）。

---

## 6b. 真实 LLM 实测记录（2026-06-11，Azure OpenAI gpt-5.4-nano）

用 6 段真实文档散文经 `ingest_text` 建世界（全 LLM 语义链路），跑
`scripts/eval_real_corpus_llm.py`（8 变体 × 2 任务，LLM 回答 + LLM 评分）：

**结构质量（确定性结论）：**
- typed_ratio **1.0**、generic **0**——源头判型直接达到目标状态，reflect 无债可治；
- 8 条负轴边语义精准（如 `raw transcript retrieval → violates_constraint →
  semantic organization of meaning`）；Hebbian LLM 判型与 reflect 再判型均在
  真实链路触发（reflect 将一条 generic 重判为 `enables`）。

**消融（LLM judge，n=2 任务，方差大，仅作方向参考）：**

| 变体 | tokens | 静默违规 | correctness | groundedness |
|---|---|---|---|---|
| raw_history | 379 | **0.50** | 4.5 | 4.0 |
| lexical_recall | 211 | 0.00 | 5.0 | 4.0 |
| projection（默认渲染） | 331 | 0.00 | 3.0 | 3.0 |
| **projection_compact** | **212** | **0.00** | **5.0** | **5.0** |

**三个教训：**
1. **渲染风格决定成败**：默认渲染的结构元数据（representation id、置信度小数）
   既贵（331 tok）又伤 groundedness（3.0）；compact 渲染同一投影 → 212 tok、
   correctness/groundedness 双 5.0，全场最佳。**Agent prompt 注入应默认 compact**；
   本次顺带修复了 compact 不渲染反信号的缺陷（⚠ 行现已保留）。
2. **小语料是 raw_history 的主场**：6 段文本（379 tok）整体塞进 prompt 时压缩
   没有价值空间——投影的适用域下界被实测确认；真正的判决需要 raw_history
   超出预算的规模（50+ chunks）与 ≥10 个任务。
3. **静默违规分离在真实回答下成立**：raw_history 0.50 vs 投影全部 0.00——
   唯一不受 judge 方差影响的确定性指标。

## 7. 开放问题（文献未给答案）

1. **声明权重与学习权重如何融合为单一边权**——相加、相乘，还是作为在不同阶段消费的独立通道（声明用于遍历资格，学习用于排序）？没有任何来源给出融合公式。
2. **时间失效模式是否适合 World 0**——鉴于 CLAUDE.md 对"记忆 / 事实存储"职责的明确边界，区分"这条概念关系不再成立"（合法认知）与"这条事实过期"（记忆）需要先做设计决策。
3. **PPR 式全图传播如何迁移到双通道（激励 + 抑制）扩散**——负边可把激活压到负值，而 PPR 假设非负转移概率，因此抑制通道可能需要单独的非 PPR 规则或有符号图变体；无来源覆盖此点。
4. **类型化清单的合适粒度**——所有被调研系统都走向开放 / 无类型，是否存在经验上的甜点（例如映射到 3 轴的约 10-30 种类型），既保留类型感知遍历的收益，又避免迫使他人转向自由标签的抽取脆性？

---

## 8. 来源与可信度

- **同行评审 / 教科书级（权重最高）：** HippoRAG［arXiv:2405.14831，NeurIPS'24］与 ACT-R 系列为经典、时间不敏感来源。
- **厂商预印本（引用机制，不引用基准）：** Zep［2501.13956］、Mem0［2504.19413］为厂商自撰、未经同行评审；其头条基准数字（如 Zep 94.8 vs 93.4 DMR、最高 18.5% LongMemEval）为自报、跨设置、且至少在一处互相质疑（Zep 与 Mem0 互相质疑对方数据）。**引用其架构机制（对自身系统的描述无争议），但将性能差值仅作指示性参考。**
- **被证伪的两条主张（核验阶段剔除）：**
  - GraphRAG 仅按出现频次加权边（1-2 否决）——实际上它**同时**有声明的 `relationship_strength`。
  - ACT-R 的 `S_ji` 由共现学习而来（0-3 否决）——实际为**声明值 + 扇出稀释**。

---

*生成方式：deep-research 工作流（5 角度 / 19 来源 / 94 主张 / 25 条对抗式核验）+ World 0 关系子系统代码全量映射。*
