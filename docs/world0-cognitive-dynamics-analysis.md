# World 0 认知动力学分析与优化

> 对 World 0 基础设计（数学模型、结构、算法、系统）的深度分析，附实证探针、
> 修复方案、参数标定与后续路线。目标：让 World 0 成为一个**可持续运行、
> 动态更新**的 Agent 认知层，而不是只在演示脚本里工作一次的图结构。
>
> 复现方式见 §8。所有探针数字均来自本仓库代码在同一台机器上的实际运行。

---

## 0. 摘要

World 0 的概念框架（概念 / 关系 / 上下文 / 激活 / 投影）是清晰且自洽的，
模块边界（Protocol 驱动的 Lego 结构）也很好。但作为一个**随时间持续演化**的
动力系统，原实现有 8 处会在真实运行中失效的数学或系统缺陷：

| # | 问题 | 后果（修复前实测） | 状态 |
|---|------|--------------------|------|
| A | 衰减不幂等：每次 `reflect()` 都对"自上次激活以来的全部时长"再衰减一次 | 48h 后调用 1 次衰减 → 0.323；同一瞬间调用 3 次 → 0.218 | ✅ 已修 |
| B | 成熟度阶梯不可达：加性递减增益 vs 乘性衰减的稳态太低 | 每日使用 120 天：confidence 0.059，永远到不了 established；每周使用：直接 FADING | ✅ 已修 + 标定 |
| C | 激活取 `max` 聚合：多路径汇聚不被奖励 | 被两个种子同时指向的概念与只被一个种子指向的概念得分完全相同 | ✅ 已修 |
| D | 常数传播下限抹平排序：深度 ≥3 的所有节点得分并列 | 链上 c3=c4=c5=c6=0.01733，弱泛化边与强依赖边同分 | ✅ 已修 |
| E | 投影跨进程不确定：循环内逐个取 `datetime.now()` 产生与文件遍历顺序相关的噪声；MMR 用 `set` 迭代 | 同一存储在不同 `PYTHONHASHSEED` 下投影结果不同 | ✅ 已修 |
| F | Hebbian 共现计数器只在内存中 | 重启后计数清零，跨会话共现永远达不到阈值 | ✅ 已修 |
| G | 关系语义概率随时间衰减（`probability = confidence`）；`weaken()` 把结构强度尺度复制进概率 | 30 天闲置：P(关系正确) 0.70 → 0.008；一次否证反而让 probability 0.70 → 0.80 | ✅ 已修 |
| H | `reinforcement_log` 无上限，任务亲和是 O(log 长度) 的子串扫描 | 500 次激活 → 500 条日志、55 KB/概念文件；`"ml"` 匹配 `"html parsing"` | ✅ 已修 |

此外本文档记录了若干**未在本轮修改、但决定系统能否长期演化**的结构性问题
（§7），并给出面向"持续更新的认知系统"的路线建议。

---

## 1. 分析范围与方法

- **阅读**：`schemas/`、`dynamics/`、`projection/`、`concepts/`、`relations/`、
  `world/`、`metrics/`、`store/` 全部核心代码（约 2.8k 行）以及
  `DesignPhilosophy.md`、`DCTMTemporalDynamics.md`、`docs/world0-paper.md`、
  `docs/relation-probability-redesign.md`、`docs/world-network-entropy-design.md`、
  `docs/world0-color-field-dynamics.md`。
- **数学推导**：对每个更新算子写出闭式或稳态公式，寻找与设计意图（AGENTS.md /
  DesignPhilosophy）不一致的地方。
- **实证探针**：为每个假设写最小可复现脚本（§4、§8），修复前后各跑一遍。
- **回归**：全部 1006 个测试通过（原 978 + 新增 28 个行为测试
  `tests/test_dynamics_analysis.py`）。

---

## 2. 系统模型的形式化

记概念 $v$ 的状态为 $(c_v, m_v, n_v, d_v, t_v)$：置信度、成熟度、激活次数、
否证次数、最近激活时间；关系 $e$ 的状态为 $(w_e, p_e, r_e, t_e)$：操作权重、
语义概率、强化次数、最近强化时间。系统由四个算子驱动：

```text
Ingest(O):    get_or_create → activate(+boost) → discover/reinforce → hebbian → color seed
Activate(S):  分层扩散  a(v) = Agg_paths[ a(u)·w·ρ_type·max(c_v, 0.3)·δ^depth·γ_task·γ_domain·τ_rel·τ_node ]
Project(a):   MMR 选择  argmax (1−λ)·rel(v) − λ·max_{s∈sel} Jaccard(N(v), N(s))
Reflect():    decay → community → color → lifecycle → prune
```

其中 `activate()` 的增益为 $b(n) = 0.06/(1+0.08n)$；衰减为
$c \leftarrow c \cdot 2^{-\Delta t / HL(m)}$，$HL$ 只依赖成熟度
（embryonic 24h、developing 168h、established 720h、core 2160h）。

一个"持续更新的认知系统"对这些算子至少要求：

1. **时间一致性**：算子的效果只依赖于物理时间，不依赖调用频率（幂等）。
2. **证据可积累**：反复确认应带来持久性；一次性噪声应被遗忘。
3. **排序信息不丢失**：激活分数应保序，投影应对同一世界确定。
4. **学习状态可持久化**：所有跨调用的中间状态必须落盘。
5. **语义与操作量分离**：`P(关系正确)` 不应因时间流逝而变化。

原实现在 1、2、3、4、5 上都有缺口，下面逐模块分析。

---

## 3. 逐模块分析

### 3.1 置信度更新与衰减（`schemas/concept.py`、`dynamics/decay.py`）

**问题 A — 衰减不幂等。** 原实现：

```python
hours = node.hours_since_activation()
node.confidence *= 0.5 ** (hours / half_life)
```

参考点是 `last_activated`，因此在没有新激活的情况下，第 $k$ 次调用会再次乘以
$2^{-\Delta t_k/HL}$，总衰减为 $2^{-(\Delta t_1+\cdots+\Delta t_k)/HL}$ 而不是
$2^{-\Delta t_k/HL}$。`reflect()` 调得越勤，概念死得越快——这直接阻止了
"高频 reflect"这种持续运行模式。

**修复。** 为 `ConceptNode`/`RelationEdge` 增加 `last_decayed_at`，衰减区间从
$\max(t_{act}, t_{decayed})$ 开始计算；宽限期内跳过时不更新参考点，因此未应用的
区间不会丢失（`test_grace_period_defers_but_does_not_lose_decay`）。

**问题 B — 成熟度阶梯不可达。** 设概念每隔 $g$ 小时被激活一次，激活后紧接衰减。
记 $f = 2^{-g/HL}$，则稳态（衰减后）为

$$c^* = \frac{b(n)\,f}{1-f}$$

- 每日使用、embryonic（$HL=24$h）：$f=0.5$，$c^* = b(n) \le 0.056 < 0.3$，
  **永远不能晋升到 developing**。实测中它之所以显示 `developing`，是因为先掉到
  FADING（$<0.05$）再被 `activate()` 的"复苏"直接改成 DEVELOPING，绕过了置信度门。
- developing（$HL=168$h）每日使用：$f = 0.906$，$c^* \approx 9.6\,b(n)$；
  $n=100$ 时 $b=0.0067$，$c^*\approx 0.064$（实测 0.059）。**established（0.6）
  不可达。**
- 每周使用：$f = 2^{-7} \approx 0.008$，$c^*\approx 0$ → FADING（实测 0.026）。

根因是"加性、随 $n$ 递减的增益"对抗"与证据量无关的乘性衰减"。系统中已经有
证据量（$n, d$）和 `evidence_balance()`（Beta 后验均值），但它们不参与衰减。

**修复（证据锚定衰减，OU 型均值回归）。**

$$HL_{\text{eff}}(v) = \min\!\big(8760\text{h},\; HL(m)\cdot\min(8,\; 1 + 0.2\,(n-1))\big)$$

$$\text{floor}(v) = 0.35\cdot \text{evidence\_balance}(v)\cdot\Big(\tfrac{n}{n+10}\Big)^2\cdot 2^{-\Delta t_{act}/4380\text{h}}$$

$$c \leftarrow \text{floor} + (c - \text{floor})\cdot 2^{-\Delta t/HL_{\text{eff}}}\quad(\text{仅当 } c > \text{floor})$$

- 半衰期随证据拉长（关系层早已如此：$1 + 0.5\,r$），并有 1 年的绝对上限——
  没有不朽的概念。
- 置信度向"证据地板"回归而不是向 0 回归。平方饱和使 $n\le 6$ 的地板低于
  FADING 阈值 0.05（噪声仍会消失并被剪枝），$n=30$ 时地板 ≈0.20，$n=100$ ≈0.29。
- 地板本身按"纪元"尺度（6 个月半衰期）遗忘：被确认 30 次后弃用的概念约 3 年
  后 FADING。否证通过 `evidence_balance` 压低地板。
- 单次确认的概念半衰期精确等于表中的名义值（$n-1=0$），既有衰减曲线测试全部
  保持通过。

新稳态 $c^* = \text{floor}(n) + b(n) f/(1-f)$，其中 $f$ 现在随 $n$ 增大而趋近 1。
标定结果见 §5。

**仍存的结构性问题（见 §7.1）：** `confidence` 同时承担"证据"和"当前显著性"
两种语义；`temporal_relevance()`（软新鲜度，1 周半衰期）与硬衰减作用在同一时钟
上，等效遗忘率是二者之和。这是有意设计（文档已说明），但意味着调参时二者必须
一起看。

### 3.2 关系概率与权重（`schemas/relation.py`、`dynamics/decay.py`）

`docs/relation-probability-redesign.md` 明确：`probability` 是"给定证据下该 typed
relation 正确的概率"，"不应被 token 共现驱动"，`weight/confidence` 是操作量。
但：

- `DecayEngine.decay_relations()` 在衰减后执行 `edge.probability = edge.confidence`，
  使 P(正确) 随闲置时间指数下降（30 天：0.70 → 0.008）。**时间流逝不是反证。**
- `RelationEdge.weaken()` 同样 `probability = confidence`。`confidence` 初始化为
  `structural_strength`（0.78），`probability` 初始化为 `propagation_strength`
  （0.70），两者不同尺度；Hebbian 复现又会把 confidence 推到 0.856，于是**一次
  否证让 probability 从 0.70 升到 0.80**。

**修复。** 衰减只作用于 `weight`/`confidence`；`weaken()` 对 `probability` 施加
同样的递减惩罚（$0.06/(1+0.1d)$）。`test_weaken_probability_tracks_confidence_*`
原本把错误行为固化成了断言，已改为断言"概率按惩罚下降"。

**未修改但需注意：** `RelationManager.adjust_strength()`（投影反馈回路使用）
仍有 `probability = confidence`；显式复观测一条关系但不带概率值时，
`discover()` 只 `reinforce()` 不更新概率——设计文档说"显式抽取证据应更新概率"，
这里是一个语义缺口（§7.3）。

### 3.3 扩散激活（`dynamics/activation.py`）

**问题 C — `max` 聚合。** 原实现对同一邻居只保留最强单路径：

```python
if propagated > old: activations[neighbor_id] = propagated
```

于是"两个种子都指向 C"与"只有一个种子指向 D"得分相同（实测均为 0.0467）。
对任务投影而言，**概念交集**（被多个激活源共同支持的概念）恰恰是最有价值的
信号。

**修复：有界 noisy-OR。** 令 $S$ 为最强种子分数，同一层内到达同一概念的贡献
$a_i$ 按

$$a = S\Big(1 - \prod_i \big(1 - \min(1, a_i/S)\big)\Big)$$

聚合。单路径时退化为原值；多路径单调递增；恒有 $a \le S$，因此**任何被传播到
的概念都不会超过最强种子**（`test_no_propagated_concept_outscores_strongest_seed`）。
6 个种子同时指向的目标得分是单路径的 2 倍以上。

**分层锁定。** 概念在首次到达的层固定，不再接受更深层的回流，种子层被冻结。
这保证：环路不会把激活反馈给自身（`test_cycle_does_not_inflate_seed`），
`record=True` 时每个概念的 `activation_count` 只加 1（原实现每次分数上升都调用
`activate()`，会虚增证据），扩展是有限的（每层每条边至多一次）。

**问题 D — 常数下限。** 原实现把低于 `3%·S` 的信号一律钳到 `3%·S`，于是深度 ≥3
的全部节点、以及"深度 3 的弱泛化边"与"深度 4 的强依赖边"都并列，MMR 只能靠
邻域重叠随机决定谁进投影。

**修复：保序下限带。** 对 $raw < F$（$F = 0.03\,S$）：

$$v = F\cdot\big(0.9 + 0.1\cdot raw/F\big) \in [0.9F, F)$$

严格单调、在 $raw = F$ 处连续，横向范围（0.9F）保证既有"3–4 跳视野"不退化，
而位置由 raw 决定：链上 $c_3 > c_4 > c_5 > c_6$ 严格成立（0.01617 > 0.01569 >
0.01564 > 0.01561）。

**任务亲和。** 原来是二值（匹配 → ×1.5）且用子串匹配。现在
$\gamma_{task} = 1 + 0.5\cdot\text{affinity}$，affinity 来自词级匹配（§3.5），
完全匹配时仍为 1.5，与全部既有基准测试兼容。

**一次性 `now`。** 原实现在内层循环里逐个调用 `datetime.now()`，
`temporal_relevance()` 因此带有与遍历顺序相关的 $10^{-10}$ 级噪声——这正是
探针 E 在 MMR 排序修复之后仍不确定的原因。`activate()`/`project()`/
`_build_coupling()` 现在各取一次 `now`。

### 3.4 投影（`projection/engine.py`）

- **确定性**：候选按 `(−score, id)` 排序后遍历，平局稳定；`selected` 以列表保持
  顺序（原来经 `set(selected)` 二次打乱）；关系按 `(−weight, id)` 排序（关系索引
  顺序依赖文件系统 `glob` 顺序）。探针：同一存储在 6 个 `PYTHONHASHSEED` 下
  渲染结果完全一致。
- **未修改**：冗余度量是邻居集合的无权 Jaccard，忽略关系类型与权重；候选筛选
  `min_activation=0.01` 是绝对值，而分数尺度随种子置信度变化。见 §7.5。

### 3.5 任务画像（`schemas/concept.py`）

- `reinforcement_log` 原来每次激活追加一条，永不截断，每次 flush 整文件重写。
  500 次激活 → 55 KB/概念。现在保留最近 64 条作为"近期活动窗口"，新增
  `task_profile: dict[str, int]`（规范化任务标签 → 次数，最多 64 个标签，按频次
  保留）作为任务关联的权威记录；旧记录加载时由 `model_validator` 从日志回填。
- `task_match_score(query, label)`：完全匹配 → 1；否则为 query 的签名词被 label
  覆盖的比例（`"ml serving"` vs `"ml training"` → 0.5）；无签名词（极短标签、
  中文）时回退到整串包含。`"ml"` 不再匹配 `"html parsing"`。
- `ConceptNode.task_affinity()` 的复杂度是 O(不同任务数)，与激活次数无关。
- `merge()` 合并 `task_profile` 并截断日志；`pkm.py` 概念卡片与可视化的
  `tasks` 字段改为读取画像。

### 3.6 Hebbian（`dynamics/hebbian.py`、`world/facade.py`）

- 计数器 `_cooccurrence` 现在通过 `snapshot()/restore()` 存入 `state.json`
  （`hebbian_pending`），`ingest()` 后按需保存；重启后第 2 次共现即建边。
  上限 50 000 对，FIFO 淘汰。
- `MAX_PAIRS=30` 原按 id 字典序截断，系统性偏向 id 最小的概念；现在按观察顺序
  截断（抽取通常按显著性排列），确定且更有认知意义。

### 3.7 生命周期（`dynamics/lifecycle.py`）

规则本身合理（动态 core 连接阈值是好设计）。问题全部来自 §3.1 的置信度动力学。
未修改的两点：

- 只有 `→ fading` 一种降级，core/established 不会因长期低置信度而软降级。
- FADING → DEVELOPING 的复苏绕过置信度门；在新地板机制下这条路径触发得少了，
  但它仍然是"晋升靠复苏而不是靠证据"的漏洞。建议在 §7.2 的复现分数就绪后收紧。

### 3.8 概念身份与整合（`concepts/_manager.py`）

- 身份键 = name + kind + domain + sense 的哈希，`Apple/公司` 与 `apple/水果`
  可共存——设计正确。
- `_find_synonym_match()` 对全部概念做 O(N) 扫描（每个带语义字段的候选各一次），
  ingest 一个 20 概念的观察在 10k 概念世界里是 200k 次比较。已有 `TokenIndex`
  可以先做候选短名单。见 §7.6。
- 签名 Jaccard 阈值 0.78、跨域 ×0.3 折扣：保守合理。

### 3.9 群落、色场、网络熵

- 标签传播 + 稳定性计数 + 结构优先生色，与 `world0-color-field-dynamics.md`
  的"命题 1/2（非负、有界）"一致；`fade_step` 的中位数归一化是稳健的。
- 只做了一处修改：`_build_coupling()` 取一次 `now`，避免耦合权重带顺序噪声
  影响标签传播的平局。
- 网络熵实现忠实于设计文档，作为只读诊断量没有发现问题。

### 3.10 存储与系统设计（`store/json_store.py`、`world/facade.py`）

- 每概念/关系一个 JSON 文件，启动 O(N) 读盘，flush 整文件重写；日志截断后单文件
  大小有界（9.7 KB vs 55 KB）。万级概念仍可用，十万级需要换后端（§7.7）。
- 跨周期状态只有 `last_reflect` 与 `communities`，Hebbian 状态是遗漏；已补。
- **幂等衰减是"持续运行"的使能条件**：现在可以在每 N 次 ingest 后安全地调用
  `reflect()`，或由外部定时器高频调用，而不会改变动力学。

---

## 4. 探针证据：修复前 / 后

| 探针 | 修复前 | 修复后 |
|------|--------|--------|
| A 48h 后 1×/3× 衰减（概念） | 0.3232 / 0.2175 | 0.3464 / 0.3464 |
| A 48h 后 1×/3× 衰减（关系权重） | 0.9194 / 0.7772 | 0.9194 / 0.9194 |
| B 每日使用（见 §5） | 120 天：developing, 0.059 | 90 天：established, 0.839（第 29 天 developing，第 58 天 established） |
| B 每周使用 | 120 天：**fading**, 0.026 | 365 天：developing, 0.358 |
| C 汇聚 C（两种子）vs D（单种子） | 0.0467 = 0.0467 | **0.0863** > 0.0467 |
| D 链 c3/c4/c5/c6/弱支 | 0.01733 ×5（全并列） | 0.01617 / 0.01569 / 0.01564 / 0.01561 / 0.01568 |
| E 6 个 PYTHONHASHSEED 下的投影 | 2 种不同结果 | 1 种 |
| F 重启后待定共现对 / 第 2 次共现建边 | 0 / 否 | 1 / 是 |
| G 30 天闲置后 P(关系) | 0.700 → 0.008 | 0.700 → 0.700（weight 0.008） |
| G 一次否证后 P(关系) | 0.70 → **0.80** | 0.70 → 0.65 |
| H 500 次激活：日志条数 / 文件大小 | 500 / 55.1 KB | 64 / 9.7 KB |
| H `"ml"` 匹配 `"html parsing"` | True | False |

---

## 5. 参数标定（`CONCEPT_EVIDENCE_HL_GAIN`）

模拟：概念每隔 `gap` 小时被激活一次（含一条 `depends_on` 关系），激活后推进时间、
执行 decay + lifecycle。`first_reached` 是首次达到某成熟度的天数。

| gain | 每 24h × 90d | 每 72h × 180d | 每 168h × 365d | 每 720h × 730d | 一次性 fading | 30× 后弃用：1 年 / 2 年 |
|------|--------------|---------------|----------------|----------------|---------------|--------------------------|
| 0.1 | established 0.62（dev 43d, est 87d） | developing 0.44 | developing 0.32 | developing 0.16 | 54h | 0.149 / 0.028 (fading) |
| **0.2** | **established 0.84（dev 29d, est 58d）** | **developing 0.52** | **developing 0.36** | **developing 0.17** | **54h** | **0.282 / 0.090** |
| 0.3 | established 0.94（dev 23d, est 47d） | developing 0.54 | developing 0.36 | developing 0.18 | 54h | 0.327 / 0.122 |

取 **0.2**：每日使用约一个月进入 developing、两个月 established；每周/每月使用
稳定为 developing 而不再消失；一次性概念两天后 fading；重度使用后弃用的概念在
1 年半衰期上限下约 3 年褪去（`test_burst_then_abandon_declines_slowly_but_surely`）。

该模型下**每周使用永远达不到 established（0.6）**——这是加性递减增益的固有
上界（$b(n)\to 0$），不是参数问题，见 §7.1/7.2。

---

## 6. 本次改动清单

| 文件 | 改动 |
|------|------|
| `schemas/concept.py` | `task_profile`、`last_decayed_at`、`MAX_REINFORCEMENT_LOG=64`、`normalize_task_label()`、`task_match_score()`、`record_task()`、`task_affinity()`、`decay_reference_time()`、`hours_since_activation(now)`/`temporal_relevance(now=)`、旧记录回填 |
| `schemas/relation.py` | `last_decayed_at`、`decay_reference_time()`、`task_affinity()`、`temporal_relevance(now=)`；`weaken()` 不再把 confidence 复制进 probability |
| `dynamics/decay.py` | 幂等衰减；`concept_half_life()`（证据缩放 + 1 年上限）；`evidence_floor()`（OU 均值回归目标，纪元遗忘）；关系衰减不触碰 `probability` |
| `dynamics/activation.py` | 有界 noisy-OR 聚合；分层锁定；保序下限带；分级任务增益；一次性 `now`；`record` 每概念一次 |
| `dynamics/hebbian.py` | `snapshot()/restore()/forget_concept()`；观察顺序截断；50k 上限 |
| `dynamics/community.py` | `_build_coupling()` 一次性 `now` |
| `projection/engine.py` | 确定性 MMR（排序遍历、列表保序、关系排序）；任务亲和改用画像 |
| `world/facade.py` | 恢复/持久化 Hebbian 状态（`state.json: hebbian_pending`） |
| `concepts/_identity_ops.py` | merge 合并 `task_profile`、截断日志、取最新 `last_decayed_at` |
| `agents/pkm.py`、`visualization/_graph_data.py` | `tasks` 读取画像 |
| `tests/test_dynamics_analysis.py` | 28 个行为测试（幂等、阶梯可达、地板、概率、画像、聚合、保序、环路、记录一次、跨进程确定性、Hebbian 持久化） |
| `tests/test_relation_axes_deep.py` | 修正把 bug 固化成断言的 `weaken` 测试 |

所有字段均有默认值，旧的 JSON 存储可直接加载（`task_profile` 自动回填）。

---

## 7. 结构性问题与建议路线

以下问题本轮**未改**，因为它们需要接口层面的设计决策；按对"持续动态更新的认知
系统"的重要性排序。

### 7.1 把 `confidence` 拆成"证据"与"显著性"

`confidence` 现在同时表示"这个概念是真的/有用的"（证据）和"它现在相关"
（显著性）。`evidence_balance()`（Beta 后验）和 `temporal_relevance()`
（新鲜度）已经分别是这两者的干净度量。建议：

```text
evidence(v)  = Beta 后验均值 × 饱和(n)         ——只被 activate/weaken 改变，慢速纪元遗忘
salience(v)  = temporal_relevance × 任务/视角亲和 ——只被时间与上下文改变
confidence   = evidence × g(salience)            ——派生量，供旧接口读取
```

成熟度门槛改为基于 `evidence`（加上 §7.2 的复现），衰减只作用于 `salience`。
这会消除 §3.1 的"双时钟"问题，也让 `render()` 里的 "confidence: 0.06" 不再误导
下游 Agent。

### 7.2 基于复现（recurrence）的晋升

`DCTMTemporalDynamics.md` §9.1 已经提出 recurrence score。在 `task_profile`
的基础上加一个按天/按 episode 去重的复现计数几乎是零成本的：

```text
recurrence(v) = |{ distinct days (or episodes) with an activation }|
embryonic → developing:  recurrence ≥ 3
developing → established: recurrence ≥ 10 ∧ evidence ≥ 0.6
```

这能让"每周用一次、用了一年"的概念正当地成为 established，同时天然免疫
"一天内刷 30 次"的突发噪声——正是 §5 里加性增益模型无法表达的区分。

### 7.3 关系的显式复观测应更新语义概率

`IngestPipeline._step_relations()` 对已存在的显式关系只 `reinforce()`（操作权重），
不更新 `probability`。建议：无概率值的显式复观测按
`update_probability(evidence_probability=spec.propagation_strength, evidence_strength=1.0)`
处理；Hebbian 复现保持不动。同时把 `adjust_strength()` 的 `probability = confidence`
改成独立的增量。

### 7.4 有向传播与视角

激活是无向遍历：`A depends_on B` 从 B 出发同样以全强度到达 A。对"debug"视角，
沿依赖方向（我依赖什么）与逆依赖方向（谁依赖我）的价值不同。建议在
`Perspective` 增加 `direction_bias: {semantic_relation: (forward, backward)}`，
默认 (1, 1) 保持现状。

### 7.5 投影冗余度量与相对阈值

- 冗余用加权 Jaccard（按 `weight × ρ_type`），使"通过一条强依赖边相连"与
  "通过一条弱泛化边相连"的重叠含义不同。
- 候选阈值改为相对值 `max(min_activation, 0.02·S)`，低置信种子也能得到完整视野；
  当前 0.01 绝对阈值意味着种子置信度 <0.33 时 3% 下限带会被整体裁掉。
- MMR 的 $\lambda=0.3$ 未经任何任务级评测标定；建议用
  `tests/_cognitive_benchmark.py` 的 precision/recall 做一次扫描。

### 7.6 身份解析索引化

`_find_synonym_match()` 改为先用 `TokenIndex.candidates()` 取短名单（同域、有
共享签名词），再打分；把 O(N) 降到 O(候选数)。这是万级概念世界 ingest 延迟的主要
来源。

### 7.7 存储层

当前 JSON-per-file 适合 <10k 概念。下一步建议 SQLite（单文件、事务、按 id 更新）
或 append-only 事件日志 + 周期快照；`Store` Protocol 已经把这一步隔离好了。

### 7.8 持续运行模式

有了幂等衰减，可以定义 `World.tick()`：每 N 次 ingest 或每 T 分钟执行一次轻量
`reflect()`（decay + lifecycle，不做群落检测），把群落/色场留给低频的完整
`reflect()`。这样"持续更新"就不再依赖使用者记得在任务结束时调用 `reflect()`。

### 7.9 边界提醒

`agents/` 目前 11.6k 行，是概念核心（concepts + relations + dynamics +
projection ≈ 2.8k 行）的 4 倍。AGENTS.md 明确警告不要滑向"伪装成认知系统的工作
流引擎"。建议后续把 agent 侧的会话/失败/恢复逻辑视为**相邻系统**维护，核心的
演进优先级放在 §7.1–7.5。

---

## 8. 复现

```bash
# 全部测试（含 28 个新行为测试）
./start_world0.sh test
# 或
python -m pytest -q tests/test_dynamics_analysis.py

# 标定扫描（§5 的表）
python scripts/calibrate_decay.py 0.1 0.2 0.3
```

`tests/test_dynamics_analysis.py` 顶部的 `_advance()` / `_simulate_cadence()`
就是探针与标定所用的时间模拟器：它把所有参考时间戳（包括 `last_decayed_at`）
整体回拨，再执行 `decay + lifecycle`。注意早期探针只回拨 `last_activated`，在新的
幂等衰减下会**低估**衰减（参考点被 `last_decayed_at` 接管），得出"置信度 1.000"
的假象——标定时务必回拨全部时间戳。
