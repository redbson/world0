# World 0 认知动力学分析与优化

> 对 World 0 基础设计（数学模型、结构、算法、系统）的深度分析，附实证探针、
> 修复方案、参数标定与后续路线。目标：让 World 0 成为一个**可持续运行、
> 动态更新**的 Agent 认知层，而不是只在演示脚本里工作一次的图结构。
>
> 复现方式见 §8。所有探针数字均来自本仓库代码在同一台机器上的实际运行。

---

## 0. 摘要

World 0 的概念框架（概念 / 关系 / 上下文 / 激活 / 投影）是清晰且自洽的，
模块边界（Protocol 驱动的 Lego 结构）也很好。但作为一个**随观察持续演化**的
动力系统，原实现有 9 处会在真实运行中失效的数学或系统缺陷：

| # | 问题 | 后果（修复前实测） | 状态 |
|---|------|--------------------|------|
| T | **时间基错误**：设计上时间应按观察次数计（times），实现却全部用墙钟小时 | 衰减取决于日历而不是 Agent 的认知活动量：一小时内处理 1000 条观察和一天处理 1 条，概念老化速度相同 | ✅ 已改为认知时钟（§3.0） |
| A | 衰减不幂等：每次 `reflect()` 都对"自上次激活以来的全部时长"再衰减一次 | 48 单位后调用 1 次衰减 → 0.323；同一瞬间调用 3 次 → 0.218 | ✅ 已修 |
| B | 成熟度阶梯不可达：加性递减增益 vs 乘性衰减的稳态太低 | 每 24 次观察复现一次、持续 2880 次：confidence 0.059，永远到不了 established；每 168 次复现一次：直接 FADING | ✅ 已修 + 标定 |
| C | 激活取 `max` 聚合：多路径汇聚不被奖励 | 被两个种子同时指向的概念与只被一个种子指向的概念得分完全相同 | ✅ 已修 |
| D | 常数传播下限抹平排序：深度 ≥3 的所有节点得分并列 | 链上 c3=c4=c5=c6=0.01733，弱泛化边与强依赖边同分 | ✅ 已修 |
| E | 投影跨进程不确定：循环内逐个取 `datetime.now()` 产生与文件遍历顺序相关的噪声；MMR 用 `set` 迭代 | 同一存储在不同 `PYTHONHASHSEED` 下投影结果不同 | ✅ 已修 |
| F | Hebbian 共现计数器只在内存中 | 重启后计数清零，跨会话共现永远达不到阈值 | ✅ 已修 |
| G | 关系语义概率随时间衰减（`probability = confidence`）；`weaken()` 把结构强度尺度复制进概率 | 720 单位闲置：P(关系正确) 0.70 → 0.008；一次否证反而让 probability 0.70 → 0.80 | ✅ 已修 |
| H | `reinforcement_log` 无上限，任务亲和是 O(log 长度) 的子串扫描 | 500 次激活 → 500 条日志、55 KB/概念文件；`"ml"` 匹配 `"html parsing"` | ✅ 已修 |

> 单位说明：修复前的实现以**小时**为单位，修复后以**观察次数（tick）**为单位；
> 半衰期等常数的数值原样保留（旧模型的 1 小时 ≡ 新模型的 1 次观察），因此
> 修复前后的数字可以直接对照。

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
- **回归**：全部 1014 个测试通过（原 978 + 新增 36 个行为测试
  `tests/test_dynamics_analysis.py`；`tests/test_temporal.py` 等按认知时钟重写）。

---

## 2. 系统模型的形式化

记概念 $v$ 的状态为 $(c_v, m_v, n_v, d_v, \tau_v)$：置信度、成熟度、激活次数、
否证次数、最近激活的认知时刻；关系 $e$ 的状态为 $(w_e, p_e, r_e, \tau_e)$：操作
权重、语义概率、强化次数、最近强化的认知时刻。系统由四个算子驱动：

```text
Ingest(O):    clock += 1 → get_or_create → activate(+boost) → discover/reinforce → hebbian → color seed
Activate(S):  分层扩散  a(v) = Agg_paths[ a(u)·w·ρ_type·max(c_v, 0.3)·δ^depth·γ_task·γ_domain·τ_rel·τ_node ]
Project(a):   MMR 选择  argmax (1−λ)·rel(v) − λ·max_{s∈sel} Jaccard(N(v), N(s))
Reflect():    decay → community → color → lifecycle → prune
```

其中 `activate()` 的增益为 $b(n) = 0.06/(1+0.08n)$；衰减为
$c \leftarrow c \cdot 2^{-\Delta t / HL(m)}$，$HL$ 只依赖成熟度
（embryonic 24、developing 168、established 720、core 2160，单位见 §3.0）。

一个"持续更新的认知系统"对这些算子至少要求：

0. **时间是认知时间**：算子以 Agent 处理了多少观察来计时，而不是日历。
1. **时间一致性**：算子的效果只依赖于经过的认知时间，不依赖调用频率（幂等）。
2. **证据可积累**：反复确认应带来持久性；一次性噪声应被遗忘。
3. **排序信息不丢失**：激活分数应保序，投影应对同一世界确定。
4. **学习状态可持久化**：所有跨调用的中间状态必须落盘。
5. **语义与操作量分离**：`P(关系正确)` 不应因时间流逝而变化。

原实现在 0–5 上都有缺口，下面逐模块分析。

---

## 3. 逐模块分析

### 3.0 认知时钟（`schemas/clock.py`）

**问题 T。** 设计上 World 0 的"时间"是**次数**：色场按 reflect 周期褪色
（`fade_step(dt=1)`）、群落稳定性按 reflect 次数计数，都体现了这一点；但概念/
关系的衰减、`temporal_relevance()`、激活与投影里的全部时间因子却用
`datetime.now()` 计小时。后果是概念老化与 Agent 的认知活动量无关：一个一小时
处理 1000 条观察的 Agent 和一个一天处理 1 条的 Agent，概念以同样的日历速度
消失；跨会话回来时，世界要么冻结（没有 reflect），要么因为日历流逝而被清空。

**修复：以 ingest 次数为主时钟，墙钟为次要漂移。** `World` 持有一个
`CognitiveClock`，每次 `ingest()` 前 `tick += 1`，并持久化在 `state.json`
（`tick`）。每条记录同时保存两套坐标：`*_tick`（认知时刻）与 `datetime`
（墙钟）。任意两个时刻之间经过的认知时间为

$$\Delta = \max(0,\, t_{now} - t_{since}) + \kappa \cdot \max(0,\, h_{now} - h_{since}),\qquad \kappa = 0.1\ \text{tick/h}$$

- 主项是观察次数差；第二项让**闲置**的世界仍缓慢老化：闲置一天 ≈ 2.4 tick，
  闲置一年 ≈ 876 tick——不到一周活跃观察的量（1 周 × 每天 100 条 ≈ 700 tick
  量级），因此日历只是休眠世界的补充项，不是主时钟。
- 所有半衰期、新鲜度常数的**数值不变**，单位由小时改为观察次数
  （`CONCEPT_HALF_LIFE`、`RELATION_BASE_HALF_LIFE`、`CONCEPT_TEMPORAL_HL`、
  `RELATION_TEMPORAL_HL`、`PROJECTION_TEMPORAL_HL`、`EVIDENCE_FLOOR_ERA_HL`、
  `CONCEPT_MAX_HALF_LIFE`）。
- 记录的时间戳由持有时钟的 `ConceptManager` / `RelationManager` /
  `ActivationEngine` 打上（`activate(tick=)`、`reinforce(tick=)`、
  `discovered_tick`、`created_tick`）；`DecayEngine`、`ProjectionEngine`、
  `CommunityDetector` 每次遍历只读一次 `clock.tick` 与 `now`。
- 旧存储无 `*_tick` 字段时默认为 0；世界时钟从 0 起，新的 ingest 才开始让
  旧概念按次数老化——正是期望的语义。
- `WorldStatus.cognitive_tick` 暴露当前认知时刻；`world.clock.advance(n)` 让 n 次
  观察"无事发生"地流逝，是测试与标定的时间模拟器（§8）。

**为什么不是 reflect 次数？** reflect 的调用时机由使用者决定；以它计时会让
衰减取决于调用者而不是认知活动。以 ingest 计时后，reflect 可以任意频繁地调用
（配合 §3.1 的幂等性），每次只结算已经过去的观察数。

### 3.1 置信度更新与衰减（`schemas/concept.py`、`dynamics/decay.py`）

**问题 A — 衰减不幂等。** 原实现：

```python
hours = node.hours_since_activation()
node.confidence *= 0.5 ** (hours / half_life)
```

参考点是最近激活时刻，因此在没有新激活的情况下，第 $k$ 次调用会再次乘以
$2^{-\Delta_k/HL}$，总衰减为 $2^{-(\Delta_1+\cdots+\Delta_k)/HL}$ 而不是
$2^{-\Delta_k/HL}$。`reflect()` 调得越勤，概念死得越快——这直接阻止了
"高频 reflect"这种持续运行模式。

**修复。** 为 `ConceptNode`/`RelationEdge` 增加 `last_decayed_tick` /
`last_decayed_at`，衰减区间从"最近激活"与"最近衰减"中较晚的时刻起算；同一观察内
再次调用时区间为 0 而被跳过，不更新参考点，因此未应用的区间不会丢失
（`test_grace_period_defers_but_does_not_lose_decay`、
`test_reflect_frequency_between_observations_is_irrelevant`）。

**问题 B — 成熟度阶梯不可达。** 设概念每隔 $g$ 次观察被复现一次，复现后紧接
衰减。记 $f = 2^{-g/HL}$，则稳态（衰减后）为

$$c^* = \frac{b(n)\,f}{1-f}$$

- 每 24 次观察复现、embryonic（$HL=24$）：$f=0.5$，$c^* = b(n) \le 0.056 < 0.3$，
  **永远不能晋升到 developing**。实测中它之所以显示 `developing`，是因为先掉到
  FADING（$<0.05$）再被 `activate()` 的"复苏"直接改成 DEVELOPING，绕过了置信度门。
- developing（$HL=168$）每 24 次复现：$f = 0.906$，$c^* \approx 9.6\,b(n)$；
  $n=100$ 时 $b=0.0067$，$c^*\approx 0.064$（实测 0.059）。**established（0.6）
  不可达。**
- 每 168 次复现：$f = 2^{-7} \approx 0.008$，$c^*\approx 0$ → FADING（实测 0.026）。

根因是"加性、随 $n$ 递减的增益"对抗"与证据量无关的乘性衰减"。系统中已经有
证据量（$n, d$）和 `evidence_balance()`（Beta 后验均值），但它们不参与衰减。

**修复（证据锚定衰减，OU 型均值回归）。**

$$HL_{\text{eff}}(v) = \min\!\big(8760,\; HL(m)\cdot\min(8,\; 1 + 0.2\,(n-1))\big)$$

$$\text{floor}(v) = 0.35\cdot \text{evidence\_balance}(v)\cdot\Big(\tfrac{n}{n+10}\Big)^2\cdot 2^{-\Delta_{act}/4380}$$

$$c \leftarrow \text{floor} + (c - \text{floor})\cdot 2^{-\Delta/HL_{\text{eff}}}\quad(\text{仅当 } c > \text{floor})$$

- 半衰期随证据拉长（关系层早已如此：$1 + 0.5\,r$），并有 8760 次观察的绝对上限——
  没有不朽的概念。
- 置信度向"证据地板"回归而不是向 0 回归。平方饱和使 $n\le 6$ 的地板低于
  FADING 阈值 0.05（噪声仍会消失并被剪枝），$n=30$ 时地板 ≈0.20，$n=100$ ≈0.29。
- 地板本身按"纪元"尺度（4380 次观察半衰期）遗忘：被确认 30 次后弃用的概念约
  26 000 次观察后 FADING。否证通过 `evidence_balance` 压低地板。
- 单次确认的概念半衰期精确等于表中的名义值（$n-1=0$），既有衰减曲线测试全部
  保持通过。

新稳态 $c^* = \text{floor}(n) + b(n) f/(1-f)$，其中 $f$ 现在随 $n$ 增大而趋近 1。
标定结果见 §5。

**仍存的结构性问题（见 §7.1）：** `confidence` 同时承担"证据"和"当前显著性"
两种语义；`temporal_relevance()`（软新鲜度，168 次观察半衰期）与硬衰减作用在
同一时钟上，等效遗忘率是二者之和。这是有意设计（文档已说明），但意味着调参时
二者必须一起看。**第六轮已处理**：传播 readiness 改读 `max(confidence, evidence)`，
新鲜度项改读 `salience()`（新鲜度 ∨ 证据持续性），见 §7.1。

### 3.2 关系概率与权重（`schemas/relation.py`、`dynamics/decay.py`）

`docs/relation-probability-redesign.md` 明确：`probability` 是"给定证据下该 typed
relation 正确的概率"，"不应被 token 共现驱动"，`weight/confidence` 是操作量。
但：

- `DecayEngine.decay_relations()` 在衰减后执行 `edge.probability = edge.confidence`，
  使 P(正确) 随闲置时间指数下降（720 次观察：0.70 → 0.008）。**时间流逝不是反证。**
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
`_build_coupling()` 现在各取一次 `(clock.tick, now)`。

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
  （`hebbian_pending`），与时钟一起在 `ingest()` 后按需保存；重启后第 2 次共现
  即建边。上限 50 000 对，FIFO 淘汰。
- `MAX_PAIRS=30` 原按 id 字典序截断，系统性偏向 id 最小的概念；现在按观察顺序
  截断（抽取通常按显著性排列），确定且更有认知意义。

### 3.7 生命周期（`dynamics/lifecycle.py`）

规则本身合理（动态 core 连接阈值是好设计）。问题全部来自 §3.1 的置信度动力学。

- 只有 `→ fading` 一种降级，core/established 不会因长期低置信度而软降级（未改）。
- FADING → DEVELOPING 的复苏原本绕过置信度门；第二轮起复苏只在
  `recurrence_count ≥ 3` 时回到 DEVELOPING，否则回到 EMBRYONIC（§7.2）。

### 3.8 概念身份与整合（`concepts/_manager.py`）

- 身份键 = name + kind + domain + sense 的哈希，`Apple/公司` 与 `apple/水果`
  可共存——设计正确。
- `_find_synonym_match()` 对全部概念做 O(N) 扫描（每个带语义字段的候选各一次），
  ingest 一个 20 概念的观察在 10k 概念世界里是 200k 次比较。已有 `TokenIndex`
  可以先做候选短名单。见 §7.6。
- 签名 Jaccard 阈值 0.78、跨域 ×0.3 折扣：保守合理。

### 3.9 群落、色场、网络熵

- 标签传播 + 稳定性计数 + 结构优先生色，与 `world0-color-field-dynamics.md`
  的"命题 1/2（非负、有界）"一致；`fade_step` 的中位数归一化是稳健的。这两个
  子系统本来就以 reflect 周期计时，与认知时钟的精神一致，未改动。
- 只做了一处修改：`_build_coupling()` 取一次 `(tick, now)`，避免耦合权重带顺序
  噪声影响标签传播的平局。
- 网络熵实现忠实于设计文档，作为只读诊断量没有发现问题。

### 3.10 存储与系统设计（`store/json_store.py`、`world/facade.py`）

- 每概念/关系一个 JSON 文件，启动 O(N) 读盘，flush 整文件重写；日志截断后单文件
  大小有界（9.7 KB vs 55 KB）。万级概念仍可用，十万级需要换后端（§7.7）。
- 跨周期状态原来只有 `last_reflect` 与 `communities`：时钟与 Hebbian 状态都是
  遗漏；已补（`tick`、`last_reflect_tick`、`hebbian_pending`）。
- **认知时钟 + 幂等衰减是"持续运行"的使能条件**：现在可以在每 N 次 ingest 后
  安全地调用 `reflect()`，或由外部定时器高频调用，而不会改变动力学。

---

## 4. 探针证据：修复前 / 后

单位：修复前为小时，修复后为观察次数（常数 1:1 对应，见 §0 注）。

| 探针 | 修复前 | 修复后 |
|------|--------|--------|
| A 48 单位后 1×/3× 衰减（概念） | 0.3232 / 0.2175 | 0.3464 / 0.3464 |
| A 48 单位后 1×/3× 衰减（关系权重） | 0.9194 / 0.7772 | 0.9194 / 0.9194 |
| B 每 24 次复现（见 §5） | 2880 次后：developing, 0.059 | 2160 次后：established, 0.86（tick 672 developing，tick 1344 established） |
| B 每 168 次复现 | 2880 次后：**fading**, 0.026 | 8760 次后：developing, 0.359 |
| C 汇聚 C（两种子）vs D（单种子） | 0.0467 = 0.0467 | **0.0863** > 0.0467 |
| D 链 c3/c4/c5/c6/弱支 | 0.01733 ×5（全并列） | 0.01617 / 0.01569 / 0.01564 / 0.01561 / 0.01568 |
| E 6 个 PYTHONHASHSEED 下的投影 | 2 种不同结果 | 1 种 |
| F 重启后待定共现对 / 第 2 次共现建边 | 0 / 否 | 1 / 是 |
| G 720 单位闲置后 P(关系) | 0.700 → 0.008 | 0.700 → 0.700（weight 0.008） |
| G 一次否证后 P(关系) | 0.70 → **0.80** | 0.70 → 0.65 |
| H 500 次激活：日志条数 / 文件大小 | 500 / 55.1 KB | 64 / 9.7 KB |
| H `"ml"` 匹配 `"html parsing"` | True | False |
| T 一周墙钟无观察 vs 168 次观察 | 二者衰减相同 | 闲置仅漂移 16.8 tick；168 次观察衰减明显更多 |

---

## 5. 参数标定（`CONCEPT_EVIDENCE_HL_GAIN`）

模拟：概念每隔 `gap` 次观察被复现一次（含一条 `depends_on` 关系），其间用
`world.clock.advance()` 让其他观察流逝，每次复现后执行 decay + lifecycle。
`first_reached` 是首次达到某成熟度的 tick。

| gain | 每 24 × 2160 | 每 72 × 4320 | 每 168 × 8760 | 每 720 × 17520 | 一次性 fading | 30× 后弃用：8760 / 17520 / 26280 |
|------|--------------|--------------|---------------|----------------|---------------|----------------------------------|
| 0.1 | established 0.62（dev 1032, est 2088） | developing 0.44 | developing 0.32 | developing 0.16 | 54 | 0.149 / 0.028 (fading) / — |
| **0.2** | **established 0.86（dev 672, est 1344）** | **developing 0.52** | **developing 0.36** | **developing 0.17** | **54** | **0.282 / 0.090 / 0.028 (fading)** |
| 0.3 | established 0.94（dev 552, est 1128） | developing 0.54 | developing 0.36 | developing 0.18 | 54 | 0.327 / 0.122 / — |

取 **0.2**：每 24 次观察复现一次的概念约 700 次观察后进入 developing、1400 次
后 established；每 168 / 720 次复现的概念稳定为 developing 而不再消失；一次性
概念约 54 次观察后 fading；重度使用后弃用的概念在 8760 的半衰期上限下约
26 000 次观察后褪去（`test_burst_then_abandon_declines_slowly_but_surely`）。

该模型下**每 168 次复现永远达不到 established（0.6）**——这是加性递减增益的
固有上界（$b(n)\to 0$），不是参数问题，见 §7.1/7.2。

> 注意：这张表的模拟只执行 decay + lifecycle，不执行 prune。第十三轮发现真实流水线
> 会在第一个长间隔里把概念删掉（§7.16），已用剪枝宽限期修正；表中的稳态数值本身不受影响。

> 这些常数的"合理性"取决于 Agent 的观察粒度：如果一次 ingest 是一轮对话，
> 24 次观察大约是一次工作会话；如果一次 ingest 是一篇文档，则是一天的阅读量。
> 需要更快/更慢的遗忘时，调 `CONCEPT_HALF_LIFE` 的整体尺度即可，无需碰墙钟。

---

## 6. 本次改动清单

| 文件 | 改动 |
|------|------|
| `schemas/clock.py`（新） | `CognitiveClock`、`cognitive_elapsed()`、`WALL_TICKS_PER_HOUR=0.1` |
| `schemas/concept.py` | `created_tick`、`last_activated_tick`、`last_decayed_tick/_at`、`task_profile`、`MAX_REINFORCEMENT_LOG=64`、`normalize_task_label()`、`task_match_score()`、`record_task()`、`task_affinity()`、`elapsed_since_activation()`、`decay_elapsed()`、`temporal_relevance(half_life, now_tick=, now=)`、`activate(tick=)`、旧记录回填 |
| `schemas/relation.py` | `discovered_tick`、`last_reinforced_tick`、`last_decayed_tick/_at`、`reinforce(tick=)`、`update_probability(tick=)`、`elapsed_since_reinforced()`、`decay_elapsed()`、`task_affinity()`、`temporal_relevance(half_life, now_tick=, now=)`；`weaken()` 不再把 confidence 复制进 probability |
| `schemas/types.py` | `WorldStatus.cognitive_tick` |
| `concepts/_manager.py`、`relations/manager.py` | 持有时钟；创建/激活/发现/强化时打 tick |
| `dynamics/decay.py` | 认知时间上的幂等衰减；`concept_half_life()`（证据缩放 + 上限）；`evidence_floor()`（OU 均值回归目标，纪元遗忘）；关系衰减不触碰 `probability` |
| `dynamics/activation.py` | 有界 noisy-OR 聚合；分层锁定；保序下限带；分级任务增益；一次性 `(tick, now)`；`record` 每概念一次并打 tick |
| `dynamics/hebbian.py` | `snapshot()/restore()/forget_concept()`；观察顺序截断；50k 上限 |
| `dynamics/community.py`、`dynamics/coefficients.py` | 时钟；一次性 `(tick, now)`；常数单位改为观察次数 |
| `projection/engine.py` | 确定性 MMR（排序遍历、列表保序、关系排序）；任务亲和改用画像；时钟 |
| `world/facade.py`、`world/_status.py` | 创建/持久化时钟（`state.json: tick`），`ingest()` 推进时钟；恢复/持久化 Hebbian 状态；`World.clock`；状态含 `cognitive_tick` |
| `concepts/_identity_ops.py` | merge 合并 `task_profile` 与 tick 坐标、截断日志 |
| `agents/pkm.py`、`visualization/_graph_data.py` | `tasks` 读取画像 |
| `tests/test_dynamics_analysis.py` | 36 个行为测试（时钟、幂等、阶梯可达、地板、概率、画像、聚合、保序、环路、记录一次、跨进程确定性、Hebbian 持久化） |
| `tests/test_temporal.py` 等 | 时间模拟改为 `world.clock.advance(n)` / `*_tick` 坐标 |
| `scripts/calibrate_decay.py` | §5 的标定扫描 |
| **第二轮（§7 落地）** | |
| `schemas/concept.py`、`dynamics/lifecycle.py` | `recurrence_count` / `last_recurrence_window`、复现晋升门、复苏收紧 |
| `schemas/relation.py`、`world/_ingest.py`、`relations/manager.py` | `RelationEdge.confirm()`、显式复观测调用、`adjust_strength` 概率增量 |
| `schemas/context.py`、`dynamics/activation.py` | `Perspective.direction_weights` 与有向传播 |
| `concepts/_manager.py`、`concepts/_indexes.py` | 同义匹配短名单；索引 sense 词 |
| `world/_reflect.py`、`world/facade.py` | `reflect(light=)`、`auto_reflect_every` |
| `tests/test_roadmap_dynamics.py` | 22 个行为测试 |
| **第三轮** | |
| `concepts/_manager.py`、`concepts/_indexes.py` | 稀有词短名单 + `NameIndex.ids_for`、签名缓存、探针签名只算一次 |
| `dynamics/activation.py`、`projection/engine.py` | 相对接受阈值 `min(min_activation, 0.02·S)`；加权 Jaccard 实测否定后保留普通 Jaccard |
| `schemas/concept.py`、`schemas/types.py` | `evidence()`、`salience()`；`render()` 同时输出 evidence |
| `tests/test_roadmap_dynamics.py` | +6 个测试（弱种子视野、证据/显著性独立、常见词短名单、缓存失效） |
| **第四轮** | |
| `schemas/concept.py` | `tokenize_signature` 保留纯数字 token（"GPT 4" ≠ "GPT 5"） |
| `projection/engine.py` | `MMR_LAMBDA` 0.3 → 0.5（同族场景标定） |
| `tests/test_roadmap_dynamics.py` | +5 个测试（数字身份、投影跨区域覆盖） |
| **第五轮** | |
| `store/sqlite_store.py`、`store/__init__.py`、`world/facade.py` | SQLite 后端；`World(backend=)` 选择（后缀自动识别） |
| `tests/test_sqlite_store.py` | 9 个测试（契约、重启一致性、剪枝、flush 成本对比） |
| **第六轮** | |
| `schemas/concept.py` | `salience()` = 新鲜度 ∨ `SALIENCE_EVIDENCE_SHARE·evidence()·2^(−Δ/SALIENCE_ERA_HL)`；`dynamics/decay.py` 共用同一纪元常数 |
| `dynamics/activation.py`、`projection/engine.py` | readiness = `max(confidence, evidence, 0.3)`；新鲜度项改读 `salience()` |
| `tests/test_roadmap_dynamics.py` | +6 个测试（休眠依赖占位、新语境仍胜、否证降低传播、噪声不持续、纪元遗忘、边界） |
| `tests/test_layer_boundaries.py`（新） | AST 级层边界测试：核心包不得导入 agents/llm/extraction/…（§7.9） |
| `scripts/sweep_salience.py` | §7.1 的标定扫描 |
| `tests/test_projection_stability.py`（新） | 投影稳定性回归（§7.10） |
| **第七轮** | |
| `schemas/context.py` | `relation_type_weights` 语义级键 + 构造校验；`weight_for(axis, default, semantic_relation)` |
| `schemas/relation.py` | `is_known_relation_label()`、`RelationEdge.is_directed` |
| `dynamics/activation.py` | 语义级类型系数；方向系数只作用于有向边 |
| `perspectives/__init__.py`（新） | 命名画像 `PROFILES`、`get_perspective()`、`perspective_names()` |
| `world/facade.py` | `project(perspective="<profile>")` |
| `tests/test_perspective.py` | +10 个测试 |
| **第八轮** | |
| `dynamics/hebbian.py` | 观察/提及统计、`HEBBIAN_MIN_ASSOCIATION = 0.2` 关联门、`association()`、`stats_snapshot()/restore_stats()` |
| `world/facade.py` | 持久化 `hebbian_stats` |
| `scripts/sweep_hebbian.py` | §7.11 的阈值扫描 |
| `dynamics/hebbian.py`、`world/_reflect.py`、`schemas/types.py`、`core/interfaces.py` | `revalidate()`（第 5b 步）、`ReflectResult.stale_relations`、`HebbianLearner.revalidate` |
| `tests/test_roadmap_dynamics.py` | +11 个测试 |
| **第九轮** | |
| `projection/engine.py` | 任务感知冗余：`redundancy = max(sim, 1 − affinity)`（仅当仍有更贴合任务的候选） |
| `tests/test_context_drift.py`（新） | 概念漂移下的上下文（4 个测试）；基准 ML/Ops 精度 0.67/0.83 → 1.00/1.00 |
| **第十轮** | |
| `projection/engine.py`、`core/interfaces.py`、`world/facade.py` | `project(..., seed_ids=)`：种子先选、不受阈值过滤 |
| `tests/test_projection_seeds.py`（新） | 4 个测试 |
| **第十一轮** | |
| `dynamics/decay.py` | `relation_floor()`、`RELATION_FLOOR_SHARE = 0.1`；显式关系向概率地板回归 |
| `tests/test_roadmap_dynamics.py` | +4 个测试 |
| **第十二轮** | |
| `dynamics/hebbian.py` | 规范字符串键；`tracked_concepts` |
| `store/base.py`、`core/interfaces.py`、`store/json_store.py`、`store/sqlite_store.py`、`core/test_doubles.py` | `save_learning_state()/load_learning_state()` |
| `world/facade.py` | 摊销持久化、旧格式迁移、`close()`/上下文管理器 |
| `tests/test_learning_state.py`（新） | 7 个测试 |
| **第十三轮** | |
| `dynamics/decay.py` | `PRUNE_MIN_IDLE_TICKS = 720`；`prune_concepts()` 增加闲置条件 |
| `tests/test_roadmap_dynamics.py` | +5 个测试 |

所有字段均有默认值，旧的 JSON 存储可直接加载（`task_profile` 自动回填，tick 与
recurrence 默认 0）。

---

## 7. 结构性问题与建议路线

按对"持续动态更新的认知系统"的重要性排序。第二轮实现了 §7.2、§7.3、§7.4、
§7.6、§7.8，第三、四轮完成 §7.5（相对阈值，加权 Jaccard 实测否定），第五轮
§7.7（SQLite 后端），第六轮 §7.1（单时钟显著性）并把 §7.9 变成测试。标注 ✅ 的
条目都有行为测试（`tests/test_roadmap_dynamics.py`、`tests/test_sqlite_store.py`、
`tests/test_layer_boundaries.py`）。

### 7.1 把 `confidence` 拆成"证据"与"显著性" ✅（单时钟）

`confidence` 同时表示"这个概念是真的/有用的"（证据）和"它现在相关"（显著性）。
第三轮把两者作为只读派生量暴露：`ConceptNode.evidence()` = Beta 后验均值 ×
$n/(n+10)$（与时间无关）；`salience()` 当时只是 `temporal_relevance` 的别名。

**第六轮探针**发现这带来了"双时钟"的实际后果：传播权重同时乘
`max(confidence, 0.3)`（confidence 已衰减）与 `temporal_relevance`（下限 0.1），
一个被确认 50 次、随后休眠的依赖在其种子的邻域里得分只有**同一次观察刚提到的
一次性概念的 0.2 倍**，在名额受限的投影（4 个名额、6 个新概念）中永远进不去——
而它的 `evidence()` 仍是 0.82。

**修复：** 时间在一次传播中只计一次。

```text
readiness(v) = max(confidence, evidence(v), 0.3)         ——"是不是真概念"，与时间无关
salience(v)  = max(freshness, 0.7·evidence(v)·2^(−Δ/4380)) ——"现在是否在场"
```

`salience()` 成为真正的量：新鲜度**或**证据支撑的持续性，二者取大；持续性按
与置信度地板相同的纪元半衰期（`SALIENCE_ERA_HL = EVIDENCE_FLOOR_ERA_HL = 4380`）
遗忘，因此信念的两半以同一速率忘记深历史。激活与投影的新鲜度项都改读
`salience()`；`temporal_relevance()` 保持纯时间量（群落耦合仍用它）。一次性概念
的证据 ≈ 0.06，永远抬不过 0.1 的新鲜度下限——噪声不会因此持续。

`scripts/sweep_salience.py` 的标定（认知基准在所有取值下 ML 0.67/0.67、
Ops 0.83/0.83 不变）：

| 方案 | Δ500 老兵/新兵 | Δ1000 | Δ3000 | Δ8000 | Δ500 时进 4 名额投影 |
|---|---|---|---|---|---|
| 修复前 | 0.22 | 0.21 | 0.21 | 0.20 | 否 |
| 仅 readiness | 0.34 | 0.25 | 0.22 | 0.21 | 否 |
| share 0.5 | 1.14 | 0.87 | 0.30 | 0.21 | 否（投影再乘一次新鲜度） |
| **share 0.7** | **1.59** | **1.22** | **0.41** | **0.21** | **是（第 2 位）** |
| share 1.0 | 2.28 | 1.74 | 0.59 | 0.21 | 是 |

取 **0.7**：被反复确认的依赖在休眠约 500 次观察内仍占据种子投影的前排，约
1 000 次时与新鲜的一次性提及打平，之后让位给新语境（`test_fresh_context_still_wins_after_long_dormancy`）——
"上下文改变相关性"仍然成立，只是不再把已建立的知识在一千次观察后从邻域里抹掉。
投影层的新鲜度混合（`0.7 + 0.3·salience`）保留：它有界，且是种子唯一的时间项。

### 7.2 基于复现（recurrence）的晋升 ✅

`DCTMTemporalDynamics.md` §9.1 提出的 recurrence score 已实现：激活按
`RECURRENCE_WINDOW = 24` tick 的窗口（"认知日"）去重，
`ConceptNode.recurrence_count` 记录**不同窗口**的激活次数——一个窗口内刷 30 次
只算 1 次，每窗口 1 次持续 30 个窗口算 30 次。生命周期的两级晋升都增加了复现
门（任一门满足即可）：

```text
embryonic → developing:   n ≥ 3 ∧ c ≥ 0.3      或  recurrence ≥ 3  ∧ c ≥ 0.15
developing → established: n ≥ 10 ∧ c ≥ 0.6     或  recurrence ≥ 10 ∧ c ≥ 0.3
```

每 168 次观察复现一次的概念现在在第 10 次复现后成为 established
（`test_sparse_but_regular_concept_reaches_established`），而突发式的 30 次
刷屏不会触发复现门。同时收紧了 §3.7 指出的复苏漏洞：FADING 概念被重新激活时，
只有 `recurrence_count ≥ 3` 才回到 DEVELOPING，否则回到 EMBRYONIC 从头爬梯子。
合并时取两者复现数的最大值（不同窗口无法事后恢复）。

### 7.3 关系的显式复观测更新语义概率 ✅

`RelationEdge.confirm()`：Agent/抽取器再次**显式陈述**一条 typed relation 时，
$p \leftarrow p + (1-p)\cdot 0.05$（递减收益，20 次把 0.70 推到 ≈0.89），并计入
`probability_observation_count`；`IngestPipeline._step_relations()` 对无概率值的
显式复观测同时调用 `reinforce()`（操作权重）与 `confirm()`（语义概率）。Hebbian
共现只走 `reinforce()`，不碰概率（`test_hebbian_cooccurrence_does_not_touch_probability`）。
`RelationManager.adjust_strength()` 改为按 `confidence_delta` 增量移动概率，不再
把 confidence 尺度复制进去。

### 7.4 有向传播与视角 ✅（第七轮：语义级权重、方向只作用于有向关系、命名画像）

`Perspective.direction_weights = {"forward": …, "backward": …}`：forward 是沿
关系 source→target 遍历（`A depends_on B` 从 A 出发："我依赖什么"），backward
相反（"谁依赖我"）。缺省 1.0，默认视角保持无向。

**第七轮探针**在认知基准世界上逐个变体检查"视角是否真的改变投影"：

| 变体 | 修复前 | 修复后 |
|---|---|---|
| `direction_weights={"forward": 0.2}` | 所有邻居等比缩小 0.2，**投影不变**（种子的 Hebbian 平行边也被按"方向"缩放） | 只有 depends_on 邻居缩小；投影从 PyTorch/deployment/FastAPI 换成 optimizer/gradient descent/… |
| `relation_type_weights={"co_occurs": 0.05}` | 键不在三轴词表里，**静默无效** | 构造时即拒绝（`ValueError`，列出可用标签） |
| `relation_type_weights={"positive": 0.05}` | 生效 | 生效（不变） |
| 语义级键 `{"dependence": 1.4, "inclusion": 0.3}` | 不支持——dependence 与 inclusion 同属 positive 轴，无法区分 | 语义键优先于轴键；别名（`depends_on`、`contains`）等价 |

三处改动：

1. `Perspective.weight_for(axis, default, semantic_relation)`：查找顺序为语义名
   （原键 → 规范名）→ 轴 → 默认；`relation_type_weights` 的键在构造时用
   `is_known_relation_label()` 校验。
2. `RelationEdge.is_directed`（positive/negative 为有向，parallel 对称）；激活引擎
   只对有向边乘方向系数——平行边的 source→target 只是存储顺序，不是语义。
3. 新增 `world0.perspectives`（AGENTS.md 建议的 `perspectives/` 模块）：
   `dependency_map`（我依赖什么）、`impact_map`（谁依赖我，同一组边反向读）、
   `taxonomy`（归属/包含）、`analogy`（共振）、`contrast`（冲突）、`default`；
   `World.project(seeds, perspective="taxonomy", task=…)` 直接按名字使用。画像
   刻意少而明确——视角是一种有文档的阅读策略，不是调参旋钮。

`tests/test_perspective.py` 新增 10 个测试：同轴语义区分、别名等价、语义键优先、
未知键拒绝、平行边不受方向影响、`is_directed`、画像可解析、依赖图与影响图对同一
组边反向排序、分类与类比前景不同、命名画像保留 task。

### 7.5 投影冗余度量与相对阈值（相对阈值 ✅，加权 Jaccard ✗ 已实测否定）

- **相对阈值 ✅**：激活与投影的接受阈值都改为
  `min(min_activation, 0.02·S)`（S = 最强种子分数）——只会放松绝对阈值、
  从不收紧。置信度 ≈0.2 的一次性种子现在与置信种子一样保留 4 跳视野
  （`test_weak_seed_keeps_its_horizon`），基准测试不受影响。
- **加权 Jaccard ✗**：按 `weight × ρ_type` 的加权冗余在认知基准上实测**没有
  收益**：λ=0.3 时只是把 ML/Ops 的精度互换（0.67/0.83 → 0.83/0.67），λ≥0.4
  两边都掉到 0.67。因此保留普通集合 Jaccard，代码里留了说明。
- **λ 扫描**（第三轮时的数字；第九轮的任务感知冗余之后基准在所有 λ 下都是 1.0，见 §7.12）：普通 Jaccard 下 λ ∈ {0.1…0.5} 对认知基准**完全不敏感**
  （ML 0.67/0.67、Ops 0.83/0.83、交集 4 恒定）——该基准由相关性主导，冗余项
  没有信号。于是构造了一个**对冗余敏感**的场景：hub 下挂 6 个共享同两个锚点的
  近似同族概念，另有一条 3 个概念的链（`TestProjectionDiversity`）。6 个名额下：

  | λ | 同族概念 | 链上概念 |
  |---|---|---|
  | 0.0（纯相关性） | 4 | 1 |
  | 0.1 / 0.3（原默认） | 3 | 1 |
  | **0.5** | **2** | **3** |
  | 0.7 | 2 | 3 |

  λ=0.5 起投影才同时覆盖两个区域；认知基准在该值下不变，全套测试通过，因此
  **默认 λ 改为 0.5**，并把该场景固化为回归测试。扫描脚本见 §8。

### 7.6 身份解析索引化 ✅

`_find_synonym_match()` 现在只对短名单打分：**精确标签命中**（`NameIndex`，覆盖
词面重叠路径）∪ **最稀有的 40%（至少 3 个）探针词的倒排列表**（覆盖签名重叠
路径——真同义词必须共享 ≥78% 的词，因此必然共享最稀有的那几个）。仅按"任一
共享词"取短名单在真实语料里会退化为全量扫描（"system"、"data"这类词几乎在
每个概念里出现），剖析显示 600 次插入触发了 179 700 次打分、每次都重新分词。
现在每个概念的签名（标签键、sense+description 词集、规范化 sense）按派生字段
构成的键缓存，探针侧签名对整个短名单只算一次；探针完全不可分词（纯中文标签且
无描述）时回退到全量扫描。候选按 `(created_tick, id)` 排序，平局结果不变。

实测（`kind`/`sense` 语义候选，同一台机器）：

| 规模 | 操作 | 修复前（O(N²)） | 修复后 |
|------|------|-----------------|--------|
| N=1000 | 建 1000 个概念 | 16.8 s | **0.14 s** |
| N=1000 | 摄入 50×10 个语义候选 | 23.6 s | **0.55 s** |
| N=5000 | 建 5000 个概念 | （未等完） | **2.4 s** |
| N=5000 | 摄入 50×10 个语义候选 | — | **1.65 s** |

顺带发现并修复了一个既有的身份风险：`tokenize_signature` 原来丢弃所有单字符
token，因此"GPT 4"/"GPT 5"这类只靠一个数字区分、其余描述相同的概念会被判为
同义并合并。现在纯数字 token 不受长度限制（"Python 3"/"Python3" 这类真同义仍
通过别名合并，`TestNumericIdentityTokens`）。

### 7.7 存储层 ✅（SQLite 后端）

`store/sqlite_store.py`：单文件 SQLite（WAL），四张 `(id, payload)` 表存放与
JSON 后端完全相同的 pydantic 载荷；一次 flush 的 N 条脏记录是**一个事务内的
N 次 upsert**，加载是每表一次顺序扫描。`World(store_path, backend=)` 选择后端：
`"auto"`（默认）在路径后缀为 `.sqlite/.sqlite3/.db` 时用 SQLite，否则保持
JSON-per-file，因此既有存储不受影响。实测 60 次 ingest × 40 个概念的 flush
总成本：JSON 0.59 s，SQLite 0.23 s。`test_sqlite_store.py` 覆盖 Store 契约、
World 全周期重启一致性（时钟、Hebbian 待定对、投影渲染逐字相同）与剪枝删行。
append-only 事件日志 + 周期快照仍是更远的选项。

### 7.8 持续运行模式 ✅

`World.reflect(light=True)` 只执行 decay + lifecycle + prune，跳过群落检测与
色场；`World(store_path, auto_reflect_every=N)` 让 `ingest()` 每 N 次观察自动
执行一次轻量 reflect（`test_auto_reflect_every_consolidates_without_explicit_calls`）。
因为衰减对认知时间幂等，自动周期只决定"何时结算"，不改变遗忘量；群落/色场仍留给
显式的完整 `reflect()`（只有完整 reflect 才更新 `last_reflect`）。
`DCTMTemporalDynamics.md` 中的 episode / phase 层可以直接建立在 tick 之上
（episode = tick 区间）。

### 7.9 边界提醒

`agents/` 目前 11.6k 行，是概念核心（concepts + relations + dynamics +
projection ≈ 2.9k 行）的 4 倍。AGENTS.md 明确警告不要滑向"伪装成认知系统的工作
流引擎"。建议后续把 agent 侧的会话/失败/恢复逻辑视为**相邻系统**维护，核心的
演进优先级放在 §7.1–7.5。

第六轮把这条边界从提醒变成测试：`tests/test_layer_boundaries.py` 用 AST 遍历
`schemas / core / concepts / relations / dynamics / projection / communities /
store / metrics` 的每个模块，断言它们不导入 `agents / llm / extraction / prompts /
models / visualization / world / spaces`；`world/`（门面）可以用 extraction 与
prompts（文本摄入需要抽取器），但不得导入 agents；仓库内没有任何包导入
`agents`。新增顶层包必须先在该测试里归类，否则测试失败。当前依赖图完全满足。

### 7.10 投影稳定性（探针，负结果 = 稳定）✅

AGENTS.md 要求测试"投影稳定性"。第六轮在认知基准世界上做了四组扰动，投影
（`model serving` / `ml training` / 6 名额 / 深度 4）的**有序**结果：

| 扰动 | 结果 |
|---|---|
| 逐条摄入 12 条无关观察（weather / cooking） | 12 次全部逐字相同 |
| 一次域内复提（PyTorch / optimizer / gradient descent / monitoring / model serving） | 只有 gradient descent 与 optimizer 互换（二者激活分严格相等 0.1732，按 id 平局） |
| 闲置 0 / 1 / 5 / 20 / 100 tick 后 `reflect()` | 全部相同 |
| 交替摄入两个无关概念 20 次 | 相邻投影变化 0 次、只出现 1 种结果 |

第 5 项探针显示纳入/排除边界上是一个四路平局（optimizer、gradient descent、
neural network、training pipeline 都是 0.1732）——这是基准世界的 Hebbian 全连接
结构造成的真实对称，不是数值噪声；平局由 `(−score, id)` 决定性打破。
`tests/test_projection_stability.py` 把前四项固化为回归测试。

### 7.11 Hebbian 发现的关联门（防止泛化边蔓延）✅

**第八轮探针**：60 个概念、每次观察随机取 6 个、400 次观察（无结构的对照世界）。
只有"共现 ≥ 2 次"这一道门时：

| 观察数 | 关系数 | 其中 generic_relation | 平均度 / 最大度 |
|---|---|---|---|
| 50 | 138 | 126 | 4.7 / 15 |
| 100 | 393 | 378 | 13.1 / 26 |
| 200 | 901 | 874 | 30.0 / 48 |
| 400 | 1503 | 1468（全部 1770 对的 83%） | 50.1 / 58 |

一次 `reflect()` 只剪掉 73 条；投影里 18 条关系全是 generic_relation，显式声明的
35 条 depends_on 骨干完全不可见——这正是 AGENTS.md 警告的"无结构语义蔓延"，也违反
规则 3（`related_to` 只能是临时兜底）。原因是概率上的：P 个概念每次取 k 个，任一
对的期望共现次数为 $N k^2/P^2$，400 次观察下 ≈ 4，**每一对都会靠运气过门**。

**修复：** `HebbianEngine` 记录观察数与每个概念的提及数（每次观察去重），一对概念
过了计数门之后还要满足 Jaccard 关联度
$J = \frac{c_{ab}}{n_a + n_b - c_{ab}} \ge$ `HEBBIAN_MIN_ASSOCIATION`（"提到两者之一
的观察里，有多大比例同时提到两者"）。总是一起出现的对 $J=1$；无处不在的枢纽概念
与偶然同框的概念 $J = n_x/N$ 很小，不会被连上。统计量随 `hebbian_pending` 一起持久化
（`hebbian_stats`），旧存储没有该字段时从零计数，在统计积累前退化为旧的计数门。

`scripts/sweep_hebbian.py`（随机世界、6 主题世界各 400 次观察，认知基准）：

| θ | 随机世界 generic 边（占全部对） | 主题内 270 对被连 | 跨主题边 | 基准 ML / Ops |
|---|---|---|---|---|
| 0.0（修复前） | 1505（85%） | 270（100%） | 127 | 0.67/0.67、0.83/0.83 |
| 0.1 | 516（29%） | 270（100%） | 12 | 不变 |
| **0.2** | **152（9%）** | **269（100%）** | **3** | **不变** |
| 0.3 | 54（3%） | 248（92%） | 0 | 不变 |
| 0.4 | 39（2%） | 183（68%） | 0 | 不变 |

取 **0.2**：主题内的关联完整保留，跨主题噪声几乎消失，随机同框的蔓延下降一个量级；
认知基准（同一批概念反复共现，$J=1$）不受影响。`TestHebbianAssociationGate` 覆盖
随机世界不成团、主题伙伴仍相连、总是同现两次即连、枢纽不与过客相连、统计量重启后
保留、旧状态兼容、删除概念清理统计。

**复验（reflect 第 5b 步）。** 门只在创建时判断，而世界早期的统计很薄：两次提及、
两次同框就是 $J=1$，于是随机世界里仍有 146 条 generic 边靠早期运气建立，之后又被
偶然同框不断强化（权重 0.15–0.42）。用当前统计重算它们的关联度
（$c_{ab}$ = `reinforcement_count` + 2）：中位数 0.063，**没有一条 ≥ 0.2**。
`HebbianEngine.revalidate()` 在每次 `reflect()`（含 light）末尾重判所有非显式的
generic 边：当两概念的提及数之和 ≥ 20（`REVALIDATION_MIN_MENTIONS`，薄统计不判）
且 $J <$ 0.2 × 0.5（`REVALIDATION_HYSTERESIS`，创建与移除阈值之间留出迟滞）时删除，
结果记入 `ReflectResult.stale_relations`。随机世界一次 reflect：151 → 10 条
（130 条复验删除、11 条衰减剪枝）；显式关系与被显式复述升级为 typed 关系的边
永不触碰（`TestHebbianRevalidation`）。

### 7.12 概念漂移下的上下文：任务感知的投影冗余 ✅

**探针**：`pipeline` 先在数据工程语境（ETL、warehouse、Airflow、Spark、schema，
task="data eng"）用 150 次观察，再在 ML 语境（training、model、GPU、dataset、
checkpoint，task="ml"）用 150 次。之后：

| 投影 | 修复前 | 修复后 |
|---|---|---|
| 无任务 | ML 邻域为主（近因） | 不变 |
| task="data eng" | Spark、**checkpoint**（ML！）、warehouse、schema、Airflow | Spark、warehouse、schema、Airflow、ETL |
| task="ml" | 全 ML | 不变 |

分解（激活 / 任务亲和 / 显著性 / 投影相关性）：数据工程概念相关性 0.24–0.25，
checkpoint 0.19（任务折扣 0.6）——相关性上并没有输。输在 **MMR 的冗余项**：数据工程
概念两两之间邻居集合完全相同（一个团），选了 Spark 之后其余成员的 Jaccard 冗余
= 1.0，罚项 0.5；checkpoint 与 Spark 只共享种子，冗余 0.14。$0.5\cdot0.19-0.5\cdot0.14
> 0.5\cdot0.25-0.5\cdot1.0$，于是"多样性"从任务外的簇里买来了第二个名额，渲染
出来的"Core Understanding"第二条就是 checkpoint。相关性被种子归一化压在 0.25 的
窄带里，而冗余项的动态范围是整个 [0, 1]——只靠调 `TASK_AFFINITY_DISCOUNT` 或 λ
都救不回来（任务外候选要赢，只需 sim 差 > 相关性差 / λ）。

**修复：任务外候选与"任务本身"冗余。** 在 MMR 中
$\text{redundancy}(c) = \max\big(\max_{s\in S}\text{sim}(c,s),\; 1-\text{affinity}(c)\big)$，
且只在剩余候选里还有比 $c$ 更贴合任务的候选时生效（`min_mismatch`）；无任务时
affinity ≡ 1，行为不变。含义：在还有任务内候选可选时，一个与任务无关的概念和一个
重复概念一样"冗余"；它仍可进入投影——只要其原始相关性超过任务内候选 1/折扣 倍
（≈1.67×），或任务内候选已用尽。

**副作用是正向的**：认知基准从 ML 0.67/0.67、Ops 0.83/0.83 变为 **1.00/1.00 /
1.00/1.00**，两个任务投影只共享种子（`scripts/sweep_mmr.py` 现在对 λ ∈ [0.1, 0.5]
全部 1.0）。原来 0.67 的损失正是同一机制：ML 团自冗余后，MMR 把名额买给了 Ops 簇。
`tests/test_context_drift.py` 固化漂移场景（任务选中对应邻域、无任务随近因、域视角
一致、漂移概念保留两个域）；`test_same_world_produces_different_rank_order_under_task_context`
改为更强的契约（任务外依赖可以不出现）。

### 7.13 种子永远在投影里 ✅

**探针**（认知基准世界，`max_depth=2`）：

| 种子 | max_concepts | 修复前投影 | 缺失的种子 |
|---|---|---|---|
| PyTorch, deployment（task="ml training"） | 3 / 4 / 6 | PyTorch + ML 邻居 | **deployment（三种规模都缺）** |
| PyTorch, neural network, optimizer | 3 | optimizer, model serving, neural network | PyTorch |
| 6 个 ML 种子 | 3 | model serving, optimizer, gradient descent | 3 个 |

种子是 Agent 明确询问的对象，却被 MMR 当作普通候选：与已选种子同团的第二个
种子 Jaccard 冗余 = 1，或者在任务视角下属于"任务外"（§7.12 的冗余下限），于是被
一个"更多样"的邻居顶掉。`Projection` 的语义是"围绕这些种子的局部视图"，缺了种子
的视图对下游 Agent 是误导。

**修复：** 门面把 `seed_ids` 传给 `ProjectionEngine.project()`；种子不受激活阈值
过滤，并在 MMR 之前按分数**先选**（超过 `max_concepts` 时按分数截断），剩余名额
才由 MMR 填充。`tests/test_projection_seeds.py`：跨域双种子在任务下都在、6 个种子
3 个名额时恰好是分数最高的 3 个种子、团种子 + 弱种子都在、种子排在 `concepts`
前面。认知基准与漂移、稳定性测试全部不变。

### 7.14 关系的概率锚定地板 ✅

**探针**（概念一直保持活跃，只让关系闲置）：

| 闲置观察数 | 显式声明 1 次（p=0.70） | 显式复述 5 次（p=0.76, r=9） | Hebbian（p=0.15） |
|---|---|---|---|
| 200 | w=0.197 | w=0.694 | w=0.054 |
| 500 | w=0.029 | w=0.410 | **剪掉** |
| 1000 | **剪掉** | w=0.171 | 剪掉 |
| 3000 | 剪掉 | **剪掉** | 剪掉 |

第一轮把 `probability`（"这条 typed relation 是对的"）从时间衰减里拿了出来，但
剪枝只看 `weight < 0.02`：边被删了，信念也就没了——一个 Agent 明确声明过的依赖
在 400–1000 次无关观察后消失，与概念侧"被确认的概念不蒸发"不对称。

**修复：** 与 §3.1 的概念地板同构。显式关系的 `weight`/`confidence` 向
$\text{floor} = 0.1\cdot p\cdot 2^{-\Delta/4380}$ 回归而不是向 0（`RELATION_FLOOR_SHARE`，
纪元半衰期与概念地板共用）；Hebbian 边没有地板（靠强化存活，由 §7.11 的复验清理）；
剪枝阈值不变。p=0.70 的一次声明地板 0.07，约 7 900 次观察后才随纪元遗忘降到 0.02
以下；`confirm()` 抬高 p 即抬高地板。基准里的半衰期测试（0.8 → 0.4 ± 0.05）仍然
成立（0.8 → 0.435）。`TestRelationEvidenceFloor`：3 000 次闲置后显式关系仍在、Hebbian
边仍消失、12 000 次后显式关系按纪元遗忘、复述过的关系地板更高。

### 7.15 学习状态的持久化成本 ✅

**探针**（100 个主题 × 20 个概念，3 000 次观察，SQLite）：轻量 reflect 0.31 s、完整
reflect 0.28 s、投影 0.3 ms、重启加载 0.28 s——都没问题；但**单次 ingest 从 22 ms
涨到 44 ms**。cProfile（第 1 500–1 700 次观察，9 212 个待定对）：`_persist_learning_state`
占 ingest 的 **76%**——`snapshot()` 为每个待定对 `sorted()` 一次（190 万次调用），
整个 state（待定对 + 每个概念的提及数）每次观察都 `json.dumps` 一遍并写库。

**修复：**

1. `HebbianEngine` 内部键改为规范字符串 `"a|b"`（排序拼接）——`snapshot()` 是一次
   `dict` 拷贝；`restore()` 格式不变，旧快照可直接加载。
2. `Store` 增加 `save_learning_state()/load_learning_state()`：JSON 后端写
   `learning.json`，SQLite 后端写 `state` 表的 `learning` 行。`state.json` 只剩时钟、
   reflect 元数据与群落快照，每次观察写它很便宜。旧存储里内联的
   `hebbian_pending/hebbian_stats` 在首次打开时迁移。
3. 摊销策略（`World._persist_learning_state`）：学习记录在条目数
   ≤ `LEARNING_EAGER_LIMIT`（2 000）时每次观察都写（小世界重启逐字一致，既有测试
   不变）；超过后最多每 `LEARNING_PERSIST_EVERY`（20）次观察写一次，并在 `reflect()`
   与新增的 `World.close()`（也支持 `with World(...) as w`）时强制写。概念与关系仍然
   每次观察 flush；崩溃最多丢失 20 次观察的**共现计数**。

agents 层：`PKMAgent.close()` 调用 `World.close()`，CLI 在退出时（`try/finally`）、Web 在
`shutdown` 事件时调用。

`tests/test_learning_state.py`：小世界不 close 也重启一致、记录与 state 分离、旧
格式迁移、大世界 60 次观察只写 2–4 次学习记录、reflect/close 强制写、close 后重启
逐字一致、SQLite 记录往返。

### 7.16 剪枝宽限期：快褪色、慢删除 ✅

**探针**（`auto_reflect_every=25`，概念 X 提到一次，`gap` 次无关观察后再提一次）：

| gap | 修复前：第二次提及时 X 还在？ | 修复后 |
|---|---|---|
| 25 / 50 / 80 | 在（同一节点，n=2） | 在 |
| 100 / 200 / 400 | **已删除，第二次提及新建节点（n=1）** | 在（同一节点，n=2） |
| 800 | 已删除 | 已删除 |

embryonic 半衰期 24、`prune` 阈值 0.02：一次性概念 ~54 次观察后 FADING，~80 次后
被删——连同它的关系、任务画像、来源引用一起；第二次提及只能从零开始。§5 的标定
模拟只跑 decay + lifecycle，**没有 prune**，所以"每 720 次复现 → developing"在真实
流水线里根本到不了——概念在第一个间隔里就被删了。100 主题 × 20 概念、每 100 次
观察 light reflect 的世界：创建过 1 937 个概念，只剩 797 个，430 个孤立，established
仅 2 个。

**修复：** `PRUNE_MIN_IDLE_TICKS = 720`——FADING 且置信度 < 0.02 的概念还要**闲置
满 720 次观察**才删除。褪色仍然很快（可逆：再提及即复苏同一节点），删除变慢（不可
逆）。同一世界修复后：1 885 个概念、1 444 个 developing（复现门现在能积累）、
421 个 FADING 痕迹、关系 274 → 1 380。代价是褪色痕迹多占一个月的存储；它们在激活
里只以 `0.3 × salience下限 0.1` 的强度传播，极少进入投影。
`TestPruneGrace`：gap 100/400/700 复苏同一节点、770 后删除、100 时已 FADING；
两个假设"立即剪枝"的旧测试改为先闲置 800 tick。

---

## 8. 复现

```bash
# 全部测试（含 36 个新行为测试）
./start_world0.sh test
# 或
python -m pytest -q tests/test_dynamics_analysis.py tests/test_temporal.py

# 标定扫描（§5 的表）
python scripts/calibrate_decay.py 0.1 0.2 0.3

# MMR λ × 冗余度量扫描（§7.5 的结论）
python scripts/sweep_mmr.py

# 显著性持续性份额扫描（§7.1 的表）
python scripts/sweep_salience.py
```

时间模拟只有一个原语：`world.clock.advance(n)` 让 n 次观察"无事发生"地流逝；
`tests/test_dynamics_analysis.py` 的 `_simulate_cadence()` 与
`scripts/calibrate_decay.py` 都建立在它之上。需要只老化某个概念时，把它的
`last_activated_tick` 设为 `world.clock.tick - n`（见 `tests/test_temporal.py`
的 `_age_concept`）。不要再用 `timedelta` 回拨墙钟时间戳——在认知时钟下那只会
产生每小时 0.1 tick 的漂移。
