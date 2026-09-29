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

> 更新（形式化轮）：这张表早于第二轮的复现晋升路径。用当前代码重跑（`python
> scripts/calibrate_decay.py 0.2`）的结果为：每 24 次 → established 0.998（dev 240，est 384）；
> 每 72 次 → established 0.82；每 168 次 → established 0.516（复现路径，est 5040）；每 720
> 次 → developing 0.173；一次性概念约 54 次后 fading；30 次后弃用在 26 280 次时 fading。
> "每 168 次永远达不到 established"已被复现路径改变，理论界见论文命题 3.5。

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
| **第十四轮** | |
| `context/`（新包：`grounding.py`） | `name_coverage()`、`ground_task()`：任务点名的概念 1.0、其直接邻居 0.5、部分点名按覆盖率（≥ 0.5） |
| `projection/engine.py` | 任务亲和度 = max(任务历史, 词汇锚定) |
| `tests/test_task_grounding.py`（新） | 14 个测试；`tests/test_layer_boundaries.py` 把 `context` 列入概念核心 |
| **第十五轮** | |
| `schemas/relation.py` | `SYMMETRIC_SEMANTIC_RELATIONS`；`is_directed` 同时看轴向与语义；`RelationEdge.connects(directed=)`、`opposes()` |
| `relations/manager.py`、`core/interfaces.py`、`core/test_doubles.py` | `find_between(..., directed=)`；`discover()` 按陈述方向匹配有向关系 |
| `world/_ingest.py` | 显式主张削弱同一对概念上相反轴向的显式主张（`weakened_relations` 中报告）；`contradicted_relations` 按方向匹配 |
| `tests/test_relation_claims.py`（新） | 11 个测试 |

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

> **后记（§7.22 / §7.23）：**本节的门已被取代。复现按"距上一次被计数的复现不少于 24 tick"计数而不是固定窗口，
> 间隔门读的是证据 $e(n,d)$ 与平衡 $\beta$ 而不是置信度（$\rho\ge10\wedge e\ge0.5\wedge\beta\ge0.8$，第 12 次使用），
> 晋升在事件上评估，见 §7.22 与论文 §3.5。

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

**后记（§7.21）：** 上表的地板 $0.1\cdot p$ 锚定的是信念。negative 轴的显式主张当时是用抑制增益
（0.05–0.12）而不是信念播种的，所以对它们地板低于剪枝阈值，一次陈述的 conflict 450 次观察就被剪；
§7.21 把信念与增益分开后，它们与 depends_on 同样存续 8 100 次。

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

### 7.17 任务锚定：没见过的任务也能改变投影 ✅

**探针**（两个簇共享种子 `deployment`：Ops = kubernetes / helm chart / container registry /
rollout / load balancer，ML = pytorch / gpu cluster / training run / checkpoint / loss curve；
每簇 12 次观察，`project(["deployment"], task=…, max_concepts=5)`，数字为种子之外的 Ops / ML 个数）：

| task | 观察**不带** task 标签：修复前 | 修复后 | 观察带 `ops`/`ml` 标签：修复前 | 修复后 |
|---|---|---|---|---|
| （无） | 1 / 3 | 1 / 3 | 1 / 3 | 1 / 3 |
| `kubernetes rollout` | **1 / 3** | 4 / 0 | **1 / 3** | 4 / 0 |
| `pytorch training` | **1 / 3** | 0 / 4 | **1 / 3** | 0 / 4 |
| `ops` | 1 / 3 | 1 / 3 | 4 / 0 | 4 / 0 |
| `ml` | 1 / 3 | 1 / 3 | 0 / 4 | 0 / 4 |

任务亲和度只来自 `task_profile`——概念**被标注过**的任务历史。于是：由不带 task 标签的
观察构成的世界（常见：Agent 往往不标注任务）里，**任何**任务字符串都不改变投影；即使
观察带标签，一个新任务——哪怕它直接点名了两个 Ops 概念——也毫无作用。这违反 AGENTS.md
Rule 4（上下文改变相关性）：上下文只有在"恰好复述过去的标签"时才生效。

**修复：** 新包 `world0.context`（AGENTS.md 建议的 `context/` 模块）。`name_coverage(task,
names)` 以概念签名同样的词级粒度计算任务提到了概念名（或别名）的多大比例；无签名词的
名称（CJK、极短标签）退化为整串包含。`ground_task()` 给投影候选打分：名称被完全提到的
概念是**锚点**（1.0），部分提到且覆盖率 ≥ 0.5 的按覆盖率计，锚点的直接邻居继承 0.5。
投影里的任务亲和度取 `max(任务历史, 词汇锚定)`，其余（折扣、§7.12 的任务冗余）不变。

- 只点名种子本身（`task="deployment"`）→ 所有邻居同为 0.5，投影与无任务时相同；
- 部分点名（`"helm"` → `helm chart` 0.5）只拉进该概念，不向邻居扩散；
- 什么都没点名（`"ops"` 在无标签世界）→ 不变，这是正确的：没有可锚定的结构；
- CJK：`"容器滚动发布"` 锚定 `容器`、`滚动发布`，投影落在同一簇；
- 成本：2 000 概念时每次投影 +0.2 ms（锚定复用冗余项已建好的邻居表）。

认知基准与全部既有测试不变（1 116 通过）。锚定只作用于投影阶段：激活的任务增益仍只看
历史，被点名但在 `max_depth` 之外的概念不会被拉进候选——那是"点名即种子"的另一种语义，
留给调用方显式传种子。`tests/test_task_grounding.py`（14 个）。

### 7.18 关系主张：方向与矛盾 ✅

**探针**（显式关系，每行为同一对概念）：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `X depends_on Y` ×10，再 `Y depends_on X` ×1 | 只有一条边 X→Y，**反向主张被当成对 X→Y 的确认** | 两条边：X→Y p=0.81 不变，Y→X p=0.70 |
| `A enables B` ×10，再 `A conflict B` ×10 | enables p=0.85 **不变**，conflict p=0.43——两个相反主张同时"可信" | enables p=0.45，conflict p=0.43：相互竞争（其中"势均力敌"部分是先验 0.76 对 0.10 不等的产物，见 §7.21） |
| `A conflict B`，再 `B conflict A` | 同一条边（碰巧对：查找本来就不看方向） | 同一条边（对称语义，按设计） |

`find_between` 只看"两端是否是这两个概念"，所以 `discover()` 把反向的有向主张合并进
已有的边并 `confirm()`——依赖方向正是 `dependency_map` / `impact_map` 视角（§7.4）读取的
信息。相反轴向的显式主张各建一条边、互不影响，只有 Agent 显式传 `contradicted_relations`
时旧主张才被削弱；从 LLM 抽取的观察里，这个字段几乎总是空的。

**修复：**

- 有向关系按陈述方向匹配：`RelationEdge.connects(a, b, directed=True)`，`find_between(...,
  directed=)`，`discover()` 与 `contradicted_relations` 都用它。平行关系和有向轴上的
  **对称语义**（`SYMMETRIC_SEMANTIC_RELATIONS`：conflict、disjointness、complement、
  incompatible_ontology、co_creation、mutual_reinforcement、future_coupling）两个方向都
  匹配；`is_directed` 同样把它们视为无向，于是方向视角也不再缩放 `conflict` 这类边。
- 显式主张是对相反主张的反证：负轴主张对同一对概念上显式的正轴 / 平行主张调用
  `weaken()`，反之亦然，并在 `IngestResult.weakened_relations` 中报告。共现（Hebbian）
  边与 `generic_relation` 不断言任何东西，不受影响；同轴主张（enables 与 dependence）
  互相兼容，也不受影响。

**当时未改（记录），后来在 §7.21 处理：** 显式关系的初始 `probability` 取语义规格的
`propagation_strength`，所以只说一次的负向主张信念很低（conflict 0.10、disjointness 0.05），
关系地板（§7.14）也随之很低——负向知识比正向知识忘得快。传播强度（激活流多少）与信念
（主张是否为真）是两个量，§7.21 把它们分开。同轴改标签（dependence ×10 后说一次
inclusion，边的标签直接被覆盖）仍然保留。

全部既有测试不变（1 141 通过）。`tests/test_relation_claims.py`（11 个）。

### 7.19 机器意识方向 → `docs/mc/`

从第十六轮起，按意识科学的"指标属性"（Butlin 等 2023）研究 World 0：全局工作空间、
元认知、注意图式、预测加工等理论被用作**设计来源**，验收仍是探针与行为测试，不做任何
关于体验的主张。记录、映射表、探针与各轮分析见 `docs/mc/README.md`。第十六轮实现了
元认知监控（`Projection.epistemic`，`docs/mc/02-metacognition.md`）。

### 7.20 形式化 → `docs/paper/`

当前设计的完整数学模型、命题与证明见 `docs/paper/world0-formal.md`，每个命题由
`docs/paper/verify.py` 在代码上数值检验。形式化过程中发现并修复了七处实现与声称性质
不一致的地方（论文 §12；F6 见 §7.22，F7 见 §7.23）。影响最大的是 F4：激活会移动衰减参照点，两次使用之间未
结算的衰减被丢弃，**遗忘量曾取决于 reflect 的调用频率**（每 24 次观察使用一次的概念，
每次都 reflect 时置信度为 0.31，从不 reflect 时为 0.93）。现在每次强化之前先结算欠下的
衰减，四种调用频率下都是 0.31。§5 的标定脚本恰好一直在每次使用前做衰减，因此标定
出的常数本来就对应修复后的动力学。

### 7.21 显式负向主张：信念与抑制增益是两个量 ✅

**问题.** `RelationManager.discover()` 用 `SemanticRelationSpec.propagation_strength` 播种显式边的
`probability`（"这条主张是对的"的信念）。在 negative 轴上这个数是**抑制通道的增益**
（0.05–0.12，`tests/test_relation_axes*.py` 断言），不是信念。后果一（存续）：地板
$0.1\,p$ 低于剪枝阈值 0.02，一次陈述的负向主张不受保护。后果二（对立）：一对对立主张
的先验是 0.76 对 0.10。后果三（元认知）：`assess()` 读的就是这个概率，一次 `enables`
加一次 `conflict` 被报告为 `leaning`（0.705 对 0.100，差 0.61）。后果四：`confirm()` 每次只
消除 5% 的疑虑，五次重述才把 0.10 推到 0.30，`weaken()` 两次就把它压到 0.01 的下限。

**探针一：存续**（真实的 `World.ingest` + `DecayEngine`，两个概念一直保持活跃，关系不重述，
时钟每次前进 50，`docs/paper/verify.py §4.2` 同款）：

| 陈述一次 | 修复前 p / 闲置多少次后被剪 | 修复后 |
|---|---|---|
| depends_on（对照） | 0.70 / 8 100 | 不变 |
| enables / part_of / similar_to / related_to（对照） | 0.76 / 8 600，0.88 / 9 500，0.64 / 7 500，0.45 / 5 300 | 不变 |
| conflict | 0.10 / **450** | 0.70 / **8 100** |
| disjointness | 0.05 / **350** | 0.70 / **8 100** |
| exclusion、violates_constraint、incompatible_ontology | 0.08 / 400，0.08 / 400，0.06 / 350 | 0.70 / 8 100 |
| instability、complement、adversarial_prediction | 0.12 / 500，0.10 / 450，0.10 / 450 | 0.70 / 8 100 |
| 重述 5 次：depends_on / conflict / disjointness | 0.768 / 9 200；0.304 / 4 000；0.265 / 3 650 | 0.768 / 9 200 三者相同 |

结算步长 10 / 50 / 200 / 1 000 下 conflict 的存续为 8 070 / 8 100 / 8 000 / 8 000（修复前
440 / 450 / 600 / 1 000），与 reflect 频率无关的性质对负向主张同样成立。真实
`World.reflect(light=True)`：闲置 500 / 3 000 / 7 000 次时 conflict 与 disjointness 都还在
（修复前 500 次就没了），9 000 次后按纪元遗忘，与 depends_on 逐点一致；共现边照旧消失。

**修复.** 传播强度与信念分开。`SemanticRelationSpec.claim_prior`：positive / parallel 仍是
传播强度（逐位不变），negative 轴是 `NEGATIVE_CLAIM_PRIOR = 0.70`。`discover()` 用它播种显式边的
信念，`weight`（= 抑制增益）与 `confidence`（= 结构强度）不动；抽取器给出的先验照旧优先。
地板公式不变，仍是 `0.1 × probability`——它锚定的是信念，只是现在负向边的信念是真的信念。
**为什么是 0.70：** 它是一次陈述的 `dependence`（论文的参照关系，也是 positive 轴上最弱的
主张）所带的信念。扫描（`NEGATIVE_CLAIM_PRIOR` 取 0.30 / 0.50 / 0.60 / 0.70 / 0.76 / 0.88）：
一次陈述的 conflict 存续 2 700 / 5 950 / 7 100 / **8 100** / 8 600 / 9 500，只有 0.70 与 dependence
的 8 100 相等；低于它，3 : 1 的对立竞争里负向一方被压得太低（0.50 时 0.19，0.70 时 0.36）。

**探针二：对立主张**（`enables` 对 `conflict`，真实摄入 + `assess()`；E = enables，C = conflict）：

| 序列 | 修复前 enables / conflict | 修复后 |
|---|---|---|
| E C | 0.705 / 0.100 leaning 0.61 | 0.705 / 0.700 **contested** 0.005 |
| C E | 0.760 / 0.045 leaning 0.72 | 0.760 / 0.645 contested 0.115 |
| E×10 再 C×10 | 0.447 / 0.433 contested | 0.447 / 0.811 leaning 0.36（后一块领先） |
| C×10 再 E×10（镜像） | 0.849 / 0.032 leaning 0.82 | 0.849 / 0.410 leaning 0.44 |
| (E C)×10 交替均等 | 0.536 / 0.150 leaning 0.39 | 0.536 / 0.528 **contested** 0.008 |
| (E E E C)×5 | 0.711 / 0.060 leaning 0.65 | 0.711 / 0.358 leaning 0.35 |
| (C C C E)×5 | 0.407 / 0.389 contested | 0.407 / 0.682 leaning 0.28 |

文档里的"`enables` ×10 再 `conflict` ×10 → 0.45 对 0.43"是先验不等的产物：修复前交换两条主张的
角色得到 0.03 对 0.85；`dependence` 与 conflict 现在以同样的 0.70 起步，交换轴是**精确镜像**
（0.410 / 0.811 ↔ 0.811 / 0.410，5 个序列由 `verify.py §4.3` 逐位检验）。成块重述时后一块
领先，因为 `confirm()` 是相对的 5% 步长，`weaken()` 是绝对的 0.06/(1+0.1δ) 步长（首次 0.0545，
约抵 3–4 次确认）；这对两条轴一样，本轮**不改**该校准。

**不变的部分.** 抑制强度：创建时 conflict / disjointness 的 `weight` 为 0.176 / 0.126，一次陈述后
从 A 抑制 X 的净分数 0.00605，修复前后逐位相同；重述 5 次后的增益 0.7815 → 0.7824（地板
减慢了两次重述之间那 1 个 tick 的衰减）。闲置的一次陈述 conflict 增益松弛到地板而不是 0：
300 / 1 000 / 3 000 次闲置后 0.033 / 0.009 / 0.006 → 0.083 / 0.061 / 0.044（这正是"受保护"的含义，
与正向边的 0.07 地板一致；对 X 的净分数的影响约 0.0002）。共现边仍无地板，概率仍不随时间衰减。

**已存的存储.** `RelationEdge.belief_prior`（新字段，`None` = 修复前存的边）标记信念的起点。
`RelationManager.load()` 对修复前的显式 negative 边一次性重定基（`adopt_claim_prior`，
与 `ensure_probability` 同一个钩子）：存储的信念若恰是"旧种子（增益）加上边自己的确认 / 否证
计数"所能解释的（容差 0.02，两种先后次序取包络），就以 0.70 为起点重放同样的计数，且只升不降；
否则那是抽取器或反馈给出的信念，原样保留。用**修复前的代码**写出的真实存储（JSON 与 SQLite
各一份）载入后：

| 修复前存的边 | 载入前 | 载入后（= 新代码同样历史下的值） |
|---|---|---|
| conflict 陈述一次 | 0.100 | 0.700 |
| conflict 陈述 6 次 | 0.304 | 0.768 |
| disjointness 陈述 3 次 | 0.143 | 0.729 |
| conflict，`contradicted_relations` 4 次 | 0.010（下限） | 0.506 |
| E×10 再 C×3 中的 conflict | 0.188，`leaning` 0.51 | 0.729，`contested` 0.03 |
| 抽取器先验 0.9 / 0.05 的 conflict | 0.9 / 0.05 | 0.9 / 0.05（保留） |
| depends_on、part_of（对照） | 0.729、0.880 | 不变 |

第二次载入与第一次逐位相同（幂等）；`weight` 从不改动。旧存储里已经被剪掉的边找不回来。

**副作用（已测）.** ① 网络熵（`metrics/entropy.py` 用 `probability × 轴系数`）现在把负向主张
按信念计入：玩具世界（enables、depends_on、conflict、similar_to、disjointness 各一条）的
负向质量 0.09 → 0.84，`avg_network_entropy` 0.724 → 0.973，`relation_type_entropy`
0.652 → 0.913。② 抽取器给出的先验恰为 0.3 时，模型校验器曾把它当成"未设置"而替换成语义
默认值（depends_on 0.70，conflict 0.10），重新载入时 `ensure_probability` 又把它换成结构置信度（0.3 → 0.376）；播种过的边现在带 `belief_prior`，两处都不再替换它。
③ 用 `relation_priors` 重述 negative 边时，`update_probability` 不再把混合后的信念写进抑制
增益（否则 0.25 → 0.72）；修复前是 0.272，现在 0.248。

**未改（记录）.** ① `weaken()` 从操作权重里减去绝对量，negative 轴的增益（0.05–0.2）几次否证
就到 0.01：端点不共现、`contradicted_relations` 连续 6 次时 conflict 信念 0.43 而权重
0.010，下一次 reflect 即被剪，同样处境的 depends_on 权重 0.505、还能存续 4 950 次观察。
共现强化通常抵消这一点，所以只出现在纯否证序列里。② 抽取器路径首次陈述时仍把信念写进
操作权重（conflict 先验 0.9 → 权重 0.976，修复前后相同）。③ `confirm` / `weaken` 的
步长校准（成块重述时后一块领先）不变。④ 投影的"Key Relations"按 `weight` 排序，抑制增益
小的负向边仍排在后面。

`tests/test_negative_claim_belief_negative_claims_final.py`（93 个；在修复前的代码上 77 个失败——
其中约 30 个是数值断言，如 `350 > 3000`、轴交换镜像，其余是新接口不存在——16 个是对照）。
既有的 1 190 个测试一个都没改：全套 1 283 通过、8 跳过。

### 7.22 成熟度：稀疏节律与 reflect 频率无关 ✅

**探针**（真实的 World / ingest / reflect；同一条观察流，只改变 reflect 频率，末尾一次 reflect）：

| 流 | 之前：reflect 每 1 / 50 / 1000 次 / 从不 | 之后 |
|---|---|---|
| 每 24 次使用，共 30 次 | est 0.667 / est 0.637 / dev 0.308 / dev 0.308 | est 0.758 × 4 |
| 每 72 次 | est 0.569 / 0.566 / 0.458 / 0.431 | est 0.596 × 4 |
| 每 168 次 | est 0.345 / 0.343 / 0.317 / 0.305 | est 0.476 × 4 |
| 每 720 次 | **dev** 0.205 / 0.204 / 0.199 / 0.196 | **est** 0.296 × 4 |
| 突发 30 次后闲置 5 000 | est 0.557 / est 0.482 / **emb** 0.087 / **emb** 0.086 | dev 0.153 × 4 |
| 一次性提及后闲置 5 000 | 四种频率都被剪枝 | 同 |
| 40 次后中途新增 5 个连接、再稀疏使用 | core 0.688 / core 0.667 / **dev** 0.245 / **dev** 0.206 | core 0.750 × 4 |
| 每 3 次使用被否证 1 次 | dev 0.246 / 0.212 / 0.174 / 0.174 | dev 0.331 × 4 |

之前的置信度最大差为 0.36–0.47（突发 / 每 24 次），之后 $\le 3\times10^{-9}$（墙钟漂移项）。

**根因（三处，均由探针确认）**：(a) 晋升只在 reflect 里评估，读的是当时衰减后的置信度（"某个 $c \ge 0.6$ 的窗口
是否被看见"取决于 reflect 节奏），而晋升改变之后所用的半衰期 $H_m$；每次 reflect 还只爬一级；(b) 褪色在结算
恰好运行时才记，整个区间用旧半衰期；`weaken` 在未结算的置信度上扣罚；(c) core 的连接数在 reflect 时点数，
且 reflect 先评估后剪枝（已死的边仍被计入）。另外，结算把地板冻结在区间终点，切分残差（0.008，长间隔可达
0.02）也让"完全相同"变成"差不超过 0.01"。稀疏节律那一侧：复现门读置信度，而稀疏节律的置信度平衡点被证据
地板封顶（命题 3.5 旧版：$T<974$ 才保证可达；$T=720$ 需 324 次使用）。

**设计（B-monotone：门只含单调 / 时间无关的量，评估在事件上）**：

1. 门：embryonic→developing 为 $n\ge3\wedge c\ge0.3$（密集）或 $\rho\ge3\wedge e\ge0.15$（间隔）；
   developing→established 为 $n\ge10\wedge\rho\ge3\wedge c\ge0.6$ 或 $\rho\ge10\wedge e\ge0.5\wedge\beta\ge0.8$；
   established→core 为 $n\ge30$ 且存活连接 $\ge \max(2,5-\lfloor(n-30)/20\rfloor)$。$e\ge0.15/0.5$ 就是元认知层的
   `TENTATIVE_EVIDENCE / WELL_EVIDENCED`。稀疏节律的 established 因此是"十个不同窗口里出现过、证据充分且
   确认远多于否证"，与置信度无关：$T=24/72/168/720$ 都在第 12 次使用（此前 13 / 15 / 27 / 324 次）。
2. 评估位置：每次激活之后、每次关系创建 / 强化之后（两端，先结算后评估），一直爬到不动点；reflect 的 `evaluate()`
   退化为对事件外变更（合并、导入）的追赶，并报告自上次 reflect 以来的晋升。
3. 褪色边界：结算是移动地板方程 $\dot c=-\lambda(c-f(t))^+$ 的精确解（精确半群，随机 20 000 组最大偏差
   $6\times10^{-13}$），穿越 0.05 的时刻二分求根，之前用成熟度半衰期、之后用 FADING 半衰期；`weaken` /
   `adjust_confidence` 先结算，并在事件上判褪色。
4. core 连接：计数用"结算后权重 $\ge0.02$"的边，与物理剪枝是否已发生无关。
5. 既有存储：不增字段；已存的成熟度不变（没有降级）；$\rho=0$ 的旧记录仍按旧的密集规则；跨版本读取
   JSON 与 SQLite 存储已验证（core / established / developing / fading 记录保留，reflect 追赶晋升）。

**反例控制（必须留在门外的）**：

| 控制 | 之前 | 之后 |
|---|---|---|
| 一个窗口内 30 次提及后闲置 | reflect 频繁时 **established**（$n\ge10\wedge c\ge0.6$ 走密集门），从不 reflect 时 embryonic | developing（四种频率一致，$\rho$ 不足） |
| 一次性提及 | fading → 第 720 次剪枝 | 同 |
| 每 1 / 2 / 3 次使用被否证 1 次（T=24 / 168 / 720） | T=24、每 3 次否证：**established**（reflect 频繁时） | fading / developing / developing，均不 established |
| 每 4–10 次否证 1 次 | T=24：established；T=168 需 $k\ge8$；T=720 从不 | T=24 / 168 / 720 都 established（$\beta$ 在评估当刻 $\ge0.8$） |
| 随机重提（间隔 24–2500，否证 30–70%） | fading / developing | 同，不 established |
| established 之后被遗弃 | 计数 fading 约 10 500（该探针中概念始终是 developing） | fading 约 23 000 观察后、剪枝约 23 500（nothing is immortal） |
| 淡出的 established 被提及一次 | developing，之后 12 次每日使用仍 developing（0.247） | developing（$\rho$ 回到 1）；再经 12 个窗口重新 established（0.226） |

**校准脚本**（`scripts/calibrate_decay.py 0.2`，每次 gap 末尾采样）：

| 每 gap 次 | 之前 | 之后 |
|---|---|---|
| 24 × 2160 | est 0.998，established 于 384 | est 0.998，established 于 288，developing 于 72（原 240） |
| 72 × 4320 | est 0.82，1152 | est 0.855，864 |
| 168 × 8760 | est 0.516，5040 | est 0.626，2016 |
| 720 × 17520 | **dev** 0.173，无 established | **est** 0.26，established 于 8640 |
| 一次性提及 fading | ≈ 54 | ≈ 54 |
| 30 次后遗弃（26 280 时） | 0.028 fading | 0.003 fading（褪色后走 FADING 半衰期） |

（168 一行的 developing 首次出现在间隔末尾的采样点 840 而非 672：概念在第 3 次使用时已是 developing，
但采样点在 gap 末尾，此时它已再次淡出——n≤6 时地板低于 0.05——所以采样看到的是 fading；两版都有一次
fading 发作。）认知基准的精确度 / 召回 / 覆盖 / Jaccard 距离与之前逐位相同（1.0 / 0.833–1.0 / 1.0 / 0.909），
基准世界的 16 个概念从全部 embryonic 变为 developing（原来没有 reflect 就永远不晋升）。

**副作用与取舍**：

- 投影原先靠等值概念之间 $10^{-11}$ 的墙钟噪声决定先后，噪声的系统性顺序（后激活者略高）被几个测试隐含依赖；
  成熟度提前生效后噪声被更少地遗忘，这些测试约一半概率不稳定（修复前后 6 / 6 稳定 → 约 50%）。
  `projection/engine.py` 现在把分数量化到 $10^{-6}$，量化后相等的按最近激活、名称、id 裁决。
- 残余（当时）：可剪枝的概念被再提及时，reflect 过的世界新建（$n=1$）、从不 reflect 的世界复苏旧节点；关系一侧
  `weaken` 不先结算、冻结地板的时代修正仍在。独立评审又找到几处同类泄漏（幽灵邻居、死边复活、稀疏节律
  T > 720 的命题 3.5 在 reflect 过的世界里不成立）——都在 §7.23（F7）里处理，本节的结论以 §7.23 为准。
- 摄入吞吐量：1 500 次摄入 + 15 次轻量 reflect，4.25 / 4.80 s（之前）对 5.14 / 4.96 s（之后），在噪声范围内。

**测试**：`tests/test_maturity_schedule_maturity_b_monotone.py`（57 个；T=2000 的"第 12 次"一条在 §7.23 改为"宽限期之外被遗忘"）。其中 30 个在旧实现上失败；随机重提、
一次性等控制在两版上都通过。`test_decay_split_matches_closed_form` 改为 `test_decay_split_is_exact`（旧闭式
断言的正是被消除的时代修正）。


### 7.23 存在性与存活：已死即不存在（F7）✅

独立评审（正确性 / 论文准确性 / 行为与性能三个视角）对 §7.22 的补丁做了对抗性检查，找到定理 3.7 与命题 3.5 的
三处不成立，以及若干钩子缺口。全部在修复前的代码上复现，并由 `tests/test_schedule_independence.py`（47 个）固定。

**根因（同一件事的三个面）**：物理删除只发生在 reflect，而事件读取状态时并不区分"已死但还没删"与"不存在"。

| 症状 | 修复前 | 修复后 |
|---|---|---|
| 间隔 800 的 12 次使用（$n\le6$ 的概念在两次使用之间已 fading 且闲置超过 720） | reflect 每 50 次：从不 established，概念被新建 11–13 次；从不 reflect：第 12 次 established | 两种世界一致：每次新建、n = 1（命题 3.5(ii)） |
| 5 个一次性伙伴的边 + 每 48 次使用 40 次 | 从不 reflect：**core** 0.804（边还在、被 core 门计入）；reflect 过：established 0.790 | established 0.790 × 4（core 门只数存活边与存活邻居） |
| 死边（权重 < 0.02）被重述 / 共现 | 从不 reflect：复活旧边；reflect 过：新建。core 判定因此不同（core / established / established / core） | 一律先回收再新建；四种频率一致 |
| 带 `relation_priors` 的重述 | 不触发 core 评估（要等下次激活或 reflect） | 触发 |
| `ActivationEngine.activate(record=True)`、正向 `adjust_confidence` | 不触发晋升 | 触发 |
| 关系 `weaken` / 带先验重述 / `adjust_strength` | 在未结算的权重上修改（惩罚被追溯衰减） | 先结算 |
| 关系结算 | 地板冻结在区间终点，残差可达 0.05；一次陈述的边存续 8 070 / 8 100 / 8 200 / 9 000（结算每 10 / 50 / 200 / 1000 次） | 精确半群（命题 4.2′）：与频率无关 |
| 复现窗口 | 固定网格 `tick // 24`：一个横跨边界的 26 tick 突发算三个窗口，31 个连续提及在 tick 20–23 开始时达到 established | 距上一次被计数的复现 ≥ 24 tick：31 个连续提及无论何时开始都是 ρ = 2 |
| 长间隔后复苏 | 复苏的置信度可低于 0.05：下一次结算又标 fading，再一次提及降回 embryonic | 复苏至少回到 0.05 |
| 投影分数量化（绝对 $10^{-6}$） | 种子置信度 $10^{-6}$ 时所有邻居量化为 0，排序退化为最近激活 | 相对最强激活量化（`SCORE_DIGITS` 不变） |
| negative 关系带先验创建 | 抑制增益被信念覆盖：`conflict` 无先验 0.176、先验 0.7 → 0.776、先验 0.9 → 0.976 | 增益恒为 0.176（信念只在 `probability`） |
| `split_concept` | 新节点 `created_tick = last_activated_tick = 0`：tick ≥ 720 时下一次 reflect 即被剪 | 以当前 tick 出生 |
| 替换 `World._lifecycle`（文档化的覆盖点） | 构造时绑定的钩子仍驱动原引擎 | 钩子在事件时经 `self._lifecycle` 调用 |

**设计**：已死是结算后状态的谓词（概念：fading ∧ $c<0.02$ ∧ 闲置 ≥ 720；关系：结算权重 < 0.02），没有事件时单调。
事件上已死的对象被当作不存在（摄入时 `IngestPipeline._reap_if_expired`，关系发现与共现学习时
`RelationManager.reap_dead_between`），判定取"reflect 最后可能运行的那个 tick"即当前观察的前一个 tick，所以恰好 720
的节律不被误伤；物理删除退化为垃圾回收。core 门只数权重 ≥ 0.02 且另一端未过期的边（`concept_expired` 对结算后状态
做非修改的读取）。

**验证**：`tests/test_schedule_independence.py` 含 8 条随机流（5 个概念、关系、否证、突发、10 种节律）在 reflect 每
1 / 50 / 1 000 次与从不下终态逐项一致；评审者的随机模糊测试在修复前 160 条里 11 条不一致，修复后 340 条
（2 500 与 4 000 tick，轻量与完整 reflect，含 Hebbian 再验证与剪枝）零不一致。`docs/paper/verify.py` 新增：
褪色边界与独立 RK4 的对照（三种成熟度的穿越）、晋升门表的边界、关系结算的精确性、稀疏节律含 reflect 与宽限期之外
的负对照、幽灵邻居与宽限期之外两条流。

**取舍与残余**：
- $T > 720$ 的稀疏使用不再能成熟。此前从不 reflect 的世界里它能，那是调度依赖而不是特性；要恢复需要延长宽限期或
  给"有复现历史的概念"豁免，是设计问题（论文 §13）。
- 默认配置（不 reflect）下概念现在在事件上成熟，`Projection.render()` 的 "Core Understanding" 段带描述与链接列表，
  一个 200 概念、3 000 次观察的世界 15 个概念的投影从 1 380 增到 3 180 token；紧凑渲染不受影响。LongRun 里出厂
  `world0` 在 1 200 token 预算下召回下降约 0.04–0.05，`world0_compact` / `tuned` / `reflect` 在 ±0.01 内。
- 读取路径仍读存储值：事件之间激活的种子分数与投影读的是上次结算时的置信度（同一条流同一 tick，0.489 对 0.317），
  依赖 reflect 的节奏。修法是读取时取结算后的非修改投影，但会改变激活里每一处 `node.confidence`——待办。
- Hebbian 再验证仍是 reflect 独有的关系删除：一个概念的第 30 次确认若恰好落在某条低关联共现边的最后强化后约 300
  tick 内，且 core 门恰好依赖它，两种世界会不同。340 条随机流没有命中，但未证明不可能。
- 评审还提到但未处理：`leaning` 变得罕见（一次 `conflict` 对 30 次 `enables` 为 contested 0.19），是校准判断；
  被撤回的冲突不重述时抑制持续数千次观察（`0.1 × 信念` 的地板，设计如此）；`dynamics/__init__` 的"互不导入"
  改为"互不导入引擎"，`decay` 的纯结算 / 存活函数是共享内核。

### 7.24 任务语境：区分性的任务词与"在别的任务里"的主张 ✅

**来源**：LongRun（`docs/eval/01-report.md`）里任务标签对 World 0 没有可测的影响（精确 / 缺失 / 错误 = 0.59 / 0.59 / 0.61），
桥接概念的语义纯度只有 0.52——而"上下文敏感"是 AGENTS.md 的第三优先级。

**探针**（seed 0 的 12 个桥接查询，`world0_tuned`，预算 1 200）：精确 / 无 / 错标签 = 0.436 / 0.449 / 0.436。两个原因：

1. **任务词没有区分度。** 评测里的任务全是"〈领域〉 work"；`task_match_score` 按词计数，"buducom work" 对 "panucom work"
   得 0.5，所以每个概念对每个任务的亲和都 ≥ 0.5。真实任务同样共享套话（"fix … bug"）。
2. **投影不看主张的语境。** 概念选择确实随任务变化，但预算够大时视图会被填满，而"选中概念之间的全部关系"都被放进来——
   桥接概念在另一个领域里的主张也在其中。

**修复**：
- `TaskVocabulary`：概念档案携带的不同任务标签上的词频；词权重 $\ln\frac{L+1}{d+1} + 0.01$。每个标签都有的词几乎不计分；
  词汇表由 `ConceptManager` 在强化 / 删除时增量维护、在合并 / 拆分 / 记录路径的激活 / 加载后重建，是档案的函数。
- 语境拆分（投影 `CONTEXT_MATCH = 0.5`）：一条显式主张的语境是它被**陈述**时的任务（`RelationEdge.claim_tasks`，共现的
  provenance 不算）；视图中的概念若在当前任务下有存活的主张，它在别的任务下的主张移到 `Projection.other_contexts`
  （渲染为 "Seen in Other Tasks"），争议判定仍看两者的并。没有匹配语境时什么都不动。

**效果**（同一探针）：精确标签 0.436 → **0.929**；无标签不变（0.449）；错标签 0.217——错标签若恰好是桥接概念的另一个领域，
视图就如实显示那个领域的主张。LongRun 的数字见报告。

**取舍**：错误的任务标签现在有代价（与 `summary_task`、`state_doc` 同类，但只作用于"在当前任务里有主张"的概念）；任务标签
不可靠时应不传任务。测试：`tests/test_task_context.py`。

### 7.25 撤回与读取时结算 ✅

**撤回**。评测里的"更正：X 不再依赖 Y"原来走 `contradicted_relations`（否证），旧主张只是信念下降、仍在每个视图里：严格
修订得分 ≈ 0，真实读者 100% 提到了过期主张。"不再成立"是世界的修订，不是主张曾为假的证据，所以新增
`Observation.retracted_relations` 与 `RelationEdge.retracted_tick`：信念不变、地板为 0（权重松弛到 0 后按定义 3.6 已死）、
不传播、不算 core 连接、不在视图里，只在端点是种子时列在 "No Longer Holds"；共现既不强化它也不把这对概念视为已联结；
显式重述使它恢复；合并时只有两边都撤回才撤回。论文 §4.4。

**读取时结算**。激活此前读取存储的置信度与权重：两次使用之间，一个 reflect 过的世界与一个没有 reflect 的世界给出不同的
视图（同一 tick 种子分数 0.317 对 0.489）。现在激活与投影读取结算值（`settled_confidence`、`projected_relation_weight`），
并把已死的边与概念当作不存在；关系按结算权重排序。论文推论 3.2′。

**独立评审**找到第一版的八处缺陷（共现复活撤回的边并阻断重新联结、注意力轨迹点名撤回的主张、撤回 / 已死 / 共现边把概念
误判为"在语境内"、一次共现把未标注的主张改标为别的任务、语境拆分隐藏了争议、撤回已死的边依赖 reflect、合并时撤回的孪生
主张压过存活的主张、读取路径仍看见未剪枝的死边与死概念），每一处都有在第一版上失败的回归测试（`tests/test_retraction.py`、
`tests/test_task_context.py`）。评审的 12 个复现脚本在修复后全部给出正确结果；加入 25% 撤回事件的随机流在四种 reflect 安排下
终态一致（差 ≤ 1e-6）。

### 7.26 紧凑渲染成为默认 ✅

**问题**。LongRun 里同一个投影，出厂 `render()` 让真实读者答对 0.57，基准自带的紧凑格式 0.90：出厂渲染按成熟度分节，
每个概念打印 `name.feature.id`、成熟度、置信度、证据，关系行打印轴、结构 / 传播强度与强化次数，最多 10 条关系；读者要自己
把 id 对回名字，预算大半花在注释上。紧凑格式只存在于基准代码里，Agent 拿不到。

**改动**。`Projection.render(style="compact")` 成为默认，`style="full"` 保留为诊断视图：

- 当前主张一行一句，按信念从高到低：`api depends on db (belief 0.82)`；短语表 `RELATION_PHRASES` 覆盖全部 26 个语义关系，
  按陈述方向读（`contains` / `part_of` / `depends_on` 的别名方向与规范名一致）。
- 共现边不算主张，只把端点列进 `Also relevant`；概念卡描述列在 `Definitions`；同名概念加上义项（`Apple (fruit)`）。
- 需要打折扣的另列、带标签，读者不会把它们当成当前主张：`No longer holds`（撤回）、`Seen under other tasks`（带任务名，
  来自 `claim_tasks`）、`Hold loosely`（有争议 / 倾向的主张对、证据单薄的概念）。
- 顺带修正：`inclusion` / `proper_inclusion` 的说明原为 "A is contained in B"，与别名 `contains`（源包含目标）方向相反；
  抽取提示词与规格说明统一为 "A contains B"。

**测量**（LongRun main，10 个 seed，基准改用库里的紧凑渲染）：1200 token 下 `world0_tuned` 0.95 → 0.95、`world0_compact`
0.90 → 0.90，token 多 14–18%（撤回与"其他任务"两节是基准自带渲染丢掉的内容）；紧预算下略降（300：0.75 → 0.71，600：
0.93 → 0.91）。打分只从主张行解析，这两节对读者的价值（不复述过期或别的任务的主张）不计分；保留它们是有意的取舍。
`tests/test_compact_render.py`。

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
