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
二者必须一起看。

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

所有字段均有默认值，旧的 JSON 存储可直接加载（`task_profile` 自动回填，tick 与
recurrence 默认 0）。

---

## 7. 结构性问题与建议路线

按对"持续动态更新的认知系统"的重要性排序。第二轮已实现 §7.2、§7.3、§7.4、
§7.6、§7.8（标注 ✅，行为测试在 `tests/test_roadmap_dynamics.py`）；§7.1、
§7.5、§7.7、§7.9 仍是建议。

### 7.1 把 `confidence` 拆成"证据"与"显著性"

`confidence` 现在同时表示"这个概念是真的/有用的"（证据）和"它现在相关"
（显著性）。`evidence_balance()`（Beta 后验）和 `temporal_relevance()`
（新鲜度）已经分别是这两者的干净度量。建议：

```text
evidence(v)  = Beta 后验均值 × 饱和(n)         ——只被 activate/weaken 改变，慢速纪元遗忘
salience(v)  = temporal_relevance × 任务/视角亲和 ——只被认知时间与上下文改变
confidence   = evidence × g(salience)            ——派生量，供旧接口读取
```

成熟度门槛改为基于 `evidence`（加上 §7.2 的复现），衰减只作用于 `salience`。
这会消除 §3.1 的"双时钟"问题，也让 `render()` 里的 "confidence: 0.06" 不再误导
下游 Agent。

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

### 7.4 有向传播与视角 ✅

`Perspective.direction_weights = {"forward": …, "backward": …}`：forward 是沿
关系 source→target 遍历（`A depends_on B` 从 A 出发："我依赖什么"），backward
相反（"谁依赖我"）。缺省 1.0，默认视角保持无向；`ActivationEngine` 把该系数乘进
`edge_strength`。这是"同一世界在不同视角下给出不同投影"的第三个杠杆（前两个是
关系轴权重与域亲和）。按语义关系细分方向权重留作后续扩展。

### 7.5 投影冗余度量与相对阈值

- 冗余用加权 Jaccard（按 `weight × ρ_type`），使"通过一条强依赖边相连"与
  "通过一条弱泛化边相连"的重叠含义不同。
- 候选阈值改为相对值 `max(min_activation, 0.02·S)`，低置信种子也能得到完整视野；
  当前 0.01 绝对阈值意味着种子置信度 <0.33 时 3% 下限带会被整体裁掉。
- MMR 的 $\lambda=0.3$ 未经任何任务级评测标定；建议用
  `tests/_cognitive_benchmark.py` 的 precision/recall 做一次扫描。

### 7.6 身份解析索引化 ✅

`_find_synonym_match()` 现在先用 `TokenIndex.candidates()` 取短名单再打分。
任何正的同义分都要求共享一个标签词或一个 sense/description 词，因此短名单是
完备的（`TokenIndex` 现在同时索引 `sense` 词，只用于短名单，不影响签名相似度）；
探针完全不可分词（如纯中文标签且无描述）时回退到全量扫描。候选按
`(created_tick, id)` 排序，平局结果与原实现一致。复杂度从 O(N) 降到 O(候选数)。

### 7.7 存储层

当前 JSON-per-file 适合 <10k 概念。下一步建议 SQLite（单文件、事务、按 id 更新）
或 append-only 事件日志 + 周期快照；`Store` Protocol 已经把这一步隔离好了。

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

---

## 8. 复现

```bash
# 全部测试（含 36 个新行为测试）
./start_world0.sh test
# 或
python -m pytest -q tests/test_dynamics_analysis.py tests/test_temporal.py

# 标定扫描（§5 的表）
python scripts/calibrate_decay.py 0.1 0.2 0.3
```

时间模拟只有一个原语：`world.clock.advance(n)` 让 n 次观察"无事发生"地流逝；
`tests/test_dynamics_analysis.py` 的 `_simulate_cadence()` 与
`scripts/calibrate_decay.py` 都建立在它之上。需要只老化某个概念时，把它的
`last_activated_tick` 设为 `world.clock.tick - n`（见 `tests/test_temporal.py`
的 `_age_concept`）。不要再用 `timedelta` 回拨墙钟时间戳——在认知时钟下那只会
产生每小时 0.1 tick 的漂移。
