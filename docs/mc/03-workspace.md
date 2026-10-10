# 03 · 第十七轮：持续焦点（GWT-4 / GWT-2）与注意图式（AST-1）

## 1. 指标

- **GWT-2**：容量有限的工作空间，进入它需要经过竞争；内容一旦进入即被"点燃"——非线性地、
  全或无地放大并维持。
- **GWT-4**：状态依赖的注意——工作空间的当前内容决定下一步注意什么，从而能沿一条思路连续
  调用各模块。
- **AST-1**：系统维护一个关于"自己此刻在注意什么、为什么"的简化模型，并用它控制注意。

## 2. 基线

`docs/mc/probes/workspace.py`，Ops / ML 两簇共享桥接概念 `deployment`，另有一个无关簇
（sourdough）。投影 5 个名额，数字为种子以外的 Ops / ML 个数：

- 冷启动 `project(["deployment"])` → Ops 1 / ML 3；
- 先看 `["kubernetes", "rollout"]`、`["helm chart", "load balancer"]` 两个 Ops 视图，再看
  `["deployment"]` → **仍是 Ops 1 / ML 3，完全相同**。投影是世界状态的纯函数，"刚才在想
  什么"不起作用；
- `Projection` 只有概念、关系、分数，不记录任何概念为何入选。

对 Agent 而言，这意味着沿一条思路追问时，每一步都从零开始；一个桥接概念总是按全局统计而
不是按当前话题来解读。

## 3. 设计

### 3.1 焦点（`world0.context.Focus`）

一个小而短暂的**注意状态**（不是记忆）。开关 `World(sustained_attention=True)`，默认关闭，
关闭时投影仍是纯函数。

| 机制 | 规则 | 参数 |
|---|---|---|
| 点燃（GWT-2） | 投影后，种子一律点燃；其他入选概念的相关度达到本视图非种子最高相关度的一半即点燃；点燃者以强度 1.0 进入焦点（全或无） | `IGNITION_THRESHOLD = 0.5` |
| 容量 | 最多 7 个，保留最强者 | `FOCUS_CAPACITY = 7` |
| 保持 | 此后每次投影，已有强度 × 0.6；低于 0.15 离开——无关视图穿插 3 次后仍在，第 4 次后消失 | `FOCUS_RETENTION = 0.6`、`FOCUS_MIN_STRENGTH = 0.15` |
| 释放 | 在一个**不同的显式任务**下投影时清空 | — |
| 偏置（GWT-4） | 下一次投影中，焦点成员的相关度 × (1 + 0.5 × 强度)，成员的直接邻居 × (1 + 0.5 × 0.5 × 强度) | `FOCUS_GAIN = 0.5`、`FOCUS_NEIGHBOR_SHARE = 0.5` |

两条关键约束：

- **只偏置、不注入。** 焦点只提升本次种子**已经激活到**的候选，从不把焦点内容加进候选集。
  所以与焦点无关的查询完全不受影响——这从结构上排除了固着（perseveration）。
- **种子不向邻居传递焦点。** 第一版中桥接概念 `deployment` 自己进了焦点，于是它的所有邻居
  （包括 ML 簇的 `pytorch`）都被当作"靠近焦点"，焦点被稀释到每个它连接的簇，注意轨迹也
  误报 pytorch "仍在焦点中"。种子本来就在视野中心，它在焦点里并不说明思路指向它的哪个
  邻居，因此本次种子不参与邻居传递。

### 3.2 注意图式（`Projection.attention`）

每个入选概念一条 `AttentionTrace`：

| 字段 | 含义 |
|---|---|
| `kind` | `seed`（Agent 问的）或 `reached`（激活到的） |
| `via` / `relation` | 贡献最大的已激活邻居及所经关系（`候选激活 × 边权重` 最大者）——对真实激活路径的简化模型，正如 AST 所说"注意图式是简化的" |
| `task_named` / `task_history` | 任务点名了它（§7.17 锚定）/ 它在该任务下出现过 |
| `sustained` / `in_focus` | 焦点提升了它 / 它本身就在焦点中（而非只是靠近焦点） |
| `ignited` | 本次视图后它进入了焦点 |

渲染新增 `### Why These Concepts`：`kubernetes: via dependence from deployment; still in focus`、
`helm chart: via generic_relation from deployment; named by the task; next to the current focus`。

"控制注意"的一面：`world.focus.items()` 可读、`world.focus.clear()` 可清空，换任务自动释放。

## 4. 结果

| 场景 | 无焦点 | 有焦点 |
|---|---|---|
| 两个 Ops 视图后看 `deployment` | Ops 1 / ML 3 | **Ops 3 / ML 1** |
| 两个 ML 视图后看 `deployment` | Ops 1 / ML 3 | Ops 1 / ML 3（本就偏 ML） |
| Ops 焦点后看无关的 `sourdough` | 与冷启动相同 | **与冷启动相同**（无固着） |
| Ops 焦点，插入 k 次无关视图后看 `deployment` | — | k = 0…3：Ops 3；k = 4：回到 Ops 1（焦点消退） |
| Ops 焦点下换任务 `model training` | Ops 1 / ML 3 | Ops 1 / ML 3（焦点释放，与无状态一致） |

默认关闭时，全部既有测试与基准不变。`tests/test_workspace.py`（13 个）。

## 5. 局限与下一步

- **点燃只作用于进入焦点，不作用于激活剖面。** 激活尾部仍是一串约 0.09 的平局（`01`
  §3 GWT-2）；可以研究在激活阶段引入非线性（例如对超过阈值者做再入放大），这与 RPT-1
  （循环加工）是同一个方向。
- **焦点只由投影驱动。** GWT 中新的感知也会争夺工作空间；可以让 `ingest()` 的高意外观察
  （第十八轮的预测误差）点燃焦点。
- **PKM Agent 尚未启用。** `sustained_attention` 目前只在 `World` 上可选；在 Agent 的多轮
  对话中开启并评估连贯性，是它真正的使用场景。
- **偏置增益未标定。** `FOCUS_GAIN = 0.5` 在探针世界中足以让桥接概念转向而不压倒种子的强
  邻居（Ops 焦点下 `pytorch` 仍在视图中）；需要在更大的世界上扫描。
