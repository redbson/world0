# 02 · 第十六轮：元认知监控（HOT-2）

## 1. 指标

HOT-2：*metacognitive monitoring distinguishing reliable perceptual representations
from noise*。落到 World 0：投影交给 Agent 的不只是"这些概念与关系"，还应包括"这些
知识有多可靠、哪里存在分歧"。对 LLM Agent 而言，这是减少"把薄弱或有争议的知识当成
确定事实复述"的直接手段。

## 2. 基线探针

世界：Ops / ML 两簇共享 `deployment`；`pytorch → gpu cluster` 先说 6 次 `enables`、
再说 4 次 `conflict`；`quantum annealer` 只出现一次。`project(["pytorch"])` 的渲染：

```
- pytorch → enables [positive] → gpu cluster (structural: 0.82, propagation: 0.76, reinforced 15×)
- pytorch → generic_relation [parallel] → gpu cluster (structural: 0.55, propagation: 0.45, reinforced 20×)
- pytorch → conflict [negative] → gpu cluster (structural: 0.84, propagation: 0.10, reinforced 7×)
```

- 两个相反主张并列，**没有显示信念**（`probability`），也没有标出分歧；
- `structural` 是关系类型的结构强度，不是信念——conflict 的 0.84 看上去比 enables 的
  0.82 还"强"；
- 共现边与主张同样排版；
- 一次性概念与被确认五十次的概念只能靠一串数字区分。

## 3. 设计

`world0.projection.metacognition.assess(concepts, relations) → EpistemicStatus`，
由投影引擎在每次投影时调用，结果挂在 `Projection.epistemic`。

**可靠度**：用与时间无关的证据 `ConceptNode.evidence()` 分级（实测：提及 1 次 0.06、
2 次 0.13、3 次 0.19、5 次 0.29、10 次 0.46、20 次 0.64、50 次 0.82）：

| 等级 | 条件 | 大致含义 |
|---|---|---|
| `tentative` | evidence < 0.15 | 至多见过两次 |
| `moderate` | 0.15 ≤ evidence < 0.5 | 见过几次到十来次 |
| `well_evidenced` | evidence ≥ 0.5 | 在十几次以上的观察中得到确认 |

选证据而不是置信度：置信度会随闲置衰减（它回答"现在还活跃吗"），证据只随确认积累
（它回答"我们有多少理由相信它存在"）——元认知关心的是后者。

**争议主张**：投影内同一对概念上的显式主张中，若存在相反轴向（负轴 vs 正轴 / 平行，
与第十五轮 `RelationEdge.opposes` 同一判据），报告全部主张及其信念，按信念排序：

- 领先者与最强反方主张的信念差 < `CONTEST_MARGIN = 0.25` → `contested`；
- 否则 → `leaning`（仍然报告：少数主张未被驳倒）。

共现边与 `generic_relation` 不是主张，不参与。

**渲染**：

- 显式主张一行显示 `belief: x`，共现边显示 `co-occurrence`；
- 新增 `### Epistemic Status`：`Contested: A enables B (belief 0.48) vs A conflict B (belief 0.44)`、
  `Leaning: … over …`、`Thin evidence (seen once or twice): …`；无事可报时不出现。

## 4. 结果

同一探针世界，第十六轮后：

```
- pytorch → enables [positive] → gpu cluster (belief: 0.62, structural: 0.82, …)
- pytorch → generic_relation [parallel] → gpu cluster (co-occurrence, structural: 0.55, …)
- pytorch → conflict [negative] → gpu cluster (belief: 0.23, structural: 0.84, …)

### Epistemic Status
- Leaning: pytorch enables gpu cluster (belief 0.62) over pytorch conflict gpu cluster (belief 0.23)
```

结构化结果（`probes/baseline_indicators.py`）：`quantum annealer` → `tentative`，其余概念
`well_evidenced`；争议对 `leaning`，margin 0.39。投影的概念选择**不变**（只加注释），
全部既有测试不变。`tests/test_metacognition.py`（9 个）。

## 5. 局限与下一步

- **HOT-3 尚未闭环。** 监控结果只呈现给 Agent，投影选择本身还不用它：基线中一次性概念
  `quantum annealer` 仍凭新鲜度排在 `deployment` 投影第 3 位，高于 `kubernetes`。下一步
  可以研究让选择对"证据薄弱"敏感，同时不压制真正的新信息。
- **负向主张的先验偏低会影响争议判断。** 显式关系的初始信念取其类型的传播强度（分析文档
  §7.18 记录），`conflict` 从 0.10 起步，所以同样次数的陈述下负向主张的信念系统性地低，
  "leaning 正向"部分来自这个先验而非证据。
- **阈值来自提及次数，未按来源可靠性区分。** 同一来源重复十次与十个独立来源各说一次得到
  相同的证据；按来源多样性加权是可能的改进。
