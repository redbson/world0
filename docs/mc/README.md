# World 0 × Machine Consciousness（机器意识方向）研究记录

本目录存放 World 0 在机器意识（Machine Consciousness）方向上的研究、分析与实验记录。

## 立场与边界

- **只讨论功能属性，不讨论体验。** 我们使用意识科学中"指标属性"（indicator
  properties）的方法：从主流意识理论（全局工作空间、高阶理论、注意图式、预测加工、
  循环加工）中抽取可计算的功能特征，检查 World 0 是否具备、缺什么、补上之后认知
  是否更好。满足任何指标都**不**意味着系统有意识，本目录也不做这种主张。
- **World 0 是 Agent 的一个组件。** 它是 LLM Agent 的概念认知层，不是完整的智能体；
  许多指标（感知、行动、具身）落在 LLM 与 Agent 循环里，不在 World 0 内。映射时我们
  标明"属于 World 0"与"属于 Agent 其余部分"的边界。
- **仍受 AGENTS.md 约束。** 每个机制必须改善概念清晰度、激活质量或投影有用性，不能
  把记忆、工作流或文档存储带进核心模型。意识理论在这里是**设计来源**，验收标准仍是
  探针与行为测试。

## 文件

| 文件 | 内容 |
|---|---|
| [`01-indicator-mapping.md`](01-indicator-mapping.md) | 理论综述；14 个指标属性逐条映射到 World 0；基线探针证据；路线图 |
| [`02-metacognition.md`](02-metacognition.md) | 第十六轮：元认知监控（HOT-2）——可靠度分级、争议主张、信念可见 |
| [`03-workspace.md`](03-workspace.md) | 第十七轮：持续焦点（GWT-4 状态依赖注意、GWT-2 点燃）与注意图式（AST-1） |
| [`04-prediction.md`](04-prediction.md) | 第十八轮：摄入时的预测误差（PP-1）——相对预期的缺席与意外组合、标定、漂移检测 |
| [`probes/baseline_indicators.py`](probes/baseline_indicators.py) | 指标基线探针（每轮后重跑，记录状态变化） |
| [`probes/workspace.py`](probes/workspace.py) | 第十七轮探针：焦点的转向、保持、释放、无固着与注意轨迹 |
| [`probes/prediction.py`](probes/prediction.py) | 第十八轮探针：常规 / 偏离、概念漂移、结构化 vs 随机世界、成本 |

## 轮次

| 轮次 | 指标 | 状态 | 记录 |
|---|---|---|---|
| 16 | HOT-2 元认知监控 | ✅ 已实现 | `02-metacognition.md` |
| 17 | GWT-4 状态依赖注意 + GWT-2 点燃 + AST-1 注意图式 | ✅ 已实现（`sustained_attention` 可选开启） | `03-workspace.md` |
| 18 | PP-1 预测误差 | ✅ 已实现（报告；尚未驱动学习或注意） | `04-prediction.md` |

## 复现

```bash
python docs/mc/probes/baseline_indicators.py        # 指标基线
python docs/mc/probes/workspace.py                  # 第十七轮探针
python docs/mc/probes/prediction.py                 # 第十八轮探针
python -m pytest -q tests/test_metacognition.py tests/test_workspace.py tests/test_prediction_error.py
```
