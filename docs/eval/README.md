# LongRun：长期运行 Agent 中 World 0 与传统记忆、直接上下文的对比

> 问题：一个长期运行的 Agent，把"过去发生的事"交给 **World 0**、交给**传统记忆模块**
> （检索、摘要、事实库、知识图谱），或者**直接放在上下文里**，有什么区别？
>
> 这是一组可复现的实验，不是宣传材料。设计者同时是 World 0 的作者，所以实验先经过
> 5 位独立评审的对抗性审查（§6），按其意见做了第二、第三版；报告里每个结论都写明了
> 它成立的条件和不成立的条件（`01-report.md`）。

## 文件

| 文件 | 内容 |
|---|---|
| `README.md`（本文） | 实验设计、被比较的系统、指标、公平性措施、局限、复现 |
| [`01-report.md`](01-report.md) | 结果与结论：区别在哪、哪种情形下谁占优、World 0 该改什么 |
| `results/*.meta.json`、`results/tables.md`、`results/readers.md` | 各研究的元数据（提交号、耗时、预算）、汇总表与真实 LLM 读者阶段的表；原始行由代码重新生成 |
| `../../benchmarks/longrun/` | 生成器、系统适配器、打分、运行器、读者阶段、汇总 |
| `../../tests/test_longrun_bench.py` | 评测台自身的测试：确定性、金标准只含已被告知的内容、预算强制、打分 |

## 1. 隐藏世界与事件流（`worldgen.py`）

一个隐藏世界由 `n_domains` 个领域组成，每个领域是一张带类型的小概念图：依赖层（`depends_on`）、
包含（`contains`）、冲突（`conflict`）、使能（`enables`），并有若干**桥接概念**：同一个名字在两个
领域里有不同的关系（一词多义）。所有名字都是生造的伪词，因此没有系统能靠先验知识猜出答案，
只能依赖被给予的信息。

Agent 每次在一个领域里工作一段（run），每个事件陈述当前真值中的 1–3 条关系加若干共现概念；
有些事件是闲聊（噪声）；有些领域在早期活跃、之后长期休眠；每条关系的被陈述频率服从 Zipf 分布，
因此存在只被说过一次的"冷门知识"；在预定时刻某条依赖被**修订**（旧关系被撤回、新关系被陈述，
事件文本里写作 "Correction: … no longer …"）；一部分事件带一个票据号（情景性细节，不是概念）；
事件文本里另有填充词，模拟真实转录里"不是概念"的部分。

同一个 seed 对应同一个隐藏世界与同一条事件流，所有系统读到完全相同的内容（按 seed 配对）。

**查询**在检查点生成，五种：`focus`（围绕两个入口概念的局部视图）、`chain`（X 最终依赖什么）、
`bridge`（桥接概念在当前任务领域里的关系）、`stale`（被修订过的依赖现在是什么）、`detail`（某条
关系被提到时的票据号）。金标准只包含 **Agent 已被告知且仍为真** 的内容——从没说过的关系，任何
系统都不可能召回；被撤回的关系永远是"过期"，不是金标准。

## 2. 被比较的系统（`systems.py`）

所有系统看到相同的事件、相同的"已链接实体"（查询里出现的已知概念名）和相同的任务文本，
返回一个不超过 token 预算的上下文。

| 系统 | 代表什么 | 读什么 |
|---|---|---|
| `none` | 无记忆（下界） | — |
| `window` | 直接上下文：最近的若干事件，塞满预算为止 | 文本 |
| `full_context` | 直接上下文：全部历史（上限 128k tokens，最新在后） | 文本 |
| `rag` | 对历史事件做 BM25 检索，取 top-k | 文本 |
| `rag_recency` | Generative-Agents 配方（相关性 + 新近度 + 重要性），不是"普通 RAG"，作消融 | 文本 |
| `summary_buffer` | ConversationSummaryBuffer：近期窗口 + 一份扁平频次摘要，作消融 | 文本 |
| `summary_task` | 按任务线程维护摘要，当前任务优先 | 文本 |
| `factstore` | Mem0 式事实库：抽取出的事实为条目，实体匹配、一跳、无任务标签、保留票据、撤回即删除 | 抽取结果 |
| `fact_task` | 上一个加任务标签、两跳、按任务过滤——"配置良好的结构化记忆" | 抽取结果 |
| `kg_temporal` | 带失效与出处的图（Zep/GraphRAG 式），两跳 | 抽取结果 |
| `kg_static` | 只增不删的图（不处理撤回），作消融 | 抽取结果 |
| `state_doc` | 把整个当前状态当作一份文档，当前任务的事实排在前面，截断到预算 | 抽取结果 |
| `world0` | World 0 按出厂方式：`World.project`（depth 2、不 reflect）+ `Projection.render()` | 抽取结果 |
| `world0_compact` | 同一个投影，用紧凑格式渲染（类型化主张 + 信念） | 抽取结果 |
| `world0_tuned` | 紧凑渲染，depth / reflect 频率**只在 dev seeds（100–104）上选定**（`tuned.json`），报告用 test seeds（0–9） | 抽取结果 |
| `world0_reflect / focus / notask / depth1 / depth3` | World 0 的消融变体 | 抽取结果 |

"抽取结果" = 一个抽取器对事件给出的概念、关系、撤回；`extract_p = 0` 时是完美抽取，`extract_p > 0` 时按
同一套误差模型损坏（漏抽 p、抽错关系类型 p/2、每个事件多出一条虚假关系 p、漏掉撤回 p）。
文本系统读原文，不经过抽取，所以不受抽取误差影响（它们付出的是 token）。

## 3. 指标（`scoring.py`）

| 类别 | 头号指标 | 说明 |
|---|---|---|
| focus | 金标准（入口概念两跳内的已知主张）的**召回率** | 精确率、F1、半径 1 / 3 的召回单独报告 |
| chain | 依赖闭包的 **F1** | 由上下文里出现的 `depends_on` 主张推出闭包，与真闭包比较 |
| bridge | 桥接概念在当前领域的主张的**召回率 × 语义纯度** | 纯度 = 该概念被展示的主张里属于当前领域的比例；与金标准半径无关 |
| stale | **严格**：当前主张出现且被撤回的主张不出现 | "两者都出现、当前的信念更高"的宽松记分单独报告，不计入头号指标 |
| detail | 票据号出现在上下文里 | 情景性细节，**不在 World 0 的职责范围**（AGENTS.md 规则 6），不计入综合分 |

**综合分（utility）** = focus、chain、bridge、stale 四类头号指标先各自取均值、再取平均。detail 单独一栏。
每个系统的实际 token 用量与分数并列报告，因为结构化系统只返回相关内容，并不会用满预算。

打分读的是**系统实际放进提示词的文本**（`parse.py` 从渲染文本里解析主张），不是背后的投影对象：
出厂的 `render()` 最多打印 10 条关系，超出的不计分。

## 4. 统计

复制单位是 **seed**（一个世界、一条事件流、全部系统）；同一次运行里的查询相互相关，所以所有统计量都
基于每个 seed 的均值。区间是对 seed 的 t 区间；系统间比较按 seed 配对，用精确符号翻转检验，
对比较族做 Holm 校正。主实验 10 个 seed；敏感性研究 6 个；大世界与 H=6000 为 4 个。
生成器的超参数（Zipf 偏斜、领域切换节奏、休眠领域数、票据率）按 seed 抽取，不让结论依赖单一设置。

## 5. 研究一览

| 研究 | 变量 |
|---|---|
| `main` | H=1000，10 个 seed，全部核心系统，预算 150–4800 tokens |
| `scale` | 视野 H = 300 / 3000 / 6000 |
| `bigworld` | 40 个领域 × 每领域 40 个概念、领域随时间出现、H=6000（已知状态约 2.3 万 tokens，装不进一个提示词） |
| `extraction` | 抽取误差 p = 0.05 / 0.1 / 0.2 / 0.3；另含"只展示至少被陈述两次的主张"的证据阈值变体 |
| `taskmode` | 查询里的任务标签：精确 / 缺失 / 错误（另一个领域的标签） |
| `chatter` | 闲聊比例 0 / 0.5 / 0.75 |
| `verbosity` | 每事件填充词 0 / 120 / 400 |
| `ablation` | World 0 变体 |
| 读者阶段 | 真实 LLM 读者：每个（问题，上下文）一份独立文件，只读这一份（见 `readers.py`） |

## 6. 独立评审与据此做的修改

第一版结果出来后，5 位互不知情的评审者分别从基线强度、指标与金标准偏向、World 0 用法、真实性、
统计方法五个角度攻击评测台，随后由一位编辑合并。合并后的结论是：**评测台的第一版并不偏向 World 0，
反而在多个格子里偏向基线**，但它有若干让比较不可信的缺陷。已按 must-fix 逐项处理：

| 评审发现 | 处理 |
|---|---|
| 宽松的过期记分只有 World 0 能得分（仅凭约 0.03 的信念差） | 严格记分为头号指标，宽松记分单列 |
| detail 占综合分 20%，World 0 按设计得 0 | detail 单独成栏，综合分只含四类 |
| focus / chain 的金标准就是图基线硬编码的两跳遍历 | 报告半径 1 / 2 / 3；`kg_*` 明确标为"图型基线"；bridge 改为与半径无关的语义纯度 |
| 基线偏弱（`kg_temporal` 不保留票据；事实库无任务标签；摘要不分领域） | 新增 `fact_task`、`summary_task`、`state_doc`；`kg_temporal` 保留出处；扁平摘要与 GA 配方标为消融 |
| 任务标签只有 World 0 与 RAG 能用；"改写"研究里领域词仍在 | 所有系统都收到任务标签；改为"精确 / 缺失 / 错误"三种 |
| 抽取、链接、读者都是理想的，且注释里提到的 LLM 读者阶段并不存在 | 加入抽取误差研究；新增真实 LLM 读者阶段 |
| 预算网格停在 1200，整个世界才约 3.9k tokens，"必须投影"从未被检验 | 预算扩到 4800；新增 40×40 的增长型大世界 |
| World 0 延迟被缓存污染；资源行不可比 | 延迟另做串行无缓存计时；资源行只列状态持有者 |
| 出厂 `render()` 的信息被计入投影对象而非文本 | 改为从渲染文本解析主张 |
| 没有 dev/test 划分，World 0 的参数没有调 | depth / reflect 只在 dev seeds 上选，报告在 test seeds |

## 7. 局限（评审与我们自己都同意的）

- **没有真实的抽取与链接。** 结构化系统读的是生成器给的抽取结果（可注入误差，但不是真 LLM 抽取）；
  实体链接是精确子串匹配。抽取的 LLM 成本没有计入任何系统。
- **读者阶段规模有限**（4 个 seed、130 个问题），且用的是同一个模型；不覆盖长上下文退化的全部形态。
- **隐藏世界是合成的**：名字唯一且无别名、无同义词、无指代；概念身份（别名、合并、拆分）、
  视角（perspective）、reflect 机制等 World 0 的其余功能在本实验里基本没有被用到，
  因此**本实验不能评价这些功能**。
- **金标准由生成器定义**（两跳球、依赖闭包、领域内主张），不是下游任务成功率。
- 压缩格式是作者写的；它把"World 0 的投影"与"渲染开销"分开，但部分优势来自渲染工程。
- token 数是估计（`tokens.py`，常见词记 1、生造词按 `ceil(len/3)`），未用真实分词器校验；
  并列报告 `chars/4` 作为敏感性。
- 延迟与吞吐是单机、单进程、非空闲机器上的测量，只用于量级比较。

## 8. 复现

```bash
python -m benchmarks.longrun.run --study tune --out docs/eval/results-dev     # dev seeds 上选 World 0 的 depth / reflect
python -m benchmarks.longrun.tune docs/eval/results-dev
for s in main ablation extraction taskmode scale bigworld chatter verbosity; do
  python -m benchmarks.longrun.run --study $s --seeds 10 --out docs/eval/results
done
python -m benchmarks.longrun.report docs/eval/results > docs/eval/results/tables.md
python -m benchmarks.longrun.run --study main --seeds 4 --timing --workers 1 --out docs/eval/results-timing   # 串行无缓存计时
python -m benchmarks.longrun.readers prepare --out <dir> --gold <manifest.json>   # 读者阶段：生成案例文件
python -m benchmarks.longrun.readers grade --gold <manifest.json> --answers <answers.json>
python -m pytest -q tests/test_longrun_bench.py
```
