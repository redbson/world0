# World 0 每日研究与开发协议 / Daily research & development protocol

每天一次（例行任务 "World 0 daily research"，北京时间 08:00 前后触发，新会话、无人值守），按下面的步骤做一轮
**研究 → 决策 → 开发 → 验证 → 提交**。每一轮的记录写在 `notes/YYYY-MM-DD.md`，一天一份；索引在本文末尾。

A fresh, unattended session runs one round a day: **research → decide → develop → verify → commit**.
Each round is one note in `notes/YYYY-MM-DD.md`; the index is at the end of this file.

## 0. 定位 / Orientation（每轮必读）

1. `AGENTS.md` — 项目定义与设计规则（概念 → 卡 → 类型化关系 → 语境 → 激活 → 投影；边界：不是记忆、不是日志、不是工作流）。
2. `docs/ROADMAP.md` — 当前版本范围、推迟到下一版的问题。
3. `CHANGELOG.md` 的 `[Unreleased]` 节 — 最近做了什么。
4. `docs/world0-cognitive-dynamics-analysis.md` §7 的最后几节 — 最近几轮的问题、规则、实测。
5. `docs/paper/world0-formal.md` §13 与 `docs/world0-api.md` §7 — 不能破坏的性质（状态是观察流的函数；Stable 级接口）。
6. 上一份 `notes/*.md` 的"未完成 / 下一步"。

分支：所有工作在 `dev` 上开发并推送；`main` 的集成 PR 是 redbson/world0#2（草稿）。不要开新分支、不要改 `main`。

## 1. 研究 / Research（≤ 30 分钟）

查**最近 7 天**（首轮：最近 30 天）的论文，主题固定为下面七类，每类至少一条查询。
**本环境的网络限制**（2026-10-08 实测）：`arxiv.org`、`export.arxiv.org`、`api.semanticscholar.org`、`huggingface.co`、`alphaxiv.org`
对网页抓取工具全部不可解析，只有**网页搜索**可用。所以用搜索工具（`site:arxiv.org <关键词> 2026`、extended 模式）
获取标题、arXiv id 与摘要片段；记录时注明"仅搜索摘要，未读正文"。若某天搜索也不可用，记下并跳到 §2。

| 主题 | 关键词（任选组合） | 对应 World 0 的环 |
|---|---|---|
| Agent 记忆与巩固 | agent memory, memory consolidation, forgetting curve, spaced repetition, long-term memory for LLM agents | 衰减 / 长期记忆（论文 §3） |
| 概念与激活 | spreading activation, concept graph, associative memory, Hebbian, activation dynamics | 激活（§6） |
| 语境与投影 | context engineering, context selection, retrieval for agents, task-conditioned retrieval, working memory | 语境 / 投影（§7） |
| 知识图与时间 | temporal knowledge graph, belief revision, contradiction, knowledge editing, retraction | 关系 / 撤回（§4） |
| 认知架构 | cognitive architecture LLM, global workspace, metacognition, ACT-R, SOAR | 元认知 / 焦点（§8–9） |
| 评测 | long-horizon agent benchmark, memory benchmark, LoCoMo, LongMemEval | LongRun（`docs/eval/`） |
| 抽取 | relation extraction LLM, schema-guided extraction, entity resolution | 抽取 prompt（`prompts/defaults.py`） |

每篇只记：标题、作者（首作者 et al.）、arXiv id 或 DOI、一句话"它做了什么"、一句话"与 World 0 哪一环相关、相同还是相反"。
**不复述摘要，不列无关论文**。读不到正文就记"仅摘要"。网页不可达时记下并跳到 §2（用已有待办继续）。

## 2. 决策 / Decide（一条，只一条）

从三处选**一项**当天可完成（含测试）的开发：(a) 研究发现里能直接转成规则或实验的；(b) `docs/ROADMAP.md` 推迟项；
(c) 上一份笔记的"下一步"。选择标准按 `AGENTS.md` 的优先级：概念完整性 > 关系质量 > 语境 > 投影 > 可演化；
"加大存储量"类不选。在笔记里写明：选了什么、为什么、不选的候选各一句。

## 3. 开发 / Develop

- 先写**探针**（一个小脚本或测试）证明问题存在并量化，再改代码；探针数字进笔记。
- 改动遵守：状态是观察流的函数（门只在事件上变真；曲线族在区间起点确定；结算是半群）；`world0.api` 的 Stable 级形状不改
  （新增可选字段可以）；关系必须类型化；不往核心加记忆 / 工作流 / 文档存储。
- 每条新规则配行为测试（`tests/test_*.py`），命题性的性质加进 `docs/paper/verify.py` 并在论文里写一段。
- 文档：`CHANGELOG.md` `[Unreleased]`、分析文档 §7 新一节（问题 / 规则 / 为什么不破坏性质 / 实测 / 代价）、
  必要时 README 与 `docs/world0-usage.md`。

## 4. 验证 / Verify（全部通过才提交）

```bash
python -m pytest -q                     # 全部测试（约 3–6 分钟；吞吐测试在高负载下偶发，单独重跑）
python docs/paper/verify.py             # 全部命题
python -m benchmarks.longrun.run --study main --seeds 10 --out /tmp/lr   # 动力学 / 投影有改动时；与 docs/eval/results 的 main 比较
```

动力学改动要报告 LongRun `main`（必要时 `bigworld`）前后数字，哪怕是"没有变化"。

## 5. 提交 / Commit

- 一轮一到两个提交，信息写清问题与规则；提交信息末尾用会话系统提示里给出的署名行；提交、PR、代码里**不写模型 ID**。
- `git push -u origin dev`。不要创建新 PR（redbson/world0#2 已存在）；若改了对外行为，在 PR 正文末尾追加一行日期 + 一句话。
- 笔记 `notes/YYYY-MM-DD.md` 与本文末尾的索引一起提交。

## 6. 笔记模板 / Note template

```markdown
# YYYY-MM-DD

## 论文 / Papers（最近 7 天）
- **标题** — 首作者 et al., arXiv:XXXX.XXXXX。做了什么。与 World 0：哪一环，相同 / 相反 / 可借用。
- （没有相关新论文时写明查询与结果数）

## 决策 / Decision
选：…。理由：…。未选：… / … 。

## 探针 / Probe
问题存在的证据与数字。

## 开发 / Development
改了什么（文件、规则、常数），为什么不破坏"状态是流的函数"。

## 验证 / Verification
pytest N passed；verify.py；LongRun main 前 / 后。

## 提交 / Commits
sha：一句话。

## 未完成 / 下一步
```

## 7. 不做的事 / Never

- 不为了"有产出"而改参数：没有探针证据的改动不提交。
- 不扩大范围：一天一项；大于一天的工作拆成有独立价值的第一步，其余写进"下一步"。
- 不改 `main`、不合并 PR、不打标签、不删除分支、不改 `.claude/` 配置。
- 不把论文摘要大段复制进仓库；记录只写一句话与引用。

## 索引 / Index

| 日期 | 主题 | 开发 | 结果 |
|---|---|---|---|
| 2026-10-08 | 记忆巩固 / 遗忘、扩散激活、双时态矛盾解决（13 篇，仅搜索摘要） | `Claim.since_tick` / `until_tick`、`ConceptCard.first_seen_tick`；两个否定探针（复述、压制共现边） | 1 598 tests；无动力学改动 |
