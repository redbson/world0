# World 0 — Architecture / 架构图

> Scope: the **World 0 core cognitive system** (the `World` facade and everything beneath it).
> The PKM / Agent application layer (PKMAgent, CLI/Web/GUI, MCP, tools, research, external, sessions) is **out of scope** for this diagram — it lives in the separate [`world0-pkm`](../../world0-pkm) project (the `pkm` package), which depends on this core.
>
> 范围：**World 0 核心认知系统**（`World` 统一接口及其下层）。PKM / Agent 应用层位于独立项目 [`world0-pkm`](../../world0-pkm)（`pkm` 包），不在本图范围内。

## 1. Component Architecture / 组件架构

```mermaid
flowchart TB
    pkm["PKM / Agent application layer<br/><i>(out of scope / 不在本图范围)</i>"]:::ext

    subgraph FACADE["world/ — World Facade + Pipelines / 统一接口与管线"]
        direction TB
        world["<b>World</b> (facade.py)<br/>ingest · ingest_text · project · reflect<br/>merge · split · weaken · status · visualize"]
        ingestp["IngestPipeline<br/>(_ingest.py)"]
        reflectp["ReflectPipeline<br/>(_reflect.py)"]
        identity["IdentityOps<br/>(_identity.py) merge/split/weaken"]
        status["build_status<br/>(_status.py)"]
        world --> ingestp & reflectp & identity & status
    end

    subgraph DOMAIN["Cognitive Domain Managers / 认知领域管理器"]
        direction LR
        concepts["<b>Concepts</b> (concepts/)<br/>lifecycle · identity<br/>signature dedup · merge/split<br/>aliases · indexes"]
        relations["<b>Relations</b> (relations/)<br/>typed 3-axis relations<br/>weighting · lifecycle"]
        sources["<b>SourceLibrary</b> (sources/)<br/>raw source provenance"]
        communities["<b>Communities</b> (communities/)<br/>community persistence"]
        spaces["<b>Spaces</b> (spaces/)<br/>isolated concept worlds"]
    end

    subgraph DYN["dynamics/ — Cognitive Dynamics Engines / 认知动力学"]
        direction LR
        activation["ActivationEngine<br/>spreading activation / 扩散激活"]
        hebbian["HebbianEngine<br/>co-occurrence learning"]
        decay["DecayEngine<br/>maturity-based decay"]
        lifecycle["LifecycleEngine<br/>maturity transitions"]
        colordiff["ColorDiffusionEngine<br/>community color field / 色场"]
        commdet["CommunityDetector<br/>community detection"]
        coeff["coefficients.py<br/>tunable constants"]
    end

    subgraph COGOUT["Projection · Extraction · Metrics / 投影·提取·诊断"]
        direction LR
        projection["<b>ProjectionEngine</b> (projection/)<br/>spreading activation + MMR<br/>→ task-relevant local view"]
        extractor["<b>ConceptExtractor</b> (extraction/)<br/>LLM-powered text → observation"]
        metrics["entropy (metrics/)<br/>network-entropy diagnostics"]
    end

    subgraph INFRA["Infrastructure / 基础设施 (cross-cutting)"]
        direction LR
        schemas["schemas/<br/>concept · relation · types<br/>context · community · space · source"]
        store["store/<br/>Store (base) · JsonStore<br/>pluggable persistence"]
        llm["llm/<br/>base · openai · anthropic · azure_openai"]
        prompts["prompts/<br/>configurable prompt registry"]
        models["models/<br/>per-operation model config"]
        viz["visualization/<br/>interactive HTML graph"]
    end

    pkm -.->|uses 4-method API| world

    ingestp --> concepts & relations & hebbian & colordiff
    reflectp --> decay & lifecycle & colordiff & communities
    identity --> concepts & relations
    status --> concepts & relations & communities

    world --> activation --> projection
    world --> extractor
    extractor --> sources
    world --> viz

    activation --> concepts & relations
    colordiff --> concepts & relations
    commdet --> concepts & relations
    communities --> commdet
    projection --> concepts & relations

    concepts & relations & sources & communities --> store
    store --> schemas
    extractor --> llm
    extractor --> prompts
    llm --> models

    classDef ext fill:#eee,stroke:#999,stroke-dasharray:5 5,color:#555;
    classDef facade fill:#dCE9ff,stroke:#3b7;
    class world,ingestp,reflectp,identity,status facade;
```

## 2. Operational Data Flow / 操作数据流

The system is an **operational chain**, not a CRUD store. / 系统是一条操作链，而非 CRUD 存储。

```mermaid
flowchart LR
    obs["Observation<br/>(concepts + relations + task)"]
    text["Raw text / 原始文本"]

    text -->|ingest_text| ex["ConceptExtractor<br/>(LLM extraction)"]
    ex --> obs
    ex -.records.-> src["SourceLibrary<br/>provenance"]

    obs -->|ingest| ing["IngestPipeline"]
    ing --> cr["concepts + relations<br/>+ Hebbian + color"]
    cr -->|flush| disk[("JsonStore<br/>(disk / 磁盘)")]

    seeds["Seeds + Task<br/>种子 + 任务"] -->|project| act["ActivationEngine<br/>扩散激活"]
    act --> mmr["ProjectionEngine<br/>MMR selection"]
    mmr --> proj["Projection"]
    proj -->|.render| md["markdown<br/>→ inject into Agent prompt"]

    refl["reflect()"] --> dec["DecayEngine"] --> lc["LifecycleEngine"] --> cd["ColorDiffusion"] --> comm["Communities"]
    comm -->|flush| disk

    cr -.reads.-> act
    disk -.loads.-> act
```

### Concept Lifecycle / 概念生命周期

```mermaid
stateDiagram-v2
    direction LR
    [*] --> embryonic
    embryonic --> developing: act≥3, conf≥0.3
    developing --> established: act≥10, conf≥0.6
    established --> core: act≥30, conn≥5
    embryonic --> fading: conf<0.05
    developing --> fading: conf<0.05
    established --> fading: conf<0.05
    fading --> developing: re-activated / 重新激活
```
