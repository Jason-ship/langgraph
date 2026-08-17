---
alwaysApply: true
description: "项目核心规范，全局生效。项目架构、技术栈、目录结构、命名规范、核心文件、依赖版本。问项目干什么的、用了哪些技术、功能在哪文件、核心文件、参考源码时触发。根图/WritingCrew/QualityPanel辩论子图、P0/P1文件、技术栈（LangGraph/FastAPI/PostgreSQL/Milvus/Neo4j/Redis/MinIO）、依赖版本、命名规范、research源码。"
---
# LangGraph 小说工厂 — 项目核心规范

**版本：** v4.0.0
**生效方式：** 始终生效
**优先级：** ⭐⭐⭐⭐⭐

---

## 一、项目概述

### 1.1 项目信息

- **项目名称**：LangGraph NovelFactory
- **项目类型**：多智能体小说创作服务器
- **主技术栈**：LangGraph（>=1.1.0）
- **框架**：FastAPI
- **部署方式**：Docker Compose（Docker Desktop）
- **当前版本**：8.0.0

### 1.2 技术栈

| 组件 | 技术 | 职责 |
|------|------|------|
| **智能体框架** | **LangGraph** >=1.1.0 | 多智能体工作流编排 |
| LLM | DeepSeek V4 Flash | 写作/审查/编排 |
| 检查点存储 | PostgreSQL (pgvector:pg16) | LangGraph 状态持久化 |
| 缓存/队列 | Redis 6-alpine | LLM 缓存、消息队列 |
| 向量检索 | Milvus v2.4.17 | 章节语义搜索 |
| 图数据库 | Neo4j 5 | 角色关系网络 |
| 对象存储 | MinIO | 多媒体文件存储 |
| 外部代理 | New-API | LLM API 统一代理（ARK + DeepSeek 自动回退） |
| Web 服务 | FastAPI + Uvicorn | API 服务 + SSE 流式 |

---

## 二、系统架构

### 2.1 根图架构（Main Supervisor）— v5.5

```
                          START
                            │
                      main_supervisor
                            │
              ┌─────────────┼──────────────┬──────────┬──────────┐
              ▼             ▼              ▼          ▼          ▼
         setup_crew    load_memory    refresh_quota volume_check quality_check
              │             │              │          │          │
              ▼             ▼              ▼          ▼          ▼
         main_supervisor    │        prepare_writing  quality_check foreshadowing
              │             │              │          (NodeSpec   (NodeSpec
              ▼             │              ▼          动态注册)    动态注册)
    wait_for_review         │        writing_crew ──→ intelligent_monitor
    (interrupt)             │              │                         │
              │             │              ▼                         ▼
              ▼             ▼        main_supervisor ────────────────┘
         main_supervisor    │              │
                            │              ▼
                            │        media_crew
                            │              │
                            │              ▼
                            │        main_supervisor
                            │              │
                            │     ┌────────┴────────┐
                            │     ▼                 ▼
                            │  volume_dispatch    sync_crew
                            │  (Send 并行分发)     (FeishuToolkit)
                            │     │                 │
                            │     ▼                 ▼
                            │  chapter_collector  main_supervisor
                            │     │                 │
                            │     ▼          ┌──────┴──────┐
                            │  volume_review  ▼             ▼
                            │     │     (more ch.)    (all done)
                            │     ▼      → writing    → save_memory
                            │  main_supervisor
                            │
                              save_memory
                                    │
                                   END
```

### 2.2 Writing Crew 子图（v7.0 VerdictEngine 融合评审）

```
  START → context_builder → chapter_writer → verdict_engine
                                                  │
                                    _score_router (双条件判定)
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       ▼
                   chapter_planner         chapter_refiner     state_extractor
                   (score<55, 重写)        (score 55-89, 润色)  (双条件通过)
                          │                       │                       │
                          ▼                       ▼                       ▼
                   verdict_engine          verdict_engine         database_writer
                   (重新评审)               (重新评审)                  │
                                                                        ▼
                                                              __exit_for_chapter__
                                                                        │
                                                                       END
```

### 2.3 InformedDebate 评审引擎（v7.0，替代 v5.5 QualityPanel）

评审已从 LangGraph 辩论子图重构为 `evaluation/` 模块中的结构化评分管线，核心引擎 `VerdictEngine` 融合多源评分：

- **四维评分**：文学性 30 + 结构 25 + 角色 20 + 节奏 15 = 90 基础分 + 10 整体
- **程序化评分**：AI 味 8 维检测 + 老书虫爽点/毒点 + 跨章一致性传感器
- **LLM 语义评分**：老书虫 LLM 评审 + AI 味 LLM 检测（失败降级为程序化）
- **辩论评分**：`evaluation/debate/engine.py` — `InformedDebateEngine` 编辑↔读者多轮辩论
- **迭代宽松加分**：根据重写/润色次数加 VerdictWeight[iteration_bonus]
- **质量衰减**：章节 > 2 时按衰减率（`QUALITY_DECAY_RATE`）扣分

参考 TradingAgents Bull↔Bear 辩论模式：
- 编辑视角：四维评分（文学性/结构/角色/节奏）
- 读者视角：爽点/代入感/AI味/毒点
- 评分差异 < 10 分 → 提前收敛
- 最多 3 轮辩论

---

## 三、目录结构（精简版）

```
langgraph/
├── src/novelfactory/          # ★ 核心源码
│   ├── graph/                 # 图构建层       → 详见 02-langgraph-development.md
│   ├── agents/                # Agent 定义层    → 详见 03-agent-patterns.md
│   ├── state/                 # 状态定义        → 详见 02-langgraph-development.md
│   ├── schemas/               # Pydantic Schema → 详见 03-agent-patterns.md
│   ├── store/                 # 持久层          → 详见 05-api-server.md
│   ├── server/                # FastAPI 服务    → 详见 05-api-server.md
│   ├── config/                # 配置层          → 详见 05-api-server.md
│   ├── pipeline/              # 创作管线        → 详见 05-api-server.md
│   ├── analysis/              # 质量分析        → 详见 04-quality-scoring.md
│   ├── integrations/          # 外部集成        → 详见 05-api-server.md
│   ├── tools/                 # LangChain @tool → 详见 05-api-server.md
│   ├── utils/                 # 工具            → 详见 03-agent-patterns.md
│   ├── middleware/            # 中间件          → 详见 05-api-server.md
│   ├── crews/                 # Crew Supervisor → 详见 02-langgraph-development.md
│   ├── skills/                # Skill 加载器    → 详见 07-operation-rules.md
│   ├── cli/                   # CLI 命令        → 详见 07-operation-rules.md
│   ├── api/                   # 时间旅行 API    → 详见 05-api-server.md
│   └── scripts/               # 数据库迁移脚本  → 详见 06-deployment-operations.md
├── deploy/                    # 部署配置        → 详见 06-deployment-operations.md
├── research/                  # 参考源代码（只读）
├── tests/                     # 测试
├── .trae/                     # 工作区规则
│   ├── rules/                 # 本文件所在目录
│   ├── skills/                # 本地 skill
│   └── specs/                 # 功能规格文档
├── langgraph.json             # LangGraph 配置
├── pyproject.toml             # 项目依赖
├── Dockerfile                 # 容器镜像
├── docker-compose.yml         # 服务编排
└── Makefile                   # 构建任务
```

---

## 四、核心文件优先级

| 优先级 | 文件 | 作用 | 修改频率 |
|--------|------|------|----------|
| P0 | `graph/new_builder.py` | 根图构建（18 节点+Send） | 高 |
| P0 | `graph/routing.py` | 路由函数（含 phase check chain） | 高 |
| P0 | `graph/node_specs.py` | NodeSpec 动态注册表 | 中 |
| P0 | `graph/crews/writing_crew.py` | 写作子图（含 quality_panel） | 高 |
| P0 | `evaluation/verdict/engine.py` | VerdictEngine 融合评审 | 高 |
| P0 | `evaluation/coordinator.py` | 评审协调器（verdict_engine_node） | 高 |
| P0 | `state/novel_state.py` | 全局状态定义 | 中 |
| P0 | `graph/checkpointer.py` | 持久化层 + 检查点清理 | 低 |
| P1 | `schemas/review_schemas.py` | 结构化评审 Schema | 低 |
| P1 | `graph/parallel/volume_dispatch.py` | Send 并行分发 | 中 |
| P1 | `server/app.py` | API 服务 | 低 |
| P1 | `server/streaming.py` | SSE 流式 + Agent 状态追踪 | 中 |
| P1 | `evaluation/debate/engine.py` | InformedDebate 辩论引擎 | 中 |
| P1 | `config/constants.py` | 全局常量中心化 | 中 |
| P1 | `config/settings.py` | 配置层（含 env 覆盖） | 中 |
| P1 | `config/llm.py` | LLM 配置 | 中 |
| P1 | `utils/wall_time_tracker.py` | 全链路耗时/Token/LLM 统计 | 低 |
| P1 | `agents/infra/` | 基础设施模块组 | 中 |
| P1 | `integrations/feishu/feishu_toolkit.py` | 飞书工具箱 | 低 |

---

## 五、关键依赖版本

```
langgraph >=1.1.0, <2
langgraph-checkpoint-postgres >=3.1.0
langchain-openai >=1.0.0
psycopg + psycopg-pool >=3.1.0
pymilvus >=3.0.0
neo4j >=6.0.0
fastapi + uvicorn >=0.110.0
pydantic >=2.5.0
lark-cli >=1.0.57
```

---

## 六、命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 目录名 | kebab-case | `postgres-store/` |
| 文件名 | snake_case | `novel_state.py` |
| 类名 | PascalCase | `NovelFactoryState`, `FeishuToolkit` |
| 函数名 | snake_case | `route_from_supervisor()` |
| 常量 | UPPER_SNAKE_CASE | `MAX_REWRITE_ATTEMPTS` |
| 私有成员 | 前缀下划线 | `_last_value()` |

---

## 七、多智能体并行模式

### 7.1 并行模式全景

| 模式 | 实现方式 | 并行度 | 适用场景 | 代码位置 |
|------|---------|:------:|---------|---------|
| **编排型并行** | Main Supervisor 条件路由 | 串行调度 | 阶段切换（setup→writing→media→sync） | `graph/routing.py` |
| **子图独立Agent** | `add_node(compiled_subgraph)` | 独立运行 | writing/media/sync/setup 子图 | `graph/new_builder.py` |
| **ThreadPoolExecutor 真并行** | `ThreadPoolExecutor(max_workers=2)` | 2 | media_crew 插图+配音同时生成 | `graph/crews/media_crew.py` |
| **评审子Agent并行** | VerdictEngine 统一编排 + try/except 降级 | 4（逻辑并行） | 程序化/LLM老书虫/LLM AI味/辩论 | `evaluation/verdict/engine.py` + `evaluation/coordinator.py` |
| **辩论式并行** | 编辑↔读者多轮辩论 | 2 views × 3 rounds | 章节质量定性评审 | `evaluation/debate/engine.py` |
| **检查链** | NodeSpec 动态注册 + 条件路由 | 按题材过滤 | volume_check/quality_check/foreshadowing | `graph/node_specs.py` |
| **扇出边** | `add_edge([A,B], C)` | 静态扇出 | 多节点汇聚 | `graph/new_builder.py` |
| ~~Send Map-Reduce~~ | ~~LangGraph Send API~~ | ~~动态分发~~ | ~~卷级并行写作（v6.3 已移除）~~ | ~~`graph/parallel/volume_dispatch.py`~~ |

### 7.2 模式详解

#### 7.2.1 编排型并行（Main Supervisor）

Main Supervisor 作为中央编排器，根据 `current_phase` 将控制权切换到不同子图。各子图按阶段串行执行，不涉及真正的时间并行，但实现了"多智能体分工协作"的逻辑并行。

```
main_supervisor (路由决策)
  ├── setup_crew      → 项目初始化（多Agent协作）
  ├── writing_crew    → 章节创作（含质量评审）
  ├── media_crew      → 媒体生成（真并行）
  └── sync_crew       → 飞书同步
```

#### 7.2.2 ThreadPoolExecutor 真并行（media_crew）

`_parallel_media_node` 中使用 `ThreadPoolExecutor(max_workers=2)` 同时执行插图和配音生成：

```python
with ThreadPoolExecutor(max_workers=2, thread_name_prefix="media_crew") as pool:
    future_ill = pool.submit(illustrator_agent, illustrator_input)
    future_tts = pool.submit(tts_agent, tts_input)
    illustrator_result = future_ill.result()
    tts_result = future_tts.result()
```

**约束**：
- Agent 必须是同步 Runnable（非 async）
- 失败时自动重试（最多 3 次）
- 单 Agent 失败不影响另一个

#### 7.2.3 评审子Agent并行（逻辑并行）

`VerdictEngine.evaluate()` 统一编排 4 个评审维度（v7.0 重构，替代旧 `run_parallel_review`）：

```
run_programmatic_analysis ──-> 程序化分析（AI味8维 + 老书虫爽点/毒点 + 跨章一致性）
llm_old_reader_analysis   ──-> LLM 老书虫语义评分
llm_ai_style_analysis     ──-> LLM AI味人类相似度
InformedDebateEngine.run  ──-> 编辑↔读者多轮辩论
```

每个维度有独立的 try/except 保护：
- 单维度失败 -> 降级为程序化评分或默认值，不阻塞整体
- 4 个维度地位平等，互不依赖
- 辩论结果作为定性分析，产出问题清单 + 严重度权重

#### 7.2.4 辩论式并行

Editor Reader 多轮辩论（最多 3 轮）：

```
editor_review ──→ reader_review ──→ debate_router
                                        │
                ┌───────────────────────┼──────────────────────┐
                ▼                       ▼                      ▼
          editor_rebuttal        reader_rebuttal         quality_gate
                │                       │                      │
                └───────────────────────┘                      │
                          │                             debate_converge
                          ▼                                     │
                    debate_router                               END
                  (max 3 rounds)
```

- 评分差异 < 10 分 → 提前收敛
- 3 轮后未收敛 → 自动取平均分

### 7.3 并行设计原则

| 原则 | 说明 |
|------|------|
| **容错优先** | 单 Agent 失败不阻塞整体（try/except + 默认值降级） |
| **无共享状态** | 并行 Agent 通过 state 通信，无共享内存变量 |
| **父图统一持久化** | 子图不传 checkpointer，由根图统一管理检查点 |
| **Reducer 防冲突** | 多节点写入同一字段使用 `_last_value` / `add_messages` |
| **递归上限保护** | 根图 5000 / 子图 200，防死循环 |

---

## 八、参考源代码

项目 `research/` 目录下保存了四份重要的上游参考源代码，**只读不可修改**。

### 7.1 research/cli-main/ — lark-cli 官方源码

| 用途 | 查阅路径 |
|------|---------|
| 理解 shortcut 命令参数和实现 | `shortcuts/<domain>/` |
| 查看飞书 API 调用方式 | `internal/client/` |
| 查看命令注册结构 | `shortcuts/register.go` |
| 查看技能定义 | `skills/<skill-name>/SKILL.md` |

### 7.2 research/langgraph-main/ — LangGraph 官方源码

| 用途 | 查阅路径 |
|------|---------|
| 查看 StateGraph/Pregel 实现 | `libs/langgraph/langgraph/graph/` |
| 查看 Checkpointer 接口 | `libs/checkpoint/langgraph/checkpoint/` |
| 查看 Postgres 检查点实现 | `libs/checkpoint-postgres/langgraph/checkpoint/postgres/` |
| 查看 CLI 工具实现 | `libs/cli/langgraph_cli/` |

### 7.3 research/langgraph-tutorial-wenwenc9/ — LangGraph 中文教程

| 章节 | 内容 | 文件 |
|------|------|------|
| 快速入门 | StateGraph 基础用法 | `1-基础章节/1、快速入门.ipynb` |
| 综合案例 | ReAct Agent、工具调用 | `1-基础章节/2、综合案例.ipynb` |
| 持久化 | Checkpointer 配置 | `2-进阶/1、持久化.ipynb` |
| 流式传输 | Streaming 实现 | `2-进阶/2、流式传输.ipynb` |
| 中断 | interrupt/resume 模式 | `2-进阶/3、中断.ipynb` |
| 时间旅行 | 状态回溯与分支 | `2-进阶/4、时间旅行.ipynb` |

### 7.4 research/TradingAgents/ — 金融参考项目（v5.5+v5.6）

| 模式 | NovelFactory 对应模块 | 来源文件 |
|------|---------------------|---------|
| Bull↔Bear 辩论 | evaluation/debate/engine.py（编辑↔读者辩论） | `agents/risk_mgmt/` |
| NodeSpec 动态注册 | node_specs.py + new_builder.py | `graph/analyst_execution.py` |
| Send Map-Reduce | parallel/volume_dispatch.py | `graph/signal_processing.py` |
| 结构化 Schema | schemas/review_schemas.py | `agents/schemas.py` |
| WallTimeTracker | utils/wall_time_tracker.py | （独立模式） |
| ~~bind_structured 异常保护~~ | ~~agents/utils/structured.py~~（v5.8 已移除，改用 JSON 解析 + validate_json_output） | `agents/utils/structured.py` |
| ~~invoke_structured_or_freetext~~ | ~~agents/utils/structured.py~~（v5.8 已移除，改用 async_llm_call_with_retry + 文本提取） | `agents/utils/structured.py` |
| StreamStateTracker | server/streaming.py | `cli/main.py` |
| _ENV_OVERRIDES | config/settings.py | `default_config.py` |
| checkpoint 终态清理 | graph/checkpointer.py | `graph/checkpointer.py` |
| WallTime LLM 统计 | utils/wall_time_tracker.py | `cli/stats_handler.py` |

### 7.5 注意事项

- research 目录**只读不可修改** — 上游源码快照，修改破坏版本匹配
- 优先查阅 research 而非在线搜索，源码版本与部署环境一致

### 7.6 项目文档速查

| 文档 | 位置 | 内容 |
|------|------|------|
| 架构全貌（v5.5） | `research/NovelFactory_v5.5_Architecture_Overview.md` | TradingAgents 模式借鉴、图拓扑、版本演进 |
| 项目分析 + SOP | `NovelFactory_v5.5_项目分析_架构_SOP.md` | 完整架构分析、8 个 SOP 流程、FAQ |
| v5.6 TradingAgents 优化 Spec | `.trae/specs/v5.6-tradingagents-optimization/` | Checklist / Spec / Tasks |
| v6.0 重构计划 | `.trae/specs/REFACTOR-PLAN-v6.0.md` | 后续重构方向 |

---

**规则版本：** v4.0.0
**生效方式：** 始终生效
**最后更新：** 2026-08-11
