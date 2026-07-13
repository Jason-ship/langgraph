# NovelFactory - 系统架构

## 概述

NovelFactory 是基于 LangGraph 的多智能体小说创作系统。根图由 18+ 节点组成，通过 Main Supervisor 条件路由编排 5 个 Crew 子图，覆盖从项目设定到章节同步的全流程。

## 根图架构

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
    wait_for_review         │        writing_crew ──-> intelligent_monitor
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
                            │     ▼      -> writing    -> save_memory
                            │  main_supervisor
                            │
                              save_memory
                                    │
                                   END
```

## 路由表 (route_from_supervisor)

| 条件 | 目标节点 |
|------|---------|
| phase=setup, !setup_complete | setup_crew |
| phase=setup, pending_review=kickoff | wait_for_review |
| phase=setup, complete | load_memory |
| phase=writing, chapter_needs_guidance | chapter_human_guidance |
| phase=writing, pending_review=chapter | wait_for_review |
| phase=writing, current_chapter>1 | check_chain[0] (体裁过滤) |
| phase=writing, current_chapter<=1 | refresh_quota |
| phase=media | media_crew |
| phase=sync | sync_crew |
| phase=done | save_memory |
| unknown | END (警告日志) |

Phase check 链 (build_check_chain)：`volume_check -> quality_check -> foreshadowing_check`（按体裁过滤）

## 子图结构

### Writing Crew（7节点）

核心创作子图，包含写作、评审、重写/润色循环。

```
START -> context_builder -> chapter_writer -> verdict_engine
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

### Setup Crew（9节点）

项目初始化子图，生成世界观、角色、大纲。

```
init_setup -> world_builder -> character_designer -> outline_writer
-> volume_detail_writer -> quality_gate -> feishu_setup -> db_persist -> setup_finalize -> END
```

### Media Crew（2节点并行）

媒体生成子图，使用 ThreadPoolExecutor 真并行。

```
START -> _parallel_media_node ──> illustrator_agent (插图生成)
                             └──> tts_agent (配音生成)
        ──> END
```

### Sync Crew（3节点）

飞书同步子图，将章节内容推送到飞书文档。

```
START -> sync_crew_node -> feishu_sync -> END
```

## 评审引擎 (VerdictEngine)

评审已从 LangGraph 辩论子图重构为 `evaluation/` 模块中的结构化评分管线。

### 评审流程

```
1. 程序化分析（纯代码，毫秒级）
   ├── ai_style_sensor: 8维AI味检测
   │   (N-gram / 句长波动 / 词汇多样性 / 模板化 / 标点节奏 / 对白比例 / 感官词 / 语义平滑)
   ├── old_reader_sensor: 老书虫视角评分（爽点/毒点）
   └── cross_chapter_sensor: 跨章一致性检测（角色声音/文风/节奏/情节连贯/伏笔）

2. 知情辩论（LLM，注入程序化结果）
   ├── editor_review -> reader_review -> debate_router -> 单轮收敛
   └── 产出: DebateReport (issues / strengths / severity_weight / transcript)

3. 四维 LLM 评分（1次）
   ├── 文学性(30) + 结构(25) + 角色(20) + 节奏(15) = 100
   └── 含跨章一致性维度(0-100)

4. 融合计算 -> 校准 -> 决议
```

### 融合公式

```
final_score = quality_score × 0.40
            + programmatic_normalized × 0.30
            + cross_chapter_consistency × 0.20
            - debate_penalty × 0.10
```

### 三路决策

| final_score | level | 路由 | 说明 |
|-------------|-------|------|------|
| >=75 | PASS | __exit_for_chapter__ | 通过 |
| >=55 | REFINE | chapter_refiner | 需润色 |
| <55 | REWRITE | chapter_writer | 重写 |
| 严重毒点 + 未用尽重写 | REWRITE | chapter_writer | 毒点强制重写 |
| 次数用尽 | PASS | __exit_for_chapter__ | 防死循环 |

题材阈值详见 [SCORING.md](SCORING.md)。

## 关键状态字段

### NovelFactoryState（根图）

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_phase` | str | setup/writing/media/sync/done |
| `current_chapter` | int | 当前章节号 |
| `target_chapters` | int | 目标总章节数 |
| `completed_chapters` | Annotated[list, _add_chapters_compressed] | 已完成章节列表 |
| `messages` | Annotated[list, add_messages] | 消息列表 |
| `total_usage` | Annotated[dict, _add_usage] | Token/成本统计 |
| `crew_result` | Annotated[dict, _last_value] | Crew 间数据传递 |

### WritingCrewLocalState

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_chapter` | int | 当前章节号 |
| `loop_count` | int | 重写次数 |
| `refine_attempts` | int | 润色次数 |
| `chapter_draft` | str | 章节草稿 |
| `quality_score` | float | 质量评分 |
| `composite_score` | float | 综合评分 |
| `verdict_result` | VerdictResult | 评审结果 |
| `final_score` | float | 最终分数 |
| `debate_transcript` | str | 辩论记录 |

## 检查点系统

- **持久化**：PostgreSQL (pgvector:pg16)，LangGraph Checkpointer
- **子图无检查点**：子图编译不传 checkpointer，由根图统一管理
- **中断恢复**：支持 interrupt 中断后从检查点恢复
- **时间旅行**：支持状态回溯与分支（`api/time_travel.py`）
- **终态清理**：`save_memory` 节点执行检查点终态清理

## 中间件链

| 中间件 | 职责 |
|--------|------|
| `summarization` | 长文本摘要压缩 |
| `large_file_storage` | 大文件存储到 MinIO |
| `skill_injection` | Skill 动态注入 |
| `todo_list` | 待办事项管理 |

## 模块职责

| 模块 | 职责 |
|------|------|
| `graph/` | 图构建、路由、检查点、节点函数、Crew 子图 |
| `agents/` | Agent 定义（写作/评审/设定/媒体/同步） |
| `evaluation/` | 评审融合引擎（VerdictEngine/辩论/程序化评分） |
| `state/` | 状态定义（NovelFactoryState/CrewState/ChapterState） |
| `server/` | FastAPI 服务（路由/SSE 流式/序列化） |
| `config/` | 配置层（LLM/数据库/常量/定价/配额） |
| `store/` | 持久层（PostgreSQL/Milvus/Neo4j/Redis） |
| `pipeline/` | 创作管线（叙事编解码器/缩放管理器） |
| `integrations/` | 外部集成（飞书 21 域工具集/MiniMax） |
| `middleware/` | 中间件链 |
| `tools/` | LangChain @tool（Neo4j/Milvus/飞书） |
| `schemas/` | Pydantic Schema |
| `skills/` | Skill 加载器 |
| `cli/` | CLI 命令行工具 |
| `api/` | 时间旅行 API |
