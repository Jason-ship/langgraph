# NovelFactory 最小单元拆分重构方案（面向 DSH 基座迁移）

> **Goal:** 把 NovelFactory 拆成可独立演进的最小单元，为"DSH 基座 + LangGraph 核心 + 能力插件化"混合架构铺路。
>
> **Architecture:** 三层拆分——L0 清理死代码 → L1 抽零依赖公共库（novelfactory-agent-sdk）→ L2 业务服务化（评审/存储/工具子图/渠道/LLM 基础设施，均可作为 DSH 插件挂载）→ L3 图编排核心保留并声明式化（supervisor 阶段机、verdict 循环、checkpointer、HITL）。DSH 侧通过 `tool-langgraph` 桥接插件调用 NovelFactory 服务。
>
> **Tech Stack:** Python 3.10 / LangGraph / FastAPI / TypeScript(Cordis, DSH) / PostgreSQL / Milvus / Neo4j / Redis

---

## 一、拆分原则（三条）

1. **依赖单向**：L0 → L1 → L2 → L3。上层可依赖下层，下层绝不依赖上层。任何"存储层反向依赖业务层"（如 `store/channel_connection.py` 反向 import `channels/service.py`）都必须消除。
2. **隐式契约显式化**：`crew_result` 是子图↔父图唯一的 dict 管道，无 schema 校验——这是迁移时最易断的接口，必须先定义 `CrewResultSchema`。
3. **执行器抽象**：凡依赖 `graph.ainvoke/astream_events` 的消费方（`channels/manager.py`）改为面向 `RunExecutor` 接口编程，LangGraph 只是其中一个实现。

## 二、最小单元清单

### L0 清理层（无迁移价值，直接删/归档）

| 文件 | 处理 | 依据 |
|---|---|---|
| `src/novelfactory/graph/parallel/volume_dispatch.py` | 删除 | DEPRECATED 死代码（Send 分发已移除） |
| `src/novelfactory/graph/nodes/._supervisor.py` | 删除 | AppleDouble 元数据垃圾文件，非代码 |
| `src/novelfactory/graph/nodes/prepare_writing.py` 的 `feishu_upload_node` | 删除 | 死代码（同步已迁入 sync_crew） |
| `src/novelfactory/graph/lightweight_setup.py` 的 `_llm_quality_gate/_persist_volume_structure_to_db/_split_outline` | 删除 | 与 `setup_nodes.py` 重复逻辑 |
| `src/novelfactory/state/chapter_state.py` | 核验引用后删除 | 内存对象，功能已被 PG-backed context_builder 替代 |
| `src/novelfactory/integrations/minimax/` | 归档 | 空包（DeepSeek 已替换 MiniMax） |
| `src/novelfactory/model_factory.py` | 归档 | 与 `config/llm.py` 功能重叠 |
| `src/novelfactory/evaluation/elo/` | 归档 | 无外部调用（v8.2 辩论移除后遗留） |
| `src/novelfactory/evaluation/llm/` | 收敛进 unified | 仅 `prompts.py(trim_text)` 与 `verdict/feedback.py` 引用 |

### L1 零依赖公共库（抽为 `novelfactory-agent-sdk` 包）

| 单元 | 文件 | 接口 |
|---|---|---|
| LLM 调用包装（同步/异步重试） | `agents/infra/retry.py`、`async_retry.py`、`_retry_common.py`、`timeout.py` | `async_llm_call_with_retry(func, step_name, ...)` |
| 熔断器 | `agents/infra/circuit_breaker.py` | `circuit_breaker_record_success/failure/is_open` |
| 语义缓存 | `agents/infra/llm_cache.py` | `LLMResponseCache.get/set` |
| 用量追踪 | `agents/infra/usage.py` | `read_usage_tracking/reset_usage_tracking` |
| 流与日志 | `agents/infra/stream.py`、`logger.py`、`helpers.py`、`serialization.py`、`context_compressor.py` | `get_crew_stream` 等 |
| 纯函数状态工具 | `state/reducers.py`、`human_input.py`、`novel_context.py` | reducer / TypedDict 校验 |
| 文件评审队列 | `crews/review_queue.py` | `review_queue.add/decide` |
| 定价/配额 | `config/pricing.py`、`config/quota.py` | `calc_cost` / `quota_settings` |

**验证**：新包独立跑通单测（不 import 任何 LangGraph/graph 模块）。

### L2 服务化单元（可独立服务 / DSH 插件）

| 单元 | 来源 | 对外最小接口 | 迁移形态 |
|---|---|---|---|
| 评审服务 | `evaluation/service.py` + `verdict/` + `unified/` | `ReviewService.evaluate()`、`UnifiedReviewEngine.quick_recheck()`、`CheckpointReplayService.replay_chapters()` | DSH 工具 ×3，契约 `evaluation/schemas.py` 原样复用 |
| 存储五插件 | `store/protocols.py` + `postgres_store/milvus_store/neo4j_store/redis_store/embedding` | `RelationalStore/VectorStore/GraphStore/KVStore/EmbeddingProvider` | DSH storage 插件（先合并双 Milvus 客户端、双 skills 实现、双源定价） |
| 上下文/提取/写库 worker | `graph/subgraphs/context_builder.py`、`state_extractor.py`、`database_writer.py` | `build_writer_context(chapter)` / `extract(chapter)` / `persist(chapter)` | asyncio worker 服务（LangGraph 仅留顺序语法糖）；修复 `database_writer.py` 自 import 循环 |
| 渠道插件 SDK | `channels/base.py`、`message_bus.py`、`commands.py`、`run_policy.py`、`feishu_run_policy.py`、`store.py`、`runtime_config_store.py`、`connection_identity.py` | `ChannelProvider`（生命周期 + 入站/出站协议） | DSH 插件平移；`manager.py` 改面向 `RunExecutor`；卡片文案抽象为回调 |
| LLM 基础设施插件 | `config/llm.py` + `llm_params.py` + L1 七件套 | 工厂×5 / `center.get_params` / 调用包装 | DSH `llm/stream` waterfall 插件；修复 `preferred_provider="ark"` 遗留默认 |
| 媒体/同步桥接 | `graph/crews/media_crew.py`、`sync_crew.py` | `render_media(chapter)` / `sync_feishu(chapter)` | HTTP 服务；解耦 `sync_agents.update_project_state` 对 checkpointer 的直写 |
| 监控旁路 | `graph/monitor_node.py` | `build_monitoring_snapshot(state)` | 订阅式旁路服务 |
| 流水线质检 | `pipeline/phase2_manager.py`、`scale_manager.py`、`phase3_manager.py`、`narrative_codec/` | `build_writer_context(chapter)` / `after_chapter(...)` | 业务编排保留；合并 VolumeManager/OutlineManager 重叠查询；弧线模板外置 |
| 配置三件套 | `config/settings.py` + `quality_params.py` + `constants.py` | env 单一入口 / 运行时覆盖回滚 / 默认值注册表 | 修复 quality_params"文档称 Redis 实现为 JSON"的不一致；Layer 1 升级 Redis |
| 调度/子代理执行框架 | `scheduler.py`、`subagents.py` | `register/start/stop`、`execute/cancel` | DSH schedule/jobs 插件；补 cron 解析语义 |

### L3 图编排核心（保留 LangGraph，声明式化）

| 单元 | 说明 |
|---|---|
| `graph/new_builder.py` 装配 | 硬编码装配 → 声明式 spec（节点/边/路由表/状态映射），LangGraph 仅运行时 |
| `graph/nodes/supervisor.py` + `routing.py` | 阶段状态机与条件边（不可替代） |
| `graph/crews/writing_crew.py` verdict 循环 | REFINE/REWRITE/PASS 三路 + 上限保护（质量控制核心） |
| `state/novel_state.py` + `reducers.py` + `managed_values.py` | 根 schema + checkpoint 序列化（注意 managed_values 依赖 langgraph 私有 API） |
| `graph/checkpointer.py` | Postgres/Redis 断点恢复（支撑"原线程断点继续"） |
| `graph/nodes/review.py` HITL | interrupt + 飞书回调恢复 |

## 三、阶段任务

### P0：清理（0.5-1 天）
- [ ] 删除 L0 全部死代码/垃圾文件/重复逻辑
- [ ] 归档 elo/、minimax/、model_factory.py
- [ ] 收敛 `evaluation/llm/` 进 unified
- 验证：全量测试通过（`pytest tests/ -q`），`ruff check src/` 无新增

### P1：抽零依赖 SDK（3-5 天）
- [ ] 新建 `packages/novelfactory-agent-sdk`（或 `src/agent_sdk/`）：L1 全部单元
- [ ] 定义 `CrewResultSchema`（隐式契约显式化第一步）
- [ ] 收敛双源定价（`usage.py` 常量 → `config/pricing.py` 单一来源）
- 验证：SDK 单测独立通过；主项目改为 import SDK 后行为不变（diff 为空）

### P2：服务化 + 执行器抽象（1-2 周）
- [ ] 评审服务独立模块（`ReviewService` 三接口）+ 单测
- [ ] 存储五插件按 `protocols.py` 拆分；合并双 Milvus/双 skills/双定价；修复 `channel_connection.py` 反向依赖
- [ ] 定义 `RunExecutor` 接口；`channels/manager.py` 改面向接口
- [ ] 三工具子图改写为 asyncio worker；修复 `database_writer.py` 自 import
- 验证：写作图用新服务实现跑通 3 章（复用既有端到端测试）

### P3：DSH 桥接验证（1 周）
- [ ] DSH 侧写 `tool-langgraph` 插件（线程创建/发起 run/进度流/章节拉取）
- [ ] DSH headless 端到端：发起写作 → 读进度 → 拿章节 → 评审工具调用
- [ ] 渠道插件 SDK 平移为 DSH 插件包 `novelfactory-dsh-plugin`
- 验证：DSH 会话中完整驱动一次 3 章写作，事件轨迹可见

### P4：核心声明式化 + 网关切换（2-3 周）
- [ ] `new_builder.py` 改造为声明式 spec
- [ ] 删除 `stub_router.py`（~40 假端点），DSH web UI 接管交互
- [ ] `memory.py`/`feedback.py` 落库（现为纯内存）
- [ ] `quality_params.py` Layer 1 升级 Redis
- 验证：DSH UI 全流程写作 + 飞书同步 + 断点恢复

## 四、关键风险

| 风险 | 应对 |
|---|---|
| `crew_result` 隐式契约断裂 | P1 先行定义 `CrewResultSchema`，双向校验 |
| `sync_agents.update_project_state` 直写 checkpointer | P2 解耦为显式服务调用 |
| `managed_values.py` 依赖 langgraph 私有 API | 升级时改用派生状态或 pinned 版本 |
| DSH 预览版接口变动 | 桥接层保持薄接口（仅 HTTP 转发），隔离变化 |
| `workflow` 无声明式图 | 写作管线永不迁入 DSH，只做编排与交互 |

## 五、迁移顺序总览

```
P0 清理 → P1 SDK 抽离 → P2 服务化（评审/存储/执行器） → P3 DSH 桥接验证 → P4 核心声明式化+UI 切换
```
每阶段结束均可独立交付、可回滚。
