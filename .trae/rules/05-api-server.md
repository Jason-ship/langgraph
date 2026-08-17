---
alwaysApply: false
description: "API/存储/集成规范，匹配server/**、middleware/**、store/**、integrations/**、tools/**、pipeline/**、config/**。飞书文档/文件夹/lark-cli参数、API端点、SSE流式、中间件、存储层、飞书回调时触发。FastAPI（20+路由）、SSE（8事件）、中间件（5个）、存储（4种）、飞书四层架构（lark-cli/FeishuToolkit/FeishuAPI/@tool）、Pipeline/Config、飞书回调。"
---
# API/存储/集成规范

**版本：** v1.3.0
**生效方式：** 智能生效
**优先级：** ⭐⭐⭐⭐
**匹配模式：** `server/**`, `middleware/**`, `store/**`, `integrations/**`, `tools/**`, `pipeline/**`, `config/**`, `crews/**`, `analysis/**`

---

## 一、FastAPI 服务

### 1.1 服务入口

`server/app.py` — FastAPI 应用实例化入口，核心配置：

```python
class CustomJSONResponse(JSONResponse):
    """JSONResponse that uses _MessageJSONEncoder for AIMessage serialization."""
```

### 1.2 路由文件（server/routes/ 21 个）

| 文件 | 路由前缀 | 职责 |
|------|---------|------|
| `routes/health.py` | `/health`, `/ready`, `/info`, `/metrics`, `/params` | 健康检查、就绪探针、版本信息、指标、动态参数 |
| `routes/threads.py` | `/threads` | 线程 CRUD（创建/查询/更新/删除/复制/历史/状态） |
| `routes/runs.py` | `/threads/{id}/runs` | 写作运行管理（启动/恢复/取消/流式回放） |
| `routes/assistants.py` | `/assistants` | Assistant 管理、图结构/Schema/子图查询 |
| `routes/store.py` | `/store` | 持久化存储 CRUD + 搜索（Item/Namespace） |
| `routes/crons.py` | `/runs/crons` | 定时任务管理（创建/搜索/删除） |
| `routes/feishu_callback.py` | `/feishu` | 飞书交互卡片回调 + 手动线程恢复 |
| `routes/agents.py` | `/agents` | Agent 管理（注册/重命名/删除，v8.5 重命名冲突 → HTTP 409） |
| `routes/skills.py` | `/skills` | Skill 管理 |
| `routes/memory.py` | `/memory` | 长期记忆查询 |
| `routes/branches.py` | `/threads/{id}/branches` | 时间旅行分支管理 |
| `routes/regenerate.py` | `/threads/{id}/regenerate` | 章节重新生成 |
| `routes/compact.py` | `/threads/{id}/compact` | 上下文压缩 |
| `routes/time_travel.py` | `/threads/{id}/time-travel` | 时间旅行 API |
| `routes/token_usage.py` | `/token-usage` | Token 用量查询 |
| `routes/quality_feedback.py` | `/quality-feedback` | 评分反馈（反馈按钮 → 参数调优） |
| `routes/channel_connections.py` | `/channels` | 通道连接管理 |
| `routes/console.py` | `/console` | 控制台 |
| `routes/features.py` | `/features` | 特性开关 |
| `routes/feedback.py` | `/feedback` | 用户反馈 |
| `routes/input_polish.py` | `/input/polish` | 输入润色 |
| `routes/suggestions.py` | `/suggestions` | 写作建议 |

### 1.3 API 基础 URL

| 环境 | 基础 URL |
|------|---------|
| 宿主机访问 | `http://localhost:8123` |
| 容器内访问 | `http://localhost:8000` |

### 1.4 其他服务文件

| 文件 | 职责 |
|------|------|
| `models.py` | Pydantic 数据模型（Assistant、ThreadModel、RunRequest 等） |
| `serialization.py` | `_MessageJSONEncoder` — AIMessage/BaseMessage 序列化 |
| `streaming.py` | SSE 流式处理 + `StreamStateTracker` Agent 状态追踪 |

---

## 二、SSE 流式

### 2.1 StreamStateTracker Agent 状态追踪

`server/streaming.py` 中的 `StreamStateTracker` 负责在流式输出过程中追踪 Agent 进度，参考 TradingAgents MessageBuffer 模式。

**核心接口**：

```python
class StreamStateTracker:
    """Tracks agent progress across SSE stream events with message dedup."""

    def __init__(self):
        self.phase: str = ""              # 当前阶段
        self.current_chapter: int = 0     # 当前章节
        self.total_chapters: int = 0      # 总章节数
        self.current_agent: str = ""      # 当前 Agent
        self.agent_status: str = "pending" # Agent 状态
        self.chapter_preview: str | None = None
        self.quality_score: float | None = None
        self.composite_score: float | None = None
        self._processed_message_ids: set[str] = set()

    def update_from_state(self, state: dict) -> bool:
        """从 state dict 提取进度信息。返回 True 表示有变更。"""

    def to_progress_event(self) -> dict:
        """构建 progress SSE 事件载荷。"""
```

**集成方式**（在 `_stream_run` 中）：

```python
if event.get("type") == "updates":
    for node_name, node_update in data.items():
        if tracker.update_from_state(node_update):
            yield _format_sse("progress", tracker.to_progress_event())
```

**progress 事件格式**：

```json
{
    "event": "progress",
    "data": {
        "phase": "writing",
        "chapter": 5,
        "agent": "chapter_writer",
        "agent_status": "in_progress",
        "chapter_preview": "...",
        "quality_score": 87.5,
        "composite_score": 0.72,
        "total_chapters": 1000
    }
}
```

**并行场景下的 progress 事件**：

多智能体并行执行时，`StreamStateTracker` 逐个追踪每个 Agent 的进度，通过 `agent` 字段区分：

```json
{
    "event": "progress",
    "data": {
        "phase": "media",
        "chapter": 5,
        "agent": "illustrator_agent",
        "agent_status": "in_progress",
        "chapter_preview": "生成插图..."
    }
}
{
    "event": "progress",
    "data": {
        "phase": "media",
        "chapter": 5,
        "agent": "tts_generator",
        "agent_status": "in_progress",
        "chapter_preview": "生成配音..."
    }
}
```

> 注意：线程级并行（ThreadPoolExecutor）中，两个 Agent 的 progress 事件按完成顺序输出，无法保证严格时序。

### 2.2 SSE 事件格式

LangGraph `astream_events(version="v2")` 映射到以下 SSE 事件类型：

| 事件类型 | 数据格式 | 说明 |
|---------|---------|------|
| `values` | 完整 State dict（含 `__interrupt__` 映射） | 节点执行完后的完整状态快照 |
| `updates` | `{nodeName: updateDict}` | 增量更新，`{节点名: 更新字典}` |
| `messages` | `[messageChunk, metadata]` 2-tuple | AI 消息 Token 流 |
| `progress` | 自定义进度对象（见 2.1 节） | StreamStateTracker 生成的阶段性摘要 |
| `interrupt` | `{thread_id, interrupts[], next[]}` | 人机交互中断通知 |
| `metadata` | `{run_id, thread_id}` | 运行元数据 |
| `end` | `{status: "completed"}` | 运行正常结束 |
| `error` | `{message: "..."}` | 运行出错 |

**values 事件** — `setStreamValues(data)`，data 为完整状态字典：

```json
{
    "event": "values",
    "data": {
        "current_phase": "writing",
        "current_chapter": 5,
        "messages": [...],
        "__interrupt__": []
    }
}
```

**updates 事件** — `onUpdateEvent(data)`，data 格式为 `{节点名: 更新字典}`：

```json
{
    "event": "updates",
    "data": {
        "chapter_writer": {
            "messages": [...],
            "chapter_content": "正文内容..."
        }
    }
}
```

**messages 事件** — 消息 Token 流，2-tuple `[消息块, 元数据]`：

```json
{
    "event": "messages",
    "data": [
        {"content": "正在写作...", "type": "ai", "id": "xxx"},
        {}
    ]
}
```

**interrupt 事件** — 人机交互中断：

```json
{
    "event": "interrupt",
    "data": {
        "thread_id": "uuid-xxx",
        "interrupts": [{"value": {...}, "id": "..."}],
        "next": ["wait_for_review"]
    }
}
```

**end / error 事件**：

```json
{"event": "end", "data": {"status": "completed"}}
{"event": "error", "data": {"message": "错误描述"}}
```

---

## 三、中间件架构

### 3.1 中间件文件（5 个）

| 文件 | 类名 | 职责 |
|------|------|------|
| `base.py` | `Middleware`（基类） + `MiddlewareChain`（链管理器） | 三个钩子接口：`before_node` / `after_node` / `modify_system_prompt` |
| `large_file_storage.py` | `LargeFileStorageMiddleware` | 大章节草稿（>8000 字符）自动存入 `tmp/chapters/`，State 中只存路径和预览 |
| `summarization.py` | `SummarizationMiddleware` | 智能上下文压缩，消息数超阈值时用 LLM 摘要中间消息，保留最近 N 条原文 |
| `skill_injection.py` | `SkillInjectionMiddleware` | 根据题材和阶段自动注入 Skill 内容到 system prompt |
| `todo_list.py` | `TodoListMiddleware` | 项目启动时自动生成任务列表，阶段完成时更新状态 |

### 3.2 链式调用顺序

```python
chain = MiddlewareChain()
chain.add(SkillInjectionMiddleware(loader))
chain.add(SummarizationMiddleware())
chain.add(LargeFileStorageMiddleware())
chain.add(TodoListMiddleware())

# 执行顺序：
# 1. before_node: SkillInjection → Summarization → LargeFileStorage → TodoList
# 2. modify_system_prompt: SkillInjection → Summarization
# 3. after_node: 按添加顺序执行
```

所有中间件的 `before_node` 返回的更新合并后注入节点输入；`modify_system_prompt` 链式修改 system prompt；`after_node` 在节点执行后处理。

---

## 四、存储层（Store）

### 4.1 文件清单

| 文件 | 核心技术 | 职责 |
|------|---------|------|
| `postgres_store.py` | PostgreSQL + pgvector (psycopg 3) | 长期记忆存储 + 向量语义搜索；角色状态、章节元数据 |
| `milvus_store.py` | MilvusClient (pymilvus) | 章节语义搜索；向量维度 1024，NLIST 128，TOP_K 3 |
| `neo4j_store.py` | Neo4j 5 (bolt 协议) | 图存储 — 角色关系网络、情节关联图 |
| `redis_store.py` | Redis 7 (redis-py) | LLM 调用缓存、消息队列 |
| `embedding.py` | 嵌入模型 | 向量嵌入生成，用于 Milvus 和 PostgreSQL pgvector 索引 |
| `tracker.py` | — | 小说状态追踪器，记录章节写入/评分/完成进度 |
| `guide_store.py` | — | 写作指导/大纲持久化 |
| `chapter_state_store.py` | — | 跨章节角色状态一致性存储 |

### 4.2 存储策略对比

| 存储 | 用途 | 查询方式 | 容量 |
|------|------|---------|------|
| PostgreSQL | 结构化数据、长期记忆、语义搜索 | SQL + pgvector 向量检索 | 结构化 |
| Milvus | 章节向量检索 | `search(collection, vectors, top_k=3)` | 非结构化 |
| Neo4j | 角色关系图 | Cypher 查询 | 图结构 |
| Redis | 缓存/队列 | Key-Value | 内存 |

---

## 五、飞书集成

### 5.1 四层架构

```
第 1 层: lark-cli (容器内 /usr/local/bin/lark-cli, v1.0.57)
第 2 层: FeishuToolkit (integrations/feishu/feishu_toolkit.py, 209 方法/21 域)
第 3 层: FeishuAPI 兼容层 (integrations/feishu/feishu_api.py, 向后兼容)
第 4 层: Agent @tool 工具绑定 (tools/feishu_tools.py, 16 个工具)
```

### 5.2 第 1 层 — lark-cli

容器内通过 npm 全局安装，手动安装路径固定为 `/usr/local/bin/lark-cli`。

```go
lark-cli --version   # v1.0.57
lark-cli auth login  # OAuth 认证
lark-cli shortcut list
```

### 5.3 第 2 层 — FeishuToolkit

`integrations/feishu/feishu_toolkit.py` — 覆盖 21 域 209 方法的完整 Python 封装：

```
FeishuToolkit
├── _core          底层 CLI 调用引擎
├── im            即时通讯 (20 命令)
├── docs          文档 (11 命令)
├── drive         云盘 (26 命令)
├── calendar      日历 (7 命令)
├── contact       通讯录 (2 命令)
├── mail          邮箱 (19 命令)
├── sheets        电子表格 (73 命令)
├── base          多维表格 (90 命令)
├── task          任务 (18 命令)
├── minutes       妙记 (8 命令)
├── vc            视频会议 (7 命令)
├── wiki          知识库 (12 命令)
├── okr           OKR (12 命令)
├── apps          应用管理 (25 命令)
├── markdown      Markdown (5 命令)
├── slides        幻灯片 (4 命令)
├── whiteboard    画板 (3 命令)
├── event         事件订阅 (1 命令)
├── note          笔记 (2 命令)
├── approval      审批
└── attendance    考勤
```

**使用示例**：

```python
from novelfactory.integrations.feishu.feishu_toolkit import FeishuToolkit

tk = FeishuToolkit()
tk.im.send_text("你好", chat_id="oc_xxx")
tk.docs.create("标题", "# 内容", folder_token="token")
tk.drive.ensure_folder("文件夹名", "parent_token")
tk.calendar.agenda(date="2026-06-29")
tk.contact.search_user("张三")
tk.task.create("完成任务", assignee="ou_xxx", due="+3d")
```

### 5.4 第 3 层 — FeishuAPI 兼容层

`integrations/feishu/feishu_api.py` — 所有实现委托给 FeishuToolkit，保持旧接口不变：

```python
from novelfactory.integrations.feishu.feishu_api import (
    send_lark_message,        # 发送飞书文本消息
    send_lark_card,           # 发送飞书交互卡片
    create_feishu_doc,        # 创建飞书文档
    ensure_folder,            # 确保云盘文件夹存在
    ensure_project_folders_idempotent,  # 幂等创建项目文件夹
    send_chapter_to_feishu,   # 推送章节到飞书文档
)
```

### 5.5 第 4 层 — Agent @tool 工具绑定

`tools/feishu_tools.py` — 16 个 `@tool` 装饰器，供 LLM Agent 自主使用：

```python
from novelfactory.tools.feishu_tools import get_feishu_tools, get_feishu_tools_basic

# 完整版（16 个工具）
tools = get_feishu_tools()
agent = create_react_agent(llm, tools=tools, prompt=...)

# 轻量版（6 个核心工具）
tools = get_feishu_tools_basic()
```

### 5.6 integrations/feishu/ 文件清单

`integrations/feishu/` 目录下共 7 个文件 + `_tools/` 子包（21 个域工具模块）：

| 文件 | 职责 |
|------|------|
| `feishu_toolkit.py` | 标准工具箱 — 209 方法/21 域完整封装（核心文件） |
| `feishu_api.py` | 向后兼容封装层 — 旧接口委托给 FeishuToolkit |
| `feishu_drive.py` | 云盘操作 — 文件上传/文件夹管理等 |
| `notify.py` | 飞书通知 — 写作进度/审核结果推送 |
| `card_builder.py` | 交互卡片构建 — 审核/确认等交互式卡片 |
| `event_handler.py` | IM 事件处理 — 接收和处理飞书消息事件 |
| `_core.py` | CLI 调用引擎 — LarkResult/LarkListResult/`_LarkCLIEngine` |

---

## 六、工具集

### 6.1 文件清单

| 文件 | 数量 | 职责 |
|------|:----:|------|
| `tools/feishu_tools.py` | 16 个 @tool | 飞书 Agent 工具（消息/文档/日历/通讯录/任务等） |
| `tools/milvus_tools.py` | 若干 | Milvus 向量检索工具（章节语义搜索） |
| `tools/neo4j_tools.py` | 若干 | Neo4j 图数据库工具（角色关系查询） |

### 6.2 使用方式

```python
# 飞书工具
from novelfactory.tools import get_feishu_tools, get_feishu_tools_basic

# Milvus 工具
from novelfactory.tools.milvus_tools import search_similar_chapters

# Neo4j 工具
from novelfactory.tools.neo4j_tools import query_character_relations
```

---

## 七、Pipeline 层

### 7.1 文件清单

| 文件 | 职责 |
|------|------|
| `pipeline/phase2_manager.py` | Phase 2 管理 — 一致性审计、伏笔管理系统（优先级/回收计划/到期提醒）、节奏控制（张弛检测/节奏建议） |
| `pipeline/phase3_manager.py` | Phase 3 管理 — 卷检查（完成检测/过渡上下文构建）、质量检测（趋势衰减分析）、写作上下文构建 |
| `pipeline/scale_manager.py` | 长篇小说扩展 — 分层大纲（卷→章→节）、滑动上下文窗口（摘要压缩/关键事件索引）、角色弧线追踪 |

### 7.2 集成方式

```python
from novelfactory.pipeline.phase2_manager import ForeshadowingManager, ConsistencyAuditor
from novelfactory.pipeline.phase3_manager import VolumeManager, QualityTrendDetector
from novelfactory.pipeline.scale_manager import ScaleManager
```

---

## 八、Config 层

### 8.1 文件清单

| 文件 | 职责 |
|------|------|
| `config/llm.py` | LLM 工厂 — ChatOpenAI 实例化（supervisor/worker/reviewer 不同 temperature；v8.5 DeepSeek key 兜底不再串用 ARK/OPENAI） |
| `config/settings.py` | pydantic-settings 配置中心 + `_ENV_OVERRIDES`（8 个环境变量类型安全覆盖）+ `LARK_PROXY_URL` 字段（v8.5-fix S5）+ URL 密码日志掩码（v8.5-fix S9） |
| `config/database.py` | 数据库连接池 — PostgreSQL/Milvus/Neo4j/Redis 统一配置 |
| `config/constants.py` | 全局常量中心化 — 阈值（VERDICT_PASS/REFINE_THRESHOLD）、题材阈值（GENRE_THRESHOLDS）、重试/超时/并行参数 |
| `config/llm_params.py` | LLM 参数中心 — 5 个 Tier + Agent 级覆盖（Temperature/Timeout） |
| `config/quality_params.py` | 动态调参中心 — verdict.* / unified.* 运行时覆盖 |
| `config/quota.py` | QuotaSettings — 配额单一来源（v8.5-fix M5 消除双源） |
| `config/pricing.py` | 定价 — Token 计费标准 |

### 8.2 环境变量覆盖

```python
# config/settings.py
_ENV_OVERRIDES = {
    "NOVELFACTORY_CHECKPOINT_TYPE":  "CHECKPOINT_TYPE",  # postgres | memory
    "NOVELFACTORY_MAX_RETRIES":      "MAX_RETRIES",
    # ... 共 8 个键，自动类型安全转换（v8.5 移除 QUOTA_THRESHOLD 死映射）
}

def _coerce_env(value: str, reference: object) -> object:
    """类型安全转换: "true" → True, "5" → 5, "3.14" → 3.14"""
```

**LARK_PROXY_URL（v8.5-fix S5）**：`settings.LARK_PROXY_URL` 字段已声明，`lark_proxy_url` 属性 = `LARK_PROXY_URL or f"http://tools_proxy:5004"`（env 优先，容器内默认可达）。启动日志 `_log_effective_config` 对 `DATABASE_URL`/`REDIS_URL` 等 URL 字段掩码 userinfo 密码。

---

## 九、飞书回调

### 9.1 飞书 URL 验证

```http
POST /feishu/callback
Content-Type: application/json

{"type": "url_verification", "challenge": "test123"}
```

**响应**：

```json
{"challenge": "test123"}
```

### 9.2 交互卡片回调处理流程

`routes/feishu_callback.py` 处理飞书卡片按钮回调，实现人工审核中断恢复：

```
1. wait_for_review_node 调用 interrupt() 暂停执行
2. 中断数据通过 SSE 或飞书卡片展示给用户
3. 用户点击飞书卡片按钮（approve/reject/modify）
4. 飞书发送回调到 /feishu/callback
5. endpoint 解析回调数据，调用 Command(resume={...}) 恢复线程
```

**核心恢复函数**：

```python
async def _resume_thread(thread_id: str, resume_data: dict) -> bool:
    """恢复被 interrupt() 暂停的线程。"""
    graph = await get_app()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 5000,
    }
    await graph.aupdate_state(config, values=None, as_node="wait_for_review")
    await graph.ainvoke(Command(resume=resume_data), config)
```

### 9.3 手动恢复线程端点

```http
GET /feishu/resume/{thread_id}?action=approve&comment=看起来不错
```

### 9.4 安全配置

| 环境变量 | 说明 |
|---------|------|
| `FEISHU_VERIFICATION_TOKEN` | 飞书应用验证 Token |
| `FEISHU_ENCRYPT_KEY` | 飞书加密 Key（回调 payload 解密） |

---

**规则版本：** v1.2.0
**生效方式：** 智能生效
**最后更新：** 2026-07-04
