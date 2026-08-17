---
alwaysApply: false
description: "LangGraph规范，匹配graph/**、state/**、langgraph.json。调图流程、改路由、加node、状态字段、并行分发、检查点恢复、子图编译、recursion_limit、Send分发、Command/interrupt/NodeSpec时触发。节点签名、TypedDict+AnnotatedReducer、Checkpointer、NodeSpec注册、幂等检查链路由、路由、子图编译、统一评审。"
---
# LangGraph 图开发规范

**版本：** v3.3.0
**生效方式：** 智能生效
**优先级：** ⭐⭐⭐⭐⭐
**匹配模式：** `graph/**`, `state/**`, `langgraph.json`

---

## 一、核心概念

| 概念 | 说明 | 关键点 |
|------|------|--------|
| **StateGraph** | 有状态图构建器，节点通过共享状态通信 | 根图和子图都使用 |
| **Node** | 执行单元，接收状态返回部分更新 | 函数签名有多种形式 |
| **Reducer** | 多节点写入同一键的合并策略 | `operator.add`, `add_messages`, `_last_value` |
| **Checkpointer** | 每步自动快照状态 | 仅根图持有 |
| **Command** | 导航 + 状态更新一体化 | `goto`, `update`, `Command.PARENT` |
| **Send** | 动态并行分发 | Map-Reduce 模式 |
| **interrupt** | 人机交互挂起 | `interrupt_before` / `interrupt_after` |

---

## 二、StateGraph 构建

### 2.1 基本模式

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

class MyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    result: str

builder = StateGraph(State)
builder.add_node("node_a", func_a)
builder.add_node("node_b", func_b)
builder.add_edge(START, "node_a")
builder.add_conditional_edges("node_a", router_fn, {"yes": "node_b", "no": END})
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_a"],
)
```

### 2.2 节点函数签名

```python
# 基础
def node(state: State) -> dict:
    return {"key": value}

# 带配置
def node(state: State, config: RunnableConfig) -> dict:
    thread_id = config["configurable"]["thread_id"]

# 带 Store（长期记忆）
async def node(state: State, *, store: BaseStore) -> dict:
    await store.put(["namespace"], "key", {"data": ...})

# 带 Writer（流式输出）
def node(state: State, *, writer: StreamWriter) -> dict:
    writer({"custom_key": "streaming_data"})

# 返回 Command（导航 + 更新）
def node(state: State) -> Command[Literal["next_a", "next_b"]]:
    return Command(goto="next_a", update={"x": 1})

# 子图退出返父图
def subgraph_exit(state: SubState) -> Command:
    return Command(update={"result": "done"}, goto="parent_node", graph=Command.PARENT)

# interrupt() 人机交互
def review_node(state: MyState) -> dict:
    decision = interrupt({"question": "approve this?"})
    return {"approved": decision.get("action") == "approve"}
```

### 2.3 并行与分发

```python
# 等待多个节点完成
builder.add_edge(["node_a", "node_b"], "node_c")

# Send 动态并行分发（Map-Reduce）
def router(state):
    return [Send("worker", {"item": item}) for item in state["items"]]
builder.add_conditional_edges(START, router)
```

### 2.4 子图集成

```python
# 子图编译（无 checkpointer）
writing_crew = build_writing_crew_graph().compile()
parent_builder.add_node("writing_crew", writing_crew)

# 子图退出返父图
def exit_node(state: CrewState) -> Command:
    return Command(goto=Command.PARENT, update={"crew_result": state})
```

---

## 三、状态定义规范

### 3.1 推荐模式：TypedDict + Annotated Reducer

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

class NovelFactoryState(TypedDict):
    # 身份与上下文
    thread_id: str
    project_context: dict

    # 累积字段
    messages: Annotated[list[AnyMessage], add_messages]
    completed_chapters: Annotated[list, operator.add]

    # 控制字段
    current_phase: str
    current_chapter: int

    # 托管值
    is_last_step: Annotated[bool, IsLastStep]
```

### 3.2 关键 Reducer

```python
# _last_value: 多源写防冲突
def _last_value(existing, update):
    return update if update is not None else existing

# _add_usage: Token 用量去重合并
def _add_usage(existing: dict, incoming: dict) -> dict:
    # 按 (chapter_number, phase) 去重，最新覆盖

# compress_completed_chapters: 保留最近 50 章完整 + 早期摘要
```

### 3.3 InvalidUpdateError 预防

**问题**：同一 tick 内多个节点写同一无 Annotated reducer 的字段会报错。

**解决**：为多节点共享字段添加 `Annotated[T, _last_value]`。

---

## 四、Checkpointer（检查点）

### 4.1 配置规范

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool(
    conninfo="postgresql://user:pass@host:5432/db?sslmode=disable",
    max_size=10, kwargs={"autocommit": True},
)
await pool.open()
checkpointer = AsyncPostgresSaver(pool)
await checkpointer.setup()

# Store（长期记忆 + 语义搜索）
store = AsyncPostgresStore(
    pool,
    index={"embed": embed_fn, "dims": 1024, "fields": ["$"]},
)
await store.setup()

# 编译
app = graph.compile(checkpointer=checkpointer, store=store)
```

### 4.2 恢复场景

| 场景 | state.next | 操作 |
|------|-----------|------|
| **正常继续** | 非空且无 interrupt | `POST /threads/{id}/runs {}` |
| **中断恢复** | 非空且有 interrupt | `POST /threads/{id}/runs {"resume":data}` |
| **新运行** | 空 | 带完整 seed 重新提交 |

### 4.3 关键规则

- **子图不传 checkpointer**，由根图统一管理持久化
- 所有编译后的图设置 `recursion_limit = 200`（子图）或 `5000`（根图）
- 中断点通过 `interrupt_before` / `interrupt_after` 定义
- **v5.6**: 完成态自动清理 — `maybe_cleanup_checkpoints()` 在 `save_memory` 节点触发

---

## 五、NodeSpec 动态注册体系

### 5.1 PhaseCheckNodeSpec

```python
@dataclass(frozen=True)
class PhaseCheckNodeSpec:
    key: str                        # 节点 ID
    node_fn: Callable               # 节点函数
    required_genres: tuple[str, ...] = ()  # 必须执行此检查的题材
    skip_genres: tuple[str, ...] = ()     # 跳过此检查的题材
    description: str = ""           # 描述
```

### 5.2 预注册节点

| key | 说明 | 必选题材 | 跳过题材 |
|-----|------|---------|---------|
| `volume_check` | 卷结构检查 | — | `短篇` |
| `quality_check` | 质量趋势检测 | — | — |
| `foreshadowing_check` | 伏笔回收检查 | `悬疑灵异`, `仙侠`, `玄幻` | `短篇` |

### 5.3 题材名统一解析

所有题材名经 `constants.resolve_genre()` 统一解析为标准中文名称：

- 输入 `"末世"`、`"末世小说"` → 解析为 `"末世"`
- 输入 `"快穿"`、`"快穿文"` → 解析为 `"系统流"`
- 解析失败 → 返回 `"default"`

NodeSpec 中 `only_for_genres` 和 `skip_genres` 必须使用解析后的标准名称。

### 5.4 路由集成

`route_from_supervisor` 从 `PHASE_CHECK_SPECS` 调用 `build_check_chain(genre)` 生成按题材过滤的检查链。

**v8.5-fix（S8）**：检查链节点绑定 `make_check_chain_router(node_key)` 幂等路由闭包，按"当前完成节点在链中的位置"精确路由到下一节点：

```python
# graph/routing.py — v8.5-fix：替代基于跨章残留状态字段的 route_phase_check_chain
def make_check_chain_router(node_key: str):
    def _route(state: NovelFactoryState) -> str:
        check_chain = build_check_chain(PHASE_CHECK_SPECS, _resolve_genre(state))
        if not check_chain:
            return "refresh_quota"
        pos = _find_in_chain(check_chain, node_key)
        if pos >= 0 and pos + 1 < len(check_chain):
            return check_chain[pos + 1]
        return "refresh_quota"
    return _route

# new_builder.py 中为每个检查节点绑定独立路由
builder.add_conditional_edges("volume_check", make_check_chain_router("volume_check"), CHECK_CHAIN_MAP)
```

> ⚠️ 旧 `route_phase_check_chain` 基于 `volume_status/quality_trend/foreshadowing_status` 等**跨章持久化字段**推断进度，第 2 章起残留状态导致后续检查被跳过（已修复）。新图统一使用 `make_check_chain_router`，旧函数仅保留兼容。

---

## 六、多智能体并行开发模式

### 6.1 并行模式总览

NovelFactory 支持 6 种多智能体并行模式（~~划掉的是 v6.3 已移除~~）：

| 模式 | 实现机制 | 并行类型 | 代码位置 |
|------|---------|---------|---------|
| **子图独立Agent** | `add_node(compiled_subgraph)` | 时序协作 | `graph/new_builder.py` |
| **ThreadPoolExecutor 真并行** | `ThreadPoolExecutor(max_workers=N)` | 线程级并行 | `graph/crews/media_crew.py` |
| **统一评审子Agent** | UnifiedReviewEngine 单次五视角调用 + 分歧仲裁 | 单次调用 | `evaluation/unified/` + `evaluation/verdict/engine.py` |
| **检查链** | NodeSpec 动态注册 + 幂等闭包路由 | 链式并行 | `graph/node_specs.py` + `graph/routing.py` |
| **扇出边** | `add_edge([A,B], C)` | 静态扇出 | `graph/new_builder.py` |
| ~~Send Map-Reduce~~ | ~~LangGraph Send API~~ | ~~动态并行~~ | ~~`graph/parallel/volume_dispatch.py`~~ |

### 6.2 模式详解

#### 6.2.1 子图独立Agent

每个 Crew 编译为独立子图，通过 `add_node()` 注入根图：

```python
graph.add_node("writing_crew", with_middleware(build_writing_crew(), _mw_chain))
graph.add_node("media_crew", with_middleware(build_media_crew(), _mw_chain))
graph.add_node("sync_crew", with_middleware(build_sync_crew(), _mw_chain))
```

**规范**：
- 子图编译时**不传 checkpointer**，由根图统一管理持久化
- 子图内部节点返回 `Command.PARENT` 时自动传播到父图
- 每个子图设置 `recursion_limit = 200`
- 状态继承 `BaseCrewState`，通过 `crew_result` 字段透传数据

#### 6.2.2 ThreadPoolExecutor 真并行

用于 media_crew 中插图与配音的并发生成：

```python
def _parallel_media_node(state: MediaCrewLocalState) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="media_crew") as pool:
        future_ill = pool.submit(illustrator_agent, illustrator_input)
        future_tts = pool.submit(tts_agent, tts_input)
        illustrator_result = future_ill.result()
        tts_result = future_tts.result()
```

**规则**：
- Agent 必须是同步 Runnable —— ThreadPoolExecutor 不支持 async
- 自动重试机制（`_media_tool_router` 条件路由，最多 3 次）
- 单 Agent 失败不影响另一个，各自 try/except 保护
- 结果通过 `crew_result` 合并返回

#### 6.2.3 统一评审子Agent（v8.2）

`UnifiedReviewEngine.evaluate()` 单次 LLM 调用完成五视角评审（替代旧 4 路并行评审）：

```python
# evaluation/unified/engine.py - UnifiedReviewEngine.evaluate()
# 一次调用：五视角（老书虫/番茄编辑/读者/评论员）+ 四维分项
ur = await unified_engine.evaluate(chapter_text=..., genre=..., ...)
# → parse_review_output（标签化解析 + _clamp 钳制）
# → apply_consistency_check（自洽校验硬约束）
# → arbitrate（仅 severe 分歧触发 1 次仲裁）
```

**容错设计**：
- 调用走 `async_llm_call_with_retry`（超时/重试/熔断/配额/缓存）
- 评审失败 `ur.failed` → 降级 PASS（v8.5-fix M3）
- 分数越界 → parser/仲裁/engine 三层钳制（v8.5-fix S2）

#### 6.2.4 VerdictEngine 统一评审融合（替代 v5.5 QualityPanel 与 v7.0 多源融合）

评审已从 LangGraph 辩论子图重构为 `evaluation/` 模块中的统一 LLM 评审管线。核心引擎 `VerdictEngine` 在 `evaluation/verdict/engine.py`，由 `evaluation/coordinator.py` 的 `verdict_engine_node` 封装为图节点：

统一评审管线：
| 阶段 | 模块 |
|------|------|
| 五视角评审（单次调用） | `evaluation/unified/engine.py` |
| 标签化解析 + 分数钳制 | `evaluation/unified/parser.py` |
| 自洽校验硬约束 | `evaluation/unified/parser.py`（apply_consistency_check） |
| 分歧仲裁 | `evaluation/unified/arbitration.py` |
| 融合 + 迭代加分 + 三级决议 | `evaluation/verdict/engine.py` |
| 三分支路由 | `evaluation/verdict/router.py` |

评分流水线：
```
verdict_engine (VerdictEngine.evaluate)
    ↓
  _decide_unified_level (三级决议: PASS/REFINE/REWRITE)
    ↓
  verdict_router → chapter_planner / chapter_refiner / __exit_for_chapter__
```

相关代码路径：
- `evaluation/unified/` — 统一 LLM 评审（engine/parser/prompts/arbitration/schemas）
- `evaluation/verdict/engine.py` — VerdictEngine 融合引擎
- `evaluation/verdict/router.py` — verdict_router 三分支路由
- `evaluation/coordinator.py` — verdict_engine_node 图节点封装 + best_version 保存
- `evaluation/schemas.py` — VerdictResult / FeedbackBundle / AttemptInfo

> 评分细节与阈值详见 [04-quality-scoring.md](file:///Users/jason/Downloads/langgraph/.trae/rules/04-quality-scoring.md)

#### 6.2.5 检查链（NodeSpec 动态注册）

```python
@dataclass(frozen=True)
class PhaseCheckNodeSpec:
    key: str                        # 节点 ID
    node_fn: Callable               # 节点函数
    required_genres: tuple[str, ...] = ()  # 必须执行此检查的题材
    skip_genres: tuple[str, ...] = ()     # 跳过此检查的题材

# 构建按题材过滤的检查链
def build_check_chain(specs: tuple[PhaseCheckNodeSpec, ...], genre: str) -> list[str]:
    return [s.key for s in specs if _should_include(s, genre)]
```

### 6.3 并行状态管理

#### 6.3.1 Reducer 防冲突

多节点并行写入同一字段时必须使用 Annotated Reducer：

```python
# 多源写防冲突
def _last_value(existing, update):
    return update if update is not None else existing

# 累积合并
completed_chapters: Annotated[list, operator.add]
messages: Annotated[list, add_messages]
total_usage: Annotated[dict, _add_usage]
```

#### 6.3.2 Send 并行上下文隔离

```python
# v6.1 P3-10: 发送完整上下文以支持并行上下文隔离
Send("writing_crew", {
    "current_chapter": ch_number,
    "thread_id": thread_id,      # 上下文隔离
    "volume_context": volume_context,
})
```

### 6.4 并行最佳实践

| 场景 | 推荐模式 | 不推荐 |
|------|---------|-------|
| 独立任务并发（插图+配音） | ThreadPoolExecutor | Send |
| 章节质量评审 | 统一 LLM 评审（五视角一次调用） | 多轮辩论 |
| 阶段切换 | Main Supervisor 编排 | 手动 Command |
| 大量独立子任务 | 线性逐一（v6.3+） | Send Map-Reduce |
| 按题材过滤的检查 | NodeSpec 动态链 + 幂等闭包路由 | if/else 硬编码 |

---

## 七、路由逻辑

`route_from_supervisor` 路由表（定义在 `graph/routing.py`）：

| 条件 | 目标 | 说明 |
|------|------|------|
| `setup_aborted=True`（任意 phase） | `END` | v8.5-fix S4：全局拦截，防空设定污染 |
| phase=setup, 未完成 | `setup_crew` | |
| phase=setup, 完成 | `load_memory` | |
| phase=writing, ch>1 | `volume_check` → `quality_check` → `foreshadowing_check` | NodeSpec 动态链 + 幂等闭包路由 |
| phase=writing, ch=1 | `refresh_quota` | |
| phase=writing, 审核待定 | `wait_for_review` | interrupt |
| phase=volume_parallel (v6.3移除) | `volume_dispatch` | Send 并行分发 |
| phase=volume_review (v6.3移除) | `volume_reviewer` | 卷级评审 |
| phase=media | `media_crew` | |
| phase=sync | `sync_crew` | |
| done | `save_memory` → END | |

> 关于评分路由（_score_router）和 scoring 判定逻辑，详见 [04-quality-scoring.md](file:///Users/jason/Downloads/langgraph/.trae/rules/04-quality-scoring.md)

---

## 八、子图编译规则

| 子图 | Checkpointer | 说明 |
|------|:-----------:|------|
| writing_crew | 无 | 父图管理持久化，内含 verdict_engine 评分节点 |
| media_crew | 无 | 含 tool self-loop 重试 |
| sync_crew | 无 | 含 tool self-loop 重试 |
| context_builder | 无 | 纯函数子图（load_state→build_context→aggregate） |
| state_extractor | 无 | 纯函数子图（6 个 LLM 节点并行→aggregate） |
| database_writer | 无 | 纯函数子图（5 路并行写入→aggregate） |
| narrative_codec_crew | 无 | Codec 子图（7 节点串联流水线） |
| **根图** | AsyncPostgresSaver | 根图持有持久化 + 终态清理 |

---

## 九、VerdictEngine 评分路由（替代 v5.5 QualityPanel 辩论子图）

QualityPanel 辩论子图已废弃，评分路由由 `evaluation/verdict/router.py` 的 `verdict_router` 函数处理（纯路由，v8.5 已移除 state 修改逻辑）：

```python
# evaluation/verdict/router.py
def verdict_router(state) -> Literal["chapter_planner", "chapter_refiner", "__exit_for_chapter__"]:
    """三分支路由: REWRITE→chapter_planner, REFINE→chapter_refiner, PASS→__exit_for_chapter__"""
```

> v8.5-fix（M1）：`best_version_*` 保存逻辑已从条件边迁移到 `verdict_engine_node` 节点返回值（条件边内改 state 不写回 checkpoint，原恢复逻辑恒不触发）。

评分决策流程详见 [04-quality-scoring.md](file:///Users/jason/Downloads/langgraph/.trae/rules/04-quality-scoring.md)

## 十、langgraph.json 配置

```json
{
    "dependencies": ["."],
    "graphs": {
        "agent": "src/novelfactory/graph/new_builder.py:create_dev_graph"
    },
    "env": {
        "NOVELFACTORY_LOG_PATH": ""
    }
}
```

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `graphs` | 是 | 图定义，`"路径:变量名"` |
| `dependencies` | 否 | `"."` 表示本地包 |
| `env` | 否 | `.env` 文件路径 |
| `store` | 否 | 长期记忆配置 |

---

## 十一、常见问题排查（图开发相关）

| 问题 | 原因 | 解决 |
|------|------|------|
| InvalidUpdateError | 多节点写同一无 Annotated 字段 | 加 `_last_value` reducer |
| recursion_limit 超限 | 递归上限太低 | 根图 5000 / 子图 200 |
| 子图不持久化 | 子图带了 checkpointer | 子图编译不传 checkpointer |
| 检查点恢复后状态丢失 | thread_id 不匹配 | 始终传正确的 `configurable.thread_id` |
| 检查链第 2 章起被跳过 | 跨章残留状态字段误判进度 | 用 `make_check_chain_router` 幂等闭包路由 |
| 子图字段被丢弃 | 字段未在子图 schema 声明 | WritingCrewLocalState 声明 critic/guidance/best_version 等 |
| VerdictEngine 评分异常 | 统一评审失败/熔断 | 检查 `llm_analysis_failed` 与熔断器状态 |

> 部署/运维相关问题参见 [06-deployment-operations.md](file:///Users/jason/Downloads/langgraph/.trae/rules/06-deployment-operations.md)

---

**规则版本：** v3.3.0
**生效方式：** 智能生效
**最后更新：** 2026-08-17
