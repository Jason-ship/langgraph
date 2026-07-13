# Sync Agent — LangGraph 模式参考

> 本文档是 LANGGRAPH_PATTERNS_REFERENCE.md 的子文档，专注 Sync 智能体

---

## 一、当前架构概览

### 涉及文件
- [sync_agents.py](src/novelfactory/agents/sync_agents.py) - FeishuSyncAgent, StateUpdate

### 智能体列表

| 智能体 | 类型 | 职责 |
|--------|------|------|
| FeishuSyncAgent | ReAct Agent（工具使用） | 将章节内容、插画、音频同步到飞书，发送进度通知 |
| StateUpdate | Utility 节点 | 持久化 Sync Crew 结果到全局 checkpointer（纯工具函数） |

### 当前图结构

```
Sync Crew:
  Agent Node (FeishuSyncAgent) → StateUpdate Node
```

### 当前模式

- **Agent（简单工具使用 Agent）**：FeishuSyncAgent 通过 `create_react_agent` 构建，绑定飞书工具集，LLM 可自主决定上传策略。
- **中断状态**：`interrupt_before=[]`，中断机制当前全部禁用。
- **动态 Prompt**：`_build_sync_dynamic_prompt` 根据项目状态调整同步指令（v6.0）。
- **StateUpdate**：非真正持久化节点，仅作为 Sync Crew 流程中的显式检查点标记。

### 关键实现细节

- `_get_context(state)`：从 `crew_result` 或根 state 中提取上下文字段，包含飞书目录 token、质量评分、token 用量等。
- `_resolve_volume_number(chapter_number, volume_structure)`：根据章节号解析所属卷号，支持 dict 和 list 两种 volume_structure 格式。
- **URL 真实性检测**：仅实际上传成功（非 LLM 幻觉生成的 URL）时才发送完成通知。

```python
# 当前中断配置（已禁用）
interrupt_before=[]
```

---

## 二、可借鉴的 LangGraph 模式

### 2.1 Customer Support — Part 2/3：中断确认 + Conditional Interrupt

**来源**：LANGGRAPH_PATTERNS_REFERENCE.md 2.1 Customer Support

**现状问题**：当前 FeishuSyncAgent 直接执行飞书上传，无人工确认步骤。在多章批量同步场景下，用户可能希望在上传前确认内容正确性。

**可借鉴模式**：

#### Part 2: Add Confirmation — `interrupt` 实现上传前人工确认

在 Sync Agent 执行飞书上传前插入中断，等待用户确认后再执行。

```python
# 在编译 Sync Crew 时设置中断点
sync_graph = sync_crew.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["feishu_sync_agent"],  # ← 上传前中断
)
```

用户通过 `Command(resume=...)` 恢复执行：

```python
# 用户确认后续传
thread_config = {"configurable": {"thread_id": "project_xxx"}}
sync_graph.invoke(
    Command(resume={"confirmed": True, "note": "确认上传"}),
    config=thread_config,
)
```

#### Part 3: Conditional Interrupt — 仅首次同步需确认

避免每次同步都中断，仅在首次同步（或重要章节）时触发确认。

```python
def should_interrupt(state: dict) -> bool:
    """判断当前同步是否需要中断确认。"""
    # 条件1：首次同步
    is_first = not state.get("folder_tokens", {}).get("project")
    # 条件2：关键章节（如转折章节）
    current_ch = state.get("current_chapter_number", 1)
    is_key_chapter = current_ch in (1, 10, 50, 100)  # 示例里程碑
    # 条件3：质量评分过低时提醒
    quality = state.get("crew_result", {}).get("quality_score", 100)
    is_low_quality = quality < 60

    return is_first or is_key_chapter or is_low_quality
```

### 2.2 Customer Support — Part 4：多个专用助手

**来源**：LANGGRAPH_PATTERNS_REFERENCE.md 2.1 Customer Support（Specialized Workflows）

**现状问题**：当前 FeishuSyncAgent 一个 Agent 承担了所有职责（上传、通知、备份），逻辑耦合度高。

**可借鉴模式**：将同步流程拆分为多个专用助手子图，通过 Supervisor 编排。

```
Sync Supervisor
  ├── SyncHandler     — 负责飞书文档上传
  ├── Notifier        — 负责发送进度通知
  └── BackupHandler   — 负责本地/云端备份
```

优势：
- 每个助手只关注单一职责，Prompt 更清晰
- 可独立测试每个子流程
- 支持条件路由：上传失败时不触发通知，上传成功才通知

### 2.3 Common Workflows — Routing：同步结果路由

**来源**：LANGGRAPH_PATTERNS_REFERENCE.md 1.2 Common Workflows - Routing

**现状问题**：当前同步结果仅有成功/失败二元区分，缺少对部分成功（如插画上传失败但文档上传成功）的处理。

**可借鉴模式**：根据同步结果路由到不同后处理节点。

```
                     ┌─── 成功 → send_completion_notification
同步结果检查 ──┬─── 部分成功 → send_warning_notification + log_partial_failure
                     └─── 失败 → retry_or_rollback
```

---

## 三、补充代码模板

### 3.1 Conditional Interrupt — `should_interrupt` 判断函数

在 Sync Crew 图中添加条件判断节点，控制是否中断。

```python
from typing import Literal


def should_interrupt_node(state: dict) -> dict:
    """评估当前同步是否需要中断确认。"""
    # 条件1：首次同步（飞书目录尚未创建）
    folder_tokens = state.get("folder_tokens", {})
    if not folder_tokens or not folder_tokens.get("project"):
        return {"needs_confirmation": True, "interrupt_reason": "首次同步，需确认上传"}

    # 条件2：关键章节（里程碑章节）
    current_ch = state.get("current_chapter_number", 1)
    milestone_chapters = {1, 10, 50, 100}
    if current_ch in milestone_chapters:
        return {"needs_confirmation": True, "interrupt_reason": f"第{current_ch}章为里程碑章节，需确认"}

    # 条件3：质量评分过低
    quality = state.get("crew_result", {}).get("quality_score", 100)
    if quality < 60:
        return {
            "needs_confirmation": True,
            "interrupt_reason": f"质量评分({quality})低于60，建议人工确认",
        }

    return {"needs_confirmation": False, "interrupt_reason": ""}
```

### 3.2 同步结果路由 — 3 路条件边实现

根据同步状态（成功 / 部分成功 / 失败）路由到不同后处理。

```python
from typing import Literal


def route_sync_result(state: dict) -> Literal["completion", "warning", "retry"]:
    """根据同步结果路由到不同后处理路径。

    同步状态判断逻辑：
    - 成功（completion）：飞书文档上传成功，所有附属资源上传成功
    - 部分成功（warning）：文档上传成功，但插画或音频上传失败
    - 失败（retry）：文档上传失败
    """
    crew_result = state.get("crew_result", {}) or {}
    feishu_url = crew_result.get("feishu_doc_url", "")
    sync_details = crew_result.get("sync_details", {})

    # 判断文档上传状态
    doc_ok = bool(feishu_url) and not feishu_url.startswith("https://feishu.cn/doc")

    if not doc_ok:
        # 文档上传失败 → 重试或回退
        return "retry"

    # 检查附属资源上传状态
    illustration_ok = sync_details.get("illustration_uploaded", True)
    audio_ok = sync_details.get("audio_uploaded", True)

    if illustration_ok and audio_ok:
        # 所有资源上传成功
        return "completion"
    else:
        # 文档成功但部分附件失败
        return "warning"


# 在图中注册条件边
sync_builder.add_conditional_edges(
    "sync_checker",
    route_sync_result,
    {
        "completion": "send_completion_notification",
        "warning": "send_warning_notification",
        "retry": "retry_sync_node",
    },
)
```

### 3.3 `interrupt_before` + `Command(resume=...)` 恢复执行模板

完整的中断 → 确认 → 恢复执行流程。

```python
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

# ── 1. 编译 Sync Crew（启用中断） ─────────────────────────────
sync_graph = sync_crew.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["feishu_sync_agent"],
)


# ── 2. 首次调用（触发中断） ──────────────────────────────────
thread_config = {"configurable": {"thread_id": "project_novel_001"}}

# 同步流程执行到 feishu_sync_agent 前中断
for event in sync_graph.stream(
    {"crew_result": {"refined_chapter": "..."}},
    config=thread_config,
    stream_mode="updates",
):
    if "__interrupt__" in event:
        interrupt_value = event["__interrupt__"][0].value
        print(f"[中断] 原因: {interrupt_value}")
        # 这里等待用户确认...（API 层等待前端回传确认信号）
        break


# ── 3. 用户确认后恢复执行（在外部 API 调用中） ────────────
confirmed = True  # 用户前端传回的确认值
sync_graph.invoke(
    Command(resume={"confirmed": confirmed}),
    config=thread_config,
)


# ── 4. 二阶段恢复：支持附带额外指令 ──────────────────────
# 用户可以在确认时附带备注或修改请求
sync_graph.invoke(
    Command(
        resume={
            "confirmed": True,
            "note": "请在文档标题中添加「终稿」标记",
            "skip_illustration": True,  # 跳过插画上传
        }
    ),
    config=thread_config,
)
```

---

## 四、与综合文档的关联

本文档是 [LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md) 的子文档，专注 Sync 智能体的 LangGraph 模式。

| 综合文档章节 | 内容 | 本文档映射 |
|-------------|------|-----------|
| 4. Sync 智能体 | 当前架构、可借鉴模式、关键 API | 第一节（当前架构）+ 第二节（模式详解） |
| 2.1 Customer Support Part 2 | interrupt 确认模式 | 2.1 + 3.3 代码模板 |
| 2.1 Customer Support Part 3 | Conditional Interrupt | 2.1 + 3.1 代码模板 |
| 2.1 Customer Support Part 4 | 专用助手工作流 | 2.2 |
| 1.2 Common Workflows - Routing | 结果路由 | 2.3 + 3.2 代码模板 |

### 下一步改进建议

1. **启用中断机制**：从 `interrupt_before=[]` 改为条件中断，首次同步和里程碑章节时触发人工确认
2. **拆分专用助手**：将 FeishuSyncAgent 拆分为 SyncHandler / Notifier / BackupHandler 三个子图
3. **结果路由**：增加同步结果检查节点（route_sync_result），支持成功/部分成功/失败三路分支
4. **子图复用**：Sync Crew 可作为子图嵌入到顶层 Writing 图或 Chapter Review 图中，实现端到端流程
