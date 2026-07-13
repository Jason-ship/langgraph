# Media Agent — LangGraph 模式参考

> 本文档是 [LANGGRAPH_PATTERNS_REFERENCE.md](./LANGGRAPH_PATTERNS_REFERENCE.md) 的子文档，专注 Media 智能体

---

## 1. 当前架构概览

### 涉及文件
- `media_agents.py` — IllustratorAgent, TTSGeneratorAgent

### 智能体列表

| 智能体 | 职责 | 输出 TypedDict | 辅助函数 |
|--------|------|---------------|----------|
| **IllustratorAgent** | 为章节生成高质量插画 | `IllustratorOutput` → `{illustration_url, illustration_prompt}` | `_generate_image_via_matrix()` |
| **TTSGeneratorAgent** | 为章节生成有声内容 | `TTSGeneratorOutput` → `{audio_url}` | `_generate_tts_via_matrix()` |

### 当前图结构：Media Crew

```
Supervisor
    │
    ├──► IllustratorAgent（ReAct Agent + subprocess image generation）
    │
    └──► TTSGeneratorAgent（ReAct Agent + subprocess TTS generation）
```

- 两个 Worker **无依赖关系**，由 Media Crew Supervisor 并行触发
- 每个 Agent 均使用 `create_react_agent` 构建，`tools=[]`
- 底层使用 `_generate_image_via_matrix` / `_generate_tts_via_matrix` 调用 Matrix MCP 服务

### 当前模式
- **Parallelization（并行化）**：Illustrator + TTS Generator 并行执行
- **Agent（简单 Agent）**：目前无 LangGraph 工具，直接调用 LLM + subprocess

### 当前异常处理方式
- 主路径失败时，函数内部已实现 `try/except` + 空字符串回退逻辑
- 但缺少**显式的图级别路由**：失败路径和成功路径在图中不可见，全在函数内部消化

---

## 2. 可借鉴的 LangGraph 模式

### 2.1 Common Workflows — Parallelization（结合聚合器）

**参考来源**：综合文档 1.2 Common Workflows — Parallelization

当前实现中，Illustrator 和 TTS Generator 的输出各自独立写入 `crew_result`，但缺少一个**聚合节点**来统一决策。建议增加 `media_aggregator` 节点：

```
Supervisor
    │
    ├──► illustrator_node ──┐
    │                        ├──► media_aggregator
    └──► tts_node ──────────┘         │
                                       │ 检查：插图成功？TTS成功？
                                       │ 决策：继续 vs 降级 vs 重试
                                       ▼
                                 最终输出
```

**聚合器的价值**：
- 统一检查两个并行分支的结果状态
- 决定是否需要重试某一分支（而不影响已完成的分支）
- 即使一个分支完全失败，聚合器仍可产出部分有效的输出

### 2.2 Corrective RAG (CRAG) 风格的回退机制

**参考来源**：综合文档 3.3 Corrective RAG

CRAG 的核心思想是：**主路径失败时，沿回退链降级，而非直接返回空结果**。对 Media Agent 意味着三级回退链：

```
                    ┌── 主路径成功 ──► 聚合
                    │
   illustration_node ── 主路径失败 ──► 回退路径_1 ──► 聚合
                    │
                    └── 回退_1也失败 ──► 二次回退 ──► 聚合（标注降级）
```

**回退层级设计**：

| 层级 | 描述 | 示例（插图） | 示例（TTS） |
|------|------|-------------|------------|
| **主路径** | Matrix MCP 服务 | `matrix_generate_image` | `matrix_batch_text_to_audio` |
| **回退路径_1** | 备选 API 服务 | DALL-E / Stable Diffusion | 备用 TTS 引擎 |
| **二次回退** | 纯文本描述/占位 | 记录 `illustration_prompt` 但无图 | 空音频 + 标注 |

### 2.3 Customer Support — 工具调用错误处理

**参考来源**：综合文档 2.1 Customer Support — Error Handling

Customer Support 模式中的重试和降级策略可应用于 `_generate_*_via_matrix` 函数：

| 模式 | 当前做法 | 可改进方向 |
|------|---------|-----------|
| 重试 | 无显式重试 | 增加指数退避重试（3 次） |
| 超时 | 硬编码 `timeout=60/120` | 可配置超时 + 超时降级 |
| 降级 | 直接返回空字符串 | 图级别降级路由 |
| 监控 | logger warning | 增加结构化错误计数/metric |

---

## 3. 补充代码模板

### 3.1 并行 + 聚合器节点的 StateGraph 构建

```python
"""media_graph.py — 并行 + 聚合器版本的结构化示例。"""

from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.runnables import RunnableLambda


class MediaState(TypedDict):
    # 输入
    refined_chapter: str
    chapter_draft: str
    world_setting: str
    character_setting: str
    current_chapter_number: int
    project_name: str

    # 输出 — 由各自 Worker 填充
    illustration_url: str
    illustration_prompt: str
    audio_url: str

    # 元数据 — 供聚合器决策
    illustration_status: str        # "success" | "fallback" | "failed"
    tts_status: str                 # "success" | "fallback" | "failed"


# ── Worker 节点 ────────────────────────────────────────────────────────────

def illustrator_node(state: MediaState) -> dict:
    """调用现有的 create_illustrator_agent 逻辑。"""
    from novelfactory.agents.media_agents import (
        _generate_image_via_matrix,
        _get_context,
    )
    ctx = _get_context(state)
    chapter_text = ctx.get("refined_chapter", "") or ctx.get("chapter_draft", "")

    # ... 复用现有 LLM 调用逻辑 ...

    # 调用 Matrix MCP
    url = _generate_image_via_matrix(
        prompt, ctx.get("project_name", ""), ctx.get("current_chapter_number", 1)
    )
    if url:
        return {
            "illustration_url": url,
            "illustration_prompt": prompt,
            "illustration_status": "success",
        }
    return {
        "illustration_url": "",
        "illustration_prompt": prompt,
        "illustration_status": "failed",
    }


def tts_node(state: MediaState) -> dict:
    """调用现有的 create_tts_generator_agent 逻辑。"""
    from novelfactory.agents.media_agents import (
        _generate_tts_via_matrix,
        _get_context,
    )
    ctx = _get_context(state)
    chapter_text = ctx.get("refined_chapter", "") or ctx.get("chapter_draft", "")

    # ... 复用现有 LLM 调用逻辑 ...

    url = _generate_tts_via_matrix(
        selected_text, ctx.get("project_name", ""), ctx.get("current_chapter_number", 1)
    )
    if url:
        return {"audio_url": url, "tts_status": "success"}
    return {"audio_url": "", "tts_status": "failed"}


# ── 聚合器节点 ────────────────────────────────────────────────────────────

def media_aggregator(state: MediaState) -> dict:
    """聚合并行结果，填充最终输出。"""
    summary = []

    if state["illustration_status"] == "success":
        summary.append(f"插图生成成功：{state['illustration_url'][:50]}...")
    else:
        summary.append("插图生成失败（已降级）")

    if state["tts_status"] == "success":
        summary.append(f"语音生成成功：{state['audio_url'][:50]}...")
    else:
        summary.append("语音生成失败（已降级）")

    return {
        "crew_result": {
            "illustration_url": state.get("illustration_url", ""),
            "illustration_prompt": state.get("illustration_prompt", ""),
            "audio_url": state.get("audio_url", ""),
            "media_summary": " | ".join(summary),
        }
    }


# ── 图构建 ────────────────────────────────────────────────────────────────

def build_media_graph() -> StateGraph:
    builder = StateGraph(MediaState)

    builder.add_node("illustrator", RunnableLambda(illustrator_node))
    builder.add_node("tts", RunnableLambda(tts_node))
    builder.add_node("aggregator", RunnableLambda(media_aggregator))

    # 并行启动两个 Worker
    builder.add_edge(START, "illustrator")
    builder.add_edge(START, "tts")

    # 聚合
    builder.add_edge("illustrator", "aggregator")
    builder.add_edge("tts", "aggregator")
    builder.add_edge("aggregator", END)

    return builder.compile()
```

### 3.2 CRAG 风格回退链的条件路由实现

```python
"""media_crag.py — CRAG 风格回退链的条件路由示例。"""

from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ── 状态定义 ───────────────────────────────────────────────────────────────

class FallbackState(TypedDict):
    prompt: str
    project_name: str
    chapter_number: int

    # 输出
    image_url: str
    fallback_level: int          # 0=主路径成功, 1=回退, 2=二次回退


# ── 回退路由函数 ──────────────────────────────────────────────────────────

def route_illustration(
    state: FallbackState,
) -> Literal["fallback_illustrator", "finalize"]:
    """主路径失败 → 回退路径；主路径成功 → 结束。"""
    if state.get("image_url", "").startswith("http"):
        return "finalize"
    return "fallback_illustrator"


def route_fallback_1(
    state: FallbackState,
) -> Literal["fallback_illustrator_2", "finalize"]:
    """一级回退失败 → 二级回退；一级回退成功 → 结束。"""
    if state.get("image_url", "").startswith("http"):
        return "finalize"
    return "fallback_illustrator_2"


# ── 节点实现 ───────────────────────────────────────────────────────────────

def primary_illustrator(state: FallbackState) -> dict:
    """主路径：Matrix MCP 生成插画（复用 _generate_image_via_matrix）。"""
    from novelfactory.agents.media_agents import _generate_image_via_matrix

    url = _generate_image_via_matrix(
        state["prompt"],
        state["project_name"],
        state["chapter_number"],
    )
    return {
        "image_url": url,
        "fallback_level": 0 if url else 0,  # fallback_level 由后续节点覆盖
    }


def fallback_illustrator_1(state: FallbackState) -> dict:
    """回退路径_1：备用 API（DALL-E / Stable Diffusion）。"""
    import subprocess, json

    cmd = [
        "mavis", "mcp", "call", "matrix", "matrix_generate_image",
        "--arg", json.dumps({
            "prompt": state["prompt"],
            "model": "dall-e-3",          # 切换模型
            "aspect_ratio": "16:9",
            "resolution": "1024x1024",
        }),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        url = json.loads(result.stdout).get("url", "") if result.stdout else ""
    except Exception:
        url = ""

    return {"image_url": url, "fallback_level": 1}


def fallback_illustrator_2(state: FallbackState) -> dict:
    """二次回退：纯文字描述（无实际图像）。"""
    return {
        "image_url": "",                    # 无图
        "fallback_level": 2,                # 标记为二级降级
    }


def finalize(state: FallbackState) -> dict:
    """最终输出节点。"""
    level_map = {0: "主路径", 1: "回退路径_1", 2: "二次回退（降级）"}
    return {
        "crew_result": {
            "illustration_url": state["image_url"],
            "illustration_prompt": state["prompt"],
            "fallback_info": level_map.get(state["fallback_level"], "unknown"),
        }
    }


# ── 图构建 ────────────────────────────────────────────────────────────────

def build_crag_media_graph() -> StateGraph:
    builder = StateGraph(FallbackState)

    builder.add_node("primary", primary_illustrator)
    builder.add_node("fallback_1", fallback_illustrator_1)
    builder.add_node("fallback_2", fallback_illustrator_2)
    builder.add_node("finalize", finalize)

    # 主路径
    builder.add_edge(START, "primary")
    builder.add_conditional_edges(
        "primary",
        route_illustration,
        {"fallback_illustrator": "fallback_1", "finalize": "finalize"},
    )

    # 回退路径_1
    builder.add_conditional_edges(
        "fallback_1",
        route_fallback_1,
        {"fallback_illustrator_2": "fallback_2", "finalize": "finalize"},
    )

    # 二次回退后直接结束
    builder.add_edge("fallback_2", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
```

### 3.3 与现有 `_generate_image_via_matrix` 和 `_generate_tts_via_matrix` 的集成点

当前 `_generate_image_via_matrix` 和 `_generate_tts_via_matrix` 是同步的 subprocess 包装函数，返回 URL 或空字符串。以下是两种集成方式：

#### 方式 A：直接复用（零侵入）

在 LangGraph 节点内部直接调用现有辅助函数，无需修改 `media_agents.py`：

```python
def illustration_node(state):
    from novelfactory.agents.media_agents import _generate_image_via_matrix
    url = _generate_image_via_matrix(prompt, project_name, chapter_number)
    # 将结果写入 state 而非隐藏
    return {"image_url": url, "illustration_status": "success" if url else "failed"}
```

**优点**：`media_agents.py` 无需任何修改，纯新增

#### 方式 B：提取为重试版（推荐长期方案）

将现有辅助函数升级为带重试和降级的版本，供 LangGraph 节点使用：

```python
# media_utils.py（新增）
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_image_with_retry(prompt: str, project_name: str, chapter: int) -> str:
    """带重试的图片生成包装器。"""
    from novelfactory.agents.media_agents import _generate_image_via_matrix
    url = _generate_image_via_matrix(prompt, project_name, chapter)
    if not url:
        raise RuntimeError("Image generation returned empty URL")
    return url
```

#### 集成上下文传递关系

```
                    ┌─────────────────────────────────────────────┐
                    │              MediaState                      │
                    │  refined_chapter, draft, world_setting, ...  │
                    └──────────┬──────────────────────┬────────────┘
                               │                      │
            ┌──────────────────▼──┐        ┌──────────▼──────────┐
            │  illustrator_node   │        │    tts_node          │
            │                     │        │                      │
            │  _get_context()     │        │  _get_context()      │
            │  → LLM → prompt     │        │  → LLM → text        │
            │  → _generate_...    │        │  → _generate_...     │
            │  → url / ""         │        │  → url / ""          │
            └──────────┬──────────┘        └──────────┬───────────┘
                       │                              │
                       └──────────┬───────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │    media_aggregator       │
                    │  合并结果 + 决策          │
                    └──────────────────────────┘
```

---

## 4. 与综合文档的关联

| 综合文档章节 | 模式 | 本文对应内容 | 优先级 |
|-------------|------|-------------|--------|
| 1.2 Common Workflows — Parallelization | 并行 + 聚合器 | 第 3.1 节 | **高** — 立即受益，改动最小 |
| 3.3 Corrective RAG | CRAG 回退链 | 第 3.2 节 | **中** — 适合基础设施完善后 |
| 2.1 Customer Support — Error Handling | 重试 + 降级 | 第 3.3 节 | **高** — 提升稳定性 |

### 实施路线图建议

1. **第一阶段**（低风险，高收益）：引入 `media_aggregator` 节点 + `generate_image_with_retry` 包装器
   - 改动量：< 100 行新增代码
   - 零侵入：不修改现有 `media_agents.py`
   - 效果：失败可观测、可重试

2. **第二阶段**（中等风险）：启用 CRAG 回退链的条件路由
   - 将一级降级路由到不同模型
   - 需要确认备选 API（DALL-E / Stable Diffusion）的可达性

3. **第三阶段**（长期）：将 Media 图作为子图嵌入顶层 Agent 图
   - 参考综合文档「通用改进建议 — 子图组合」一节
   - 使 Media Crew 成为可复用的编译子图

---

> 本文档基于 LangGraph 官方教程 (v5.4.0+) 整理，建议配合官方文档阅读。
> 官方地址：https://langchain-ai.github.io/langgraph/tutorials/
