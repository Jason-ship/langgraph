# Setup Agent — LangGraph 模式参考

> 本文档是 [LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md) 的子文档，专注 Setup 智能体

---

## 一、当前架构概览

### 涉及文件

- **[setup_agents.py](setup_agents.py)** — 所有 Setup 智能体的实现
- **[LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md)** — 综合模式参考（本文档的父文档）

### 智能体列表

| 智能体 | 类/函数 | 工具绑定 | 输出类型 | 说明 |
|--------|---------|----------|----------|------|
| **WorldBuilder** | `create_world_builder_agent()` | 无（纯 LLM） | `WorldBuilderOutput.world_setting` | 构建世界观，纯推理 |
| **CharacterDesigner** | `create_character_designer_agent()` | 无（纯 LLM） | `CharacterDesignerOutput.character_setting` | 设计角色，纯推理 |
| **OutlineWriter** | `create_outline_writer_agent()` | Neo4j 图谱工具 | `OutlineWriterOutput.story_outline` + `volume_structure` | 卷级大纲，JSON 输出 |
| **VolumeDetailWriter** | `create_volume_detail_writer_agent()` | Neo4j 图谱工具 | `VolumeDetailOutput.chapter_outlines_detail` | 逐章大纲，JSON 数组输出 |

### 当前图结构

```
Setup Crew（Supervisor）
    │
    ▼
WorldBuilder ──→ CharacterDesigner ──→ OutlineWriter ──→ VolumeDetailWriter
    │                   │                     │
    ▼                   ▼                     ▼
crew_result         crew_result           crew_result
.world_setting      .character_setting    .story_outline
                                         .volume_structure  .chapter_outlines_detail
```

所有智能体通过 `crew_result` 字典传递上下文。每个节点从 `crew_result` 读取前序输出，追加自身输出后写回。

### 当前模式

- **Prompt Chaining（线性链）**：WorldBuilder → CharacterDesigner → OutlineWriter → VolumeDetailWriter，严格顺序执行
- **Orchestrator-Worker**：Setup Supervisor 编排 4 个 Worker，每个 Worker 是独立的 `create_react_agent`
- **动态 Prompt**：OutlineWriter 和 VolumeDetailWriter 使用 `_build_outline_dynamic_prompt` / `_build_volume_detail_dynamic_prompt` 函数，在运行时根据状态动态组装 SystemMessage（含工具使用指引）

---

## 二、可借鉴的 LangGraph 模式

### 2.1 Plan-and-Execute（规划执行）

> 来源：[LANGGRAPH_PATTERNS_REFERENCE.md 第 5.1 节](LANGGRAPH_PATTERNS_REFERENCE.md#1-setup-智能体)

**当前映射关系**：

| Plan-and-Execute 角色 | 当前实现 | 说明 |
|------------------------|----------|------|
| **Planner** | `OutlineWriter` | 生成卷级大纲（`volume_structure`），决定"写什么" |
| **Executor** | `VolumeDetailWriter` | 根据卷级大纲逐章细化执行，决定"怎么写" |
| **Replanner** | 缺失 | 当前无回退机制 |

**核心问题**：当 VolumeDetailWriter 发现某卷规划不合理（如章节数过多/过少、因果链断裂、伏笔无法回收）时，当前只能硬处理（`_parse_chapter_outlines_json` 返回空列表），无法回退到 OutlineWriter 重新规划。

**建议增强**：

```
VolumeDetailWriter
    │
    ├── 执行成功 ──→ 完成（结束）
    │
    └── 发现规划问题 ──→ Replanner 评估
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              调整当前卷   回退重规划   标记降级
              (补丁)     (重新调用     (跳过该卷)
                          OutlineWriter)
```

#### 联合类型定义（Act）

```python
from typing import TypedDict, Union, Literal

# Plan-and-Execute: 区分"继续规划" vs "结束" vs "调整"
class ActComplete(TypedDict):
    action: Literal["complete"]
    message: str

class ActReplan(TypedDict):
    action: Literal["replan"]
    volume_number: int
    reason: str
    suggestion: str

class ActAdjust(TypedDict):
    action: Literal["adjust"]
    volume_number: int
    patch: dict  # 局部调整建议，不触发完整重规划

Act = Union[ActComplete, ActReplan, ActAdjust]
```

#### Replanner 条件边实现

```python
from typing import Literal
from langgraph.graph import StateGraph, END

def replanner_node(state: SetupState) -> dict:
    """Replanner 节点：评估 VolumeDetailWriter 的输出质量。

    检查维度：
    1. 章节数量是否在合理范围（30-50 章）
    2. 章节之间的因果链是否完整（相邻章节是否有 core_events 继承）
    3. 伏笔是否有头有尾（cliffhanger 非空检查）
    4. importance 分布是否合理（高潮章与铺垫章比例）
    """
    outlines = state["crew_result"].get("chapter_outlines_detail", [])
    volume_number = state["crew_result"].get("volume_number", 0)

    if not outlines:
        return {
            "act": {
                "action": "replan",
                "volume_number": volume_number,
                "reason": "章节大纲为空，LLM 未能生成有效输出",
                "suggestion": "请重新规划本卷，注意章节数量与因果链",
            }
        }

    # 检查因果链：相邻章节是否有 core_events
    broken_chains = []
    for i in range(1, len(outlines)):
        prev = outlines[i - 1]
        curr = outlines[i]
        if not prev.get("core_events") or not curr.get("core_events"):
            broken_chains.append(curr["chapter_number"])

    if len(broken_chains) > len(outlines) * 0.3:
        return {
            "act": {
                "action": "replan",
                "volume_number": volume_number,
                "reason": f"超过 30% 的章节缺乏因果链（章节: {broken_chains}）",
                "suggestion": "确保每章都有明确的 core_events，相邻章节存在因果",
            }
        }

    # 检查高潮分布：预期 10% 章节 importance >= 8
    high_importance = [c for c in outlines if c.get("importance", 5) >= 8]
    if len(high_importance) < max(1, len(outlines) // 10):
        return {
            "act": {
                "action": "adjust",
                "volume_number": volume_number,
                "patch": {
                    "suggestion": "本卷缺乏高潮章节，建议在 25%, 50%, 75% 位置增加 importance >= 8 的章节"
                },
            }
        }

    return {"act": {"action": "complete", "message": "本卷规划通过质量检查"}}


def route_after_replanner(state: SetupState) -> Literal["outline_writer", "volume_detail_writer", "__end__"]:
    """根据 Replanner 的决策路由。"""
    act = state["act"]
    if act["action"] == "complete":
        return "__end__"
    elif act["action"] == "replan":
        # 回退到 OutlineWriter 重新规划
        return "outline_writer"
    else:
        # adjust: 回到 VolumeDetailWriter 做微调
        return "volume_detail_writer"


# ── 在 StateGraph 中注册 ──
graph = StateGraph(SetupState)

# ... 注册各节点 ...

graph.add_conditional_edges(
    "volume_detail_writer",
    route_after_replanner,
    {
        "outline_writer": "outline_writer",
        "volume_detail_writer": "volume_detail_writer",
        "__end__": END,
    },
)
```

### 2.2 Basic Reflection（基础反思）

> 来源：[LANGGRAPH_PATTERNS_REFERENCE.md 第 6.1 节](LANGGRAPH_PATTERNS_REFERENCE.md#1-setup-智能体)

**当前问题**：WorldBuilder 和 CharacterDesigner 的输出没有自检环节，可能出现世界观自洽性问题或角色弧线不完整的情况。

**建议实现**：

```python
REFLECTION_PROMPT = """\
你是一个质量控制专家，请对以下 {agent_name} 的输出进行反思检查。

## 检查清单
1. **内部一致**：设定内部是否有矛盾？（如力量体系与地理环境的冲突）
2. **完整性**：是否覆盖了所有必要维度？
3. **可扩展性**：设定是否支持 100+ 章节的剧情容量？
4. **独特性**：设定是否有区别于同类作品的特色？

## 输出格式
{
  "passed": bool,          // 是否通过反思
  "issues": [string],      // 发现的问题列表
  "suggestions": [string], // 改进建议
  "refined_output": str    // 优化后的完整输出（或保持原样）
}
"""

def reflection_node(state: SetupState) -> dict:
    """通用反思节点，对指定的 agent 输出进行自我检查。"""
    agent_name = state.get("reflection_target", "WorldBuilder")
    
    if agent_name == "WorldBuilder":
        output_key = "world_setting"
    elif agent_name == "CharacterDesigner":
        output_key = "character_setting"
    else:
        return {"reflection_result": {"passed": True}}
    
    raw_output = state["crew_result"].get(output_key, "")
    if not raw_output:
        return {"reflection_result": {"passed": False, "issues": ["输出为空"]}}
    
    # 调用 LLM 进行反思（使用独立的小模型或同一模型）
    # ... reflection LLM call ...
    
    result = {"passed": True, "issues": [], "suggestions": [], "refined_output": raw_output}
    # 如果发现严重问题，不通过反思
    # result["passed"] = False
    # result["refined_output"] = refined_text
    
    return {"reflection_result": result}


def route_after_reflection(state: SetupState) -> Literal["retry", "next"]:
    """反思条件边：通过则进入下一阶段，不通过则重试。"""
    result = state.get("reflection_result", {})
    if result.get("passed", True):
        return "next"
    return "retry"
```

**整合到 Setup Crew**：

```
WorldBuilder → Reflect → [通过 → CharacterDesigner]
                       → [不通过 → WorldBuilder（重试）]

CharacterDesigner → Reflect → [通过 → OutlineWriter]
                             → [不通过 → CharacterDesigner（重试）]
```

### 2.3 STORM（并行角色生成）

> 来源：[LANGGRAPH_PATTERNS_REFERENCE.md 第 8.1 节](LANGGRAPH_PATTERNS_REFERENCE.md#1-setup-智能体)

**核心思路**：借鉴 STORM 的 `InterviewState` 独立子图模式，使用 `Send()` API 实现多个角色的并行生成。每个角色从不同视角出发，最终汇总成统一的角色设定文档。

**适用场景**：

1. **多视角角色分析**：对同一世界观，从"正派视角"、"反派视角"、"中立视角"三个角度并行生成角色设定
2. **批量角色生成**：当需求大量配角时（如学院故事需要 10+ 配角），并行生成显著缩短耗时
3. **角色关系拓扑**：每个角色生成时独立查询 Neo4j 关系图谱，互不阻塞

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict, List
import operator


# ── Interview Subgraph State ───────────────────────────────────────────────────

class InterviewState(TypedDict):
    """单个角色的"面试"子图状态。
    
    类比 STORM 中的 InterviewState，每个角色视为一次独立的"采访"。
    """
    role_id: str                     # 角色标识（如 "protagonist", "antagonist", "supporter_1"）
    role_perspective: str            # 角色视角描述（如 "被压迫的底层少年"）
    world_setting: str               # 共享的世界观设定
    character_draft: str             # 该角色的生成草稿
    iteration_count: int             # 反思迭代次数


class CharacterInterviewSubgraph:
    """独立子图：一次完整的角色"面试"。
    
    每个子图独立运行，包含：生成 → 反思 → 可能的重试。
    """
    
    @staticmethod
    def build() -> StateGraph:
        graph = StateGraph(InterviewState)
        
        # 节点
        graph.add_node("generate_character", _generate_character)
        graph.add_node("reflect_character", _reflect_character)
        graph.add_node("finalize_character", _finalize_character)
        
        # 边
        graph.add_edge(START, "generate_character")
        graph.add_edge("generate_character", "reflect_character")
        graph.add_conditional_edges(
            "reflect_character",
            _decide_after_reflection,
            {
                "retry": "generate_character",      # 反思不通过，重新生成
                "finalize": "finalize_character",    # 反思通过，进入最终节点
            },
        )
        graph.add_edge("finalize_character", END)
        
        return graph.compile()


# ── 顶层并行调度 ───────────────────────────────────────────────────────────────

class SetupState(TypedDict):
    seed_idea: str
    world_setting: str
    character_settings: Annotated[list, operator.add]  # 用 operator.add 实现聚合
    ...


def spawn_character_interviews(state: SetupState) -> list[Send]:
    """动态分发：为每个需要生成的角色创建一个 Send 任务。
    
    每个 Send 启动一个 InterviewState 子图，并行执行。
    类似 STORM 中为每个资料生成 InterviewState 的模式。
    """
    # 定义需要并行生成的角色列表
    roles = [
        {
            "role_id": "protagonist",
            "role_perspective": "故事的主角，从底层崛起的少年",
        },
        {
            "role_id": "antagonist",
            "role_perspective": "核心反派，与主角形成镜像对立",
        },
        {
            "role_id": "supporter_1",
            "role_perspective": "主角的挚友，提供情感支持",
        },
        {
            "role_id": "supporter_2",
            "role_perspective": "中立势力代表，亦敌亦友",
        },
    ]
    
    sends = []
    for role in roles:
        sends.append(
            Send(
                "character_interview_subgraph",  # 目标子图节点
                {
                    "role_id": role["role_id"],
                    "role_perspective": role["role_perspective"],
                    "world_setting": state["world_setting"],
                    "character_draft": "",
                    "iteration_count": 0,
                },
            )
        )
    return sends


def aggregate_characters(state: SetupState) -> dict:
    """聚合器：收集所有并行子图的输出，合并为统一文档。
    
    在 STORM 模式中对应 finalize 阶段，将所有"采访"结果
    综合成一篇完整的文章。此处将所有角色设定合并。
    """
    all_characters = state.get("character_settings", [])
    
    # 按角色重要性排序
    priority = {"protagonist": 0, "antagonist": 1, "supporter": 2}
    all_characters.sort(
        key=lambda c: priority.get(c.get("role_id", "supporter"), 99)
    )
    
    # 合并为完整文档
    merged = "# 角色设定文档\n\n"
    for char in all_characters:
        merged += f"## {char.get('role_id', '未知角色')}\n\n"
        merged += f"视角：{char.get('role_perspective', '')}\n\n"
        merged += f"{char.get('character_draft', '')}\n\n---\n\n"
    
    return {"crew_result": {"character_setting": merged}}


# ── 注册到主图 ─────────────────────────────────────────────────────────────────

def build_setup_storm_graph():
    """构建带有 STORM 并行角色生成能力的 Setup 图。"""
    
    graph = StateGraph(SetupState)
    
    # 常规节点
    graph.add_node("world_builder", world_builder_node)
    graph.add_node("character_interview_subgraph", character_interview_subgraph)
    graph.add_node("aggregate_characters", aggregate_characters)
    graph.add_node("outline_writer", outline_writer_node)
    graph.add_node("volume_detail_writer", volume_detail_writer_node)
    
    # 边：STORM 并行分发
    graph.add_edge(START, "world_builder")
    graph.add_conditional_edges(
        "world_builder",
        spawn_character_interviews,                 # 动态分发 Send
        path_map=["character_interview_subgraph"],   # 全部分发到子图
    )
    # 所有子图完成后，聚合到 aggregate_characters
    graph.add_edge("character_interview_subgraph", "aggregate_characters")
    
    # 后续线性执行
    graph.add_edge("aggregate_characters", "outline_writer")
    graph.add_edge("outline_writer", "volume_detail_writer")
    graph.add_edge("volume_detail_writer", END)
    
    return graph.compile()
```

#### 与当前实现的差异对比

| 维度 | 当前实现 | STORM 模式改造后 |
|------|----------|-----------------|
| 角色生成方式 | 单次 LLM 调用，一次性输出 3-5 个角色 | 每个角色独立子图，并行生成 |
| 反思机制 | 无 | 每个角色子图内置反射节点 |
| 总耗时 | 线性：O(N) | 并行：O(1)（受并发限制） |
| 输出质量 | 角色间质量不均，独立性差 | 每个角色独立优化，质量更可控 |
| 复杂度 | 简单函数 | 子图 + Send() 分发 + 聚合器 |

---

## 三、与综合文档的关系

本文档是 [LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md) 在 **Setup 智能体** 维度上的深度展开。综合文档中的以下章节与本文档直接相关：

| 综合文档章节 | 本文档对应内容 | 关系说明 |
|-------------|----------------|----------|
| 第 1 节（Setup 智能体） | 全文 | 父级概述，本文档是其详细实现参考 |
| 第 5.1 节（Plan-and-Execute） | 第 2.1 节 | 提供了 Replanner 模式的完整代码模板 |
| 第 6.1 节（Basic Reflection） | 第 2.2 节 | 提供了反思节点的可复用实现 |
| 第 8.1 节（STORM） | 第 2.3 节 | 提供了并行角色生成的完整子图方案 |
| 第 7 节（通用改进建议） | 第 1 节 | 中断机制、子图组合、状态持久化均适用于 Setup |

---

> 本文档基于 LangGraph 官方教程 (v5.4.0+) 整理，建议配合 [LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md) 和 [setup_agents.py](setup_agents.py) 阅读。
> 官方地址：https://langchain-ai.github.io/langgraph/tutorials/
