# Writing Agent — LangGraph 模式参考

> 本文档是 [LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md) 的子文档，专注 Writing 智能体

---

## 第一节 — 当前架构概览

### 涉及文件

| 文件 | 作用 |
|------|------|
| [writing_agents.py](src/novelfactory/agents/writing_agents.py) | 三个智能体的 Agent 工厂函数（create_react_agent + RunnableLambda 封装） |
| [writing_crew.py](src/novelfactory/graph/crews/writing_crew.py) | Writing Crew 子图的 StateGraph 构建（节点注册 + 条件边路由） |
| [writer.py](src/novelfactory/graph/crews/writing_nodes/writer.py) | `_chapter_writer_node` — 调用 ChapterWriter 写入草稿 |
| [reviewer.py](src/novelfactory/graph/crews/writing_nodes/reviewer.py) | `_chapter_reviewer_node` 和 `_chapter_refiner_node` — 评分 + 润色 |
| [routing.py](src/novelfactory/graph/crews/writing_nodes/routing.py) | `_score_router` 条件边路由 + `_exit_for_chapter` 出口节点 |

### 智能体列表

| 智能体 | 工厂函数 | 绑定工具 | 输出 |
|--------|----------|----------|------|
| **ChapterWriter** | `create_chapter_writer_agent` | Neo4j（角色关系）+ Milvus（相似章节语义检索） | `{ "chapter_draft": str }` |
| **ChapterReviewer** | `create_chapter_reviewer_agent` | 无（纯 LLM 评分） | `{ "quality_score": float, "review_comments": str, "needs_refine": bool }` |
| **ChapterRefiner** | `create_chapter_refiner_agent` | Neo4j（润色时确认角色关系一致性） | `{ "refined_chapter": str }` |

### 当前图结构：Writing Crew

```
START
  │
  ▼
context_builder_node    ← 构建分层上下文（ContextBuilder 子图）
  │
  ▼
chapter_writer          ← ChapterWriter ReAct Agent（写作 + 工具调用）
  │
  ▼
chapter_reviewer        ← ChapterReviewer + 综合评分器（AI味/老书虫/复合分）
  │
  ▼
_score_router           ← 条件路由（双条件门控）
  ├─── "chapter_writer"              → 分数 < 阈值（重写循环）
  ├─── "chapter_refiner"             → 分数 60-89（润色环）
  │       │
  │       ▼
  │   chapter_reviewer               ← 重审
  │       │
  │       └─── 再次路由
  │
  └─── "__exit_for_chapter__"
          │
          ▼
    state_extractor_node → database_writer_node → _exit_for_chapter → END
```

### 当前模式：Evaluator-Optimizer + Supervisor

**Evaluator-Optimizer（评估器-优化器）**：
- ChapterWriter 生成草稿（Actor）
- ChapterReviewer 评分（Evaluator）
- ChapterRefiner 润色（Optimizer）
- 循环条件：`quality_score`（四维） + `composite_score`（综合）双条件门控

**Supervisor（监督路由）**：
- `_score_router` 作为条件边控制器，根据评分结果动态路由：
  - `quality_score >= genre_threshold` 且 `composite_score >= composite_ok` → 通过
  - `quality_score >= 60` → chapter_refiner（润色环，最多 1-2 次）
  - `quality_score < 60` → chapter_writer（重写循环，最多 3 次）
  - 循环用尽 → 强制通过（`__exit_for_chapter__`）

**题材感知评分**（v5.5）：
- `_score_router` 从 `crew_result.genre` 读取题材
- 调用 `get_genre_thresholds(genre)` 获取动态阈值
- 不同题材（爽文/仙侠/都市/悬疑等）有不同的通过标准
- 题材感知指引注入 Reviewer 的 scoring_prompt

**循环护盾**：
- `MAX_REWRITE_ATTEMPTS = 3`：重写次数上限
- `REFINE_MAX_ATTEMPTS = { "high": 1, "mid": 2 }`：润色次数上限
- 循环用尽 → 自动通过（防死循环）+ 请求人工指导

---

## 第二节 — 可借鉴的 LangGraph 模式

### 5.1 Plan-and-Execute（规划执行）

**当前状态**：ChapterWriter 已有 `<thinking>` 标签内的章节规划（核心情节点、衔接设计、人物心理轨迹、感官清单、字数分配），但规划是自由文本嵌入在 prompt 中的。

**升级建议**：
- 将 `<thinking>` 升级为结构化的 `Plan` Pydantic 模型输出
- Plan 包含：步骤列表、每段字数分配、感官描写计划、伏笔埋设计划
- 增加 **Replanner** 节点，在 ChapterWriter 完成后检查是否按计划执行
- 当 Replanner 检测到偏离计划时，可触发重新规划或局部修正

**关键 API 参考**：
```python
class Plan(BaseModel):
    steps: list[str]          # 执行步骤列表
    word_counts: dict[str, int]  # 每段字数分配
    sensory_plan: list[str]   # 感官描写计划
    foreshadowing: list[str]  # 伏笔埋设计划

class Act(BaseModel):
    response: str
    plan: list[str]  # 可继续规划或结束
```

### 5.2 ReWOO（减少冗余 LLM 调用）

**当前状态**：ChapterWriter 在写作过程中可能多次调用 Neo4j/Milvus 工具，每次工具调用都消耗 LLM 的推理 token。

**优化建议**：
- 采用 `#E1 = Tool[args]` 变量替换模式
- 阶段 1（Planner）：先规划所有需要的工具调用（仅一次 LLM 调用）
- 阶段 2（Worker）：一次性执行所有工具调用（无 LLM 调用）
- 阶段 3（Solver）：基于工具返回的结果写作（一次 LLM 调用）
- 从 3+ 次 LLM 调用 → 2 次，节约 30-40% token

```python
# ReWOO 模式
# 1. Planner 输出结构化工具调用计划
{
  "plan": [
    "#E1 = get_character_network(林默)",
    "#E2 = search_similar_chapters(突破金丹的经典场景)",
    "#E3 = get_plot_threads(unfinished)"
  ]
}
# 2. Worker 执行所有工具，替换 #E1, #E2, #E3 为实际结果
# 3. Solver 基于替换后的结果写作
```

### 5.3 LLMCompiler（流式 DAG 任务执行）

**当前状态**：写作 → 评分 → 润色是严格串行的（writer → reviewer → refiner → reviewer），无法并行。

**优化建议**：
- 将写作者、评分者、润色者组织成流式 DAG
- **Joiner** 聚合器节点加速循环：
  - 写作者完成 → 同步启动评分和润色（并行）
  - 评分者完成 → 如果分数高直接通过，无需等待润色结果
  - 润色者完成 → 重新评分（如分数不够）

```python
# LLMCompiler 流式 DAG
def joiner(state):
    if state.writing_done and state.scoring_done:
        if state.quality_score >= 90:
            return "approve"
        elif state.refine_done:
            return "re-review"
        else:
            return "refine"
```

### 6.3 Tree of Thoughts (ToT)

**当前状态**：每次只生成一个章节草稿，没有候选选择机制。

**优化建议**：
- **Expand**：ChapterWriter 并行生成多个章节草稿候选（2-3 个）
- **Score**：ChapterReviewer 评估每个候选的质量
- **Prune**：保留最高分候选，或并行推进多个候选到下一轮
- `Send()` API 实现并行扩展
- 适用场景：关键转折章节（第 10% / 50% / 90% 章节）

```python
# ToT: Send() API 并行扩展
def expand(state):
    candidates = [
        Send("chapter_writer", {"chapter_setting": f"风格{i}"})
        for i in range(num_candidates)
    ]
    return candidates  # Send() 列表

# 评分 + 剪枝
def score_candidates(state):
    scored = sorted(
        state.candidates,
        key=lambda c: c["quality_score"],
        reverse=True
    )
    state.candidates = scored[:beam_size]  # 保留前 K 个
```

### 6.2 Reflexion（高级反思）

**当前状态**：ChapterReviewer 进行四维评分，但仅评估"质量问题"（好不好），不评估"信息完整性"（缺不缺 / 多不多）。

**优化建议**：
- 增加对"信息完整性"的检查：
  - **缺失检查**：章节是否遗漏了关键情节点（如前文伏笔、角色成长节点）
  - **多余检查**：章节是否包含不必要的描述（如与主线无关的支线）
- 反思结果可作为 ChapterWriter 重写的直接输入

```python
# Reflexion 反思 Prompt
reflection_prompt = """
## 信息完整性分析
请分析以下章节，输出两部分评估：

### 缺失信息
- 前文伏笔未回收：______（例如：玉佩在第3章出现后本章再无提及）
- 角色发展遗漏：______（例如：主角的金丹突破过程缺少心境变化描写）
- 情节因果断裂：______（例如：第5段与第6段之间缺少过渡事件）

### 多余信息
- 冗余描述：______（例如：第3段对环境的300字描写与剧情无关）
- 旁支情节：______（例如：第4段引入的支线人物对主线无贡献）
- 重复表达：______（例如：第2段与第7段对同一事物的描述重复）
"""
```

### 8.4 Complex Data Extraction（复杂数据提取）

**当前状态**：`validate_json_output` 实现了基本的 JSON 解析和必填字段检查，使用 `fail_closed=False` 的容错策略。

**优化建议**：
- 引入 `ValidatorNode` 模式，在数据提取节点中绑定 Pydantic 的 `field_validator`
- 字段级验证：`quality_score` 范围检查（0-100）、`review_comments` 长度检查
- 自修复重试：验证失败时自动调整满足约束

```python
class ReviewScore(BaseModel):
    quality_score: float
    review_comments: str
    needs_refine: bool

    # 字段级验证
    @field_validator("quality_score")
    @classmethod
    def check_score_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError(f"quality_score {v} 超出 0-100 范围")
        return v

    @field_validator("review_comments")
    @classmethod
    def check_comments_not_empty(cls, v):
        if not v.strip():
            raise ValueError("review_comments 不能为空")
        return v

# ValidatorNode 模式
review_extractor = ValidatorNode(
    ReviewScore,
    llm=llm,
    max_retries=3,
    fallback={"quality_score": 70.0, "review_comments": "审核失败", "needs_refine": True},
)
```

---

## 第三节 — 补充代码模板

### 3.1 ReWOO 变量替换模式 — 完整实现

```python
"""ReWOO 模式：Planner → Worker → Solver 三步替代 ReAct 循环。

在 ChapterWriter 中应用：
  阶段1（Planner）：LLM 一次性规划所有工具调用
  阶段2（Worker）：逐一执行工具，将结果替换回变量
  阶段3（Solver）：LLM 基于替换后的完整上下文写作
"""

from typing import Any

import re


# ── Pydantic Models ─────────────────────────────────────────────────────

class ToolPlan(BaseModel):
    """结构化工具调用计划。"""
    plan: list[str]  # ["#E1 = get_character_network(林默)", "#E2 = search_similar_chapters(突破)"]


# ── State ────────────────────────────────────────────────────────────────

class ReWOOState(TypedDict):
    plan: list[str]               # Planner 生成的调用计划
    tool_results: dict[str, str]  # {"#E1": "林默——师傅：青云真人", "#E2": "..."}
    chapter_draft: str            # Solver 输出的章节草稿


# ── Nodes ────────────────────────────────────────────────────────────────

def planner_node(state: ReWOOState, llm: BaseChatModel) -> dict:
    """阶段1：LLM 规划工具调用 —— 仅一次 LLM 调用。"""
    context = _build_planning_context(state)
    prompt = f"""请规划本章写作所需的工具调用。
当前上下文：{context}

输出格式：按需列出工具调用，每行一个。
每个调用使用 #E1, #E2, ... 编号。
示例：
#E1 = get_character_network(林默)
#E2 = search_similar_chapters(突破金丹的情节设计)

至少调用 get_all_characters() 获取角色列表，
其他工具按需调用。

## 规则
- 只在必要时调用工具
- 最多规划 3 个工具调用
- 不要重复调用相同工具"""

    response = llm.invoke(prompt)
    plan = _parse_plan(response.content)
    return {"plan": plan}


def worker_node(state: ReWOOState, tools: dict[str, Any]) -> dict:
    """阶段2：执行所有工具调用 —— 无 LLM 调用，纯执行。"""
    from novelfactory.tools import get_neo4j_tools, get_milvus_tools

    all_tools = {t.name: t for t in get_neo4j_tools() + get_milvus_tools()}
    results = {}
    for step in state["plan"]:
        # 解析 "#E1 = tool_name(arg1, arg2)"
        match = re.match(r"(#E\d+)\s*=\s*(\w+)\(([^)]*)\)", step)
        if not match:
            continue
        var_name, tool_name, args_str = match.groups()
        tool = all_tools.get(tool_name)
        if not tool:
            results[var_name] = f"工具 {tool_name} 不存在"
            continue
        # 解析参数（简化版）
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        try:
            result = tool.invoke({"input": args[0] if args else ""})
            results[var_name] = str(result)[:2000]  # 截断避免过长
        except Exception as e:
            results[var_name] = f"调用失败：{e}"
    return {"tool_results": results}


def solver_node(state: ReWOOState, llm: BaseChatModel) -> dict:
    """阶段3：基于工具结果写作 —— 仅一次 LLM 调用。"""
    # 变量替换：将 prompt 中的 #E1 替换为实际结果
    context = _build_writing_context(state)

    # 将工具结果替换到 prompt 中
    replaced_context = context
    for var_name, result in state["tool_results"].items():
        placeholder = f"[{var_name}]"
        replaced_context = replaced_context.replace(placeholder, result)

    prompt = f"""请撰写本章正文。

## 工具查询结果
{replaced_context}

## 写作要求
{CHAPTER_WRITER_PROMPT[:500]}

请输出章节正文。"""
    response = llm.invoke(prompt)
    return {"chapter_draft": response.content}


# ── Graph Builder ─────────────────────────────────────────────────────────

def build_rewoo_writer_crew(llm: BaseChatModel) -> CompiledStateGraph:
    """ReWOO 模式的写作子图。

    优势：3 次 LLM 调用（Planner + Solver + ?）→ 2 次（Planner + Solver）。
    当前 ReAct 模式约为 3-5 次 LLM 调用（每轮工具调用都消耗推理 token）。
    """
    graph = StateGraph(ReWOOState)

    graph.add_node("planner", lambda s: planner_node(s, llm))
    graph.add_node("worker", worker_node)
    graph.add_node("solver", lambda s: solver_node(s, llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "solver")
    graph.add_edge("solver", END)

    return graph.compile()
```

### 3.2 ToT（Tree of Thoughts）— Expand/Score/Prune 模型定义 + Send() 并行

```python
"""Tree of Thoughts 模式应用于 ChapterWriter。

在关键转折章节（第 10%/50%/90%）：
  1. Expand: 并行生成 N 个候选草稿
  2. Score: 评审每个候选的质量
  3. Prune: 保留 Beam 个最佳候选，推进到下一轮
"""

from langgraph.types import Send
from pydantic import BaseModel, Field


# ── Pydantic 模型定义 ─────────────────────────────────────────────────

class ChapterCandidate(BaseModel):
    """单个章节候选。"""
    candidate_id: str = Field(description="候选唯一标识")
    draft: str = Field(description="章节草稿")
    style_label: str = Field(description="风格标签", examples=["快节奏冲突", "细腻感情戏", "悬念驱动"])
    word_count: int = Field(ge=1000, le=8000, description="字数")

    @field_validator("draft")
    @classmethod
    def draft_not_empty(cls, v):
        if not v.strip():
            raise ValueError("草稿不能为空")
        return v


class ScoredCandidate(BaseModel):
    """带评分的候选。"""
    candidate: ChapterCandidate
    quality_score: float = Field(ge=0, le=100, description="四维评分总分")
    plot_score: float = Field(ge=0, le=30, description="剧情逻辑得分")
    style_score: float = Field(ge=0, le=25, description="文笔表达得分")
    character_score: float = Field(ge=0, le=25, description="人物一致性得分")
    world_score: float = Field(ge=0, le=20, description="世界观契合得分")

    @field_validator("quality_score")
    @classmethod
    def score_consistent(cls, v, info):
        """验证总分与分项和一致。"""
        data = info.data
        expected = data.get("plot_score", 0) + data.get("style_score", 0) \
                 + data.get("character_score", 0) + data.get("world_score", 0)
        if abs(v - expected) > 0.5:
            raise ValueError(f"总分 {v} 与分项和 {expected} 不一致")
        return v


class ToTState(TypedDict):
    """ToT 状态。"""
    candidates: list[ChapterCandidate]  # 当前轮候选
    scored: list[ScoredCandidate]       # 评分后的候选
    round: int                          # 当前轮次
    max_rounds: int                     # 最大轮次
    beam_size: int                      # 每轮保留数 Beam = K


# ── Nodes ──────────────────────────────────────────────────────────────

def expand_candidates(state: ToTState) -> list[Send]:
    """Expand 节点：并行生成 N 个候选草稿。

    使用 Send() 实现 beam search 风格的并行扩展。
    """
    num_candidates = state.get("beam_size", 2) * 2  # 2x beam
    styles = ["快节奏冲突推进", "细腻情感+悬念铺垫", "场景描写+角色互动"]

    # 每个 Send 启动一个独立的 chapter_writer 调用
    return [
        Send(
            "generate_candidate",
            {
                "candidate_id": f"cand_{state['round']}_{i}",
                "style_label": styles[i % len(styles)],
                "writing_context": state.get("writing_context", ""),
            },
        )
        for i in range(num_candidates)
    ]


def generate_candidate(state: dict) -> dict:
    """单个候选生成节点（被 Send 调用）。"""
    candidate_id = state["candidate_id"]
    style = state["style_label"]
    context = state["writing_context"]

    # 调用 LLM 生成
    prompt = f"请以「{style}」的风格撰写本章。上下文：{context}"
    # llm.invoke(prompt) ...
    return {"draft": "...", "candidate_id": candidate_id, "style_label": style}


def score_candidates(state: ToTState) -> dict:
    """Score 节点：评估所有候选并排序。"""
    scored = []
    for cand in state["candidates"]:
        # 调用 ChapterReviewer 评估
        score = _evaluate_candidate(cand)
        scored.append(score)
    return {"scored": scored}


def prune_candidates(state: ToTState) -> dict:
    """Prune 节点：剪枝，仅保留 Beam 个最佳候选。

    可用于：
    - 最终选择：直接保留最高分候选
    - 继续迭代：保留多个候选进入下一轮
    """
    beam = state.get("beam_size", 2)
    sorted_candidates = sorted(
        state["scored"],
        key=lambda x: x.quality_score,
        reverse=True,
    )
    pruned = sorted_candidates[:beam]

    return {
        "candidates": [sc.candidate for sc in pruned],
        "round": state["round"] + 1,
    }


# ── Graph Builder ──────────────────────────────────────────────────────

def build_tot_writer_crew() -> CompiledStateGraph:
    """ToT 模式的写作子图。

    Flow:
      START → expand_candidates ──Send()──→ [generate_candidate * N]
                ↓
            merge_candidates → score_candidates → prune_candidates
                ↓
            [round < max_rounds] → expand_candidates（继续迭代）
            [round >= max_rounds] → select_best → END
    """
    graph = StateGraph(ToTState)

    graph.add_node("expand_candidates", expand_candidates)
    graph.add_node("generate_candidate", generate_candidate)
    graph.add_node("score_candidates", score_candidates)
    graph.add_node("prune_candidates", prune_candidates)

    # 并行扩展
    graph.add_conditional_edges(
        "expand_candidates",
        lambda s: s["candidates"],
        path_map=[Send("generate_candidate", ...)],
    )

    # 聚合评分
    graph.add_edge("generate_candidate", "score_candidates")
    graph.add_edge("score_candidates", "prune_candidates")

    # 迭代或结束
    graph.add_conditional_edges(
        "prune_candidates",
        lambda s: "expand" if s["round"] < s["max_rounds"] else "select",
    )

    return graph.compile()
```

### 3.3 Reflexion — 缺失/多余检查的 Prompt 模板

```python
"""Reflexion 模式：评估缺失和多余信息。

在 ChapterReviewer 评分后增加反射节点，
检查信息完整性（不仅评估质量）。
"""

# ── 缺失分析 Prompt ──────────────────────────────────────────────────

REFLEXION_MISSING_PROMPT = """\
## 缺失信息分析（Reflexion 模式）

分析以下章节，找出「应当出现但缺失」的内容。

### 1. 伏笔回收缺失
对照前文埋设的伏笔列表（若有），检查本章是否应该回收：
- 上一章结尾的悬念是否得到回应？
- 前 3 章以内埋设的道具/人物/事件是否得到推进？
- 主角设定的"金手指"词条系统是否按规范出现？

### 2. 情节因果链缺失
逐段检查因果连续性：
- 段落 A → 段落 B 之间是否有合理的因果关系？
- 是否有"跳跃式"缺失？例如从"A出门"直接到"C到达目的地"，中间缺少"B赶路过程"
- 第 X 段到第 Y 段是否需要补充过渡？

### 3. 角色发展缺失
- 主角在本章应有的心理变化是否完整？（情绪起点 → 触发事件 → 情绪终点）
- 关键配角的戏份是否与本章定位匹配？
- 角色关系中是否有该交互但未交互的情况？

### 4. 感官描写缺失
- 是否有超过 200 字无感官描写的段落？（标出具体段落号）
- 本章当前感官描写类型分布：视觉___处 / 听觉___处 / 嗅觉___处 / 触觉___处

### 输出格式
```json
{
  "missing_foreshadowing": ["第3段：玉佩伏笔在前3章埋设，但本章完全未提及"],
  "missing_causal_links": ["第5段→第6段：主角突然突破金丹，缺少灵气积累的描写过渡"],
  "missing_character_development": ["主角第4段受重伤后第5段无任何心理描写就继续战斗"],
  "missing_sensory_detail": [{"paragraph": 7, "start_char": 0, "gap_length": 250}]
}
```
"""


# ── 多余分析 Prompt ──────────────────────────────────────────────────

REFLEXION_REDUNDANT_PROMPT = """\
## 多余信息分析（Reflexion 模式）

分析以下章节，找出「可以删除或精简」的内容。

### 1. 冗余描写
- 对同一事物的反复描写（例如：环境描述出现 3 次且内容相似）
- 无信息量的对话（例如："嗯"、"好"、"是吗" 等填充对话）

### 2. 偏离主线
- 与当前情节点无关的支线展开
- 对无关配角的过多描写
- 世界观说明插入不当（"信息 dump"）

### 3. 重复表达
- 前后段表达了相同含义
- 同一个角色性格特点被多次强调
- 同一段情节通过不同方式说了两遍

### 输出格式
```json
{
  "redundant_descriptions": [{"paragraph": 4, "reason": "与第2段环境描写重复", "estimated_chars": 150}],
  "off_topic_content": [{"paragraph": 8, "reason": "引入的算命先生支线与当前情节无关"}],
  "repeated_expressions": ["第6段与第3段都强调了'主角很愤怒'"]
}
```
"""


# ── Reflexion 节点 ────────────────────────────────────────────────────

def reflexion_node(state: WritingCrewLocalState, llm: BaseChatModel) -> dict:
    """Reflexion 节点：分析缺失和多余信息。

    在 ChapterReviewer 评分后执行，输出作为 ChapterWriter 重写的补充输入。
    """
    chapter_draft = state.get("chapter_draft", "")

    # 调用缺失分析
    missing_response = llm.invoke([
        {"role": "system", "content": REFLEXION_MISSING_PROMPT},
        {"role": "user", "content": f"分析以下章节的缺失信息：\n\n{chapter_draft[:4000]}"},
    ])
    missing_result = json.loads(extract_json(missing_response.content))

    # 调用多余分析
    redundant_response = llm.invoke([
        {"role": "system", "content": REFLEXION_REDUNDANT_PROMPT},
        {"role": "user", "content": f"分析以下章节的多余信息：\n\n{chapter_draft[:4000]}"},
    ])
    redundant_result = json.loads(extract_json(redundant_response.content))

    return {
        "crew_result": {
            "reflexion_missing": missing_result,
            "reflexion_redundant": redundant_result,
            # 综合反射结果 → 注入重写 prompt
            "reflexion_guidance": _build_reflexion_guidance(missing_result, redundant_result),
        }
    }


def _build_reflexion_guidance(missing: dict, redundant: dict) -> str:
    """将缺失/多余分析结果转为自然语言写作指引。"""
    parts = ["## Reflexion 指导："]
    if missing.get("missing_foreshadowing"):
        parts.append("### 需要补充的内容：")
        for item in missing["missing_foreshadowing"]:
            parts.append(f"- {item}")
    if redundant.get("redundant_descriptions"):
        parts.append("### 可以精简的内容：")
        for item in redundant["redundant_descriptions"]:
            parts.append(f"- {item['reason']}（约节省{item.get('estimated_chars', 0)}字）")
    return "\n".join(parts)
```

### 3.4 Complex Data Extraction — ValidatorNode + field_validator

```python
"""复杂数据提取模式 — ChapterReviewer 输出验证。

使用 Pydantic field_validator 做字段级验证，
验证失败时自动重试（自修复）。
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ── 验证模型 ──────────────────────────────────────────────────────────

class ReviewOutput(BaseModel):
    """审核输出模型，带字段级验证。"""
    quality_score: float = Field(ge=0, le=100, description="四维评分总分")
    review_comments: str = Field(min_length=10, max_length=2000, description="审核意见")
    needs_refine: bool = Field(description="是否需要润色（True: <90分）")

    @field_validator("quality_score")
    @classmethod
    def score_must_be_rounded_to_1dp(cls, v):
        """强制小数位不超过1位（避免LLM给出 89.5678 这种不自然的分数）。"""
        return round(v, 1)

    @field_validator("review_comments")
    @classmethod
    def comments_must_be_specific(cls, v):
        """审核意见必须包含段落标记。"""
        if "第" not in v or "段" not in v:
            raise ValueError("审核意见必须包含具体段落编号（如'第3段'）")
        return v

    @field_validator("needs_refine")
    @classmethod
    def refine_must_match_score(cls, v, info):
        """needs_refine 必须与 quality_score 逻辑一致。"""
        data = info.data
        if "quality_score" in data:
            expected = data["quality_score"] < 90
            if v != expected:
                raise ValueError(
                    f"needs_refine={v} 与 quality_score={data['quality_score']} "
                    f"不一致（应改为 {expected}）"
                )
        return v


# ── ValidatorNode 工厂函数 ────────────────────────────────────────────

def create_review_extractor(
    llm: BaseChatModel,
    max_retries: int = 3,
) -> Runnable:
    """创建 ReviewOutput 的 ValidatorNode。

    验证失败时自动重试，将错误信息反馈给 LLM 修正。
    """
    def _extract_with_validation(state: dict) -> dict:
        """带 Pydantic 验证的审核结果提取器。"""
        response_text = state.get("raw_review", "")
        last_error = ""

        for attempt in range(max_retries):
            try:
                # 从 LLM 回复中提取 JSON
                json_str = extract_json(response_text)
                if attempt > 0:
                    # 重试时，将验证错误注入 prompt 让 LLM 自修复
                    corrected = llm.invoke(
                        f"修正以下 JSON 使其通过验证。\n"
                        f"原始 JSON：{json_str}\n"
                        f"验证错误：{last_error}\n"
                        f"修复后输出有效的 JSON。"
                    )
                    json_str = extract_json(corrected.content)

                # Pydantic 字段级验证
                validated = ReviewOutput.model_validate_json(json_str)
                return validated.model_dump()

            except (ValueError, json.JSONDecodeError) as e:
                last_error = str(e)
                continue

        # 所有重试失败 → 返回兜底值
        return {
            "quality_score": 70.0,
            "review_comments": "审核结果解析失败",
            "needs_refine": True,
        }

    return RunnableLambda(_extract_with_validation)


# ── 集成到 ChapterReviewer ───────────────────────────────────────────

def create_reviewer_with_validation(llm: BaseChatModel) -> Runnable:
    """带 ValidatorNode 的 ChapterReviewer。"""
    reviewer = create_chapter_reviewer_agent(llm)
    extractor = create_review_extractor(llm)

    def _review_with_validation(state: dict) -> dict:
        # 第一步：LLM 原始审核
        raw_result = reviewer.invoke(state)

        # 第二步：Pydantic 验证 + 自修复重试
        validated = extractor.invoke({
            "raw_review": raw_result.get("crew_result", {}).get("review_comments", ""),
        })

        # 第三步：合并结果
        cr = raw_result.get("crew_result", {})
        return {
            "crew_result": {
                **cr,
                **validated,
            }
        }

    return RunnableLambda(_review_with_validation)
```

---

## 第四节 — 与综合文档的关联

本文档是 `LANGGRAPH_PATTERNS_REFERENCE.md` 的子文档，专注于 Writing 智能体的模式分析和代码模板。与综合文档的对应关系如下：

| 综合文档模式 | 本文档覆盖 |
|-------------|-----------|
| 5.1 Plan-and-Execute | 第二节 — 升级 `<thinking>` 为结构化 Plan，增加 Replanner 节点 |
| 5.2 ReWOO | 第二节 + 第三节 — Planner → Worker → Solver 三步替换，含完整图实现 |
| 5.3 LLMCompiler | 第二节 — Joiner 聚合器加速写评润循环 |
| 6.1 Basic Reflection | 第二节 — 角色反转技巧已在 `chapter_draft` 传递中实现 |
| 6.2 Reflexion | 第二节 + 第三节 — 缺失/多余检查 Prompt 模板 + Reflexion 节点实现 |
| 6.3 Tree of Thoughts | 第二节 + 第三节 — Expand/Score/Prune Pydantic 模型 + Send() 并行代码 |
| 8.4 Complex Data Extraction | 第二节 + 第三节 — ValidatorNode + field_validator 完整实现 |

### 综合文档中的通用改进建议

综合文档的「通用改进建议」直接适用于 Writing 智能体：

1. **中断机制**（interrupt）：当前 `interrupt_before=[]` 全部禁用，可在 Writing Crew 低分循环用尽时启用 interrupt 请求人工指导
2. **子图组合**：Writing Crew 本身已作为子图注入父图（`graph.add_node("writing_crew", build_writing_crew())`）
3. **状态持久化**：当前使用 `MemorySaver`，可通过 checkpointer 传参切换为 SQLite/PostgreSQL
4. **动态 Prompt**：v6.0 已引入 `_build_writer_dynamic_prompt`，可进一步借鉴 Self-Discover 模式
5. **Token 优化**：ReWOO 可减 30-40% LLM 调用，LLMCompiler 可并行执行降低延迟，ToT 的 beam search 已在模式中内置剪枝

---

> 本文档基于 LangGraph 官方教程 (v5.4.0+) 及 NovelFactory Writing 智能体实现编写。
> 建议配合 `writing_agents.py`、`writing_crew.py` 和综合文档 `LANGGRAPH_PATTERNS_REFERENCE.md` 阅读。
