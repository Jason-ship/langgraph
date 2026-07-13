# Review Agent — LangGraph 模式参考

> 本文档是 [LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md) 的子文档，专注 Review 智能体

---

## 目录

1. [当前架构概览](#一当前架构概览)
2. [可借鉴的 LangGraph 模式](#二可借鉴的-langgraph-模式)
3. [补充代码模板](#三补充代码模板)
4. [与综合文档的关联](#四与综合文档的关联)

---

## 一、当前架构概览

### 涉及文件

- [review_agents.py](src/novelfactory/agents/review_agents.py) - 两个独立 Review Agent 工厂函数

### 智能体列表

| 智能体 | 工厂函数 | 职责 |
|--------|----------|------|
| **KickoffReviewAgent** | `create_kickoff_review_agent()` | 开篇审核：验证世界观/角色/大纲三大产出的完整性和质量 |
| **ChapterFinalReviewAgent** | `create_chapter_final_review_agent()` | 章节终审：Writing Crew 质量门控后的合规和一致性确认 |

### 当前图结构

两个 Agent 均作为独立审核子图使用，位于 Writing Crew 质量门控之后、Sync Crew 同步之前，充当**终审把关**角色：

```
Writing Crew (quality_score >= 90 通过)
        │
        ▼
  Review Agent（当前架构）
        │
        ├── KickoffReviewAgent（开篇审核：四维评分 + 飞书通知）
        └── ChapterFinalReviewAgent（章节终审：合规检查 + 飞书通知）
        │
        ▼
  Sync Crew（飞书同步）
```

### 当前实现模式

- **模式**：Agent（简单工具使用 Agent），基于 `langgraph.prebuilt.create_react_agent`
- **工具**：绑定飞书工具（`send_feishu_message`、`send_review_request`），LLM 可自主决定通知策略
- **输出验证**：`validate_json_output` + `fail_closed` 策略（解析失败默认返回 False）
- **动态 Prompt**：`_build_kickoff_review_dynamic_prompt` / `_build_chapter_review_dynamic_prompt` 在 SystemMessage 中附加工具使用指引
- **审核标准**：
  - KickoffReview：四维量化评分（世界观30分 / 角色25分 / 大纲25分 / 自洽性20分），总分 >= 80 通过
  - ChapterFinalReview：合规检查（零容忍）+ 字数检查（>= 500 字）+ 结构完整性

### 核心代码结构

```python
# review_agents.py 工厂函数模式
def create_kickoff_review_agent(llm: BaseChatModel) -> Runnable:
    """Build the KickoffReview ReAct agent with Feishu tools."""
    tools = get_feishu_tools()
    agent = create_react_agent(llm, tools=tools,
        prompt=_build_kickoff_review_dynamic_prompt)

    def _node(state: dict) -> dict[str, Any]:
        # 1. 提取审核上下文（world_setting / character_setting / story_outline）
        # 2. 组装 review_prompt（含字数统计）
        # 3. llm_call_with_retry(agent.invoke, ...)
        # 4. validate_json_output(response_text, required_keys=[...], fail_closed=True)
        # 5. 返回 crew_result 更新

    return RunnableLambda(_node)
```

---

## 二、可借鉴的 LangGraph 模式

### 2.1 Basic Reflection（基础反思）— 文档 6.1

**问题**：当前 Review Agent 的输出是一锤子买卖，没有对审核结果本身进行复审。

**改进方案**：增加**自反思循环**，让 Review Agent 对自己的审核结果进行复审。

```
当前流程：
  Review Agent → 输出审核结果 → 结束

改进后流程：
  Review Agent → 输出审核结果
       │
       ▼
  Self-Reflect Node（评审审核意见是否准确、是否遗漏关键问题）
       │
  ┌────┴────┐
  ▼         ▼
 通过     不通过
  │         │
  │         ▼
  │    Revise Node（修正审核意见）
  │         │
  └─────────┘
       │
       ▼
    结束
```

**适用场景**：
- KickoffReview 的四维评分（世界观/角色/大纲/自洽性）可进行逐维复审
- ChapterFinalReview 的合规检查可进行二次确认

### 2.2 Agent-based Evaluation（基于模拟用户的评估）— 文档 7.1

**问题**：当前的两个 Review Agent 都是"专业评分者"视角，缺少"普通读者"视角。

**改进方案**：引入 **Simulated Reader Agent**（模拟用户），评估章节的可读性和吸引力。

```
当前视角：
  ChapterFinalReviewAgent → 合规检查（专业视角）

增加视角：
  SimulatedReaderAgent → 可读性 + 吸引力评估（读者视角）
         │
         ▼
   结果聚合：专业评分 + 读者评分 → 综合审核结论
```

**设计思路**：

| 维度 | 专业审核（现有） | 模拟读者（新增） |
|------|----------------|----------------|
| 视角 | 编辑/审校 | 普通读者 |
| 评估重点 | 合规、结构、衔接 | 趣味性、代入感、情感共鸣 |
| 输出 | review_passed + comments | engagement_score + reader_feedback |
| 触发时机 | 质量门控后 | 与终审并行或串行 |

### 2.3 Reflexion（高级反思）— 文档 6.2

**问题**：当前只有对"被审核内容"的评估，缺少对"审核自身质量"的评估。

**改进方案**：在 Basic Reflection 的基础上，增加对审核质量的深层评估。

```
审核质量评估维度：
1. 扣分点是否准确？是否有误判？
2. 是否遗漏了重要问题？
3. 审核意见是否足够具体？是否有建设性？
4. 评分是否与审核意见一致？
```

**与 Basic Reflection 的区别**：

| 维度 | Basic Reflection | Reflexion |
|------|-----------------|-----------|
| 深度 | 表层复核 | 深层批判 |
| 范围 | 仅审核结果 | 审核过程 + 审核标准 |
| 输出 | 修正版审核意见 | 审核质量报告 + 改进建议 |
| 适用 | 常规复审 | 审核质量不达标时触发 |

### 2.4 Complex Data Extraction（复杂数据提取）— 文档 8.4

**问题**：当前 `validate_json_output` 是通用的 JSON 验证函数，`_bind_validator_with_retries` 模式可进一步通用化。

**改进方案**：将 `_bind_validator_with_retries` 模式封装为通用工具函数，支持：
- 字段级 Pydantic 验证（不仅仅是 `required_keys` 检查）
- 可配置的最大重试次数（当前默认 3 次）
- 自定义 fallback 策略

```python
# 通用化思路
def _bind_validator_with_retries(
    llm_call: Callable,
    validator: Callable[[str], tuple[dict | None, str]],
    max_retries: int = 3,
    fallback: dict | None = None,
) -> Callable:
    """通用验证器 + 重试机制。"""
```

---

## 三、补充代码模板

### 3.1 Simulated User Agent — Prompt 模板和状态定义

```python
"""Simulated Reader Agent — 模拟读者评估章节可读性和吸引力。"""

from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

# ── 状态定义 ─────────────────────────────────────────────────────────────

class SimulatedReaderState(TypedDict):
    """模拟读者评估状态。"""
    chapter_text: str                              # 章节正文
    current_chapter: int                           # 当前章节号
    genre: str                                     # 小说类型（玄幻/都市/科幻...）
    reader_experience: str | None                  # 读者阅读体验描述
    engagement_score: float | None                 # 吸引力评分（0-100）
    reader_feedback: str | None                    # 读者反馈意见
    review_passed: bool                            # 是否通过
    review_comments: str                           # 综合审核意见


# ── System Prompt 模板 ──────────────────────────────────────────────────

SIMULATED_READER_PROMPT = """\
你是一个{genre}小说的忠实读者，刚刚读完第{current_chapter}章。

## 你的读者画像
- 阅读偏好：{genre}网文，每天阅读 2-3 小时
- 阅读平台：起点中文网 / 番茄小说
- 对章节质量的期望：有代入感、节奏紧凑、人物鲜活

## 评估任务
请基于普通读者的视角评估本章的可读性和吸引力。

## 评估维度

### 1. 阅读体验（0-40分）
- 开篇是否抓人？是否想继续读下去？
- 阅读过程中是否有"停不下来"的感觉？
- 文字是否流畅？是否需要反复回看？

### 2. 代入感（0-30分）
- 是否能代入主角视角和情感？
- 场景描写是否让你"身临其境"？
- 对话是否自然？是否符合角色性格？

### 3. 情感共鸣（0-30分）
- 是否有让你产生情绪波动的场景？
- 是否能理解角色的选择和动机？
- 悬念或冲突是否让你产生期待？

## 输出格式
```json
{{
  "reader_experience": "<详细的阅读体验描述，至少50字>",
  "engagement_score": <0-100的吸引力评分>,
  "reader_feedback": "<给作者的具体建议，至少30字>",
  "review_passed": <true/false，engagement_score>=60为true>,
  "review_comments": "<综合读者视角的审核意见>"
}}
```

## 注意事项
- 诚实反馈：不要为了"通过"而给出虚高评分
- 具体建议：指出"哪里好"、"哪里不好"、"如何改进"
- 不要使用专业术语，用读者语言表达
"""


# ── 节点函数 ─────────────────────────────────────────────────────────────

def simulated_reader_node(state: SimulatedReaderState, llm) -> dict:
    """模拟读者评估节点。"""
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = SIMULATED_READER_PROMPT.format(
        genre=state.get("genre", "网络小说"),
        current_chapter=state.get("current_chapter", 1),
    )
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["chapter_text"][:5000]),
    ]
    result = llm.invoke(messages)
    # 解析 JSON 输出...
    return {
        "engagement_score": parsed_score,
        "reader_feedback": parsed_feedback,
        "review_passed": parsed_passed,
        "review_comments": parsed_comments,
    }


# ── 与现有四维评分体系集成 ──────────────────────────────────────────────

class CompositeReviewState(TypedDict):
    """综合审核状态（专业审核 + 模拟读者）。"""
    # 专业审核输出
    quality_score: float | None                    # 四维质量评分
    review_comments_pro: str | None                # 专业审核意见
    # 模拟读者输出
    engagement_score: float | None                 # 读者吸引力评分
    review_comments_reader: str | None             # 读者反馈
    # 综合结果
    review_passed: bool                            # 综合判定
    final_comments: str                            # 最终审核意见
    review_weight_pro: float                       # 专业权重（默认 0.6）
    review_weight_reader: float                    # 读者权重（默认 0.4）
```

### 3.2 自反思循环 — 图结构代码

```python
"""自反思审核循环 — Self-Reflect → Revise → Re-Review。"""

from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ── 状态定义 ─────────────────────────────────────────────────────────────

class SelfReflectionState(TypedDict):
    """自反思审核状态。"""
    # 输入
    content_to_review: str                          # 待审核内容
    initial_review: str | None                      # 初始审核意见
    # 反思结果
    reflection_result: str | None                   # 反思结论
    reflection_score: int | None                    # 反思评分（0-100）
    needs_revision: bool | None                     # 是否需要修正
    revision_instructions: str | None               # 修正指引
    # 修订结果
    revised_review: str | None                      # 修订后审核意见
    revised_review_passed: bool | None              # 修订后是否通过
    # 控制
    reflection_attempts: int                        # 反思尝试次数
    max_reflection_attempts: int                    # 最大反思次数
    reflection_complete: bool                       # 反思是否完成


# ── 节点函数 ─────────────────────────────────────────────────────────────

SELF_REFLECT_PROMPT = """\
你是一名审核质量评审专家，请评价你对以下内容的审核意见。

## 被审核内容
{content_summary}

## 你的审核意见
{review_text}

## 评价维度
1. **准确性**：扣分点是否准确？有无误判？
2. **完整性**：是否遗漏了重要问题？
3. **具体性**：审核意见是否具体？是否有建设性？
4. **一致性**：评分与审核意见是否一致？

## 输出
```json
{
  "reflection_score": <0-100>,
  "needs_revision": <true/false, <=70 为 true>,
  "reflection_result": "<详细的反思结论>",
  "revision_instructions": "<如需修正，请给出具体修改指引>"
}
```
"""


REVISE_REVIEW_PROMPT = """\
请根据以下反思结论修正你的审核意见。

## 原始审核意见
{original_review}

## 反思结论
{reflection_result}

## 修正指引
{revision_instructions}

## 输出
```json
{
  "review_passed": <true/false>,
  "review_comments": "<修正后的审核意见>"
}
```
"""


def self_reflect_node(state: SelfReflectionState, llm) -> dict:
    """自反思节点：评估审核意见质量。"""
    # 调用 LLM 反思
    messages = [
        SystemMessage(content=SELF_REFLECT_PROMPT.format(
            content_summary=state["content_to_review"][:2000],
            review_text=state["initial_review"],
        )),
    ]
    result = llm.invoke(messages)
    # 解析结果...
    return {
        "reflection_score": parsed_score,
        "needs_revision": parsed_needs_revision,
        "reflection_result": parsed_reflection,
        "revision_instructions": parsed_instructions,
        "reflection_attempts": state.get("reflection_attempts", 0) + 1,
    }


def revise_node(state: SelfReflectionState, llm) -> dict:
    """修正节点：根据反思结论修正审核意见。"""
    # 调用 LLM 修正
    messages = [
        SystemMessage(content=REVISE_REVIEW_PROMPT.format(
            original_review=state["initial_review"],
            reflection_result=state["reflection_result"],
            revision_instructions=state["revision_instructions"],
        )),
    ]
    result = llm.invoke(messages)
    # 解析结果...
    return {
        "revised_review": revised_comments,
        "revised_review_passed": revised_passed,
    }


def re_review_node(state: SelfReflectionState, llm) -> dict:
    """再审核节点：对修订后的审核意见进行二次审核。"""
    # 重新调用自反思逻辑
    return self_reflect_node(
        {"content_to_review": state["content_to_review"],
         "initial_review": state["revised_review"]},
        llm,
    )


def route_after_reflection(state: SelfReflectionState) -> Literal["revise", "re_review", "__end__"]:
    """条件路由：根据反思结果决定下一步。"""
    if state.get("reflection_complete"):
        return "__end__"
    if state.get("needs_revision") and state["reflection_attempts"] <= state.get("max_reflection_attempts", 2):
        if state.get("revised_review"):
            return "re_review"  # 已有修订版，再审核
        return "revise"         # 需要修正
    return "__end__"


# ── 图构建 ────────────────────────────────────────────────────────────────

def build_self_reflection_graph(llm) -> StateGraph:
    """构建自反思审核循环图。"""
    builder = StateGraph(SelfReflectionState)

    # 添加节点
    builder.add_node("self_reflect", lambda s: self_reflect_node(s, llm))
    builder.add_node("revise", lambda s: revise_node(s, llm))
    builder.add_node("re_review", lambda s: re_review_node(s, llm))

    # 添加边
    builder.add_edge(START, "self_reflect")
    builder.add_conditional_edges(
        "self_reflect",
        route_after_reflection,
        {"revise": "revise", "re_review": "re_review", "__end__": END},
    )
    builder.add_conditional_edges(
        "revise",
        route_after_reflection,
        {"revise": "revise", "re_review": "re_review", "__end__": END},
    )
    builder.add_conditional_edges(
        "re_review",
        route_after_reflection,
        {"revise": "revise", "re_review": "re_review", "__end__": END},
    )

    return builder


# ── 使用示例 ────────────────────────────────────────────────────────────────

# 在现有 Review Agent 节点中集成自反思循环
def review_with_self_reflection(state: dict, llm) -> dict:
    """带自反思的审核节点。"""
    # Step 1: 执行初始审核
    initial_result = review_agent.invoke(state)
    initial_review = extract_ai_message_text(initial_result)

    # Step 2: 自反思循环
    reflection_graph = build_self_reflection_graph(llm).compile()
    reflection_result = reflection_graph.invoke({
        "content_to_review": state.get("content", ""),
        "initial_review": initial_review,
        "reflection_attempts": 0,
        "max_reflection_attempts": 2,
        "reflection_complete": False,
    })

    # Step 3: 返回最终审核结果
    if reflection_result.get("revised_review"):
        return {
            "review_passed": reflection_result["revised_review_passed"],
            "review_comments": reflection_result["revised_review"],
        }
    return {
        "review_passed": parsed_initial_passed,
        "review_comments": initial_review,
    }
```

### 3.3 与现有四维评分体系的集成方式

```python
"""将自反思循环集成到现有的 KickoffReview 四维评分体系中。"""

# 集成方案一：审核后反思
# 在现有 KickoffReview 的 _node 函数末尾增加自反思

def kickoff_review_with_reflection(state: dict, llm) -> dict:
    """带自反思的 KickoffReview 节点。"""
    # 1. 执行现有审核逻辑（四维评分）
    existing_cr = state.get("crew_result", {})
    ctx = _get_context(state)

    review_prompt = (
        f"请审核以下 Setup 阶段产出：\n"
        f"【世界观设定】{len(ctx['world_setting'])}字\n"
        f"【角色设定】{len(ctx['character_setting'])}字\n"
        f"【故事大纲】{len(ctx['story_outline'])}字\n"
        f"参考审核标准：≥80分为通过..."
    )
    result = llm_call_with_retry(agent.invoke, {"messages": [("user", review_prompt)]})
    response_text = extract_ai_message_text(result)
    parsed, _ = validate_json_output(response_text, required_keys=["review_passed", "review_comments"])

    # 2. 只有在审核未通过时，才触发自反思循环
    #    防止对明显通过的审核做不必要的反思
    if parsed and not parsed["review_passed"]:
        reflection_result = run_self_reflection_cycle(
            content=review_prompt,
            initial_review=parsed["review_comments"],
            llm=llm,
        )
        # 如果反思发现误判（例如扣分点不准确），修正审核结论
        if reflection_result["revised_review_passed"]:
            return {
                "crew_result": {
                    **existing_cr,
                    "review_passed": True,
                    "review_comments": (
                        f"[反思修正] 初始审核误判，经反思后修正。\n"
                        f"修正说明：{reflection_result.get('reflection_result', '')}\n"
                        f"修正意见：{reflection_result['revised_review']}"
                    ),
                }
            }

    # 3. 正常返回
    return {
        "crew_result": {
            **existing_cr,
            "review_passed": parsed["review_passed"] if parsed else False,
            "review_comments": parsed["review_comments"] if parsed else "审核失败",
        }
    }


# 集成方案二：模拟读者与专业审核并行评估
# 在 ChapterFinalReview 中增加并行模拟读者评估

def chapter_review_with_simulated_reader(state: dict, llm_pro: BaseChatModel, llm_reader: BaseChatModel) -> dict:
    """章节终审 + 模拟读者并行评估。"""
    from langgraph.types import Send
    from langgraph.graph import StateGraph, START, END

    # 定义并行分发
    chapter_text = state.get("refined_chapter") or state.get("chapter_draft", "")
    current_ch = state.get("current_chapter_number", 1)

    def pro_review(state):
        """专业审核节点。"""
        # ... 现有 ChapterFinalReview 逻辑 ...
        return {"pro_result": ...}

    def reader_review(state):
        """模拟读者节点。"""
        # ... 使用 SIMULATED_READER_PROMPT ...
        return {"reader_result": ...}

    def aggregate_reviews(state):
        """聚合两个审核结果。"""
        pro = state["pro_result"]
        reader = state["reader_result"]

        # 综合判定策略
        # - 专业审核 FAIL → 最终 FAIL（合规优先）
        # - 专业审核 PASS + 读者审核 PASS → PASS
        # - 专业审核 PASS + 读者审核 FAIL → PASS with warning（记录读者反馈）
        if not pro.get("review_passed"):
            return {
                "review_passed": False,
                "review_comments": f"[专业审核] {pro['review_comments']}\n[读者反馈] {reader.get('reader_feedback', '')}",
            }
        if reader.get("review_passed", True):
            return {
                "review_passed": True,
                "review_comments": f"[专业审核] {pro['review_comments']}\n[读者评分] {reader.get('engagement_score', 'N/A')}/100\n[读者反馈] {reader.get('reader_feedback', '')}",
            }
        return {
            "review_passed": True,
            "review_comments": f"[PASS with warning] 专业审核通过，但模拟读者反馈需关注。\n[读者评分] {reader.get('engagement_score', 'N/A')}/100\n[读者反馈] {reader.get('reader_feedback', '')}",
        }

    # 构建并执行并行图
    builder = StateGraph(dict)
    builder.add_node("pro_review", pro_review)
    builder.add_node("reader_review", reader_review)
    builder.add_node("aggregate", aggregate_reviews)
    builder.add_edge(START, ["pro_review", "reader_review"])
    builder.add_edge(["pro_review", "reader_review"], "aggregate")
    builder.add_edge("aggregate", END)

    graph = builder.compile()
    return graph.invoke(state)
```

---

## 四、与综合文档的关联

### 4.1 文档层级

```
LANGGRAPH_PATTERNS_REFERENCE.md（综合文档）
  ├── 1. Setup 智能体
  ├── 2. Writing 智能体
  ├── 3. Review 智能体          ← 本文档的父级章节
  ├── 4. Sync 智能体
  ├── 5. Media 智能体
  └── 通用改进建议
```

### 4.2 对应关系

| 本文档章节 | 综合文档章节 | 说明 |
|-----------|-------------|------|
| 一、当前架构概览 | 3. Review 智能体 | 本文档详细展开当前架构的具体实现 |
| 二、2.1 Basic Reflection | 6.1 Basic Reflection | 自反思循环的具体图结构设计 |
| 二、2.2 Agent-based Evaluation | 7.1 Agent-based Evaluation | 模拟用户评估的完整 Prompt 和状态定义 |
| 二、2.3 Reflexion | 6.2 Reflexion | 审核质量评估的深层机制 |
| 二、2.4 Complex Data Extraction | 8.4 Complex Data Extraction | `_bind_validator_with_retries` 通用化方案 |
| 三、补充代码模板 | 综合文档关键 API 参考 | 将 API 参考扩展为可直接使用的代码模板 |

### 4.3 实施优先级建议

| 优先级 | 模式 | 预估工时 | 收益 | 推荐实施阶段 |
|--------|------|---------|------|------------|
| P0 | Basic Reflection（自反思循环） | 2-3天 | 降低误判率，提升审核可靠性 | v6.1 |
| P1 | Complex Data Extraction 通用化 | 1天 | 减少重复代码，统一验证策略 | v6.1 |
| P2 | Agent-based Evaluation（模拟读者） | 3-5天 | 新增读者视角，提升章节可读性 | v6.2 |
| P3 | Reflexion（审核质量评估） | 2-3天 | 建立审核质量监控体系 | v6.3 |

---

> 本文档基于 LANGGRAPH_PATTERNS_REFERENCE.md 的 "3. Review 智能体" 章节扩展而成。
> 综合文档地址：[LANGGRAPH_PATTERNS_REFERENCE.md](LANGGRAPH_PATTERNS_REFERENCE.md)
> 源文件地址：[review_agents.py](src/novelfactory/agents/review_agents.py)
