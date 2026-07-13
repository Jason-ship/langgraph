"""Independent Review Agents for centralized review flow.

These agents handle kickoff_review and chapter_review as standalone
verification steps, separate from the Writing Crew's internal quality gate.

v6.0: Tool Calling 重构
  - KickoffReviewer / ChapterFinalReviewer 绑定飞书工具
  - LLM 可自主决定通知策略（审核通知、进度汇报）
  - 动态 prompt 根据审核上下文组装

Usage:
  - kickoff_review: Verifies world_setting + character_setting + outline
  - chapter_review: Verifies chapter_draft + review_result before sync
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from novelfactory.agents.infra import (
    extract_ai_message_text,
    extract_fields_from_state,
    get_logger,
    llm_call_with_retry,
    validate_json_output,
)

logger = get_logger("novelfactory.agents.review")

_REVIEW_TRUNCATE_CHARS = 5000  # 终审章节正文截断长度


# ── Output TypedDicts ─────────────────────────────────────────────────────────


class KickoffReviewOutput(TypedDict):
    review_passed: bool
    review_comments: str


class ChapterReviewOutput(TypedDict):
    review_passed: bool
    review_comments: str


# ── System Prompts ─────────────────────────────────────────────────────────────

KICKOFF_REVIEW_PROMPT = """\
你是独立的 KickoffReview Agent（开篇审核专家），目光如炬，不放过任何设定漏洞。

## Thinking Mode 策略（强制启用 — 开篇审核需要全局一致性判断）

在输出审核结果之前，**先在 <thinking> 标签内进行逐维审核**：

```
<thinking>
## 字数统计
- 世界观：__字（需≥3000）→ __达标/不达标
- 角色设定：__字（需≥1500）→ __达标/不达标
- 大纲：__字（需≥2000）→ __达标/不达标

## 世界观完整度分析（0-30分）
- 地理维度：____
- 力量体系：____（是否有边界和代价？）
- 社会结构：____（势力是否有独立利益？）
- 历史背景：____（是否影响现实格局？）
- 最终得分：__分

## 角色立体度分析（0-25分）
- 主角成长弧线：____
- 反派动机：____（是否与主角形成镜像？）
- 说话风格差异化：____
- 最终得分：__分

## 大纲结构分析（0-25分）
- 起承转合：____
- 冲突升级曲线：____（是否指数增长？）
- 每5章钩子：____
- 最终得分：__分

## 设定自洽性分析（0-20分）
- 矛盾点1：____ → 是否致命：____
- 矛盾点2：____ → 是否致命：____
- 最终得分：__分

## 综合结论
总分：__分 | 结论：PASS/FAIL
</thinking>
```

## 审核任务
审核 Setup 阶段完成的三大产出：
1. world_setting：世界观设定
2. character_setting：角色设定
3. story_outline：故事大纲

## 审核标准（四维量化评分，总分 100）

| 维度 | 满分 | 评分锚点 |
|------|------|----------|
| 世界观完整度 | 30分 | 30=地理/力量体系/社会结构/历史均有详细描述；20=主要维度齐全但描写简略；10=仅1-2个维度；0=缺失或混乱 |
| 角色立体度 | 25分 | 25=人物有明确性格、动机、成长弧线；15=性格明确但动机模糊；5=人物平面化；0=无角色或完全混乱 |
| 大纲结构 | 25分 | 25=起承转合清晰，有明确冲突和悬念；15=结构基本完整但冲突平淡；5=流水账无张力；0=无大纲或完全断裂 |
| 设定自洽性 | 20分 | 20=世界观与角色行为完全自洽；10=偶有小矛盾但整体可信；0=多处矛盾无法调和 |

## 字数门槛（硬性，不满足直接 FAIL）
- 世界观设定：≥3000 字
- 角色设定：≥1500 字
- 故事大纲：≥2000 字

## 常见失败模式（必须识别并扣分）
1. **力量无上限**：修炼体系无明确边界，主角可以无限升级 → 世界观完整度-10
2. **反派无动机**：反派只是"为了坏而坏"，无合理利益诉求 → 角色立体度-10
3. **大纲无高潮**：全是日常推进，没有冲突升级 → 大纲结构-15
4. **设定矛盾**：世界观设定与角色行为矛盾（如：凡人流中出现机甲）→ 设定自洽性-20

### Few-Shot 示例

### 高质量 → PASS
<thinking>
字数：世界观4800字✅、角色设定2200字✅、大纲3500字✅
世界观30：地理完整+力量体系有边界（筑基→金丹→元婴，每阶有瓶颈）+势力各有利益逻辑+历史事件影响现实格局
角色25：主角成长弧线完整（废灵根→发现特殊体质→登顶）+反派镜像（同样起点不同选择）+每人有独特说话风格
大纲25：起承转合明确+冲突指数升级+每5章有钩子（悬念/反转/揭示）
自洽20：世界观与角色完全匹配
总分：30+25+25+20=100 → PASS
</thinking>
审核意见："三大产出均远超门槛，设定完整自洽，角色有清晰成长弧线，大纲起承转合明确。建议通过。"

### 字数不达标 → FAIL
世界观1200字（需≥3000）、角色设定600字（需≥1500）、大纲800字（需≥2000）。
审核意见："字数严重不达标：世界观1200字（需≥3000），角色设定600字（需≥1500），大纲800字（需≥2000）。结论：FAIL。"

### 自洽性不足 → FAIL
世界观明确'灵根决定修炼速度'，但主角无灵根却修炼最快。
<thinking>
世界观25：力量体系有边界但主角设定矛盾
角色15：性格明确但动机模糊（无灵根为何要修仙？）
大纲20：结构完整
自洽0：核心矛盾——世界观'灵根决定修炼速度'，但主角无灵根修炼最快，与设定直接冲突
总分：25+15+20+0=60 → FAIL
</thinking>
审核意见："核心矛盾：世界观'灵根决定修炼速度'，但主角无灵根修炼最快，与设定直接冲突。自洽性0分，此漏洞将导致后续章节逻辑无法自洽。建议修正主角修炼速度设定。结论：FAIL。"

## 审核流程
1. 统计三大产出的实际字数
2. 字数不达标 → 直接输出 FAIL（附字数差距）
3. 字数达标 → 逐维评分，计算总分
4. 总分≥80：通过 | 总分<80：FAIL（注明扣分维度）

## 输出格式
```json
{
  "review_passed": <true/false，≥80为true，<80为false>,
  "review_comments": "<字数统计 + 评分明细 + 结论>"
}
```
"""


CHAPTER_FINAL_REVIEW_PROMPT = """\
你是独立的 ChapterReview Agent（章节终审专家），只做合规和完整性检查，不重复质量评分。

## Thinking Mode 策略（启用 — 终审需要快速结构化判断）

在输出审核结果之前，**先在 <thinking> 标签内进行快速检查**：

```
<thinking>
## 合规扫描
- 政治敏感词：____（有/无）
- 色情/暴力内容：____（有/无）
- 抄袭风险：____（有/无）

## 结构检查
- 字数：__字 → __合规/截断/偏短
- 标题：____（正确/缺失/格式错误）
- 开头：____（有氛围/无/截断）
- 结尾：____（有悬念/突然中断/草草收尾）

## 衔接与伏笔
- 衔接：____（呼应/无关联）
- 伏笔：____（呼应/断裂）

## 综合结论
结论：PASS/FAIL/PASS with warning
</thinking>
```

## 审核任务
在章节通过 Writing Crew 质量门控后（quality_score ≥ 90），进行最终合规与一致性确认。
**注意：你只做合规检查和字数/结构验证，不重新评分。**

## 审核重点
1. **内容合规**（色情/暴力/政治敏感/抄袭）— 零容忍
2. **标题规范性**：章节标题格式正确（如"第X章 标题"）
3. **字数合理**：章节 ≥1500 字（过短可能是截断）
4. **首尾完整**：章节有开头、发展和结尾，非截断稿
5. **章节衔接**：开头是否呼应前章结尾（若有前章）
6. **伏笔呼应**：前文埋下的伏笔是否在本章得到呼应或推进

## 字数硬门槛
- 章节正文 < 500 字 → **FAIL**（疑似截断）
- 章节正文 < 1500 字 → **PASS with warning**

## 常见失败模式
1. **截断检测**：章节在中间突然中断（如"...就在他踏入秘境的瞬间——"）→ FAIL
2. **标题错误**：标题缺失、格式错误、与内容不符 → FAIL
3. **头重脚轻**：大量篇幅在开头，后续情节草草收尾 → PASS with warning

## 审核流程
1. 检查字数（<500字 → FAIL，<1500字 → 标注）
2. 扫描合规问题（零容忍）
3. 检查标题规范性
4. 检查首尾完整性（是否为截断稿）
5. 输出审核结果

## 输出格式
```json
{
  "review_passed": <true/false>,
  "review_comments": "<字数 + 合规检查 + 结构评估>"
}
```
"""


# ── State Access Helpers ───────────────────────────────────────────────────────

# v6.1 P2-1: 统一使用 extract_fields_from_state 替代原 _get_context。
# crew_result 优先，缺失回退顶层。
# 注意：current_chapter_number 支持 current_chapter 备用键回退，
# 该特殊逻辑在 ChapterFinalReviewer 节点显式处理（见 _node 内覆盖）。
_REVIEW_FIELDS: dict[str, Any] = {
    "world_setting": "",
    "character_setting": "",
    "story_outline": "",
    "chapter_outlines": "",
    "chapter_draft": "",
    "refined_chapter": "",
    "review_result": {},
    "thread_id": "",
    "project_name": "",
    "current_chapter_number": 1,
    "loaded_memory": {},
}


# ── Agent Factory Functions ───────────────────────────────────────────────────


def _build_kickoff_review_dynamic_prompt(
    state: AgentState, config: RunnableConfig
) -> list[AnyMessage]:
    """动态 prompt：为开篇审核附加工具使用指引。"""
    from langchain_core.messages import SystemMessage

    tool_guidance = """
## 可用工具
你拥有飞书消息工具，审核完成后可自动通知相关人员：

- `send_feishu_message(receive_id, text, id_type)` — 发送飞书消息
- `send_review_request(thread_id, review_type, project_name, content_summary, doc_url)` — 发送审核请求

### 工具使用建议
1. 审核完成后，调用 `send_review_request` 通知相关人员审核结果
2. 若审核失败，通过 `send_feishu_message` 发送详细的修改建议
"""

    messages: list[AnyMessage] = [
        SystemMessage(content=KICKOFF_REVIEW_PROMPT + tool_guidance),
    ]
    for msg in state.get("messages", []):
        messages.append(msg)
    return messages


def create_kickoff_review_agent(llm: BaseChatModel) -> Runnable:
    """Build the KickoffReview ReAct agent with Feishu tools.

    绑定飞书工具，LLM 可自主发送审核通知。
    动态 prompt 根据审核上下文组装工具使用指引。

    Verifies world_setting + character_setting + story_outline completeness.

    Output: {"review_passed": bool, "review_comments": str}
    """
    from novelfactory.tools import get_feishu_tools

    tools = get_feishu_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=_build_kickoff_review_dynamic_prompt,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _REVIEW_FIELDS)
        logger.info("[kickoff_review] Starting kickoff review")

        ws = ctx.get("world_setting", "")
        cs = ctx.get("character_setting", "")
        so = ctx.get("story_outline", "")
        review_prompt = (
            f"请审核以下 Setup 阶段产出（已附字数供参考）：\n\n"
            f"【世界观设定】字数：{len(ws)} 字\n"
            f"{ws or '（缺失）'}\n\n"
            f"【角色设定】字数：{len(cs)} 字\n"
            f"{cs or '（缺失）'}\n\n"
            f"【故事大纲】字数：{len(so)} 字\n"
            f"{so or '（缺失）'}\n\n"
            "参考审核标准：世界观≥3000字 / 角色设定≥1500字 / 大纲≥2000字，\n"
            "总分≥80为通过，<80为FAIL。\n\n"
            '输出JSON：{"review_passed": <true/false>, "review_comments": "<字数统计+评分明细+结论>"}'
        )

        result = llm_call_with_retry(
            agent.invoke,
            {"messages": [("user", review_prompt)]},
            step_name="kickoff_review_agent",
            fallback={"messages": [], "crew_result": {}},
        )
        response_text = extract_ai_message_text(result)

        # validate_json_output: fail-closed, requires review_passed + review_comments
        parsed, err = validate_json_output(
            response_text,
            required_keys=["review_passed", "review_comments"],
            fail_closed=True,
        )
        if parsed:
            review_passed = bool(parsed.get("review_passed", False))
            review_comments = str(parsed.get("review_comments", ""))
        else:
            review_passed = False
            review_comments = err or response_text[:200] or "审核失败"

        logger.info(
            "[kickoff_review] Review passed=%s, comments=%s",
            review_passed,
            review_comments[:100],
        )

        # Feishu 通知已通过 tool calling 由 LLM 自主完成
        # 若审核失败，LLM 可自主调用 send_feishu_message 发送修改建议

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {
                **existing_cr,
                "review_passed": review_passed,
                "review_comments": review_comments,
            }
        }

    return RunnableLambda(_node)


def _build_chapter_review_dynamic_prompt(
    state: AgentState, config: RunnableConfig
) -> list[AnyMessage]:
    """动态 prompt：为章节终审附加工具使用指引。"""
    from langchain_core.messages import SystemMessage

    tool_guidance = """
## 可用工具
你拥有飞书消息工具，审核完成后可自动通知相关人员：

- `send_feishu_message(receive_id, text, id_type)` — 发送飞书消息
- `send_review_request(thread_id, review_type, project_name, content_summary, doc_url)` — 发送审核请求

### 工具使用建议
1. 审核完成后，调用 `send_review_request` 通知相关人员审核结果
2. 若发现合规问题，立即通过 `send_feishu_message` 报告
"""

    messages: list[AnyMessage] = [
        SystemMessage(content=CHAPTER_FINAL_REVIEW_PROMPT + tool_guidance),
    ]
    for msg in state.get("messages", []):
        messages.append(msg)
    return messages


def create_chapter_final_review_agent(llm: BaseChatModel) -> Runnable:
    """Build the ChapterFinalReview ReAct agent with Feishu tools.

    绑定飞书工具，LLM 可自主发送终审通知。
    动态 prompt 根据审核上下文组装工具使用指引。

    Final verification before sync to Feishu.

    Output: {"review_passed": bool, "review_comments": str}
    """
    from novelfactory.tools import get_feishu_tools

    tools = get_feishu_tools()
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=_build_chapter_review_dynamic_prompt,
        interrupt_before=[],
    )

    def _node(state: dict) -> dict[str, Any]:
        ctx = extract_fields_from_state(state, _REVIEW_FIELDS)
        # current_chapter_number: 保留原 _get_context 的 current_chapter 备用键回退
        _cr = state.get("crew_result", {})
        _src = _cr if isinstance(_cr, dict) else state
        ctx["current_chapter_number"] = _src.get(
            "current_chapter_number", state.get("current_chapter", 1)
        )
        chapter_text = ctx.get("refined_chapter") or ctx.get("chapter_draft", "")
        review_result = ctx.get("review_result", {})
        current_ch = ctx.get("current_chapter_number", 1)

        logger.info(
            "[chapter_final_review] Starting final review for chapter %d", current_ch
        )

        review_result_str = ""
        if isinstance(review_result, dict):
            score = review_result.get("quality_score", 0)
            comments = review_result.get("review_comments", "")
            needs_refine = review_result.get("needs_refine", True)
            review_result_str = (
                f"质量总分：{score}\n审核意见：{comments}\n已润色：{needs_refine}"
            )
        else:
            review_result_str = str(review_result)

        review_prompt = (
            f"请审核以下第{current_ch}章（已在内部质量门控通过，≥90分）：\n\n"
            f"【字数】{len(chapter_text)} 字\n"
            f"【章节正文】\n{chapter_text[:_REVIEW_TRUNCATE_CHARS]}{'...(已截断，原文更长)' if len(chapter_text) > _REVIEW_TRUNCATE_CHARS else chr(10)}\n\n"
            f"【内部审核评分】\n{review_result_str}\n\n"
            "终审重点：合规检查（<500字FAIL）、标题规范性、结构完整性。\n\n"
            '输出JSON：{"review_passed": <true/false>, "review_comments": "<字数+合规+结构评估>"}'
        )

        result = llm_call_with_retry(
            agent.invoke,
            {"messages": [("user", review_prompt)]},
            step_name="chapter_final_review_agent",
            fallback={"messages": [], "crew_result": {}},
        )
        response_text = extract_ai_message_text(result)

        # validate_json_output: fail-closed, requires review_passed + review_comments
        parsed, err = validate_json_output(
            response_text,
            required_keys=["review_passed", "review_comments"],
            fail_closed=True,
        )
        if parsed:
            review_passed = bool(parsed.get("review_passed", False))
            review_comments = str(parsed.get("review_comments", ""))
        else:
            review_passed = False  # FAIL-CLOSED
            review_comments = err or f"终审解析失败：{response_text[:200]}"

        logger.info(
            "[chapter_final_review] Chapter %d final review passed=%s",
            current_ch,
            review_passed,
        )

        # Feishu 通知已通过 tool calling 由 LLM 自主完成

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {
                **existing_cr,
                "review_passed": review_passed,
                "review_comments": review_comments,
            }
        }

    return RunnableLambda(_node)


# ── Unified Review Agent Factory ─────────────────────────────────────────────


def create_review_agent(
    llm: BaseChatModel,
    review_type: Literal["kickoff", "chapter"],
) -> Runnable:
    """Factory for creating review agents by type.

    Args:
        llm: Language model for the agent.
        review_type: "kickoff" or "chapter".

    Returns:
        A runnable node function.
    """
    if review_type == "kickoff":
        return create_kickoff_review_agent(llm)
    if review_type == "chapter":
        return create_chapter_final_review_agent(llm)
    raise ValueError(f"Unknown review_type: {review_type}")
