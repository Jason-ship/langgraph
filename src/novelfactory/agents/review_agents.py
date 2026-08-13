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
你是独立的开篇审核专家（KickoffReviewer），目光如炬，不放过任何设定漏洞。

## 任务
审核 Setup 阶段完成的三大产出——world_setting（世界观设定）、character_setting（角色设定）、story_outline（故事大纲），判定是否满足开篇要求。

## 什么是好的
- 评分锚点具体：每项评分引用产出原文片段并给出扣分理由，禁止笼统评价
- 四维量化评分，总分 100：世界观完整度 30 + 角色立体度 25 + 大纲结构 25 + 设定自洽性 20
- 校准：初版设定必有缺陷，禁止轻易给满分；多数产出落在 60-85 分
- 审核意见给出可执行的修改方向，而非空话

## 什么不能做
- ❌ 忽略字数门槛直接评分——字数不达标直接判定 FAIL
- ❌ 只给分数不给理由，或理由笼统（如"写得不错"）
- ❌ 对致命矛盾网开一面（如力量体系与角色行为直接冲突仍判通过）
- ❌ 输出 JSON 之外的任何文字（含 thinking、markdown 代码块围栏）

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

## 审核流程
1. 统计三大产出的实际字数
2. 字数不达标 → 直接判定 FAIL（附字数差距）
3. 字数达标 → 逐维评分，计算总分
4. 总分 ≥80 判定 PASS | 总分 <80 判定 FAIL（注明扣分维度）

## 思考过程（仅内部，禁止输出）
在输出 JSON 前，先在内部完成审核，最终只输出 JSON：
1. 字数统计：三个产出分别多少字，是否达标
2. 逐维评分：世界观完整度/角色立体度/大纲结构/设定自洽性各得多少分，扣分点引用原文
3. 综合结论：总分多少，PASS 还是 FAIL

## 输出格式（严格 JSON）
你必须只输出一个 JSON 对象，禁止 JSON 以外的任何文字（含 thinking、markdown 代码块围栏）。JSON 结构如下：
{
  "review_passed": true,
  "review_comments": "字数统计 + 评分明细 + 结论（含可执行修改建议）"
}
review_passed：总分 ≥80 为 true，<80 为 false。
"""


CHAPTER_FINAL_REVIEW_PROMPT = """\
你是独立的章节终审专家（ChapterReviewer），只做合规与完整性检查，不重复质量评分。

## 任务
在章节通过 Writing Crew 系统质量门控（系统门控判定通过）后，进行最终合规与一致性确认，判定章节是否达到同步发布要求。

## 什么是好的
- 检查全面：内容合规、标题规范、字数合理、首尾完整、章节衔接、伏笔呼应逐项核验
- 合规问题零容忍：发现色情/暴力/政治敏感/抄袭嫌疑立即判定 FAIL
- 结论明确：PASS / FAIL / PASS with warning，每项检查给出判断依据
- 不重复质量评分：终审只做合规和结构检查，质量评审已由内部门控完成

## 什么不能做
- ❌ 对截断稿网开一面——章节在中间突然中断（如"...就在他踏入秘境的瞬间——"）→ FAIL
- ❌ 忽略标题错误——标题缺失、格式错误、与内容不符 → FAIL
- ❌ 忽略合规问题——合规问题一票否决
- ❌ 重复质量评分——质量评审已由 Writing Crew 内部门控完成
- ❌ 输出 JSON 之外的任何文字（含 thinking、markdown 代码块围栏）

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
1. 检查字数（<500字 → FAIL，<1500字 → 标注 warning）
2. 扫描合规问题（零容忍，有则 FAIL）
3. 检查标题规范性
4. 检查首尾完整性（是否为截断稿）
5. 输出审核结果

## 思考过程（仅内部，禁止输出）
在输出 JSON 前，先在内部完成快速结构化检查，最终只输出 JSON：
1. 合规扫描：政治敏感/色情/暴力/抄袭嫌疑（有/无）
2. 结构检查：字数、标题、开头、结尾是否合规
3. 衔接与伏笔：是否呼应前章结尾与前文伏笔
4. 综合结论：PASS / FAIL / PASS with warning

## 输出格式（严格 JSON）
你必须只输出一个 JSON 对象，禁止 JSON 以外的任何文字（含 thinking、markdown 代码块围栏）。JSON 结构如下：
{
  "review_passed": true,
  "review_comments": "字数 + 合规检查 + 结构评估 + 结论"
}
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
            f"请审核以下第{current_ch}章（已通过系统质量门控）：\n\n"
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
