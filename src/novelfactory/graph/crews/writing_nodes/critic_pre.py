"""Critic 前置大纲评估节点（v7.3 新增）。

在 chapter_writer 写正文前评估大纲合理性。
参考 MAGNET (Pocket FM 2026) 的 Critic 前置评估设计。

流程:
    chapter_planner → critic_pre_assessment_node → chapter_writer (PASS/FLAG)
                                                   → chapter_planner (FAIL→重规划)
"""

from __future__ import annotations

from typing import Any

from novelfactory.agents.infra import get_logger, async_llm_call_with_retry
from novelfactory.evaluation.debate.parser import parse_markdown_sections
from novelfactory.evaluation.debate.prompts import CRITIC_PRE_ASSESSMENT_PROMPT

logger = get_logger(__name__)


async def critic_pre_assessment_node(state: dict[str, Any]) -> dict[str, Any]:
    """Critic 前置大纲评估节点（async）。

    Args:
        state: writing_crew 状态，需含 chapter_plan, volume_info, story_outline 等

    Returns:
        dict 含 critic_assessment（PASS/FLAG/FAIL）+ critic_feedback
    """
    chapter_plan = state.get("chapter_plan", {})
    if not chapter_plan:
        logger.info("[Critic前置] 无章节大纲，跳过评估")
        return {"critic_assessment": "PASS", "critic_feedback": "（无大纲，跳过）"}

    # 提取需要的信息
    story_outline = state.get("story_outline", "")
    volume_info = str(state.get("volume_info", {}))
    prev_summary = str(state.get("writer_context", ""))[:2000]

    # 格式化大纲
    chapter_outline = _format_outline(chapter_plan)
    if not chapter_outline:
        return {"critic_assessment": "PASS", "critic_feedback": "（大纲为空，跳过）"}

    prompt = CRITIC_PRE_ASSESSMENT_PROMPT.format(
        story_outline=story_outline[:3000] if story_outline else "（无）",
        chapter_outline=chapter_outline,
        volume_info=volume_info[:1000] if volume_info else "（无）",
        prev_summary=prev_summary if prev_summary else "（无）",
    )

    try:
        from novelfactory.config.llm import get_reviewer_llm

        llm = get_reviewer_llm()
        response = await async_llm_call_with_retry(llm, prompt, step_name="critic_pre_assessment")
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_markdown_sections(raw)

        # 提取综合判定
        final_verdict = _extract_verdict(parsed.get("综合判定", ""), parsed)
        feedback = _build_feedback(parsed, chapter_plan)

        logger.info(
            "[Critic前置] 评估结果=%s 大纲=%s",
            final_verdict,
            chapter_plan.get("chapter_title", "?"),
        )

        return {
            "critic_assessment": final_verdict,
            "critic_feedback": feedback,
        }

    except Exception as e:
        logger.warning("[Critic前置] 评估失败: %s（降级为PASS）", e)
        return {"critic_assessment": "PASS", "critic_feedback": f"（评估失败: {e}）"}


def _format_outline(plan: dict) -> str:
    """将 chapter_plan 格式化为可读文本。"""
    parts = []
    title = plan.get("chapter_title", "")
    if title:
        parts.append(f"章节标题: {title}")

    goal = plan.get("writing_goal", "") or plan.get("goal", "")
    if goal:
        parts.append(f"写作目标: {goal}")

    key_beats = plan.get("key_beats", []) or plan.get("beats", [])
    if key_beats:
        parts.append("关键情节节点:")
        for i, beat in enumerate(key_beats, 1):
            parts.append(f"  {i}. {beat}")

    characters = plan.get("characters_involved", []) or plan.get(
        "target_characters", []
    )
    if characters:
        parts.append(f"涉及角色: {', '.join(characters)}")

    # 伏笔
    f_plant = plan.get("foreshadowing_plant", [])
    if f_plant:
        parts.append(f"埋下伏笔: {', '.join(f_plant)}")
    f_resolve = plan.get("foreshadowing_resolve", [])
    if f_resolve:
        parts.append(f"回收伏笔: {', '.join(f_resolve)}")

    return "\n".join(parts) if parts else str(plan)


def _extract_verdict(verdict_section: str, parsed: dict) -> str:
    """从综合判定中提取 PASS/FLAG/FAIL。"""
    if not verdict_section:
        # 从各维度判定中推断
        verdicts = []
        for key in (
            "角色行为一致性",
            "情节合理性",
            "世界观一致",
            "伏笔衔接",
            "难度递进",
        ):
            v = parsed.get(key, "")
            if v.startswith("FAIL"):
                return "FAIL"
            if v.startswith("FLAG"):
                verdicts.append("FLAG")
        return (
            "FAIL" if verdicts.count("FAIL") > 2 else ("FLAG" if verdicts else "PASS")
        )

    text = verdict_section.strip().upper()
    if text.startswith("FAIL"):
        return "FAIL"
    if text.startswith("FLAG"):
        return "FLAG"
    return "PASS"


def _build_feedback(parsed: dict, plan: dict) -> str:
    """构建反馈信息。"""
    parts = []
    for key in ("角色行为一致性", "情节合理性", "世界观一致", "伏笔衔接", "难度递进"):
        v = parsed.get(key, "")
        if v:
            parts.append(f"{key}: {v[:200]}")
    verdict = parsed.get("综合判定", "")
    if verdict:
        parts.append(f"综合判定: {verdict[:200]}")

    return "\n".join(parts) if parts else "（无反馈）"
