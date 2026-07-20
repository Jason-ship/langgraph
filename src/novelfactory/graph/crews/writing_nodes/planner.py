"""Chapter planner node — plan before writing.

Creates a structured ChapterPlan that the chapter_writer executes.
On REWRITE path, planner receives review feedback and produces a revised plan.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from novelfactory.agents.infra import get_crew_stream, get_logger
from novelfactory.agents.writing_agents import create_chapter_planner_agent
from novelfactory.config.llm import get_worker_llm

logger = get_logger(__name__)


async def chapter_planner_node(state: dict) -> dict[str, Any]:
    """Plan the next chapter before writing (async).

    Invokes the planner agent with:
      - Story outline, character setting, previous chapter summary
      - Cross-chapter context from ContextBuilder
      - Review feedback (if this is a rewrite)

    Returns structured ChapterPlan for the writer to execute.
    """
    cr = state.get("crew_result", {})
    current_ch = state.get("current_chapter", cr.get("current_chapter_number", 1))
    prefix = f"chapter_{current_ch}"
    sw = get_crew_stream("writing", prefix)
    loop_count = state.get("loop_count", 0)

    rewrite_suffix = f"（第{loop_count + 1}次重写）" if loop_count > 0 else ""
    if sw:
        sw.write(f"\n[chapter_planner] 规划第{current_ch}章{rewrite_suffix}...\n")

    review_result = cr.get("review_result", {}) or state.get("review_result", {})
    critic_feedback = review_result.get("feedback", {}).get("summary", "") or cr.get(
        "critic_feedback", ""
    )

    planner_input = {
        "crew_result": cr,
        "writer_context": state.get("writer_context", ""),
        "current_chapter": current_ch,
        "loop_count": loop_count,
        "critic_feedback": critic_feedback,
    }

    planner_agent = create_chapter_planner_agent(get_worker_llm())
    result = await planner_agent.ainvoke(planner_input)
    chapter_plan = result.get("chapter_plan", {})

    if chapter_plan:
        scenes = chapter_plan.get("scenes", [])
        if sw:
            sw.write(
                f"[chapter_planner] 计划：{chapter_plan.get('title', '')} | "
                f"{len(scenes)}个场景 | {chapter_plan.get('target_word_count', 0)}字\n"
            )
            for s in scenes:
                sw.write(
                    f"  [{s.get('scene_number', '?')}] {s.get('purpose', '')[:60]}\n"
                )

        logger.info(
            "Chapter %d plan: %s | %d scenes | %d chars",
            current_ch,
            chapter_plan.get("title", ""),
            len(scenes),
            chapter_plan.get("target_word_count", 0),
        )
    else:
        if sw:
            sw.write("[chapter_planner] ⚠ 规划失败，writer 将直接创作\n")

    return {
        "chapter_plan": chapter_plan,
        # On rewrite, pass loop_count through so writer sees correct count
        "loop_count": loop_count,
        # Chat UI message
        "messages": [
            AIMessage(
                content=f"第{current_ch}章规划完成{'（第' + str(loop_count + 1) + '次重写）' if loop_count > 0 else ''}：{chapter_plan.get('title', '')}",
                name="chapter_planner",
            )
        ],
    }
