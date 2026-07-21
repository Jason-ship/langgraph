"""Bridge — connects Lead Agent (conversational) to batch processing pipeline.

This is the key fusion component that allows the conversational Lead Agent
to delegate to the existing batch processing pipeline (Writing Crew, Setup Crew)
when the user requests full automatic novel generation.

State Mapping:
    LeadAgentState → NovelFactoryState (for batch invocation)
    NovelFactoryState → LeadAgentState (for conversational continuation)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  LeadAgentState → NovelFactoryState
# ═══════════════════════════════════════════════════════════════════════════════


def map_lead_to_batch(lead_state: dict[str, Any]) -> dict[str, Any]:
    """Map LeadAgentState fields to NovelFactoryState for batch pipeline invocation.

    The batch pipeline (Writing Crew, Setup Crew) expects NovelFactoryState fields.
    This function maps overlapping fields and sets appropriate defaults.

    Args:
        lead_state: Current LeadAgentState as a plain dict.

    Returns:
        NovelFactoryState-compatible dict ready for batch graph invocation.
    """
    batch_input: dict[str, Any] = {}

    # ── Identity ──
    if lead_state.get("thread_id"):
        batch_input["thread_id"] = lead_state["thread_id"]
    if lead_state.get("project_name"):
        batch_input["project_name"] = lead_state["project_name"]

    # ── User Input ──
    if lead_state.get("genre"):
        batch_input["genre"] = lead_state["genre"]
    if lead_state.get("target_chapters"):
        batch_input["target_chapters"] = lead_state["target_chapters"]
    if lead_state.get("current_chapter"):
        batch_input["current_chapter"] = lead_state["current_chapter"]

    # ── Setup Phase Outputs ──
    if lead_state.get("world_setting"):
        batch_input["world_setting"] = lead_state["world_setting"]
    if lead_state.get("character_setting"):
        batch_input["character_setting"] = lead_state["character_setting"]
    if lead_state.get("story_outline"):
        batch_input["story_outline"] = lead_state["story_outline"]
    if lead_state.get("chapter_outlines"):
        batch_input["chapter_outlines"] = lead_state["chapter_outlines"]
    if lead_state.get("setup_complete"):
        batch_input["setup_complete"] = lead_state["setup_complete"]
    else:
        # If not complete, the batch pipeline will enter setup phase
        batch_input["setup_complete"] = False

    # ── Crew Result ──
    crew_result: dict[str, Any] = {}
    if lead_state.get("chapter_draft"):
        crew_result["chapter_draft"] = lead_state["chapter_draft"]
    if lead_state.get("quality_score"):
        crew_result["quality_score"] = lead_state["quality_score"]
    crew_result["genre"] = lead_state.get("genre", "")
    if lead_state.get("world_setting"):
        crew_result["world_setting"] = lead_state["world_setting"]
    if lead_state.get("character_setting"):
        crew_result["character_setting"] = lead_state["character_setting"]
    if lead_state.get("story_outline"):
        crew_result["story_outline"] = lead_state["story_outline"]
    if lead_state.get("chapter_outlines"):
        crew_result["chapter_outlines"] = lead_state["chapter_outlines"]
    batch_input["crew_result"] = crew_result

    # ── Phase Control ──
    # Determine which phase to start from
    if lead_state.get("setup_complete"):
        # Setup is done, start writing
        batch_input["current_phase"] = "writing"
        batch_input["pending_review"] = ""
    else:
        # Setup not done, start from setup
        batch_input["current_phase"] = "setup"

    # ── Seed Idea (from conversation summary or project name) ──
    seed = lead_state.get("seed_idea", "") or lead_state.get("project_name", "")
    if not seed and lead_state.get("genre"):
        seed = f"创作一部{lead_state['genre']}题材的小说"
    if not seed and lead_state.get("world_setting"):
        seed = lead_state["world_setting"][:200]
    if not seed and lead_state.get("context_summary"):
        seed = lead_state["context_summary"][:200]
    batch_input["seed_idea"] = seed

    # ── Conversation-derived context (mapped for batch pipeline context) ──
    if lead_state.get("context_summary"):
        batch_input["intent_analysis"] = lead_state["context_summary"]

    logger.info(
        "[Bridge] Mapped LeadAgentState → NovelFactoryState: "
        "genre=%s, setup_complete=%s, phase=%s, chapters=%s",
        batch_input.get("genre"),
        batch_input.get("setup_complete"),
        batch_input.get("current_phase"),
        batch_input.get("target_chapters"),
    )

    return batch_input


# ═══════════════════════════════════════════════════════════════════════════════
#  NovelFactoryState → LeadAgentState
# ═══════════════════════════════════════════════════════════════════════════════


def map_batch_to_lead(batch_state: dict[str, Any]) -> dict[str, Any]:
    """Map NovelFactoryState results back to LeadAgentState for conversational continuation.

    After the batch pipeline completes, this function extracts relevant fields
    for the Lead Agent to continue the conversation.

    Args:
        batch_state: Resulting NovelFactoryState from batch pipeline.

    Returns:
        LeadAgentState-compatible dict with mapped fields + status message.
    """
    lead_updates: dict[str, Any] = {}

    # ── Progress ──
    if batch_state.get("current_chapter"):
        lead_updates["current_chapter"] = batch_state["current_chapter"]
    if batch_state.get("quality_score"):
        lead_updates["quality_score"] = batch_state["quality_score"]

    # ── Latest Chapter Content ──
    # Prefer refined_chapter over raw draft for better quality
    if batch_state.get("refined_chapter"):
        lead_updates["chapter_draft"] = batch_state["refined_chapter"]
    elif batch_state.get("chapter_draft"):
        lead_updates["chapter_draft"] = batch_state["chapter_draft"]

    # ── Generated World Building (if setup was completed by batch pipeline) ──
    if batch_state.get("world_setting"):
        lead_updates["world_setting"] = batch_state["world_setting"]
    if batch_state.get("character_setting"):
        lead_updates["character_setting"] = batch_state["character_setting"]
    if batch_state.get("story_outline"):
        lead_updates["story_outline"] = batch_state["story_outline"]
    if batch_state.get("chapter_outlines"):
        lead_updates["chapter_outlines"] = batch_state["chapter_outlines"]
    if batch_state.get("setup_complete"):
        lead_updates["setup_complete"] = batch_state["setup_complete"]

    # ── Phase ──
    if batch_state.get("current_phase"):
        lead_updates["current_phase"] = batch_state["current_phase"]
        if batch_state["current_phase"] == "done":
            lead_updates["current_agent"] = "done"

    # ── Set Lead Agent back to chat for continuation ──
    if lead_updates.get("current_agent") != "done":
        lead_updates["current_agent"] = "chat_agent"
    lead_updates["agent_context"] = {"type": "batch_completed"}

    # Build status message
    chapter_count = len(batch_state.get("completed_chapters", []))
    word_count = batch_state.get("word_count", 0)
    quality_score = batch_state.get("quality_score", 0)

    _status_lines = [
        f"✅ 自动创作已完成！\n",
        f"**进度**: 已完成 {chapter_count} 章",
        f"**当前章节**: 第{batch_state.get('current_chapter', 1)}章",
    ]
    if quality_score:
        _status_lines.append(f"**质量评分**: {quality_score:.1f}")
    if word_count:
        _status_lines.append(f"**总字数**: 约 {word_count} 字")
    _status_lines.append(
        "\n你可以:\n"
        "1. 继续对话式创作 — 讨论剧情或角色\n"
        "2. 让评审编辑检查章节质量 — 输入 /review\n"
        "3. 继续自动生成更多章节 — 告诉我继续"
    )

    lead_updates["messages"] = [
        AIMessage(
            content="\n".join(_status_lines),
            name="bridge_agent",
        )
    ]

    logger.info(
        "[Bridge] Mapped batch results: %d chapters, phase=%s, score=%s",
        chapter_count,
        batch_state.get("current_phase"),
        quality_score,
    )

    return lead_updates
