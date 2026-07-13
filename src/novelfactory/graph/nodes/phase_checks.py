"""Phase 2/3 Check Nodes: volume, quality, foreshadowing checks (v4.1+)."""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.config.constants import FALLBACK_TARGET_CHAPTERS
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)

_DRAFT_PREVIEW_LENGTH = 500
_ENDGAME_CHAPTERS_REMAINING = 10
_FORESHADOWING_HIGH_PRIORITY = 7


# ── Phase 3: Volume Check Node (v4.1+) ─────────────────────────────────────────


def volume_check_node(state: NovelFactoryState) -> dict:
    """Phase 3 volume completion check + lazy outline generation.

    Detects volume boundaries and generates next volume chapter outlines
    on-demand. Updates volume_status and appends transition guidance to
    auto_guidance.
    """
    project_name = state.get("project_name", "未命名项目")
    current_ch = state.get("current_chapter", 1)

    updates: dict = {}
    guidance_parts: list[str] = []

    try:
        from novelfactory.config.database import DatabaseManager
        from novelfactory.pipeline.phase3_manager import Phase3Manager
        from novelfactory.pipeline.scale_manager import ScaleManager

        with DatabaseManager.get_instance().get_connection() as conn:
            scale = ScaleManager(conn, project_name)
            phase3 = Phase3Manager(conn, project_name)

            try:
                volume_completion = phase3.volume.check_volume_completion(
                    project_name, current_ch
                )
                if volume_completion and volume_completion.get("volume_complete"):
                    transition_ctx = phase3.volume.build_transition_context(
                        project_name, current_ch
                    )
                    if transition_ctx:
                        guidance_parts.append(
                            f"[卷间过渡] {transition_ctx[:_DRAFT_PREVIEW_LENGTH]}"
                        )
                        logger.info(
                            "[volume_check] Volume transition detected at ch%d",
                            current_ch,
                        )

                    next_vol = volume_completion.get("next_volume")

                    # ── 生成下一卷大纲（懒加载，卷到卷） ──
                    if next_vol:
                        _generate_next_volume_outlines(state, scale, next_vol)

                    # 记录卷状态信息
                    next_vol_idx = (
                        next_vol
                        if next_vol
                        else (
                            volume_completion.get("current_volume", 0) + 1
                            if volume_completion
                            else 0
                        )
                    )
                    logger.info(
                        "[volume_check] 卷边界 — ch%d 完成, 下一卷 %d (线性创作)",
                        current_ch,
                        next_vol_idx,
                    )

                updates["volume_status"] = {
                    "checked": True,
                    "volume_complete": volume_completion.get("volume_complete", False)
                    if volume_completion
                    else False,
                    "next_volume": volume_completion.get("next_volume")
                    if volume_completion
                    else None,
                    "current_chapter": current_ch,
                }
            except Exception as e:
                logger.warning("[volume_check] Volume check failed: %s", e)
                updates["volume_status"] = {
                    "checked": True,
                    "error": str(e),
                    "current_chapter": current_ch,
                }
    except Exception as e:
        logger.warning("[volume_check] ScaleManager init failed (non-fatal): %s", e)

    # Append to existing auto_guidance (preserve guidance from previous nodes)
    existing_guidance = state.get("auto_guidance", "") or ""
    if guidance_parts:
        new_guidance = "\n".join(guidance_parts)
        combined = (existing_guidance + "\n" + new_guidance).strip()
        updates["auto_guidance"] = combined
        logger.info(
            "[volume_check] Auto-guidance appended for ch%d: %d items",
            current_ch + 1,
            len(guidance_parts),
        )

    return updates


# ── Phase 3: Quality Check Node (v4.1+) ────────────────────────────────────────


def quality_check_node(state: NovelFactoryState) -> dict:
    """Phase 3 quality decay detection.

    Analyzes recent chapter quality trends and injects corrective guidance
    if quality is declining. Updates quality_trend and appends guidance to
    auto_guidance.
    """
    project_name = state.get("project_name", "未命名项目")
    current_ch = state.get("current_chapter", 1)

    updates: dict = {}
    guidance_parts: list[str] = []

    try:
        from novelfactory.pipeline.phase3_manager import Phase3Manager

        with Phase3Manager(project_name=project_name) as phase3:
            quality_info = phase3.quality.detect_decay(project_name)

        if quality_info and quality_info.get("decaying"):
            alerts = quality_info.get("alerts", [])
            decay_desc = (
                "; ".join(alerts)
                if alerts
                else quality_info.get("pattern", "质量持续下降")
            )
            avg_score = quality_info.get("recent_avg", 0)
            guidance_parts.append(
                f"[质量干预] 检测到质量衰减: {decay_desc[:200]}，"
                f"近期均分{avg_score:.1f}。请在下一章提升描写细节、"
                f"增加角色内心独白、强化场景氛围渲染。"
            )
            logger.warning(
                "[quality_check] Quality decay detected at ch%d: %s",
                current_ch,
                decay_desc,
            )

        updates["quality_trend"] = {
            "checked": True,
            "decaying": quality_info.get("decaying", False) if quality_info else False,
            "recent_avg": quality_info.get("recent_avg", 0) if quality_info else 0,
            "alerts": quality_info.get("alerts", []) if quality_info else [],
            "pattern": quality_info.get("pattern", "") if quality_info else "",
        }
    except Exception as e:
        logger.warning("[quality_check] Quality check init failed (non-fatal): %s", e)
        updates["quality_trend"] = {
            "checked": True,
            "error": str(e),
        }

    # Append to existing auto_guidance (preserve guidance from previous nodes)
    existing_guidance = state.get("auto_guidance", "") or ""
    if guidance_parts:
        new_guidance = "\n".join(guidance_parts)
        combined = (existing_guidance + "\n" + new_guidance).strip()
        updates["auto_guidance"] = combined
        logger.info(
            "[quality_check] Auto-guidance appended for ch%d: %d items",
            current_ch + 1,
            len(guidance_parts),
        )

    return updates


# ── Phase 2: Foreshadowing Check Node (v4.1+) ──────────────────────────────────


def foreshadowing_check_node(state: NovelFactoryState) -> dict:
    """Phase 2 foreshadowing enforcement check.

    Detects overdue high-priority foreshadowing and unresolved items near
    novel end. Injects resolution guidance into auto_guidance. Updates
    foreshadowing_status, pacing_status, and audit_results.
    """
    project_name = state.get("project_name", "未命名项目")
    current_ch = state.get("current_chapter", 1)
    target_ch = state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS

    updates: dict = {}
    guidance_parts: list[str] = []
    audit_entries: list[dict] = []

    try:
        from novelfactory.config.database import DatabaseManager
        from novelfactory.pipeline.phase2_manager import Phase2Manager

        with DatabaseManager.get_instance().get_connection() as conn:
            phase2 = Phase2Manager(conn, project_name)

            overdue_count = 0
            unresolved_count = 0

            # Check overdue foreshadowing
            try:
                overdue = phase2.foreshadowing.get_overdue(project_name, current_ch)
                if overdue:
                    urgent = [
                        f
                        for f in overdue
                        if getattr(f, "priority", 0) >= _FORESHADOWING_HIGH_PRIORITY
                    ]
                    if urgent:
                        hints = "; ".join(
                            [
                                f"回收「{getattr(f, 'description', '')[:40]}」"
                                f"(第{getattr(f, 'planted_chapter', '?')}章埋设)"
                                for f in urgent[:5]
                            ]
                        )
                        guidance_parts.append(
                            f"[伏笔回收] 以下高优先级伏笔已过期，必须在近期章节回收: {hints}"
                        )
                        logger.info(
                            "[foreshadowing_check] %d overdue high-priority foreshadowing items",
                            len(urgent),
                        )
                    overdue_count = len(urgent)
                    for f in urgent:
                        audit_entries.append(
                            {
                                "type": "overdue_foreshadowing",
                                "description": getattr(f, "description", "")[:60],
                                "planted_chapter": getattr(f, "planted_chapter", 0),
                                "priority": getattr(f, "priority", 0),
                            }
                        )

                # Near end of novel: force resolution of all unresolved
                chapters_remaining = target_ch - current_ch
                if (
                    chapters_remaining <= _ENDGAME_CHAPTERS_REMAINING
                    and chapters_remaining > 0
                ):
                    unresolved = phase2.foreshadowing.get_all_active(project_name)
                    if unresolved and len(unresolved) > 0:
                        all_hints = "; ".join(
                            [
                                f"「{getattr(f, 'description', '')[:30]}」"
                                for f in unresolved[:8]
                            ]
                        )
                        guidance_parts.append(
                            f"[伏笔终局] 距完结仅{chapters_remaining}章，"
                            f"尚有{len(unresolved)}条伏笔未回收: {all_hints}。"
                            f"必须在剩余章节中完成回收。"
                        )
                        logger.info(
                            "[foreshadowing_check] %d unresolved foreshadowing with %d chapters left",
                            len(unresolved),
                            chapters_remaining,
                        )
                    unresolved_count = len(unresolved)
                    for f in unresolved:
                        audit_entries.append(
                            {
                                "type": "unresolved_foreshadowing",
                                "description": getattr(f, "description", "")[:60],
                                "planted_chapter": getattr(f, "planted_chapter", 0),
                                "priority": getattr(f, "priority", 0),
                            }
                        )

            except Exception as e:
                logger.warning(
                    "[foreshadowing_check] Foreshadowing check failed: %s", e
                )

            updates["foreshadowing_status"] = "overdue" if overdue_count > 0 else "ok"
            updates["pacing_status"] = (
                "final_push"
                if (target_ch - current_ch) <= _ENDGAME_CHAPTERS_REMAINING
                and unresolved_count > 0
                else "normal"
            )
            updates["audit_results"] = audit_entries

    except Exception as e:
        logger.warning(
            "[foreshadowing_check] Phase2Manager init failed (non-fatal): %s", e
        )

    # Append to existing auto_guidance (preserve guidance from previous nodes)
    existing_guidance = state.get("auto_guidance", "") or ""
    if guidance_parts:
        new_guidance = "\n".join(guidance_parts)
        combined = (existing_guidance + "\n" + new_guidance).strip()
        updates["auto_guidance"] = combined
        logger.info(
            "[foreshadowing_check] Auto-guidance appended for ch%d: %d items",
            current_ch + 1,
            len(guidance_parts),
        )

    return updates


def _generate_next_volume_outlines(
    state: NovelFactoryState,
    scale: Any,
    next_vol: int,
) -> None:
    """Lazy-generate chapter outlines for the next volume on-demand.

    This solves the max_tokens limitation: instead of generating all 1000
    chapter outlines during setup, we generate them volume-by-volume as
    the writing loop progresses.
    """
    project_name = state.get("project_name", "未命名项目")
    volume_structure = state.get("volume_structure", {})
    volumes = volume_structure.get("volumes", [])

    if next_vol > len(volumes):
        logger.warning("[lazy_outline] Volume %d not in structure", next_vol)
        return

    vol_info = volumes[next_vol - 1]
    ch_start, ch_end = vol_info.get("chapter_range", [0, 0])
    if ch_start == 0:
        logger.warning("[lazy_outline] Invalid chapter range for vol %d", next_vol)
        return

    # Check if outlines already exist
    try:
        existing = scale.outline.get_chapter_outlines(project_name, next_vol)
        if existing:
            logger.info("[lazy_outline] Volume %d outlines already exist", next_vol)
            return
    except Exception:
        pass  # If check fails, proceed with generation

    # Generate outlines for this volume
    try:
        from novelfactory.agents.setup_agents import create_volume_detail_writer_agent
        from novelfactory.config.llm import get_worker_llm

        agent = create_volume_detail_writer_agent(get_worker_llm())
        result = agent.invoke(
            {
                "crew_result": {
                    "volume_number": next_vol,
                    "volume_title": vol_info.get("title", f"第{next_vol}卷"),
                    "volume_theme": vol_info.get("theme", ""),
                    "volume_summary": vol_info.get("summary", ""),
                    "chapter_start": ch_start,
                    "chapter_end": ch_end,
                    "world_setting": state.get("world_setting", ""),
                    "character_setting": state.get("character_setting", ""),
                    "story_outline": state.get("story_outline", ""),
                }
            }
        )

        outlines = result.get("crew_result", {}).get("chapter_outlines_detail", [])
        for ol in outlines:
            try:
                scale.outline.save_chapter_outline(
                    project_name=project_name,
                    volume_number=next_vol,
                    chapter_number=ol.get("chapter_number", 0),
                    title=ol.get("title", ""),
                    core_events=ol.get("core_events", ""),
                    importance=ol.get("importance", 5),
                )
            except Exception as e:
                logger.warning(
                    "[lazy_outline] Failed to save ch%d: %s",
                    ol.get("chapter_number", 0),
                    e,
                )

        logger.info(
            "[lazy_outline] Generated %d chapter outlines for volume %d (ch%d-%d)",
            len(outlines),
            next_vol,
            ch_start,
            ch_end,
        )
    except Exception as e:
        logger.error("[lazy_outline] Generation failed for vol %d: %s", next_vol, e)
