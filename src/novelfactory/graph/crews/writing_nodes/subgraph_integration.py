"""Subgraph integration nodes extracted from writing_crew.py.

Provides thin wrappers that invoke compiled subgraphs as LangGraph nodes:
    - context_builder_node_fn  →  ContextBuilder subgraph
    - state_extractor_node_fn  →  StateExtractor subgraph
    - database_writer_node_fn  →  DatabaseWriter subgraph
"""

from __future__ import annotations

import logging

from novelfactory.agents.infra import read_usage_tracking
from novelfactory.state.crew_state import BaseCrewState

logger = logging.getLogger(__name__)

# ── Node: context_builder ────────────────────────────────────────────────────────


def context_builder_node_fn(state: BaseCrewState) -> dict:
    """Build writing context via ContextBuilder subgraph — as LangGraph node."""
    cr = state.get("crew_result", {})
    # v7.3-fix: 优先 crew_result，兜底顶层 state（确保项目名一致性）
    project_name = cr.get("project_name") or state.get("project_name", "") or ""
    # v5.9 P0-fix: Send 并行分发时 state.current_chapter 为权威来源
    current_ch = state.get("current_chapter", cr.get("current_chapter_number", 1))
    try:
        from novelfactory.graph.subgraphs.context_builder import build_context_builder

        ctx = build_context_builder().invoke(
            {
                "project_name": project_name,
                "chapter_number": current_ch,
            }
        )
        return {"writer_context": ctx.get("writer_context", "")}
    except (ImportError, ValueError, OSError) as e:
        logger.warning("ContextBuilder failed: %s", e)
        return {"writer_context": ""}


# ── Node: state_extractor ────────────────────────────────────────────────────────


def state_extractor_node_fn(state: BaseCrewState) -> dict:
    """Extract state after chapter completion — as LangGraph node."""
    from novelfactory.config.constants import MIN_CHAPTER_TEXT_LENGTH

    cr = state.get("crew_result", {})
    chapter_text = cr.get("refined_chapter", "") or cr.get("chapter_draft", "")
    current_ch = cr.get("current_chapter_number", 1)
    # v7.3-fix: 兜底顶层 state project_name
    project_name = cr.get("project_name") or state.get("project_name", "") or ""
    if chapter_text and len(chapter_text) > MIN_CHAPTER_TEXT_LENGTH:
        try:
            from novelfactory.graph.subgraphs.state_extractor import (
                build_state_extractor,
            )

            extractor = build_state_extractor()
            result = extractor.invoke(
                {
                    "project_name": project_name,
                    "chapter_number": current_ch,
                    "chapter_text": chapter_text,
                    "world_setting": cr.get("world_setting", ""),
                    "character_setting": cr.get("character_setting", ""),
                }
            )
            return {"extracted": result.get("extracted", {})}
        except (ImportError, ValueError, OSError) as e:
            logger.error("state extraction error: %s", e)
    return {"extracted": {}}


# ── Node: database_writer ────────────────────────────────────────────────────────


def database_writer_node_fn(state: BaseCrewState) -> dict:
    """Write chapter data to databases — as LangGraph node."""
    from novelfactory.config.constants import (
        EXCELLENT_THRESHOLD as _EXCELLENT_THRESHOLD,
    )
    from novelfactory.config.constants import (
        MIN_CHAPTER_TEXT_LENGTH,
    )

    cr = state.get("crew_result", {})
    chapter_text = cr.get("refined_chapter", "") or cr.get("chapter_draft", "")
    current_ch = cr.get("current_chapter_number", 1)
    # v7.3-fix: 兜底顶层 state project_name
    project_name = cr.get("project_name") or state.get("project_name", "") or ""

    # v5.12 FIX: 优先从 crew_result 读取 quality_score（子图状态隔离的权威来源），
    # 不再依赖 state 顶层（可能不存在），也不 fallback 到 review_result（可能不含顶层评分）。
    quality = float(
        cr.get("quality_score")
        or cr.get("review_result", {}).get("quality_score")
        or _EXCELLENT_THRESHOLD
    )
    logger.info(
        "[database_writer] ch%d quality=%.1f (from=%s)",
        current_ch,
        quality,
        "crew_result" if cr.get("quality_score") else "review_result",
    )

    extracted = state.get("extracted", {})
    # v5.12: extracted 为空时记录调试信息但不告警 — 这是 state_extractor 的行为，
    # 不影响创作流程（只是 DB 持久化降级）。
    if not extracted or len(extracted.get("characters") or []) == 0:
        logger.info(
            "[database_writer] ch%d extracted 为空，DB 角色/审计写入将跳过",
            current_ch,
        )

    if chapter_text and len(chapter_text) > MIN_CHAPTER_TEXT_LENGTH:
        try:
            from novelfactory.graph.subgraphs.database_writer import (
                build_database_writer,
            )

            writer = build_database_writer()
            writer.invoke(
                {
                    "project_name": project_name,
                    "chapter_number": current_ch,
                    "chapter_text": chapter_text,
                    "chapter_title": f"第{current_ch}章",
                    "quality_score": quality,
                    "extracted": extracted,
                    "actual_usage": read_usage_tracking(),
                }
            )
        except (ImportError, ValueError, OSError):
            logger.exception("database write error")
    return {}
