"""verdict_router — 3 分支路由，替代 _score_router 的 12 分支。

纯代码，零 LLM。所有兜底逻辑（次数用尽、短文本、scorer 故障）
已在 VerdictEngine 中处理，router 只做纯路由。
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.evaluation.schemas import VerdictLevel

logger = logging.getLogger(__name__)


def verdict_router(state: dict[str, Any]) -> str:
    """三级决议路由 — 替代 _score_router 的 12 分支。

    v7.4-fix: REWRITE 时保留最佳版本文本+评分到 state，
    防止最终耗尽重试时垃圾版本冲掉高分版本。

    路由规则极简：
        verdict.level == PASS    → "__exit_for_chapter__"
        verdict.level == REFINE  → "chapter_refiner"
        verdict.level == REWRITE → "chapter_planner" (v6.3: 重新规划后再写)

    所有兜底逻辑（次数用尽、短文本、scorer 故障）已在 VerdictEngine 中处理，
    router 只做纯路由。
    """
    verdict_data = state.get("verdict_result", {})
    level = verdict_data.get("level", "rewrite")
    quality_score = verdict_data.get("quality_score", 0.0)

    # 尝试从 VerdictLevel 枚举值匹配
    if isinstance(level, str):
        level_str = level
    elif isinstance(level, VerdictLevel):
        level_str = level.value
    else:
        level_str = str(level)

    if level_str == VerdictLevel.PASS.value:
        logger.info("[verdict_router] → __exit_for_chapter__ (PASS)")
        return "__exit_for_chapter__"

    if level_str == VerdictLevel.REFINE.value:
        logger.info("[verdict_router] → chapter_refiner (REFINE)")
        return "chapter_refiner"

    # ── REWRITE: 保存最佳版本，防止耗尽后垃圾版本冲掉高分版 ──
    # NOTE: 此处直接修改 mutable state dict 是 LangGraph 的已知设计模式。
    # 在 conditional_edge router 中，返回值仅用于路由（节点名），
    # 无法通过返回 dict 写入 state。LangGraph 在执行期间传入的 state dict
    # 是 mutable 的，直接修改会被持久化到 checkpoint，这是官方推荐的
    # "side-effect in router" 模式（参见 LangGraph docs: Side effects in nodes）。
    # 如果未来 LangGraph 提供 Command-based router state mutation API，应迁移过去。
    cr = state.get("crew_result", {}) or {}
    chapter_text = state.get("chapter_draft", "") or cr.get(
        "refined_chapter", cr.get("chapter_draft", "")
    )
    best_quality = state.get("best_version_quality", 0.0)
    if quality_score > best_quality and chapter_text:
        logger.info(
            "[verdict_router] REWRITE: 保存最佳版本 (quality=%.1f > best=%.1f)",
            quality_score, best_quality,
        )
        state["best_version_text"] = chapter_text
        state["best_version_quality"] = quality_score

    logger.info("[verdict_router] → chapter_planner (REWRITE)")
    return "chapter_planner"
