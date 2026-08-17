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

    v7.4-fix: REWRITE 时保留最佳版本文本+评分到 state。
    v8.5-fix: 最佳版本保存已迁移到 verdict_engine_node（coordinator.py）——
    条件边 router 的返回值仅用于路由，对传入 state 的顶层键修改不会写回
    checkpoint（LangGraph Branch._route 不提交 state），原实现实际从未生效。

    路由规则极简：
        verdict.level == PASS    → "__exit_for_chapter__"
        verdict.level == REFINE  → "chapter_refiner"
        verdict.level == REWRITE → "chapter_planner" (v6.3: 重新规划后再写)

    所有兜底逻辑（次数用尽、短文本、scorer 故障）已在 VerdictEngine 中处理，
    router 只做纯路由。
    """
    verdict_data = state.get("verdict_result", {})
    level = verdict_data.get("level", "rewrite")

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

    logger.info("[verdict_router] → chapter_planner (REWRITE)")
    return "chapter_planner"
