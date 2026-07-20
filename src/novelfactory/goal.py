"""Goal 管理 — 目标状态和评估。

Migrated from DeerFlow runtime/goal.py + agents/goal_state.py.

实现类似 Claude Code 风格的目标循环：Agent 持续执行直到目标达成。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

logger = logging.getLogger(__name__)

GoalBlocker = Literal[
    "none", "missing_evidence", "needs_user_input",
    "run_failed", "external_wait", "goal_not_met_yet",
]


@dataclass
class GoalEvaluation:
    """目标评估结果。"""

    satisfied: bool = False
    blocker: GoalBlocker = "none"
    reason: str = ""
    evidence_summary: str = ""


@dataclass
class GoalState:
    """目标状态。"""

    objective: str = ""
    status: str = "active"  # active, completed, failed, cancelled
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    consecutive_count: int = 0
    max_consecutive: int = 10
    no_progress_count: int = 0
    max_no_progress: int = 3


class GoalManager:
    """目标管理器 — 管理单个目标的生命周期。"""

    def __init__(self, max_consecutive: int = 10, max_no_progress: int = 3):
        self._goal: GoalState | None = None
        self._max_consecutive = max_consecutive
        self._max_no_progress = max_no_progress

    def set_goal(self, objective: str) -> GoalState:
        """设置新目标。"""
        self._goal = GoalState(
            objective=objective,
            max_consecutive=self._max_consecutive,
            max_no_progress=self._max_no_progress,
        )
        logger.info("[goal] Set: %s", objective[:100])
        return self._goal

    def get_goal(self) -> GoalState | None:
        """获取当前目标。"""
        return self._goal

    def clear_goal(self) -> None:
        """清除目标。"""
        self._goal = None
        logger.info("[goal] Cleared")

    def evaluate(self, evidence: str, reason: str = "", llm=None) -> GoalEvaluation:
        """评估目标是否达成。

        支持两种评估模式：
        1. LLM 评估：传入 llm 参数，使用 LLM 对 evidence 做语义理解
        2. 计数评估：无 LLM 时回退到简化版计数逻辑

        Args:
            evidence: 评估证据文本。
            reason: 可选原因。
            llm: 可选的 LLM 实例，用于语义评估。

        Returns:
            GoalEvaluation 评估结果。
        """
        if self._goal is None:
            return GoalEvaluation(satisfied=True, reason="No active goal")

        # 更新计数
        self._goal.consecutive_count += 1
        self._goal.updated_at = datetime.now(UTC).isoformat()

        # LLM 评估模式
        if llm is not None and evidence:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a goal evaluation assistant. Determine if the following evidence "
                        "demonstrates that the goal has been achieved. "
                        f"Goal: {self._goal.objective}\n\n"
                        "Respond with only 'YES' or 'NO' followed by a brief reason."
                    ),
                    HumanMessage(content=f"Evidence: {evidence[:2000]}"),
                ]
                response = llm.invoke(messages)
                result_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

                if result_text.upper().startswith("YES"):
                    return GoalEvaluation(
                        satisfied=True,
                        reason=f"LLM assessment: {result_text[:200]}",
                        evidence_summary=evidence[:500],
                    )
                elif result_text.upper().startswith("NO"):
                    return GoalEvaluation(
                        satisfied=False,
                        blocker="goal_not_met_yet",
                        reason=f"LLM assessment: {result_text[:200]}",
                        evidence_summary=evidence[:500],
                    )
            except Exception:
                logger.warning("[goal] LLM evaluation failed, falling back to counting logic", exc_info=True)

        # 计数评估模式（回退）
        # 检查是否超限
        if self._goal.consecutive_count >= self._goal.max_consecutive:
            return GoalEvaluation(
                satisfied=True,
                reason=f"Max consecutive attempts reached ({self._goal.max_consecutive})",
                evidence_summary=evidence,
            )

        return GoalEvaluation(
            satisfied=False,
            blocker="goal_not_met_yet",
            reason=reason or "Goal not yet met, continuing...",
            evidence_summary=evidence,
        )


__all__ = [
    "GoalState",
    "GoalEvaluation",
    "GoalManager",
    "GoalBlocker",
]