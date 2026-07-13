"""自定义 ManagedValue — 为 NovelFactory 提供声明式托管状态。

ManagedValue 是 LangGraph v1.x 提供的托管值框架：
  - 继承 ``ManagedValue[V]`` 并实现 ``get(scratchpad) -> V``
  - 在图状态中通过 ``Annotated[V, ManagedClass]`` 注册
  - 每次节点执行前自动计算，不持久化到检查点

v6.0 新增：
  - ChapterProgress: 实时写作进度（当前/目标/已完成）
  - PhaseStatus: 当前阶段状态机状态
"""

from __future__ import annotations

import logging

# v6.0.1: PregelScratchpad 是 LangGraph 私有 API (_internal._scratchpad)。
# 如果 LangGraph 升级改变了该模块路径，ManagedValue 会直接导入失败。
# 添加防护：导入失败时回退到安全的运行时占位类型。
try:
    from langgraph._internal._scratchpad import PregelScratchpad
except ImportError:
    logger = logging.getLogger(__name__)
    logger.critical(
        "[managed_values] Cannot import PregelScratchpad — "
        "ChapterProgress/PhaseStatus will return empty dicts. "
        "LangGraph may have moved PregelScratchpad from "
        "langgraph._internal._scratchpad; update the import path. "
        "falling back to object type for PregelScratchpad."
    )
    # 回退：用 Any 类型替代，ManagedValue.get() 内部仍通过 hasattr 安全访问
    PregelScratchpad: type = object  # type: ignore[no-redef]

from langgraph.managed.base import ManagedValue

_log = logging.getLogger(__name__)


class ChapterProgress(ManagedValue[dict]):
    """实时写作进度追踪。

    每次节点执行前自动计算，返回当前进度快照：
      - current: 当前写作章节号
      - total: 目标章节总数
      - completed: 已完成章节数
      - percent: 完成百分比（0-100）
    """

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> dict:
        state = scratchpad.state if hasattr(scratchpad, "state") else {}
        current = state.get("current_chapter", 0) if isinstance(state, dict) else 0
        total = state.get("target_chapters", 0) if isinstance(state, dict) else 0
        completed = (
            len(state.get("completed_chapters", [])) if isinstance(state, dict) else 0
        )
        percent = round((completed / max(total, 1)) * 100, 1) if total > 0 else 0.0
        return {
            "current": current,
            "total": total,
            "completed": completed,
            "percent": percent,
        }


class PhaseStatus(ManagedValue[dict]):
    """当前阶段状态机状态。

    - phase: 当前阶段名称
    - setup_complete: 设置是否完成
    - chapter_approved: 当前章节是否已批准
    """

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> dict:
        state = scratchpad.state if hasattr(scratchpad, "state") else {}
        return {
            "phase": state.get("current_phase", "setup")
            if isinstance(state, dict)
            else "setup",
            "setup_complete": state.get("setup_complete", False)
            if isinstance(state, dict)
            else False,
            "chapter_approved": state.get("chapter_approved", False)
            if isinstance(state, dict)
            else False,
        }
