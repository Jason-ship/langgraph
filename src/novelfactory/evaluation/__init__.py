"""Evaluation module — 评分模块一体化重构 (v6.3)。

统一评分体系架构：
    Programmatic Sensors (纯代码传感器)
        ↓ 注入
    Informed Debate (知情辩论 LLM)
        ↓ + Four-Dim LLM Score
    Verdict Engine (融合引擎 纯代码)
        ↓
    VerdictResult → VerdictRouter (3级路由) + FeedbackBundle (统一反馈)

替代旧的 coordinator.py + reviewer.py 评分逻辑 + routing.py 路由逻辑。
"""

from novelfactory.evaluation.schemas import (
    AttemptInfo,
    CrossChapterSignals,
    DebateReport,
    FeedbackBundle,
    FourDimReviewResult,
    ProgrammaticReport,
    VerdictLevel,
    VerdictResult,
)

__all__ = [
    "AttemptInfo",
    "CrossChapterSignals",
    "DebateReport",
    "FeedbackBundle",
    "FourDimReviewResult",
    "ProgrammaticReport",
    "VerdictLevel",
    "VerdictResult",
]
