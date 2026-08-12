"""Evaluation module — 统一 LLM 评审体系（v8.2）。

统一评分体系架构（v8.2）：
    Unified LLM Review（单次调用五视角评审）
        ↓ 标签化解析 + 自洽校验 + 分歧仲裁
    Verdict Engine（融合 + 路由三态）
        ↓
    VerdictResult → VerdictRouter (PASS/REFINE/REWRITE) + FeedbackBundle

v8.2: 清除程序化传感器（Programmatic Sensors）/ 多轮辩论 / 加权融合 / 校准，
评分职责 100% 由统一 LLM 评审（evaluation/unified）承担。
"""

from novelfactory.evaluation.schemas import (
    AttemptInfo,
    FeedbackBundle,
    VerdictLevel,
    VerdictResult,
)
from novelfactory.evaluation.unified import (
    UnifiedFourDim,
    UnifiedReviewEngine,
    UnifiedReviewResult,
    apply_consistency_check,
    parse_review_output,
)

__all__ = [
    "AttemptInfo",
    "FeedbackBundle",
    "VerdictLevel",
    "VerdictResult",
    "UnifiedReviewEngine",
    "UnifiedReviewResult",
    "UnifiedFourDim",
    "parse_review_output",
    "apply_consistency_check",
]
