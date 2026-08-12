"""统一 LLM 评审系统（v8.2）。

将分散的"程序化传感器 + 5 个 LLM 维度 + 多轮辩论 + 加权融合"收敛为：
单次 LLM 调用五视角评审 + 标签化解析 + 自洽校验 + 分歧驱动仲裁 + 回归复查。
"""

from novelfactory.evaluation.unified.arbitration import arbitrate, parse_arbitration
from novelfactory.evaluation.unified.engine import UnifiedReviewEngine
from novelfactory.evaluation.unified.parser import (
    apply_consistency_check,
    parse_review_output,
)
from novelfactory.evaluation.unified.schemas import UnifiedFourDim, UnifiedReviewResult

__all__ = [
    "UnifiedReviewResult",
    "UnifiedFourDim",
    "UnifiedReviewEngine",
    "parse_review_output",
    "apply_consistency_check",
    "parse_arbitration",
    "arbitrate",
]
