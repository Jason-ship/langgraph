"""Quality scoring and analysis."""

from novelfactory.analysis.ai_style_analyzer import (
    AIStyleMetrics,
    AIStyleResult,
    analyze_ai_style,
)
from novelfactory.analysis.old_reader_reviewer import (
    OldReaderResult,
    review_as_old_reader,
)

# 阈值常量从 config.constants 中心化导入：
#   AI_STYLE_THRESHOLD, COMPOSITE_THRESHOLD, LAO_SHU_THRESHOLD, QUALITY_SCORE_THRESHOLD
# 或使用 get_genre_thresholds() 获取题材感知阈值

__all__ = [
    "OldReaderResult",
    "review_as_old_reader",
    "AIStyleMetrics",
    "AIStyleResult",
    "analyze_ai_style",
]
