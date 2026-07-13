"""Verdict subpackage — 融合引擎 + 路由 + 反馈 + 校准。

组件：
    - engine.py:        VerdictEngine 融合引擎（纯代码编排，内部调用 LLM）
    - router.py:        verdict_router 3 分支路由（纯代码）
    - feedback.py:      FeedbackBuilder 统一反馈包构建（纯代码）
    - calibration.py:   CalibrationModule 校准（纯代码）
"""

from novelfactory.evaluation.verdict.calibration import CalibrationModule
from novelfactory.evaluation.verdict.engine import VerdictEngine
from novelfactory.evaluation.verdict.feedback import FeedbackBuilder
from novelfactory.evaluation.verdict.router import verdict_router

__all__ = [
    "CalibrationModule",
    "FeedbackBuilder",
    "VerdictEngine",
    "verdict_router",
]
