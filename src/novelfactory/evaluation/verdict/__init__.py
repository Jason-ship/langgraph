"""Verdict subpackage — 统一评审融合 + 路由 + 反馈（v8.2）。

组件：
    - engine.py:        VerdictEngine 融合引擎（统一评审结果 → VerdictResult + 路由三态）
    - router.py:        verdict_router 3 分支路由（纯代码）
    - feedback.py:      FeedbackBuilder 反馈包构建（兼容保留）

v8.2: calibration.py 已随统一 LLM 评审移除（自洽校验在 unified/parser.py 承接）。
"""

from novelfactory.evaluation.verdict.engine import VerdictEngine
from novelfactory.evaluation.verdict.feedback import FeedbackBuilder
from novelfactory.evaluation.verdict.router import verdict_router

__all__ = [
    "FeedbackBuilder",
    "VerdictEngine",
    "verdict_router",
]
