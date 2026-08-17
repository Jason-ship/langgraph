"""Verdict subpackage — 统一评审融合 + 路由 + 反馈（v8.2）。

组件：
    - engine.py:        VerdictEngine 融合引擎（统一评审结果 → VerdictResult + 路由三态）
    - router.py:        verdict_router 3 分支路由（纯代码）

v8.2: calibration.py 已随统一 LLM 评审移除（自洽校验在 unified/parser.py 承接）。
v8.5-clean: feedback.py（FeedbackBuilder）为 v8.2 前遗产，无任何调用点，已删除。
"""

from novelfactory.evaluation.verdict.engine import VerdictEngine
from novelfactory.evaluation.verdict.router import verdict_router

__all__ = [
    "VerdictEngine",
    "verdict_router",
]
