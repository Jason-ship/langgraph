"""Informed debate subpackage — 知情辩论子模块。

核心改进（v6.3）：
    程序化分析结果 + 跨章信号注入辩论 prompt，
    辩论双方从"盲评"变为"知情辩论"，基于程序化事实做增量分析。

组件：
    - prompts.py: 知情辩论 prompt 模板（含程序化注入）
    - parser.py:  Markdown 解析（迁移自 quality_panel_agents.py）
    - agents.py:  辩论 Agent（编辑↔读者多轮辩论）
    - engine.py:  辩论引擎（普通函数，非子图）
"""

from novelfactory.evaluation.debate.engine import InformedDebateEngine

__all__ = ["InformedDebateEngine"]
