"""Programmatic sensors subpackage — 程序化传感器子模块。

包含四类纯代码传感器（零 LLM）：
    1. AIStyleSensor — AI味 8 维检测
    2. OldReaderSensor — 老书虫毒点/爽点检测
    3. CrossChapterSensor — 跨章一致性信号采集
    4. ItemStateTracker — 跨章物品状态追踪（v7.3）

设计原则：只产出客观数据，不做判断。判断交给 LLM。
"""

from novelfactory.evaluation.programmatic.ai_style_sensor import AIStyleSensor
from novelfactory.evaluation.programmatic.cross_chapter_sensor import (
    CrossChapterSensor,
    ItemStateTracker,
    SentimentConsistencyFilter,
)
from novelfactory.evaluation.programmatic.old_reader_sensor import OldReaderSensor
from novelfactory.evaluation.programmatic.runner import run_programmatic_analysis

__all__ = [
    "AIStyleSensor",
    "CrossChapterSensor",
    "ItemStateTracker",
    "SentimentConsistencyFilter",
    "OldReaderSensor",
    "run_programmatic_analysis",
]
