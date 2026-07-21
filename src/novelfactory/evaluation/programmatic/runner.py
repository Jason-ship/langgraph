"""程序化分析运行器 — 统一调用三个传感器，产出 ProgrammaticReport + CrossChapterSignals。

这是 VerdictEngine 调用程序化分析的入口。
三个传感器并行运行（均为纯代码，毫秒级），结果融合为统一报告。
"""

from __future__ import annotations

import logging

from novelfactory.evaluation.programmatic.ai_style_sensor import AIStyleSensor
from novelfactory.evaluation.programmatic.cross_chapter_sensor import (
    CrossChapterSensor,
    ItemStateTracker,
)
from novelfactory.evaluation.programmatic.old_reader_sensor import (
    GENRE_SEVERE_TOXIC_EXEMPT,
    OldReaderSensor,
)
from novelfactory.evaluation.schemas import (
    CrossChapterSignals,
    ProgrammaticReport,
)

logger = logging.getLogger(__name__)

# 严重毒点类型集合（全局默认）
_SEVERE_TOXIC_TYPES = {"NTR", "NUE_ZHU", "SHENGMU"}


def _resolve_severe_toxic_types(genre: str | None) -> set[str]:
    """题材感知严重毒点分类 — 某些题材中特定毒点是其核心情感元素。

    v7.6: 都市亲情治愈题材中 SHENGMU（原谅/宽恕）不是毒点，是主题。
    匹配策略：
      1. 先精确匹配豁免配置键（处理"治愈""日常""搞笑"等非标准题材名）
      2. 未命中时使用 resolve_genre 标准化后精确匹配（处理"都市亲情"→"都市"等映射）
    """
    if not genre:
        return _SEVERE_TOXIC_TYPES

    # 1. 精确匹配 — 直接命中豁免配置
    if genre in GENRE_SEVERE_TOXIC_EXEMPT:
        return _SEVERE_TOXIC_TYPES - GENRE_SEVERE_TOXIC_EXEMPT[genre]

    # 2. resolve_genre 标准化后精确匹配
    from novelfactory.config.constants import resolve_genre

    resolved = resolve_genre(genre)
    if resolved != "default" and resolved in GENRE_SEVERE_TOXIC_EXEMPT:
        return _SEVERE_TOXIC_TYPES - GENRE_SEVERE_TOXIC_EXEMPT[resolved]

    return _SEVERE_TOXIC_TYPES


def run_programmatic_analysis(
    chapter_text: str,
    chapter_index: int = 1,
    genre: str | None = None,
    prev_chapters_summary: str = "",
    character_setting: str = "",
    story_outline: str = "",
    tracker: ItemStateTracker | None = None,
) -> tuple[ProgrammaticReport, CrossChapterSignals]:
    """执行全部程序化分析，返回统一报告。

    Args:
        chapter_text: 章节文本
        chapter_index: 章节序号
        genre: 题材名称
        prev_chapters_summary: 前文摘要
        character_setting: 角色设定
        story_outline: 故事大纲
        tracker: 可选的物品状态追踪器实例（每次调用应传入独立实例，避免跨请求状态污染）

    Returns:
        (ProgrammaticReport, CrossChapterSignals)
    """
    # === 1. AI 味检测 ===
    ai_sensor = AIStyleSensor()
    ai_score, ai_metrics, ai_issues, ai_fix, ai_short = ai_sensor.analyze(
        chapter_text, genre=genre
    )

    # === 2. 老书虫评审 ===
    old_reader = OldReaderSensor()
    lao_score, toxic_points, shuangdian_points, lao_fix, verdict_text, lao_short = (
        old_reader.analyze(chapter_text, genre=genre)
    )

    # === 3. 跨章分析 ===
    # 每次调用创建独立的 ItemStateTracker，避免跨请求/项目状态污染
    item_tracker = tracker if tracker is not None else ItemStateTracker()
    cross_sensor = CrossChapterSensor()
    cross_signals = cross_sensor.analyze(
        chapter_text=chapter_text,
        chapter_index=chapter_index,
        prev_chapters_summary=prev_chapters_summary,
        character_setting=character_setting,
        story_outline=story_outline,
        tracker=item_tracker,
    )

    # === 融合 ===
    is_short_text = ai_short or lao_short
    # v7.6: 题材感知严重毒点 — 都市亲情治愈中 SHENGMU 是主题而非毒点
    effective_severe_types = _resolve_severe_toxic_types(genre)
    has_severe_toxic = any(t.type in effective_severe_types for t in toxic_points)
    severe_toxic_types = [
        t.type for t in toxic_points if t.type in effective_severe_types
    ]

    report = ProgrammaticReport(
        ai_style_score=ai_score,
        ai_style_metrics=ai_metrics,
        ai_style_issues=ai_issues,
        ai_style_fix=ai_fix,
        lao_shu_chong_score=lao_score,
        toxic_points=toxic_points,
        shuangdian_points=shuangdian_points,
        lao_shu_chong_fix=lao_fix,
        verdict_text=verdict_text,
        has_severe_toxic=has_severe_toxic,
        severe_toxic_types=severe_toxic_types,
        is_short_text=is_short_text,
    )

    logger.info(
        "[程序化分析] ai=%.3f lao=%.1f short=%s severe_toxic=%s toxic=%d shuangdian=%d "
        "cross_prev=%s",
        ai_score,
        lao_score,
        is_short_text,
        has_severe_toxic,
        len(toxic_points),
        len(shuangdian_points),
        cross_signals.has_prev_context,
    )

    return report, cross_signals
