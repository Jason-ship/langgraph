"""老书虫传感器 — 封装 analysis/old_reader_reviewer.py 为 ProgrammaticReport 产出。

适配器模式：不修改原始评审器，在新模块中封装为统一产出格式。
原始评审器保持不变，其他模块仍可直接调用。

职责：
    调用 review_as_old_reader() → 转换为 ToxicDetail + ShuangdianDetail + fix
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.analysis.old_reader_reviewer import review_as_old_reader
from novelfactory.evaluation.schemas import ShuangdianDetail, ToxicDetail

logger = logging.getLogger(__name__)

# 严重毒点类型集合 — 用于 has_severe_toxic 判定（全局默认）
_SEVERE_TOXIC_TYPES = {"NTR", "NUE_ZHU", "SHENGMU"}

# v7.6: 题材感知严重毒点豁免 — 某些题材中特定"毒点"是其核心情感元素
# 都市亲情/治愈：原谅/宽恕是主题，不应一票否决 → 移除 SHENGMU
# 都市搞笑：副线情节中的 NTR 提及可能是误判 → 降级 NTR
GENRE_SEVERE_TOXIC_EXEMPT: dict[str, set[str]] = {
    "都市": {"SHENGMU"},  # 都市亲情中"原谅"是情感核心，不是毒点
    "都市亲情": {"SHENGMU"},
    "治愈": {"SHENGMU"},
    "日常": {"SHENGMU"},
    "搞笑": {"SHENGMU", "NTR"},  # 搞笑文中 NTR 类情节通常是玩梗，非真毒点
}


class OldReaderSensor:
    """老书虫传感器 — 纯代码，零 LLM。

    调用现有的 review_as_old_reader() 函数，将结果转换为统一的毒点/爽点详情。
    """

    def analyze(
        self,
        chapter_text: str,
        genre: str | None = None,
    ) -> tuple[float, list[ToxicDetail], list[ShuangdianDetail], str, str, bool]:
        """执行老书虫评审。

        Args:
            chapter_text: 章节文本
            genre: 题材名称

        Returns:
            (score, toxic_points, shuangdian_points, fix, verdict_text, is_short_text)
        """
        context = {"genre": genre} if genre else {}
        result: dict[str, Any] = review_as_old_reader(chapter_text, context=context)

        score: float = result["lao_shu_chong_score"]
        verdict_text: str = result["verdict"]

        # 判断是否短文本
        is_short_text = "文本过短" in verdict_text or "无法" in verdict_text

        # 转换毒点
        toxic_points: list[ToxicDetail] = []
        for td in result.get("toxic_details", []):
            toxic_points.append(
                ToxicDetail(
                    type=td["type"],
                    description=td.get("description", ""),
                    severity=td.get("severity", "medium"),
                    weight=td.get("weight", 0.0),
                )
            )

        # 转换爽点
        shuangdian_points: list[ShuangdianDetail] = []
        for sd in result.get("shuangdian_details", []):
            shuangdian_points.append(
                ShuangdianDetail(
                    type=sd["type"],
                    description=sd.get("description", ""),
                    weight=sd.get("weight", 0.0),
                )
            )

        # 生成修改建议
        issues: list[str] = result.get("issues", [])
        fix = self._build_fix_suggestion(issues)

        logger.info(
            "[老书虫传感器] score=%.1f toxic=%d shuangdian=%d verdict=%s",
            score,
            len(toxic_points),
            len(shuangdian_points),
            verdict_text,
        )

        return score, toxic_points, shuangdian_points, fix, verdict_text, is_short_text

    def _build_fix_suggestion(self, issues: list[str]) -> str:
        """根据检测结果生成修改建议。"""
        if not issues:
            return ""

        parts: list[str] = []
        for issue in issues:
            parts.append(f"- {issue}")

        return "老书虫修改建议：\n" + "\n".join(parts)
