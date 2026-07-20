"""Config 常量统一性测试。

覆盖:
    - 题材阈值完整性
    - 关键词映射唯一性
    - 融合权重总和
"""

from __future__ import annotations

from novelfactory.config.constants import (
    GENRE_THRESHOLDS,
    VERDICT_WEIGHTS,
    resolve_genre,
)


class TestGenreThresholds:
    """题材阈值完整性测试。"""

    def test_all_genres_have_required_keys(self):
        """所有题材必须包含 quality_score / composite / ai_style / lao_shu。"""
        required = {"quality_score", "composite", "ai_style", "lao_shu"}
        for genre, thresholds in GENRE_THRESHOLDS.items():
            if genre == "default":
                continue
            missing = required - set(thresholds.keys())
            assert not missing, f"题材 '{genre}' 缺少字段: {missing}"

    def test_default_threshold_exists(self):
        """兜底 default 配置必须存在。"""
        assert "default" in GENRE_THRESHOLDS

    def test_resolve_genre_valid(self):
        """已知题材应正确解析。"""
        assert resolve_genre("玄幻") == "玄幻"
        assert resolve_genre("仙侠") == "仙侠"

    def test_resolve_genre_unknown(self):
        """未知题材应返回 default。"""
        assert resolve_genre("未知题材123") == "default"

    def test_resolve_genre_none(self):
        """None 应返回 default。"""
        assert resolve_genre(None) == "default"

    def test_resolve_genre_by_keyword(self):
        """关键词匹配应正确。"""
        assert resolve_genre("打脸爽文") == "爽文"
        assert resolve_genre("甜宠文") == "现代言情"

    def test_genre_description_not_empty(self):
        """每个题材应有非空 description。"""
        for genre, thresholds in GENRE_THRESHOLDS.items():
            desc = thresholds.get("description", "")
            assert desc, f"题材 '{genre}' description 为空"


class TestVerdictWeights:
    """融合权重验证测试。"""

    def test_weights_sum_approx_one(self):
        """VERDICT_WEIGHTS 各维度权重之和应 ≈ 1.0。"""
        total = sum(VERDICT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"权重总和 {total} 偏离 1.0"

    def test_all_weights_positive(self):
        """所有权重应为正值。"""
        for name, w in VERDICT_WEIGHTS.items():
            assert w > 0, f"权重 {name}={w} 应为正值"

    def test_required_weight_keys(self):
        """必须包含所有核心维度。"""
        required = {"quality", "programmatic", "cross_chapter", "debate_penalty"}
        missing = required - set(VERDICT_WEIGHTS.keys())
        assert not missing, f"缺少核心权重: {missing}"
