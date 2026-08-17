"""Evaluation schemas — 评分模块核心数据结构 (v6.3)。

这是评分体系的唯一契约层。所有评分源的产出
最终汇聚到 VerdictResult，由 VerdictRouter 消费。

v8.2: 统一 LLM 评审（evaluation/unified）承接全部评分职责，
程序化传感器 / 多轮辩论 / 加权融合 / 校准均已移除。
v8.5-clean: 清理 v8.2 前遗留 schema（ProgrammaticReport / CrossChapterSignals /
DebateReport / FourDimReviewResult / EvidenceItem 等，无任何调用点）。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
#  统一反馈包
# ═══════════════════════════════════════════════════════════════════════════════


class AttemptInfo(BaseModel):
    """重写/润色次数追踪。"""

    loop_count: int = Field(default=0, ge=0, description="重写次数(score过低触发)")
    refine_attempts: int = Field(default=0, ge=0, description="润色次数(score中等触发)")
    max_rewrite: int = Field(default=2, ge=0, description="最大重写次数")
    max_refine: int = Field(default=2, ge=0, description="最大润色次数")

    @property
    def rewrite_exhausted(self) -> bool:
        """重写次数是否用尽。"""
        return self.loop_count >= self.max_rewrite

    @property
    def refine_exhausted(self) -> bool:
        """润色次数是否用尽。"""
        return self.refine_attempts >= self.max_refine


class FeedbackBundle(BaseModel):
    """统一反馈包 — 同时服务 refiner 和 writer。

    消除 refiner/writer 反馈注入不一致问题。
    所有反馈源汇聚到一个结构，翻修节点统一消费。
    """

    # === 评分概要 ===
    score_summary: str = Field(default="", description="评分概要行")

    # === 核心问题 ===
    review_comments: str = Field(default="", description="LLM四维评审意见")
    ai_style_fix: str = Field(default="", description="AI味修改建议")
    lao_shu_chong_fix: str = Field(default="", description="老书虫修改建议")

    # === 结构化信号 ===
    toxic_points: list[str] = Field(
        default_factory=list, description="毒点类型列表(必须规避)"
    )
    shuangdian_points: list[str] = Field(
        default_factory=list, description="爽点类型列表(保留增强)"
    )

    # === 辩论产出 ===
    debate_issues: list[str] = Field(default_factory=list, description="辩论发现的问题")
    debate_strengths: list[str] = Field(
        default_factory=list, description="辩论认可的亮点(必须保留)"
    )
    debate_suggestions: str = Field(default="", description="辩论改进建议")
    debate_transcript: str = Field(default="", description="完整辩论记录(供深度参考)")

    # === 程序化指标摘要 ===
    ai_style_metrics_brief: str = Field(default="", description="AI味8维指标精简摘要")

    # === 跨章反馈 ===
    cross_chapter_brief: str = Field(default="", description="跨章一致性摘要")
    cross_chapter_issues: list[str] = Field(
        default_factory=list, description="跨章问题列表"
    )

    def to_unified_body(self) -> str:
        """生成统一的反馈正文 — refiner 和 writer 共用同一格式。"""
        parts: list[str] = []

        # 评分概要
        if self.score_summary:
            parts.append(f"【评分概要】\n{self.score_summary}")

        # 核心问题
        if self.review_comments:
            parts.append(f"【审核意见】\n{self.review_comments}")

        # 程序化指标
        if self.ai_style_metrics_brief:
            parts.append(f"【程序化指标】\n{self.ai_style_metrics_brief}")

        # AI味修改建议
        if self.ai_style_fix:
            parts.append(f"【AI味修改建议】\n{self.ai_style_fix}")

        # 老书虫修改建议
        if self.lao_shu_chong_fix:
            parts.append(f"【老书虫修改建议】\n{self.lao_shu_chong_fix}")

        # 毒点
        if self.toxic_points:
            parts.append(f"【毒点（必须规避）】\n{', '.join(self.toxic_points)}")

        # 爽点
        if self.shuangdian_points:
            parts.append(f"【爽点（保留增强）】\n{', '.join(self.shuangdian_points)}")

        # 跨章反馈
        if self.cross_chapter_brief:
            parts.append(f"【跨章一致性指导】\n{self.cross_chapter_brief}")
        if self.cross_chapter_issues:
            issues_str = "\n".join(f"  - {i}" for i in self.cross_chapter_issues)
            parts.append(f"【跨章问题】\n{issues_str}")

        # 辩论产出
        if self.debate_issues:
            issues_str = "\n".join(f"  - {i}" for i in self.debate_issues)
            parts.append(f"【辩论发现问题】\n{issues_str}")
        if self.debate_strengths:
            strengths_str = "\n".join(f"  - {s}" for s in self.debate_strengths)
            parts.append(f"【辩论认可亮点（必须保留）】\n{strengths_str}")
        if self.debate_suggestions:
            parts.append(f"【辩论改进建议】\n{self.debate_suggestions}")
        if self.debate_transcript:
            parts.append(f"【完整辩论记录（供深度参考）】\n{self.debate_transcript}")

        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  统一评审决议 — 核心契约
# ═══════════════════════════════════════════════════════════════════════════════


class VerdictLevel(str, Enum):
    """三级评审决议 — 替代 12 分支路由。"""

    PASS = "pass"
    REFINE = "refine"
    REWRITE = "rewrite"


class VerdictResult(BaseModel):
    """统一评审决议 — 评分模块的唯一输出契约。

    消除 quality_score / composite_score 三源问题：
    所有评分字段只在 VerdictResult 中定义一次。
    """

    # === 决议 ===
    level: VerdictLevel = Field(default=VerdictLevel.REWRITE, description="三级决议")
    passed: bool = Field(default=False, description="level == PASS")

    # === 评分（融合计算） ===
    final_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="融合后最终评分(0-100)"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="LLM四维原始分"
    )
    programmatic_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="程序化融合分(0-1)"
    )
    cross_chapter_consistency: float = Field(
        default=75.0, ge=0.0, le=100.0, description="跨章一致性分(0-100)"
    )
    debate_penalty: float = Field(
        default=0.0, ge=0.0, le=30.0, description="辩论问题惩罚(0-30)"
    )

    # === 子评分明细 ===
    ai_style_score: float = Field(default=0.0, ge=0.0, le=1.0, description="AI味原始分")
    lao_shu_chong_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="老书虫原始分"
    )

    # === v7.1: LLM 语义分析追踪 ===
    llm_semantic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="LLM 老书虫语义评分 (0-100)，失败时=0",
    )
    llm_human_like_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="LLM AI味人类相似度 (0-100)，失败时=0",
    )
    llm_attraction_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="LLM 吸引力专家团队评分 (0-100)，失败时=0",
    )
    llm_attraction_fix: str = Field(
        default="",
        description="LLM 吸引力专家团队修改建议（LLMAttractionResult.fix），失败时为空串",
    )
    llm_severe_toxic_detected: bool = Field(
        default=False,
        description="LLM 检测到严重毒点",
    )
    llm_implicit_toxic_found: bool = Field(
        default=False,
        description="LLM 检测到隐式毒点（程序化无法发现）",
    )
    llm_analysis_failed: bool = Field(
        default=False,
        description="LLM 语义分析是否失败",
    )

    # === 反馈产出 ===
    feedback: FeedbackBundle = Field(
        default_factory=FeedbackBundle, description="统一反馈包"
    )

    # === 元信息 ===
    is_short_text: bool = Field(default=False, description="短文本标记")
    is_calibrated: bool = Field(default=False, description="是否经过校准")
    calibration_reason: str = Field(default="", description="校准原因")
    attempt_info: AttemptInfo = Field(
        default_factory=AttemptInfo, description="次数追踪"
    )
    has_severe_toxic: bool = Field(default=False, description="是否有严重毒点")

    def to_state_dict(self) -> dict[str, Any]:
        """转换为可写入 LangGraph state 的扁平字典。

        v7.1: 新增 LLM 分析追踪字段。
        """
        return {
            # 唯一权威结构
            "verdict_result": self.model_dump(),
            # 评分字段
            "quality_score": self.quality_score,
            "final_score": self.final_score,
            "ai_style_score": self.ai_style_score,
            "lao_shu_chong_score": self.lao_shu_chong_score,
            "programmatic_score": self.programmatic_score,
            "is_short_text": self.is_short_text,
            # v7.1: LLM 语义分析追踪
            "llm_semantic_score": self.llm_semantic_score,
            "llm_human_like_score": self.llm_human_like_score,
            "llm_attraction_score": self.llm_attraction_score,
            "llm_attraction_fix": self.llm_attraction_fix,
            "llm_severe_toxic_detected": self.llm_severe_toxic_detected,
            "llm_implicit_toxic_found": self.llm_implicit_toxic_found,
            "llm_analysis_failed": self.llm_analysis_failed,
            # 反馈字段（供旧代码消费）
            "review_comments": self.feedback.review_comments,
            "ai_style_fix": self.feedback.ai_style_fix,
            "lao_shu_chong_fix": self.feedback.lao_shu_chong_fix,
            "toxic_points": self.feedback.toxic_points,
            "shuangdian_points": self.feedback.shuangdian_points,
            "debate_issues": self.feedback.debate_issues,
            "debate_strengths": self.feedback.debate_strengths,
            "debate_suggestions": self.feedback.debate_suggestions,
            "debate_transcript": self.feedback.debate_transcript,
        }
