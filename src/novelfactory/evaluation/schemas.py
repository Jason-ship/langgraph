"""Evaluation schemas — 评分模块核心数据结构 (v6.3)。

这是评分体系的唯一契约层。所有评分源（程序化 + LLM + 辩论）的产出
最终汇聚到 VerdictResult，由 VerdictRouter 和 FeedbackBuilder 消费。

设计原则：
    1. 单一权威 — VerdictResult 是评分的唯一来源，消除三源问题
    2. 程序化只做传感器 — CrossChapterSignals / ProgrammaticReport 只产出客观数据，不做判断
    3. LLM 做裁判 — 四维评分和辩论负责语义判断
    4. 反馈统一 — FeedbackBundle 同时服务 refiner 和 writer，消除注入不一致
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from novelfactory.schemas.review_schemas import FourDimScores

# ═══════════════════════════════════════════════════════════════════════════════
#  程序化传感器产出
# ═══════════════════════════════════════════════════════════════════════════════


class AIStyleMetricsBrief(BaseModel):
    """AI味 8 维指标摘要 — 程序化传感器产出，客观数据。"""

    repetition_ngram: float = Field(
        default=0.0, ge=0.0, le=1.0, description="N-gram重复率(0-1)越高越AI"
    )
    sentence_length_variance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="句长波动(0-1)越高越AI"
    )
    lexical_diversity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="词汇多样性(0-1)越高越AI"
    )
    cliche_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="模板化表达比例(0-1)越高越AI"
    )
    punctuation_rhythm: float = Field(
        default=0.0, ge=0.0, le=1.0, description="标点节奏变异(0-1)越高越AI"
    )
    dialogue_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="对白比例偏离(0-1)越高越AI"
    )
    sensory_emotion_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="感官情绪词密度(0-1)越高越AI"
    )
    semantic_smoothness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="语义平滑度(0-1)越高越AI"
    )

    def to_brief_string(self) -> str:
        """生成给翻修 prompt 的精简摘要。"""
        parts: list[str] = []
        if self.repetition_ngram > 0.4:
            parts.append(f"N-gram重复{self.repetition_ngram:.2f}偏高")
        if self.sentence_length_variance > 0.5:
            parts.append(f"句长波动{self.sentence_length_variance:.2f}趋同")
        if self.lexical_diversity > 0.4:
            parts.append(f"词汇多样性{self.lexical_diversity:.2f}偏低")
        if self.cliche_ratio > 0.3:
            parts.append(f"模板化{self.cliche_ratio:.2f}偏高")
        if self.punctuation_rhythm > 0.5:
            parts.append(f"标点节奏{self.punctuation_rhythm:.2f}规律")
        if not parts:
            return "各项指标正常"
        return " | ".join(parts)


class ToxicDetail(BaseModel):
    """毒点详情。"""

    type: str = Field(description="毒点类型，如 NTR / NUE_ZHU / SHENGMU")
    description: str = Field(description="毒点描述")
    severity: str = Field(
        default="medium", description="严重程度: extreme/high/medium/low"
    )
    weight: float = Field(default=0.0, description="扣分权重")


class ShuangdianDetail(BaseModel):
    """爽点详情。"""

    type: str = Field(description="爽点类型，如 打脸 / 装逼 / 升级")
    description: str = Field(description="爽点描述")
    weight: float = Field(default=0.0, description="加分权重")


class ProgrammaticReport(BaseModel):
    """程序化分析统一报告 — AI味检测 + 老书虫评审的融合产出。

    只产出客观数据和检测结果，不做好坏判断。
    判断交给 LLM（四维评分 + 辩论）。
    """

    # === AI味检测 ===
    ai_style_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="AI味指数(0-1)越低越好"
    )
    ai_style_metrics: AIStyleMetricsBrief = Field(
        default_factory=AIStyleMetricsBrief, description="8维原始指标"
    )
    ai_style_issues: list[str] = Field(default_factory=list, description="具体问题列表")
    ai_style_fix: str = Field(default="", description="修改建议")

    # === 老书虫评审 ===
    lao_shu_chong_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="老书虫评分(0-100)越高越好"
    )
    toxic_points: list[ToxicDetail] = Field(
        default_factory=list, description="毒点详情"
    )
    shuangdian_points: list[ShuangdianDetail] = Field(
        default_factory=list, description="爽点详情"
    )
    lao_shu_chong_fix: str = Field(default="", description="修改建议")
    verdict_text: str = Field(default="", description="优秀/可读/需改/弃书")

    # === 融合信号 ===
    has_severe_toxic: bool = Field(
        default=False, description="是否有严重毒点(NTR/虐主/圣母)"
    )
    severe_toxic_types: list[str] = Field(
        default_factory=list, description="严重毒点类型列表"
    )
    is_short_text: bool = Field(
        default=False, description="短文本标记(程序化分析无法执行)"
    )

    @property
    def programmatic_score(self) -> float:
        """程序化融合分 (0-1) — 用于 VerdictEngine 融合计算。"""
        return (self.lao_shu_chong_score / 100.0) * (1.0 - self.ai_style_score)

    def to_debate_briefing(self) -> str:
        """生成给辩论 Agent 的程序化分析摘要。"""
        if self.is_short_text:
            return "（文本过短，程序化分析无法执行，请基于文本直接分析）"

        parts: list[str] = []

        # AI味摘要
        parts.append(
            f"AI味指数: {self.ai_style_score:.2f}({'合格' if self.ai_style_score <= 0.3 else '不合格'})"
        )
        metrics_str = self.ai_style_metrics.to_brief_string()
        if metrics_str != "各项指标正常":
            parts.append(f"  指标详情: {metrics_str}")
        if self.ai_style_issues:
            parts.append(f"  AI味问题: {'; '.join(self.ai_style_issues[:3])}")

        # 老书虫摘要
        parts.append(
            f"老书虫评分: {self.lao_shu_chong_score:.0f}/100({self.verdict_text})"
        )
        if self.toxic_points:
            toxic_strs = [f"{t.type}({t.severity})" for t in self.toxic_points]
            parts.append(f"  毒点: {', '.join(toxic_strs)}")
        if self.shuangdian_points:
            sd_strs = [s.type for s in self.shuangdian_points]
            parts.append(f"  爽点: {', '.join(sd_strs)}")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  跨章传感器产出
# ═══════════════════════════════════════════════════════════════════════════════


class CrossChapterSignals(BaseModel):
    """跨章一致性信号 — 纯代码传感器产出。

    只产出客观数据（"句长+40%"），不做判断（"声音漂移"）。
    判断交给 LLM（注入四维评分 prompt 和辩论 prompt）。
    """

    # === 角色声音信号 ===
    cur_avg_sentence_length: float = Field(default=0.0, description="本章平均句长")
    prev_avg_sentence_length: float = Field(default=0.0, description="前文平均句长")
    sentence_length_delta: float = Field(
        default=0.0, description="句长变化百分比(如+40=-40%)"
    )

    # === 节奏信号 ===
    cur_dialogue_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="本章对话密度"
    )
    prev_dialogue_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="前文对话密度"
    )
    dialogue_density_delta: float = Field(default=0.0, description="对话密度变化")
    cur_action_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="本章动作描写密度"
    )
    prev_action_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="前文动作描写密度"
    )

    # === 文风信号 ===
    cur_vocab_richness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="本章词汇丰富度"
    )
    prev_vocab_richness: float = Field(
        default=0.0, ge=0.0, le=1.0, description="前文词汇丰富度"
    )
    style_drift_delta: float = Field(default=0.0, description="文风偏移度")

    # === 伏笔信号 ===
    potential_new_foreshadowing: list[str] = Field(
        default_factory=list, description="本章潜在新伏笔"
    )
    unexplained_elements: list[str] = Field(
        default_factory=list, description="前文未解释元素"
    )

    # === 情节连贯信号 ===
    setting_keywords: list[str] = Field(
        default_factory=list, description="前文关键设定词"
    )
    chapter_keywords: list[str] = Field(default_factory=list, description="本章关键词")
    keyword_overlap_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="关键词重叠率"
    )

    # === 元信息 ===
    has_prev_context: bool = Field(default=False, description="是否有前文上下文")

    # v7.3: 物品状态不一致问题（SCORE UC Berkeley 2025）
    item_state_issues: list[str] = Field(
        default_factory=list,
        description="物品状态不一致问题（如 destroyed 物品重新出现）",
    )

    chapter_index: int = Field(default=1, description="当前章节序号")

    def to_debate_briefing(self) -> str:
        """生成给辩论/四维 prompt 的跨章信号摘要。"""
        if not self.has_prev_context:
            return "（无前文上下文，跨章分析跳过）"

        parts: list[str] = []

        # 句长变化
        if abs(self.sentence_length_delta) > 0.2:
            direction = "增加" if self.sentence_length_delta > 0 else "减少"
            parts.append(
                f"句长变化: 本章平均{self.cur_avg_sentence_length:.0f}字 vs 前文{self.prev_avg_sentence_length:.0f}字"
                f"({direction}{abs(self.sentence_length_delta) * 100:.0f}%)"
            )

        # 对话密度变化
        if abs(self.dialogue_density_delta) > 0.15:
            parts.append(
                f"对话密度: 本章{self.cur_dialogue_density:.2f} vs 前文{self.prev_dialogue_density:.2f}"
                f"(变化{self.dialogue_density_delta * 100:+.0f}%)"
            )

        # 文风偏移
        if abs(self.style_drift_delta) > 0.2:
            parts.append(
                f"词汇丰富度: 本章{self.cur_vocab_richness:.2f} vs 前文{self.prev_vocab_richness:.2f}"
                f"(偏移{abs(self.style_drift_delta) * 100:.0f}%)"
            )

        # 伏笔
        if self.unexplained_elements:
            parts.append(f"前文未解释元素: {', '.join(self.unexplained_elements[:5])}")
        if self.potential_new_foreshadowing:
            parts.append(
                f"本章潜在新伏笔: {', '.join(self.potential_new_foreshadowing[:3])}"
            )

        # v7.3: 物品状态不一致
        if self.item_state_issues:
            parts.append(f"物品状态问题: {'; '.join(self.item_state_issues[:3])}")

        # 关键词重叠
        if self.keyword_overlap_ratio < 0.3:
            parts.append(
                f"关键词重叠率: {self.keyword_overlap_ratio:.2f}(偏低，可能与前文关联弱)"
            )

        if not parts:
            return "跨章信号正常，无显著偏移"
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  辩论产出
# ═══════════════════════════════════════════════════════════════════════════════


class PerspectiveReview(BaseModel):
    """单视角评审结果（编辑或读者的首轮评审）。"""

    review_comments: str = Field(default="", description="综合评审意见")
    issues: list[str] = Field(default_factory=list, description="问题列表")
    strengths: list[str] = Field(default_factory=list, description="亮点列表")
    suggestions: str = Field(default="", description="改进建议")


class Rebuttal(BaseModel):
    """单轮反驳结果。"""

    rebuttal_comments: str = Field(default="", description="辩论意见正文")
    new_issues: list[str] = Field(default_factory=list, description="本轮新增问题")
    revised_suggestions: str = Field(default="", description="修正建议")
    has_dissent: bool = Field(default=True, description="是否仍有异议")


class DebateReport(BaseModel):
    """多轮辩论统一报告 — 编辑↔读者知情辩论的融合产出。"""

    # === 首轮评审 ===
    editor_review: PerspectiveReview = Field(
        default_factory=PerspectiveReview, description="编辑视角首轮评审"
    )
    reader_review: PerspectiveReview = Field(
        default_factory=PerspectiveReview, description="读者视角首轮评审"
    )

    # === 多轮辩论 ===
    editor_rebuttals: list[Rebuttal] = Field(
        default_factory=list, description="编辑反驳列表"
    )
    reader_rebuttals: list[Rebuttal] = Field(
        default_factory=list, description="读者反驳列表"
    )
    debate_rounds: int = Field(default=0, description="实际辩论轮次")
    debate_transcript: str = Field(default="", description="完整辩论记录")

    # === 融合产出 ===
    merged_issues: list[str] = Field(
        default_factory=list, description="全部轮次去重合并的问题"
    )
    merged_strengths: list[str] = Field(
        default_factory=list, description="首轮亮点(去重)"
    )
    merged_suggestions: str = Field(default="", description="全部轮次建议合并")
    convergence_achieved: bool = Field(default=False, description="是否提前收敛")
    debate_failed: bool = Field(default=False, description="辩论是否完全失败(降级)")

    @property
    def severity_weight(self) -> float:
        """辩论发现的问题严重度权重 — 回馈评分。

        v7.6-fix: 改用 constants 中的容差，降低辩论惩罚力度。
        辩论本质是对章节的深入分析，发现问题是正常的，不应过度惩罚。
        v7.8-fix: 动态 CAP — 辩论收敛时惩罚可信度高，CAP 用满；
                 辩论未收敛（双方各执一词）时惩罚可信度低，CAP 折半。
        """
        from novelfactory.config.constants import (
            VERDICT_DEBATE_PENALTY_CAP as _CAP,
        )
        from novelfactory.config.constants import (
            VERDICT_DEBATE_PENALTY_PER_ISSUE as _PER_ISSUE,
        )
        from novelfactory.config.constants import (
            VERDICT_DEBATE_PENALTY_PER_SEVERE as _PER_SEVERE,
        )

        # 动态 CAP：收敛时完全信任，分歧时减半
        cap = _CAP if self.convergence_achieved else _CAP * 0.5

        base = len(self.merged_issues) * _PER_ISSUE
        # NOTE: 简单子串匹配。辩论输出为结构化 issue 描述，此处"严重"几乎总是
        # severity 语义。若辩论格式变化引入非 severity 用法，需升级检测逻辑。
        severe = sum(1 for i in self.merged_issues if "严重" in i or "毒点" in i) * _PER_SEVERE
        return min(cap, base + severe)


# ═══════════════════════════════════════════════════════════════════════════════
#  四维 LLM 评分产出
# ═══════════════════════════════════════════════════════════════════════════════


class EvidenceItem(BaseModel):
    """证据链条目 — 参考 ConStory-Checker (微软 2026) 的 evidence chain 设计。

    每个扣分点都应附带一对矛盾文本引用 + 推理说明。
    """

    issue: str = Field(default="", description="发现的问题")
    span_a: str = Field(default="", description="矛盾片段A（原文引用含段落编号）")
    span_b: str = Field(default="", description="矛盾片段B（原文引用含段落编号）")
    reasoning: str = Field(default="", description="为什么这两个片段矛盾")
    error_type: str = Field(
        default="",
        description="错误类型: 时间矛盾/角色矛盾/世界观违反/细节不一致/风格偏离",
    )


class FourDimReviewResult(BaseModel):
    """四维 LLM 评分结果 — 含跨章一致性维度。

    v7.3: 新增 evidence_chain 字段，每个扣分点附带证据链。
    """

    quality_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="四维总分(0-100)"
    )
    four_dim_scores: FourDimScores = Field(
        default_factory=FourDimScores, description="四维分项"
    )
    review_comments: str = Field(default="", description="评审综合意见")
    cross_chapter_consistency: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="跨章一致性评分(0-100)，由LLM基于跨章信号判断",
    )
    cross_chapter_issues: list[str] = Field(
        default_factory=list, description="跨章一致性问题"
    )
    evidence_chain: list[EvidenceItem] = Field(
        default_factory=list, description="证据链列表（每个扣分点附带矛盾文本引用）"
    )
    failed: bool = Field(default=False, description="LLM评分是否失败(降级)")


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
    four_dim_scores: FourDimScores = Field(
        default_factory=FourDimScores, description="四维分项"
    )
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
