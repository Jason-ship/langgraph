"""Pydantic structured output schemas for review results.

Replaces fragile dict-based review result parsing with typed, validated schemas.
Each field's ``description`` doubles as the model's output instruction,
following the TradingAgents pattern (schemas.py).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FourDimScores(BaseModel):
    """四维评分（文学性/结构/角色/节奏）。"""

    literary: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="文学性评分（0-30）：语言美感、描写细腻度、文笔流畅度",
    )
    structure: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="结构评分（0-25）：章内结构、悬念设置、情节推进",
    )
    character: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="角色评分（0-20）：角色一致性、性格刻画、对话自然度",
    )
    pacing: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="节奏评分（0-15）：叙事节奏、信息密度、阅读体验",
    )

    @property
    def total(self) -> float:
        return self.literary + self.structure + self.character + self.pacing


class ChapterReviewResult(BaseModel):
    """章节评审结构化结果 — 替代 review_result dict。"""

    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="综合质量评分（0-100），四维评分之和",
    )
    four_dim_scores: FourDimScores = Field(
        default_factory=FourDimScores,
        description="四维分项评分明细",
    )
    composite_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="综合指标 = 老书虫分/100 × (1 − AI味指数)，0-1之间",
    )
    ai_style_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="AI味指数（0-1），越低越好，≤0.4为合格",
    )
    lao_shu_chong_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="老书虫评分（0-100），≥65为合格",
    )
    passed: bool = Field(
        default=False,
        description="是否通过评审：quality_score >= 题材阈值 且 composite_score >= 题材阈值",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="发现的问题列表（如有）",
    )
    suggestions: str = Field(
        default="",
        description="改进建议（自由文本，供 refiner/writer 使用）",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="本章亮点",
    )
    review_comments: str = Field(
        default="",
        description="评审综合意见（自由文本），供下游展示与反馈构建",
    )
    needs_refine: bool = Field(
        default=False,
        description="是否需要润色（passed=False 时为 True）",
    )
    is_short_text: bool = Field(
        default=False,
        description="短文本标记 — 程序化子Agent均无法分析时为 True",
    )
    guide_references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="写作指南引用片段列表",
    )
    ai_style_fix: str = Field(
        default="",
        description="AI味修改建议（自由文本）",
    )
    lao_shu_chong_fix: str = Field(
        default="",
        description="老书虫视角修改建议（自由文本）",
    )
    toxic_points: list[str] = Field(
        default_factory=list,
        description="毒点列表（令读者反感的元素）",
    )
    shuangdian_points: list[str] = Field(
        default_factory=list,
        description="爽点列表（吸引读者的亮点）",
    )
    debate_issues: list[str] = Field(
        default_factory=list,
        description="编辑↔读者辩论合并后的问题列表",
    )
    debate_strengths: list[str] = Field(
        default_factory=list,
        description="编辑↔读者辩论合并后的亮点列表",
    )
    debate_suggestions: str = Field(
        default="",
        description="编辑↔读者辩论合并后的修改建议",
    )


class AIReviewResult(BaseModel):
    """AI味检测结构化结果。"""

    ai_style_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="AI生成痕迹指数（0-1），越低越好",
    )
    ai_patterns_detected: list[str] = Field(
        default_factory=list,
        description="检测到的AI写作模式列表",
    )
    human_like_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="人类写作风格占比（0-1），越高越好",
    )
    detailed_feedback: str = Field(
        default="",
        description="详细反馈（自由文本）",
    )


class ScenePlan(BaseModel):
    """单场景计划 —— chapter_planner 产出的结构化场景描述。"""

    scene_number: int = Field(default=1, ge=1, le=10, description="场景序号（1-based）")
    purpose: str = Field(
        default="", description="本场景的核心目的（如'揭示反派动机''推进感情线'）"
    )
    location: str = Field(default="", description="场景发生地点")
    pov_character: str = Field(default="", description="本场景的主视角角色")
    characters: list[str] = Field(default_factory=list, description="出场角色列表")
    key_content: str = Field(
        default="",
        description="关键对话/动作/情节发展点，50-100字描述",
    )
    sensory_focus: str = Field(
        default="",
        description="感官描写的侧重方向（如'视觉：废墟景象''触觉：冰冷雨水'）",
    )
    target_length_ratio: float = Field(
        default=0.2,
        ge=0.05,
        le=0.6,
        description="本场景占全章字数比例（0.05-0.6）",
    )


class ChapterPlan(BaseModel):
    """章节写作计划 —— 由 chapter_planner 节点生成，writer 按计划执行。

    在 context_builder 之后、chapter_writer 之前由 planner agent 生成。
    如果本章是 rewrite（loop_count > 0），planner 会收到上一轮的评审反馈，
    产出修正后的计划。
    """

    chapter_number: int = Field(default=1, ge=1, description="当前章节号")
    title: str = Field(default="", description="章节标题")
    core_plot_point: str = Field(
        default="",
        description="本章最核心的情节点（一句话），所有场景围绕此展开",
    )
    pov_character: str = Field(default="", description="本章主视角角色")
    characters_involved: list[str] = Field(
        default_factory=list, description="本章出场角色（含 POV）"
    )
    scenes: list[ScenePlan] = Field(
        default_factory=list, description="场景列表（3-5个场景）"
    )
    emotional_arc: str = Field(
        default="",
        description="本章情感弧线（如'从希望→绝望→决绝'）",
    )
    foreshadowing_plant: list[str] = Field(
        default_factory=list, description="本章需埋设的伏笔"
    )
    foreshadowing_resolve: list[str] = Field(
        default_factory=list, description="本章需回收的伏笔"
    )
    target_word_count: int = Field(
        default=3000, ge=500, le=10000, description="本章目标字数"
    )
    cliffhanger: str = Field(default="", description="本章结尾悬念/勾子")
    review_feedback: str = Field(
        default="",
        description="上一轮评审反馈（仅 rewrite 路径有值，planner 需针对性修正计划）",
    )


class OutlineReviewResult(BaseModel):
    """大纲/卷结构评审结构化结果。"""

    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="大纲整体评分（0-100）",
    )
    plot_coherence: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="情节连贯性（0-25）",
    )
    foreshadowing_setup: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="伏笔设计（0-25）",
    )
    pacing_curve: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="节奏曲线（0-25）",
    )
    character_arc_alignment: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="角色弧线对齐（0-25）",
    )
    volume_boundary_suggestions: list[str] = Field(
        default_factory=list,
        description="卷边界调整建议",
    )
    passed: bool = Field(default=False, description="是否通过")
    issues: list[str] = Field(default_factory=list, description="发现的问题")
