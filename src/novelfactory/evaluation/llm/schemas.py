"""LLM 评价分析的结构化输出 Schema。

参考 WritingBench (NeurIPS 2025) 的 criteria-aware scoring 和
WebNovelBench (ACL 2025) 的 8-dimension narrative quality evaluation 范式。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
#  LLM 老书虫评审
# ═══════════════════════════════════════════════════════════════════════════════


class LLMToxicDetail(BaseModel):
    """LLM 检测到的毒点详情（语义级，非关键词匹配）。"""

    type: str = Field(
        description="毒点类型: NTR / SHENGMU / PROTAGONIST_STUPID / NUE_ZHU / "
        "POWER_BREAK / ANTAGONIST_STUPID / WATER_CONTENT / "
        "SENTIMENTAL_TORTURE / MORAL_WRONG / LOGIC_HOLE / PLOT_ARMOR / OTHER"
    )
    severity: str = Field(description="严重程度: extreme / high / medium / low")
    description: str = Field(
        description="语义化描述：为什么这是毒点？具体情节上下文是什么？"
    )
    quote: str = Field(default="", description="原文引用（相关段落，50-150字）")
    narrative_context: str = Field(
        default="", description="叙事上下文分析：是作者刻意为之还是 AI 无意识输出？"
    )


class LLMShuangdianDetail(BaseModel):
    """LLM 检测到的爽点详情（语义级）。"""

    type: str = Field(
        description="爽点类型: 打脸 / 装逼 / 逆袭 / 升级 / 感情 / 悬念 / "
        "智斗 / 热血 / 感动 / 幽默 / 其他"
    )
    impact_rating: float = Field(
        ge=1.0,
        le=10.0,
        description="爽点冲击力评分 (1-10)：读者情绪调动强度",
    )
    description: str = Field(description="语义化描述：这个爽点为什么有效？")
    quote: str = Field(default="", description="原文引用（相关段落，50-150字）")


class LLMOldReaderResult(BaseModel):
    """LLM 老书虫评审结果。

    融合 EQ-Bench Creative Writing v3 的多视角评估理念：
    从老书虫（资深读者）的视角看"好不好看"、"毒不毒"、"爽不爽"。
    """

    # === 语义评分 ===
    semantic_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="LLM 语义级综合评分 (0-100)，基于对毒点/爽点/节奏/代入感的深度理解",
    )
    reading_absorption: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="阅读代入感 (0-10)：能否让读者沉浸？是否有「停不下来」的感觉？",
    )
    logic_consistency: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="逻辑自洽性 (0-10)：情节逻辑/角色行为/世界观一致性",
    )
    freshness: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="新颖感 (0-10)：是否有套路化模板感？创意是否老套？",
    )

    # === 毒点分析 ===
    toxic_points: list[LLMToxicDetail] = Field(
        default_factory=list,
        description="语义级毒点列表（程序化检测不到的隐式毒点）",
    )
    has_severe_toxic: bool = Field(
        default=False,
        description="是否有严重毒点（NTR/虐主/圣母等）",
    )
    implicit_toxic_found: bool = Field(
        default=False,
        description="是否发现了程序化检测不到的隐式毒点",
    )

    # === 爽点分析 ===
    shuangdian_points: list[LLMShuangdianDetail] = Field(
        default_factory=list,
        description="语义级爽点列表",
    )
    emotional_arc_detected: bool = Field(
        default=False,
        description="是否检测到情感弧线（情绪起伏/情感变化）",
    )

    # === 针对翻修的反馈 ===
    strengths: list[str] = Field(
        default_factory=list,
        description="本章写作亮点（附具体段落实例）",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="本章核心问题（附关键段落实例）",
    )
    concrete_suggestions: str = Field(
        default="",
        description="具体的、可操作的修改建议（而非泛泛的'增加爽点'）",
    )

    # === 元信息 ===
    failed: bool = Field(
        default=False,
        description="LLM 分析是否失败（降级标记）",
    )
    failure_reason: str = Field(default="", description="失败原因")


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM AI味语义分析
# ═══════════════════════════════════════════════════════════════════════════════


class LLMAIStyleIssue(BaseModel):
    """LLM 检测到的 AI 味问题（语义级）。"""

    category: str = Field(
        description="问题类别: 过渡生硬 / 描写模板化 / 对话不自然 / "
        "情绪虚假 / 信息过载 / 句式重复 / 逻辑过顺 / 引用过多 / 说明过多 / 其他"
    )
    description: str = Field(description="语义化描述：这段文本为何像 AI 写的？")
    quote: str = Field(default="", description="原文引用")
    replacement_suggestion: str = Field(
        default="", description="具体的改写建议（不少于50字），展示如何让这段文本更自然"
    )


class LLMAIStyleResult(BaseModel):
    """LLM AI 味语义分析结果。

    程序化 AI 味检测只能抓统计特征（重复率/句长/模板词），
    LLM 可以理解"为什么这段读起来像 AI 写的"——关注语义层面的自然度。
    """

    # === 语义评分 ===
    human_like_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="人类写作相似度 (0-100)：读起来像人的程度，越高越好",
    )
    naturalness: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="自然度 (0-10)：语言是否自然流畅，有无 AI 腔调",
    )
    voice_consistency: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="叙事声音一致性 (0-10)：POV 视角是否统一，叙述语气是否一致",
    )
    emotional_authenticity: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="情绪真实感 (0-10)：角色情绪是否真实可信，有无「演」的感觉",
    )

    # === 问题列表 ===
    semantic_issues: list[LLMAIStyleIssue] = Field(
        default_factory=list,
        description="语义级 AI 味问题列表",
    )
    has_obvious_ai: bool = Field(
        default=False,
        description="是否有明显 AI 生成痕迹",
    )

    # === 总体评估 ===
    summary: str = Field(
        default="",
        description="AI味总体评估（几句话概括，供翻修节点使用）",
    )

    # === 元信息 ===
    failed: bool = Field(
        default=False,
        description="LLM 分析是否失败（降级标记）",
    )
    failure_reason: str = Field(default="", description="失败原因")
