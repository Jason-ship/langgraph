"""Narrative Codec Engine 核心 Schema 定义。

本模块定义编解码引擎中使用的所有结构化数据模型，涵盖：
- WorldStateV1 核心类型（Shadow-Loom arXiv 2026）
- STAC 叙事角色四分类 + 11种因果连接（Beyond LLMs ACL 2025）
- Expert Index 7维语言学特征 + 15维 one-hot（Beyond LLMs §3.3）
- 结构化协议（LLM×MapReduce ACL 2025 §3.2）
- ConStory-Bench 5大类错误类型（ConStory-Bench arXiv 2026 §2.2）
- 情绪曲线 + 因果图输出

参考论文:
- Shadow-Loom: arXiv 2026, WorldStateV1 + 双时间轴 + 因果推理
- Beyond LLMs: ACL 2025, STAC + Expert Index
- LLM×MapReduce: ACL 2025, 结构化协议 + 置信度校准
- ConStory-Bench: arXiv 2026, 5×19 一致性错误分类
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════
# 1. WorldStateV1 核心类型
# ═══════════════════════════════════════════════════════════════════


class TraitVector(BaseModel):
    """特质向量 — 角色/世界特质的值+惯性+噪声+证据强度。

    参考: Shadow-Loom arXiv 2026 WorldStateV1 §3.1
    """

    value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="特质当前水平 [0.0, 1.0]",
    )
    inertia: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="惯性系数 [0.0, 1.0]，改变所需冲击力大小；1.0=永久不变",
    )
    exogenous_noise: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="外生噪声项，模拟外部环境引入的随机波动",
    )
    evidence_strength: Literal["weak", "moderate", "strong"] = Field(
        default="weak",
        description="证据强度等级：weak（弱）/ moderate（中）/ strong（强）",
    )


class EntityNode(BaseModel):
    """实体节点 — 角色/叙事实体。

    参考: Shadow-Loom arXiv 2026 WorldStateV1 §3.2
    """

    entity_id: str = Field(
        default="",
        description="实体唯一标识符",
    )
    name: str = Field(
        default="",
        description="实体名称",
    )
    traits: dict[str, TraitVector] = Field(
        default_factory=dict,
        description="特质字典，key=特质名称，value=特质向量（含value/inertia/noise/strength）",
    )
    beliefs: dict[str, float] = Field(
        default_factory=dict,
        description="信念字典，key=信念目标/命题ID，value=置信度[0,1]",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="别名列表（如昵称、代号、不同翻译名）",
    )


class EventNode(BaseModel):
    """事件节点 — 含双时间轴（fabula / syuzhet）。

    参考: Shadow-Loom arXiv 2026 §4.1 双时间轴理论
    """

    event_id: str = Field(
        default="",
        description="事件唯一标识符",
    )
    label: str = Field(
        default="",
        description="事件标签/简述",
    )
    event_type: Literal["choice", "outcome", "revelation", "utterance"] = Field(
        default="outcome",
        description="事件类型：choice（选择）/ outcome（结果）/ revelation（揭示）/ utterance（话语）",
    )
    fabula_time: int = Field(
        default=0,
        ge=0,
        description="故事时间轴（编年顺序），fabula 时间戳",
    )
    syuzhet_index: int = Field(
        default=0,
        ge=0,
        description="叙事时间轴（呈现顺序），syuzhet 下标",
    )
    participants: list[str] = Field(
        default_factory=list,
        description="参与此事件的实体 ID 列表",
    )
    location_id: str | None = Field(
        default=None,
        description="事件发生地点 ID（可为 None）",
    )
    description: str = Field(
        default="",
        description="事件描述文本",
    )


class Scene(BaseModel):
    """场景单元 — 场景分割的产出。

    参考: Shadow-Loom arXiv 2026 §4.3
    """

    scene_id: int = Field(
        default=0,
        ge=0,
        description="场景编号（0-based）",
    )
    start_char: int = Field(
        default=0,
        ge=0,
        description="原文起始字符位置（包含）",
    )
    end_char: int = Field(
        default=0,
        ge=0,
        description="原文结束字符位置（不包含）",
    )
    characters: list[str] = Field(
        default_factory=list,
        description="此场景中出现的角色名列表",
    )
    location: str = Field(
        default="",
        description="场景发生地点",
    )
    time_marker: str = Field(
        default="",
        description="时间标记（如'傍晚''三天后'）",
    )
    text: str = Field(
        default="",
        description="场景原文内容",
    )


class Location(BaseModel):
    """地点节点 — 故事中出现的空间位置。

    参考: Shadow-Loom arXiv 2026 §3.3
    """

    location_id: str = Field(
        default="",
        description="地点唯一标识符",
    )
    name: str = Field(
        default="",
        description="地点名称",
    )
    description: str = Field(
        default="",
        description="地点描述",
    )


class RelationshipEdge(BaseModel):
    """关系边 — STAGE 6类叙事关系。

    参考: Shadow-Loom arXiv 2026 §3.4 STAGE 关系分类
    """

    source_id: str = Field(
        default="",
        description="源节点 ID",
    )
    target_id: str = Field(
        default="",
        description="目标节点 ID",
    )
    rel_type: Literal[
        "event_role",
        "social",
        "inter_event",
        "spatiotemporal",
        "object_related",
        "semantic",
    ] = Field(
        default="semantic",
        description="关系类型：event_role（事件角色）/ social（社会）/ inter_event（事件间）"
        "/ spatiotemporal（时空）/ object_related（物相关）/ semantic（语义）",
    )
    properties: dict[str, float] = Field(
        default_factory=dict,
        description="关系属性字典（如权重、强度等数值属性）",
    )


# ═══════════════════════════════════════════════════════════════════
# 2. STAC 枚举和连接
# ═══════════════════════════════════════════════════════════════════


class STACLabel(str, Enum):
    """STAC 四分类 — 叙事功能角色标注。

    参考: Beyond LLMs ACL 2025 §3.1 STAC Taxonomy
    """

    SITUATION = "situation"
    """背景上下文 — 场景设定、环境描述"""
    TASK = "task"
    """任务目标 — 需要完成的目标子目标"""
    ACTION = "action"
    """主动执行的动作 — 角色采取的行动"""
    CONSEQUENCE = "consequence"
    """状态改变结果 — 动作/事件造成的结果"""


class STACBond(str, Enum):
    """STAC 有效连接 — 11种叙事因果链。

    参考: Beyond LLMs ACL 2025 §3.2 Causal Bond Taxonomy
    """

    S_TO_S = "situation→situation"
    S_TO_T = "situation→task"
    S_TO_A = "situation→action"
    S_TO_C = "situation→consequence"
    T_TO_A = "task→action"
    T_TO_C = "task→consequence"
    A_TO_A = "action→action"
    A_TO_C = "action→consequence"
    C_TO_S = "consequence→situation"
    C_TO_T = "consequence→task"
    C_TO_C = "consequence→consequence"


# 有效 STAC 连接映射表 — (from_label, to_label) → STACBond。
# 只有此字典中定义的11种连接在叙事因果链中合法。
VALID_STAC_BONDS: dict[tuple[STACLabel, STACLabel], STACBond] = {
    (STACLabel.SITUATION, STACLabel.SITUATION): STACBond.S_TO_S,
    (STACLabel.SITUATION, STACLabel.TASK): STACBond.S_TO_T,
    (STACLabel.SITUATION, STACLabel.ACTION): STACBond.S_TO_A,
    (STACLabel.SITUATION, STACLabel.CONSEQUENCE): STACBond.S_TO_C,
    (STACLabel.TASK, STACLabel.ACTION): STACBond.T_TO_A,
    (STACLabel.TASK, STACLabel.CONSEQUENCE): STACBond.T_TO_C,
    (STACLabel.ACTION, STACLabel.ACTION): STACBond.A_TO_A,
    (STACLabel.ACTION, STACLabel.CONSEQUENCE): STACBond.A_TO_C,
    (STACLabel.CONSEQUENCE, STACLabel.SITUATION): STACBond.C_TO_S,
    (STACLabel.CONSEQUENCE, STACLabel.TASK): STACBond.C_TO_T,
    (STACLabel.CONSEQUENCE, STACLabel.CONSEQUENCE): STACBond.C_TO_C,
}


class STACLabeledSentence(BaseModel):
    """STAC 标注后的句子 — 分类 + 置信度。

    参考: Beyond LLMs ACL 2025 §3.3 Sentence Labeling Pipeline
    """

    text: str = Field(
        default="",
        description="原始句子文本",
    )
    refined_text: str = Field(
        default="",
        description="精炼后的句子文本（去停用词/规范化后）",
    )
    stac_label: STACLabel | None = Field(
        default=None,
        description="STAC 分类标签（None 表示未标注）",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="标注置信度 [0.0, 1.0]",
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Expert Index
# ═══════════════════════════════════════════════════════════════════


class ExpertIndexFeature(str, Enum):
    """Expert Index 7维语言学特征枚举。

    参考: Beyond LLMs ACL 2025 §3.3 Expert Index
    """

    GENERICITY = "genericity"
    """通指性 — specific（特指）vs generic（泛指）"""
    EVENTIVITY = "eventivity"
    """事件性 — dynamic（动态）vs stative（静态）"""
    BOUNDEDNESS = "boundedness"
    """有界性 — episodic（一次性）/ habitual（习惯性）/ static（静态）"""
    INITIATIVITY = "initiativity"
    """主动性 — initiate（主动触发）vs receive（被动接受）"""
    TIME_START = "time_start"
    """起始时间 — past（过去）vs current（当前）"""
    TIME_END = "time_end"
    """结束时间 — current（持续）vs future（未来结束）"""
    IMPACT = "impact"
    """影响性 — impactful（有影响）vs resolved（已解决）"""


class ExpertIndex(BaseModel):
    """Expert Index — 7维语言学特征 + 15维 one-hot 编码。

    参考文献: Beyond LLMs ACL 2025 §3.3
    7个语言学维度，每个维度取2-3个离散值，总计15维 one-hot 编码。

    编码映射:
        genericity:    specific=[1,0],        generic=[0,1]
        eventivity:    dynamic=[1,0],         stative=[0,1]
        boundedness:   episodic=[1,0,0],      habitual=[0,1,0],   static=[0,0,1]
        initiativity:  initiate=[1,0],        receive=[0,1]
        time_start:    past=[1,0],            current=[0,1]
        time_end:      current=[1,0],         future=[0,1]
        impact:        impactful=[1,0],       resolved=[0,1]
    """

    genericity: Literal["specific", "generic"] = Field(
        default="specific",
        description="通指性：specific（特指）/ generic（泛指）",
    )
    eventivity: Literal["dynamic", "stative"] = Field(
        default="dynamic",
        description="事件性：dynamic（动态）/ stative（静态）",
    )
    boundedness: Literal["episodic", "habitual", "static"] = Field(
        default="episodic",
        description="有界性：episodic（一次性）/ habitual（习惯性）/ static（静态）",
    )
    initiativity: Literal["initiate", "receive"] = Field(
        default="initiate",
        description="主动性：initiate（主动触发）/ receive（被动接受）",
    )
    time_start: Literal["past", "current"] = Field(
        default="past",
        description="起始时间：past（过去）/ current（当前）",
    )
    time_end: Literal["current", "future"] = Field(
        default="current",
        description="结束时间：current（持续中）/ future（未来结束）",
    )
    impact: Literal["impactful", "resolved"] = Field(
        default="impactful",
        description="影响性：impactful（有影响）/ resolved（已解决）",
    )

    def to_onehot(self) -> list[int]:
        """转换为15维 one-hot 编码向量。

        维度顺序与编码映射表一致：
        [genericity(2), eventivity(2), boundedness(3), initiativity(2),
         time_start(2), time_end(2), impact(2)] = 15维
        """
        onehot: list[int] = []

        # genericity (2维)
        if self.genericity == "specific":
            onehot.extend([1, 0])
        else:
            onehot.extend([0, 1])

        # eventivity (2维)
        if self.eventivity == "dynamic":
            onehot.extend([1, 0])
        else:
            onehot.extend([0, 1])

        # boundedness (3维)
        if self.boundedness == "episodic":
            onehot.extend([1, 0, 0])
        elif self.boundedness == "habitual":
            onehot.extend([0, 1, 0])
        else:
            onehot.extend([0, 0, 1])

        # initiativity (2维)
        if self.initiativity == "initiate":
            onehot.extend([1, 0])
        else:
            onehot.extend([0, 1])

        # time_start (2维)
        if self.time_start == "past":
            onehot.extend([1, 0])
        else:
            onehot.extend([0, 1])

        # time_end (2维)
        if self.time_end == "current":
            onehot.extend([1, 0])
        else:
            onehot.extend([0, 1])

        # impact (2维)
        if self.impact == "impactful":
            onehot.extend([1, 0])
        else:
            onehot.extend([0, 1])

        return onehot


# ═══════════════════════════════════════════════════════════════════
# 4. 结构化协议
# ═══════════════════════════════════════════════════════════════════


class StructuredProtocol(BaseModel):
    """LLM×MapReduce 结构化信息协议。

    每次 Map 操作的单元输出：从文本 chunk 中提取关键信息+推理过程+中间答案。
    参考: LLM×MapReduce ACL 2025 §3.2 Structured Protocol
    """

    extracted_info: str = Field(
        default="",
        description="从文本 chunk 中提取的关键信息",
    )
    rationale: str = Field(
        default="",
        description="LLM 推理过程和依据",
    )
    answer: str | None = Field(
        default=None,
        description="中间答案（None 表示 NO INFORMATION — 当前 chunk 无相关信息）",
    )
    confidence: float = Field(
        default=0.0,
        ge=1.0,
        le=5.0,
        description="置信度评分 [1, 5]，1=很低，5=非常高",
    )


# ═══════════════════════════════════════════════════════════════════
# 5. ConStory 错误类型
# ═══════════════════════════════════════════════════════════════════


class ConsistencyErrorType(str, Enum):
    """ConStory 5大类一致性错误类型枚举。

    参考: ConStory-Bench arXiv 2026 §2.2
    覆盖时间线/情节逻辑/角色塑造/世界观构建/事实/叙事风格六大维度。
    """

    # ── TimeLine & Plot Logic (时间线与情节逻辑) ──
    ABSOLUTE_TIME = "absolute_time_contradiction"
    """绝对时间矛盾 — 同一事件被赋予两个不同的时间点"""
    CAUSELESS_EFFECT = "causeless_effect"
    """无因之果 — 事件发生但前文未建立必要因由"""
    ABANDONED_PLOT = "abandoned_plot_element"
    """弃置情节元素 — 前文铺垫未回收"""

    # ── Characterization (角色塑造) ──
    MEMORY_CONTRADICTION = "memory_contradiction"
    """记忆矛盾 — 角色对同一事件有前后不一致的记忆"""
    SKILL_FLUCTUATION = "skill_fluctuation"
    """能力波动 — 角色能力水平无合理原因地忽高忽低"""

    # ── World Building (世界观构建) ──
    CORE_RULE_VIOLATION = "core_rule_violation"
    """核心规则违反 — 事件结果与世界观设定规则矛盾"""

    # ── Factual (事实性) ──
    NOMENCLATURE_CONFUSION = "nomenclature_confusion"
    """命名混淆 — 同一事物在不同位置使用不同名称"""

    # ── Narrative Style (叙事风格) ──
    PERSPECTIVE_CONFUSION = "perspective_confusion"
    """视角混淆 — 叙事视角非预期切换或信息超出视角限制"""


ErrorCategory = Literal[
    "timeline_plot",
    "characterization",
    "world_building",
    "factual",
    "narrative_style",
]
"""一致性错误类别联合类型 — 对应 ConStory 5
大类。"""


# ═══════════════════════════════════════════════════════════════════
# 6. 情绪曲线
# ═══════════════════════════════════════════════════════════════════


class NarrativeStage(str, Enum):
    """叙事弧阶段 — Freytag金字塔四阶段映射。

    参考: Shadow-Loom arXiv 2026 §4.4 Emotion Arc Framework
    """

    EXPOSITION = "exposition"
    """开端/铺垫 — 引入背景、角色和设定"""
    RISING = "rising"
    """上升/发展 — 冲突加剧，情节推进"""
    CLIMAX = "climax"
    """高潮 — 冲突顶点，核心转折"""
    FALLING = "falling"
    """下降/结局 — 冲突解决，余波收束"""


class EmotionArc(BaseModel):
    """情绪曲线 — 滑动窗口分析结果。

    基于滑动窗口逐段计算情感价值和唤起度，同时划分叙事弧阶段。
    参考: Shadow-Loom arXiv 2026 §4.4
    """

    valence_sequence: list[float] = Field(
        default_factory=list,
        description="情感价值序列，每窗口值 [-1.0, 1.0]，负=消极，正=积极",
    )
    arousal_sequence: list[float] = Field(
        default_factory=list,
        description="唤起度序列，每窗口值 [0.0, 1.0]，0=平静，1=强烈",
    )
    window_positions: list[int] = Field(
        default_factory=list,
        description="滑动窗口对应的原文字符位置（起始偏移量列表）",
    )
    stages: list[tuple[int, NarrativeStage]] = Field(
        default_factory=list,
        description="叙事弧阶段划分列表，每项=(阶段起始窗口索引, 阶段类型)",
    )


# ═══════════════════════════════════════════════════════════════════
# 7. 因果图输出
# ═══════════════════════════════════════════════════════════════════


class CausalGraph(BaseModel):
    """因果图输出 — 事件节点 + 因果边 + 元数据。

    参考: Shadow-Loom arXiv 2026 §5 Causal Reasoning & Graph Construction
    """

    events: list[EventNode] = Field(
        default_factory=list,
        description="因果图中所有事件节点列表",
    )
    edges: list[tuple[str, str, str]] = Field(
        default_factory=list,
        description="因果边列表，每项=(from_id, to_id, relation_type)",
    )
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="图元数据（如构建时间、模型版本、配置参数等）",
    )
