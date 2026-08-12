"""统一 LLM 评审结果数据契约。

v8.2: 统一评审系统（unified）的数据模型。
替代分散的 ProgrammaticReport/CrossChapterSignals/DebateReport/FourDimReviewResult/LLM*Result。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnifiedFourDim:
    """四维评分（满分 30/25/25/20，合计 100）。"""

    logic: float = 0.0      # 剧情逻辑 0-30
    writing: float = 0.0    # 文笔表达 0-25
    character: float = 0.0  # 人物一致性 0-25
    world: float = 0.0      # 世界观契合 0-20

    def total(self) -> float:
        return self.logic + self.writing + self.character + self.world


@dataclass
class UnifiedReviewResult:
    """统一 LLM 评审结果（标签化双通道解析产物）。

    一次 LLM 调用输出五视角结论 + 四维分项 + 毒点/爽点/水段/跳跃定位 + 综合分。
    """

    final_score: float = 60.0
    four_dim: UnifiedFourDim = field(default_factory=UnifiedFourDim)
    toxic_points: list[dict] = field(default_factory=list)   # [{type, paragraph, severity, reason}]
    severe_toxic: bool = False
    shuangdian_points: list[str] = field(default_factory=list)
    shuangdian_count: int = 0
    water_paragraphs: list[int] = field(default_factory=list)
    scene_transition_issues: list[dict] = field(default_factory=list)  # [{from, to, reason}]
    human_like_score: float = 0.0
    attraction_score: float = 0.0
    immersion_score: float = 0.0
    cross_chapter_score: float = 0.0
    decay_hint: str = "none"
    perspective_disagreements: list[str] = field(default_factory=list)
    review_comments: str = ""
    fix_suggestions: list[str] = field(default_factory=list)
    raw_text: str = ""
    failed: bool = False
    retried: bool = False
    consistency_fixed: bool = False  # 自洽校验是否修正过分数
