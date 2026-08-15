"""CrewResult 显式契约（v8.4）。

crew_result 是 LangGraph 根 state 与各 crew 子图之间唯一的 dict 管道。
历史上该管道为"隐式契约"（无 schema、无校验），是跨模块最易断的接口。

本模块把它显式化：
- CrewResult: 宽松 TypedDict（total=False，向后兼容）
- validate_crew_result(): 诊断性校验，返回缺失/异常清单（不抛异常，不阻断运行）

接入策略：写入方（prepare_writing 等）构造后调用 validate 打日志，
逐步让所有读写方共享此契约。未来可升级为 pydantic 严格校验。
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ── 核心必选字段（写作管线正常运行所需）──────────────────────────────────────
CORE_FIELDS = (
    "project_name",
    "genre",
    "current_chapter_number",
    "target_chapters",
    "world_setting",
    "character_setting",
    "story_outline",
    "chapter_outlines",
)

# ── 可选字段（按功能域归组，文档化来源）─────────────────────────────────────
# 章节进度域
COMPLETED_FIELDS = ("completed_chapters", "volume_structure")
# 上下文注入域
CONTEXT_FIELDS = ("loaded_memory", "skill_context", "auto_guidance")
# 写作过程域（writing_crew 写入）
WRITING_FIELDS = (
    "chapter_draft",
    "cross_chapter_state",
    "writer_context",
    "prev_chapters_summary",
)
# 评审结果域（verdict_engine 写入）
REVIEW_FIELDS = (
    "review_result",
    "quality_score",
    "final_score",
    "programmatic_score",
    "ai_style_score",
    "lao_shu_chong_score",
    "loop_count",
    "refine_attempts",
    "ai_style_fix",
    "lao_shu_chong_fix",
    "toxic_points",
    "shuangdian_points",
    "guide_references",
    "debate_issues",
    "debate_strengths",
    "debate_suggestions",
    "is_short_text",
)
# 内部计数域（coordinator/refiner 写入，跨 checkpoint 备份）
INTERNAL_FIELDS = ("_rewrite_count", "_refine_count")


class CrewResult(TypedDict, total=False):
    """CrewResult 显式契约（宽松 TypedDict）。"""

    # 核心
    project_name: str
    genre: str
    current_chapter_number: int
    target_chapters: int
    world_setting: str
    character_setting: str
    story_outline: str
    chapter_outlines: str
    # 章节进度
    completed_chapters: list[dict[str, Any]]
    volume_structure: dict[str, Any]
    # 上下文注入
    loaded_memory: dict[str, Any]
    skill_context: str
    auto_guidance: str
    # 写作过程
    chapter_draft: str
    cross_chapter_state: str
    writer_context: str
    prev_chapters_summary: str
    # 评审结果
    review_result: dict[str, Any]
    quality_score: float
    final_score: float
    programmatic_score: float
    ai_style_score: float
    lao_shu_chong_score: float
    loop_count: int
    refine_attempts: int
    ai_style_fix: str
    lao_shu_chong_fix: str
    toxic_points: list[str]
    shuangdian_points: list[str]
    guide_references: list[Any]
    debate_issues: list[str]
    debate_strengths: list[str]
    debate_suggestions: str
    is_short_text: bool
    # 内部计数
    _rewrite_count: int
    _refine_count: int
    # 未知扩展字段（写入方自由扩展，保持兼容）
    # 示例: tokens, estimated_cost_cny, chapter_usages, extracted, results


def validate_crew_result(cr: Any, context: str = "") -> list[str]:
    """诊断性校验 crew_result，返回问题清单（不抛异常）。

    Args:
        cr: crew_result dict（可能为 None / 非 dict）
        context: 调用场景标识（如 "prepare_writing ch=1"），仅用于日志

    Returns:
        list[str]: 问题描述列表；空列表表示通过核心检查。
    """
    problems: list[str] = []
    if cr is None:
        problems.append("crew_result 为 None")
        return problems
    if not isinstance(cr, dict):
        problems.append(f"crew_result 类型异常: {type(cr).__name__}")
        return problems

    for field in CORE_FIELDS:
        if field not in cr:
            problems.append(f"缺少核心字段: {field}")
        elif cr[field] in (None, ""):
            problems.append(f"核心字段为空: {field}")

    # 类型抽查（仅对高价值字段，避免误报）
    if "current_chapter_number" in cr and not isinstance(
        cr["current_chapter_number"], int
    ):
        problems.append(
            f"current_chapter_number 类型异常: {type(cr['current_chapter_number']).__name__}"
        )

    if problems:
        logger.warning(
            "[crew_result] 契约校验未通过 (%s): %s",
            context or "unknown",
            "; ".join(problems),
        )
    return problems
