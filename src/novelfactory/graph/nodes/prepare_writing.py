"""Prepare Writing Node (feishu_upload_node 已删除 — 同步迁入 sync_crew)."""

from __future__ import annotations

import logging

from novelfactory.config.constants import (
    COMPRESS_KEEP_MESSAGES,
    FALLBACK_TARGET_CHAPTERS,
)
from novelfactory.state.crew_result import validate_crew_result
from novelfactory.state.novel_state import (
    NovelFactoryState,
    compress_completed_chapters,
)

logger = logging.getLogger(__name__)

# v5.5: 引入 SkillLoader（在函数内延迟导入避免循环依赖）


def prepare_writing_node(state: NovelFactoryState) -> dict:
    """Prepare crew_result from root state fields before entering writing_crew subgraph.

    This node assembles the crew_result dict that the writing_crew subgraph
    expects, pulling from root state fields set during setup (world_setting,
    character_setting, story_outline, etc.) and completed chapters.
    """
    existing_cr = state.get("crew_result", {})
    # Prefer top-level completed_chapters (accumulated via operator.add),
    # fall back to crew_result.completed_chapters
    completed = state.get("completed_chapters") or existing_cr.get(
        "completed_chapters", []
    )
    # v4.2: Compress to prevent checkpoint state bloat.
    # _make_record() already strips full text; this provides defense-in-depth.
    completed = compress_completed_chapters(completed, keep_recent=COMPRESS_KEEP_MESSAGES)

    genre = state.get("genre", "")

    # v5.5: 加载当前题材的 Skill 内容注入到 writer_context
    skill_context = ""
    try:
        from novelfactory.skills.loader import SkillLoader

        loader = SkillLoader()
        loader.discover()
        if genre:
            skills = loader.get_skills_by_genre(genre)
            if skills:
                skill_context = "\n\n".join(
                    f"【{s.name}】\n{s.body[:2000]}" for s in skills if s.body
                )
    except Exception:
        pass

    cr = {
        "project_name": state.get("project_name", "未命名项目"),
        "story_outline": state.get("story_outline", ""),
        "chapter_outlines": state.get("chapter_outlines", ""),
        "world_setting": state.get("world_setting", ""),
        "character_setting": state.get("character_setting", ""),
        "volume_structure": state.get("volume_structure", {}),
        "genre": genre,
        "current_chapter_number": state.get("current_chapter", 1),
        "target_chapters": state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS,
        "completed_chapters": completed,
        "loaded_memory": state.get("loaded_memory", {}),
        # v5.5: 注入题材 Skill 到写作上下文
        "skill_context": skill_context,
    }

    # v8.4: 显式契约校验（诊断性，不阻断运行）
    validate_crew_result(
        cr,
        context=f"prepare_writing ch={cr.get('current_chapter_number')}",
    )

    # Propagate auto_guidance from volume/quality/foreshadowing checks (if any)
    auto_guidance = state.get("auto_guidance", "")
    if auto_guidance:
        cr["auto_guidance"] = auto_guidance

    # v6.2 FIX (R4): 重置章节级字段，防止上一章状态泄漏到新章节
    # writing_crew 子图每次调用共享父图 state，不清除会导致：
    # 1. chapter_draft 残留上章内容 → reviewer 评审错误文本
    # 2. quality_score/composite_score 残留 → verdict_router 错误路由
    # 3. loop_count/refine_attempts 残留 → 上限保护失效
    return {
        "crew_result": cr,
        "chapter_draft": "",
        "quality_score": 0.0,
        "ai_style_score": 0.0,
        "lao_shu_chong_score": 0.0,
        "loop_count": 0,
        "refine_attempts": 0,
        "review_result": {},
        "ai_style_fix": "",
        "lao_shu_chong_fix": "",
        "toxic_points": [],
        "shuangdian_points": [],
        "guide_references": [],
        "debate_issues": [],
        "debate_strengths": [],
        "debate_suggestions": "",
        "is_short_text": False,
    }

