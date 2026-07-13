"""Prepare Writing Node + Feishu Drive Upload Node."""

from __future__ import annotations

import logging
import os

from novelfactory.config.constants import (
    COMPRESS_KEEP_RECENT,
    FALLBACK_TARGET_CHAPTERS,
)
from novelfactory.integrations.feishu.feishu_api import (
    ensure_project_folders_idempotent,
    upload_chapter_as_doc,
)
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
    completed = compress_completed_chapters(completed, keep_recent=COMPRESS_KEEP_RECENT)

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


# ── Feishu Drive Upload Node ──────────────────────────────────────────────────


def feishu_upload_node(state: NovelFactoryState) -> dict:
    """LangGraph 节点：将完成的章节上传到飞书云盘。

    插入位置：
      sync_crew → [feishu_upload_node] → main_supervisor

    行为：
      - 读取 state 中的 folder_tokens（优先初始化，首次上传时自动创建）
      - 读取最新完成的章节内容
      - 调用 feishu_drive.upload_chapter_to_drive() 上传
      - 上传结果写入 state.folder_tokens.volume_folder_tokens

    容错：上传失败不阻塞主流程，仅记录日志。
    """
    updates: dict = {}
    folder_tokens = state.get("folder_tokens", {}) or {}
    chapters_token = folder_tokens.get("chapters", "")
    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st

    root_token = _st.FEISHU_ROOT_FOLDER or os.environ.get("FEISHU_ROOT_FOLDER", "")
    project_name = state.get("project_name", "未命名项目")

    # 首次上传时自动创建目录树（最佳尝试，失败不阻塞）
    if not chapters_token and root_token:
        logger.info("[feishu_upload] folder_tokens 未初始化，首次创建目录树...")
        try:
            new_folder_tokens = ensure_project_folders_idempotent(project_name)
            if new_folder_tokens.get("chapters"):
                logger.info(
                    "[feishu_upload] 目录树创建成功: %s",
                    new_folder_tokens.get("project"),
                )
                chapters_token = new_folder_tokens.get("chapters", "")
                # v5.9 P2-fix: 保留已有的 volume_folder_tokens，防止覆写
                existing_vol_tokens = folder_tokens.get("volume_folder_tokens", {})
                updates["folder_tokens"] = {
                    **new_folder_tokens,
                    "volume_folder_tokens": existing_vol_tokens,
                }
            else:
                logger.info("[feishu_upload] 目录创建返回空，降级到 Bot 根目录上传")
        except Exception as e:
            logger.warning("[feishu_upload] 目录创建失败，降级到 Bot 根目录: %s", e)

    completed = state.get("completed_chapters", [])
    if not completed:
        return updates

    latest = completed[-1]
    ch_num = latest.get("chapter_number", 0)
    chapter_text = (
        latest.get("refined_chapter")
        or latest.get("chapter_draft", "")
        or latest.get("content", "")
    )
    if not chapter_text:
        logger.debug("[feishu_upload] 第%d章无正文，跳过上传", ch_num)
        return updates

    # 确定卷号
    volume_number = 1
    volume_structure = state.get("volume_structure", {})
    volumes = volume_structure.get("volumes", [])
    for vol in volumes:
        ch_range = vol.get("chapter_range", [0, 0])
        if ch_range[0] <= ch_num <= ch_range[1]:
            volume_number = vol.get("volume_number", volumes.index(vol) + 1)
            break

    vol_tokens = folder_tokens.get("volume_folder_tokens", {})
    try:
        url = upload_chapter_as_doc(
            project_name=project_name,
            chapter_number=ch_num,
            chapter_text=chapter_text,
            volume_number=volume_number,
            folder_tokens=folder_tokens,
        )
        if url:
            logger.info("[feishu_upload] 第%d章上传成功: %s", ch_num, url)
            updates["folder_tokens"] = {
                **folder_tokens,
                "volume_folder_tokens": vol_tokens,
            }
        else:
            logger.warning("[feishu_upload] 第%d章上传失败", ch_num)
    except Exception as e:
        logger.error("[feishu_upload] 第%d章上传异常: %s", ch_num, e)

    return updates
