"""Setup Crew 独立节点函数 — 从 lightweight_setup.py 拆出的 7 个节点（v5.4）。

管线顺序:
  world_builder → character_designer → outline_writer
  → volume_detail_writer → quality_gate → feishu_setup → db_persist

每个节点从 state 读取输入，将结果写回 state 的同名字段。
所有 LLM 调用均通过 async_llm_call_with_retry / _retry_invoke 获得保护。
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from novelfactory.agents.infra import (
    llm_call_with_retry,
    read_usage_tracking,
    reset_usage_tracking,
    validate_json_output,
)
from novelfactory.agents.setup_agents import (
    create_character_designer_agent,
    create_outline_writer_agent,
    create_volume_detail_writer_agent,
    create_world_builder_agent,
)
from novelfactory.config.constants import FALLBACK_TARGET_CHAPTERS
from novelfactory.config.llm import get_reviewer_llm, get_worker_llm
from novelfactory.integrations.feishu.feishu_api import (
    ensure_project_folders_idempotent,
    upload_setup_docs,
)
from novelfactory.integrations.feishu.notify import send_review_notification

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_SETUP_QUALITY_THRESHOLD = 70.0
_CHAPTER_RANGE_SIZE_START = 1
_CHAPTER_RANGE_SIZE_END = 2
_MIN_TEXT_LENGTH_FOR_EXTRACTION = 100


# ── State type (imported for type annotation only) ─────────────────────────
# SetupCrewState 定义在 lightweight_setup.py 中，此处不重复定义以避免循环导入。
# 节点函数接受 dict 以便灵活传递。

# ── Helpers ────────────────────────────────────────────────────────────────


def _retry_invoke(agent: Any, input_dict: dict, step_name: str) -> dict:
    """Agent.invoke with timeout + exponential-backoff retry."""
    result = llm_call_with_retry(
        agent.invoke,
        input_dict,
        step_name=f"setup.{step_name}",
        fallback={"messages": [], "crew_result": {}},
    )
    return result if result is not None else {"messages": [], "crew_result": {}}


async def _invoke_with_retry(agent: Any, input_dict: dict, step_name: str) -> dict:
    """异步 Agent.invoke 包装 — 走 async_llm_call_with_retry 获得超时+重试保护。

    当前 agent 仍使用同步 .invoke()，通过 ThreadPoolExecutor 包装为异步。
    未来创建 async agent 后可直接替换为 agent.ainvoke()。
    """
    import asyncio

    from novelfactory.agents.infra.async_retry import async_llm_call_with_retry

    async def _invoke():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, agent.invoke, input_dict)

    result = await async_llm_call_with_retry(
        _invoke,
        step_name=f"setup.{step_name}",
        fallback={"messages": [], "crew_result": {}},
    )
    return result if isinstance(result, dict) else {"messages": [], "crew_result": {}}


def _write_log(streaming_path: str | None, msg: str) -> None:
    """写流式输出到临时文件。"""
    if streaming_path:
        try:
            with open(streaming_path, "a", encoding="utf-8") as f:
                f.write(msg)
                f.flush()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Node 1: WorldBuilder
# ═══════════════════════════════════════════════════════════════════════════


async def world_builder_node(state: dict) -> dict:
    """Step 1: 世界观构建。

    Inputs (from state):
      - seed_idea, genre, project_name, target_chapters
      - streaming_path (optional)

    Outputs (to state):
      - world_setting: 世界观设定文本
      - messages: HumanMessage + AIMessage for Chat UI
    """
    seed_idea = state.get("seed_idea", "")
    genre = state.get("genre", "")
    project_name = state.get("project_name", "未命名项目")
    target_chapters = state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
    streaming_path = state.get("streaming_path") or state.get("_streaming_path")

    llm = get_worker_llm()
    _write_log(streaming_path, "\n\n## [WorldBuilder 开始构建世界观...]\n\n")
    logger.info("Step 1/4: WorldBuilder starting...")

    wb_agent = create_world_builder_agent(llm)
    wb_input = {
        "messages": [],
        "crew_result": {
            "seed_idea": seed_idea,
            "genre": genre,
            "project_name": project_name,
            "target_chapters": target_chapters,
        },
    }
    wb_result = await _invoke_with_retry(wb_agent, wb_input, "WorldBuilder")
    wb_cr = wb_result.get("crew_result", {})
    wb_text = wb_cr.get("world_setting", "") or "世界观构建失败"
    _write_log(streaming_path, wb_text)
    logger.info("WorldBuilder done (%d chars)", len(wb_text))

    return {
        "world_setting": wb_text,
        "messages": [
            AIMessage(
                content=f"## 世界观设定\n\n{str(wb_text)[:3000]}",
                name="world_builder",
            ),
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node 2: CharacterDesigner
# ═══════════════════════════════════════════════════════════════════════════


async def character_designer_node(state: dict) -> dict:
    """Step 2: 角色设计。

    Inputs: seed_idea, genre, project_name, target_chapters, world_setting
    Outputs: character_setting
    """
    seed_idea = state.get("seed_idea", "")
    genre = state.get("genre", "")
    project_name = state.get("project_name", "未命名项目")
    target_chapters = state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
    world_setting = state.get("world_setting", "")
    streaming_path = state.get("streaming_path") or state.get("_streaming_path")

    llm = get_worker_llm()
    _write_log(streaming_path, "\n\n## [CharacterDesigner 开始设计角色...]\n\n")
    logger.info("Step 2/4: CharacterDesigner starting...")

    cd_agent = create_character_designer_agent(llm)
    cd_input = {
        "messages": [],
        "crew_result": {
            "seed_idea": seed_idea,
            "genre": genre,
            "project_name": project_name,
            "target_chapters": target_chapters,
            "world_setting": world_setting,
        },
    }
    cd_result = await _invoke_with_retry(cd_agent, cd_input, "CharacterDesigner")
    cd_cr = cd_result.get("crew_result", {})
    cd_text = cd_cr.get("character_setting", "") or "角色设计失败"
    _write_log(streaming_path, cd_text)
    logger.info("CharacterDesigner done (%d chars)", len(cd_text))

    return {
        "character_setting": cd_text,
        "messages": [
            AIMessage(
                content=f"## 角色设定\n\n{str(cd_text)[:3000]}",
                name="character_designer",
            ),
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node 3: OutlineWriter (卷级大纲)
# ═══════════════════════════════════════════════════════════════════════════


async def outline_writer_node(state: dict) -> dict:
    """Step 3: 故事主线 + 卷级大纲。

    Inputs: seed_idea, genre, project_name, target_chapters, world_setting, character_setting
    Outputs: story_outline, volume_structure
    """
    seed_idea = state.get("seed_idea", "")
    genre = state.get("genre", "")
    project_name = state.get("project_name", "未命名项目")
    target_chapters = state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
    world_setting = state.get("world_setting", "")
    character_setting = state.get("character_setting", "")
    streaming_path = state.get("streaming_path") or state.get("_streaming_path")

    llm = get_worker_llm()
    _write_log(streaming_path, "\n\n## [OutlineWriter 开始构建卷级故事大纲...]\n\n")
    logger.info("Step 3/4: OutlineWriter starting...")

    ow_agent = create_outline_writer_agent(llm)
    ow_input = {
        "messages": [],
        "crew_result": {
            "seed_idea": seed_idea,
            "genre": genre,
            "project_name": project_name,
            "target_chapters": target_chapters,
            "world_setting": world_setting,
            "character_setting": character_setting,
        },
    }
    ow_result = await _invoke_with_retry(ow_agent, ow_input, "OutlineWriter")
    ow_cr = ow_result.get("crew_result", {})
    story_outline = ow_cr.get("story_outline", "") or "大纲创作失败"
    volume_structure = ow_cr.get("volume_structure") or {
        "story_theme": "",
        "total_volumes": 0,
        "volumes": [],
    }
    _write_log(streaming_path, story_outline)
    logger.info(
        "OutlineWriter done (outline=%d chars, volumes=%d)",
        len(story_outline),
        volume_structure.get("total_volumes", len(volume_structure.get("volumes", []))),
    )

    return {
        "story_outline": story_outline,
        "volume_structure": volume_structure,
        "messages": [
            AIMessage(
                content=f"## 故事大纲\n\n{str(story_outline)[:3000]}",
                name="outline_writer",
            ),
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node 4: VolumeDetailWriter (前3卷逐章大纲)
# ═══════════════════════════════════════════════════════════════════════════


async def volume_detail_writer_node(state: dict) -> dict:
    """Step 3b: 前3卷逐章大纲。

    Inputs: seed_idea, genre, project_name, target_chapters,
            world_setting, character_setting, story_outline, volume_structure
    Outputs: chapter_outlines, first_3_volumes_detail, volume_structure (enriched)
    """
    seed_idea = state.get("seed_idea", "")
    genre = state.get("genre", "")
    project_name = state.get("project_name", "未命名项目")
    target_chapters = state.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
    world_setting = state.get("world_setting", "")
    character_setting = state.get("character_setting", "")
    story_outline = state.get("story_outline", "")
    volume_structure = state.get("volume_structure", {})
    streaming_path = state.get("streaming_path") or state.get("_streaming_path")

    llm = get_worker_llm()
    _write_log(
        streaming_path, "\n\n## [VolumeDetailWriter 开始生成前3卷逐章大纲...]\n\n"
    )
    logger.info("Step 3b: VolumeDetailWriter starting (first 3 volumes)...")

    vd_agent = create_volume_detail_writer_agent(llm)
    volumes = volume_structure.get("volumes", [])
    first_3_volumes_detail: list[dict] = []
    chapter_outlines_parts = ["## 章节大纲（前3卷）\n"]
    previous_volume_summary = ""

    for vol in volumes[:3]:
        vol_num = vol.get("volume_number", 0)
        vol_title = vol.get("title", "")
        ch_range = vol.get("chapter_range", [1, 40])
        chapter_start = (
            ch_range[0]
            if isinstance(ch_range, list) and len(ch_range) >= _CHAPTER_RANGE_SIZE_START
            else 1
        )
        chapter_end = (
            ch_range[1]
            if isinstance(ch_range, list) and len(ch_range) >= _CHAPTER_RANGE_SIZE_END
            else 40
        )

        vd_input = {
            "messages": [],
            "crew_result": {
                "seed_idea": seed_idea,
                "genre": genre,
                "project_name": project_name,
                "target_chapters": target_chapters,
                "world_setting": world_setting,
                "character_setting": character_setting,
                "story_outline": story_outline,
                "volume_number": vol_num,
                "volume_title": vol_title,
                "chapter_start": chapter_start,
                "chapter_end": chapter_end,
                "previous_volume_summary": previous_volume_summary,
            },
        }
        vd_result = await _invoke_with_retry(
            vd_agent,
            vd_input,
            f"VolumeDetailWriter_V{vol_num}",
        )
        vd_cr = vd_result.get("crew_result", {})
        ch_outlines = vd_cr.get("chapter_outlines_detail", [])

        vol["chapter_outlines"] = ch_outlines
        first_3_volumes_detail.append(vol)

        chapter_outlines_parts.append(
            f"\n### 第{vol_num}卷《{vol_title}》（第{chapter_start}-{chapter_end}章）\n"
        )
        for ch in ch_outlines:
            chapter_outlines_parts.append(
                f"  第{ch['chapter_number']}章《{ch['title']}》：{ch['core_events']}"
                f"（悬念：{ch['cliffhanger']}，重要性：{ch['importance']}/10）\n"
            )

        previous_volume_summary = vol.get("summary", "")
        logger.info(
            "VolumeDetailWriter V%d done (%d chapters)", vol_num, len(ch_outlines)
        )
        _write_log(
            streaming_path,
            f"第{vol_num}卷《{vol_title}》: {len(ch_outlines)}章大纲已生成\n",
        )

    chapter_outlines = "".join(chapter_outlines_parts)
    _write_log(streaming_path, chapter_outlines)
    logger.info(
        "VolumeDetailWriter done (first 3 volumes, total %d chapters)",
        sum(len(v.get("chapter_outlines", [])) for v in first_3_volumes_detail),
    )

    return {
        "chapter_outlines": chapter_outlines,
        "first_3_volumes_detail": first_3_volumes_detail,
        "volume_structure": volume_structure,  # enriched with chapter_outlines
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node 5: Quality Gate (M3-5: LLM scoring + retry weakest output)
# ═══════════════════════════════════════════════════════════════════════════


async def quality_gate_node(state: dict) -> dict:
    """Step 4: M3-5 Quality Gate — LLM 评分 + 最多 2 轮重试最弱项。

    Inputs: world_setting, character_setting, story_outline, chapter_outlines
    Outputs: quality_score, quality_comments, review_text,
             (可能更新 world_setting, character_setting, story_outline)
    """
    world_setting = state.get("world_setting", "")
    character_setting = state.get("character_setting", "")
    story_outline = state.get("story_outline", "")
    chapter_outlines = state.get("chapter_outlines", "")

    from novelfactory.agents.setup_agents import (
        create_character_designer_agent,
        create_outline_writer_agent,
        create_world_builder_agent,
    )
    from novelfactory.graph.lightweight_setup import _SETUP_QUALITY_THRESHOLD

    threshold = state.get("_setup_quality_threshold", _SETUP_QUALITY_THRESHOLD)

    # ── Quality Gate helper (async, with retry protection) ──
    async def _score(w: str, c: str, s: str, o: str) -> tuple[float, str]:
        from novelfactory.agents.infra.async_retry import async_llm_call_with_retry

        reviewer_llm = get_reviewer_llm()
        from novelfactory.config.prompts import get_prompt

        prompt_template = get_prompt("setup", "quality_gate")
        prompt = prompt_template.format(
            world_setting_len=len(w),
            world_setting=w[:3000],
            character_setting_len=len(c),
            character_setting=c[:2000],
            story_outline_len=len(s),
            story_outline=s[:2000],
            chapter_outlines_len=len(o),
            chapter_outlines=o[:2000],
        )

        async def _invoke():
            response = await reviewer_llm.ainvoke([("user", prompt)])
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            # Record usage
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                usage = meta.get("usage") or meta.get("token_usage", {})
                if isinstance(usage, dict):
                    pt = int(usage.get("prompt_tokens", 0) or 0)
                    ct = int(usage.get("completion_tokens", 0) or 0)
                    if pt or ct:
                        from novelfactory.agents.infra.usage import _record_usage

                        _record_usage("setup_quality_gate", pt, ct)
            return {"messages": [response], "_text": response_text}

        result = await async_llm_call_with_retry(
            _invoke,
            step_name="setup_quality_gate",
            timeout_seconds=120,
            fallback={"messages": [], "_text": ""},
        )
        text = result.get("_text", "") if isinstance(result, dict) else ""

        parsed, err = validate_json_output(
            text,
            required_keys=["quality_score", "review_comments"],
            fail_closed=False,
        )
        if parsed:
            score = float(parsed.get("quality_score", 50.0))
            comments = str(parsed.get("review_comments", ""))
            return max(0.0, min(100.0, score)), comments
        return 50.0, f"评分解析失败：{err or text[:200]}"

    # ── First pass ──
    quality_score, quality_comments = await _score(
        world_setting,
        character_setting,
        story_outline,
        chapter_outlines,
    )
    logger.info(
        "M3-5 setup quality gate: score=%.1f (%s)",
        quality_score,
        quality_comments[:200],
    )

    # ── Retry weakest up to 2 times if below threshold ──
    if quality_score < threshold:
        logger.warning(
            "M3-5 setup quality gate FAILED: score=%.1f < %.1f. Retrying weakest...",
            quality_score,
            threshold,
        )

        # Identify weakest from comments
        if "世界观" in quality_comments or "world" in quality_comments.lower():
            weakest_field = "world_setting"
        elif "角色" in quality_comments or "character" in quality_comments.lower():
            weakest_field = "character_setting"
        else:
            weakest_field = "story_outline"

        llm = get_worker_llm()
        wb_agent = create_world_builder_agent(llm)
        cd_agent = create_character_designer_agent(llm)
        ow_agent = create_outline_writer_agent(llm)

        for retry_round in range(2):
            logger.info("Setup retry round %d: %s", retry_round + 1, weakest_field)
            retry_agent_name = None
            retry_input = {}

            if weakest_field == "world_setting":
                retry_agent_name = "WorldBuilder"
                retry_input = {
                    "messages": [],
                    "crew_result": {
                        "seed_idea": state.get("seed_idea", ""),
                        "genre": state.get("genre", ""),
                        "project_name": state.get("project_name", ""),
                        "target_chapters": state.get("target_chapters")
                        or FALLBACK_TARGET_CHAPTERS,
                    },
                }
            elif weakest_field == "character_setting":
                retry_agent_name = "CharacterDesigner"
                retry_input = {
                    "messages": [],
                    "crew_result": {
                        "seed_idea": state.get("seed_idea", ""),
                        "genre": state.get("genre", ""),
                        "project_name": state.get("project_name", ""),
                        "target_chapters": state.get("target_chapters")
                        or FALLBACK_TARGET_CHAPTERS,
                        "world_setting": world_setting,
                    },
                }
            else:  # story_outline
                retry_agent_name = "OutlineWriter"
                retry_input = {
                    "messages": [],
                    "crew_result": {
                        "seed_idea": state.get("seed_idea", ""),
                        "genre": state.get("genre", ""),
                        "project_name": state.get("project_name", ""),
                        "target_chapters": state.get("target_chapters")
                        or FALLBACK_TARGET_CHAPTERS,
                        "world_setting": world_setting,
                        "character_setting": character_setting,
                    },
                }

            agent_map = {
                "WorldBuilder": wb_agent,
                "CharacterDesigner": cd_agent,
                "OutlineWriter": ow_agent,
            }
            retry_result = await _invoke_with_retry(
                agent_map[retry_agent_name],
                retry_input,
                retry_agent_name + f"_retry_{retry_round + 1}",
            )
            retry_cr = retry_result.get("crew_result", {})

            if weakest_field == "world_setting":
                world_setting = retry_cr.get("world_setting", world_setting)
            elif weakest_field == "character_setting":
                character_setting = retry_cr.get("character_setting", character_setting)
            else:
                new_outline = retry_cr.get("story_outline", "")
                if new_outline and new_outline != story_outline:
                    story_outline = new_outline

            quality_score, quality_comments = await _score(
                world_setting,
                character_setting,
                story_outline,
                chapter_outlines,
            )
            logger.info(
                "Setup retry %d: quality_score=%.1f (%s)",
                retry_round + 1,
                quality_score,
                quality_comments[:200],
            )
            if quality_score >= threshold:
                logger.info(
                    "M3-5 setup quality gate PASSED after retry %d", retry_round + 1
                )
                break

        if quality_score < threshold:
            logger.warning(
                "M3-5 setup quality gate STILL BELOW %.1f after 2 retries: score=%.1f. Proceeding with warning.",
                threshold,
                quality_score,
            )
            review_text = f"质量警告：总分{quality_score:.1f}仍低于{threshold}分，可能影响后续章节质量"
        else:
            review_text = f"质量门控通过（重试后达标，分数：{quality_score:.1f}）"
    else:
        review_text = f"质量门控通过（分数：{quality_score:.1f}）"

    return {
        "quality_score": quality_score,
        "quality_comments": quality_comments,
        "review_text": review_text,
        # 重试后可能更新的字段
        "world_setting": world_setting,
        "character_setting": character_setting,
        "story_outline": story_outline,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node 6: Feishu Setup (通知 + 目录创建 + 文档上传)
# ═══════════════════════════════════════════════════════════════════════════


async def feishu_setup_node(state: dict) -> dict:
    """Steps 5+6: 飞书通知 + 目录创建 + 设定文档上传。

    Inputs: project_name, thread_id, world_setting, character_setting, story_outline
    Outputs: folder_tokens
    """
    project_name = state.get("project_name", "未命名项目")
    thread_id = state.get("thread_id", "")
    world_setting = state.get("world_setting", "")
    character_setting = state.get("character_setting", "")
    story_outline = state.get("story_outline", "")

    # Step 5: Feishu notification
    try:
        send_review_notification(
            thread_id=thread_id,
            review_type="kickoff",
            project_name=project_name,
            content_summary=(
                f"世界观：{world_setting[:100]}...\n"
                f"角色：{character_setting[:100]}...\n"
                f"大纲：{story_outline[:100]}..."
            ),
        )
        logger.info("Feishu kickoff notification sent")
    except Exception as e:
        logger.warning("Feishu notification failed: %s", e)

    # Step 6: Create Feishu Drive folder structure + upload setup docs
    folder_tokens: dict = {
        "project": "",
        "setup": "",
        "chapters": "",
        "volume_folder_tokens": {},
    }
    try:
        folder_tokens = ensure_project_folders_idempotent(project_name)
        if folder_tokens.get("project"):
            logger.info(
                "[setup] 飞书目录树创建成功: project=%s", folder_tokens["project"]
            )
            doc_urls = upload_setup_docs(
                project_name,
                world_setting,
                character_setting,
                story_outline,
                folder_tokens,
            )
            if doc_urls.get("world"):
                logger.info("[setup] 世界观设定已上传: %s", doc_urls["world"])
        else:
            logger.info("[setup] 飞书未配置，跳过目录创建")
    except Exception as e:
        logger.warning("[setup] 飞书目录创建失败（不影响主流程）: %s", e)

    return {"folder_tokens": folder_tokens}


# ═══════════════════════════════════════════════════════════════════════════
#  Node 7: Database Persist (卷结构持久化)
# ═══════════════════════════════════════════════════════════════════════════


async def db_persist_node(state: dict) -> dict:
    """Step 7: 卷级大纲持久化到数据库。

    使用 OutlineManager 将卷和章大纲写入 novel_volumes 和 novel_chapter_outlines 表。
    如果数据库不可用，静默跳过（不影响 Setup 流程）。

    Inputs:  project_name, volume_structure, first_3_volumes_detail
    Outputs: (none — side effect only)
    """
    project_name = state.get("project_name", "")
    volume_structure = state.get("volume_structure", {})
    first_3_volumes_detail = state.get("first_3_volumes_detail", [])

    try:
        from novelfactory.config.database import DatabaseManager
        from novelfactory.pipeline.scale_manager import ChapterOutline, OutlineManager

        with DatabaseManager.get_instance().get_connection() as conn:
            manager = OutlineManager(conn)

            volumes = volume_structure.get("volumes", [])
            detail_map: dict[int, list[dict]] = {
                v["volume_number"]: v.get("chapter_outlines", [])
                for v in first_3_volumes_detail
            }

            for vol in volumes:
                vol_num = vol.get("volume_number", 0)
                ch_range = vol.get("chapter_range", [0, 0])
                start_ch = (
                    ch_range[0]
                    if isinstance(ch_range, list) and len(ch_range) >= 1
                    else 0
                )
                end_ch = (
                    ch_range[1]
                    if isinstance(ch_range, list) and len(ch_range) >= 2
                    else 0
                )

                manager.create_volume(
                    project=project_name,
                    volume_number=vol_num,
                    title=vol.get("title", f"第{vol_num}卷"),
                    theme=vol.get("theme", ""),
                    summary=vol.get("summary", ""),
                    start_chapter=start_ch,
                    end_chapter=end_ch,
                )

                if vol_num in detail_map:
                    for ch in detail_map[vol_num]:
                        outline = ChapterOutline(
                            chapter_number=ch.get("chapter_number", 0),
                            volume_number=vol_num,
                            title=ch.get("title", ""),
                            goal=ch.get("core_events", ""),
                            key_beats=[],
                            pov_character="",
                            characters_involved=[],
                            foreshadowing_plant=[],
                            foreshadowing_resolve=[],
                            word_count_target=3000,
                            status="pending",
                        )
                        manager.save_chapter_outline(project_name, outline)

            logger.info(
                "卷级大纲已持久化到数据库（%d 卷，前3卷含章大纲）",
                len(volumes),
            )
    except Exception as e:
        logger.warning("卷级大纲持久化失败（不影响 Setup 流程）: %s", e)

    return {}


# ═══════════════════════════════════════════════════════════════════════════
#  Node: Init Setup (流式文件 + 用量重置)
# ═══════════════════════════════════════════════════════════════════════════


async def init_setup_node(state: dict) -> dict:
    """Setup 入口节点：初始化流式文件 + 重置用量追踪。

    如果 setup 已完成，跳过并返回 setup_complete 标记。
    """
    # Guard: skip re-running if setup already completed (checkpoint recovery)
    if state.get("setup_complete", False):
        folder_tokens = state.get("folder_tokens", {}) or {}
        if folder_tokens.get("project"):
            logger.info("[setup_crew] Setup already complete, skipping")
            return {
                "current_phase": "writing",
                "setup_complete": True,
            }
        logger.info(
            "[setup_crew] Setup complete but folder_tokens missing, retrying folders..."
        )
        try:
            project_name = state.get("project_name", "未命名项目")
            folder_tokens = ensure_project_folders_idempotent(project_name)
            if folder_tokens.get("project"):
                upload_setup_docs(
                    project_name,
                    state.get("world_setting", ""),
                    state.get("character_setting", ""),
                    state.get("story_outline", ""),
                    folder_tokens,
                )
        except Exception as e:
            logger.warning("[setup_crew] Folder retry failed: %s", e)
        return {
            "current_phase": "writing",
            "setup_complete": True,
            "folder_tokens": folder_tokens,
        }

    reset_usage_tracking()

    # Create streaming temp file
    streaming_path: str | None = None
    enable_streaming = state.get("enable_streaming", True)
    if enable_streaming:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
            encoding="utf-8",
            dir=os.environ.get("TEMP", os.environ.get("TMP", "/tmp")),
        )
        streaming_path = tmp.name
        tmp.close()
        logger.info("Streaming to %s", streaming_path)

    messages = []
    seed_idea = state.get("seed_idea", "")
    if seed_idea:
        messages.append(
            HumanMessage(
                content=f"项目：{state.get('project_name', '')}\n"
                f"类型：{state.get('genre', '')}\n"
                f"创意：{str(seed_idea)[:500]}",
                name="user_input",
            )
        )

    return {
        "_streaming_path": streaming_path,
        "messages": messages,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node: Setup Finalize (汇总 + 用量快照 + handoff)
# ═══════════════════════════════════════════════════════════════════════════


async def setup_finalize_node(state: dict) -> Command:
    """Setup 出口节点：汇总所有结果 + 用量快照 + 通过 Command.PARENT 推送状态到父图。

    这是 Setup 子图的最后一个节点。使用 ``Command(graph=Command.PARENT)``
    显式退出子图并将状态更新推送到父图（NovelFactoryState），
    解决子图局部状态无法在 astream_events 模式下自动合并到父图的问题。
    """
    world_setting = state.get("world_setting", "")
    character_setting = state.get("character_setting", "")
    story_outline = state.get("story_outline", "")
    chapter_outlines = state.get("chapter_outlines", "")
    volume_structure = state.get("volume_structure", {})
    first_3_volumes_detail = state.get("first_3_volumes_detail", [])
    folder_tokens = state.get("folder_tokens", {})
    review_text = state.get("review_text", "质量门控测试")
    project_name = state.get("project_name", "未命名项目")

    streaming_path = state.get("_streaming_path")
    if streaming_path:
        try:
            with open(streaming_path, "a", encoding="utf-8") as f:
                f.write("\n\n## [Setup 完成，准备进入写作阶段]\n")
            logger.info("Full output written to: %s", streaming_path)
        except Exception:
            pass

    # Token usage snapshot
    setup_usage = read_usage_tracking()
    logger.info(
        "[setup] usage: prompt=%d completion=%d total=%d cost≈¥%.4f",
        setup_usage["prompt_tokens"],
        setup_usage["completion_tokens"],
        setup_usage["total_tokens"],
        setup_usage["estimated_cost_cny"],
    )

    crew_result = {
        "project_name": project_name,
        "folder_tokens": folder_tokens,
        "world_setting": world_setting,
        "character_setting": character_setting,
        "story_outline": story_outline,
        "chapter_outlines": chapter_outlines,
        "volume_structure": volume_structure,
        "first_3_volumes_detail": first_3_volumes_detail,
        "setup_complete": True,
        "review_text": review_text,
    }

    # Build Chat UI messages
    messages = []
    seed_idea = state.get("seed_idea", "")
    if seed_idea:
        messages.append(
            HumanMessage(
                content=f"项目：{state.get('project_name', '')}\n"
                f"类型：{state.get('genre', '')}\n"
                f"创意：{str(seed_idea)[:500]}",
                name="user_input",
            )
        )
    if world_setting:
        messages.append(
            AIMessage(
                content=f"## 世界观设定\n\n{str(world_setting)[:3000]}",
                name="world_builder",
            )
        )
    if character_setting:
        messages.append(
            AIMessage(
                content=f"## 角色设定\n\n{str(character_setting)[:3000]}",
                name="character_designer",
            )
        )
    if story_outline:
        messages.append(
            AIMessage(
                content=f"## 故事大纲\n\n{str(story_outline)[:3000]}",
                name="outline_writer",
            )
        )

    return Command(
        graph=Command.PARENT,
        update={
            "crew_result": {
                "crew_name": "setup",
                **crew_result,
                "setup_usage": setup_usage,
            },
            "world_setting": world_setting,
            "character_setting": character_setting,
            "story_outline": story_outline,
            "chapter_outlines": chapter_outlines,
            "volume_structure": volume_structure,
            "folder_tokens": folder_tokens,
            "setup_complete": True,
            "current_phase": "writing",
            "messages": messages,
            "total_usage": {
                "chapter_usages": [
                    {
                        "chapter_number": 0,
                        "phase": "setup",
                        "prompt_tokens": setup_usage["prompt_tokens"],
                        "completion_tokens": setup_usage["completion_tokens"],
                        "total_tokens": setup_usage["total_tokens"],
                        "estimated_cost_cny": setup_usage["estimated_cost_cny"],
                        "model_breakdown": setup_usage.get("model_breakdown", {}),
                        "quality_score": 0.0,
                    }
                ],
                "prompt_tokens": setup_usage["prompt_tokens"],
                "completion_tokens": setup_usage["completion_tokens"],
                "total_tokens": setup_usage["total_tokens"],
                "estimated_cost_cny": setup_usage["estimated_cost_cny"],
                "model_breakdown": setup_usage.get("model_breakdown", {}),
            },
        },
    )
