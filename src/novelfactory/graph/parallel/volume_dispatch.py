"""Volume-level parallel chapter dispatch via LangGraph Send (DEPRECATED).

v6.3: Send 并行分发已移除，改为线性逐一创作。
此文件保留仅用于检查点向后兼容，所有函数不再被图调用。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Send

from novelfactory.config.llm import get_reviewer_llm
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)


def volume_dispatch(state: NovelFactoryState) -> list[Send]:
    """将卷内的章节分发为并行写作任务。

    参考 LangGraph 原生 Send Map-Reduce 模式：
      每个 Send 产生一个独立的 chapter_writer 执行路径，
      所有路径完成后自动汇聚到 chapter_collector。

    Args:
        state: NovelFactoryState with pending_chapters list.

    Returns:
        list of Send instances, each dispatching one chapter to writing_crew.
    """
    pending_chapters: list[dict[str, Any]] = state.get("pending_chapters", [])  # type: ignore[assignment]
    thread_id: str = state.get("thread_id", "")

    if not pending_chapters:
        logger.warning(
            "[volume_dispatch] No pending chapters — skipping parallel dispatch"
        )
        return []

    logger.info(
        "[volume_dispatch] Dispatching %d chapters in parallel for thread %s",
        len(pending_chapters),
        thread_id[:12],
    )

    # ── v6.1 P0-fix: pending_chapters 合理性校验 ──
    # 验证章节编号是否连续，防止 volume_structure 数据损坏导致章节跳跃
    current_ch: int = state.get("current_chapter", 0)
    chapter_numbers = [
        c.get("number", 0) for c in pending_chapters if isinstance(c, dict)
    ]
    if chapter_numbers:
        min_ch = min(chapter_numbers)
        if (
            current_ch
            and current_ch < min_ch
            and min_ch - current_ch > max(50, len(chapter_numbers) * 2)
        ):
            logger.error(
                "[volume_dispatch] 章节编号异常！current_chapter=%d, 但 pending_chapters 从 %d 开始 "
                "(跨度 %d 章)。已阻止并行分发以免写入错误章节！",
                current_ch,
                min_ch,
                min_ch - current_ch,
            )
            return []

        # 检查编号是否连续
        sorted_nums = sorted(chapter_numbers)
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] - sorted_nums[i - 1] != 1:
                logger.warning(
                    "[volume_dispatch] pending_chapters 编号不连续: %s",
                    sorted_nums,
                )
                break

    sends: list[Send] = []
    volume_context: dict[str, Any] = state.get("volume_context", {})  # type: ignore[assignment]
    for chapter in pending_chapters:
        ch_number = chapter.get("number", 0)
        sends.append(
            Send(
                "writing_crew",
                # v6.1 P3-10: 传递完整上下文（thread_id + volume_context），
                # 解决懒生成大纲场景的上下文断链问题。
                {
                    "current_chapter": ch_number,
                    "thread_id": thread_id,
                    "volume_context": volume_context,
                },
            )
        )
    return sends


def chapter_collector(state: NovelFactoryState) -> dict[str, Any]:
    """汇聚所有并行章的结果，进入卷级评审。

    All Send paths converge here automatically.
    The completed_chapters reducer (operator.add) has already merged results.

    Args:
        state: NovelFactoryState with accumulated completed_chapters.

    Returns:
        State update with volume_result and phase transition.
    """
    completed: list[dict[str, Any]] = state.get("completed_chapters", [])
    thread_id: str = state.get("thread_id", "")
    pending: list[dict[str, Any]] = state.get("pending_chapters", [])  # type: ignore[assignment]

    logger.info(
        "[chapter_collector] Collected %d chapters (expected %d) for thread %s",
        len(completed),
        len(pending),
        thread_id[:12],
    )

    # v5.9 P0-fix: 显式设置 current_chapter = max(chapter_numbers) + 1，
    # 避免并行 _exit_for_chapter 竞争 _last_value reducer 导致的非确定性值
    chapter_numbers = [
        c.get("chapter_number", 0) for c in completed if isinstance(c, dict)
    ]
    next_chapter: int = (
        max(chapter_numbers) + 1 if chapter_numbers else state.get("current_chapter", 1)
    )

    # v6.0.1: 并发一致性校验 — 验证实际收集章节数与期望数的匹配
    if len(completed) < len(pending):
        logger.warning(
            "[chapter_collector] 缺少章节 — collected=%d < expected=%d (可能部分 Send 分支失败)",
            len(completed),
            len(pending),
        )

    return {
        "volume_result": {
            "chapters": completed,
            "collected_count": len(completed),
            "expected_count": len(pending),
            "collected_at": thread_id,
            # v6.0.1: 标记缺失（如有），供 volume_reviewer 做降级处理
            "missing_count": max(0, len(pending) - len(completed)),
        },
        "current_phase": "volume_review",
        "current_chapter": next_chapter,
        "pending_chapters": [],
    }


def volume_reviewer(state: NovelFactoryState) -> dict[str, Any]:
    """卷级评审节点 — 对并行汇聚后的卷进行整体质量评审。

    在所有并行章节汇聚后执行，评估卷级一致性：
      - 章节间衔接是否流畅
      - 伏笔跨章一致性
      - 卷级节奏把控
      - 人物弧光连贯性

    Args:
        state: NovelFactoryState with volume_result.

    Returns:
        State update with volume_review_result and phase transition.
    """
    volume_result: dict[str, Any] = state.get("volume_result", {})  # type: ignore[assignment]
    collected_count: int = volume_result.get("collected_count", 0)
    expected_count: int = volume_result.get("expected_count", 0)
    thread_id: str = state.get("thread_id", "")

    logger.info(
        "[volume_reviewer] 开始卷级评审 — 已收集 %d/%d 章 (thread=%s)",
        collected_count,
        expected_count,
        thread_id[:12] if thread_id else "?",
    )

    if collected_count == 0:
        logger.warning("[volume_reviewer] 无章节可评审 — 跳过卷级评审")
        return {
            "volume_review_result": {
                "passed": True,
                "score": 0.0,
                "comment": "无章节，跳过卷级评审",
                "skipped": True,
            },
            "current_phase": "writing",
        }

    chapters: list[dict[str, Any]] = volume_result.get("chapters", [])
    chapter_indices = sorted(
        {c.get("chapter_number", 0) for c in chapters if isinstance(c, dict)}
    )
    chapter_count = len(chapter_indices)

    # v5.7 P1-fix: 用实际 LLM 评审替代占位值 85.0
    # v5.9 P1-fix: 从 chapter_summary 读取（make_record 剥离了 chapter_draft 以防检查点膨胀），
    # 卷级评审仅需摘要级别的上下文（每章 200 字），详细内容评审已在单章节 reviewer 完成。
    chapter_summaries = "\n".join(
        f"第{c.get('chapter_number', '?')}章 (评分{c.get('quality_score', '?')}): "
        f"{str(c.get('chapter_summary', c.get('title', '')))}"
        for c in chapters
        if isinstance(c, dict)
    )[:8000]

    review_prompt = (
        "你是卷级审稿编辑，请对以下并行写作的卷进行整体质量评审。\n\n"
        "## 评审维度\n"
        "1. 章节间衔接是否流畅（跨章过渡）\n"
        "2. 伏笔跨章一致性\n"
        "3. 卷级节奏把控\n"
        "4. 人物弧光连贯性\n\n"
        f"收录章节: {chapter_indices[0]}-{chapter_indices[-1]}（共{chapter_count}章）\n\n"
        f"{chapter_summaries}\n\n"
        "输出格式: {'score': <0-100>, 'passed': <true/false>, 'comment': '<评价>'}"
    )

    try:
        llm = get_reviewer_llm()
        response = llm.invoke(review_prompt)
        text = (
            response.content.strip() if hasattr(response, "content") else str(response)
        )
        import json as _json

        if "{" in text:
            json_str = text[text.find("{") : text.rfind("}") + 1]
            parsed_raw: dict[str, Any] = _json.loads(json_str)
        else:
            parsed_raw = {"score": 85.0, "passed": True, "comment": text[:500]}
        parsed = parsed_raw
    except Exception as exc:
        logger.warning("[volume_reviewer] LLM 评审失败 (%s)，使用降级保守评分", exc)
        # v5.9 FIX: 所有 LLM endpoint 全挂时使用保守默认值（75 而非 85），
        # 附带 degraded=True 标记供下游监控
        parsed = {
            "score": 75.0,
            "passed": True,
            "comment": f"LLM评审异常, 降级通过(degraded): {str(exc)[:100]}",
            "degraded": True,
        }

    review_result: dict[str, Any] = {
        "passed": parsed.get("passed", True),
        "score": float(parsed.get("score", 85.0)),
        "comment": f"{parsed.get('comment', '')} | 卷级评审完成 — {chapter_count} 章已收集",
        "chapter_range": (
            f"{chapter_indices[0]}-{chapter_indices[-1]}" if chapter_indices else "N/A"
        ),
        "collected_count": collected_count,
        "expected_count": expected_count,
        "thread_id": thread_id,
    }

    logger.info(
        "[volume_reviewer] LLM 卷级评审 — 范围 %s, 评分 %.1f, 通过=%s",
        review_result["chapter_range"],
        review_result["score"],
        review_result["passed"],
    )

    return {
        "volume_review_result": review_result,
        "current_phase": "writing",
    }
