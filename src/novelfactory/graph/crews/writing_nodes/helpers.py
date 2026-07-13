"""Helper functions extracted from writing_crew.py.

- _sanitize_human_guidance: 清洗用户输入，防止 Prompt 注入攻击
- _make_record: 构建 completed_chapters 中的章节记录
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _sanitize_human_guidance(guidance: str) -> str:
    """清洗用户输入，防止 Prompt 注入攻击。

    移除潜在的注入模式：分隔符、指令覆盖、角色切换等。
    """
    if not guidance:
        return ""
    # 移除常见的注入分隔符
    guidance = re.sub(r"={3,}", "", guidance)
    guidance = re.sub(r"-{3,}", "", guidance)
    # 移除指令覆盖模式 (支持 "ignore all previous instructions" 等多词组合)
    guidance = re.sub(
        r"(?i)(ignore|disregard|override|forget)\s+[\w\s]*?\b(instructions?|prompts?)",
        "[FILTERED]",
        guidance,
    )
    # 移除角色切换模式
    guidance = re.sub(
        r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)",
        "[FILTERED]",
        guidance,
    )
    # 截断过长输入
    from novelfactory.config.constants import GUIDANCE_MAX_LENGTH

    if len(guidance) > GUIDANCE_MAX_LENGTH:
        guidance = guidance[:GUIDANCE_MAX_LENGTH] + "..."
    return guidance.strip()


def _make_record(
    number: int,
    text: str,
    score: float,
    chapter_summary: str | None = None,
    review_result: dict | None = None,
    composite_score: float = 0.0,
) -> dict:
    """Build a chapter record for completed_chapters.

    Stores metadata only — NO full chapter text. Full text lives in PostgreSQL
    chapters table (written by database_writer_node). This keeps checkpoint
    state small (~KB per chapter) instead of accumulating MB per chapter
    and crashing PostgreSQL with "invalid memory alloc request size".

    v4.2 fix (2026-06-20): removed refined_chapter (full text) from record.
    Before this fix, 15 chapters = 211MB state → PG allocation error at 1GB.
    Now 600 chapters ≈ ~300KB state.

    v6.1: composite_score renamed to programmatic_score for consistency
    with VerdictResult schema.

    v7.4-fix: chapter_summary 参数 — 由调用方 LLM 生成摘要，
    不再简单截断前 200 字。调用方未传入时回退截断。
    """
    record: dict = {
        "chapter_number": number,
        "chapter_summary": chapter_summary or text[:200].replace("\n", " ").strip(),
        "quality_score": score,
        "word_count": len(text),
    }
    # v5.0: Store composite score in record
    if review_result:
        record["review_result_snapshot"] = {
            "quality_score": review_result.get("quality_score", score),
            "review_comments": review_result.get("review_comments", ""),
            "needs_refine": review_result.get("needs_refine", False),
        }
    # v6.1: renamed from composite_score → programmatic_score
    record["programmatic_score"] = composite_score
    return record
