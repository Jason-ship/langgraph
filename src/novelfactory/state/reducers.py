"""LangGraph 自定义 Reducer 与状态合并工具。

从 ``novel_state.py`` 拆出，使 ``novel_state.py`` 聚焦于 TypedDict schema 定义，
reducer 逻辑独立可测。

包含：
  - ``_last_value``: last-write-wins 标量 reducer（修复 v3/v7 InvalidUpdateError）
  - ``_chapter_key``: 章节用量记录去重键
  - ``_add_usage``: total_usage 合并 + 去重 reducer
  - ``_add_chapters_compressed``: completed_chapters 合并 + 压缩 reducer
  - ``compress_completed_chapters``: 章节列表压缩工具
"""

from __future__ import annotations

from typing import Any

from novelfactory.config.constants import (
    COMPRESS_KEEP_RECENT_CHAPTERS,
    COMPRESS_OLD_TRUNC_LEN,
    COST_ROUND_DIGITS,
    DEEPSEEK_COMPLETION_TOKEN_RATE,
    DEEPSEEK_PROMPT_TOKEN_RATE,
    TOKENS_PER_MILLION,
)


def _last_value(existing: Any, update: Any) -> Any:
    """Last-write-wins reducer for multi-source scalar fields.

    Per LangGraph 1.1.10 official pattern (Annotated[T, reducer]):
    signature (existing, update) -> T. Returns the new value if present,
    otherwise preserves the existing one. Allows multiple nodes to write
    the same non-list field (e.g. thread_id, seed_idea, current_phase)
    in the same tick without raising InvalidUpdateError.

    P0-fix 2026-06-07: added to fix v3 (thread_id) and v7 (seed_idea)
    InvalidUpdateError on chapter 2 startup. See flight_mode_log.md
    for full root cause analysis.
    """
    return update if update is not None else existing


def _chapter_key(chapter_record: dict[str, Any]) -> str:
    """Build a unique key for a chapter usage record.

    The combination of (chapter_number, phase) identifies a single logical
    write. If a chapter is rewritten multiple times (e.g. because
    quality_score < 60 after rewrite), the new chapter_record should
    REPLACE the old one rather than append — so the key must not be
    "chapter_number" alone (which would overwrite legitimate distinct
    "setup"/"writing"/"refine" phases for the same chapter).
    """
    ch = chapter_record.get("chapter_number", 0)
    phase = chapter_record.get("phase", "unknown")
    return f"ch{ch}_{phase}"


def _add_usage(existing: dict, incoming: dict) -> dict:
    """Merge two total_usage dicts (for cross-chapter accumulation).

    Key fix (2026-06-15): chapter_usages is now de-duped using the
    (chapter_number, phase) tuple. A chapter rewrite (score < 60)
    produces a new chapter_record with the SAME chapter_number — the
    old one must be REPLACED, not appended. Otherwise the total token
    count keeps growing on every retry loop iteration, producing
    misleading /usage endpoint reports.

    Args:
        existing: previous total_usage dict (may be empty on first call).
        incoming: new data to merge in (expected to have "chapter_usages"
                  and "prompt_tokens" / "completion_tokens" fields).
    """
    if not incoming:
        return existing
    if existing is None:
        return incoming

    # ── Deduplicate chapter_usages by (chapter_number, phase) ───────────
    incoming_usages = incoming.get("chapter_usages") or []
    existing_usages = existing.get("chapter_usages") or []

    # Build dedup map from existing records
    dedup: dict = {}
    for rec in existing_usages:
        if not isinstance(rec, dict):
            continue
        key = _chapter_key(rec)
        dedup[key] = rec

    # Apply incoming records: if key exists, replace (newer is better),
    # otherwise append (new chapter / new phase).
    for rec in incoming_usages:
        if not isinstance(rec, dict):
            continue
        key = _chapter_key(rec)
        dedup[key] = rec

    chapter_usages_final = list(dedup.values())
    # Sort by (chapter_number, phase) for deterministic output
    chapter_usages_final.sort(
        key=lambda r: (
            int(r.get("chapter_number", 0)),
            r.get("phase", ""),
        )
    )

    # ── Recompute totals from chapter_usages (the canonical source) ────
    # This is MORE CORRECT than summing existing + incoming because
    # a rewrite replaces an older usage record — summing would double-count.
    prompt_total = 0
    completion_total = 0
    for rec in chapter_usages_final:
        prompt_total += int(rec.get("prompt_tokens", 0) or 0)
        completion_total += int(rec.get("completion_tokens", 0) or 0)

    # ── Merge model breakdown (keep latest per model, NOT summed) ────
    # Rationale: model_breakdown tracks per-model cumulative stats reported
    # by the LLM gateway. Each incoming report is a SNAPSHOT from the gateway,
    # not an incremental delta — so the latest value should replace, not be
    # added to the existing value. Adding would double-count on every tick.
    existing_models = existing.get("model_breakdown") or {}
    incoming_models = incoming.get("model_breakdown") or {}
    merged_models: dict = dict(existing_models)  # start with existing
    for model, i in incoming_models.items():
        merged_models[model] = i  # latest snapshot wins

    # Estimated cost: recompute from total tokens using DeepSeek pricing.
    estimated_cost = round(
        (prompt_total / TOKENS_PER_MILLION) * DEEPSEEK_PROMPT_TOKEN_RATE
        + (completion_total / TOKENS_PER_MILLION) * DEEPSEEK_COMPLETION_TOKEN_RATE,
        COST_ROUND_DIGITS,
    )

    return {
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "total_tokens": prompt_total + completion_total,
        "estimated_cost_cny": estimated_cost,
        "chapter_usages": chapter_usages_final,
        "model_breakdown": merged_models,
    }


def compress_completed_chapters(
    chapters: list, keep_recent: int = COMPRESS_KEEP_RECENT_CHAPTERS
) -> list:
    """压缩已完成章节列表，只保留最近N章的完整内容 + 早期章节的摘要。

    用于防止 1000+ 章时内存溢出。早期章节只保留标题和摘要，
    最近N章保留完整内容供 ContextBuilder 使用。

    Args:
        chapters: 章节列表，每个元素是 dict 或 str
        keep_recent: 保留最近几章的完整内容

    Returns:
        压缩后的章节列表
    """
    if len(chapters) <= keep_recent:
        return chapters
    recent = chapters[-keep_recent:]
    old = chapters[:-keep_recent]
    # 早期章节只保留标题
    compressed_old = []
    for ch in old:
        if isinstance(ch, dict):
            compressed_old.append(
                {
                    "chapter_number": ch.get("chapter_number"),
                    "title": ch.get("title", ""),
                    "compressed": True,  # 标记为已压缩
                }
            )
        else:
            # 字符串格式：取前 COMPRESS_OLD_TRUNC_LEN 字作为摘要
            compressed_old.append(
                ch[:COMPRESS_OLD_TRUNC_LEN] + "..."
                if len(ch) > COMPRESS_OLD_TRUNC_LEN
                else ch
            )
    return compressed_old + recent


def _add_chapters_compressed(existing: list, update: list) -> list:
    """Reducer for completed_chapters — merge + compress to prevent checkpoint bloat.

    Replaces operator.add for the parent graph's completed_chapters.
    After merging, applies compress_completed_chapters so that older chapter
    records are truncated to {chapter_number, title, compressed: True}.
    This prevents unbounded checkpoint growth over 1000+ chapter runs.

    Args:
        existing: previous accumulated list (already compressed by prior calls).
        update: new chapter records to append.

    Returns:
        Compressed merged list.
    """
    merged = (existing or []) + (update or [])
    return compress_completed_chapters(
        merged, keep_recent=COMPRESS_KEEP_RECENT_CHAPTERS
    )
