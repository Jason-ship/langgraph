"""Annotated Reducer 模式 — 优化 NovelFactory 状态管理。

参考 DeerFlow agents/thread_state.py 的 Reducer 设计模式。

为 NovelFactoryState 字段提供更健壮的合并策略：

- merge_todos: None = 未修改，非 None = 显式更新
- merge_delegations: 终态不可逆，上限 50 条
- merge_goal: 保留旧值（首次写入后不可覆盖）
- merge_artifacts: 合并去重保序
- merge_viewed_images: 空字典 = 清空信号
"""

from __future__ import annotations

from typing import Any

from langgraph.graph.message import add_messages  # noqa: F401 — re-export


def merge_todos(old: list[dict] | None, new: list[dict] | None) -> list[dict] | None:
    """Todo 合并器：None = 节点未修改，非 None = 显式更新。

    使用方式：
        from typing import Annotated
        todos: Annotated[list[dict], merge_todos]

    在 NovelFactoryState 中替代普通的 list[dict] 声明。
    """
    if new is None:
        return old
    return new


def merge_goal(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any] | None:
    """目标合并器：首次写入后不可覆盖。

    一旦 goal 被设置，后续节点写入 None 不会清除它。
    只有显式传入空 dict 才会清除。
    """
    if new is None:
        return old
    if new == {}:
        return None
    if old is None:
        return new
    # 已有目标时，只更新非 None 字段
    merged = dict(old)
    for k, v in new.items():
        if v is not None:
            merged[k] = v
    return merged


def merge_delegations(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """委托记录合并器：终态不可逆，上限 50 条。

    终态（completed/failed/cancelled）的记录不会被非终态记录覆盖。
    最新记录在前，最多保留 50 条。
    """
    _terminal_statuses = {"completed", "failed", "cancelled"}
    _max_delegations = 50

    if not old:
        old = []
    if not new:
        new = []

    # 建立索引：task_id → 记录
    old_by_id: dict[str, dict] = {}
    for item in old:
        tid = item.get("task_id") or item.get("id", "")
        if tid:
            old_by_id[tid] = item

    result = list(old)
    for item in new:
        tid = item.get("task_id") or item.get("id", "")
        if not tid:
            result.append(item)
            continue

        existing = old_by_id.get(tid)
        if existing:
            # 终态不可逆
            if existing.get("status") in _terminal_statuses:
                continue
            # 替换
            idx = next((i for i, r in enumerate(result) if (r.get("task_id") or r.get("id", "")) == tid), None)
            if idx is not None:
                result[idx] = item
        else:
            result.append(item)

    # 上限 50 条，最新在前
    if len(result) > _max_delegations:
        result = result[-_max_delegations:]
    return result


def merge_artifacts(old: list[str] | None, new: list[str] | None) -> list[str]:
    """产物合并器：合并去重保序。"""
    combined = list(old or []) + list(new or [])
    seen: set[str] = set()
    result: list[str] = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def merge_quality_scores(old: dict[str, float] | None, new: dict[str, float] | None) -> dict[str, float]:
    """质量评分合并器：相同键取最新值。

    用于追踪每次评审的评分变化。
    """
    merged = dict(old or {})
    if new:
        merged.update(new)
    return merged


def _add_chapters_compressed(
    old: list[dict] | None, new: list[dict] | None
) -> list[dict]:
    """Chapter reducer with compression - append new chapters, compress old ones.

    v6.1: Optimized to avoid O(N²) list rebuild on every call. Only appends
    new records to the end and delegates compression to
    ``compress_completed_chapters`` when the list exceeds
    ``COMPRESS_KEEP_RECENT_CHAPTERS``. Already-compressed records are skipped
    during compression to prevent redundant reprocessing.
    """
    result = list(old or [])
    if new:
        result.extend(new)

    from novelfactory.config.constants import COMPRESS_KEEP_RECENT_CHAPTERS

    if len(result) > COMPRESS_KEEP_RECENT_CHAPTERS:
        return compress_completed_chapters(result, COMPRESS_KEEP_RECENT_CHAPTERS)
    return result


def _last_value(old: Any, new: Any) -> Any:
    """Reducer that keeps the last non-None value.

    Standard LangGraph pattern for scalar fields that may be written by
    multiple nodes in the same tick. Prevents InvalidUpdateError.
    """
    if new is None:
        return old
    return new


def _chapter_key(chapter: dict) -> str:
    """Build dedup key from (chapter_number, phase) for usage records.

    Returns a string like ``"ch3_writing"`` so records from the same
    chapter+phase can be deduplicated (newest wins).
    """
    return f"ch{int(chapter.get('chapter_number', 0))}_{chapter.get('phase', 'unknown')}"


def compress_completed_chapters(
    completed: list[dict], keep_recent: int = 50
) -> list[dict]:
    """Compress completed chapters list - keep recent N full, truncate older ones.

    Older chapters are reduced to {chapter_number, chapter_summary} only.
    Already-compressed records (containing only chapter_number + chapter_summary
    keys) are passed through without reprocessing to avoid redundant work.
    """
    if len(completed) <= keep_recent:
        return completed
    recent = completed[-keep_recent:]
    older = completed[:-keep_recent]
    compressed = []
    for ch in older:
        if isinstance(ch, dict):
            # Skip already-compressed records (only chapter_number + chapter_summary)
            if set(ch.keys()) <= {"chapter_number", "chapter_summary"}:
                compressed.append(ch)
            else:
                compressed.append({
                    "chapter_number": ch.get("chapter_number", "?"),
                    "chapter_summary": (ch.get("chapter_summary", "") or "")[:200],
                })
        else:
            compressed.append(ch)
    return compressed + recent


def _add_usage(old: dict | None, new: dict | None) -> dict:
    """Token usage accumulator — merges chapter usages with dedup + recompute.

    v6.0: Replaces operator.add for total_usage to prevent duplicate
    accumulation when subgraph states merge.
    v8.0-fix: Dedup on (chapter_number, phase) with newest-wins, then
    recompute prompt/completion/total tokens from the merged chapter_usages
    to eliminate double-counting on chapter rewrites. Sorted by
    (chapter_number, phase) for deterministic output.
    """
    old = old or {}
    new = new or {}
    if not new:
        # 返回浅拷贝而非原引用，避免 reducer 结果与 state 中旧对象共享可变引用
        return dict(old)

    # 1. 合并 chapter_usages — 按 (chapter_number, phase) 去重，新记录覆盖旧记录
    merged_usages: dict[str, dict] = {}
    for usage in (old.get("chapter_usages") or []) + (new.get("chapter_usages") or []):
        if isinstance(usage, dict):
            merged_usages[_chapter_key(usage)] = usage

    usages = sorted(
        merged_usages.values(),
        key=lambda u: (int(u.get("chapter_number", 0)), str(u.get("phase", ""))),
    )

    # 2. 重算 totals — 从去重后的 chapter_usages 重新求和
    prompt_tokens = sum(int(u.get("prompt_tokens") or 0) for u in usages)
    completion_tokens = sum(int(u.get("completion_tokens") or 0) for u in usages)
    total_tokens = prompt_tokens + completion_tokens

    # 3. 合并 model_breakdown — 同模型最新记录覆盖旧记录
    old_breakdown = old.get("model_breakdown") or {}
    new_breakdown = new.get("model_breakdown") or {}
    merged_breakdown = dict(old_breakdown)
    for model, data in new_breakdown.items():
        merged_breakdown[model] = dict(data) if isinstance(data, dict) else data

    return {
        "chapter_usages": usages,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model_breakdown": merged_breakdown,
        # 顶层费用字段保留累加语义（与 chapter_usages 独立记账）
        "estimated_cost_cny": (old.get("estimated_cost_cny") or 0.0)
        + (new.get("estimated_cost_cny") or 0.0),
    }


__all__ = [
    "add_messages",
    "merge_todos",
    "merge_goal",
    "merge_delegations",
    "merge_artifacts",
    "merge_quality_scores",
    "_add_chapters_compressed",
    "_add_usage",
    "_last_value",
    "_chapter_key",
    "compress_completed_chapters",
]
