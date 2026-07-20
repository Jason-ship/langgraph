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

from langgraph.graph.message import add_messages

# 重新导出 LangGraph 内置 Reducer
add_messages = add_messages


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
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
    MAX_DELEGATIONS = 50

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
            if existing.get("status") in TERMINAL_STATUSES:
                continue
            # 替换
            idx = next((i for i, r in enumerate(result) if (r.get("task_id") or r.get("id", "")) == tid), None)
            if idx is not None:
                result[idx] = item
        else:
            result.append(item)

    # 上限 50 条，最新在前
    if len(result) > MAX_DELEGATIONS:
        result = result[-MAX_DELEGATIONS:]
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
    """Chapter reducer with compression — append new chapters, compress old ones.

    v6.0: Replaces operator.add to prevent unbounded checkpoint growth in 1000+
    chapter novels. Keeps only the most recent N chapters full, compresses older
    entries to chapter_summary only.
    """
    result = list(old or [])
    if new:
        result.extend(new)

    from novelfactory.config.constants import COMPRESS_KEEP_RECENT_CHAPTERS

    if len(result) > COMPRESS_KEEP_RECENT_CHAPTERS:
        # Keep recent chapters full, compress older ones to summary-only
        recent = result[-COMPRESS_KEEP_RECENT_CHAPTERS:]
        old_compressed = result[:-COMPRESS_KEEP_RECENT_CHAPTERS]
        compressed = []
        for ch in old_compressed:
            if isinstance(ch, dict):
                compressed.append({
                    "chapter_number": ch.get("chapter_number", "?"),
                    "chapter_summary": (ch.get("chapter_summary", "") or "")[:200],
                })
            else:
                compressed.append(ch)
        return compressed + recent
    return result


__all__ = [
    "add_messages",
    "merge_todos",
    "merge_goal",
    "merge_delegations",
    "merge_artifacts",
    "merge_quality_scores",
    "_add_chapters_compressed",
]