"""Sync Crew entry node for the NovelFactory graph.

State machine (v5.5 tool self-loop):

    START ──→ feishu_sync ──→ _sync_tool_router ──→ [feishu_sync(retry) | state_update]
                                                           │
                                            state_update ──→ _exit_node ──→ END

v5.5: feishu upload failure auto-retry up to 3 times via _sync_tool_router.
_exit_node reads current_chapter_number vs target_chapters to decide
whether to continue writing or return to the main supervisor.

Usage:
    from novelfactory.graph.crews.sync_crew import build_sync_crew
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from novelfactory.agents.infra import (
    cleanup_crew_stream,
    count_tokens,
    get_crew_stream,
)
from novelfactory.agents.sync_agents import (
    create_feishu_sync_agent,
    update_project_state,
)
from novelfactory.config.constants import (
    FALLBACK_TARGET_CHAPTERS,
    SUBGRAPH_RECURSION_LIMIT,
)
from novelfactory.config.llm import get_supervisor_llm
from novelfactory.state.crew_state import BaseCrewState

logger = logging.getLogger(__name__)

_COMPACT_THRESHOLD = 8  # 超过此章节数开始压缩旧章节详细内容
_COMPRESS_AT_COUNT = 20  # 超过此章节数开始压缩 completed_chapters 列表
# ── Local State ────────────────────────────────────────────────────────────────


class SyncCrewLocalState(BaseCrewState):
    """Local state for the Sync Crew subgraph (extends BaseCrewState).

    Inherits ``messages``, ``crew_result``, ``crew_error`` from BaseCrewState.
    """

    feishu_doc_url: str
    state_updated: bool
    folder_tokens: dict  # 飞书目录结构（从父图继承）
    # v5.1.1-fix: 从父图继承 total_usage（否则 _get_context 的 fallback
    # state.get("total_usage", {}) 在 sync 子图内返回空 dict）
    total_usage: dict

    # v5.5: Tool self-loop retry tracking
    feishu_retry_count: int
    feishu_upload_error: str
    # v6.0.1: 飞书重试耗尽标记 — 供监控/报警使用
    feishu_sync_exhausted: bool


# ── Node: feishu_sync ─────────────────────────────────────────────────────────


def _feishu_sync_node(state: SyncCrewLocalState) -> dict[str, Any]:
    """Upload chapter content, illustrations, and audio to Feishu.

    v5.5: Added retry tracking — sets feishu_upload_error and increments
    feishu_retry_count on failure. _sync_tool_router decides retry/continue.

    Returns plain dict with feishu_doc_url.
    Skips immediately if Feishu is not configured.
    """
    cr = state.get("crew_result", {})
    current_ch = cr.get("current_chapter_number", "?")

    sw = get_crew_stream("sync", f"ch{current_ch}")
    logger.info(f"[sync_crew] 开始同步第{current_ch}章")

    # Skip Feishu sync if not configured (no LARK_APP_ID)
    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st

    if not (_st.LARK_APP_ID or os.environ.get("LARK_APP_ID", "")):
        sw.write("[sync_crew] 飞书未配置，跳过同步\n")
        logger.info(
            "[sync_crew] 飞书未配置(LARK_APP_ID为空)，跳过第%s章同步", current_ch
        )
        return {
            "feishu_doc_url": "",
            "feishu_upload_error": "",
            "feishu_retry_count": 0,
        }

    sw.section(f"第{current_ch}章 - 飞书同步")
    sw.write(f"[sync_crew] 开始上传第{current_ch}章内容到飞书...\n")
    logger.info("[sync_crew] 开始上传第%s章到飞书", current_ch)

    # Create and invoke FeishuSync agent
    # Pass the FULL subgraph state (incl. folder_tokens) so the agent can
    # upload docs & files to the correct Feishu project folders.
    feishu_agent = create_feishu_sync_agent(get_supervisor_llm())
    result = feishu_agent.invoke(state)

    # feishu agent 返回的 feishu_doc_url 可能在顶层也可能在 crew_result 中
    feishu_doc_url = str(
        result.get("feishu_doc_url")
        or (result.get("crew_result", {}) or {}).get("feishu_doc_url", "")
    )
    # v5.4-fix: 同步 sync agent 返回的最新 folder_tokens（含新建的卷文件夹）
    agent_cr = result.get("crew_result", {}) or {}
    updated_folder_tokens = (
        agent_cr.get("folder_tokens")
        or result.get("folder_tokens")
        or state.get("folder_tokens", {})
    )

    # v5.5: Error tracking for tool self-loop
    prev_retry = state.get("feishu_retry_count", 0)
    upload_error = ""

    if feishu_doc_url and feishu_doc_url != "None":
        sw.write(f"[sync_crew] 飞书文档已更新: {feishu_doc_url[:80]}...\n")
        logger.info("[sync_crew] 第%s章飞书同步成功: %s", current_ch, feishu_doc_url)
        upload_error = ""
        next_retry = 0
    else:
        sw.write("[sync_crew] 飞书同步失败，请检查网络或权限\n")
        upload_error = "飞书上传失败（无有效URL返回）"
        next_retry = prev_retry + 1
        logger.warning(
            "[sync_crew] 第%s章飞书同步失败(retry=%d/3, feishu_doc_url=%r, result keys=%s)",
            current_ch,
            next_retry,
            feishu_doc_url,
            list(result.keys()) if isinstance(result, dict) else type(result).__name__,
        )

    return {
        "feishu_doc_url": feishu_doc_url,
        "folder_tokens": updated_folder_tokens,
        "feishu_upload_error": upload_error,
        "feishu_retry_count": next_retry,
        "messages": [
            AIMessage(
                content=f"第{current_ch}章飞书同步{'完成' if feishu_doc_url else '跳过（未配置）'}",
                name="feishu_sync",
            )
        ],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """M3-8: Accurate token count using tiktoken (with regex fallback).

    Delegates to count_tokens() from agents/infra which uses tiktoken when
    available, falling back to the regex heuristic for CJK text.

    This replaces the previous inline regex-based heuristic.
    """
    return count_tokens(text)


def _compact_completed_chapters(
    completed: list[dict], *, token_budget: int = 50_000
) -> list[dict]:
    """Compress early chapters to keep state within token budget.

    Strategy (token-aware, replaces hard chapter-count threshold):
      - Estimate total tokens in completed_chapters field
      - Only compact when total exceeds token_budget (default 50k)
      - Recent 5 chapters: keep full record
      - Early chapters: keep only metadata (chapter_number + summary[:500])
      - Never compact below 3 full records (preserve recent context)

    Claude Code research: context window at 95% triggers auto-compact.
    We budget ~50k tokens for completed_chapters (~1/3 of 128k context).
    """
    if not completed:
        return completed

    # Estimate total token footprint of all chapter records
    total = sum(
        _estimate_tokens(
            str(ch.get("chapter_summary", ""))
            + str(ch.get("chapter_draft", ""))
            + str(ch.get("refined_chapter", ""))
        )
        for ch in completed
    )

    if total <= token_budget:
        return completed

    # Keep recent N full records (never fewer than 3)
    recent = completed[-5:] if len(completed) >= _COMPACT_THRESHOLD else completed[-3:]
    # P0 5 fix (2026-06-03): the early-chapter dict previously dropped
    # refined_chapter/chapter_draft entirely. Downstream code uses
    # `cr.get("refined_chapter") or cr.get("chapter_draft", "")` (P0 9
    # fallback pattern), so missing fields would silently substitute "" —
    # misleading the chapter_writer on next iteration. Keep a truncated
    # preview (first 200 chars) of refined_chapter/chapter_draft so the
    # downstream fallback can still find something coherent, and add an
    # explicit `compacted: True` marker so audit logs can tell.
    early = [
        {
            "chapter_number": ch.get("chapter_number"),
            "chapter_summary": (ch.get("chapter_summary") or "")[:1000],
            "quality_score": ch.get("quality_score", 0),
            "refined_chapter": (ch.get("refined_chapter") or "")[:200],
            "chapter_draft": (ch.get("chapter_draft") or "")[:200],
            "compacted": True,
        }
        for ch in completed[: -len(recent)]
    ]

    # Early chapters: only the metadata we just built (chapter_number/summary/score).
    # The full refined_chapter/chapter_draft fields are NOT in `early` (we just
    # stripped them above), so the token estimate must only count what's there.
    new_total = sum(
        _estimate_tokens(str(ch.get("chapter_summary", ""))) for ch in early
    ) + sum(
        _estimate_tokens(
            str(ch.get("chapter_summary", ""))  # full summary (up to 200 chars)
            + str(ch.get("refined_chapter") or ch.get("chapter_draft", ""))
        )
        for ch in recent
    )

    logger.info(
        "[sync_crew] Compacting %d chapters (~%s→~%s tokens) → %d metadata + %d full",
        len(completed),
        f"{total:,}",
        f"{new_total:,}",
        len(early),
        len(recent),
    )
    return early + recent


def _state_update_node(state: SyncCrewLocalState) -> dict[str, Any]:
    """Persist crew results to the global checkpointer.

    This is a pure utility node — no LLM involved.
    Returns plain dict with state_updated flag.

    M3-7: Calls checkpoint GC after each chapter to prune old checkpoints
    (keeps the 5 most recent, deletes older ones).
    """
    cr = state.get("crew_result", {})
    feishu_doc_url = state.get("feishu_doc_url", "")
    thread_id = cr.get("thread_id", "")
    current_ch = cr.get("current_chapter_number", "?")

    sw = get_crew_stream("sync", f"ch{current_ch}")
    sw.section("状态持久化")
    sw.write("[sync_crew] 保存项目状态到 checkpointer...\n")

    # Merge feishu_doc_url into crew_result before persistence
    final_result = {
        **cr,
        "feishu_doc_url": feishu_doc_url,
        "sync_complete": True,
    }

    # Compress completed_chapters if list is too long
    completed = final_result.get("completed_chapters", [])
    if len(completed) > _COMPRESS_AT_COUNT:
        final_result["completed_chapters"] = _compact_completed_chapters(completed)

    update_result = update_project_state(
        thread_id=thread_id,
        state_updates={"crew_result": final_result},
    )

    state_updated = update_result.get("state_updated", False)

    if state_updated:
        sw.write(f"[sync_crew] 状态已保存，共完成 {len(completed)} 章\n")
    else:
        sw.write("[sync_crew] 状态保存失败\n")
    cleanup_crew_stream("sync", f"ch{current_ch}")

    return {
        "state_updated": state_updated,
        "crew_result": final_result,
    }


# ── Exit Node ────────────────────────────────────────────────────────────────


def _exit_node(state: SyncCrewLocalState) -> dict[str, Any]:
    """Exit the Sync Crew and return state updates to the parent graph.

    This is the ONLY crew-exit point. State updates are merged into the
    parent graph's root state. Routing is determined by the parent graph's
    main_supervisor node via current_phase / current_chapter.
    """
    cr = state.get("crew_result", {})
    current_ch = cr.get("current_chapter_number", 1)
    target = cr.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
    feishu_doc_url = state.get("feishu_doc_url", "")

    if current_ch < target:
        # More chapters remain — increment chapter number and hand off to writing
        next_ch = current_ch + 1

        sw = get_crew_stream("sync", f"ch{current_ch}")
        sw.write(f"[sync_crew] 第{current_ch}章同步完成 → 进入第{next_ch}章写作\n\n")

        return {
            "crew_result": {
                **cr,
                "crew_name": "sync",
                "current_chapter_number": next_ch,
                "target_chapters": target,
                "feishu_doc_url": feishu_doc_url,
                "completed_chapters": cr.get("completed_chapters", []),
                "last_synced_chapter": current_ch,
            },
            "messages": [
                AIMessage(
                    content=f"第{current_ch}章同步完成 → 进入第{next_ch}章写作",
                    name="sync_crew",
                )
            ],
        }
    # All chapters complete — hand off to main supervisor
    sw = get_crew_stream("sync", f"ch{current_ch}")
    sw.write(f"[sync_crew] 全部 {target} 章同步完成 → 进入完成阶段\n\n")
    cleanup_crew_stream("sync", f"ch{current_ch}")

    return {
        "crew_result": {
            **cr,
            "crew_name": "sync",
            "writing_complete": True,
            "media_complete": True,
            "sync_complete": True,
            "completed_chapters": cr.get("completed_chapters", []),
            "feishu_doc_url": feishu_doc_url,
        },
        "messages": [
            AIMessage(
                content=f"全部 {target} 章同步完成！写作结束。",
                name="sync_crew",
            )
        ],
    }


# ── Sync Tool Router ───────────────────────────────────────────────────────


def _sync_tool_router(state: SyncCrewLocalState) -> str:
    """v5.5: 飞书上传路由 — 失败自动重试（最多 3 次）。

    Returns:
        "feishu_sync" if retry needed, "state_update" if OK or exhausted
    """
    upload_error = state.get("feishu_upload_error", "")
    retry_count = state.get("feishu_retry_count", 0)

    if upload_error and retry_count <= 2:
        logger.warning(
            "[sync_crew] 飞书上传失败(retry=%d/3), 重试上传 — err=%s",
            retry_count,
            upload_error,
        )
        return "feishu_sync"

    if upload_error:
        logger.error(
            "[sync_crew] 飞书上传重试耗尽(retry=%d), 放弃 — err=%s",
            retry_count,
            upload_error,
        )
        # v6.0.1: 不再静默跳过 — 设置 feishu_sync_exhausted 标记供监控/报警使用
        return "state_update"

    logger.info("[sync_crew] 飞书上传成功或放弃, 继续 state_update")
    return "state_update"


# ── Graph Builder ─────────────────────────────────────────────────────────────


def build_sync_crew(_checkpointer: Any = None) -> Any:
    """Build the Sync Crew StateGraph.

    Architecture (sequential with conditional exit):

      START ──→ feishu_sync ──→ state_update ──→ _exit_node ──→ END
                                                         │
                                        ├── _goto="writing_crew"  (more chapters)
                                        └── _goto="main_supervisor" (all chapters done)

    The exit node returns a plain dict with a `_goto` routing hint that the
    parent graph reads to decide the next step.

    Returns:
        Compiled StateGraph.  Add as a node in the root graph:
            graph.add_node("sync_crew", build_sync_crew())
    """
    graph = StateGraph(SyncCrewLocalState)

    graph.add_node("feishu_sync", _feishu_sync_node)
    graph.add_node("state_update", _state_update_node)

    # Entry
    graph.add_edge(START, "feishu_sync")

    # v5.5: feishu_sync → _sync_tool_router (conditional: retry or continue)
    graph.add_conditional_edges(
        "feishu_sync",
        _sync_tool_router,
        {
            "feishu_sync": "feishu_sync",
            "state_update": "state_update",
        },
    )

    # state_update → _exit_node (subgraph exit; parent graph picks up state updates)
    graph.add_node("_exit_node", _exit_node)
    graph.add_edge("state_update", "_exit_node")
    graph.add_edge("_exit_node", END)

    # Native add_node: compile without checkpointer.
    # The parent graph's checkpointer handles all persistence.
    compiled = graph.compile()
    compiled.recursion_limit = SUBGRAPH_RECURSION_LIMIT
    return compiled
