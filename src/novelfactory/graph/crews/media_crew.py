"""Media Crew entry node for the NovelFactory graph.

State machine (v5.5 tool self-loop):

    START ──→ illustrator ──┐
                           ├──→ media_supervisor ──→ _media_tool_router
    START ──→ tts_generator ┘                         │
                                         ┌──────────────┼──────────────┐
                                         ▼              ▼              ▼
                                    _parallel_      _parallel_        END
                                    media_node      media_node
                                    (插图失败重试)  (配音失败重试)

illustrator and tts_generator run in parallel via ThreadPoolExecutor.
v5.5: Added tool self-loop — failed illustration/audio generation
retries up to 3 times before giving up.

Usage:
    from novelfactory.graph.crews.media_crew import build_media_crew
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from novelfactory.agents.infra import cleanup_crew_stream, get_crew_stream
from novelfactory.agents.media_agents import (
    create_illustrator_agent,
    create_tts_generator_agent,
)
from novelfactory.config.constants import SUBGRAPH_RECURSION_LIMIT
from novelfactory.config.llm import get_worker_llm
from novelfactory.state.crew_state import BaseCrewState

logger = logging.getLogger(__name__)


# ── Local State ────────────────────────────────────────────────────────────────


class MediaCrewLocalState(BaseCrewState):
    """Local state for the Media Crew subgraph (extends BaseCrewState).

    Inherits ``messages``, ``crew_result``, ``crew_error`` from BaseCrewState.
    """

    # Parallel results
    illustration_url: str
    illustration_prompt: str
    audio_url: str

    # v5.5: Tool self-loop retry tracking
    media_retry_count: int
    illustration_error: str
    audio_error: str


# ── Parallel Execution Node ─────────────────────────────────────────────────────


def _parallel_media_node(state: MediaCrewLocalState) -> dict[str, Any]:
    """Run illustrator and tts_generator in parallel via asyncio.gather.

    This is the single entry point — both agents execute concurrently.
    Returns plain dict with illustration_url, illustration_prompt, audio_url.

    Streaming progress is written to a temp file via StreamWriter so the user
    can watch image generation and TTS synthesis in real time.
    """
    current_ch = state.get("crew_result", {}).get("current_chapter_number", 1)

    # Module-level StreamWriter (never stored in state, avoids msgpack crash)
    cr = state.get("crew_result", {})
    sw = get_crew_stream("media", f"ch{current_ch}")

    sw.section(f"第{current_ch}章 - 媒体生成")
    sw.write("[media] 开始生成插图 + 配音...\n")

    # Prepare inputs for both agents
    illustrator_input = {"crew_result": cr}
    tts_input = {"crew_result": cr}

    # Create agent nodes (they are factories — call them to get the runnable)
    illustrator_agent = create_illustrator_agent(get_worker_llm())
    tts_agent = create_tts_generator_agent(get_worker_llm())

    # ── Phase 1: parallel generation ────────────────────────────────────────────
    # Refactor 2026-06-01: sync runnables don't need an asyncio event loop.
    # Use ThreadPoolExecutor for true concurrent execution (no event loop
    # creation, no RuntimeError handling, no loop leak on Windows).
    sw.write("[media] 生成插图中...\n")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="media_crew") as pool:
        future_ill = pool.submit(illustrator_agent, illustrator_input)
        future_tts = pool.submit(tts_agent, tts_input)
        illustrator_result = future_ill.result()
        tts_result = future_tts.result()

    illustration_url = illustrator_result.get("illustration_url", "")
    illustration_prompt = illustrator_result.get("illustration_prompt", "")
    audio_url = tts_result.get("audio_url", "")

    # ── Phase 2: log results ────────────────────────────────────────────────────
    ill_status = "✓" if illustration_url else "✗"
    audio_status = "✓" if audio_url else "✗"
    sw.write(f"[media] 插图生成 {ill_status} | 配音生成 {audio_status}\n")

    logger.info(
        "[media_crew] Parallel media complete: illustration=%s, audio=%s",
        bool(illustration_url),
        bool(audio_url),
    )

    return {
        "crew_result": {
            **cr,
            "illustration_url": illustration_url,
            "illustration_prompt": illustration_prompt,
            "audio_url": audio_url,
        },
        "messages": [
            AIMessage(
                content=f"第{current_ch}章媒体生成：插图{'✓' if illustration_url else '✗'} 配音{'✓' if audio_url else '✗'}",
                name="media_crew",
            )
        ],
    }


# ── Media Supervisor Node ───────────────────────────────────────────────────────


def _media_supervisor_node(state: MediaCrewLocalState) -> dict[str, Any]:
    """Coordinate media results and exit the subgraph.

    v5.5: Added error detection and retry count tracking for tool self-loop.
    If illustration or audio fails, sets error flags and increments retry count.
    _media_tool_router will decide whether to retry or exit.

    Closes the StreamWriter and writes a final summary to the temp file.
    """
    logger.info("[media_crew] media_supervisor node")

    cr = state.get("crew_result", {})
    illustration_url = cr.get("illustration_url", "")
    illustration_prompt = cr.get("illustration_prompt", "")
    audio_url = cr.get("audio_url", "")
    current_ch = cr.get("current_chapter_number", 1)
    retry_count = state.get("media_retry_count", 0)

    # v5.5: Error detection
    illustration_error = "" if illustration_url else "插图生成失败（无URL）"
    audio_error = "" if audio_url else "配音生成失败（无URL）"

    # Close StreamWriter and write final summary (module-level cache)
    sw = get_crew_stream("media", f"ch{current_ch}")
    ill_ok = bool(illustration_url)
    audio_ok = bool(audio_url)
    sw.write(
        f"[media] 第{current_ch}章媒体生成完成 — "
        f"插图: {'✓' if ill_ok else '✗'} | 配音: {'✓' if audio_ok else '✗'}\n"
    )
    sw.section(f"第{current_ch}章 - 完成")
    cleanup_crew_stream("media", f"ch{current_ch}")

    # Log media generation results
    if illustration_url:
        logger.info(
            "[media_crew] Chapter %s illustration URL: %s",
            current_ch,
            illustration_url[:80],
        )
    if audio_url:
        logger.info("[media_crew] Chapter %s audio URL: %s", current_ch, audio_url[:80])

    # Update crew_result with media outputs
    updated_crew_result = {
        **cr,
        "illustration_url": illustration_url,
        "illustration_prompt": illustration_prompt,
        "audio_url": audio_url,
        "illustration_error": illustration_error,
        "audio_error": audio_error,
        "media_retry_count": retry_count,
        "media_complete": ill_ok and audio_ok,
    }

    # v5.5: Increment retry count if there are errors
    next_retry = retry_count + 1 if (illustration_error or audio_error) else 0

    # Return state updates as plain dict (subgraph will exit to END or retry)
    return {
        "crew_result": {
            "crew_name": "media",
            **updated_crew_result,
        },
        "media_retry_count": next_retry,
        "illustration_error": illustration_error,
        "audio_error": audio_error,
        "messages": [
            AIMessage(
                content=f"第{current_ch}章媒体生成{'完成' if ill_ok and audio_ok else '失败(retry=' + str(next_retry) + '/3)'}",
                name="media_supervisor",
            )
        ],
    }


# ── Media Tool Router ──────────────────────────────────────────────────────


def _media_tool_router(state: MediaCrewLocalState) -> str:
    """v5.5: 工具调用后路由 — 成功→继续，失败→重试（最多 3 次）。

    Returns:
        "_parallel_media_node" if retry needed, otherwise END key
    """
    cr = state.get("crew_result", {})
    illustration_error = cr.get("illustration_error", "")
    audio_error = cr.get("audio_error", "")
    retry_count = state.get("media_retry_count", 0)

    has_error = bool(illustration_error) or bool(audio_error)

    if has_error and retry_count < 3:
        logger.warning(
            "[media_crew] 媒体生成失败(retry=%d/%d), 重试 — ill_err=%s, audio_err=%s",
            retry_count,
            3,
            bool(illustration_error),
            bool(audio_error),
        )
        return "_parallel_media_node"

    if has_error:
        logger.error(
            "[media_crew] 媒体生成重试耗尽(retry=%d), 放弃 — ill_err=%s, audio_err=%s",
            retry_count,
            bool(illustration_error),
            bool(audio_error),
        )

    logger.info("[media_crew] 媒体生成完成或放弃, 退出子图")
    return "END"


# ── Graph Builder ─────────────────────────────────────────────────────────────


def build_media_crew(_checkpointer: Any = None) -> Any:
    """Build the Media Crew StateGraph.

    Architecture (v5.5 tool self-loop):

      START ──→ _parallel_media_node ──→ _media_supervisor_node
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              _parallel_media_node       END
                              (retry up to 3 times)

    illustrator and tts_generator run concurrently in _parallel_media_node.
    If either fails, _media_tool_router retries up to 3 times.

    Returns:
        Compiled StateGraph.  Add as a node in the root graph:
            graph.add_node("media_crew", build_media_crew())
    """
    graph = StateGraph(MediaCrewLocalState)

    graph.add_node("_parallel_media_node", _parallel_media_node)
    graph.add_node("_media_supervisor_node", _media_supervisor_node)

    # Entry → parallel execution
    graph.add_edge(START, "_parallel_media_node")

    # Parallel → supervisor (unconditional)
    graph.add_edge("_parallel_media_node", "_media_supervisor_node")

    # v5.5: Supervisor → tool router (conditional retry loop)
    graph.add_conditional_edges(
        "_media_supervisor_node",
        _media_tool_router,
        {
            "_parallel_media_node": "_parallel_media_node",
            "END": END,
        },
    )

    # Native add_node: compile without checkpointer.
    # The parent graph's checkpointer handles all persistence.
    compiled = graph.compile()
    compiled.recursion_limit = SUBGRAPH_RECURSION_LIMIT
    return compiled
