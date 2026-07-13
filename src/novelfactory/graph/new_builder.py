"""Main LangGraph builder - NovelFactory Multi-Agent Architecture (v4.1).

Architecture:
  - Main Supervisor (root coordinator)
  - Setup: lightweight sequential pipeline with hierarchical outline generation
  - Writing/Media/Sync Crews: native compiled subgraphs via add_node()
  - ContextBuilder/StateExtractor/DatabaseWriter subgraphs integrated
  - volume_check / quality_check / foreshadowing_check: Phase 2/3 checks
    split from monolithic post_sync_check (300万字生产保障)
  - wait_for_review node (standard interrupt() pattern)

Changes in v4.0 (2026-06-17):
  - Native LangGraph: add_node(compiled_subgraph) replaces _run_crew_until_done
  - Command.PARENT propagates automatically from subgraphs to root graph
  - ContextBuilder/StateExtractor/DatabaseWriter subgraphs replace
    NovelStateTracker.before/after_chapter() calls
  - prepare_writing node assembles crew_result before writing_crew
  - Crew subgraphs compiled without checkpointer (parent handles persistence)

Changes in v4.1 (2026-06-17):
  - Split monolithic post_sync_check into 3 independent nodes:
    • volume_check: Volume transition detection + transition context injection
      + lazy outline generation (volume-by-volume, solves max_tokens limit)
    • quality_check: Quality decay detection + corrective guidance injection
    • foreshadowing_check: Foreshadowing enforcement (overdue + end-game)
  - Hierarchical outline: story → volume → chapter (setup generates first 3
    volumes, remaining generated on-demand by volume_check)
  - Volume structure persisted to database during setup
  - auto_guidance field propagates interventions to writing_crew

Changes in v4.2 (2026-06-23):
  - Refactored: node functions extracted to graph/nodes/*.py and graph/routing.py
  - new_builder.py now only contains build_novel_factory_graph() + compile_app()
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from novelfactory.config.constants import (
    RECURSION_LIMIT,
)
from novelfactory.graph.checkpointer import (
    create_checkpointer,
    create_store,
    set_checkpointer_instance,
)
from novelfactory.graph.crews.media_crew import build_media_crew
from novelfactory.graph.crews.sync_crew import build_sync_crew
from novelfactory.graph.crews.writing_crew import build_writing_crew
from novelfactory.graph.lightweight_setup import build_setup_crew
from novelfactory.graph.monitor_node import intelligent_monitor_node
from novelfactory.graph.node_specs import PHASE_CHECK_SPECS
from novelfactory.graph.nodes.memory import load_longterm_memory, save_longterm_memory
from novelfactory.graph.nodes.prepare_writing import prepare_writing_node
from novelfactory.graph.nodes.quota import refresh_quota_node
from novelfactory.graph.nodes.review import chapter_human_guidance, wait_for_review_node
from novelfactory.graph.nodes.supervisor import main_supervisor_node
from novelfactory.graph.routing import route_from_supervisor, route_phase_check_chain
from novelfactory.middleware import get_middleware_chain, with_middleware
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Note: All node functions have been extracted to separate modules:
#   - graph/routing.py:          route_from_supervisor, _resolve_target_chapters
#   - graph/nodes/quota.py:      refresh_quota_node
#   - graph/nodes/memory.py:     load_longterm_memory, save_longterm_memory
#   - graph/nodes/supervisor.py: main_supervisor_node
#   - graph/lightweight_setup.py: build_setup_crew (compiled subgraph)
#   - graph/nodes/prepare_writing.py: prepare_writing_node, feishu_upload_node
#   - graph/nodes/phase_checks.py: volume_check_node, quality_check_node,
#                                  foreshadowing_check_node,
#                                  _generate_next_volume_outlines
#   - graph/nodes/review.py:     wait_for_review_node, chapter_human_guidance
# ═══════════════════════════════════════════════════════════════════════════════


# ── Graph Builder ─────────────────────────────────────────────────────────────


def build_novel_factory_graph() -> StateGraph:
    """Build the root StateGraph with Main Supervisor + native compiled subgraphs.

    Native LangGraph migration (2026-06-17):
      - Crews (writing/media/sync) added as compiled subgraphs via add_node()
      - Command.PARENT from subgraph internal nodes propagates automatically
      - _run_crew_until_done manual loop removed
      - prepare_writing node assembles crew_result before writing_crew
    """
    graph = StateGraph(NovelFactoryState)

    # ── Root Supervisor ──
    graph.add_node("main_supervisor", main_supervisor_node)

    # ── Setup (compiled subgraph, v4.3) ──
    graph.add_node("setup_crew", build_setup_crew())

    # ── Prepare writing input ──
    graph.add_node("prepare_writing", prepare_writing_node)

    # ── Phase 2 / Phase 3 Checks (v4.1+ → v5.5 NodeSpec dynamic) ──
    # Dynamic node registration via PHASE_CHECK_SPECS registry.
    # Genre-aware filtering happens in route_from_supervisor (routing.py).
    for spec in PHASE_CHECK_SPECS:
        graph.add_node(spec.key, spec.node_fn)

    # ── Crew subgraphs (native add_node with compiled subgraphs) ──
    # When internal nodes return Command.PARENT, LangGraph propagates
    # the command to the parent graph automatically.
    # v6.1: 通过 with_middleware 包装中间件链
    _mw_chain = get_middleware_chain()
    graph.add_node("writing_crew", with_middleware(build_writing_crew(), _mw_chain))
    graph.add_node("media_crew", with_middleware(build_media_crew(), _mw_chain))
    graph.add_node("sync_crew", with_middleware(build_sync_crew(), _mw_chain))

    # ── Long-Term Memory ──
    graph.add_node("load_memory", load_longterm_memory)
    graph.add_node("save_memory", save_longterm_memory)

    # ── Quota ──
    graph.add_node("refresh_quota", refresh_quota_node)

    # ── Interrupt / Recovery ──
    graph.add_node("wait_for_review", wait_for_review_node)
    graph.add_node("chapter_human_guidance", chapter_human_guidance)

    # ── Intelligent Monitor (v5.0+) ──
    # Inserts LLM-powered monitoring between writing_crew and main_supervisor
    graph.add_node("intelligent_monitor", intelligent_monitor_node)

    # Feishu upload now handled by sync_crew subgraph internally (v5.4+)

    # ── Edges ──
    graph.add_edge(START, "main_supervisor")

    # Dynamic routing map from main_supervisor (v5.5)
    # Phase check nodes are derived from PHASE_CHECK_SPECS.
    _route_map: dict[str, str] = {
        "setup_crew": "setup_crew",
        "load_memory": "load_memory",
        "save_memory": "save_memory",
        "refresh_quota": "refresh_quota",
        "writing_crew": "writing_crew",
        "media_crew": "media_crew",
        "sync_crew": "sync_crew",
        "wait_for_review": "wait_for_review",
        "chapter_human_guidance": "chapter_human_guidance",
    }
    for spec in PHASE_CHECK_SPECS:
        _route_map[spec.key] = spec.key
    _route_map[END] = END

    graph.add_conditional_edges(
        "main_supervisor",
        route_from_supervisor,
        _route_map,
    )

    graph.add_edge("setup_crew", "main_supervisor")
    graph.add_edge("prepare_writing", "writing_crew")
    # Phase check chain: dynamic conditional routing (v5.5)
    # Each check node routes to next in chain or refresh_quota, based on genre.
    _check_chain_keys = [s.key for s in PHASE_CHECK_SPECS] + [
        "refresh_quota",
        "main_supervisor",
    ]
    _check_chain_map = {k: k for k in _check_chain_keys}
    for spec in PHASE_CHECK_SPECS:
        graph.add_conditional_edges(spec.key, route_phase_check_chain, _check_chain_map)
    graph.add_edge("load_memory", "main_supervisor")
    graph.add_edge("save_memory", END)
    graph.add_edge("refresh_quota", "prepare_writing")
    # writing_crew → intelligent_monitor → main_supervisor
    graph.add_edge("writing_crew", "intelligent_monitor")
    graph.add_edge("intelligent_monitor", "main_supervisor")
    graph.add_edge("media_crew", "main_supervisor")
    # sync_crew → main_supervisor (feishu upload in sync_crew subgraph)
    graph.add_edge("sync_crew", "main_supervisor")
    graph.add_edge("wait_for_review", "main_supervisor")
    graph.add_edge("chapter_human_guidance", "main_supervisor")

    return graph


async def compile_app(
    checkpointer: Any = None, store: Any = None
) -> CompiledStateGraph:
    """Compile the graph into an executable app."""
    logger.info("[compile_app] Building graph from %s", __file__)
    graph = build_novel_factory_graph()
    kwargs: dict[str, Any] = {}

    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    else:
        kwargs["checkpointer"] = await create_checkpointer()

    # Share the checkpointer instance for runtime GC use (agc_checkpoints etc.)
    set_checkpointer_instance(kwargs["checkpointer"])

    if store is not None:
        kwargs["store"] = store
    else:
        kwargs["store"] = await create_store()

    # No interrupt_before — using standard interrupt() inside nodes
    # allowed_msgpack_modules is configured at the checkpointer serde level
    # (checkpointer.py), not at graph.compile().
    app = graph.compile(**kwargs)
    # Override default recursion_limit (25) for long-running production flows.
    # Config-level recursion_limit may be ignored by subgraph invocations
    # and astream_events; setting on the compiled object ensures propagation.
    app.recursion_limit = RECURSION_LIMIT

    logger.info(
        "[app] Compiled — checkpointer=%s, store=%s",
        type(kwargs["checkpointer"]).__name__,
        type(kwargs["store"]).__name__,
    )
    return app


def create_dev_graph():
    """Create a lightweight graph for dev mode.

    langgraph dev mode factory: returns the un-compiled StateGraph,
    letting the in-memory runtime handle compilation with its own checkpointer.
    """
    return build_novel_factory_graph()
