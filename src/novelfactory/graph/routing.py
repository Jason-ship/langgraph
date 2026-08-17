"""Main Supervisor routing: decide which Crew/phase to invoke next."""

from __future__ import annotations

import logging

from langgraph.graph import END

from novelfactory.config.constants import (
    FALLBACK_TARGET_CHAPTERS,
    PHASE_DONE,
    PHASE_MEDIA,
    PHASE_SETUP,
    PHASE_SYNC,
    PHASE_WRITING,
)
from novelfactory.graph.node_specs import PHASE_CHECK_SPECS, build_check_chain
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)

# ── Check node → status field 显式映射 ──────────────────────────────────────
# 消除 route_phase_check_chain 中的隐式字段名匹配。
# 键 = node key (与 PHASE_CHECK_SPECS.key 对齐)
# 值 = state 中的状态字段名
_CHECK_STATUS_FIELDS: dict[str, str] = {
    "volume_check": "volume_status",
    "quality_check": "quality_trend",
    "foreshadowing_check": "foreshadowing_status",
}

# ── Routing Functions ──────────────────────────────────────────────────────────


def _resolve_target_chapters(state: NovelFactoryState) -> int:
    """Resolve ``target_chapters`` from state, falling back through multiple sources.

    Priority:
      1. ``target_chapters`` key in state (set via input or graph)
      2. ``project_context.target_chapters`` (persisted project config)
      3. Hard default: 600 (适用于长篇网文)
    """
    if state.get("target_chapters"):
        return state["target_chapters"]
    pc = state.get("project_context")
    # v8.5-fix: ProjectContext 为 TypedDict — 兼容 dict 索引与对象属性访问
    if pc:
        if isinstance(pc, dict):
            return pc.get("target_chapters") or FALLBACK_TARGET_CHAPTERS
        return getattr(pc, "target_chapters", 0) or FALLBACK_TARGET_CHAPTERS
    return FALLBACK_TARGET_CHAPTERS  # fallback


def _resolve_genre(state: NovelFactoryState) -> str:
    """Extract genre from project_context and resolve to standard genre name."""
    from novelfactory.config.constants import resolve_genre

    pc = state.get("project_context")
    raw_genre = ""
    if isinstance(pc, dict):
        raw_genre = pc.get("genre", "")
    elif pc is not None:
        raw_genre = getattr(pc, "genre", "")
    return resolve_genre(raw_genre)


def _find_in_chain(chain: list[str], key: str) -> int:
    """Find the index of a key in a chain, returning -1 if not found."""
    try:
        return chain.index(key)
    except ValueError:
        return -1


def route_phase_check_chain(state: NovelFactoryState) -> str:
    """Route to next phase check node or refresh_quota based on genre and progress.

    Called from conditional_edges on each phase check node.
    Uses the explicit _CHECK_STATUS_FIELDS mapping to determine which check
    just completed, then routes to the next node in the genre-filtered chain.

    v8.5-fix: 新增 make_check_chain_router(node_key) 按"当前完成节点"精确路由，
    替代本函数基于跨章持久化状态字段的进度推断（volume_status/quality_trend/
    foreshadowing_status 跨章残留导致第 2 章起误判链尾，后续检查被跳过）。
    新图构建统一使用 make_check_chain_router；本函数保留供兼容/测试。
    """
    genre = _resolve_genre(state)
    check_chain = build_check_chain(PHASE_CHECK_SPECS, genre)

    if not check_chain:
        return "refresh_quota"

    # Find the last completed check via explicit field mapping (not implict kw naming)
    pos = -1
    for node_key, status_field in _CHECK_STATUS_FIELDS.items():
        # 按 check_chain 顺序遍历，找到第一个有状态值的节点
        if node_key in check_chain and state.get(status_field):
            pos = _find_in_chain(check_chain, node_key)

    if pos >= 0 and pos + 1 < len(check_chain):
        return check_chain[pos + 1]

    return "refresh_quota"


def make_check_chain_router(node_key: str):
    """为指定检查节点绑定独立路由闭包（v8.5-fix，S8）。

    直接从 node_key 在链中的位置路由到下一节点，不依赖跨章持久化的
    状态字段（volume_status/quality_trend/foreshadowing_status 会跨章残留，
    旧推断逻辑在第 2 章起误判全部完成 → quality/foreshadowing 检查被跳过）。
    """
    def _route(state: NovelFactoryState) -> str:
        genre = _resolve_genre(state)
        check_chain = build_check_chain(PHASE_CHECK_SPECS, genre)
        if not check_chain:
            return "refresh_quota"
        pos = _find_in_chain(check_chain, node_key)
        if pos >= 0 and pos + 1 < len(check_chain):
            return check_chain[pos + 1]
        return "refresh_quota"
    return _route


def route_from_supervisor(state: NovelFactoryState) -> str:
    """Main Supervisor routing: decide which Crew to invoke next."""
    phase = state.get("current_phase", PHASE_SETUP)
    setup_complete = state.get("setup_complete", False)
    next_node = END

    # v8.5-fix: setup_aborted 全局拦截（原实现仅 PHASE_SETUP 分支检查，
    # checkpoint 恢复 / 其他 phase 入口可绕过 fail-closed → 空设定进入写作）。
    # 当再次携带有效 seed 发起时，init_setup_node 会复位该标记。
    if state.get("setup_aborted"):
        logger.warning(
            "[supervisor] setup_aborted=True → routing to END "
            "(需要携带 seed_idea 重新发起运行)"
        )
        return END

    if phase == PHASE_SETUP:
        if not setup_complete:
            next_node = "setup_crew"
        elif state.get("pending_review") == "kickoff":
            next_node = "wait_for_review"
        else:
            next_node = "load_memory"

    elif phase == PHASE_WRITING:
        if state.get("chapter_needs_guidance") and not state.get("guidance_complete"):
            next_node = "chapter_human_guidance"
        elif state.get("pending_review") == "chapter":
            next_node = "wait_for_review"
        elif state.get("current_chapter", 1) > 1:
            genre = _resolve_genre(state)
            check_chain = build_check_chain(PHASE_CHECK_SPECS, genre)
            if check_chain:
                next_node = check_chain[0]
            else:
                next_node = "refresh_quota"
        else:
            next_node = "refresh_quota"

    elif phase == PHASE_MEDIA:
        next_node = "media_crew"
    elif phase == PHASE_SYNC:
        next_node = "sync_crew"

    elif phase == PHASE_DONE:
        next_node = "save_memory"

    else:
        # v5.9 FIX: 未知 phase → 警告并路由到 END，避免无声终止
        logger.warning(
            "[supervisor] UNKNOWN phase=%s → routing to END (check for typo or missing handler)",
            phase,
        )
        next_node = END

    logger.info(
        "[supervisor] phase=%s setup_complete=%s chapter=%s review=%s → %s",
        phase,
        setup_complete,
        state.get("current_chapter"),
        state.get("pending_review"),
        next_node,
    )
    return next_node
