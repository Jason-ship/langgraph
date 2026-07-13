"""Main Supervisor node — top-level phase machine orchestrator."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from novelfactory.graph.routing import _resolve_target_chapters
from novelfactory.state.novel_state import NovelFactoryState

logger = logging.getLogger(__name__)


def main_supervisor_node(state: NovelFactoryState) -> dict:
    """Main Supervisor entry point. Orchestrates the top-level phase machine."""
    phase = state.get("current_phase", "setup")
    logger.info(f"[main_supervisor] phase={phase}")

    updates: dict = {}

    if phase == "setup":
        if not state.get("setup_complete"):
            updates["current_phase"] = "setup"
        elif state.get("pending_review") == "kickoff":
            pass
        else:
            updates["current_phase"] = "writing"

    elif phase == "writing":
        if state.get("pending_review") == "chapter":
            pass
        elif state.get("chapter_approved"):
            # Chapter writing complete — check if media generation is needed
            if state.get("has_media"):
                logger.info(
                    "[main_supervisor] chapter_approved (ch=%d), has_media → media",
                    state.get("current_chapter"),
                )
                updates["current_phase"] = "media"
            else:
                updates["current_phase"] = "sync"
        else:
            # ── target_chapters may be unset in state; fall back to project_context
            target = _resolve_target_chapters(state)
            if state.get("current_chapter", 1) > target:
                logger.info(
                    "[main_supervisor] All chapters done (ch=%d > target=%d)",
                    state.get("current_chapter"),
                    target,
                )
                if state.get("has_media"):
                    updates["current_phase"] = "media"
                else:
                    updates["current_phase"] = "sync"
                updates["target_chapters"] = target
            else:
                updates["current_phase"] = "sync"

    elif phase == "media":
        if state.get("media_complete"):
            logger.info(
                "[main_supervisor] media_complete (ch=%d), → sync",
                state.get("current_chapter"),
            )
            updates["current_phase"] = "sync"
        else:
            # media_crew hasn't completed yet; stay in media phase
            # (route_from_supervisor will route to media_crew again)
            logger.info(
                "[main_supervisor] media not yet complete (ch=%d), staying in media",
                state.get("current_chapter"),
            )

    elif phase == "sync":
        current = state.get("current_chapter", 1)
        target = _resolve_target_chapters(state)
        if current >= target:
            updates["current_phase"] = "done"
        else:
            updates["current_phase"] = "writing"
    elif phase == "done":
        # ── Resume from "done": if more chapters remain, restart writing ──
        target = _resolve_target_chapters(state)
        if state.get("current_chapter", 1) <= target:
            updates["current_phase"] = "sync"
            logger.info(
                "[main_supervisor] done→sync (ch=%d, target=%d, resuming)",
                state.get("current_chapter"),
                target,
            )
        else:
            logger.info(
                "[main_supervisor] done, all chapters complete (ch=%d/%d)",
                state.get("current_chapter"),
                target,
            )
    else:
        logger.warning(
            "[main_supervisor] UNKNOWN phase=%s — unexpected state, routing will handle",
            phase,
        )

    # ── Phase transition message for Chat UI ───────────────────────────────
    # Only emits when phase transitions occur (setup→writing, writing→sync, etc.)
    # Routine re-entries (same phase) are no-ops.
    from novelfactory.config.constants import MAX_MESSAGES, PHASE_LABELS

    previous_phase = state.get("_last_supervisor_phase", "")
    new_phase = updates.get("current_phase", phase)
    if new_phase != previous_phase and new_phase:
        label = PHASE_LABELS.get(new_phase, f"阶段: {new_phase}")
        ch = state.get("current_chapter", 1)
        total = _resolve_target_chapters(state)
        updates.setdefault("messages", []).append(
            AIMessage(
                content=f"**{label}**（第{ch}/{total}章）",
                name="supervisor",
            )
        )
        updates["_last_supervisor_phase"] = new_phase

    # ── Messages field compression cap ──────────────────────────────────────
    messages_state = state.get("messages", [])
    if len(messages_state) > MAX_MESSAGES:
        existing_msgs = updates.get("messages", [])
        updates["messages"] = messages_state[-MAX_MESSAGES:] + existing_msgs
        logger.info(
            "[main_supervisor] Compressed messages %d→%d",
            len(messages_state),
            MAX_MESSAGES,
        )

    return updates
