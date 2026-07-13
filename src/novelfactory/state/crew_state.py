"""Crew-local state definitions.

These TypedDicts are used only within individual Crew subgraphs and are not
persisted to the global NovelFactoryState (only overlapping keys propagate).

Design principle:
    All subgraph states share a ``BaseCrewState`` with the ``messages``
    and ``crew_result`` fields.  This follows the official LangGraph
    ``MessagesState`` pattern (``message.py`` L372-373):

        class MessagesState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

    Each Crew subgraph extends ``BaseCrewState`` with domain-specific fields.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# ── Base Crew State (shared by all subgraph states) ─────────────────────────────


class BaseCrewState(TypedDict):
    """Minimal shared state for all Crew subgraphs.

    All three Crew subgraphs (writing, media, sync) extend this state.
    The ``messages`` field uses the official ``add_messages`` reducer,
    matching ``NovelFactoryState.messages`` for automatic subgraph→parent
    message accumulation.
    """

    messages: Annotated[list, add_messages]
    crew_result: dict
    crew_error: str | None


# ── Project Context ─────────────────────────────────────────────────────────────


class ProjectContext(TypedDict):
    """Project metadata shared between Crews."""

    project_id: str
    project_name: str
    genre: str
    target_chapters: int
    current_chapter: int
