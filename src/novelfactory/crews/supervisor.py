"""Crew Supervisor factory and base classes.

Provides a reusable pattern for creating Crew supervisors without requiring
the langgraph-supervisor package. Each Crew consists of:
  - One supervisor node (LLM-driven routing)
  - One node per agent
  - Conditional edges from supervisor → agents
  - Return edges from agents → supervisor
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

# ── Crew State ─────────────────────────────────────────────────────────────────


class _CrewState(TypedDict, total=False):
    """Minimal state for a crew subgraph."""

    messages: Annotated[list, add_messages]
    current_agent: str
    crew_result: dict
    crew_error: str | None
    _iteration: int  # Supervisor loop counter — prevents infinite loops


# ── Supervisor Factory ──────────────────────────────────────────────────────────


def create_crew_supervisor(
    crew_name: str,
    agents: dict[str, Any],
    supervisor_prompt: str,
    model: Any,
    default_agent: str | None = None,
    checkpointer: Any = None,
) -> StateGraph:
    """Build a crew subgraph with a supervisor node that routes to agents.

    The supervisor is a simple LLM-driven router. It receives the crew's
    messages and decides which agent to call next.

    Args:
        crew_name: Human-readable name for logging.
        agents: Dict of {agent_name: agent_runnable}.
        supervisor_prompt: System prompt instructing the supervisor how to route.
        model: LLM used for routing decisions.
        default_agent: Agent to call on first turn if supervisor hasn't produced
            a routing decision yet.

    Returns:
        A compiled StateGraph that can be used as a node in the parent graph.
    """
    if not agents:
        raise ValueError(f"Crew '{crew_name}' must have at least one agent")

    graph = StateGraph(_CrewState)

    # Register all agent nodes
    for name in agents:
        graph.add_node(name, agents[name])

    from novelfactory.config.constants import (
        MAX_SUPERVISOR_ITERATIONS as _max_supervisor_iterations,  # noqa: N811
    )

    # Supervisor decision node
    def supervisor_node(state: _CrewState) -> dict:
        messages = state.get("messages", [])
        current = state.get("current_agent", "")
        iteration = state.get("_iteration", 0)

        # Guard against infinite loops — if we've iterated too many times, exit
        if iteration >= _max_supervisor_iterations:
            logging.getLogger(__name__).warning(
                "[supervisor] Crew '%s' hit iteration cap (%d), forcing exit",
                crew_name,
                _max_supervisor_iterations,
            )
            return {"current_agent": "", "crew_error": "max_iterations_reached"}

        # If we already have a routing decision, stay with it
        if current and current in agents:
            return {"_iteration": iteration + 1}

        # First turn or supervisor needs to re-route
        first_agent = default_agent or next(iter(agents))

        # Ask the LLM which agent to call
        try:
            response = model.invoke(
                [
                    *messages,
                    {"role": "system", "content": supervisor_prompt},
                ]
            )
            chosen = _parse_agent_from_response(response, list(agents.keys()))
        except Exception:
            logging.getLogger(__name__).warning(
                "[supervisor] LLM routing failed for crew '%s', defaulting to '%s'",
                crew_name,
                first_agent,
                exc_info=True,
            )
            chosen = first_agent

        return {"current_agent": chosen, "_iteration": iteration + 1}

    graph.add_node("supervisor", supervisor_node)

    # Agent → supervisor unconditional edges
    for name in agents:
        graph.add_edge(name, "supervisor")

    # supervisor → agents conditional routing
    def route_agent(state: _CrewState) -> str:
        agent = state.get("current_agent", "")
        if agent in agents:
            return agent
        return END

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_agent,
        {name: name for name in agents},
    )

    return graph.compile(
        checkpointer=checkpointer,
        # interrupt_before intentionally omitted — agents have interrupt_before=[] for tools,
        # and crew is always run to completion via _run_crew_until_done().
        # For human-in-the-loop review, use the root graph's wait_for_review node instead.
    )


def _parse_agent_from_response(response: Any, available: list[str]) -> str:
    """Extract an agent name from LLM text response."""
    content = ""
    if hasattr(response, "content"):
        content = response.content
    elif isinstance(response, str):
        content = response

    content_lower = content.lower()
    for name in available:
        if name.lower() in content_lower:
            return name

    return available[0]


# ── Handoff helpers ────────────────────────────────────────────────────────────


def crew_handoff(
    goto: str,
    crew_name: str,
    updates: dict,
) -> Command:
    """Return a Command that hands off to the parent graph.

    Used inside a crew node to jump to a named node in the parent (root) graph.

    Args:
        goto: Node name in the parent graph.
        crew_name: Name of the current crew (for logging).
        updates: State updates to merge before returning to parent.

    Returns:
        Command targeting the parent graph.
    """
    return Command(
        goto=goto,
        update={
            "crew_result": {
                "crew_name": crew_name,
                **updates,
            },
        },
        graph=Command.PARENT,
    )
