"""Lead Agent Graph builder — DeerFlow-compatible conversational agent.

This is the main entry point for the conversational novel writing interface.
It wraps all nodes with the middleware chain.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from novelfactory.config.constants import RECURSION_LIMIT
from novelfactory.graph.chat.agents.chat_agent import chat_agent_node
from novelfactory.graph.chat.agents.review_agent import review_agent_node
from novelfactory.graph.chat.agents.story_agent import story_agent_node
from novelfactory.graph.chat.agents.writing_agent import writing_agent_node
from novelfactory.graph.chat.bridge import map_batch_to_lead, map_lead_to_batch
from novelfactory.middleware import get_lead_agent_middleware, with_middleware
from novelfactory.state.lead_agent_state import LeadAgentState

logger = logging.getLogger(__name__)


def build_lead_agent_graph(checkpointer: Any = None) -> CompiledStateGraph:
    """Build the Lead Agent graph — a conversational agent that delegates to sub-agents.

    This graph wraps all nodes with the middleware chain for input/output processing.
    """
    builder = StateGraph(LeadAgentState)

    # ── Build middleware chain ──
    _mw_chain = get_lead_agent_middleware()

    def _wrap(fn: Any) -> Any:
        """Wrap a node function with middleware if available."""
        if _mw_chain is not None:
            return with_middleware(fn, _mw_chain)
        return fn

    # ── Core nodes (wrapped with middleware) ──
    builder.add_node("chat_supervisor", _wrap(chat_supervisor_node))
    builder.add_node("story_agent", _wrap(story_agent_node))
    builder.add_node("writing_agent", _wrap(writing_agent_node))
    builder.add_node("review_agent", _wrap(review_agent_node))
    builder.add_node("chat_agent", _wrap(chat_agent_node))
    builder.add_node("save_memory", _wrap(save_memory_node))

    # ── Bridge Agent (batch pipeline delegation) ──
    builder.add_node("bridge_agent", _wrap(bridge_agent_node))

    if _mw_chain is not None:
        logger.info(
            "[LeadAgent] Middleware chain deployed to %d nodes (chain=%s)",
            len(builder.nodes) - 2,
            type(_mw_chain).__name__,
        )

    # ── Routing ──
    builder.add_conditional_edges(
        "chat_supervisor",
        _route_from_supervisor,
        {
            "story_agent": "story_agent",
            "writing_agent": "writing_agent",
            "review_agent": "review_agent",
            "chat_agent": "chat_agent",
            "bridge_agent": "bridge_agent",
            "save_memory": "save_memory",
        },
    )

    # Sub-agents return to supervisor
    for agent in ["story_agent", "writing_agent", "review_agent", "chat_agent", "bridge_agent"]:
        builder.add_edge(agent, "chat_supervisor")

    builder.add_edge("save_memory", END)
    builder.add_edge(START, "chat_supervisor")

    # ── Compile ──
    compiled = builder.compile(checkpointer=checkpointer)
    compiled.recursion_limit = RECURSION_LIMIT
    logger.info("[LeadAgent] Graph compiled with %d nodes", len(builder.nodes))
    return compiled


async def chat_supervisor_node(state: LeadAgentState) -> dict[str, Any]:
    """Chat supervisor node — analyzes user intent and routes to the right sub-agent.

    Delegates to :func:`analyze_intent` for slash-command, keyword, and LLM-based routing.
    """
    from novelfactory.graph.chat.supervisor import analyze_intent

    target_agent = await analyze_intent(dict(state))
    logger.info("[ChatSupervisor] Routed to: %s", target_agent)
    return {"current_agent": target_agent}


async def bridge_agent_node(state: LeadAgentState) -> dict[str, Any]:
    """Bridge agent — delegates to the batch processing pipeline.

    Maps LeadAgentState → NovelFactoryState via :func:`map_lead_to_batch`,
    invokes the batch graph (Writing Crew, Setup Crew), then maps results
    back to LeadAgentState via :func:`map_batch_to_lead` for conversational
    continuation.

    The batch graph is fetched lazily from the server's GraphRouter singleton
    to avoid circular imports at module load time.

    Args:
        state: Current LeadAgentState from the conversational graph.

    Returns:
        LeadAgentState updates including status message and mapped batch results.
    """
    from novelfactory.server.app import get_router

    logger.info("[BridgeAgent] Delegating to batch pipeline...")

    try:
        # Get the batch graph from the router
        router = await get_router()
        batch_graph = router.batch_graph
        if batch_graph is None:
            raise RuntimeError("Batch graph not available — server not fully initialized")

        # Map LeadAgentState → NovelFactoryState
        batch_input = map_lead_to_batch(dict(state))

        # Configure run with thread_id for state persistence
        config: RunnableConfig = {
            "configurable": {"thread_id": state.get("thread_id", "")},
            "recursion_limit": 5000,
        }

        # Invoke batch pipeline
        logger.info(
            "[BridgeAgent] Invoking batch pipeline with input keys: %s",
            list(batch_input.keys()),
        )
        result = await batch_graph.ainvoke(batch_input, config=config)

        # Map NovelFactoryState → LeadAgentState updates
        lead_updates = map_batch_to_lead(result)

        logger.info("[BridgeAgent] Batch pipeline completed successfully")
        return lead_updates

    except Exception as e:
        logger.exception("[BridgeAgent] Batch pipeline failed: %s", e)
        return {
            "current_agent": "chat_agent",
            "messages": [
                AIMessage(
                    content=(
                        f"❌ 自动创作遇到问题: {e}\n\n"
                        "你可以:\n"
                        "1. 继续对话式创作 — 在对话中调整需求\n"
                        "2. 重试自动生成 — 准备好后告诉我"
                    ),
                    name="bridge_agent",
                )
            ],
        }


def save_memory_node(state: LeadAgentState) -> dict[str, Any]:
    """Save memory and finalize."""
    return {"current_agent": "done"}


def _route_from_supervisor(state: LeadAgentState) -> str:
    """Route from supervisor to the appropriate sub-agent.

    Reads the ``current_agent`` field set by the previous node.
    If the sub-agent indicates it is done (current_agent == "done"),
    routes to ``save_memory``.  Otherwise, falls back to ``chat_agent``.
    """
    agent = state.get("current_agent", "chat_agent")
    if agent == "done":
        return "save_memory"
    # All sub-agents (story_agent, writing_agent, review_agent, chat_agent)
    # are valid routing targets.  The supervisor itself is never a valid target.
    if agent not in {"story_agent", "writing_agent", "review_agent", "chat_agent", "bridge_agent", "save_memory"}:
        logger.warning("[ChatSupervisor] Unknown agent '%s', routing to chat_agent", agent)
        return "chat_agent"
    return agent
