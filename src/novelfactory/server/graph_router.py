"""Graph Router — dispatches between batch and conversational graphs.

The core fusion component that enables both modes to coexist:
- assistant_id == "novelfactory" → batch graph (NovelFactoryState)
- assistant_id == "lead_agent" → conversational graph (LeadAgentState)

Both graphs share the same checkpointer and store for seamless state sharing.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class GraphRouter:
    """Holds both compiled graphs and dispatches based on assistant_id.

    This is the central fusion point that allows the batch processing pipeline
    and the conversational Lead Agent to coexist in the same server.
    """

    def __init__(
        self,
        batch_graph: CompiledStateGraph | None = None,
        lead_graph: CompiledStateGraph | None = None,
    ) -> None:
        self._batch_graph = batch_graph
        self._lead_graph = lead_graph
        self._default = batch_graph

    @property
    def batch_graph(self) -> CompiledStateGraph | None:
        return self._batch_graph

    @batch_graph.setter
    def batch_graph(self, graph: CompiledStateGraph) -> None:
        self._batch_graph = graph
        if self._default is None:
            self._default = graph

    @property
    def lead_graph(self) -> CompiledStateGraph | None:
        return self._lead_graph

    @lead_graph.setter
    def lead_graph(self, graph: CompiledStateGraph) -> None:
        self._lead_graph = graph

    def get_graph(self, assistant_id: str = "novelfactory") -> CompiledStateGraph:
        """Get the appropriate graph for the given assistant_id.

        Args:
            assistant_id: "novelfactory" for batch, "lead_agent" for conversational

        Returns:
            The compiled graph for the given assistant_id.

        Raises:
            RuntimeError: If no graph is available.
        """
        if assistant_id == "lead_agent":
            if self._lead_graph is None:
                logger.warning("[GraphRouter] Lead graph not available, falling back to batch")
                return self._get_default()
            return self._lead_graph
        elif assistant_id in ("novelfactory", "agent", ""):
            return self._get_default()
        else:
            # Try lead graph for unknown assistants (extensible)
            if self._lead_graph is not None:
                logger.info("[GraphRouter] Unknown assistant '%s', trying lead graph", assistant_id)
                return self._lead_graph
            return self._get_default()

    def _get_default(self) -> CompiledStateGraph:
        if self._batch_graph is None:
            raise RuntimeError("No graph available — server not fully initialized")
        return self._batch_graph

    @property
    def store(self) -> Any:
        """Get the shared store from whichever graph has it."""
        if self._batch_graph is not None and hasattr(self._batch_graph, "store"):
            return self._batch_graph.store
        if self._lead_graph is not None and hasattr(self._lead_graph, "store"):
            return self._lead_graph.store
        return None

    @property
    def checkpointer(self) -> Any:
        """Get the shared checkpointer from whichever graph has it."""
        if self._batch_graph is not None and hasattr(self._batch_graph, "checkpointer"):
            if self._batch_graph.checkpointer is not None:
                return self._batch_graph.checkpointer
        if self._lead_graph is not None and hasattr(self._lead_graph, "checkpointer"):
            return self._lead_graph.checkpointer
        return None