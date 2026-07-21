"""Lead Agent chat graph package.

The Lead Agent is a conversational agent that delegates to sub-agents
for story planning, writing, review, and general chat.
"""

from novelfactory.graph.chat.builder import build_lead_agent_graph

__all__ = [
    "build_lead_agent_graph",
]
