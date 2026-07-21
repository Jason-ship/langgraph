# ==============================================================================
# SDK: /assistants
# ==============================================================================

from __future__ import annotations

import json as _json
import logging

from fastapi import APIRouter, HTTPException

from novelfactory.config.settings import settings
from novelfactory.server.models import Assistant
from novelfactory.server.serialization import _MessageJSONEncoder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/assistants", tags=["assistants"])
async def list_assistants() -> dict:
    """List available assistants — both batch and conversational."""
    # Batch assistant (existing)
    batch = Assistant(
        assistant_id="novelfactory",
        name="NovelFactory (批处理)",
        description="全自动小说创作流水线",
    )
    # Lead agent assistant (new)
    lead = Assistant(
        assistant_id="lead_agent",
        name="Lead Agent (对话式)",
        description="对话式小说创作助手",
    )
    return {"assistants": [batch.model_dump(), lead.model_dump()]}


@router.post("/assistants/search", tags=["assistants"])
async def search_assistants() -> dict:
    """Search/list assistants (SDK compatibility)."""
    batch = Assistant(
        assistant_id="novelfactory",
        name="NovelFactory (批处理)",
        description="全自动小说创作流水线",
    )
    lead = Assistant(
        assistant_id="lead_agent",
        name="Lead Agent (对话式)",
        description="对话式小说创作助手",
    )
    return {"assistants": [batch.model_dump(), lead.model_dump()]}


@router.get("/assistants/{assistant_id}", tags=["assistants"])
async def get_assistant(assistant_id: str) -> dict:
    """Get assistant details."""
    if assistant_id == "lead_agent":
        return Assistant(
            assistant_id="lead_agent",
            name="Lead Agent (对话式)",
            description="对话式小说创作助手",
        ).model_dump()
    if assistant_id in ("novelfactory", "agent"):
        return Assistant().model_dump()
    raise HTTPException(status_code=404, detail="Assistant not found")


@router.post("/assistants", tags=["assistants"])
async def create_assistant() -> dict:
    """Create an assistant (stub — single assistant mode)."""
    return Assistant().model_dump()


@router.patch("/assistants/{assistant_id}", tags=["assistants"])
async def update_assistant(assistant_id: str) -> dict:
    """Update assistant (stub)."""
    if assistant_id not in ("novelfactory", "agent", "lead_agent"):
        raise HTTPException(status_code=404, detail="Assistant not found")
    return Assistant().model_dump()


@router.delete("/assistants/{assistant_id}", tags=["assistants"])
async def delete_assistant(assistant_id: str) -> dict:
    """Delete assistant (stub)."""
    return {"deleted": assistant_id}


@router.get("/assistants/{assistant_id}/graph", tags=["assistants"])
async def get_assistant_graph(assistant_id: str, xray: bool | int = False) -> dict:
    """Get serialized graph structure from the compiled graph."""
    if assistant_id not in ("novelfactory", "agent", "lead_agent"):
        raise HTTPException(status_code=404, detail="Assistant not found")
    try:
        from novelfactory.server.deps import get_graph_router

        router = await get_graph_router()
        graph = router.get_graph(assistant_id)
        xray_depth = int(xray) if xray else 0
        g = graph.get_graph(xray=xray_depth)
        if hasattr(g, "to_json"):
            return g.to_json()
        return {"nodes": [], "edges": []}
    except Exception as e:
        logger.warning("[graph] Failed to get graph structure: %s", e)
        return {"nodes": [], "edges": []}


@router.get("/assistants/{assistant_id}/schemas", tags=["assistants"])
async def get_assistant_schemas(assistant_id: str) -> dict:
    """Get graph and config schemas from the compiled graph."""
    if assistant_id not in ("novelfactory", "agent", "lead_agent"):
        raise HTTPException(status_code=404, detail="Assistant not found")

    try:
        from novelfactory.server.deps import get_graph_router

        router = await get_graph_router()
        graph = router.get_graph(assistant_id)
        return {
            "graph": {
                "input": _json.loads(
                    _json.dumps(
                        graph.input_schema.model_json_schema(), cls=_MessageJSONEncoder
                    )
                ),
                "output": _json.loads(
                    _json.dumps(
                        graph.output_schema.model_json_schema(), cls=_MessageJSONEncoder
                    )
                ),
            },
            "config": {},
            "metadata": {
                "name": getattr(graph, "name", assistant_id),
                "nodes": list((getattr(graph, "nodes", {}) or {}).keys()),
            },
        }
    except Exception as e:
        logger.warning("[schemas] Failed to get schemas: %s", e)
        return {"graph": {}, "config": {}, "metadata": {}}


@router.get("/assistants/{assistant_id}/subgraphs", tags=["assistants"])
async def get_assistant_subgraphs(assistant_id: str) -> list:
    """Get subgraphs — lists all compiled subgraphs from the graph."""
    if assistant_id not in ("novelfactory", "agent", "lead_agent"):
        raise HTTPException(status_code=404, detail="Assistant not found")

    try:
        from novelfactory.server.deps import get_graph_router

        router = await get_graph_router()
        graph = router.get_graph(assistant_id)
        subgraphs = []
        for node_name, node_def in (getattr(graph, "nodes", {}) or {}).items():
            subs = getattr(node_def, "subgraphs", [])
            if subs:
                subgraphs.append(
                    {
                        "name": node_name,
                        "subgraph_id": f"{assistant_id}/{node_name}",
                        "metadata": {},
                    }
                )
        return subgraphs
    except Exception as e:
        logger.warning("[subgraphs] Failed to list subgraphs: %s", e)
        return []


@router.get(
    "/assistants/{assistant_id}/subgraphs/{namespace:path}", tags=["assistants"]
)
async def get_assistant_subgraph(assistant_id: str, namespace: str) -> dict:
    """Get a specific subgraph's serialized graph structure."""
    if assistant_id not in ("novelfactory", "agent", "lead_agent"):
        raise HTTPException(status_code=404, detail="Assistant not found")

    try:
        from novelfactory.server.deps import get_graph_router

        router = await get_graph_router()
        graph = router.get_graph(assistant_id)
        node_def = (getattr(graph, "nodes", {}) or {}).get(namespace)
        if node_def is None:
            return {}
        subs = getattr(node_def, "subgraphs", [])
        if not subs:
            return {}
        g = subs[0].get_graph(xray=1)
        if hasattr(g, "to_json"):
            return g.to_json()
        return {}
    except Exception as e:
        logger.warning("[subgraphs] Failed to get subgraph %s: %s", namespace, e)
        return {}


@router.post("/assistants/{assistant_id}/versions", tags=["assistants"])
async def list_assistant_versions(assistant_id: str) -> list:
    """List assistant versions — returns the current deployed version."""
    if assistant_id not in ("novelfactory", "agent", "lead_agent"):
        raise HTTPException(status_code=404, detail="Assistant not found")

    try:
        from novelfactory.server.deps import get_graph_router

        router = await get_graph_router()
        graph = router.get_graph(assistant_id)
        assistant = Assistant()
        return [
            {
                "version": settings.APP_VERSION,
                "assistant_id": assistant_id,
                "name": assistant.name,
                "description": assistant.description,
                "created_at": assistant.created_at,
                "graph_name": getattr(graph, "name", assistant_id),
                "node_count": len(getattr(graph, "nodes", {}) or {}),
                "is_current": True,
            }
        ]
    except Exception:
        return []


@router.post("/assistants/{assistant_id}/latest", tags=["assistants"])
async def set_assistant_latest(assistant_id: str) -> dict:
    """Switch assistant version (stub)."""
    return Assistant().model_dump()
