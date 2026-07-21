"""Agent management API — dynamic CRUD for novel-writing agents."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from novelfactory.agents.registry import AgentDefinition, AgentRegistry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agents"])


class CreateAgentRequest(BaseModel):
    name: str
    description: str = ""
    model: str = "deepseek-chat"
    tool_groups: list[str] = []
    skills: list[str] = []
    soul: str = ""


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    model: str | None = None
    tool_groups: list[str] | None = None
    skills: list[str] | None = None
    soul: str | None = None


@router.get("/agents")
async def list_agents() -> dict:
    """List all registered agents."""
    agents = AgentRegistry.list()
    return {"agents": [a.to_dict() for a in agents]}


@router.get("/agents/check")
async def check_agent_name(name: str) -> dict:
    """Check if an agent name is available.

    NOTE: This route MUST be registered before ``/agents/{name}``
    so that FastAPI matches the literal path first.
    """
    exists = AgentRegistry.get(name) is not None
    return {"available": not exists, "name": name}


@router.get("/agents/{name}")
async def get_agent(name: str) -> dict:
    """Get a single agent by name."""
    agent = AgentRegistry.get(name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return agent.to_dict()


@router.post("/agents", status_code=201)
async def create_agent(req: CreateAgentRequest) -> dict:
    """Create a new agent."""
    if AgentRegistry.get(req.name):
        raise HTTPException(status_code=409, detail=f"Agent '{req.name}' already exists")

    agent = AgentDefinition(
        name=req.name,
        description=req.description,
        model=req.model,
        tool_groups=req.tool_groups,
        skills=req.skills,
        soul=req.soul,
    )
    AgentRegistry.register(req.name, agent)
    return agent.to_dict()


@router.put("/agents/{name}")
async def update_agent(name: str, req: UpdateAgentRequest) -> dict:
    """Update an existing agent."""
    agent = AgentRegistry.get(name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    updated = AgentRegistry.update(
        name,
        name=req.name,
        description=req.description,
        model=req.model,
        tool_groups=req.tool_groups,
        skills=req.skills,
        soul=req.soul,
    )
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to update agent")
    return updated.to_dict()


@router.delete("/agents/{name}")
async def delete_agent(name: str) -> dict:
    """Delete an agent."""
    if name in ("lead_agent",):
        raise HTTPException(status_code=400, detail="Cannot delete built-in agent")

    deleted = AgentRegistry.delete(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return {"deleted": name, "status": "ok"}