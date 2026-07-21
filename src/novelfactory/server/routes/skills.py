"""Skills API — list, enable, disable, install skills."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from novelfactory.skills.manager import SkillManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["skills"])


class SkillToggleRequest(BaseModel):
    enabled: bool


class SkillInstallRequest(BaseModel):
    path: str


@router.get("/skills")
async def list_skills() -> dict:
    """List all available skills."""
    skills = SkillManager.list()
    return {"skills": [s.to_dict() for s in skills]}


@router.get("/skills/{name}")
async def get_skill(name: str) -> dict:
    """Get a single skill by name."""
    skill = SkillManager.get(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return skill.to_dict()


@router.put("/skills/{name}")
async def toggle_skill(name: str, req: SkillToggleRequest) -> dict:
    """Enable or disable a skill."""
    success = SkillManager.enable(name) if req.enabled else SkillManager.disable(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return {"name": name, "enabled": req.enabled}


@router.post("/skills/install", status_code=201)
async def install_skill(req: SkillInstallRequest) -> dict:
    """Install a skill from a file path."""
    skill = SkillManager.install(req.path)
    if not skill:
        raise HTTPException(status_code=400, detail=f"Failed to install skill from {req.path}")
    return {"status": "installed", "skill": skill.to_dict()}