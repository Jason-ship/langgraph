"""Agent Registry — dynamic agent registration and management."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Agent definition — matches the frontend Agent type in core/agents/types.ts"""

    name: str
    description: str = ""
    model: str = "deepseek-chat"
    tool_groups: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    soul: str = ""  # system prompt
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRegistry:
    """Dynamic agent registry — singleton, used by Agent API routes."""

    _agents: dict[str, AgentDefinition] = {}
    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, name: str, definition: AgentDefinition) -> None:
        with cls._lock:
            cls._agents[name] = definition
        logger.info("[AgentRegistry] Registered agent: %s", name)

    @classmethod
    def get(cls, name: str) -> AgentDefinition | None:
        return cls._agents.get(name)

    @classmethod
    def list(cls) -> list[AgentDefinition]:
        return list(cls._agents.values())

    @classmethod
    def delete(cls, name: str) -> bool:
        with cls._lock:
            if name in cls._agents:
                del cls._agents[name]
                logger.info("[AgentRegistry] Deleted agent: %s", name)
                return True
        return False

    @classmethod
    def update(cls, agent_name: str, **kwargs: Any) -> AgentDefinition | None:
        with cls._lock:
            agent = cls._agents.get(agent_name)
            if not agent:
                return None
            new_name = kwargs.pop("name", None)
            for key, value in kwargs.items():
                if hasattr(agent, key) and value is not None:
                    setattr(agent, key, value)
            if new_name:
                setattr(agent, "name", new_name)
            agent.updated_at = datetime.now().isoformat()
            # Sync dict key if name changed
            if new_name and new_name != agent_name and new_name not in cls._agents:
                del cls._agents[agent_name]
                cls._agents[new_name] = agent
        return agent

    @classmethod
    def init_defaults(cls) -> None:
        """Register default agents on startup."""
        with cls._lock:
            if cls._initialized:
                return
            cls._agents["lead_agent"] = AgentDefinition(
                name="lead_agent",
                description="NovelFactory 主智能体，协调所有创作任务",
                model="deepseek-chat",
                tool_groups=["writing", "review", "memory"],
                skills=["story_planning", "chapter_writing", "quality_review"],
                soul="你是小说创作助手，通过对话方式帮助用户完成小说创作。",
            )
            cls._agents["story_planner"] = AgentDefinition(
                name="story_planner",
                description="故事策划师，帮助构建世界观和角色设定",
                model="deepseek-chat",
                tool_groups=["writing", "neo4j"],
                skills=["world_building", "character_design"],
                soul="你是专业的故事策划师，擅长构建世界观和角色设定。",
            )
            cls._agents["chapter_writer"] = AgentDefinition(
                name="chapter_writer",
                description="章节写手，根据大纲和用户指导生成章节内容",
                model="deepseek-chat",
                tool_groups=["writing"],
                skills=["chapter_writing"],
                soul="你是专业的章节写手，擅长创作引人入胜的小说章节。",
            )
            cls._agents["review_editor"] = AgentDefinition(
                name="review_editor",
                description="评审编辑，对已写章节提供评审意见和修改建议",
                model="deepseek-chat",
                tool_groups=["review"],
                skills=["quality_review"],
                soul="你是专业的评审编辑，擅长发现章节问题并提供改进建议。",
            )
            cls._initialized = True
        logger.info("[AgentRegistry] Initialized %d default agents", len(cls._agents))