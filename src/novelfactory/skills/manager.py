"""Skill Manager — load, enable, disable Markdown skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Skill definition — loaded from Markdown files."""

    name: str
    description: str = ""
    category: str = "general"
    license: str = "MIT"
    enabled: bool = True
    editable: bool = True
    body: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["enabled"] = self.enabled
        return result

    @classmethod
    def parse_markdown(cls, filepath: str | Path) -> Skill:
        """Parse a Markdown skill file with YAML frontmatter."""
        path = Path(filepath)
        content = path.read_text(encoding="utf-8")
        name = path.stem

        # Simple frontmatter parsing
        description = ""
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                body = parts[2].strip()
                for line in frontmatter.strip().split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        if key.strip() == "description":
                            description = val.strip()

        return cls(
            name=name,
            description=description,
            body=body,
        )


class SkillManager:
    """Singleton skill manager."""

    _skills: dict[str, Skill] = {}
    _initialized: bool = False

    @classmethod
    def discover(cls, skills_dir: str = "skills") -> None:
        """Discover skills from a directory of Markdown files."""
        path = Path(skills_dir)
        if not path.exists():
            logger.info("[SkillManager] Skills directory not found: %s", skills_dir)
            return

        for filepath in path.glob("*.md"):
            try:
                skill = Skill.parse_markdown(filepath)
                cls._skills[skill.name] = skill
                logger.info("[SkillManager] Discovered skill: %s", skill.name)
            except Exception as e:
                logger.warning("[SkillManager] Failed to parse skill %s: %s", filepath, e)

        logger.info("[SkillManager] Discovered %d skills from %s", len(cls._skills), skills_dir)

    @classmethod
    def list(cls) -> list[Skill]:
        return list(cls._skills.values())

    @classmethod
    def get(cls, name: str) -> Skill | None:
        return cls._skills.get(name)

    @classmethod
    def enable(cls, name: str) -> bool:
        skill = cls._skills.get(name)
        if skill:
            skill.enabled = True
            return True
        return False

    @classmethod
    def disable(cls, name: str) -> bool:
        skill = cls._skills.get(name)
        if skill:
            skill.enabled = False
            return True
        return False

    @classmethod
    def install(cls, filepath: str) -> Skill | None:
        """Install a skill from a Markdown file path."""
        try:
            skill = Skill.parse_markdown(filepath)
            cls._skills[skill.name] = skill
            logger.info("[SkillManager] Installed skill: %s from %s", skill.name, filepath)
            return skill
        except Exception as e:
            logger.warning("[SkillManager] Failed to install skill from %s: %s", filepath, e)
            return None

    @classmethod
    def init_defaults(cls, skills_dir: str = "skills") -> None:
        """Initialize the skill manager with default skills."""
        if cls._initialized:
            return
        cls.discover(skills_dir)
        cls._initialized = True