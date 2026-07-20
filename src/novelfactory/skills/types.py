"""Skills 系统 — 技能数据类型定义。

Migrated from DeerFlow skills/types.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class SkillCategory(StrEnum):
    """技能来源分类。"""

    PUBLIC = "public"
    CUSTOM = "custom"
    LEGACY = "legacy"


@dataclass(frozen=True)
class SecretRequirement:
    """技能声明的环境变量需求。"""

    name: str
    optional: bool = False


@dataclass(frozen=True)
class Skill:
    """技能元数据。"""

    name: str
    description: str
    license: str | None
    skill_dir: Path
    skill_file: Path
    relative_path: Path
    category: SkillCategory
    allowed_tools: tuple[str, ...] | None = None
    enabled: bool = False
    required_secrets: tuple[SecretRequirement, ...] = field(default_factory=tuple)
    secrets_autonomous: bool = True


__all__ = [
    "SkillCategory",
    "SecretRequirement",
    "Skill",
]