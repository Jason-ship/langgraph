"""Skills 系统 — 技能前导码解析。

Migrated from DeerFlow skills/frontmatter.py + skills/parser.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from novelfactory.skills.types import SecretRequirement, Skill, SkillCategory

logger = logging.getLogger(__name__)

ALLOWED_FRONTMATTER_PROPERTIES = frozenset({
    "name", "description", "license", "allowed-tools",
    "required-secrets", "secrets-autonomous",
})


@dataclass
class SkillMarkdownParts:
    """SKILL.md 解析结果。"""

    metadata: dict[str, Any]
    frontmatter_text: str
    body: str


def split_skill_markdown(content: str) -> SkillMarkdownParts:
    """将 SKILL.md 拆分为前导码和正文。"""
    content = content.strip()
    if not content.startswith("---"):
        return SkillMarkdownParts(metadata={}, frontmatter_text="", body=content)

    # 找到第二个 ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return SkillMarkdownParts(metadata={}, frontmatter_text="", body=content)

    frontmatter_text = content[3:end_idx].strip()
    body = content[end_idx + 3 :].strip()

    try:
        metadata = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        logger.warning("[skills] Failed to parse YAML frontmatter")
        metadata = {}

    if not isinstance(metadata, dict):
        metadata = {}

    return SkillMarkdownParts(metadata=metadata, frontmatter_text=frontmatter_text, body=body)


def parse_skill_file(skill_dir: Path, category: SkillCategory) -> Skill | None:
    """解析 SKILL.md 文件，返回 Skill 对象。"""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        logger.warning("[skills] No SKILL.md in %s", skill_dir)
        return None

    content = skill_file.read_text(encoding="utf-8")
    parts = split_skill_markdown(content)
    meta = parts.metadata

    name = meta.get("name", "")
    description = meta.get("description", "")

    if not name or not description:
        logger.warning("[skills] Missing name or description in %s", skill_file)
        return None

    # 解析 allowed-tools
    allowed_tools_raw = meta.get("allowed-tools")
    allowed_tools: tuple[str, ...] | None = None
    if isinstance(allowed_tools_raw, list):
        allowed_tools = tuple(str(t) for t in allowed_tools_raw if isinstance(t, str))
    elif isinstance(allowed_tools_raw, str):
        allowed_tools = (allowed_tools_raw,)

    # 解析 required-secrets
    required_secrets_raw = meta.get("required-secrets")
    required_secrets: list[SecretRequirement] = []
    if isinstance(required_secrets_raw, list):
        for item in required_secrets_raw:
            if isinstance(item, str):
                required_secrets.append(SecretRequirement(name=item))
            elif isinstance(item, dict):
                required_secrets.append(
                    SecretRequirement(
                        name=str(item.get("name", "")),
                        optional=bool(item.get("optional", False)),
                    )
                )

    secrets_autonomous = bool(meta.get("secrets-autonomous", True))

    return Skill(
        name=name,
        description=description,
        license=meta.get("license"),
        skill_dir=skill_dir,
        skill_file=skill_file,
        relative_path=skill_dir.relative_to(skill_dir.parent.parent) if skill_dir.parent.parent else Path(name),
        category=category,
        allowed_tools=allowed_tools,
        required_secrets=tuple(required_secrets),
        secrets_autonomous=secrets_autonomous,
    )


__all__ = [
    "SkillMarkdownParts",
    "split_skill_markdown",
    "parse_skill_file",
    "ALLOWED_FRONTMATTER_PROPERTIES",
]