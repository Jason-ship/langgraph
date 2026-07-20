"""Skills 系统 — 技能存储与发现。

Migrated from DeerFlow skills/storage/ + skills/catalog.py.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

from novelfactory.skills.parser import parse_skill_file
from novelfactory.skills.types import Skill, SkillCategory

logger = logging.getLogger(__name__)

SKILL_MD_FILE = "SKILL.md"


class SkillStorage:
    """技能存储 — 本地文件系统实现。

    目录布局:
        <root>/public/<name>/SKILL.md   — 内置技能
        <root>/custom/<name>/SKILL.md   — 用户自定义技能
    """

    def __init__(self, root_path: str | Path | None = None) -> None:
        if root_path is None:
            root_path = Path(".novelfactory") / "skills"
        self._root = Path(root_path)

    def load_skills(self, enabled_only: bool = False) -> list[Skill]:
        """加载所有技能。"""
        skills: list[Skill] = []

        # 加载 public 技能
        public_dir = self._root / "public"
        if public_dir.exists():
            for skill_dir in sorted(public_dir.iterdir()):
                if skill_dir.is_dir():
                    skill = parse_skill_file(skill_dir, SkillCategory.PUBLIC)
                    if skill:
                        skill.enabled = True
                        skills.append(skill)

        # 加载 custom 技能
        custom_dir = self._root / "custom"
        if custom_dir.exists():
            for skill_dir in sorted(custom_dir.iterdir()):
                if skill_dir.is_dir():
                    skill = parse_skill_file(skill_dir, SkillCategory.CUSTOM)
                    if skill:
                        skills.append(skill)

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        return skills

    def get_skill(self, name: str) -> Skill | None:
        """按名称获取技能。"""
        for skill in self.load_skills():
            if skill.name == name:
                return skill
        return None

    def skill_exists(self, name: str, category: SkillCategory | None = None) -> bool:
        """检查技能是否存在。"""
        base = self._root / (category.value if category else "")
        return (base / name / SKILL_MD_FILE).exists()

    def get_skill_dir(self, name: str, category: SkillCategory) -> Path:
        """获取技能目录路径。"""
        return self._root / category.value / name


# 全局单例 + 工厂
_skill_storage: SkillStorage | None = None
_skill_storage_lock = threading.Lock()


def get_or_new_skill_storage() -> SkillStorage:
    """获取或创建全局技能存储单例。"""
    global _skill_storage
    if _skill_storage is None:
        with _skill_storage_lock:
            if _skill_storage is None:
                _skill_storage = SkillStorage()
    return _skill_storage


__all__ = [
    "SkillStorage",
    "get_or_new_skill_storage",
]