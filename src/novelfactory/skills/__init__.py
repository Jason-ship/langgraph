"""Skills 系统包入口。

提供技能类型、解析、存储、发现和运行时管理功能。
"""

from novelfactory.skills.manager import Skill, SkillManager
from novelfactory.skills.parser import parse_skill_file, split_skill_markdown
from novelfactory.skills.storage import SkillStorage, get_or_new_skill_storage
from novelfactory.skills.types import SecretRequirement, Skill as SkillType, SkillCategory

__all__ = [
    "Skill",
    "SkillType",
    "SkillCategory",
    "SecretRequirement",
    "SkillStorage",
    "SkillManager",
    "get_or_new_skill_storage",
    "parse_skill_file",
    "split_skill_markdown",
]