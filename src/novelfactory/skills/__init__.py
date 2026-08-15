"""Skills 系统包入口。

提供技能类型、解析、存储、发现和运行时管理功能。

模型命名约定（v8.4 收敛，避免同名歧义）:
    - ``skills.loader.Skill``       题材注入模型（genre/triggers/body）— 供 SkillLoader 使用
    - ``skills.types.Skill``        文件系统元数据模型（skill_dir/category/allowed_tools）
    - ``skills.manager.Skill``      API 管理模型（editable/body/enabled）

三者职责不同、各自独立工作，包入口不导出名为 ``Skill`` 的符号，
统一用全限定名引用，杜绝隐式混淆。
"""

from novelfactory.skills.loader import SkillLoader
from novelfactory.skills.manager import SkillManager
from novelfactory.skills.parser import parse_skill_file, split_skill_markdown
from novelfactory.skills.storage import SkillStorage, get_or_new_skill_storage
from novelfactory.skills.types import SecretRequirement, SkillCategory

__all__ = [
    "SkillCategory",
    "SecretRequirement",
    "SkillStorage",
    "SkillManager",
    "SkillLoader",
    "get_or_new_skill_storage",
    "parse_skill_file",
    "split_skill_markdown",
]
