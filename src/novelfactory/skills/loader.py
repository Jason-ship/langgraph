"""SkillLoader — 发现、解析、查询 SKILL.md 文件。

使用方式:
    loader = SkillLoader()
    skills = loader.discover()
    writing_skills = loader.get_skills_by_genre("仙侠")
    summary = loader.build_skills_summary()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Skill:
    """单个 Skill 的数据结构。

    Attributes:
        name: Skill 唯一标识（如 "xian-xia"）
        description: 简短描述
        genre: 题材分类（如 "仙侠"、"爽文"）
        version: 版本号
        triggers: 触发关键词列表
        body: YAML frontmatter 之后的 Markdown 正文
        filepath: SKILL.md 的绝对路径
    """

    name: str
    description: str
    genre: str = ""
    version: str = "1.0.0"
    triggers: list[str] = field(default_factory=list)
    body: str = ""
    filepath: str = ""


class SkillLoader:
    """Skill 加载器 — 遍历 skills/ 目录发现、解析和查询 SKILL.md。

    discover() 会缓存所有解析结果，后续 get_* 方法直接从缓存读取。
    """

    def __init__(self, base_path: str | Path = "skills"):
        self.base_path = Path(base_path)
        if not self.base_path.is_absolute():
            # 尝试相对项目根目录解析
            self.base_path = Path.cwd() / base_path
        self._cache: dict[str, Skill] = {}

    # ── 发现 ────────────────────────────────────────────────────────────────────

    def discover(self) -> list[Skill]:
        """遍历 base_path 下所有 SKILL.md，解析并缓存。

        Returns:
            所有已发现的 Skill 列表。
        """
        skills: list[Skill] = []
        if not self.base_path.exists():
            return skills

        for root, _dirs, files in os.walk(self.base_path):
            for fn in files:
                if fn.lower() == "skill.md":
                    path = os.path.join(root, fn)
                    skill = self.parse_skill(path)
                    if skill:
                        self._cache[skill.name] = skill
                        skills.append(skill)
        return skills

    # ── 解析 ────────────────────────────────────────────────────────────────────

    def parse_skill(self, path: str | Path) -> Skill | None:
        """解析单个 SKILL.md 文件。

        格式:
            ---
            name: xian-xia
            description: ...
            ---
            # Markdown 正文

        Args:
            path: SKILL.md 的路径。

        Returns:
            解析成功返回 Skill，失败返回 None。
        """
        path = Path(path)
        if not path.exists():
            return None

        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        # ── 解析 YAML frontmatter ────────────────────────────────────────────
        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        yaml_block = parts[1].strip()
        body = parts[2].strip()

        try:
            meta = yaml.safe_load(yaml_block)
        except yaml.YAMLError:
            return None

        if not isinstance(meta, dict):
            return None

        # ── 补齐缺失字段 ─────────────────────────────────────────────────────
        # 允许部分文件只有 name 和 description
        return Skill(
            name=str(meta.get("name", path.parent.name)),
            description=str(meta.get("description", "")),
            genre=str(meta.get("genre", "")),
            version=str(meta.get("version", "1.0.0")),
            triggers=meta.get("triggers", []),
            body=body,
            filepath=str(path.resolve()),
        )

    # ── 查询 ────────────────────────────────────────────────────────────────────

    def get_skill(self, name: str) -> Skill | None:
        """按 name 精确查询 Skill。

        Args:
            name: Skill 的 name 字段。

        Returns:
            匹配的 Skill，未找到返回 None。
        """
        return self._cache.get(name)

    def get_all_skills(self) -> list[Skill]:
        """获取所有已缓存的 Skill 列表。

        Returns:
            当前缓存中的所有 Skill（按 name 排序）。
        """
        return sorted(self._cache.values(), key=lambda s: s.name)

    def get_skills_by_genre(self, genre: str) -> list[Skill]:
        """按题材过滤 Skill。

        Args:
            genre: 题材名称（如 "仙侠"、"爽文"）。

        Returns:
            匹配题材的 Skill 列表。匹配规则：
              1. skill.genre 精确匹配
              2. skill.name 包含 genre 关键词
              3. skill.triggers 包含 genre
        """
        genre_lower = genre.lower()
        results: list[Skill] = []
        for skill in self._cache.values():
            if skill.genre == genre:
                results.append(skill)
            elif genre_lower in skill.name.lower():
                results.append(skill)
            elif any(genre_lower in t.lower() for t in skill.triggers):
                results.append(skill)
        return results

    def build_skills_summary(self, genre: str | None = None) -> str:
        """生成渐进式披露的摘要列表。

        Args:
            genre: 可选，只输出指定题材的摘要。

        Returns:
            格式化的摘要文本，每行 "name: description"。
        """
        if genre:
            skills = self.get_skills_by_genre(genre)
        else:
            skills = list(self._cache.values())

        if not skills:
            return "（暂无可用 Skill）"

        lines = []
        for s in skills:
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        """缓存中的 Skill 数量。"""
        return len(self._cache)

    def clear_cache(self):
        """清除缓存，下次 discover() 会重新加载。"""
        self._cache.clear()
