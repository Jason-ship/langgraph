"""SkillInjectionMiddleware — 根据题材和阶段自动注入 Skill 内容到 system prompt。"""

from __future__ import annotations

from novelfactory.middleware.base import Middleware
from novelfactory.skills.loader import SkillLoader


class SkillInjectionMiddleware(Middleware):
    """根据 state.current_phase 和 genre 注入对应 Skill。

    - setup phase: 注入写作指南类 Skill
    - writing phase: 注入题材写作指南 + 评分标准
    """

    def __init__(self, loader: SkillLoader | None = None):
        self._loader = loader or SkillLoader()
        self._loader.discover()

    def modify_system_prompt(self, prompt: str, state: dict, config: dict) -> str:
        phase = state.get("current_phase", "")
        genre = state.get("genre", "")

        if phase == "setup":
            # setup 阶段注入通用写作指南
            skills = self._loader.get_skills_by_genre(genre) if genre else []
            if not skills:
                skills = self._loader.get_all_skills()[:3]
            if skills:
                skill_block = "\n\n".join(
                    f"## {s.name}\n{s.body[:1000]}" for s in skills if s.body
                )
                prompt += f"\n\n## 写作指南参考\n{skill_block}"

        elif phase == "writing":
            # writing 阶段注入题材 Skill + 评分标准
            parts = []
            if genre:
                genre_skills = self._loader.get_skills_by_genre(genre)
                for s in genre_skills:
                    if s.body:
                        parts.append(f"## {s.name}\n{s.body[:2000]}")

            # 追加 review skill
            review_skills = self._loader.get_skills_by_genre("四维评分")
            for s in review_skills:
                if s.body:
                    parts.append(f"## 评分标准\n{s.body[:1500]}")

            if parts:
                prompt += "\n\n## 题材写作指南\n" + "\n\n".join(parts)

        return prompt
