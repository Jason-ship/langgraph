"""
NovelFactory 三库集成状态管理器
==================================
NovelStateTracker 编排层 — 协调 PGStore/MilvusStore/Neo4jStore/EmbeddingService。

使用方式（在 writing_crew 中嵌入）：
    tracker = NovelStateTracker(project_name, config)
    tracker.after_chapter(chapter_text, chapter_number, quality_score)
    state_prompt = tracker.before_chapter(chapter_number)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from novelfactory.store.embedding import EmbeddingService
from novelfactory.store.milvus_store import MilvusStore
from novelfactory.store.neo4j_store import Neo4jStore
from novelfactory.store.postgres_store import DBConfig, PGStore

logger = logging.getLogger(__name__)

# ── 模块级单例：避免每次创建 NovelStateTracker 时重复初始化连接池 ──
_pg_store: PGStore | None = None
_milvus_store: MilvusStore | None = None
_neo4j_store: Neo4jStore | None = None
_embedding: EmbeddingService | None = None


# ── Tracker-Specific Constants ────────────────────────────────────────────────

# After-Chapter Pipeline
CHAPTER_SUMMARY_LEN = 500
MAX_THREADS_SAVE = 5
THREAD_DESC_LEN = 200
THREAD_NAME_LEN = 100
EMBED_SUMMARY_LEN = 1000

# Before-Chapter Pipeline
THREAD_DISPLAY_LEN = 150
MAX_THREADS_DISPLAY = 5
MAX_CHARS_DISPLAY = 15

# State Extraction
EXTRACT_INPUT_MAX = 6000

# Setup Integration
MIN_CHAR_NAME_LEN = 2
SETUP_DOC_MAX = 3000
SETUP_SUMMARY_LEN = 200


# ── Main NovelStateTracker ─────────────────────────────────────────────────


class NovelStateTracker:
    """Main orchestrator — coordinates PG, Milvus, Neo4j for cross-chapter consistency."""

    def __init__(self, project_name: str, config: DBConfig | None = None) -> None:
        global _pg_store, _milvus_store, _neo4j_store, _embedding

        self.project = project_name
        self.config = config or DBConfig()

        if _pg_store is None:
            _pg_store = PGStore(self.config)
        self.pg = _pg_store

        if _milvus_store is None:
            _milvus_store = MilvusStore(self.config)
        self.milvus = _milvus_store

        if _neo4j_store is None:
            _neo4j_store = Neo4jStore(self.config)
        self.neo4j = _neo4j_store

        if _embedding is None:
            _embedding = EmbeddingService()
        self.embedding = _embedding

        self._llm = None
        self._scale: Any = None

    @property
    def scale(self) -> Any:
        if self._scale is None:
            try:
                from novelfactory.pipeline.scale_manager import ScaleManager

                self._scale = ScaleManager(self.pg, self.project)
            except Exception as e:
                logger.warning("ScaleManager init failed: %s", e)
                self._scale = None
        return self._scale

    def _get_llm(self) -> object:
        if self._llm is None:
            from novelfactory.config.llm import get_reviewer_llm

            self._llm = get_reviewer_llm()
        return self._llm

    # ── After Chapter ─────────────────────────────────────────────────────

    def after_chapter(
        self,
        chapter_text: str,
        chapter_number: int,
        quality_score: float = 0.0,
        chapter_title: str = "",
    ) -> dict:
        result = {"pg": False, "milvus": False, "neo4j": False, "characters": 0}

        state_data = self._extract_state(chapter_text, chapter_number)
        characters = state_data.get("characters", {})
        threads = state_data.get("unresolved_threads", [])

        try:
            for char_name, char_state in characters.items():
                self.pg.save_character_state(
                    self.project, chapter_number, char_name, char_state
                )
            result["characters"] = len(characters)
            result["pg"] = True
        except Exception as e:
            logger.warning("PG save error: %s", e)

        try:
            summary = chapter_text[:CHAPTER_SUMMARY_LEN].replace("\n", " ")
            self.pg.save_chapter(
                self.project,
                chapter_number,
                chapter_title,
                len(chapter_text),
                quality_score,
                summary,
            )
        except Exception as e:
            logger.warning("PG chapter save error: %s", e)

        for thread in threads[:MAX_THREADS_SAVE]:
            try:
                desc = (
                    thread[:THREAD_DESC_LEN] if isinstance(thread, str) else str(thread)
                )
                self.pg.save_plot_thread(
                    self.project,
                    desc[:THREAD_NAME_LEN],
                    desc[:THREAD_DESC_LEN],
                    chapter_number,
                )
                self.neo4j.upsert_plot_thread(
                    desc[:THREAD_NAME_LEN], desc[:THREAD_DESC_LEN], chapter_number
                )
            except Exception:
                pass

        try:
            summary = chapter_text[:EMBED_SUMMARY_LEN].replace("\n", " ")
            vec = self.embedding.embed(summary)
            if self.milvus.is_connected():
                self.milvus.store_embedding(self.project, chapter_number, vec, summary)
                result["milvus"] = True
        except Exception as e:
            logger.warning("Milvus error: %s", e)

        try:
            if self.neo4j.is_connected():
                for char_name, char_state in characters.items():
                    self.neo4j.upsert_character(
                        char_name,
                        {
                            "location": char_state.get("location", ""),
                            "mood": char_state.get("mood", ""),
                            "power": char_state.get("power_level", ""),
                            "status": char_state.get("status", "健在"),
                            "last_chapter": chapter_number,
                        },
                    )
                    if char_state.get("location"):
                        self.neo4j.create_location_relationship(
                            char_name, char_state["location"]
                        )

                for name, char_state in characters.items():
                    relationships = char_state.get("relationships", {})
                    for target, rel_type in relationships.items():
                        self.neo4j.create_relationship(
                            name, rel_type, target, {"chapter": chapter_number}
                        )
                    if char_state.get("location"):
                        self.neo4j.create_location_relationship(
                            name, char_state["location"]
                        )

                char_names = list(characters.keys())
                if len(char_names) > 1:
                    pairs = [
                        {"c1": char_names[i], "c2": char_names[j]}
                        for i in range(len(char_names))
                        for j in range(i + 1, len(char_names))
                    ]
                    if pairs:
                        self.neo4j.create_relationships_batch(
                            "KNOWS", pairs, {"chapter": chapter_number}
                        )
                result["neo4j"] = True
        except Exception as e:
            logger.warning("Neo4j error: %s", e)

        try:
            if self.scale:
                project = self.pg.get_project(self.project)
                ws = project.get("world_setting", "") if project else ""
                cs = project.get("character_setting", "") if project else ""
                scale_result = self.scale.after_chapter(
                    chapter_number,
                    chapter_text,
                    world_setting=ws,
                    character_setting=cs,
                    quality_score=quality_score,
                )
                result["scale"] = scale_result
        except Exception as e:
            logger.warning("Scale error: %s", e)

        return result

    # ── Before Chapter ────────────────────────────────────────────────────

    def before_chapter(self, chapter_number: int) -> str:
        parts = []

        try:
            states = self.pg.get_latest_character_states(self.project)
            if states:
                char_lines = ["【所有角色当前状态】"]
                for name, info in sorted(
                    states.items(),
                    key=lambda x: (
                        0 if x[1].get("status", "") not in ("未出场", "") else 1,
                        x[0],
                    ),
                ):
                    loc = info.get("location", "未知")
                    mood = info.get("mood", "")
                    power = info.get("power_level", "")
                    status = info.get("status", "健在")
                    line = f"  - {name}"
                    if loc and loc != "未知":
                        line += f" | {loc}"
                    if status not in ("健在", "未出场"):
                        line += f" | [{status}]"
                    if power and power != "未知":
                        line += f" | {power}"
                    if mood:
                        line += f" | 心境：{mood}"
                    char_lines.append(line)
                parts.append("\n".join(char_lines))
        except Exception as e:
            logger.warning("PG read error: %s", e)

        try:
            threads = self.pg.get_open_threads(self.project)
            if threads:
                parts.append(
                    "【待处理伏笔/线索】\n"
                    + "\n".join(
                        f"  - {t['description'][:THREAD_DISPLAY_LEN]}"
                        for t in threads[:MAX_THREADS_DISPLAY]
                    )
                )
        except Exception:
            pass

        try:
            if self.milvus.is_connected():
                query_vec = self.embedding.embed(f"第{chapter_number}章 上下文检索")
                import novelfactory.store.milvus_store as ms

                similar = (
                    self.milvus.search_similar(
                        query_vec, top_k=ms.SEARCH_TOP_K, project=self.project
                    )
                    if any(query_vec)
                    else []
                )
                if similar:
                    lines = ["【相关章节参考】"]
                    for s in similar:
                        lines.append(f"  - 第{s['chapter']}章 (相似度{s['score']:.2f})")
                    parts.append("\n".join(lines))
        except Exception:
            pass

        try:
            if self.neo4j.is_connected():
                chars = self.neo4j.get_all_characters()
                if chars:
                    parts.append(
                        "【故事中已有角色】\n" + "、".join(chars[:MAX_CHARS_DISPLAY])
                    )
        except Exception:
            pass

        try:
            if self.scale:
                scale_ctx = self.scale.build_writer_context(chapter_number)
                if scale_ctx:
                    parts.append(scale_ctx)
        except Exception:
            pass

        if not parts:
            return ""
        return "\n\n".join(parts)

    # ── Private: LLM State Extraction ─────────────────────────────────────

    def _extract_state(self, chapter_text: str, chapter_number: int) -> dict:
        llm: Any = self._get_llm()
        prompt = f"""你是一位小说编辑。请分析以下章节，提取角色状态和剧情线索。

【章节编号】第{chapter_number}章

【章节正文】
{chapter_text[:EXTRACT_INPUT_MAX]}

请输出JSON（仅JSON，不要其他内容）：
{{
  "characters": {{
    "角色名": {{
      "location": "当前位置",
      "mood": "心境（2-4字）",
      "power_level": "修为（未知填未知）",
      "status": "健在/受伤/失踪/死亡/未出场",
      "knowledge": ["知道的事实"],
      "items": ["持有的道具"]
    }}
  }},
  "unresolved_threads": ["未解决线索"],
  "current_location": "本章主要地点",
  "time_since_start": "时间跨度"
}}"""
        try:
            resp = llm.invoke([("user", prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {
            "characters": {},
            "unresolved_threads": [],
            "current_location": "",
            "time_since_start": "",
        }

    # ── Setup Integration ────────────────────────────────────────────────

    def save_setup(
        self,
        world_setting: str,
        character_setting: str,
        story_outline: str,
        chapter_outlines: str,
        genre: str,
        chapter_count: int,
    ) -> None:
        self.pg.save_project(
            self.project,
            genre,
            chapter_count,
            world_setting,
            character_setting,
            story_outline,
            chapter_outlines,
        )

        try:
            if self.neo4j.is_connected():
                llm: Any = self._get_llm()
                prompt = f"从以下角色设定中提取所有角色名，只输出JSON数组：\n{character_setting[:SETUP_DOC_MAX]}"
                resp = llm.invoke([("user", prompt)])
                text = resp.content if hasattr(resp, "content") else str(resp)
                match = re.search(r"\[[\s\S]*\]", text)
                if match:
                    names = json.loads(match.group())
                    for name in names:
                        if isinstance(name, str) and len(name) >= MIN_CHAR_NAME_LEN:
                            self.neo4j.upsert_character(
                                name,
                                {
                                    "status": "未出场",
                                    "location": "未知",
                                    "chapter": 0,
                                    "created_in_setup": True,
                                },
                            )
        except Exception:
            pass

        try:
            if self.milvus.is_connected():
                for doc_name, doc_text in [
                    ("世界观设定", world_setting),
                    ("角色设定", character_setting),
                    ("故事大纲", story_outline),
                ]:
                    vec = self.embedding.embed(doc_text[:SETUP_DOC_MAX])
                    self.milvus.store_embedding(
                        self.project,
                        0,
                        vec,
                        f"[{doc_name}] {doc_text[:SETUP_SUMMARY_LEN]}",
                    )
        except Exception:
            pass

    def close(self) -> None:
        """Close all connections (PG, Milvus, Neo4j).

        NovelStateTracker 使用模块级单例，close() 仅为向后兼容占位，
        实际连接由全局生命周期管理，不在此处关闭。
        """
        pass

    def __del__(self) -> None:
        pass
