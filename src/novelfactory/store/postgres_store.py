"""PostgreSQL + pgvector storage for character states and chapter metadata.

Uses psycopg 3 connection pool. All cursors managed via with-statement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ── PG Store Constants ────────────────────────────────────────────────────────

SUMMARY_TRUNC_LEN = 2000  # PG chapter summary truncation


class DBConfig:
    """Database connection configuration.

    v6.1: 统一从 settings 读取，os.environ 兜底。
    """

    def __init__(self) -> None:
        from novelfactory.config.settings import settings as _st

        self.pg_host = _st.DB_HOST or os.environ.get("DB_HOST", "localhost")
        self.pg_port = str(_st.DB_PORT or os.environ.get("DB_PORT", "5432"))
        self.pg_db = _st.DB_NAME or os.environ.get("DB_NAME", "novelfactory")
        self.pg_user = _st.DB_USER or os.environ.get("DB_USER", "noveluser")
        self.pg_password = _st.DB_PASSWORD or os.environ.get("DB_PASSWORD", "")
        self.pg_connect_timeout = int(
            os.environ.get("DB_CONNECT_TIMEOUT", str(_st.DB_CONNECT_TIMEOUT))
        )
        self.pg_statement_timeout = int(
            os.environ.get("DB_STATEMENT_TIMEOUT", str(_st.DB_STATEMENT_TIMEOUT))
        )
        self.pg_pool_min = int(
            os.environ.get("DB_POOL_MIN_SIZE", str(_st.DB_POOL_MIN_SIZE))
        )
        self.pg_pool_max = int(
            os.environ.get("DB_POOL_MAX_SIZE", str(_st.DB_POOL_MAX_SIZE))
        )

        self.milvus_host = _st.MILVUS_HOST or os.environ.get("MILVUS_HOST", "localhost")
        self.milvus_port = str(
            _st.MILVUS_PORT or os.environ.get("MILVUS_PORT", "19530")
        )

        self.neo4j_host = _st.NEO4J_HOST or os.environ.get("NEO4J_HOST", "localhost")
        self.neo4j_port = str(_st.NEO4J_PORT or os.environ.get("NEO4J_PORT", "7687"))
        self.neo4j_user = _st.NEO4J_USER or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = _st.NEO4J_PASSWORD or os.environ.get("NEO4J_PASSWORD", "")


class PGStore:
    """PostgreSQL + pgvector storage for character states and chapter metadata.

    If *db_manager* is provided (a ``DatabaseManager`` instance), it is used
    for all connection acquisition instead of creating a separate
    ``ConnectionPool``.  This avoids the "double pool" conflict when
    ``DatabaseManager`` is already managing connections elsewhere.
    """

    def __init__(self, config: DBConfig, db_manager: Any = None) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._db_manager = db_manager

        if db_manager is not None:
            logger.info("PGStore using shared DatabaseManager")
            return

        from psycopg_pool import ConnectionPool

        try:
            self._pool = ConnectionPool(
                min_size=config.pg_pool_min,
                max_size=config.pg_pool_max,
                kwargs={
                    "host": config.pg_host,
                    "port": config.pg_port,
                    "dbname": config.pg_db,
                    "user": config.pg_user,
                    "password": config.pg_password,
                    "autocommit": True,
                    "connect_timeout": config.pg_connect_timeout,
                    "options": f"-c statement_timeout={config.pg_statement_timeout}",
                },
                open=True,
            )
            logger.info(
                "PGStore pool created: %d-%d connections to %s:%s/%s",
                config.pg_pool_min,
                config.pg_pool_max,
                config.pg_host,
                config.pg_port,
                config.pg_db,
            )
        except Exception as e:
            logger.error("PGStore pool creation failed: %s", e)
            raise

    @contextmanager
    def _get_cursor(self) -> Iterator[Any]:
        if self._db_manager is not None:
            conn = self._db_manager.get_connection()
            try:
                conn.autocommit = True
                cur = conn.cursor()
                try:
                    yield cur
                finally:
                    cur.close()
            finally:
                conn.close()
            return

        conn = self._pool.getconn()
        try:
            conn.autocommit = True
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
        finally:
            self._pool.putconn(conn)

    def cursor(self) -> Any:
        if self._db_manager is not None:
            conn = self._db_manager.get_connection()
            conn.autocommit = True
            real_cur = conn.cursor()

            class _CursorWrapper:
                def __init__(self, cursor_obj: Any, connection_obj: Any) -> None:
                    self._cursor = cursor_obj
                    self._conn = connection_obj

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._cursor, name)

                def close(self) -> None:
                    try:
                        self._cursor.close()
                    finally:
                        self._conn.close()

                def __iter__(self) -> Any:
                    return iter(self._cursor)

            return _CursorWrapper(real_cur, conn)

        conn = self._pool.getconn()
        conn.autocommit = True
        real_cur = conn.cursor()

        class _CursorWrapper:
            def __init__(
                self, cursor_obj: Any, connection_obj: Any, pool_obj: Any
            ) -> None:
                self._cursor = cursor_obj
                self._conn = connection_obj
                self._pool = pool_obj

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cursor, name)

            def close(self) -> None:
                try:
                    self._cursor.close()
                finally:
                    self._pool.putconn(self._conn)

            def __iter__(self) -> Any:
                return iter(self._cursor)

        return _CursorWrapper(real_cur, conn, self._pool)

    def save_character_state(
        self, project: str, chapter: int, character: str, state: dict
    ) -> None:
        with self._get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO novel_character_states
                    (project_name, chapter_number, character_name, location, mood,
                     power_level, status, relationships, knowledge, items, raw_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    project,
                    chapter,
                    character,
                    state.get("location", ""),
                    state.get("mood", ""),
                    state.get("power_level", ""),
                    state.get("status", "健在"),
                    json.dumps(state.get("relationships", {}), ensure_ascii=False),
                    json.dumps(state.get("knowledge", []), ensure_ascii=False),
                    json.dumps(state.get("items", []), ensure_ascii=False),
                    json.dumps(state, ensure_ascii=False),
                ),
            )

    def get_latest_character_states(self, project: str) -> dict[str, dict]:
        with self._get_cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (character_name) character_name, raw_state
                FROM novel_character_states
                WHERE project_name = %s
                ORDER BY character_name, chapter_number DESC
            """,
                (project,),
            )
            result = {}
            for name, raw in cur.fetchall():
                result[name] = raw if isinstance(raw, dict) else json.loads(raw)
            return result

    def get_chapter_states(self, project: str, chapter: int) -> list[dict]:
        with self._get_cursor() as cur:
            cur.execute(
                """
                SELECT character_name, raw_state
                FROM novel_character_states
                WHERE project_name = %s AND chapter_number = %s
            """,
                (project, chapter),
            )
            return [row[1] for row in cur.fetchall()]

    def save_chapter(
        self,
        project: str,
        chapter: int,
        title: str,
        word_count: int,
        quality_score: float,
        summary: str,
    ) -> None:
        with self._get_cursor() as cur:
            text_hash = hashlib.sha256(summary.encode()).hexdigest()[:16]
            cur.execute(
                """
                INSERT INTO novel_chapters
                    (project_name, chapter_number, title, word_count, quality_score, summary, full_text_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_name, chapter_number) DO UPDATE SET
                    title = EXCLUDED.title, word_count = EXCLUDED.word_count,
                    quality_score = EXCLUDED.quality_score, summary = EXCLUDED.summary,
                    full_text_hash = EXCLUDED.full_text_hash
            """,
                (
                    project,
                    chapter,
                    title,
                    word_count,
                    quality_score,
                    summary[:2000],
                    text_hash,
                ),
            )

    def save_project(
        self,
        project: str,
        genre: str,
        chapter_count: int,
        world_setting: str,
        character_setting: str,
        story_outline: str,
        chapter_outlines: str,
    ) -> None:
        with self._get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO novel_projects
                    (project_name, genre, chapter_count, world_setting, character_setting,
                     story_outline, chapter_outlines)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_name) DO UPDATE SET
                    genre = EXCLUDED.genre, chapter_count = EXCLUDED.chapter_count,
                    world_setting = EXCLUDED.world_setting, character_setting = EXCLUDED.character_setting,
                    story_outline = EXCLUDED.story_outline, chapter_outlines = EXCLUDED.chapter_outlines,
                    updated_at = NOW()
            """,
                (
                    project,
                    genre,
                    chapter_count,
                    world_setting,
                    character_setting,
                    story_outline,
                    chapter_outlines,
                ),
            )

    def get_project(self, project: str) -> dict | None:
        with self._get_cursor() as cur:
            cur.execute(
                "SELECT * FROM novel_projects WHERE project_name = %s", (project,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "project_name": row[0],
                    "genre": row[1],
                    "chapter_count": row[2],
                    "world_setting": row[3],
                    "character_setting": row[4],
                    "story_outline": row[5],
                    "chapter_outlines": row[6],
                }
            return None

    def save_plot_thread(
        self,
        project: str,
        thread_name: str,
        description: str,
        chapter: int,
        status: str = "open",
        related_chars: list = None,
    ) -> None:
        with self._get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO novel_plot_threads
                    (project_name, thread_name, status, created_chapter, description, related_characters)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_name, thread_name) DO UPDATE SET
                    status = EXCLUDED.status,
                    description = EXCLUDED.description,
                    related_characters = EXCLUDED.related_characters
            """,
                (
                    project,
                    thread_name,
                    status,
                    chapter,
                    description,
                    json.dumps(related_chars or [], ensure_ascii=False),
                ),
            )

    def get_open_threads(self, project: str) -> list[dict]:
        with self._get_cursor() as cur:
            cur.execute(
                """
                SELECT thread_name, description, created_chapter, related_characters
                FROM novel_plot_threads
                WHERE project_name = %s AND status = 'open'
                ORDER BY created_chapter
            """,
                (project,),
            )
            rows = cur.fetchall()
            return [
                {
                    "name": r[0],
                    "description": r[1],
                    "chapter": r[2],
                    "related_chars": r[3],
                }
                for r in rows
            ]

    def is_connected(self) -> bool:
        """Whether the PG connection pool is open and usable."""
        if self._db_manager is not None:
            return True
        return (
            hasattr(self, "_pool") and self._pool is not None and not self._pool.closed
        )

    def close(self) -> None:
        if self._db_manager is not None:
            # 不关闭共享的 DatabaseManager 池
            logger.debug("PGStore skipping close for shared DatabaseManager")
        elif hasattr(self, "_pool") and self._pool:
            self._pool.close()
            logger.info("PGStore pool closed")
