"""LangGraph persistence layer — production-grade checkpointer & store.

Aligns with official ``langgraph-checkpoint-postgres`` v3.1.0:
  - Uses ``AsyncPostgresSaver`` from ``langgraph.checkpoint.postgres.aio``
  - ``setup()`` handles all DDL via the official MIGRATIONS list
  - No runtime SQL patches, no manual DDL management
  - ``SELECT_SQL`` uses official ``bytea`` casts — v3.1.0 is fully
    compatible with psycopg 3 ``binary=True`` query mode

Checkpointer: AsyncPostgresSaver (psycopg 3 async pool)
Store:       AsyncPostgresStore (psycopg 3 async pool, with optional semantic search)

Fallback chain:
  1. Postgres (production)
  2. SQLite (lightweight persistent)
  3. InMemory (development / testing)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

from novelfactory.config.constants import EMBEDDING_DIMS_DEFAULT
from novelfactory.config.settings import settings

logger = logging.getLogger(__name__)

# ── Default configuration constants ────────────────────────────────────────────
_RETRY_ATTEMPTS = 3
_RETRY_SLEEP_SECONDS = 1
_DB_MIN_SIZE = 2
_DB_MAX_SIZE = 10
_DB_STORE_MIN_SIZE = 1
_DB_STORE_MAX_SIZE = 5
_AGC_RETAIN_LAST = 10
_GC_MAX_AGE_DAYS = 30
_AGC_LIST_LIMIT = 1000

# ── Custom serde with explicit msgpack allowlist ───────────────────────────
# Register custom project types to suppress the deserialization warning:
#   "Deserializing unregistered type ... from checkpoint."
# When allowed_msgpack_modules is set explicitly, the serde operates in
# strict-explicit mode: only SAFE_MSGPACK_TYPES (built-in safe types) plus
# the listed modules are allowed; all others are blocked.
_MSGPACK_ALLOWLIST: list[tuple[str, ...]] = [
    ("novelfactory.evaluation.schemas", "VerdictLevel"),
]


def _create_serde() -> JsonPlusSerializer:
    """Create a JsonPlusSerializer with the project's custom allowlist.

    By passing an explicit ``allowed_msgpack_modules``, we tell LangGraph
    that VerdictLevel is a trusted type.  All SAFE_MSGPACK_TYPES (langchain
    messages, langgraph types, datetime, etc.) remain automatically allowed.
    Any other non-safe type will be blocked at deserialization time (instead
    of just warned).
    """
    return JsonPlusSerializer(allowed_msgpack_modules=_MSGPACK_ALLOWLIST)


# ── Node-level RetryPolicy defaults ────────────────────────────────────────────
# v5.7 P0-fix: 已移至 config/constants.py，消除 agents/infra ↔ graph 循环导入。
# 此处保留兼容性重导出，graph/ 内部模块仍可通过此模块引用。
# 新代码应直接从 config.constants 导入。


# ── Connection helpers ─────────────────────────────────────────────────────────


def _db_url_from_env() -> str | None:
    """Build Postgres DSN — delegated to settings.database_url.

    v6.1: 统一从 settings 读取，支持 DATABASE_URL 和 DB_* 组件两种格式。
    """
    url = settings.database_url
    return url if url else None


def _checkpoint_type() -> str:
    return settings.CHECKPOINT_TYPE.lower()


def _store_type() -> str:
    return settings.STORAGE_TYPE.lower()


# ── Checkpointer factory ───────────────────────────────────────────────────────


# ── Singleton cache (v6.0.1: prevent pool leak on repeated create_checkpointer calls) ──

_checkpointer_singleton: Any = None
_store_singleton: Any = None
_checkpointer_lock: asyncio.Lock = asyncio.Lock()
_store_lock: asyncio.Lock = asyncio.Lock()


async def create_checkpointer(db_url: str | None = None) -> AsyncPostgresSaver | Any:
    """Create or reuse the production checkpointer (singleton with cache).

    v6.0.1: Added singleton cache — repeated calls reuse the same pool
    instead of creating new AsyncConnectionPool instances that leak.
    Call ``reset_checkpointer_singleton()`` to force recreation (e.g. after DB restart).

    Returns an AsyncPostgresSaver connected to the Postgres pool, or falls
    back to SQLite / InMemory if Postgres is unavailable.
    """
    global _checkpointer_singleton

    # v6.0.1: Reuse cached instance if available
    if _checkpointer_singleton is not None:
        return _checkpointer_singleton

    async with _checkpointer_lock:
        # Double-check after acquiring lock
        if _checkpointer_singleton is not None:
            return _checkpointer_singleton

        cp_type = _checkpoint_type()

        if cp_type == "postgres":
            db_url = db_url or _db_url_from_env()
            if db_url:
                for attempt in range(_RETRY_ATTEMPTS):
                    try:
                        pool = AsyncConnectionPool(
                            conninfo=db_url,
                            max_size=_DB_MAX_SIZE,
                            min_size=_DB_MIN_SIZE,
                            name="checkpointer",
                            kwargs={"autocommit": True},
                            open=False,
                        )
                        await pool.open()
                        saver = AsyncPostgresSaver(conn=pool, serde=_create_serde())
                        await saver.setup()
                        logger.info(
                            "[checkpointer] AsyncPostgresSaver ready (pool size=%d..%d)",
                            _DB_MIN_SIZE,
                            _DB_MAX_SIZE,
                        )
                        _checkpointer_singleton = saver
                        return saver
                    except Exception as exc:
                        msg = str(exc)
                        if "already exists" in msg and attempt < _RETRY_ATTEMPTS - 1:
                            logger.info(
                                "[checkpointer] DDL race (attempt %d/%d), retrying in %ds...",
                                attempt + 1,
                                _RETRY_ATTEMPTS,
                                _RETRY_SLEEP_SECONDS,
                            )
                            await asyncio.sleep(_RETRY_SLEEP_SECONDS)
                            continue
                        logger.warning(
                            "[checkpointer] Postgres unavailable (%s), falling back",
                            exc,
                        )
                        break

        if cp_type == "redis":
            try:
                from urllib.parse import quote, urlparse

                from langgraph.checkpoint.redis.aio import AsyncRedisSaver

                # v6.1: 统一从 settings 读取 Redis 配置
                redis_url = settings.REDIS_URL or settings.redis_url
                password = settings.REDIS_PASSWORD
                if password and redis_url:
                    encoded_password = quote(password)
                    # urlparse 安全解析：提取 host/port/db，插入密码
                    parsed = urlparse(redis_url)
                    if parsed.hostname:
                        db_path = (
                            f"/{parsed.path.lstrip('/')}"
                            if parsed.path and parsed.path != "/"
                            else "/0"
                        )
                        redis_url = (
                            f"{parsed.scheme}://:{encoded_password}"
                            f"@{parsed.hostname}:{parsed.port or 6379}{db_path}"
                        )
                saver = AsyncRedisSaver.from_conn_info(
                    url=redis_url,
                    serde=_create_serde(),
                )
                await saver.setup()
                logger.info("[checkpointer] AsyncRedisSaver ready (url=%s)", redis_url)
                _checkpointer_singleton = saver
                return saver
            except Exception as exc:
                logger.warning("[checkpointer] Redis unavailable (%s), falling back", exc)

        if cp_type in ("sqlite",):
            try:
                import aiosqlite
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                sqlite_path = os.environ.get(
                    "SQLITE_PATH", settings.STORAGE_PATH + "/checkpoints.sqlite"
                )
                conn = await aiosqlite.connect(sqlite_path)
                saver = AsyncSqliteSaver(conn, serde=_create_serde())
                await saver.setup()
                logger.info("[checkpointer] AsyncSqliteSaver ready")
                _checkpointer_singleton = saver
                return saver
            except Exception as exc:
                logger.warning("[checkpointer] SQLite unavailable (%s), falling back", exc)

        # Final fallback: InMemory (development only)
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver(serde=_create_serde())
        logger.warning(
            "[checkpointer] No persistent backend — using InMemorySaver (DATA LOST ON RESTART)"
        )
        _checkpointer_singleton = saver
        return saver


def reset_checkpointer_singleton() -> None:
    """Reset the singleton cache so the next call recreates the pool."""
    global _checkpointer_singleton
    _checkpointer_singleton = None


def reset_store_singleton() -> None:
    """Reset the store singleton cache so the next call recreates the pool."""
    global _store_singleton
    _store_singleton = None


# ── Store embedding helper ─────────────────────────────────────────────────────


def _create_embed_function() -> Callable[[list[str]], list[list[float]]] | None:
    """Create an embedding function for PostgresStore semantic search.

    Uses EMBEDDING_* env vars to configure the embedding provider.
    Falls back through: EMBEDDING_API_KEY → OPENAI_API_KEY → ARK_API_KEY.

    Returns a callable ``fn(texts: list[str]) -> list[list[float]]``, or None
    if no embedding API key is configured.
    """
    # v6.1: 统一从 settings 读取 embedding 配置
    api_key = (
        settings.EMBEDDING_API_KEY
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ARK_API_KEY", "")
    )
    base_url = (
        settings.EMBEDDING_BASE_URL
        or os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("ARK_BASE_URL", "")
    )
    model = settings.EMBEDDING_MODEL or os.getenv(
        "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
    )

    if not api_key:
        logger.info(
            "[store] No embedding API key configured — semantic search disabled"
        )
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        def embed(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in response.data]

        logger.info(
            "[store] Embedding provider ready: model=%s base_url=%s", model, base_url
        )
        return embed
    except Exception as exc:
        logger.warning(
            "[store] Embedding init failed (%s) — semantic search disabled", exc
        )
        return None


# ── Store factory ──────────────────────────────────────────────────────────────


async def create_store(db_url: str | None = None) -> AsyncPostgresStore | Any:
    """Create or reuse the cross-thread long-term memory store (singleton with cache).

    v6.0.1: Added singleton cache — repeated calls reuse the same pool
    instead of creating new AsyncConnectionPool instances that leak.
    Call ``reset_store_singleton()`` to force recreation (e.g. after DB restart).

    Returns an AsyncPostgresStore with optional semantic search (powered by
    EMBEDDING_* env vars), or falls back to InMemoryStore only when Postgres
    itself is unreachable.
    """
    global _store_singleton

    if _store_singleton is not None:
        return _store_singleton

    async with _store_lock:
        # Double-check after acquiring lock
        if _store_singleton is not None:
            return _store_singleton

        st_type = _store_type()

        if st_type == "postgres":
            db_url = db_url or _db_url_from_env()
            if db_url:
                try:
                    pool = AsyncConnectionPool(
                        conninfo=db_url,
                        max_size=_DB_STORE_MAX_SIZE,
                        min_size=_DB_STORE_MIN_SIZE,
                        name="store",
                        kwargs={"autocommit": True},
                        open=False,
                    )
                    await pool.open()
                    embed_fn = _create_embed_function()
                    if embed_fn:
                        dims = int(
                            os.getenv(
                                "EMBEDDING_DIMS",
                                str(settings.EMBEDDING_DIMS or EMBEDDING_DIMS_DEFAULT),
                            )
                        )
                        store = AsyncPostgresStore(
                            pool,
                            index={
                                "embed": embed_fn,
                                "dims": dims,
                                "fields": ["$"],
                            },
                        )
                        await store.setup()
                        logger.info(
                            "[store] AsyncPostgresStore ready (semantic search: dims=%d)",
                            dims,
                        )
                    else:
                        store = AsyncPostgresStore(pool)
                        await store.setup()
                        logger.info("[store] AsyncPostgresStore ready (no semantic search)")
                    _store_singleton = store
                    return store
                except Exception as exc:
                    logger.warning(
                        "[store] Postgres store unavailable (%s), falling back", exc
                    )

        # Fallback: InMemoryStore
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        logger.warning(
            "[store] No persistent store — using InMemoryStore (DATA LOST ON RESTART)"
        )
        _store_singleton = store
        return store


# ── Global checkpointer accessor (for GC & runtime use) ────────────────────────

_checkpointer_instance = None


def set_checkpointer_instance(cp: Any) -> None:
    """Store a reference to the active checkpointer for runtime GC use."""
    global _checkpointer_instance
    _checkpointer_instance = cp


def get_checkpointer_instance() -> Any:
    """Retrieve the stored checkpointer instance."""
    return _checkpointer_instance


# ── Checkpoint GC Utilities ────────────────────────────────────────────────────
# 官方 AsyncPostgresSaver API 不支持逐条删除检查点,
# 只支持 adelete_thread() 删除整个线程。
# cleanup_thread_full() — 调用 adelete_thread() 删除线程所有检查点。
#                         适用于 completed/completed_sync 等终态线程。


async def cleanup_thread_full(checkpointer: Any, thread_id: str) -> int:
    """删除指定线程的全部检查点 (使用官方 adelete_thread API)。

    仅在以下场景调用:
      - 小说创作完成后 (phase == "done")
      - 用户主动删除项目
      - 废弃的测试/调试线程

    注意: 此操作不可逆, 调用前应确认线程数据已完成持久化到业务数据库。

    Returns:
        删除的检查点数量 (成功时返回 1 表示线程已删除, 失败时返回 0)
    """
    if checkpointer is None:
        return 0
    try:
        # 先查询检查点数量用于日志
        config = {"configurable": {"thread_id": thread_id}}
        count = 0
        try:
            async for _ in checkpointer.alist(config, limit=1):
                count += 1
        except (NotImplementedError, AttributeError):
            pass

        # 执行线程级删除
        await checkpointer.adelete_thread(thread_id)
        logger.info(
            "[gc] Cleaned up thread %s (%d+ checkpoints deleted via adelete_thread).",
            thread_id,
            count,
        )
        return 1
    except AttributeError:
        # adelete_thread 不存在 (旧版本 checkpointer)
        logger.debug(
            "[gc] adelete_thread not available for %s (upgrade langgraph-checkpoint-postgres to >=2.0).",
            thread_id,
        )
        return 0
    except Exception as exc:
        logger.warning("[gc] Failed to cleanup thread %s: %s", thread_id, exc)
        return 0


# ── Terminal-phase checkpoint cleanup (v5.6: M1-CLEAN) ─────────────────────────


async def maybe_cleanup_checkpoints(
    state: dict,
    config: dict | None = None,
    checkpointer: object = None,
) -> int:
    """Clean up checkpoints when a project reaches terminal phase.

    Only acts when ``current_phase`` is ``"done"``.  Returns the number of
    threads cleaned (0 or 1).

    This is designed to be called from a terminal node (e.g.
    ``save_longterm_memory``).  The actual deletion runs inside the caller's
    asyncio task — the function itself is safe to call synchronously or
    asynchronously.
    """
    if state.get("current_phase") != "done":
        return 0

    if checkpointer is None:
        checkpointer = get_checkpointer_instance()
    if checkpointer is None:
        return 0

    thread_id = ""
    if config:
        thread_id = config.get("configurable", {}).get("thread_id", "")
    if not thread_id:
        thread_id = state.get("thread_id", "")
    if not thread_id:
        return 0

    return await cleanup_thread_full(checkpointer, thread_id)
