"""Database configuration and connection management for NovelFactory.

Provides DatabaseManager — a connection pool wrapper built on psycopg 3.
Used by writing_guide_store, migration scripts, and crew subgraphs.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ── Default pool configuration ──────────────────────────────────────────────
_DEFAULT_MINCONN = 2
_DEFAULT_MAXCONN = 10


class DatabaseManager:
    """PostgreSQL connection pool manager.

    Uses psycopg.pool.ConnectionPool for concurrent access.
    All cursors must be used with context managers to return connections
    to the pool.
    """

    _instance: DatabaseManager | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        minconn: int = _DEFAULT_MINCONN,
        maxconn: int = _DEFAULT_MAXCONN,
        connect_timeout: int = 10,
        statement_timeout: int = 30000,
    ) -> None:
        from psycopg_pool import ConnectionPool

        from novelfactory.config.settings import settings as _st

        # v6.1: 统一从 settings 读取，参数优先 > settings > os.environ 兜底
        self._host = host or _st.DB_HOST or os.environ.get("DB_HOST", "localhost")
        self._port = int(port or _st.DB_PORT or os.environ.get("DB_PORT", "5432"))
        self._dbname = (
            dbname or _st.DB_NAME or os.environ.get("DB_NAME", "novelfactory")
        )
        self._user = user or _st.DB_USER or os.environ.get("DB_USER", "noveluser")
        self._password = (
            password or _st.DB_PASSWORD or os.environ.get("DB_PASSWORD", "")
        )
        self._connect_timeout = connect_timeout
        self._statement_timeout = statement_timeout
        self._minconn = minconn
        self._maxconn = maxconn

        try:
            self._pool: ConnectionPool = ConnectionPool(
                min_size=minconn,
                max_size=maxconn,
                name="novelfactory_db",
                kwargs={
                    "host": self._host,
                    "port": self._port,
                    "dbname": self._dbname,
                    "user": self._user,
                    "password": self._password,
                    "autocommit": False,
                    "connect_timeout": connect_timeout,
                    "options": f"-c statement_timeout={statement_timeout}",
                },
                open=True,
            )
            logger.info(
                "DatabaseManager pool created: %s:%s/%s (%d-%d conns)",
                self._host,
                self._port,
                self._dbname,
                minconn,
                maxconn,
            )
        except Exception as e:
            logger.error("DatabaseManager pool creation failed: %s", e)
            raise

    @classmethod
    def get_instance(cls) -> DatabaseManager:
        """Get or create the singleton DatabaseManager instance (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @contextmanager
    def cursor(self, commit: bool = True) -> Iterator[Any]:
        """Get a database cursor.

        Usage:
            with db.cursor() as cur:
                cur.execute("SELECT * FROM table")
                results = cur.fetchall()

        Args:
            commit: Whether to commit after the cursor is closed. Default True.
                    Previously set conn.autocommit = True, which conflicted with
                    get_connection() that always sets autocommit = False.
                    Now uses explicit conn.commit() to avoid inconsistencies.

        Yields:
            A psycopg cursor.
        """
        conn = self._pool.getconn()
        try:
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
                if commit:
                    conn.commit()
        finally:
            self._pool.putconn(conn)

    def execute(self, sql: str, params: tuple | None = None) -> None:
        """Execute a SQL statement (no return value)."""
        with self.cursor() as cur:
            cur.execute(sql, params)

    def fetchone(self, sql: str, params: tuple | None = None) -> tuple | None:
        """Execute SQL and fetch one row."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def fetchall(self, sql: str, params: tuple | None = None) -> list[tuple]:
        """Execute SQL and fetch all rows."""
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def get_connection(self) -> _PooledConnection:
        """Get a connection from the pool.

        Returns a _PooledConnection wrapper that provides the standard
        connection interface (cursor, commit, close).
        """
        conn = self._pool.getconn()
        conn.autocommit = False
        return _PooledConnection(conn, self._pool)

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.close()
            logger.info("DatabaseManager pool closed")
            DatabaseManager._instance = None

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class _PooledConnection:
    """Wrapper around a psycopg connection from the pool.

    Provides cursor() and commit() that work with the pool.
    close() returns the connection to the pool.

    Proxies ``.autocommit`` to the underlying connection so callers
    can use ``with DatabaseManager.get_instance().get_connection() as conn:
    conn.autocommit = True`` for bulk operations.

    Falls back to ``__getattr__`` for any attribute not explicitly proxied,
    ensuring forward compatibility with psycopg connection API changes.
    """

    def __init__(self, conn: Any, pool: Any) -> None:
        self._conn: Any = conn
        self._pool: Any = pool

    @property
    def autocommit(self) -> bool:
        return self._conn.autocommit

    @autocommit.setter
    def autocommit(self, value: bool) -> None:
        self._conn.autocommit = value

    def __getattr__(self, name: str) -> Any:
        """Delegate any unknown attribute/method to the underlying connection."""
        return getattr(self._conn, name)

    def cursor(self) -> Any:
        return self._conn.cursor()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._pool.putconn(self._conn)

    def __enter__(self) -> _PooledConnection:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self.close()
