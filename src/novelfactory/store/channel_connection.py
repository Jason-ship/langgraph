"""Channel connection persistence for NovelFactory.

Migrated from DeerFlow persistence/channel_connections/.
Simplified: uses asyncpg-style raw SQL instead of SQLAlchemy.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Database table names
TABLE_CONNECTIONS = "channel_connections"
TABLE_OAUTH_STATES = "channel_oauth_states"


class ChannelConnectionRepository:
    """Persistence facade for channel connections, OAuth states, and conversations."""

    def __init__(self, pool: Any = None) -> None:
        """Initialize with a database connection pool.

        Args:
            pool: An asyncpg pool or similar connection pool with execute/fetch/fetchrow methods.
                  If None, only in-memory operations are supported.
        """
        self._pool = pool

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _normalize_optional_identity(value: str | None) -> str:
        return value or ""

    # -- connection CRUD ---------------------------------------------------

    async def upsert_connection(
        self,
        *,
        owner_user_id: str,
        provider: str,
        external_account_id: str | None = None,
        external_account_name: str | None = None,
        workspace_id: str | None = None,
        workspace_name: str | None = None,
        scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "connected",
    ) -> dict[str, Any]:
        """Create or update a channel connection."""
        if self._pool is None:
            return {"id": self._new_id(), "owner_user_id": owner_user_id, "provider": provider, "status": status}

        conn_id = self._new_id()
        ext_id = self._normalize_optional_identity(external_account_id)
        ws_id = self._normalize_optional_identity(workspace_id)
        scopes_json = json.dumps(scopes or [], ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = datetime.now(UTC)

        async with self._pool.acquire() as conn:
            # Check for existing connection
            row = await conn.fetchrow(
                f"""SELECT id FROM {TABLE_CONNECTIONS}
                WHERE owner_user_id = $1 AND provider = $2
                AND external_account_id = $3 AND workspace_id = $4""",
                owner_user_id, provider, ext_id, ws_id,
            )
            if row:
                await conn.execute(
                    f"""UPDATE {TABLE_CONNECTIONS} SET status = $1, external_account_name = $2,
                    workspace_name = $3, scopes = $4, metadata_json = $5, updated_at = $6
                    WHERE id = $7""",
                    status, external_account_name, workspace_name, scopes_json, metadata_json, now, row["id"],
                )
                conn_id = row["id"]
            else:
                await conn.execute(
                    f"""INSERT INTO {TABLE_CONNECTIONS}
                    (id, owner_user_id, provider, status, external_account_id, external_account_name,
                     workspace_id, workspace_name, scopes, metadata_json, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11)""",
                    conn_id, owner_user_id, provider, status, ext_id, external_account_name,
                    ws_id, workspace_name, scopes_json, metadata_json, now,
                )

        return {"id": conn_id, "owner_user_id": owner_user_id, "provider": provider, "status": status}

    async def find_connection_by_external_identity(
        self,
        *,
        provider: str,
        external_account_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Find a connection by external identity."""
        if self._pool is None:
            return None

        ext_id = self._normalize_optional_identity(external_account_id)
        ws_id = self._normalize_optional_identity(workspace_id)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT * FROM {TABLE_CONNECTIONS}
                WHERE provider = $1 AND external_account_id = $2
                AND workspace_id = $3 AND status = 'connected'
                ORDER BY updated_at DESC LIMIT 1""",
                provider, ext_id, ws_id,
            )
            if row is None:
                return None
            return dict(row)

    async def disconnect_connection(self, *, connection_id: str, owner_user_id: str) -> bool:
        """Disconnect (revoke) a connection."""
        if self._pool is None:
            return False
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {TABLE_CONNECTIONS} SET status = 'revoked' WHERE id = $1 AND owner_user_id = $2",
                connection_id, owner_user_id,
            )
            return result != "UPDATE 0"

    async def list_connections(self, owner_user_id: str) -> list[dict[str, Any]]:
        """List all connections for a user."""
        if self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {TABLE_CONNECTIONS} WHERE owner_user_id = $1 ORDER BY updated_at DESC",
                owner_user_id,
            )
            return [dict(row) for row in rows]

    # -- OAuth states ------------------------------------------------------

    async def create_oauth_state(
        self,
        *,
        owner_user_id: str,
        provider: str,
        state: str,
        expires_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create an OAuth state record."""
        if self._pool is None:
            return
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {TABLE_OAUTH_STATES}
                (id, provider, code, owner_user_id, expires_at, metadata_json, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                self._new_id(), provider, state, owner_user_id, expires_at, metadata_json, datetime.now(UTC),
            )

    async def consume_oauth_state(self, *, provider: str, state: str) -> dict[str, Any] | None:
        """Consume (atomically) an OAuth state code."""
        if self._pool is None:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""UPDATE {TABLE_OAUTH_STATES} SET consumed_at = $1
                WHERE provider = $2 AND code = $3 AND consumed_at IS NULL
                AND expires_at > $1
                RETURNING *""",
                datetime.now(UTC), provider, state,
            )
            if row is None:
                return None
            return {
                "owner_user_id": row["owner_user_id"],
                "provider": row["provider"],
                "metadata": json.loads(row.get("metadata_json", "{}")),
            }

    # -- thread mapping ----------------------------------------------------

    async def set_thread_id(
        self,
        *,
        connection_id: str,
        owner_user_id: str,
        provider: str,
        external_conversation_id: str,
        thread_id: str,
        external_topic_id: str | None = None,
    ) -> None:
        """Store thread mapping for a connection."""
        # Fall back to ChannelStore for now
        from novelfactory.channels.service import get_channel_service

        service = get_channel_service()
        if service and service.store:
            service.store.set_thread_id(
                provider,
                external_conversation_id,
                thread_id,
                topic_id=external_topic_id,
                user_id=owner_user_id,
            )

    async def get_thread_id(
        self,
        connection_id: str,
        external_conversation_id: str,
        external_topic_id: str | None = None,
    ) -> str | None:
        """Get thread mapping for a connection."""
        from novelfactory.channels.service import get_channel_service

        service = get_channel_service()
        if service and service.store:
            return service.store.get_thread_id(
                "feishu",
                external_conversation_id,
                topic_id=external_topic_id,
            )
        return None