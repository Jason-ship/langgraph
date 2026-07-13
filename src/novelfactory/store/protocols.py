"""Store layer Protocol interfaces for structural typing.

These Protocols define the common contracts that store implementations
should satisfy. Python Protocols are structural — classes do NOT need
to explicitly inherit from them; if they have the matching methods,
they automatically satisfy the Protocol.

Usage in type hints::

    def process(store: VectorStoreProtocol) -> None:
        store.store_embedding(...)

Or use ``isinstance(obj, ConnectionProtocol)`` for runtime checks
(decorated with ``@runtime_checkable``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConnectionProtocol(Protocol):
    """Connection lifecycle interface for all stores."""

    def is_connected(self) -> bool:
        """Whether the store has an active, usable connection."""
        ...

    def close(self) -> None:
        """Release connection resources (idempotent, safe to call multiple times)."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Interface for vector embedding storage and semantic search."""

    def store_embedding(
        self, project: str, chapter: int, embedding: list[float], summary: str
    ) -> None:
        """Store a chapter embedding with metadata."""
        ...

    def search_similar(
        self, query_embedding: list[float], top_k: int = 3, project: str = ""
    ) -> list[dict]:
        """Search for similar chapter embeddings."""
        ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Interface for knowledge graph storage (character/place/relationship)."""

    def upsert_character(self, name: str, properties: dict) -> None:
        """Create or update a character node."""
        ...

    def create_relationship(
        self,
        char1: str,
        rel_type: str,
        char2: str,
        properties: dict | None = None,
    ) -> None:
        """Create a relationship between two entities."""
        ...

    def get_character_network(self, char_name: str, max_depth: int = 2) -> list[dict]:
        """Get the relationship network around a character."""
        ...


@runtime_checkable
class KVStoreProtocol(Protocol):
    """Interface for async key-value stores (Redis-style)."""

    async def get(self, key: str) -> str | None:
        """Get a value by key."""
        ...

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Set a value with optional TTL."""
        ...

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        ...

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        ...
