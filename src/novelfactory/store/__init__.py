"""Data persistence layer — PG, Milvus, Neo4j coordination."""

from novelfactory.store.chapter_state_store import (
    load_tracker_from_store,
    save_tracker_to_store,
)
from novelfactory.store.embedding import EmbeddingService
from novelfactory.store.guide_store import (
    WritingGuide,
    WritingGuideResult,
    WritingGuideStore,
    get_guide_store,
)
from novelfactory.store.milvus_store import MilvusStore
from novelfactory.store.neo4j_store import Neo4jStore
from novelfactory.store.postgres_store import DBConfig, PGStore
from novelfactory.store.protocols import (
    ConnectionProtocol,
    GraphStoreProtocol,
    KVStoreProtocol,
    VectorStoreProtocol,
)
from novelfactory.store.redis_store import RedisStore, get_redis_store
from novelfactory.store.tracker import NovelStateTracker

__all__ = [
    "DBConfig",
    "PGStore",
    "MilvusStore",
    "Neo4jStore",
    "EmbeddingService",
    "NovelStateTracker",
    "RedisStore",
    "get_redis_store",
    "WritingGuideStore",
    "WritingGuide",
    "WritingGuideResult",
    "get_guide_store",
    "save_tracker_to_store",
    "load_tracker_from_store",
    "ConnectionProtocol",
    "GraphStoreProtocol",
    "KVStoreProtocol",
    "VectorStoreProtocol",
]
