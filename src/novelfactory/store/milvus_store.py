"""Milvus vector storage for chapter semantic search (MilvusClient API)."""

from __future__ import annotations

import logging
import os

from novelfactory.config.constants import EMBEDDING_DIMS_DEFAULT

logger = logging.getLogger(__name__)

# ── Milvus Store Constants ─────────────────────────────────────────────────────

MILVUS_TIMEOUT = 10
MILVUS_NLIST = 128
EMBEDDING_TRUNC_LEN = 8000
SEARCH_TOP_K = 3
MILVUS_NPROBE = 10
RESULT_SUMMARY_LEN = 200
MILVUS_VARCHAR_LEN = 128
MILVUS_SUMMARY_LEN = 8192


class MilvusStore:
    """Milvus vector storage for chapter semantic search (MilvusClient API)."""

    COLLECTION_NAME = "novel_chapters"
    DEFAULT_DIM = EMBEDDING_DIMS_DEFAULT

    @staticmethod
    def _resolve_dim() -> int:
        try:
            # v6.1: 统一从 settings 读取
            from novelfactory.config.settings import settings as _st_milvus

            return int(
                os.environ.get(
                    "EMBEDDING_DIMS",
                    str(_st_milvus.EMBEDDING_DIMS or MilvusStore.DEFAULT_DIM),
                )
            )
        except (ValueError, TypeError):
            return MilvusStore.DEFAULT_DIM

    def __init__(self, config) -> None:
        from pymilvus import DataType, MilvusClient

        self._client: MilvusClient | None = None
        self._connected = False
        self._dim = self._resolve_dim()
        host = getattr(
            config, "MILVUS_HOST", getattr(config, "milvus_host", "localhost")
        )
        port = getattr(config, "MILVUS_PORT", getattr(config, "milvus_port", "19530"))
        try:
            uri = f"http://{host}:{port}"
            self._client = MilvusClient(
                uri=uri, timeout=MILVUS_TIMEOUT, db_name="default"
            )

            if self._client.has_collection(self.COLLECTION_NAME):
                existing_dim = self._get_existing_dim()
                if existing_dim and existing_dim != self._dim:
                    logger.warning(
                        "MilvusStore dimension mismatch: existing=%d, target=%d. Dropping and recreating.",
                        existing_dim,
                        self._dim,
                    )
                    self._client.drop_collection(self.COLLECTION_NAME)

            if not self._client.has_collection(self.COLLECTION_NAME):
                logger.info(
                    "MilvusStore creating collection '%s' with dim=%d",
                    self.COLLECTION_NAME,
                    self._dim,
                )
                schema = self._client.create_schema(
                    auto_id=True,
                    enable_dynamic_field=False,
                )
                schema.add_field(
                    field_name="id", datatype=DataType.INT64, is_primary=True
                )
                schema.add_field(field_name="chapter_number", datatype=DataType.INT64)
                schema.add_field(
                    field_name="project_name",
                    datatype=DataType.VARCHAR,
                    max_length=MILVUS_VARCHAR_LEN,
                )
                schema.add_field(
                    field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._dim
                )
                schema.add_field(
                    field_name="summary",
                    datatype=DataType.VARCHAR,
                    max_length=MILVUS_SUMMARY_LEN,
                )

                index_params = self._client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="IVF_FLAT",
                    metric_type="IP",
                    params={"nlist": MILVUS_NLIST},
                )
                self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    schema=schema,
                    index_params=index_params,
                )
            self._connected = True
        except Exception as e:
            logger.warning("MilvusStore init failed: %s", e)
            self._connected = False

    def _get_existing_dim(self) -> int | None:
        try:
            desc = self._client.describe_collection(self.COLLECTION_NAME)
            for field in desc.get("fields", []):
                if field.get("field_name") == "vector" or field.get("name") == "vector":
                    params = field.get("params", {}) or {}
                    dim = params.get("dim")
                    if dim is not None:
                        return int(dim)
                    type_params = field.get("type_params", {}) or {}
                    dim = type_params.get("dim")
                    if dim is not None:
                        return int(dim)
            return None
        except Exception:
            return None

    def is_connected(self) -> bool:
        return self._connected

    def store_embedding(
        self, project: str, chapter: int, embedding: list[float], summary: str
    ) -> None:
        """Store a chapter embedding with metadata.

        v6.1: 添加运行时调用计数日志。
        """
        logger.info("[Milvus] store_embedding project=%s chapter=%s", project, chapter)
        if not self._connected:
            return
        try:
            self._client.insert(
                collection_name=self.COLLECTION_NAME,
                data=[
                    {
                        "chapter_number": chapter,
                        "project_name": project,
                        "vector": embedding,
                        "summary": summary[:EMBEDDING_TRUNC_LEN],
                    }
                ],
            )
        except Exception as e:
            logger.warning("MilvusStore store error: %s", e)

    def search_similar(
        self, query_embedding: list[float], top_k: int = SEARCH_TOP_K, project: str = ""
    ) -> list[dict]:
        """Search for similar chapter embeddings.

        v6.1: 添加运行时调用计数日志。
        """
        logger.info("[Milvus] search_similar top_k=%s project=%s", top_k, project)
        if not self._connected:
            return []
        try:
            expr = f'project_name == "{project}"' if project else ""
            results = self._client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_embedding],
                limit=top_k,
                filter=expr,
                output_fields=["chapter_number", "summary"],
                search_params={
                    "metric_type": "IP",
                    "params": {"nprobe": MILVUS_NPROBE},
                },
            )
            hits = []
            for hits_group in results:
                for hit in hits_group:
                    entity = hit.get("entity", {})
                    hits.append(
                        {
                            "chapter": entity.get("chapter_number"),
                            "score": hit.get("distance", 0),
                            "summary": entity.get("summary", "")[:RESULT_SUMMARY_LEN],
                        }
                    )
            return hits
        except Exception as e:
            logger.warning("MilvusStore search error: %s", e)
            return []

    def close(self) -> None:
        if self._client:
            self._client.close()
