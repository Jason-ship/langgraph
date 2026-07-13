"""Embedding service — multi-provider with graceful fallback."""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.config.constants import EMBEDDING_DIMS_DEFAULT

logger = logging.getLogger(__name__)

# ── Embedding Service Constants ───────────────────────────────────────────────

EMBED_INPUT_MAX = 8192
BASE_URL_DISPLAY_LEN = 40


class EmbeddingService:
    """Embedding service — multi-provider with graceful fallback.

    Priority:
      1. EMBEDDING_API_KEY + EMBEDDING_BASE_URL
      2. OPENAI_API_KEY
      3. ARK_API_KEY
      4. No key → zero vector (degraded)
    """

    def __init__(self) -> None:
        import os

        # v6.1: 统一从 settings 读取
        from novelfactory.config.settings import settings as _st_emb

        self._api_key = (
            _st_emb.EMBEDDING_API_KEY
            or os.getenv("EMBEDDING_API_KEY")
            or _st_emb.DEEPSEEK_API_KEY
            or os.getenv("OPENAI_API_KEY")
            or _st_emb.ARK_API_KEY
            or os.getenv("ARK_API_KEY", "")
        )
        self._base_url: str = (
            _st_emb.EMBEDDING_BASE_URL
            or os.getenv("EMBEDDING_BASE_URL")
            or _st_emb.DEEPSEEK_BASE_URL
            or os.getenv("OPENAI_BASE_URL")
            or _st_emb.ARK_BASE_URL
            or os.getenv("ARK_BASE_URL", "")
            or ""
        )
        self._model = _st_emb.EMBEDDING_MODEL or os.getenv(
            "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
        )
        self._dims = int(
            os.getenv(
                "EMBEDDING_DIMS",
                str(_st_emb.EMBEDDING_DIMS or EMBEDDING_DIMS_DEFAULT),
            )
        )
        self._client: Any = None
        logger.info(
            "EmbeddingService init: model=%s dims=%d base_url=%s key=%s",
            self._model,
            self._dims,
            self._base_url[:BASE_URL_DISPLAY_LEN] + "..."
            if len(self._base_url) > BASE_URL_DISPLAY_LEN
            else self._base_url,
            "***" if self._api_key else "(empty)",
        )

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            if self._base_url:
                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            else:
                self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        if not self._api_key:
            logger.warning("No API key available for embedding, returning zero vector")
            return [0.0] * self._dims
        try:
            client = self._get_client()
            resp = client.embeddings.create(
                model=self._model,
                input=text[:EMBED_INPUT_MAX],
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error("Embedding failed (model=%s): %s", self._model, e)
            return [0.0] * self._dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key or not texts:
            return [[0.0] * self._dims for _ in texts]
        try:
            client = self._get_client()
            resp = client.embeddings.create(
                model=self._model,
                input=[t[:EMBED_INPUT_MAX] for t in texts],
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            logger.error("Batch embedding failed (model=%s): %s", self._model, e)
            return [[0.0] * self._dims for _ in texts]

    def is_connected(self) -> bool:
        """Whether the embedding client is initialized."""
        return self._client is not None

    def close(self) -> None:
        """Release embedding client resources (clears cached client)."""
        self._client = None
