"""SearXNG — 自托管元搜索引擎工具。

Migrated from DeerFlow community/searxng/.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

SEARXNG_DEFAULT_URL = "http://localhost:8888"


async def web_search(query: str, count: int = 5) -> str:
    """SearXNG Search — 通过自托管 SearXNG 实例搜索网页。

    使用自托管的元搜索引擎，聚合多个搜索引擎结果。
    适用于：隐私敏感场景或需要自定义搜索源。

    Args:
        query: 搜索关键词。
        count: 返回结果数量（1-10）。

    Returns:
        JSON 字符串，包含 title, url, content 等字段。
    """
    base_url = os.environ.get("SEARXNG_URL", SEARXNG_DEFAULT_URL)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{base_url}/search",
                params={"q": query, "format": "json", "language": "zh-CN", "categories": "general"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in (data.get("results", []) or [])[:count]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        return json.dumps({"query": query, "results": results, "provider": "searxng"}, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc), "query": query}, ensure_ascii=False)


__all__ = ["web_search"]