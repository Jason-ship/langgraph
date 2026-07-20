"""Brave 搜索工具。

Migrated from DeerFlow community/brave/tools.py.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from novelfactory.community.url_safety import validate_public_http_url

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMAGE_SEARCH_URL = "https://api.search.brave.com/res/v1/images/search"


def _safe_public_url(url: str) -> str:
    """验证 URL 是否为安全的公共 URL。"""
    try:
        return validate_public_http_url(url)
    except ValueError:
        return ""


async def web_search(query: str, count: int = 5) -> str:
    """Brave Web Search — 通过 Brave Search API 搜索网页。

    Args:
        query: 搜索关键词。
        count: 返回结果数量（1-10）。

    Returns:
        JSON 字符串，包含 title, url, content 等字段。
    """
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "BRAVE_SEARCH_API_KEY not configured", "query": query}, ensure_ascii=False)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                BRAVE_SEARCH_URL,
                headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": api_key},
                params={"q": query, "count": min(count, 10), "safesearch": "moderate"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in (data.get("web", {}) or {}).get("results", [])[:count]:
            url = item.get("url", "")
            result = {
                "title": item.get("title", ""),
                "url": _safe_public_url(url),
                "content": item.get("description", ""),
            }
            results.append(result)

        return json.dumps({"query": query, "results": results, "provider": "brave"}, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc), "query": query}, ensure_ascii=False)


async def image_search(query: str, count: int = 5) -> str:
    """Brave Image Search — 通过 Brave Search API 搜索图片。

    Args:
        query: 搜索关键词。
        count: 返回结果数量（1-10）。

    Returns:
        JSON 字符串，包含 title, url, image_url 等字段。
    """
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "BRAVE_SEARCH_API_KEY not configured", "query": query}, ensure_ascii=False)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                BRAVE_IMAGE_SEARCH_URL,
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                params={"q": query, "count": min(count, 10)},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in (data.get("results", []) or [])[:count]:
            thumbnail_url = _safe_public_url(item.get("thumbnail_url", ""))
            url = _safe_public_url(item.get("url", ""))
            results.append({
                "title": item.get("title", ""),
                "url": url,
                "image_url": thumbnail_url,
                "content": item.get("description", ""),
            })

        return json.dumps({"query": query, "results": results, "provider": "brave"}, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc), "query": query}, ensure_ascii=False)


__all__ = ["web_search", "image_search"]