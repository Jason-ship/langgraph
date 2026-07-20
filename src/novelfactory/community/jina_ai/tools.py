"""Jina AI — 网页内容抓取工具。

Migrated from DeerFlow community/jina_ai/.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

JINA_BASE_URL = "https://r.jina.ai"


async def web_fetch(url: str) -> str:
    """Jina AI Reader — 抓取网页内容并返回 Markdown 格式。

    使用 Jina AI 的 Reader API 将网页内容转换为干净的 Markdown 文本。
    适用于：从文档、博客、文章等提取可读内容。

    Args:
        url: 要抓取的网页 URL。

    Returns:
        Markdown 格式的网页内容（截断到 4096 字符）。
    """
    api_key = os.environ.get("JINA_API_KEY", "")
    headers = {"Accept": "text/markdown"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(f"{JINA_BASE_URL}/{url}", headers=headers)
            resp.raise_for_status()
            content = resp.text

        # 截断到 4096 字符
        if len(content) > 4096:
            content = content[:4096] + "\n\n...(truncated)"

        return json.dumps({"url": url, "content": content, "provider": "jina"}, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc), "url": url}, ensure_ascii=False)


__all__ = ["web_fetch"]