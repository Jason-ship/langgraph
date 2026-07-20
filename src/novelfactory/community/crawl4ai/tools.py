"""Crawl4AI — 网页抓取工具。

Migrated from DeerFlow community/crawl4ai/.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

CRAWL4AI_DEFAULT_URL = "http://localhost:11235"


async def web_fetch(url: str) -> str:
    """Crawl4AI — 抓取网页内容并返回 Markdown 格式。

    使用自托管的 Crawl4AI 服务进行网页抓取。
    支持 JavaScript 渲染、智能内容提取。

    Args:
        url: 要抓取的网页 URL。

    Returns:
        Markdown 格式的网页内容（截断到 4096 字符）。
    """
    base_url = os.environ.get("CRAWL4AI_URL", CRAWL4AI_DEFAULT_URL)
    api_token = os.environ.get("CRAWL4AI_API_TOKEN", "")

    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/crawl",
                headers=headers,
                json={"url": url, "priority": 10, "max_pages": 1, "extract_type": "markdown"},
            )
            resp.raise_for_status()
            data = resp.json()

        content = ""
        if isinstance(data, dict):
            content = data.get("result", {}).get("markdown", "") or data.get("markdown", "")

        if len(content) > 4096:
            content = content[:4096] + "\n\n...(truncated)"

        return json.dumps({"url": url, "content": content, "provider": "crawl4ai"}, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc), "url": url}, ensure_ascii=False)


__all__ = ["web_fetch"]