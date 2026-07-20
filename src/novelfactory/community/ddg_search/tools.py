"""DuckDuckGo 搜索工具。

Migrated from DeerFlow community/ddg_search/tools.py.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def web_search(query: str, count: int = 5) -> str:
    """DuckDuckGo Search — 通过 DuckDuckGo 搜索网页。

    无需 API 密钥，适用于快速、轻量级的网页搜索。
    注意：DuckDuckGo 搜索可能被限流，建议使用 Brave/Serper 作为替代。

    Args:
        query: 搜索关键词。
        count: 返回结果数量（1-10）。

    Returns:
        JSON 字符串，包含 title, url, content 等字段。
    """
    try:
        # 使用 duckduckgo_search 库（需安装）
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps({
                "error": "duckduckgo_search library not installed. Install with: pip install duckduckgo_search",
                "query": query,
            }, ensure_ascii=False)

        # 在线程池中运行同步库
        import asyncio

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=count))

        results_data = await asyncio.to_thread(_search)

        results = []
        for item in results_data[:count]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "content": item.get("body", ""),
            })

        return json.dumps({"query": query, "results": results, "provider": "duckduckgo"}, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc), "query": query}, ensure_ascii=False)


__all__ = ["web_search"]