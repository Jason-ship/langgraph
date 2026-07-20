"""Community 集成 — 搜索和网页抓取工具。

提供多个搜索和网页抓取提供者的统一接口：
- Brave Search
- DuckDuckGo Search
- SearXNG (自托管元搜索引擎)
- Jina AI Reader (网页抓取)
- Crawl4AI (网页抓取)
"""

from novelfactory.community.url_safety import validate_public_http_url

__all__ = ["validate_public_http_url"]