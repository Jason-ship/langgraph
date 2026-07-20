"""MCP 模块 — Model Context Protocol 集成。

提供 MCP 会话池、客户端配置构建和工具加载能力。
"""

from novelfactory.mcp.client import build_server_params, build_servers_config
from novelfactory.mcp.session_pool import McpSessionPool

__all__ = [
    "McpSessionPool",
    "build_server_params",
    "build_servers_config",
]