"""MCP 客户端 — 连接配置构建和工具加载。

Migrated from DeerFlow mcp/client.py + mcp/tools.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_server_params(server_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """构建 MCP 服务器连接参数。

    Args:
        server_name: 服务器名称。
        config: 服务器配置字典，支持:
            - command: stdio 模式的命令
            - args: stdio 模式的参数列表
            - url: SSE/HTTP 模式的 URL
            - headers: SSE/HTTP 模式的自定义请求头
            - env: stdio 模式的环境变量

    Returns:
        连接参数字典。
    """
    transport = config.get("transport", "stdio")
    params: dict[str, Any] = {"transport": transport}

    if transport == "stdio":
        command = config.get("command", "")
        if not command:
            raise ValueError(f"MCP server {server_name} has no command configured")
        params["command"] = command
        params["args"] = config.get("args", [])
        if "env" in config:
            params["env"] = config["env"]
    elif transport in ("sse", "http"):
        url = config.get("url", "")
        if not url:
            raise ValueError(f"MCP server {server_name} has no url configured")
        params["url"] = url
        if "headers" in config:
            params["headers"] = config["headers"]
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")

    return params


def build_servers_config(extensions_config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """构建多服务器配置字典。

    Args:
        extensions_config: 扩展配置，包含 mcp_servers 列表。

    Returns:
        服务器名称 → 连接参数的映射。
    """
    if not extensions_config:
        return {}

    servers = extensions_config.get("mcp_servers", []) if isinstance(extensions_config, dict) else []
    result: dict[str, dict[str, Any]] = {}

    for server_config in servers:
        if not isinstance(server_config, dict):
            continue
        name = server_config.get("name", "")
        if not name or not server_config.get("enabled", True):
            continue
        try:
            params = build_server_params(name, server_config)
            result[name] = params
            logger.info("[MCP] Configured server %s (transport=%s)", name, params.get("transport"))
        except (ValueError, KeyError) as exc:
            logger.warning("[MCP] Skipping server %s: %s", name, exc)

    return result


__all__ = [
    "build_server_params",
    "build_servers_config",
]