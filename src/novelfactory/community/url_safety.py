"""URL 安全验证 — SSRF 防护。

Migrated from DeerFlow community/url_safety.py.

跨所有 web 工具的共享安全层，防止 SSRF 攻击。
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_blocked_address(address: str) -> bool:
    """检查地址是否为私有/回环/链路本地/保留/多播地址。"""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def resolve_host_addresses(hostname: str) -> list[str]:
    """解析主机名到 IP 地址列表。"""
    try:
        info = socket.getaddrinfo(hostname, None)
        return list(set(addr[4][0] for addr in info))
    except socket.gaierror:
        logger.warning("[url_safety] Failed to resolve hostname: %s", hostname)
        return []


def validate_public_http_url(url: str, allow_private: bool = False, resolver: Callable | None = None) -> str:
    """验证 URL 是否为安全的公共 HTTP URL。

    Args:
        url: 要验证的 URL。
        allow_private: 是否允许私有地址。
        resolver: 自定义 DNS 解析函数。

    Returns:
        验证通过的 URL。

    Raises:
        ValueError: 如果 URL 不安全。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are allowed, got: {parsed.scheme}")

    hostname = parsed.hostname or ""
    if hostname in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(f"URL points to localhost: {url}")

    if allow_private:
        return url

    # DNS 解析检查
    resolve = resolver or resolve_host_addresses
    addresses = resolve(hostname)
    for addr in addresses:
        try:
            if is_blocked_address(addr):
                raise ValueError(f"URL resolves to a blocked address: {addr} ({url})")
        except ValueError as exc:
            # Re-raise ValueError from is_blocked_address check
            if "blocked" in str(exc):
                raise
            continue

    return url


__all__ = [
    "validate_public_http_url",
    "is_blocked_address",
    "resolve_host_addresses",
]