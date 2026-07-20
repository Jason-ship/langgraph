"""Gateway 授权系统 — AuthContext + 权限装饰器。

Migrated from DeerFlow app/gateway/authz.py + internal_auth.py + auth_disabled.py.
"""

from __future__ import annotations

import logging
import os
import secrets
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request

from novelfactory.server.auth import User

logger = logging.getLogger(__name__)

# ── 内部认证 ──────────────────────────────────────────────────────────────

INTERNAL_AUTH_HEADER = "X-NovelFactory-Internal-Token"
INTERNAL_OWNER_HEADER = "X-NovelFactory-Owner-User-Id"

_internal_auth_token: str = ""


def _get_internal_token() -> str:
    global _internal_auth_token
    if not _internal_auth_token:
        _internal_auth_token = os.environ.get("NOVELFACTORY_INTERNAL_TOKEN", "")
        if not _internal_auth_token:
            _internal_auth_token = secrets.token_hex(32)
    return _internal_auth_token


def create_internal_auth_headers(owner_user_id: str | None = None) -> dict[str, str]:
    """创建内部调用认证头。"""
    headers = {INTERNAL_AUTH_HEADER: _get_internal_token()}
    if owner_user_id:
        headers[INTERNAL_OWNER_HEADER] = owner_user_id
    return headers


def is_valid_internal_auth(request: Request) -> bool:
    """验证内部认证 token。"""
    token = request.headers.get(INTERNAL_AUTH_HEADER, "")
    return secrets.compare_digest(token, _get_internal_token())


# ── 免认证模式 ────────────────────────────────────────────────────────────

AUTH_DISABLED_ENV = "NOVELFACTORY_AUTH_DISABLED"


def is_auth_disabled() -> bool:
    """检查是否启用了免认证模式。"""
    env = os.environ.get("NOVELFACTORY_ENV", "")
    if env in ("prod", "production"):
        return False
    return os.environ.get(AUTH_DISABLED_ENV, "").strip() == "1"


# ── 授权上下文 ────────────────────────────────────────────────────────────


class AuthContext:
    """授权上下文，存储在 request.state.auth。"""

    def __init__(self, user: User | None = None, permissions: list[str] | None = None):
        self.user = user
        self.permissions = permissions or []

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None

    def require_user(self) -> User:
        if self.user is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return self.user


# ── 授权装饰器 ────────────────────────────────────────────────────────────


def require_auth(func):
    """强制认证装饰器。"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if request is None:
            for _, v in kwargs.items():
                if isinstance(v, Request):
                    request = v
                    break

        if request is not None:
            auth = getattr(request.state, "auth", None)
            if auth is None or not auth.is_authenticated:
                # 检查内部认证
                if not is_valid_internal_auth(request) and not is_auth_disabled():
                    raise HTTPException(status_code=401, detail="Not authenticated")
                if is_auth_disabled():
                    request.state.auth = AuthContext(user=User(id="dev", email="dev@local", system_role="admin"))

        return await func(*args, **kwargs)
    return wrapper


def require_permission(resource: str, action: str, owner_check: bool = False):
    """权限检查装饰器工厂。"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None:
                for _, v in kwargs.items():
                    if isinstance(v, Request):
                        request = v
                        break

            if request is not None:
                auth = getattr(request.state, "auth", None)
                if auth is None or not auth.is_authenticated:
                    if not is_valid_internal_auth(request) and not is_auth_disabled():
                        raise HTTPException(status_code=401, detail="Not authenticated")

            return await func(*args, **kwargs)
        return wrapper
    return decorator