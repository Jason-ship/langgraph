"""Gateway 认证系统 — JWT 令牌、密码哈希、用户模型、认证提供者。

Migrated from DeerFlow app/gateway/auth/.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import bcrypt
import jwt as pyjwt
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# ── 错误码 ────────────────────────────────────────────────────────────────


class AuthErrorCode(StrEnum):
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    EMAIL_ALREADY_EXISTS = "EMAIL_ALREADY_EXISTS"
    NOT_AUTHENTICATED = "NOT_AUTHENTICATED"


class TokenError(StrEnum):
    EXPIRED = "expired"
    INVALID_SIGNATURE = "invalid_signature"
    MALFORMED = "malformed"


class AuthErrorResponse(BaseModel):
    code: AuthErrorCode
    detail: str = ""


def token_error_to_code(err: TokenError) -> AuthErrorCode:
    mapping = {
        TokenError.EXPIRED: AuthErrorCode.TOKEN_EXPIRED,
        TokenError.INVALID_SIGNATURE: AuthErrorCode.TOKEN_INVALID,
        TokenError.MALFORMED: AuthErrorCode.TOKEN_INVALID,
    }
    return mapping.get(err, AuthErrorCode.TOKEN_INVALID)


# ── 用户模型 ──────────────────────────────────────────────────────────────


class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    password_hash: str | None = None
    system_role: str = "user"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    oauth_provider: str | None = None
    oauth_id: str | None = None
    token_version: int = 0
    needs_setup: bool = False


class UserResponse(BaseModel):
    id: str
    email: str
    system_role: str
    created_at: str
    oauth_provider: str | None = None
    needs_setup: bool = False


# ── JWT 令牌 ──────────────────────────────────────────────────────────────


class TokenPayload(BaseModel):
    sub: str
    exp: datetime
    iat: datetime
    ver: int = 0


_JWT_SECRET: str | None = None
_JWT_ALGORITHM = "HS256"


def _get_jwt_secret() -> str:
    global _JWT_SECRET
    if _JWT_SECRET is None:
        _JWT_SECRET = os.environ.get("AUTH_JWT_SECRET", "")
        if not _JWT_SECRET:
            _JWT_SECRET = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
            logger.warning("[auth] JWT_SECRET not set, generated temporary secret")
    return _JWT_SECRET


def create_access_token(user_id: str, *, expires_days: int = 7, token_version: int = 0) -> str:
    """创建 JWT access token。"""
    now = datetime.now(UTC)
    payload = TokenPayload(
        sub=user_id,
        exp=now + timedelta(days=expires_days),
        iat=now,
        ver=token_version,
    )
    return pyjwt.encode(payload.model_dump(), _get_jwt_secret(), algorithm=_JWT_ALGORITHM)


def decode_token(token: str) -> TokenPayload | TokenError:
    """解码 JWT token，返回 TokenPayload 或 TokenError。"""
    try:
        payload = pyjwt.decode(token, _get_jwt_secret(), algorithms=[_JWT_ALGORITHM])
        return TokenPayload(**payload)
    except pyjwt.ExpiredSignatureError:
        return TokenError.EXPIRED
    except pyjwt.InvalidSignatureError:
        return TokenError.INVALID_SIGNATURE
    except pyjwt.DecodeError:
        return TokenError.MALFORMED


# ── 密码哈希 ──────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """版本化密码哈希: SHA-256 预哈希 + bcrypt。"""
    prehash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(prehash.encode("utf-8"), salt)
    return f"$nfv2${hashed.decode('ascii')}"


def verify_password(password: str, password_hash: str) -> bool:
    """验证密码，自动检测版本。"""
    if password_hash.startswith("$nfv2$"):
        prehash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        stored = password_hash[len("$nfv2$"):]
        return bcrypt.checkpw(prehash.encode("utf-8"), stored.encode("ascii"))
    # 兼容旧版纯 bcrypt 哈希
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))


def needs_rehash(password_hash: str) -> bool:
    """检查是否需要升级哈希版本。"""
    return not password_hash.startswith("$nfv2$")


# ── 认证提供者 ────────────────────────────────────────────────────────────

# 简易内存用户存储
_users_store: dict[str, User] = {}


class LocalAuthProvider:
    """本地邮箱/密码认证提供者。"""

    async def authenticate(self, email: str, password: str) -> User | None:
        """验证邮箱和密码。"""
        user = self._find_user_by_email(email)
        if user is None or user.password_hash is None:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    async def create_user(self, email: str, password: str | None = None, role: str = "user") -> User:
        """创建新用户。"""
        user = User(email=email, system_role=role)
        if password:
            user.password_hash = hash_password(password)
        _users_store[user.id] = user
        return user

    def _find_user_by_email(self, email: str) -> User | None:
        for user in _users_store.values():
            if user.email == email:
                return user
        return None

    def get_user(self, user_id: str) -> User | None:
        return _users_store.get(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        return self._find_user_by_email(email)

    def count_users(self) -> int:
        return len(_users_store)


__all__ = [
    "AuthErrorCode",
    "TokenError",
    "AuthErrorResponse",
    "token_error_to_code",
    "User",
    "UserResponse",
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
    "needs_rehash",
    "LocalAuthProvider",
]