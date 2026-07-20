"""Secret Context — 请求级安全秘钥注入。

Migrated from DeerFlow runtime/secret_context.py.

秘钥仅在需要时作为环境变量注入，不会出现在提示词或工具参数中。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 保留的上下文键
SECRETS_KEY = "secrets"
ACTIVE_SKILL_SECRETS_KEY = "__active_skill_secrets"


def redact_secret_context_keys(context: dict[str, Any]) -> dict[str, Any]:
    """在序列化时移除秘钥字段，防止泄露。"""
    return {k: v for k, v in context.items() if not k.startswith("__secret") and k != SECRETS_KEY}


def redact_config_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """递归移除配置中的 secret 字段。"""
    if not isinstance(config, dict):
        return config
    return {k: redact_config_secrets(v) if isinstance(v, dict) else v for k, v in config.items() if "secret" not in k.lower()}


__all__ = [
    "redact_secret_context_keys",
    "redact_config_secrets",
    "SECRETS_KEY",
]