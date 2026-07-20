"""Model Factory — 模型创建工厂。

Migrated from DeerFlow models/factory.py.

提供统一的 LLM 模型创建接口，支持配置归一化。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def create_chat_model(
    model: str = "",
    temperature: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """创建 LLM 聊天模型。

    统一入口，支持配置归一化：
    - 自动处理 api_base → base_url 映射
    - 支持自定义模型名称

    Args:
        model: 模型名称。
        temperature: 温度参数。
        api_key: API 密钥。
        base_url: API 基础 URL。
        max_tokens: 最大输出 token 数。
        **kwargs: 其他 ChatOpenAI 参数。

    Returns:
        BaseChatModel 实例。
    """
    # 归一化 base_url
    if base_url and "api_base" not in kwargs:
        kwargs["base_url"] = base_url

    model_kwargs: dict[str, Any] = {
        "model": model or "deepseek-chat",
        "temperature": temperature,
        **kwargs,
    }
    if api_key:
        model_kwargs["api_key"] = api_key
    if max_tokens:
        model_kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**model_kwargs)


__all__ = ["create_chat_model"]