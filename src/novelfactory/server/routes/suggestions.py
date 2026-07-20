"""Suggestions API — 后续问题建议生成。

Migrated from DeerFlow app/gateway/routers/suggestions.py.

基于对话历史生成相关的 follow-up 问题建议。
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["suggestions"])


class SuggestionMessage(BaseModel):
    role: str = Field(..., description="消息角色: user|assistant")
    content: str = Field(..., description="消息内容")


class SuggestionsRequest(BaseModel):
    messages: list[SuggestionMessage] = Field(..., description="最近对话消息")
    n: int = Field(default=3, ge=1, le=5, description="建议数量")


class SuggestionsResponse(BaseModel):
    suggestions: list[str] = Field(default_factory=list, description="建议问题列表")


class SuggestionsConfigResponse(BaseModel):
    enabled: bool = Field(..., description="是否启用建议功能")


# 默认建议模板
_DEFAULT_SUGGESTIONS = {
    "writing": [
        "继续创作下一章",
        "修改当前章节的情节",
        "为角色添加更多背景故事",
    ],
    "review": [
        "评审当前章节质量",
        "检查情节连贯性",
        "优化角色对话",
    ],
}


def _generate_suggestions(messages: list[SuggestionMessage], n: int) -> list[str]:
    """基于对话历史生成建议。

    当前实现：根据最后一条消息的内容关键词生成建议。
    TODO: 集成 LLM 生成更智能的建议。
    """
    if not messages:
        return _DEFAULT_SUGGESTIONS["writing"][:n]

    last_msg = messages[-1].content.lower()
    if any(kw in last_msg for kw in ["评审", "审查", "review", "quality"]):
        pool = _DEFAULT_SUGGESTIONS["review"]
    else:
        pool = _DEFAULT_SUGGESTIONS["writing"]

    return pool[:n]


@router.post("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(body: SuggestionsRequest):
    """生成后续问题建议。

    基于对话历史，生成相关的 follow-up 问题建议。
    """
    suggestions = _generate_suggestions(body.messages, body.n)
    return SuggestionsResponse(suggestions=suggestions)


@router.get("/suggestions/config", response_model=SuggestionsConfigResponse)
async def suggestions_config():
    """获取建议功能配置。"""
    return SuggestionsConfigResponse(enabled=True)