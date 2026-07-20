"""Input Polish API — 输入润色端点。

Migrated from DeerFlow app/gateway/routers/input_polish.py.

在用户发送前自动优化 prompt 质量。
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["input-polish"])


class InputPolishRequest(BaseModel):
    text: str = Field(..., description="需要润色的草稿文本")
    locale: str | None = Field(default=None, description="语言提示")


class InputPolishResponse(BaseModel):
    rewritten_text: str = Field(..., description="润色后的文本")
    changed: bool = Field(..., description="是否有修改")


def _strip_think_blocks(text: str) -> str:
    """移除 思考... 块。"""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def _polish_text(text: str) -> tuple[str, bool]:
    """对输入文本进行基本润色。

    当前实现：
    - 去除首尾空白
    - 确保以句号/问号/感叹号结尾
    - 如果文本为空，则原样返回

    TODO: 集成 LLM 进行更智能的润色。
    """
    original = text
    text = text.strip()
    text = _strip_think_blocks(text)

    # 确保以标点结尾
    if text and text[-1] not in ".!?。！？":
        text += "。"
        changed = True
    else:
        changed = text != original

    return text, changed


@router.post("/input-polish", response_model=InputPolishResponse)
async def polish_input(body: InputPolishRequest):
    """润色用户输入文本。

    在发送给 LLM 之前优化用户输入的 prompt 质量。
    """
    rewritten, changed = _polish_text(body.text)
    return InputPolishResponse(rewritten_text=rewritten, changed=changed or rewritten != body.text)