"""Human Input 处理 — 结构化人类输入响应。

参考 DeerFlow agents/human_input.py 的版本化 TypedDict 模式。

用于 NovelFactory 的 interrupt/resume 流程中，承载结构化的人类输入数据。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class HumanInputTextResponse(TypedDict):
    """文本输入响应 (version 1)。"""

    version: Literal[1]
    type: Literal["text"]
    text: str


class HumanInputOptionResponse(TypedDict):
    """选项选择响应 (version 1)。"""

    version: Literal[1]
    type: Literal["option"]
    selected: str
    options: list[str]


class ReviewActionResponse(TypedDict):
    """评审操作响应 (version 1)。

    用于 NovelFactory 的 wait_for_review 节点。
    """

    version: Literal[1]
    type: Literal["review"]
    action: Literal["approve", "reject", "modify"]
    comment: str


# 联合类型
HumanInputResponse = HumanInputTextResponse | HumanInputOptionResponse | ReviewActionResponse


def _non_empty_string(value: Any) -> str | None:
    """提取非空字符串。"""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_human_input_response(data: dict[str, Any]) -> HumanInputResponse | None:
    """从 additional_kwargs 中读取并验证人类输入响应。

    Args:
        data: 从 Command(resume=...) 或 additional_kwargs 中获取的数据。

    Returns:
        验证通过的 HumanInputResponse，或 None（无效）。
    """
    version = data.get("version")
    if version != 1:
        return None

    msg_type = data.get("type")

    if msg_type == "text":
        text = _non_empty_string(data.get("text"))
        if text is None:
            return None
        return HumanInputTextResponse(version=1, type="text", text=text)

    if msg_type == "option":
        selected = _non_empty_string(data.get("selected"))
        options = data.get("options", [])
        if selected is None or not isinstance(options, list) or not options:
            return None
        return HumanInputOptionResponse(version=1, type="option", selected=selected, options=[str(o) for o in options])

    if msg_type == "review":
        action = data.get("action")
        if action not in ("approve", "reject", "modify"):
            return None
        comment = _non_empty_string(data.get("comment")) or ""
        return ReviewActionResponse(version=1, type="review", action=action, comment=comment)

    return None


__all__ = [
    "HumanInputTextResponse",
    "HumanInputOptionResponse",
    "ReviewActionResponse",
    "HumanInputResponse",
    "read_human_input_response",
]