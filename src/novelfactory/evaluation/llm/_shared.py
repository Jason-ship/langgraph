"""LLM 评价分析模块共享工具函数。

提取公共函数消除 old_reader_llm.py 与 ai_style_llm.py 之间的重复。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from novelfactory.evaluation.llm.prompts import LLM_REVIEW_MAX_CHARS

if TYPE_CHECKING:
    pass


def trim_text(text: str, max_chars: int = LLM_REVIEW_MAX_CHARS) -> str:
    """智能裁剪章节文本，保留开头 + 中间采样 + 结尾。

    裁剪策略（60%保留率，约 75% 情节信息）：
      40% 开头
      30% 结尾
      30% 中间随机采样（取中间区域）
    """
    if not text or len(text) <= max_chars:
        return text

    head_len = int(max_chars * 0.4)
    tail_len = int(max_chars * 0.3)
    mid_len = max_chars - head_len - tail_len

    head = text[:head_len]
    tail = text[-tail_len:]

    mid_start = head_len
    mid_end = len(text) - tail_len
    if mid_end > mid_start:
        mid_center = (mid_start + mid_end) // 2
        mid_half = mid_len // 2
        mid = text[mid_center - mid_half : mid_center + mid_half]
    else:
        mid = ""

    return f"{head}\n[...]\n{mid}\n[...]\n{tail}"


def safe_match(pattern: re.Pattern[str], text: str) -> str | None:
    """安全执行正则匹配，失败返回 None。

    WebNovelBench 风格：标签化输出的解析使用简单正则，
    比 JSON 解析（validate_json_output）更鲁棒。
    """
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def extract_float(text: str | None, default: float = 0.0) -> float:
    """安全提取浮点数，失败返回 default。"""
    if text is None:
        return default
    try:
        return float(text.strip())
    except (ValueError, TypeError):
        return default


def extract_bool(text: str | None, default: bool = False) -> bool:
    """安全提取布尔值（true/false 不区分大小写）。"""
    if text is None:
        return default
    return text.strip().lower() == "true"


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """裁剪值到 [lo, hi] 区间。"""
    return max(lo, min(hi, value))
