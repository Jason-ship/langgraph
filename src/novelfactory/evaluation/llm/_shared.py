"""LLM 评价分析模块共享工具函数。

v8.5-clean: 清理 v8.2 前遗留的解析函数（safe_match/extract_float/extract_bool/clamp），
仅保留 trim_text（被 unified/prompts.py 引用）。
"""

from __future__ import annotations

from novelfactory.evaluation.llm.prompts import LLM_REVIEW_MAX_CHARS


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
