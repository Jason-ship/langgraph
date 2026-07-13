# ==============================================================================
# 确定性 Thread ID 生成 — 借鉴 TradingAgents 的可重现性设计
# ==============================================================================

from __future__ import annotations

import hashlib
import time


def make_thread_id(
    project_name: str, date_str: str | None = None, *, short: bool = True
) -> str:
    """生成确定性 thread_id: SHA256(project_name:date)[:16]。

    借鉴 TradingAgents 的确定性 ID 设计：
    - 相同 project + date → 相同 thread_id → 可重复运行、可回溯
    - short=True 取前 16 字符 (8 字节 hex)，满足 UUID 兼容格式

    Args:
        project_name: 项目/小说名称。
        date_str: ISO 日期字符串 (YYYY-MM-DD)，默认为今天。
        short: True 返回 16 字符 hex，False 返回完整 64 字符。
    """
    if date_str is None:
        date_str = time.strftime("%Y-%m-%d")
    seed = f"{project_name}:{date_str}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return digest[:16] if short else digest
