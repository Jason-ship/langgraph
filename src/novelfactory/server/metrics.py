"""
Prometheus 指标收集模块

使用 Python 标准库实现 Prometheus 文本格式指标端点。
无需安装 prometheus_client 等外部依赖。
"""

from __future__ import annotations

import threading
from collections import defaultdict

__all__ = [
    "increment_llm_calls",
    "increment_chapters",
    "increment_verdict_level",
    "generate_metrics",
]

# ---------------------------------------------------------------------------
# 全局指标字典
# ---------------------------------------------------------------------------

_llm_calls_total: dict[tuple[str, str, str], int] = defaultdict(int)
_chapters_total: int = 0
_verdict_level_total: dict[str, int] = defaultdict(int)

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

_ESCAPE_TABLE = str.maketrans({"\\": "\\\\", "\n": "\\n", '"': '\\"'})


def _sanitize_label_value(value: str) -> str:
    """转义 Prometheus 标签值中的特殊字符。"""
    return value.translate(_ESCAPE_TABLE)


# ---------------------------------------------------------------------------
# Increment 函数
# ---------------------------------------------------------------------------


def increment_llm_calls(model: str = "", step: str = "", status: str = "") -> None:
    """记录一次 LLM 调用。

    Parameters
    ----------
    model : str
        使用的模型名称。
    step : str
        调用发生的步骤/阶段名称。
    status : str
        调用结果状态（success / error / timeout 等）。
    """
    key = (_sanitize_label_value(model),
           _sanitize_label_value(step),
           _sanitize_label_value(status))
    with _lock:
        _llm_calls_total[key] += 1


def increment_chapters() -> None:
    """记录一章生成完成。"""
    global _chapters_total
    with _lock:
        _chapters_total += 1


def increment_verdict_level(level: str) -> None:
    """记录一次评分等级分布。

    Parameters
    ----------
    level : str
        评分等级标识，例如 ``"excellent"``, ``"good"``, ``"needs_rewrite"`` 等。
    """
    safe = _sanitize_label_value(level)
    with _lock:
        _verdict_level_total[safe] += 1


# ---------------------------------------------------------------------------
# 指标导出
# ---------------------------------------------------------------------------

_METRIC_HELP = {
    "novel_llm_calls_total": "Total number of LLM calls, partitioned by model / step / status.",
    "novel_chapters_total": "Total number of generated chapters.",
    "novel_verdict_level_total": "Distribution of verdict quality levels.",
}

_METRIC_TYPE = {
    "novel_llm_calls_total": "counter",
    "novel_chapters_total": "counter",
    "novel_verdict_level_total": "counter",
}

# 编译一次正则避免运行时重复构造
import re  # noqa: E402
_INVALID_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")


def _valid_metric_name(name: str) -> str:
    """确保指标名仅含 [a-zA-Z0-9_:]，否则替换为下划线。"""
    return _INVALID_NAME_RE.sub("_", name)


def generate_metrics() -> str:
    """生成 Prometheus 文本格式的完整指标输出。

    Returns
    -------
    str
        Prometheus ``text/plain`` 格式的指标字符串，
        可直接作为 HTTP Response body 返回。
    """
    with _lock:
        llm_snapshot = dict(_llm_calls_total)
        chapters_snapshot = _chapters_total
        verdict_snapshot = dict(_verdict_level_total)

    lines: list[str] = []

    # ---- novel_llm_calls_total ----
    name = _valid_metric_name("novel_llm_calls_total")
    lines.append(f"# HELP {name} {_METRIC_HELP[name]}")
    lines.append(f"# TYPE {name} {_METRIC_TYPE[name]}")
    for (model, step, status), cnt in sorted(llm_snapshot.items()):
        labels = f'model="{model}",step="{step}",status="{status}"'
        lines.append(f"{name}{{{labels}}} {cnt}")
    lines.append("")

    # ---- novel_chapters_total ----
    name = _valid_metric_name("novel_chapters_total")
    lines.append(f"# HELP {name} {_METRIC_HELP[name]}")
    lines.append(f"# TYPE {name} {_METRIC_TYPE[name]}")
    lines.append(f"{name} {chapters_snapshot}")
    lines.append("")

    # ---- novel_verdict_level_total ----
    name = _valid_metric_name("novel_verdict_level_total")
    lines.append(f"# HELP {name} {_METRIC_HELP[name]}")
    lines.append(f"# TYPE {name} {_METRIC_TYPE[name]}")
    for level, cnt in sorted(verdict_snapshot.items()):
        labels = f'level="{level}"'
        lines.append(f"{name}{{{labels}}} {cnt}")
    lines.append("")

    return "\n".join(lines)
