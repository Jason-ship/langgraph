# ==============================================================================
# 实时报告持久化 — 借鉴 TradingAgents reporting.py 装饰器模式
# ==============================================================================

from __future__ import annotations

import functools
import json
import logging
import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RealtimeReportWriter:
    """实时报告写入器 — 线程安全，装饰器模式实时写文件。

    借鉴 TradingAgents: 在流式执行中增量写出 Markdown 报告，
    支持断点续传查看当前进度，避免流式崩溃丢失全部输出。
    """

    def __init__(self, output_dir: str | pathlib.Path = "./reports") -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffers: dict[str, list[str]] = {}

    def get_report_path(self, thread_id: str) -> pathlib.Path:
        return self._output_dir / f"{thread_id}_report.md"

    def append(self, thread_id: str, content: str) -> None:
        """增量追加报告内容。"""
        with self._lock:
            self._buffers.setdefault(thread_id, []).append(content)

    def flush(self, thread_id: str) -> None:
        """将所有缓冲内容写入磁盘。"""
        with self._lock:
            lines = self._buffers.pop(thread_id, [])
        if not lines:
            return
        path = self.get_report_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(lines))
        logger.info("Report flushed: %s (%d lines)", path, len(lines))


_report_writer: RealtimeReportWriter | None = None


def get_report_writer(output_dir: str = "./reports") -> RealtimeReportWriter:
    global _report_writer
    if _report_writer is None:
        _report_writer = RealtimeReportWriter(output_dir)
    return _report_writer


def log_chapter_to_report(thread_id: str, chapter_num: int, content: str) -> None:
    """记录章节内容到实时报告。"""
    writer = get_report_writer()
    header = f"\n\n## Chapter {chapter_num}\n\n*Written at {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    writer.append(thread_id, header)
    writer.append(thread_id, content)
    writer.flush(thread_id)


def log_quality_to_report(
    thread_id: str,
    chapter_num: int,
    quality: float,
    composite: float,
    passed: bool,
    issues: list[str] | None = None,
) -> None:
    """记录质量评审结果到实时报告。"""
    writer = get_report_writer()
    status = "✓ PASSED" if passed else "✗ FAILED"
    summary = [
        f"\n### Chapter {chapter_num} Quality Report\n\n",
        "| Metric | Score |\n|--------|-------|\n",
        f"| Quality | {quality:.1f}/100 |\n",
        f"| Composite | {composite:.3f} |\n",
        f"| Status | **{status}** |\n",
    ]
    if issues:
        summary.append("\n**Issues:**\n")
        for issue in issues:
            summary.append(f"- {issue}\n")
    writer.append(thread_id, "".join(summary))
    writer.flush(thread_id)


def with_report_logging(
    output_dir: str = "./reports",
) -> Callable[[F], F]:
    """装饰器 — 自动为线程创建报告文件并记录执行结果。

    借鉴 TradingAgents: 报告函数被 Agent 调用时自动写出 Markdown。
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            thread_id = kwargs.get("thread_id", "") or (
                args[0] if args and isinstance(args[0], str) else ""
            )
            result = func(*args, **kwargs)
            if thread_id and result:
                writer = get_report_writer(output_dir)
                if isinstance(result, str):
                    writer.append(thread_id, result)
                elif isinstance(result, dict):
                    writer.append(
                        thread_id, json.dumps(result, ensure_ascii=False, indent=2)
                    )
                writer.flush(thread_id)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
