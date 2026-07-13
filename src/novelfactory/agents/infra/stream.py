"""Streaming output to temp files for crew visibility."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class StreamWriter:
    """Write incremental output to a temp file for user visibility (v5.1.1: buffered).

    优化: 内部缓冲多个 write 调用, 每 5 次或 close 时批量写入, 减少 fsync 次数。
    """

    _BUFFER_SIZE = 5  # 缓冲阈值: 积累 5 次 write 后批量刷盘

    def __init__(self, prefix: str = "novelfactory", project_name: str = "") -> None:
        import tempfile

        tmp_dir = Path(tempfile.gettempdir())
        safe_prefix = "".join(c if c.isalnum() else "_" for c in prefix)
        if project_name:
            safe_prefix = f"{safe_prefix}_{project_name}"
        self._path = tmp_dir / f"{safe_prefix}.md"
        self._closed = False
        self._buffer: list[str] = []
        self._lock = threading.Lock()  # v5.2: instance-level thread safety
        with Path.open(self._path, "w", encoding="utf-8") as f:
            f.write("")
        self._write_count = 0

    @property
    def path(self) -> str:
        return self._path

    def write(self, content: str) -> None:
        """Append content to the stream file (buffered batch flush, thread-safe)."""
        if self._closed:
            return
        try:
            with self._lock:
                self._buffer.append(content)
                self._write_count += 1
                # 每积累 _BUFFER_SIZE 次写入触发批量刷盘（锁内调用 _flush 保持原子性）
                should_flush = len(self._buffer) >= self._BUFFER_SIZE
            if should_flush:
                self._flush()
        except OSError:
            self._closed = True

    def _flush(self) -> None:
        """将缓冲区内容批量写入磁盘（线程安全）。"""
        try:
            with self._lock:
                if not self._buffer:
                    return
                with Path.open(self._path, "a", encoding="utf-8") as f:
                    f.write("".join(self._buffer))
                    f.flush()
                self._buffer.clear()
        except OSError:
            self._closed = True

    def section(self, title: str) -> None:
        """Write a section header."""
        self.write(f"\n\n## {title}\n\n")

    def close(self) -> None:
        """Close the stream, flushing any remaining buffered content."""
        self._flush()
        self._closed = True

    def __enter__(self) -> StreamWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ── Crew Stream Manager ───────────────────────────────────────────────────────
# Shared across writing_crew, media_crew, sync_crew to avoid duplicating
# the _get_stream_writer / _cleanup_stream_writer pattern in each file.

_crew_streams: dict[str, StreamWriter] = {}
_crew_stream_lock = threading.Lock()


def get_crew_stream(crew_name: str, prefix: str) -> StreamWriter:
    """Get or create a StreamWriter for a crew node.

    Args:
        crew_name: Short identifier like "writing", "media", "sync".
        prefix: Per-chapter prefix like f"ch{chapter_number}".

    Returns:
        A StreamWriter instance (shared within the same crew+prefix combo).
    """
    key = f"{crew_name}_{prefix}"
    with _crew_stream_lock:
        if key not in _crew_streams:
            _crew_streams[key] = StreamWriter(prefix=prefix)
        return _crew_streams[key]


def cleanup_crew_stream(crew_name: str, prefix: str) -> None:
    """Close and remove a StreamWriter for a crew node."""
    key = f"{crew_name}_{prefix}"
    with _crew_stream_lock:
        sw = _crew_streams.pop(key, None)
    if sw:
        try:
            sw.close()
        except OSError:
            import logging as _logging

            _logging.getLogger("novelfactory.agents.infra.stream").warning(
                "Failed to close crew stream %s/%s", crew_name, prefix
            )


def cleanup_all_crew_streams() -> None:
    """Close all crew streams (e.g. on subgraph exit)."""
    global _crew_streams
    with _crew_stream_lock:
        streams = _crew_streams
        _crew_streams = {}
    for key, sw in streams.items():
        try:
            sw.close()
        except OSError:
            pass
