"""Wall-time tracker for full pipeline performance monitoring.

Borrowed from TradingAgents' AnalystWallTimeTracker pattern,
optimized with DeerFlow RunJournal CallbackHandler auto-capture pattern:
  - Per-node wall-time tracking with automatic reporting
  - Phase-level aggregation
  - Token cost × wall time efficiency ratio
  - Callback-based auto-capture (no manual instrumentation needed)
  - Tag-based caller identification for per-agent stats

Usage:
    tracker = WallTimeTracker()
    tracker.start("context_builder", phase="writing")
    ... node execution ...
    tracker.end("context_builder", token_count=15000)
    tracker.report()  # 打印全链路报告

    # Auto-capture with CallbackHandler:
    handler = tracker.get_callback_handler()
    # Pass handler to LangChain as a callback
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


@dataclass
class WallTimeRecord:
    node_name: str
    start_time: float
    end_time: float = 0.0
    duration_seconds: float = 0.0
    token_count: int = 0
    llm_calls: int = 0
    phase: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class WallTimeTracker:
    """全链路时延追踪器。

    Usage:
        tracker = WallTimeTracker()
        tracker.start("context_builder", phase="writing")
        ... node execution ...
        tracker.end("context_builder", token_count=15000)
        tracker.report()  # 打印全链路报告
    """

    def __init__(self) -> None:
        self._records: dict[str, WallTimeRecord] = {}
        self._phase_start: dict[str, float] = {}
        self._phase_records: dict[str, list[WallTimeRecord]] = {}
        # CallbackHandler 自动采集
        self._callback_handler: WallTimeCallbackHandler | None = None

    def start(self, node_name: str, phase: str = "", **metadata: Any) -> None:
        self._records[node_name] = WallTimeRecord(
            node_name=node_name,
            start_time=time.time(),
            phase=phase,
            metadata=metadata,
        )
        if phase and phase not in self._phase_start:
            self._phase_start[phase] = time.time()

    def end(self, node_name: str, token_count: int = 0, llm_calls: int = 0) -> float:
        record = self._records.get(node_name)
        if not record:
            return 0.0
        record.end_time = time.time()
        record.duration_seconds = record.end_time - record.start_time
        record.token_count = token_count
        record.llm_calls = llm_calls

        phase = record.phase
        if phase:
            self._phase_records.setdefault(phase, []).append(record)

        return record.duration_seconds

    def get_callback_handler(self) -> WallTimeCallbackHandler:
        """获取 CallbackHandler 实例，用于自动采集 LLM 调用。

        将此 handler 传递给 LangChain 的 callbacks 参数，
        即可自动捕获 LLM 调用的耗时和 Token 消耗。
        """
        if self._callback_handler is None:
            self._callback_handler = WallTimeCallbackHandler(tracker=self)
        return self._callback_handler

    def get_total_seconds(self) -> float:
        return sum(r.duration_seconds for r in self._records.values())

    def get_total_tokens(self) -> int:
        return sum(r.token_count for r in self._records.values())

    def get_total_llm_calls(self) -> int:
        return sum(r.llm_calls for r in self._records.values())

    def get_phase_totals(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, Any]] = {}
        for phase, records in self._phase_records.items():
            total_seconds = sum(r.duration_seconds for r in records)
            total_tokens = sum(r.token_count for r in records)
            total_llm_calls = sum(r.llm_calls for r in records)
            result[phase] = {
                "seconds": total_seconds,
                "tokens": total_tokens,
                "llm_calls": total_llm_calls,
                "node_count": len(records),
            }
        return result

    def get_summary(self) -> dict[str, Any]:
        """获取汇总数据（用于 Console API）。"""
        return {
            "total_runs": len(self._records),
            "active_runs": sum(1 for r in self._records.values() if r.end_time == 0),
            "failed_runs": sum(1 for r in self._records.values() if r.metadata.get("error")),
            "total_tokens": self.get_total_tokens(),
            "total_llm_calls": self.get_total_llm_calls(),
            "total_seconds": self.get_total_seconds(),
            "total_threads": len(set(r.metadata.get("thread_id", "") for r in self._records.values() if r.metadata.get("thread_id"))),
        }

    def get_recent_runs(self, limit: int = 20, offset: int = 0, cutoff: str | None = None) -> list[dict[str, Any]]:
        """获取最近运行记录（用于 Console API）。"""
        records = sorted(self._records.values(), key=lambda r: r.start_time, reverse=True)
        if cutoff:
            records = [r for r in records if time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r.start_time)) >= cutoff]
        return [
            {
                "run_id": r.node_name,
                "thread_id": r.metadata.get("thread_id", ""),
                "status": "completed" if r.end_time > 0 else "running",
                "duration_seconds": r.duration_seconds,
                "total_tokens": r.token_count,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r.start_time)),
            }
            for r in records[offset : offset + limit]
        ]

    def report(self) -> str:
        lines: list[str] = ["\n===== Wall Time Report ====="]
        total = 0.0
        total_tokens = 0
        total_llm_calls = 0
        for name, record in self._records.items():
            lines.append(
                f"  {name:35s} {record.duration_seconds:7.1f}s  "
                f"{record.token_count:>8,d} tokens  "
                f"{record.llm_calls:>4,d} LLM calls  "
                f"phase={record.phase}"
            )
            total += record.duration_seconds
            total_tokens += record.token_count
            total_llm_calls += record.llm_calls
        lines.append("-" * 70)
        lines.append(
            f"  {'TOTAL':35s} {total:7.1f}s  {total_tokens:>8,d} tokens  "
            f"{total_llm_calls:>4,d} LLM calls"
        )
        if total_tokens > 0 and total > 0:
            lines.append(f"  {'Efficiency':35s} {total_tokens / total:>7,.0f} tokens/sec")

        phase_totals = self.get_phase_totals()
        if phase_totals:
            lines.append("\n--- Phase Breakdown ---")
            for phase, stats in phase_totals.items():
                lines.append(f"  {phase:20s} {stats['seconds']:7.1f}s  {stats['tokens']:>8,d} tokens")

        return "\n".join(lines)


class WallTimeCallbackHandler(BaseCallbackHandler):
    """LangChain CallbackHandler，自动采集 LLM 调用的耗时和 Token 消耗。

    参考 DeerFlow RunJournal 的自动采集模式。
    将此 handler 添加到 LangChain 的 callbacks 列表中即可自动捕获。

    Usage:
        tracker = WallTimeTracker()
        handler = tracker.get_callback_handler()
        llm = ChatOpenAI(callbacks=[handler])
        # 或
        agent = create_react_agent(llm, tools, callbacks=[handler])
    """

    name = "wall_time_tracker"

    def __init__(self, tracker: WallTimeTracker) -> None:
        super().__init__()
        self._tracker = tracker
        self._start_times: dict[str, float] = {}
        self._counted_run_ids: set[str] = set()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """LLM 调用开始时记录时间戳。"""
        run_id = str(kwargs.get("run_id", ""))
        self._start_times[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束时记录耗时和 Token 消耗。"""
        run_id = str(kwargs.get("run_id", ""))
        if run_id in self._counted_run_ids:
            return
        self._counted_run_ids.add(run_id)

        start = self._start_times.pop(run_id, None)
        if start is None:
            return

        duration = time.time() - start
        token_count = 0
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {}) if isinstance(response.llm_output, dict) else {}
            if isinstance(token_usage, dict):
                token_count = token_usage.get("total_tokens", 0) or 0

        # 记录到 tracker
        node_name = f"llm_{run_id[:8]}"
        self._tracker._records[node_name] = WallTimeRecord(
            node_name=node_name,
            start_time=start,
            end_time=time.time(),
            duration_seconds=duration,
            token_count=token_count,
            llm_calls=1,
            phase="llm",
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """LLM 调用出错时记录。"""
        run_id = str(kwargs.get("run_id", ""))
        self._start_times.pop(run_id, None)
                    f"  {phase:35s} {stats['seconds']:7.1f}s  "
                    f"{stats['tokens']:>8,d} tokens  "
                    f"{stats['llm_calls']:>4,d} LLM calls  "
                    f"({stats['node_count']} nodes)"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [
                {
                    "node": r.node_name,
                    "duration_s": round(r.duration_seconds, 2),
                    "tokens": r.token_count,
                    "llm_calls": r.llm_calls,
                    "phase": r.phase,
                    "metadata": r.metadata,
                }
                for r in self._records.values()
            ],
            "total_s": round(self.get_total_seconds(), 2),
            "total_tokens": self.get_total_tokens(),
            "total_llm_calls": sum(r.llm_calls for r in self._records.values()),
            "phase_breakdown": {
                phase: {
                    "seconds": round(s["seconds"], 2),
                    "tokens": s["tokens"],
                    "llm_calls": s["llm_calls"],
                    "nodes": s["node_count"],
                }
                for phase, s in self.get_phase_totals().items()
            },
        }

    def reset(self) -> None:
        self._records.clear()
        self._phase_start.clear()
        self._phase_records.clear()

    def __repr__(self) -> str:
        return f"WallTimeTracker(records={len(self._records)}, total={self.get_total_seconds():.1f}s)"


# ── State-compatible tracker factory ───────────────────────────────────────


def create_tracker_from_state(state: dict[str, Any]) -> WallTimeTracker:
    """从状态中恢复或创建新的 WallTimeTracker。

    如果 state 中有 serialized_wall_time 数据，恢复 tracker；
    否则创建新实例。
    """
    serialized = state.get("wall_time_data", "")
    tracker = WallTimeTracker()

    if serialized:
        try:
            data = json.loads(serialized)
            for r_data in data.get("records", []):
                record = WallTimeRecord(
                    node_name=r_data["node"],
                    start_time=0,
                    end_time=0,
                    duration_seconds=r_data["duration_s"],
                    token_count=r_data["tokens"],
                    llm_calls=r_data.get("llm_calls", 0),
                    phase=r_data.get("phase", ""),
                )
                tracker._records[record.node_name] = record
        except (json.JSONDecodeError, KeyError):
            pass

    return tracker
