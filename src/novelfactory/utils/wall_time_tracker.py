"""Wall-time tracker for full pipeline performance monitoring.

Borrowed from TradingAgents' AnalystWallTimeTracker pattern:
  - Per-node wall-time tracking with automatic reporting
  - Phase-level aggregation
  - Token cost × wall time efficiency ratio

v5.5: NovelFactory integration — track every node in main supervisor + crews.

Usage:
    tracker = WallTimeTracker()
    tracker.start("context_builder", phase="writing")
    ... node execution ...
    tracker.end("context_builder", token_count=15000)
    tracker.report()  # 打印全链路报告
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


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

    def get_total_seconds(self) -> float:
        return sum(r.duration_seconds for r in self._records.values())

    def get_total_tokens(self) -> int:
        return sum(r.token_count for r in self._records.values())

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
            lines.append(
                f"  {'Efficiency':35s} {total_tokens / total:>7,.0f} tokens/sec"
            )

        # Phase breakdown
        phase_totals = self.get_phase_totals()
        if phase_totals:
            lines.append("\n--- Phase Breakdown ---")
            for phase, stats in phase_totals.items():
                lines.append(
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
