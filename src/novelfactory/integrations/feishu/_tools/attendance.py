from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _AttendanceTools:
    """考勤 — 通过 lark-cli 命令实现。

    功能：
      + 查询个人考勤记录
      + 查询考勤组信息
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def get_records(
        self,
        start_date: str,
        end_date: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """查询个人考勤记录。"""
        args = [
            "service",
            "attendance",
            "record",
            "list",
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--format",
            "json",
        ]
        return self._e.run(args, timeout=timeout)

    def get_groups(self, *, timeout: int = 15) -> LarkResult:
        """查询考勤组信息。"""
        args = ["service", "attendance", "group", "list", "--format", "json"]
        return self._e.run(args, timeout=timeout)
