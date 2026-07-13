from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _CalendarTools:
    """日历 — 对应 lark-cli calendar 域。

    命令列表（7 个）：
      +agenda, +create, +update, +freebusy, +room-find, +rsvp, +suggestion
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def agenda(
        self,
        *,
        date: str = "",
        page_size: int = 20,
        timeout: int = 15,
    ) -> LarkResult:
        """查看日程列表。"""
        args = ["calendar", "+agenda", "--page-size", str(page_size)]
        if date:
            args.extend(["--date", date])
        return self._e.run(args, timeout=timeout)

    def create(
        self,
        summary: str,
        *,
        start_time: str = "",
        end_time: str = "",
        description: str = "",
        attendees: list[str] | None = None,
        timeout: int = 15,
    ) -> LarkResult:
        """创建日程。"""
        args = ["calendar", "+create", "--summary", summary]
        if start_time:
            args.extend(["--start-time", start_time])
        if end_time:
            args.extend(["--end-time", end_time])
        if description:
            args.extend(["--description", description])
        if attendees:
            for att in attendees:
                args.extend(["--attendee", att])
        return self._e.run(args, timeout=timeout)

    def update(
        self,
        event_id: str,
        *,
        summary: str = "",
        start_time: str = "",
        end_time: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """更新日程。"""
        args = ["calendar", "+update", "--event-id", event_id]
        if summary:
            args.extend(["--summary", summary])
        if start_time:
            args.extend(["--start-time", start_time])
        if end_time:
            args.extend(["--end-time", end_time])
        return self._e.run(args, timeout=timeout)

    def freebusy(
        self,
        start_time: str,
        end_time: str,
        *,
        user_ids: list[str] | None = None,
        timeout: int = 15,
    ) -> LarkResult:
        """查询忙闲状态。"""
        args = [
            "calendar",
            "+freebusy",
            "--start-time",
            start_time,
            "--end-time",
            end_time,
        ]
        if user_ids:
            for uid in user_ids:
                args.extend(["--user-id", uid])
        return self._e.run(args, timeout=timeout)

    def room_find(
        self,
        *,
        capacity: int = 0,
        building: str = "",
        floor: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """查找会议室。"""
        args = ["calendar", "+room-find"]
        if capacity > 0:
            args.extend(["--capacity", str(capacity)])
        if building:
            args.extend(["--building", building])
        if floor:
            args.extend(["--floor", floor])
        return self._e.run(args, timeout=timeout)

    def rsvp(
        self,
        event_id: str,
        response: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """回复日程邀请（accepted/declined/tentative）。"""
        args = ["calendar", "+rsvp", "--event-id", event_id, "--response", response]
        return self._e.run(args, timeout=timeout)

    def suggestion(
        self,
        start_time: str,
        end_time: str,
        *,
        user_ids: list[str] | None = None,
        timeout: int = 15,
    ) -> LarkResult:
        """推荐空闲时段。"""
        args = [
            "calendar",
            "+suggestion",
            "--start-time",
            start_time,
            "--end-time",
            end_time,
        ]
        if user_ids:
            for uid in user_ids:
                args.extend(["--user-id", uid])
        return self._e.run(args, timeout=timeout)
