from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _VCTools:
    """视频会议 — 对应 lark-cli vc 域。

    命令列表（7 个）：
      +search, +notes, +recording, +meeting-join,
      +meeting-leave, +meeting-list-active, +meeting-events
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def search(
        self,
        query: str = "",
        *,
        page_size: int = 20,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """搜索历史会议。"""
        args = ["vc", "+search", "--page-size", str(page_size)]
        if query:
            args.extend(["--query", query])
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def notes(
        self,
        meeting_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取会议纪要。"""
        args = ["vc", "+notes", "--meeting-id", meeting_id]
        return self._e.run(args, timeout=timeout)

    def recording(
        self,
        meeting_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取会议录制。"""
        args = ["vc", "+recording", "--meeting-id", meeting_id]
        return self._e.run(args, timeout=timeout)

    def meeting_join(
        self,
        meeting_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """加入会议。"""
        args = ["vc", "+meeting-join", "--meeting-id", meeting_id]
        return self._e.run(args, timeout=timeout)

    def meeting_leave(
        self,
        meeting_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """离开会议。"""
        args = ["vc", "+meeting-leave", "--meeting-id", meeting_id]
        return self._e.run(args, timeout=timeout)

    def meeting_list_active(self, *, timeout: int = 15) -> LarkResult:
        """列出进行中的会议。"""
        return self._e.run(["vc", "+meeting-list-active"], timeout=timeout)

    def meeting_events(
        self,
        meeting_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取会议实时事件。"""
        args = ["vc", "+meeting-events", "--meeting-id", meeting_id]
        return self._e.run(args, timeout=timeout)
