from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _EventTools:
    """事件订阅 — 对应 lark-cli event 域。

    命令列表：
      +subscribe
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def subscribe(
        self,
        event_key: str,
        *,
        webhook_url: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """订阅事件。"""
        args = ["event", "+subscribe", "--event-key", event_key]
        if webhook_url:
            args.extend(["--webhook-url", webhook_url])
        return self._e.run(args, timeout=timeout)
