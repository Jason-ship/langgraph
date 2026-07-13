from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _WhiteboardTools:
    """画板 — 对应 lark-cli whiteboard 域。

    命令列表（3 个）：
      +update, +update-old, +query
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def update(
        self,
        doc_token: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """更新画板（DSL 格式）。"""
        args = ["whiteboard", "+update", "--doc-token", doc_token]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".json",
            timeout=timeout,
        )

    def query(
        self,
        doc_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """查询画板内容。"""
        args = ["whiteboard", "+query", "--doc-token", doc_token]
        return self._e.run(args, timeout=timeout)
