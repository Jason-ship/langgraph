from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _NoteTools:
    """笔记 — 对应 lark-cli note 域。

    命令列表（2 个）：
      +detail, +transcript
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def detail(
        self,
        note_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取笔记详情。"""
        args = ["note", "+detail", "--note-id", note_id]
        return self._e.run(args, timeout=timeout)

    def transcript(
        self,
        note_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取笔记转写。"""
        args = ["note", "+transcript", "--note-id", note_id]
        return self._e.run(args, timeout=timeout)
