from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _MarkdownTools:
    """Markdown 文件 — 对应 lark-cli markdown 域。

    命令列表（5 个）：
      +create, +fetch, +patch, +overwrite, +diff
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def create(
        self,
        title: str,
        content: str,
        *,
        folder_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建 Markdown 文件。"""
        args = ["markdown", "+create", "--title", title]
        if folder_token:
            args.extend(["--folder-token", folder_token])
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".md",
            timeout=timeout,
        )

    def fetch(
        self,
        file_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取 Markdown 文件内容。"""
        args = ["markdown", "+fetch", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def patch(
        self,
        file_token: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """局部更新 Markdown 文件。"""
        args = ["markdown", "+patch", "--file-token", file_token]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".md",
            timeout=timeout,
        )

    def overwrite(
        self,
        file_token: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """覆盖 Markdown 文件。"""
        args = ["markdown", "+overwrite", "--file-token", file_token]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".md",
            timeout=timeout,
        )

    def diff(
        self,
        file_token: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """比较 Markdown 文件差异。"""
        args = ["markdown", "+diff", "--file-token", file_token]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".md",
            timeout=timeout,
        )
