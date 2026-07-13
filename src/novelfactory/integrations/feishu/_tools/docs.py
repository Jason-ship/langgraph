from __future__ import annotations

from novelfactory.integrations.feishu._core import (
    _LARK_DOC_TIMEOUT,
    LarkResult,
    _LarkCLIEngine,
)


class _DocsTools:
    """云文档 — 对应 lark-cli docs 域。

    命令列表（11 个）：
      +search, +create, +fetch, +update,
      +media-insert, +media-upload, +media-preview, +media-download,
      +resource-download, +resource-update, +resource-delete
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def search(self, query: str, *, timeout: int = 30) -> LarkResult:
        """搜索文档。"""
        return self._e.run(
            ["docs", "+search", "--query", query],
            timeout=timeout,
        )

    def create(
        self,
        title: str,
        content_md: str,
        *,
        folder_token: str = "",
        timeout: int = _LARK_DOC_TIMEOUT,
    ) -> LarkResult:
        """创建文档。

        v5.7.2-fix: lark-cli v2 API 要求 ``--api-version v2``，标题嵌入内容开头
        （markdown 格式以 ``# 标题`` 开头），移除废弃的 ``--title`` 参数。
        章节正文中的 heading 全部降一级（#→##, ##→###），避免多 h1 导致标题变成 Untitled。
        """
        if not content_md or not content_md.strip():
            return LarkResult(success=False, error="content is empty")
        # 章节正文中的 #/## heading 全部降一级，避免与新加的文档标题 h1 冲突
        import re

        shifted = re.sub(r"^(#{1,5}) ", r"#\1 ", content_md, flags=re.MULTILINE)
        # v2 API: 标题嵌入内容，markdown 格式用 # 一级标题
        full_content = f"# {title}\n\n{shifted}"
        args = [
            "docs",
            "+create",
            "--api-version",
            "v2",
            "--doc-format",
            "markdown",
            "--as",
            "bot",
        ]
        if folder_token:
            args.extend(["--parent-token", folder_token])
        return self._e._run_with_tmpfile(
            args,
            full_content,
            timeout=timeout,
        )

    def fetch(
        self,
        doc_token: str,
        *,
        doc_format: str = "markdown",
        timeout: int = 30,
    ) -> LarkResult:
        """获取文档内容。"""
        args = ["docs", "+fetch", "--doc", doc_token, "--doc-format", doc_format]
        return self._e.run(args, timeout=timeout)

    def update(
        self,
        doc_token: str,
        content_md: str,
        *,
        timeout: int = _LARK_DOC_TIMEOUT,
    ) -> LarkResult:
        """更新文档内容（覆盖整篇）。"""
        if not content_md or not content_md.strip():
            return LarkResult(success=False, error="content is empty")
        args = [
            "docs",
            "+update",
            "--doc",
            doc_token,
            "--doc-format",
            "markdown",
            "--command",
            "overwrite",
        ]
        return self._e._run_with_tmpfile(args, content_md, timeout=timeout)

    def media_insert(
        self,
        doc_token: str,
        media_type: str,
        file_key: str,
        *,
        width: int = 0,
        height: int = 0,
        timeout: int = 30,
    ) -> LarkResult:
        """在文档中插入媒体（图片/文件）。"""
        args = [
            "docs",
            "+media-insert",
            "--doc-token",
            doc_token,
            "--media-type",
            media_type,
            "--file-key",
            file_key,
        ]
        if width > 0:
            args.extend(["--width", str(width)])
        if height > 0:
            args.extend(["--height", str(height)])
        return self._e.run(args, timeout=timeout)

    def media_upload(
        self,
        file_path: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """上传媒体文件到文档。"""
        args = ["docs", "+media-upload", "--file", file_path]
        return self._e.run(args, timeout=timeout)

    def media_preview(
        self,
        file_key: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """预览文档媒体。"""
        args = ["docs", "+media-preview", "--file-key", file_key]
        return self._e.run(args, timeout=timeout)

    def media_download(
        self,
        file_key: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """下载文档媒体。"""
        args = ["docs", "+media-download", "--file-key", file_key]
        return self._e.run(args, timeout=timeout)

    def resource_download(
        self,
        doc_token: str,
        file_token: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """下载文档资源。"""
        args = [
            "docs",
            "+resource-download",
            "--doc-token",
            doc_token,
            "--file-token",
            file_token,
        ]
        return self._e.run(args, timeout=timeout)

    def resource_update(
        self,
        doc_token: str,
        file_token: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """更新文档资源。"""
        args = [
            "docs",
            "+resource-update",
            "--doc-token",
            doc_token,
            "--file-token",
            file_token,
        ]
        return self._e.run(args, timeout=timeout)

    def resource_delete(
        self,
        file_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """删除文档资源。"""
        args = ["docs", "+resource-delete", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)
