from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _SlidesTools:
    """幻灯片 — 对应 lark-cli slides 域。

    命令列表（4 个）：
      +create, +media-upload, +replace-slide, +screenshot
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def create(
        self,
        title: str,
        *,
        folder_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建幻灯片。"""
        args = ["slides", "+create", "--title", title]
        if folder_token:
            args.extend(["--folder-token", folder_token])
        return self._e.run(args, timeout=timeout)

    def media_upload(
        self,
        file_path: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """上传幻灯片媒体。"""
        args = ["slides", "+media-upload", "--file", file_path]
        return self._e.run(args, timeout=timeout)

    def replace_slide(
        self,
        presentation_token: str,
        slide_id: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """替换幻灯片页面。"""
        return self._e._run_with_tmpfile(
            [
                "slides",
                "+replace-slide",
                "--presentation-token",
                presentation_token,
                "--slide-id",
                slide_id,
            ],
            content,
            suffix=".md",
            timeout=timeout,
        )

    def screenshot(
        self,
        presentation_token: str,
        slide_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """截图幻灯片页面。"""
        args = [
            "slides",
            "+screenshot",
            "--presentation-token",
            presentation_token,
            "--slide-id",
            slide_id,
        ]
        return self._e.run(args, timeout=timeout)
