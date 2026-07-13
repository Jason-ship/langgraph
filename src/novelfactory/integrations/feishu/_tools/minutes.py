from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _MinutesTools:
    """妙记 — 对应 lark-cli minutes 域。

    命令列表（8 个）：
      +search, +download, +upload, +update,
      +summary, +todo, +speaker-replace, +word-replace
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
        """搜索妙记。"""
        args = ["minutes", "+search", "--page-size", str(page_size)]
        if query:
            args.extend(["--query", query])
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def download(
        self,
        minute_token: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """下载妙记音视频。"""
        args = ["minutes", "+download", "--minute-token", minute_token]
        return self._e.run(args, timeout=timeout)

    def upload(
        self,
        file_path: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """上传音视频生成妙记。"""
        args = ["minutes", "+upload", "--file", file_path]
        return self._e.run(args, timeout=timeout)

    def update(
        self,
        minute_token: str,
        *,
        title: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """更新妙记信息。"""
        args = ["minutes", "+update", "--minute-token", minute_token]
        if title:
            args.extend(["--title", title])
        return self._e.run(args, timeout=timeout)

    def summary(
        self,
        minute_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取妙记总结。"""
        args = ["minutes", "+summary", "--minute-token", minute_token]
        return self._e.run(args, timeout=timeout)

    def todo(
        self,
        minute_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取妙记待办。"""
        args = ["minutes", "+todo", "--minute-token", minute_token]
        return self._e.run(args, timeout=timeout)

    def speaker_replace(
        self,
        minute_token: str,
        old_speaker: str,
        new_speaker: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """替换妙记发言人。"""
        args = [
            "minutes",
            "+speaker-replace",
            "--minute-token",
            minute_token,
            "--old-speaker",
            old_speaker,
            "--new-speaker",
            new_speaker,
        ]
        return self._e.run(args, timeout=timeout)

    def word_replace(
        self,
        minute_token: str,
        old_word: str,
        new_word: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """替换妙记文字。"""
        args = [
            "minutes",
            "+word-replace",
            "--minute-token",
            minute_token,
            "--old-word",
            old_word,
            "--new-word",
            new_word,
        ]
        return self._e.run(args, timeout=timeout)
