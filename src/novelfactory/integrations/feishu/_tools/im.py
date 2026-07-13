from __future__ import annotations

import json

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _IMTools:
    """即时通讯 — 对应 lark-cli im 域。

    命令列表（20 个）：
      +messages-send, +messages-reply, +messages-mget, +messages-search,
      +messages-resources-download, +threads-messages-list, +chat-create,
      +chat-list, +chat-search, +chat-update, +chat-messages-list,
      +flag-create, +flag-cancel, +flag-list, +feed-shortcut-create,
      +feed-shortcut-remove, +feed-shortcut-list, +feed-group-list,
      +feed-group-list-item, +feed-group-query-item
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    # ── 消息发送 ─────────────────────────────────────────────────────

    def send_text(
        self,
        text: str,
        *,
        chat_id: str = "",
        user_id: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """发送纯文本消息。"""
        args = ["im", "+messages-send", "--as", "bot"]
        if chat_id:
            args.extend(["--chat-id", chat_id])
        elif user_id:
            args.extend(["--user-id", user_id])
        args.extend(["--text", text])
        return self._e.run(args, format_json=False, timeout=timeout)

    def send_markdown(
        self,
        markdown: str,
        *,
        chat_id: str = "",
        user_id: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """发送 Markdown 格式消息。"""
        args = ["im", "+messages-send", "--as", "bot"]
        if chat_id:
            args.extend(["--chat-id", chat_id])
        elif user_id:
            args.extend(["--user-id", user_id])
        args.extend(["--markdown", markdown])
        return self._e.run(args, format_json=False, timeout=timeout)

    def send_card(
        self,
        card: dict,
        *,
        chat_id: str = "",
        user_id: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """发送交互卡片消息。"""
        args = ["im", "+messages-send", "--as", "bot"]
        if chat_id:
            args.extend(["--chat-id", chat_id])
        elif user_id:
            args.extend(["--user-id", user_id])
        args.extend(
            [
                "--msg-type",
                "interactive",
                "--card",
                json.dumps(card, ensure_ascii=False),
            ]
        )
        return self._e.run(args, format_json=False, timeout=timeout)

    def send_image(
        self,
        image_key: str,
        *,
        chat_id: str = "",
        user_id: str = "",
        timeout: int = 30,
    ) -> LarkResult:
        """发送图片消息。"""
        args = ["im", "+messages-send", "--as", "bot"]
        if chat_id:
            args.extend(["--chat-id", chat_id])
        elif user_id:
            args.extend(["--user-id", user_id])
        args.extend(["--image", image_key])
        return self._e.run(args, format_json=False, timeout=timeout)

    def send_file(
        self,
        file_key: str,
        *,
        chat_id: str = "",
        user_id: str = "",
        timeout: int = 30,
    ) -> LarkResult:
        """发送文件消息。"""
        args = ["im", "+messages-send", "--as", "bot"]
        if chat_id:
            args.extend(["--chat-id", chat_id])
        elif user_id:
            args.extend(["--user-id", user_id])
        args.extend(["--file", file_key])
        return self._e.run(args, format_json=False, timeout=timeout)

    # ── 消息操作 ─────────────────────────────────────────────────────

    def reply_message(
        self,
        message_id: str,
        content: str,
        *,
        msg_type: str = "text",
        timeout: int = 15,
    ) -> LarkResult:
        """回复消息。"""
        args = [
            "im",
            "+messages-reply",
            "--message-id",
            message_id,
            "--content",
            json.dumps({"text": content}),
        ]
        return self._e.run(args, timeout=timeout)

    def get_messages(
        self,
        message_ids: list[str],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """批量获取消息。"""
        args = ["im", "+messages-mget"]
        for mid in message_ids:
            args.extend(["--message-id", mid])
        return self._e.run(args, timeout=timeout)

    def search_messages(
        self,
        query: str,
        *,
        page_size: int = 20,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """搜索消息历史。"""
        args = [
            "im",
            "+messages-search",
            "--query",
            query,
            "--page-size",
            str(page_size),
        ]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def download_resource(
        self,
        message_id: str,
        file_key: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """下载消息中的资源文件。"""
        args = [
            "im",
            "+messages-resources-download",
            "--message-id",
            message_id,
            "--file-key",
            file_key,
        ]
        return self._e.run(args, timeout=timeout)

    # ── 群管理 ───────────────────────────────────────────────────────

    def create_chat(
        self,
        name: str,
        *,
        description: str = "",
        user_ids: list[str] | None = None,
        timeout: int = 15,
    ) -> LarkResult:
        """创建群聊。"""
        args = ["im", "+chat-create", "--name", name]
        if description:
            args.extend(["--description", description])
        if user_ids:
            for uid in user_ids:
                args.extend(["--user-id", uid])
        return self._e.run(args, timeout=timeout)

    def list_chats(
        self,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取群列表。"""
        args = ["im", "+chat-list", "--page-size", str(page_size)]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def search_chat(
        self,
        query: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """搜索群聊。"""
        args = ["im", "+chat-search", "--query", query]
        return self._e.run(args, timeout=timeout)

    def update_chat(
        self,
        chat_id: str,
        *,
        name: str = "",
        description: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """更新群信息。"""
        args = ["im", "+chat-update", "--chat-id", chat_id]
        if name:
            args.extend(["--name", name])
        if description:
            args.extend(["--description", description])
        return self._e.run(args, timeout=timeout)

    def list_chat_messages(
        self,
        chat_id: str,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取群消息列表。"""
        args = [
            "im",
            "+chat-messages-list",
            "--chat-id",
            chat_id,
            "--page-size",
            str(page_size),
        ]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def list_thread_messages(
        self,
        chat_id: str,
        thread_id: str,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取主题消息列表。"""
        args = [
            "im",
            "+threads-messages-list",
            "--chat-id",
            chat_id,
            "--thread-id",
            thread_id,
            "--page-size",
            str(page_size),
        ]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    # ── 消息标记 ─────────────────────────────────────────────────────

    def create_flag(self, message_id: str, *, timeout: int = 15) -> LarkResult:
        """标记消息（星标）。"""
        args = ["im", "+flag-create", "--message-id", message_id]
        return self._e.run(args, timeout=timeout)

    def cancel_flag(self, message_id: str, *, timeout: int = 15) -> LarkResult:
        """取消消息标记。"""
        args = ["im", "+flag-cancel", "--message-id", message_id]
        return self._e.run(args, timeout=timeout)

    def list_flags(
        self,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取标记消息列表。"""
        args = ["im", "+flag-list", "--page-size", str(page_size)]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    # ── 消息中心（feed）─────────────────────────────────────────────

    def create_feed_shortcut(
        self,
        title: str,
        url: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建消息中心快捷方式。"""
        args = ["im", "+feed-shortcut-create", "--title", title, "--url", url]
        return self._e.run(args, timeout=timeout)

    def remove_feed_shortcut(
        self,
        shortcut_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """删除消息中心快捷方式。"""
        args = ["im", "+feed-shortcut-remove", "--id", shortcut_id]
        return self._e.run(args, timeout=timeout)

    def list_feed_shortcuts(self, *, timeout: int = 15) -> LarkResult:
        """列出消息中心快捷方式。"""
        return self._e.run(["im", "+feed-shortcut-list"], timeout=timeout)

    def list_feed_groups(self, *, timeout: int = 15) -> LarkResult:
        """列出消息中心分组。"""
        return self._e.run(["im", "+feed-group-list"], timeout=timeout)

    def query_feed_group_item(
        self,
        item_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """查询消息中心条目。"""
        args = ["im", "+feed-group-query-item", "--item-id", item_id]
        return self._e.run(args, timeout=timeout)
