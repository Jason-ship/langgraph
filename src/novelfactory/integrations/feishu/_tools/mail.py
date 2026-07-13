from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _MailTools:
    """邮箱 — 对应 lark-cli mail 域。

    命令列表（19 个）：
      +message, +messages, +thread, +triage, +watch,
      +reply, +reply-all, +send, +draft-create, +draft-send,
      +draft-edit, +forward, +send-receipt, +decline-receipt,
      +signature, +share-to-chat, +template-create, +template-update,
      +lint-html
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        *,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        timeout: int = 30,
    ) -> LarkResult:
        """发送邮件。"""
        args = ["mail", "+send", "--to", ",".join(to), "--subject", subject]
        if cc:
            args.extend(["--cc", ",".join(cc)])
        if bcc:
            args.extend(["--bcc", ",".join(bcc)])
        # body 通过临时文件传递
        return self._e._run_with_tmpfile(
            args,
            body,
            suffix=".html",
            timeout=timeout,
        )

    def draft_create(
        self,
        to: list[str],
        subject: str,
        body: str,
        *,
        cc: list[str] | None = None,
        timeout: int = 30,
    ) -> LarkResult:
        """创建草稿。"""
        args = ["mail", "+draft-create", "--to", ",".join(to), "--subject", subject]
        if cc:
            args.extend(["--cc", ",".join(cc)])
        return self._e._run_with_tmpfile(
            args,
            body,
            suffix=".html",
            timeout=timeout,
        )

    def draft_send(self, draft_id: str, *, timeout: int = 15) -> LarkResult:
        """发送草稿。"""
        args = ["mail", "+draft-send", "--draft-id", draft_id]
        return self._e.run(args, timeout=timeout)

    def draft_edit(
        self,
        draft_id: str,
        *,
        body: str = "",
        subject: str = "",
        timeout: int = 30,
    ) -> LarkResult:
        """编辑草稿。"""
        args = ["mail", "+draft-edit", "--draft-id", draft_id]
        if subject:
            args.extend(["--subject", subject])
        if body:
            return self._e._run_with_tmpfile(
                args,
                body,
                suffix=".html",
                timeout=timeout,
            )
        return self._e.run(args, timeout=timeout)

    def messages(
        self,
        *,
        page_size: int = 20,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取邮件列表。"""
        args = ["mail", "+messages", "--page-size", str(page_size)]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def message(self, message_id: str, *, timeout: int = 15) -> LarkResult:
        """获取单封邮件详情。"""
        args = ["mail", "+message", "--message-id", message_id]
        return self._e.run(args, timeout=timeout)

    def thread(
        self,
        thread_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取邮件会话。"""
        args = ["mail", "+thread", "--thread-id", thread_id]
        return self._e.run(args, timeout=timeout)

    def triage(
        self,
        *,
        folder: str = "INBOX",
        page_size: int = 20,
        timeout: int = 15,
    ) -> LarkResult:
        """分类整理邮件。"""
        args = ["mail", "+triage", "--folder", folder, "--page-size", str(page_size)]
        return self._e.run(args, timeout=timeout)

    def watch(
        self,
        *,
        max_results: int = 10,
        timeout: int = 15,
    ) -> LarkResult:
        """监听新邮件。"""
        args = ["mail", "+watch", "--max-results", str(max_results)]
        return self._e.run(args, timeout=timeout)

    def reply(
        self,
        message_id: str,
        body: str,
        *,
        reply_all: bool = False,
        timeout: int = 30,
    ) -> LarkResult:
        """回复邮件。"""
        cmd = "+reply-all" if reply_all else "+reply"
        args = ["mail", cmd, "--message-id", message_id]
        return self._e._run_with_tmpfile(
            args,
            body,
            suffix=".html",
            timeout=timeout,
        )

    def reply_all(
        self,
        message_id: str,
        body: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """回复全部。"""
        return self.reply(message_id, body, reply_all=True, timeout=timeout)

    def forward(
        self,
        message_id: str,
        to: list[str],
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """转发邮件。"""
        args = ["mail", "+forward", "--message-id", message_id, "--to", ",".join(to)]
        return self._e.run(args, timeout=timeout)

    def send_receipt(
        self,
        message_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """发送已读回执。"""
        args = ["mail", "+send-receipt", "--message-id", message_id]
        return self._e.run(args, timeout=timeout)

    def decline_receipt(
        self,
        message_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """拒绝已读回执。"""
        args = ["mail", "+decline-receipt", "--message-id", message_id]
        return self._e.run(args, timeout=timeout)

    def signature(self, *, content: str = "", timeout: int = 15) -> LarkResult:
        """获取/设置签名。"""
        args = ["mail", "+signature"]
        if content:
            args.extend(["--content", content])
        return self._e.run(args, timeout=timeout)

    def share_to_chat(
        self,
        message_id: str,
        chat_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """分享邮件到群聊。"""
        args = [
            "mail",
            "+share-to-chat",
            "--message-id",
            message_id,
            "--chat-id",
            chat_id,
        ]
        return self._e.run(args, timeout=timeout)

    def template_create(
        self,
        name: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建邮件模板。"""
        args = ["mail", "+template-create", "--name", name]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".html",
            timeout=timeout,
        )

    def template_update(
        self,
        template_id: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """更新邮件模板。"""
        args = ["mail", "+template-update", "--template-id", template_id]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".html",
            timeout=timeout,
        )

    def lint_html(
        self,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """检查邮件 HTML 合规性。"""
        args = ["mail", "+lint-html"]
        return self._e._run_with_tmpfile(
            args,
            content,
            suffix=".html",
            timeout=timeout,
        )
