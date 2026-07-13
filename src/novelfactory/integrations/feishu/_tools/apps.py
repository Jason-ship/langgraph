from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _AppsTools:
    """应用管理 — 对应 lark-cli apps 域。

    命令列表（25 个）：
      +create, +update, +list, +access-scope-set, +access-scope-get,
      +html-publish, +init, +release-create, +release-list, +release-get,
      +env-pull, +db-table-list, +db-table-get, +db-execute, +db-env-create,
      +git-credential-init, +git-credential-list, +git-credential-remove,
      +session-create, +session-list, +session-get, +session-stop,
      +session-messages-list, +chat
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def create(
        self,
        name: str,
        *,
        description: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建应用。"""
        args = ["apps", "+create", "--name", name]
        if description:
            args.extend(["--description", description])
        return self._e.run(args, timeout=timeout)

    def update(
        self,
        app_id: str,
        *,
        name: str = "",
        description: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """更新应用。"""
        args = ["apps", "+update", "--app-id", app_id]
        if name:
            args.extend(["--name", name])
        if description:
            args.extend(["--description", description])
        return self._e.run(args, timeout=timeout)

    def list(self, *, timeout: int = 15) -> LarkResult:
        """列出应用。"""
        return self._e.run(["apps", "+list"], timeout=timeout)

    def init(self, *, timeout: int = 15) -> LarkResult:
        """初始化应用。"""
        return self._e.run(["apps", "+init"], timeout=timeout)

    def html_publish(
        self,
        source_dir: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """发布 HTML 站点（妙搭）。"""
        args = ["apps", "+html-publish", "--source-dir", source_dir]
        return self._e.run(args, timeout=timeout)

    def release_create(
        self,
        version: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建应用发布。"""
        args = ["apps", "+release-create", "--version", version]
        return self._e.run(args, timeout=timeout)

    def release_list(self, *, timeout: int = 15) -> LarkResult:
        """列出发布记录。"""
        return self._e.run(["apps", "+release-list"], timeout=timeout)

    def env_pull(self, *, timeout: int = 15) -> LarkResult:
        """拉取环境变量。"""
        return self._e.run(["apps", "+env-pull"], timeout=timeout)

    def db_execute(
        self,
        query: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """执行数据库查询。"""
        args = ["apps", "+db-execute", "--query", query]
        return self._e.run(args, timeout=timeout)

    def session_create(
        self,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建会话。"""
        return self._e.run(["apps", "+session-create"], timeout=timeout)

    def session_list(self, *, timeout: int = 15) -> LarkResult:
        """列出会话。"""
        return self._e.run(["apps", "+session-list"], timeout=timeout)

    def session_messages_list(
        self,
        session_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取会话消息列表。"""
        args = ["apps", "+session-messages-list", "--session-id", session_id]
        return self._e.run(args, timeout=timeout)

    def chat(
        self,
        message: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """与 AI 应用聊天。"""
        args = ["apps", "+chat", "--message", message]
        return self._e.run(args, timeout=timeout)

    def access_scope_set(
        self,
        scopes: list[str],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """设置应用权限范围。"""
        args = ["apps", "+access-scope-set", "--scopes", ",".join(scopes)]
        return self._e.run(args, timeout=timeout)

    def access_scope_get(self, *, timeout: int = 15) -> LarkResult:
        """获取应用权限范围。"""
        return self._e.run(["apps", "+access-scope-get"], timeout=timeout)
