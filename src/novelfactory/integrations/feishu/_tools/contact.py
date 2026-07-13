from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _ContactTools:
    """通讯录 — 对应 lark-cli contact 域。

    命令列表（2 个）：
      +search-user, +get-user
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def search_user(
        self,
        query: str,
        *,
        page_size: int = 20,
        timeout: int = 15,
    ) -> LarkResult:
        """搜索用户。"""
        args = [
            "contact",
            "+search-user",
            "--query",
            query,
            "--page-size",
            str(page_size),
        ]
        return self._e.run(args, timeout=timeout)

    def get_user(
        self,
        user_id: str,
        *,
        user_id_type: str = "open_id",
        timeout: int = 15,
    ) -> LarkResult:
        """获取用户信息。"""
        args = [
            "contact",
            "+get-user",
            "--user-id",
            user_id,
            "--user-id-type",
            user_id_type,
        ]
        return self._e.run(args, timeout=timeout)

    def resolve_name(
        self,
        name: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """按姓名查询用户 open_id。"""
        return self.search_user(name, timeout=timeout)
