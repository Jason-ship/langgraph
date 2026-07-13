from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _WikiTools:
    """知识库 — 对应 lark-cli wiki 域。

    命令列表（12 个）：
      +move, +node-create, +delete-space, +space-list, +space-create,
      +node-list, +node-copy, +node-get, +node-delete,
      +member-add, +member-remove, +member-list
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def space_list(self, *, timeout: int = 15) -> LarkResult:
        """获取知识空间列表。"""
        return self._e.run(["wiki", "+space-list"], timeout=timeout)

    def space_create(
        self,
        name: str,
        *,
        description: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建知识空间。"""
        args = ["wiki", "+space-create", "--name", name]
        if description:
            args.extend(["--description", description])
        return self._e.run(args, timeout=timeout)

    def delete_space(self, space_id: str, *, timeout: int = 15) -> LarkResult:
        """删除知识空间。"""
        args = ["wiki", "+delete-space", "--space-id", space_id]
        return self._e.run(args, timeout=timeout)

    def node_list(
        self,
        space_id: str,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取知识库节点列表。"""
        args = [
            "wiki",
            "+node-list",
            "--space-id",
            space_id,
            "--page-size",
            str(page_size),
        ]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def node_get(
        self,
        node_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取知识库节点信息。"""
        args = ["wiki", "+node-get", "--node-token", node_token]
        return self._e.run(args, timeout=timeout)

    def node_create(
        self,
        space_id: str,
        title: str,
        *,
        parent_node_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建知识库节点。"""
        args = ["wiki", "+node-create", "--space-id", space_id, "--title", title]
        if parent_node_token:
            args.extend(["--parent-node-token", parent_node_token])
        return self._e.run(args, timeout=timeout)

    def node_copy(
        self,
        node_token: str,
        target_space_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """复制知识库节点。"""
        args = [
            "wiki",
            "+node-copy",
            "--node-token",
            node_token,
            "--target-space-id",
            target_space_id,
        ]
        return self._e.run(args, timeout=timeout)

    def node_delete(self, node_token: str, *, timeout: int = 15) -> LarkResult:
        """删除知识库节点。"""
        args = ["wiki", "+node-delete", "--node-token", node_token]
        return self._e.run(args, timeout=timeout)

    def move(
        self,
        node_token: str,
        target_parent_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """移动知识库节点。"""
        args = [
            "wiki",
            "+move",
            "--node-token",
            node_token,
            "--target-parent-token",
            target_parent_token,
        ]
        return self._e.run(args, timeout=timeout)

    def member_add(
        self,
        space_id: str,
        member_id: str,
        member_type: str = "open_id",
        *,
        role: str = "viewer",
        timeout: int = 15,
    ) -> LarkResult:
        """添加知识库成员。"""
        args = [
            "wiki",
            "+member-add",
            "--space-id",
            space_id,
            "--member-id",
            member_id,
            "--member-type",
            member_type,
            "--role",
            role,
        ]
        return self._e.run(args, timeout=timeout)

    def member_remove(
        self,
        space_id: str,
        member_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """移除知识库成员。"""
        args = [
            "wiki",
            "+member-remove",
            "--space-id",
            space_id,
            "--member-id",
            member_id,
        ]
        return self._e.run(args, timeout=timeout)

    def member_list(
        self,
        space_id: str,
        *,
        page_size: int = 50,
        timeout: int = 15,
    ) -> LarkResult:
        """列出知识库成员。"""
        args = [
            "wiki",
            "+member-list",
            "--space-id",
            space_id,
            "--page-size",
            str(page_size),
        ]
        return self._e.run(args, timeout=timeout)
