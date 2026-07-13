from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _TaskTools:
    """任务 — 对应 lark-cli task 域。

    命令列表（18 个）：
      +create, +update, +set-ancestor, +comment, +complete, +reopen,
      +assign, +followers, +reminder, +get-my-tasks, +get-related-tasks,
      +search, +upload-attachment, +create-tasklist, +search-tasklist,
      +add-to-tasklist, +members-tasklist
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def create(
        self,
        summary: str,
        *,
        description: str = "",
        assignee: str = "",
        due: str = "",
        tasklist_id: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建任务。"""
        args = ["task", "+create", "--summary", summary]
        if description:
            args.extend(["--description", description])
        if assignee:
            args.extend(["--assignee", assignee])
        if due:
            args.extend(["--due", due])
        if tasklist_id:
            args.extend(["--tasklist-id", tasklist_id])
        return self._e.run(args, timeout=timeout)

    def update(
        self,
        task_id: str,
        *,
        summary: str = "",
        description: str = "",
        due: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """更新任务。"""
        args = ["task", "+update", "--task-id", task_id]
        if summary:
            args.extend(["--summary", summary])
        if description:
            args.extend(["--description", description])
        if due:
            args.extend(["--due", due])
        return self._e.run(args, timeout=timeout)

    def complete(self, task_id: str, *, timeout: int = 15) -> LarkResult:
        """完成任务。"""
        args = ["task", "+complete", "--task-id", task_id]
        return self._e.run(args, timeout=timeout)

    def reopen(self, task_id: str, *, timeout: int = 15) -> LarkResult:
        """重新打开任务。"""
        args = ["task", "+reopen", "--task-id", task_id]
        return self._e.run(args, timeout=timeout)

    def assign(
        self,
        task_id: str,
        assignee: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """分配任务。"""
        args = ["task", "+assign", "--task-id", task_id, "--assignee", assignee]
        return self._e.run(args, timeout=timeout)

    def get_my_tasks(
        self,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取我的任务。"""
        args = ["task", "+get-my-tasks", "--page-size", str(page_size)]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def search(
        self,
        query: str,
        *,
        page_size: int = 20,
        timeout: int = 15,
    ) -> LarkResult:
        """搜索任务。"""
        args = ["task", "+search", "--query", query, "--page-size", str(page_size)]
        return self._e.run(args, timeout=timeout)

    def create_tasklist(
        self,
        name: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建任务清单。"""
        args = ["task", "+create-tasklist", "--name", name]
        return self._e.run(args, timeout=timeout)

    def search_tasklist(
        self,
        query: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """搜索任务清单。"""
        args = ["task", "+search-tasklist", "--query", query]
        return self._e.run(args, timeout=timeout)

    def add_to_tasklist(
        self,
        task_id: str,
        tasklist_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """将任务加入清单。"""
        args = [
            "task",
            "+add-to-tasklist",
            "--task-id",
            task_id,
            "--tasklist-id",
            tasklist_id,
        ]
        return self._e.run(args, timeout=timeout)

    def comment(
        self,
        task_id: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """添加任务评论。"""
        args = ["task", "+comment", "--task-id", task_id, "--content", content]
        return self._e.run(args, timeout=timeout)

    def set_reminder(
        self,
        task_id: str,
        remind_time: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """设置任务提醒。"""
        args = ["task", "+reminder", "--task-id", task_id, "--remind-time", remind_time]
        return self._e.run(args, timeout=timeout)

    def set_followers(
        self,
        task_id: str,
        follower_ids: list[str],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """设置任务关注者。"""
        args = [
            "task",
            "+followers",
            "--task-id",
            task_id,
            "--followers",
            ",".join(follower_ids),
        ]
        return self._e.run(args, timeout=timeout)

    def get_related_tasks(
        self,
        task_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取关联任务。"""
        args = ["task", "+get-related-tasks", "--task-id", task_id]
        return self._e.run(args, timeout=timeout)
