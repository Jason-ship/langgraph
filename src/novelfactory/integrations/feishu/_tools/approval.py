from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _ApprovalTools:
    """审批 — 通过 lark-cli 命令或直接 API 实现。

    功能：
      + 查询审批实例
      + 审批/拒绝/转交任务
      + 抄送/撤回
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def list_tasks(
        self,
        *,
        page_size: int = 20,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """查询审批任务列表。"""
        result = self._e.run(
            [
                "service",
                "approval",
                "task",
                "list",
                "--page-size",
                str(page_size),
                "--format",
                "json",
            ],
            timeout=timeout,
        )
        return result

    def approve(
        self,
        task_id: str,
        *,
        comment: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """审批通过。"""
        args = ["service", "approval", "task", "approve", "--task-id", task_id]
        if comment:
            args.extend(["--comment", comment])
        return self._e.run(args, timeout=timeout)

    def reject(
        self,
        task_id: str,
        *,
        comment: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """审批拒绝。"""
        args = ["service", "approval", "task", "reject", "--task-id", task_id]
        if comment:
            args.extend(["--comment", comment])
        return self._e.run(args, timeout=timeout)

    def transfer(
        self,
        task_id: str,
        user_id: str,
        *,
        comment: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """转交审批任务。"""
        args = [
            "service",
            "approval",
            "task",
            "transfer",
            "--task-id",
            task_id,
            "--user-id",
            user_id,
        ]
        if comment:
            args.extend(["--comment", comment])
        return self._e.run(args, timeout=timeout)
