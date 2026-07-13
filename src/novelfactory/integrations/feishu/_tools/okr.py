from __future__ import annotations

import json

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _OKRTools:
    """OKR — 对应 lark-cli okr 域。

    命令列表（12 个）：
      +list-cycles, +cycle-detail, +list-progress, +get-progress-record,
      +create-progress-record, +update-progress-record, +delete-progress-record,
      +upload-image, +batch-create, +reorder, +weight, +indicator-update
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def list_cycles(self, *, timeout: int = 15) -> LarkResult:
        """列出 OKR 周期。"""
        return self._e.run(["okr", "+list-cycles"], timeout=timeout)

    def cycle_detail(self, cycle_id: str, *, timeout: int = 15) -> LarkResult:
        """获取 OKR 周期详情。"""
        args = ["okr", "+cycle-detail", "--cycle-id", cycle_id]
        return self._e.run(args, timeout=timeout)

    def batch_create(
        self,
        cycle_id: str,
        objectives: list[dict],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """批量创建 OKR。"""
        args = [
            "okr",
            "+batch-create",
            "--cycle-id",
            cycle_id,
            "--objectives",
            json.dumps(objectives, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)

    def reorder(
        self,
        cycle_id: str,
        objective_ids: list[str],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """重排 OKR 顺序。"""
        args = [
            "okr",
            "+reorder",
            "--cycle-id",
            cycle_id,
            "--objective-ids",
            ",".join(objective_ids),
        ]
        return self._e.run(args, timeout=timeout)

    def weight(
        self,
        cycle_id: str,
        objectives: list[dict],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """设置 OKR 权重。"""
        args = [
            "okr",
            "+weight",
            "--cycle-id",
            cycle_id,
            "--objectives",
            json.dumps(objectives, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)

    def list_progress(
        self,
        objective_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """列出 OKR 进展。"""
        args = ["okr", "+list-progress", "--objective-id", objective_id]
        return self._e.run(args, timeout=timeout)

    def get_progress_record(
        self,
        progress_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取进展记录。"""
        args = ["okr", "+get-progress-record", "--progress-id", progress_id]
        return self._e.run(args, timeout=timeout)

    def create_progress_record(
        self,
        objective_id: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建进展记录。"""
        args = [
            "okr",
            "+create-progress-record",
            "--objective-id",
            objective_id,
            "--content",
            content,
        ]
        return self._e.run(args, timeout=timeout)

    def update_progress_record(
        self,
        progress_id: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """更新进展记录。"""
        args = [
            "okr",
            "+update-progress-record",
            "--progress-id",
            progress_id,
            "--content",
            content,
        ]
        return self._e.run(args, timeout=timeout)

    def delete_progress_record(
        self,
        progress_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """删除进展记录。"""
        args = ["okr", "+delete-progress-record", "--progress-id", progress_id]
        return self._e.run(args, timeout=timeout)

    def upload_image(
        self,
        file_path: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """上传 OKR 图片。"""
        args = ["okr", "+upload-image", "--file", file_path]
        return self._e.run(args, timeout=timeout)

    def indicator_update(
        self,
        objective_id: str,
        indicators: dict,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """更新 OKR 指标。"""
        args = [
            "okr",
            "+indicator-update",
            "--objective-id",
            objective_id,
            "--indicators",
            json.dumps(indicators, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)
