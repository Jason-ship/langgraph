from __future__ import annotations

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _DriveTools:
    """云盘 — 对应 lark-cli drive 域。

    命令列表（26 个）：
      +upload, +create-folder, +create-shortcut, +download, +preview,
      +cover, +add-comment, +export, +export-download, +import,
      +version-history, +version-get, +version-revert, +version-delete,
      +move, +delete, +status, +push, +pull, +sync, +task-result,
      +apply-permission, +member-add, +secure-label-list, +secure-label-update,
      +search, +inspect
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def upload(
        self,
        file_path: str,
        *,
        folder_token: str = "",
        file_name: str = "",
        timeout: int = 60,
    ) -> LarkResult:
        """上传文件到云盘。"""
        args = ["drive", "+upload", "--file", file_path]
        if folder_token:
            args.extend(["--folder-token", folder_token])
        if file_name:
            args.extend(["--file-name", file_name])
        return self._e.run(args, timeout=timeout)

    def create_folder(
        self,
        name: str,
        *,
        parent_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建文件夹。"""
        args = ["drive", "+create-folder", "--name", name, "--as", "bot"]
        if parent_token:
            args.extend(["--folder-token", parent_token])
        return self._e.run(args, timeout=timeout)

    def create_shortcut(
        self,
        name: str,
        target_token: str,
        *,
        parent_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建快捷方式。"""
        args = [
            "drive",
            "+create-shortcut",
            "--name",
            name,
            "--target-token",
            target_token,
        ]
        if parent_token:
            args.extend(["--folder-token", parent_token])
        return self._e.run(args, timeout=timeout)

    def download(
        self,
        file_token: str,
        *,
        output_path: str = "",
        timeout: int = 60,
    ) -> LarkResult:
        """下载云盘文件。"""
        args = ["drive", "+download", "--file-token", file_token]
        if output_path:
            args.extend(["--output", output_path])
        return self._e.run(args, timeout=timeout)

    def preview(
        self,
        file_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """预览云盘文件。"""
        args = ["drive", "+preview", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def cover(self, file_token: str, *, timeout: int = 15) -> LarkResult:
        """获取文件封面。"""
        args = ["drive", "+cover", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def add_comment(
        self,
        file_token: str,
        content: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """添加评论。"""
        args = [
            "drive",
            "+add-comment",
            "--file-token",
            file_token,
            "--content",
            content,
        ]
        return self._e.run(args, timeout=timeout)

    def export(
        self,
        file_token: str,
        export_format: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """导出文件。"""
        args = [
            "drive",
            "+export",
            "--file-token",
            file_token,
            "--format",
            export_format,
        ]
        return self._e.run(args, timeout=timeout)

    def export_download(
        self,
        task_id: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """下载导出结果。"""
        args = ["drive", "+export-download", "--task-id", task_id]
        return self._e.run(args, timeout=timeout)

    def import_file(
        self,
        file_path: str,
        file_type: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """导入文件到云盘。"""
        args = ["drive", "+import", "--file", file_path, "--file-type", file_type]
        return self._e.run(args, timeout=timeout)

    def version_history(
        self,
        file_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取版本历史。"""
        args = ["drive", "+version-history", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def version_get(
        self,
        file_token: str,
        version_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取版本详情。"""
        args = [
            "drive",
            "+version-get",
            "--file-token",
            file_token,
            "--version-id",
            version_id,
        ]
        return self._e.run(args, timeout=timeout)

    def version_revert(
        self,
        file_token: str,
        version_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """还原到指定版本。"""
        args = [
            "drive",
            "+version-revert",
            "--file-token",
            file_token,
            "--version-id",
            version_id,
        ]
        return self._e.run(args, timeout=timeout)

    def version_delete(
        self,
        file_token: str,
        version_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """删除版本。"""
        args = [
            "drive",
            "+version-delete",
            "--file-token",
            file_token,
            "--version-id",
            version_id,
        ]
        return self._e.run(args, timeout=timeout)

    def move(
        self,
        file_token: str,
        target_folder_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """移动文件。"""
        args = [
            "drive",
            "+move",
            "--file-token",
            file_token,
            "--folder-token",
            target_folder_token,
        ]
        return self._e.run(args, timeout=timeout)

    def delete(self, file_token: str, *, timeout: int = 15) -> LarkResult:
        """删除文件到回收站。"""
        args = ["drive", "+delete", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def status(self, file_token: str, *, timeout: int = 15) -> LarkResult:
        """获取文件状态。"""
        args = ["drive", "+status", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def push(
        self,
        file_path: str,
        *,
        folder_token: str = "",
        timeout: int = 60,
    ) -> LarkResult:
        """推送本地文件到云盘（同步式上传）。"""
        args = ["drive", "+push", "--file", file_path]
        if folder_token:
            args.extend(["--folder-token", folder_token])
        return self._e.run(args, timeout=timeout)

    def pull(
        self,
        file_token: str,
        *,
        output_path: str = "",
        timeout: int = 60,
    ) -> LarkResult:
        """从云盘拉取文件到本地（同步式下载）。"""
        args = ["drive", "+pull", "--file-token", file_token]
        if output_path:
            args.extend(["--output", output_path])
        return self._e.run(args, timeout=timeout)

    def sync(
        self,
        local_dir: str,
        remote_folder_token: str,
        *,
        timeout: int = 60,
    ) -> LarkResult:
        """同步本地目录到云盘。"""
        args = [
            "drive",
            "+sync",
            "--local-dir",
            local_dir,
            "--folder-token",
            remote_folder_token,
        ]
        return self._e.run(args, timeout=timeout)

    def task_result(self, task_id: str, *, timeout: int = 15) -> LarkResult:
        """查询异步任务结果。"""
        args = ["drive", "+task-result", "--task-id", task_id]
        return self._e.run(args, timeout=timeout)

    def apply_permission(
        self,
        file_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """申请文件权限。"""
        args = ["drive", "+apply-permission", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def member_add(
        self,
        file_token: str,
        member_id: str,
        member_type: str = "open_id",
        *,
        perm: str = "view",
        timeout: int = 15,
    ) -> LarkResult:
        """添加文件协作者。"""
        args = [
            "drive",
            "+member-add",
            "--file-token",
            file_token,
            "--member-id",
            member_id,
            "--member-type",
            member_type,
            "--perm",
            perm,
        ]
        return self._e.run(args, timeout=timeout)

    def secure_label_list(
        self,
        file_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取文件密级标签。"""
        args = ["drive", "+secure-label-list", "--file-token", file_token]
        return self._e.run(args, timeout=timeout)

    def secure_label_update(
        self,
        file_token: str,
        label_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """更新文件密级标签。"""
        args = [
            "drive",
            "+secure-label-update",
            "--file-token",
            file_token,
            "--label-id",
            label_id,
        ]
        return self._e.run(args, timeout=timeout)

    def search(
        self,
        query: str,
        *,
        page_size: int = 50,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """搜索云盘文件。"""
        args = ["drive", "+search", "--query", query, "--page-size", str(page_size)]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def inspect(self, url: str, *, timeout: int = 15) -> LarkResult:
        """解析飞书资源 URL 为 token。"""
        args = ["drive", "+inspect", "--url", url]
        return self._e.run(args, timeout=timeout)

    # ── 旧兼容接口 ────────────────────────────────────────────────

    def find_folder(
        self,
        parent_token: str,
        name: str,
        *,
        timeout: int = 15,
    ) -> str | None:
        """在父目录下查找子文件夹（返回 folder_token）。"""
        r = self._e.run(
            [
                "drive",
                "files",
                "list",
                "--folder-token",
                parent_token,
                "--page-size",
                "200",
                "--page-all",
                "--as",
                "user",
            ],
            timeout=timeout,
        )
        if not r.success:
            return None
        data = r.data or {}
        inner = data.get("data", {}) if isinstance(data, dict) else {}
        files = inner.get("files", [])
        if isinstance(files, list):
            for f in files:
                if (
                    isinstance(f, dict)
                    and f.get("name") == name
                    and f.get("type") == "folder"
                ):
                    return f.get("token", "")
        return None

    def ensure_folder(
        self,
        name: str,
        parent_token: str,
        *,
        timeout: int = 15,
    ) -> str | None:
        """幂等创建文件夹：搜索→创建→再搜索。"""
        existing = self.find_folder(parent_token, name, timeout=timeout)
        if existing:
            return existing
        r = self.create_folder(name, parent_token=parent_token, timeout=timeout)
        if r.success:
            data = r.data or {}
            inner = data.get("data", {}) if isinstance(data, dict) else {}
            token = inner.get("folder_token", "") or inner.get("token", "")
            if token:
                return token
        return self.find_folder(parent_token, name, timeout=timeout)
