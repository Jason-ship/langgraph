from __future__ import annotations

import json

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _BaseTools:
    """多维表格 — 对应 lark-cli base 域。

    命令列表（90 个）：
      基础:       +resolve-url, +resolve-title, +block-list, +block-create,
                  +block-move, +block-rename, +block-delete, +get, +copy, +create
      数据表:     +table-list, +table-get, +table-create, +table-update, +table-delete
      字段:       +field-list, +field-get, +field-create, +field-update, +field-delete
                  +field-search-options
      视图:       +view-list, +view-get, +view-create, +view-delete, +view-rename,
                  +view-get-filter, +view-set-filter, +view-get-visible-fields,
                  +view-set-visible-fields, +view-get-group, +view-set-group,
                  +view-get-sort, +view-set-sort, +view-get-timebar, +view-set-timebar,
                  +view-get-card, +view-set-card
      记录:       +record-list, +record-search, +record-get, +record-upsert,
                  +record-batch-create, +record-batch-update,
                  +record-share-link-create, +record-upload-attachment,
                  +record-download-attachment, +record-remove-attachment,
                  +record-delete, +record-history-list
      角色权限:   +role-create, +role-delete, +role-update, +role-list, +role-get
                  +advperm-enable, +advperm-disable
      工作流:     +workflow-list, +workflow-get, +workflow-create,
                  +workflow-update, +workflow-enable, +workflow-disable
      仪表盘:     +dashboard-list, +dashboard-get, +dashboard-create,
                  +dashboard-update, +dashboard-delete, +dashboard-arrange,
                  +dashboard-block-list, +dashboard-block-get,
                  +dashboard-block-get-data, +dashboard-block-create,
                  +dashboard-block-update, +dashboard-block-delete
      数据查询:   +data-query
      表单:       +form-create, +form-delete, +form-list, +form-update, +form-get,
                  +form-detail, +form-questions-create, +form-questions-delete,
                  +form-questions-update, +form-questions-list, +form-submit
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    def resolve_url(
        self,
        url: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """通过 URL 解析多维表格信息。"""
        args = ["base", "+resolve-url", "--url", url]
        return self._e.run(args, timeout=timeout)

    def resolve_title(
        self,
        title: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """通过标题搜索多维表格。"""
        args = ["base", "+resolve-title", "--title", title]
        return self._e.run(args, timeout=timeout)

    def table_list(
        self,
        base_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取数据表列表。"""
        args = ["base", "+table-list", "--base-token", base_token]
        return self._e.run(args, timeout=timeout)

    def table_get(
        self,
        base_token: str,
        table_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取数据表信息。"""
        args = [
            "base",
            "+table-get",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
        ]
        return self._e.run(args, timeout=timeout)

    def record_list(
        self,
        base_token: str,
        table_id: str,
        *,
        page_size: int = 20,
        page_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """获取记录列表。"""
        args = [
            "base",
            "+record-list",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--page-size",
            str(page_size),
        ]
        if page_token:
            args.extend(["--page-token", page_token])
        return self._e.run(args, timeout=timeout)

    def record_get(
        self,
        base_token: str,
        table_id: str,
        record_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取单条记录。"""
        args = [
            "base",
            "+record-get",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--record-id",
            record_id,
        ]
        return self._e.run(args, timeout=timeout)

    def record_upsert(
        self,
        base_token: str,
        table_id: str,
        fields: dict,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建/更新记录。"""
        args = [
            "base",
            "+record-upsert",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--fields",
            json.dumps(fields, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)

    def record_batch_create(
        self,
        base_token: str,
        table_id: str,
        records: list[dict],
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """批量创建记录。"""
        args = [
            "base",
            "+record-batch-create",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--records",
            json.dumps(records, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)

    def record_search(
        self,
        base_token: str,
        table_id: str,
        query: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """搜索记录。"""
        args = [
            "base",
            "+record-search",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--query",
            query,
        ]
        return self._e.run(args, timeout=timeout)

    def record_delete(
        self,
        base_token: str,
        table_id: str,
        record_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """删除记录。"""
        args = [
            "base",
            "+record-delete",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--record-id",
            record_id,
        ]
        return self._e.run(args, timeout=timeout)

    def field_list(
        self,
        base_token: str,
        table_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取字段列表。"""
        args = [
            "base",
            "+field-list",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
        ]
        return self._e.run(args, timeout=timeout)

    def field_create(
        self,
        base_token: str,
        table_id: str,
        field_name: str,
        field_type: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建字段。"""
        args = [
            "base",
            "+field-create",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--field-name",
            field_name,
            "--field-type",
            field_type,
        ]
        return self._e.run(args, timeout=timeout)

    def data_query(
        self,
        base_token: str,
        query: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """执行数据查询（SQL-like）。"""
        args = ["base", "+data-query", "--base-token", base_token, "--query", query]
        return self._e.run(args, timeout=timeout)

    def form_list(
        self,
        base_token: str,
        table_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取表单列表。"""
        args = [
            "base",
            "+form-list",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
        ]
        return self._e.run(args, timeout=timeout)

    def form_submit(
        self,
        base_token: str,
        form_id: str,
        fields: dict,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """提交表单。"""
        args = [
            "base",
            "+form-submit",
            "--base-token",
            base_token,
            "--form-id",
            form_id,
            "--fields",
            json.dumps(fields, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)

    def dashboard_list(
        self,
        base_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取仪表盘列表。"""
        args = ["base", "+dashboard-list", "--base-token", base_token]
        return self._e.run(args, timeout=timeout)

    def role_list(
        self,
        base_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取角色列表。"""
        args = ["base", "+role-list", "--base-token", base_token]
        return self._e.run(args, timeout=timeout)

    def workflow_list(
        self,
        base_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取工作流列表。"""
        args = ["base", "+workflow-list", "--base-token", base_token]
        return self._e.run(args, timeout=timeout)
