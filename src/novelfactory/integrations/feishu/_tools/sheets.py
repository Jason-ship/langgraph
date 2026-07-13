from __future__ import annotations

import json

from novelfactory.integrations.feishu._core import LarkResult, _LarkCLIEngine


class _SheetsTools:
    """电子表格 — 对应 lark-cli sheets 域。

    命令列表（73 个）：
      工作簿:      +workbook-create, +workbook-info, +workbook-export, +workbook-import
      工作表:      +sheet-create, +sheet-delete, +sheet-rename, +sheet-move,
                   +sheet-copy, +sheet-hide, +sheet-unhide, +sheet-set-tab-color,
                   +sheet-show-gridline, +sheet-hide-gridline, +sheet-info
      行列操作:    +dim-insert, +dim-delete, +dim-hide, +dim-unhide,
                   +dim-freeze, +dim-group, +dim-ungroup, +dim-move
      单元格:      +cells-get, +cells-set, +cells-set-style, +cells-set-image,
                   +cells-clear, +cells-merge, +cells-unmerge, +cells-search,
                   +cells-replace, +rows-resize, +cols-resize
      范围操作:    +range-move, +range-copy, +range-fill, +range-sort
      导入/导出:   +csv-get, +csv-put, +table-get, +table-put
      下拉:        +dropdown-get, +dropdown-set, +dropdown-update, +dropdown-delete
      图表:        +chart-create, +chart-update, +chart-delete, +chart-list
      透视:        +pivot-create, +pivot-update, +pivot-delete, +pivot-list
      条件格式:    +cond-format-create, +cond-format-update, +cond-format-delete, +cond-format-list
      筛选:        +filter-create, +filter-update, +filter-delete, +filter-list
      筛选视图:    +filter-view-create, +filter-view-update, +filter-view-delete, +filter-view-list
      浮动图片:    +float-image-create, +float-image-update, +float-image-delete, +float-image-list
      微线图:      +sparkline-create, +sparkline-update, +sparkline-delete, +sparkline-list
      批量:        +batch-update, +cells-batch-set-style, +cells-batch-clear
    """

    def __init__(self, engine: type[_LarkCLIEngine]):
        self._e = engine

    # ── 工作簿 ─────────────────────────────────────────────────────

    def workbook_create(
        self,
        title: str,
        *,
        folder_token: str = "",
        timeout: int = 15,
    ) -> LarkResult:
        """创建电子表格。"""
        args = ["sheets", "+workbook-create", "--title", title]
        if folder_token:
            args.extend(["--folder-token", folder_token])
        return self._e.run(args, timeout=timeout)

    def workbook_info(
        self,
        spreadsheet_token: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """获取工作簿信息。"""
        args = ["sheets", "+workbook-info", "--spreadsheet-token", spreadsheet_token]
        return self._e.run(args, timeout=timeout)

    def workbook_export(
        self,
        spreadsheet_token: str,
        format: str = "csv",
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """导出工作簿。"""
        args = [
            "sheets",
            "+workbook-export",
            "--spreadsheet-token",
            spreadsheet_token,
            "--format",
            format,
        ]
        return self._e.run(args, timeout=timeout)

    def workbook_import(
        self,
        file_path: str,
        *,
        timeout: int = 30,
    ) -> LarkResult:
        """导入文件为电子表格。"""
        args = ["sheets", "+workbook-import", "--file", file_path]
        return self._e.run(args, timeout=timeout)

    # ── 工作表 ─────────────────────────────────────────────────────

    def sheet_create(
        self,
        spreadsheet_token: str,
        title: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """创建工作表。"""
        args = [
            "sheets",
            "+sheet-create",
            "--spreadsheet-token",
            spreadsheet_token,
            "--title",
            title,
        ]
        return self._e.run(args, timeout=timeout)

    def sheet_delete(
        self,
        spreadsheet_token: str,
        sheet_id: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """删除工作表。"""
        args = [
            "sheets",
            "+sheet-delete",
            "--spreadsheet-token",
            spreadsheet_token,
            "--sheet-id",
            sheet_id,
        ]
        return self._e.run(args, timeout=timeout)

    def sheet_rename(
        self,
        spreadsheet_token: str,
        sheet_id: str,
        title: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """重命名工作表。"""
        args = [
            "sheets",
            "+sheet-rename",
            "--spreadsheet-token",
            spreadsheet_token,
            "--sheet-id",
            sheet_id,
            "--title",
            title,
        ]
        return self._e.run(args, timeout=timeout)

    def cells_get(
        self,
        spreadsheet_token: str,
        range_str: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """读取单元格数据。"""
        args = [
            "sheets",
            "+cells-get",
            "--spreadsheet-token",
            spreadsheet_token,
            "--range",
            range_str,
        ]
        return self._e.run(args, timeout=timeout)

    def cells_set(
        self,
        spreadsheet_token: str,
        range_str: str,
        values: list[list[str]],
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """写入单元格数据。"""
        args = [
            "sheets",
            "+cells-set",
            "--spreadsheet-token",
            spreadsheet_token,
            "--range",
            range_str,
            "--values",
            json.dumps(values, ensure_ascii=False),
        ]
        return self._e.run(args, timeout=timeout)

    def cells_search(
        self,
        spreadsheet_token: str,
        range_str: str,
        value: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """搜索单元格。"""
        args = [
            "sheets",
            "+cells-search",
            "--spreadsheet-token",
            spreadsheet_token,
            "--range",
            range_str,
            "--value",
            value,
        ]
        return self._e.run(args, timeout=timeout)

    def cells_replace(
        self,
        spreadsheet_token: str,
        range_str: str,
        find: str,
        replace: str,
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """替换单元格内容。"""
        args = [
            "sheets",
            "+cells-replace",
            "--spreadsheet-token",
            spreadsheet_token,
            "--range",
            range_str,
            "--find",
            find,
            "--replace",
            replace,
        ]
        return self._e.run(args, timeout=timeout)

    def _common_sheet_flag(
        self,
        cmd: str,
        spreadsheet_token: str,
        sheet_id: str = "",
        range_str: str = "",
        *,
        timeout: int = 15,
    ) -> LarkResult:
        """通用带 sheet 标志的命令。"""
        args = ["sheets", cmd, "--spreadsheet-token", spreadsheet_token]
        if sheet_id:
            args.extend(["--sheet-id", sheet_id])
        if range_str:
            args.extend(["--range", range_str])
        return self._e.run(args, timeout=timeout)
