"""飞书标准工具箱 — 基于 lark-cli 200+ 命令的完整 Python 封装。

架构（v6.6 按域拆分为子包）：
  FeishuToolkit (根门面)
  ├── _core              底层 CLI 调用引擎 + 结果类型 + 常量
  ├── _tools/            领域工具集合（21 个域）
  │   ├── im             即时通讯 (20 命令)
  │   ├── docs           文档 (11 命令)
  │   ├── drive          云盘 (26 命令)
  │   ├── calendar       日历 (7 命令)
  │   ├── contact        通讯录 (2 命令)
  │   ├── mail           邮箱 (19 命令)
  │   ├── sheets         电子表格 (73 命令)
  │   ├── base           多维表格 (90 命令)
  │   ├── task           任务 (18 命令)
  │   ├── minutes        妙记 (8 命令)
  │   ├── vc             视频会议 (7 命令)
  │   ├── wiki           知识库 (12 命令)
  │   ├── okr            OKR (12 命令)
  │   ├── apps           应用管理 (25 命令)
  │   ├── markdown       Markdown 文件 (5 命令)
  │   ├── slides         幻灯片 (4 命令)
  │   ├── whiteboard     画板 (3 命令)
  │   ├── event          事件订阅 (1 命令)
  │   ├── note           笔记 (2 命令)
  │   ├── approval       审批
  │   └── attendance     考勤

本文件仅保留 ``FeishuToolkit`` 门面类，底层实现见 ``_core`` 与 ``_tools``。
``LarkResult`` / ``LarkListResult`` 自本模块 re-export，兼容历史导入路径
（如 ``feishu_api.py`` 从本模块导入 ``LarkResult``）。
"""

from __future__ import annotations

import novelfactory.integrations.feishu._core as _core
from novelfactory.integrations.feishu._core import (
    _LARK_TIMEOUT,
    LarkListResult,
    LarkResult,
    _LarkCLIEngine,
)
from novelfactory.integrations.feishu._tools import (
    _ApprovalTools,
    _AppsTools,
    _AttendanceTools,
    _BaseTools,
    _CalendarTools,
    _ContactTools,
    _DocsTools,
    _DriveTools,
    _EventTools,
    _IMTools,
    _MailTools,
    _MarkdownTools,
    _MinutesTools,
    _NoteTools,
    _OKRTools,
    _SheetsTools,
    _SlidesTools,
    _TaskTools,
    _VCTools,
    _WhiteboardTools,
    _WikiTools,
)

__all__ = [
    "FeishuToolkit",
    "LarkResult",
    "LarkListResult",
]


class FeishuToolkit:
    """飞书标准工具箱 — 覆盖 lark-cli 全部 200+ 命令的 Python 封装。

    用法：
        tk = FeishuToolkit()

        # 发送飞书消息
        tk.im.send_text("你好", chat_id="oc_xxx")
        tk.im.send_card(card_data, user_id="ou_xxx")

        # 操作云盘文件夹
        tk.drive.ensure_folder("新文件夹", "token_xxx")

        # 创建文档
        tk.docs.create("标题", "# 正文内容", folder_token="token_xxx")

        # 日历操作
        tk.calendar.agenda(date="2026-06-25")

        # 通讯录
        contact = tk.contact.search_user("张三")

        # 多维表格
        records = tk.base.record_list("base_token", "table_id")

        # 任务管理
        task = tk.task.create("完成报告", assignee="ou_xxx", due="+3d")

        # 妙记
        summary = tk.minutes.summary("minute_token")

        # 视频会议
        tk.vc.meeting_join("meeting_id")

        # 知识库
        tk.wiki.space_create("新知识空间")
    """

    def __init__(self, lark_proxy_url: str = ""):
        if lark_proxy_url:
            _core._LARK_PROXY_URL = lark_proxy_url

        self._engine = _LarkCLIEngine

        # 领域工具
        self.im = _IMTools(self._engine)
        self.docs = _DocsTools(self._engine)
        self.drive = _DriveTools(self._engine)
        self.calendar = _CalendarTools(self._engine)
        self.contact = _ContactTools(self._engine)
        self.mail = _MailTools(self._engine)
        self.sheets = _SheetsTools(self._engine)
        self.base = _BaseTools(self._engine)
        self.task = _TaskTools(self._engine)
        self.minutes = _MinutesTools(self._engine)
        self.vc = _VCTools(self._engine)
        self.wiki = _WikiTools(self._engine)
        self.okr = _OKRTools(self._engine)
        self.apps = _AppsTools(self._engine)
        self.markdown = _MarkdownTools(self._engine)
        self.slides = _SlidesTools(self._engine)
        self.whiteboard = _WhiteboardTools(self._engine)
        self.event = _EventTools(self._engine)
        self.note = _NoteTools(self._engine)
        self.approval = _ApprovalTools(self._engine)
        self.attendance = _AttendanceTools(self._engine)

    # ── 底层接口暴露 ────────────────────────────────────────────────

    @property
    def engine(self) -> type[_LarkCLIEngine]:
        """底层 CLI 调用引擎。"""
        return self._engine

    def run_raw(
        self,
        args: list[str],
        *,
        timeout: int = _LARK_TIMEOUT,
    ) -> LarkResult:
        """直接执行任意 lark-cli 命令。"""
        return self._engine.run(args, timeout=timeout)

    # ── 认证管理 ────────────────────────────────────────────────────

    def auth_login(
        self,
        *,
        domain: str = "drive,docs,im",
        recommend: bool = True,
        timeout: int = 15,
    ) -> LarkResult:
        """登录飞书。"""
        args = ["auth", "login", "--json"]
        if domain:
            args.extend(["--domain", domain])
        if recommend:
            args.append("--recommend")
        return self._engine.run(args, timeout=timeout, format_json=False)

    def auth_status(self, *, timeout: int = 15) -> LarkResult:
        """查看认证状态。"""
        return self._engine.run(["auth", "status"], timeout=timeout)

    def auth_logout(self, *, timeout: int = 15) -> LarkResult:
        """退出登录。"""
        return self._engine.run(["auth", "logout"], timeout=timeout)
