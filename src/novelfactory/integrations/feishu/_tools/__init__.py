"""飞书工具箱 — 领域工具集合。

re-export 所有基于 ``_LarkCLIEngine`` 的领域工具类，供 ``FeishuToolkit`` 门面装配。

包含 21 个域：
  im / docs / drive / calendar / contact / mail / sheets / base / task /
  minutes / vc / wiki / okr / apps / markdown / slides / whiteboard /
  event / note / approval / attendance
"""

from novelfactory.integrations.feishu._tools.approval import _ApprovalTools
from novelfactory.integrations.feishu._tools.apps import _AppsTools
from novelfactory.integrations.feishu._tools.attendance import _AttendanceTools
from novelfactory.integrations.feishu._tools.base import _BaseTools
from novelfactory.integrations.feishu._tools.calendar import _CalendarTools
from novelfactory.integrations.feishu._tools.contact import _ContactTools
from novelfactory.integrations.feishu._tools.docs import _DocsTools
from novelfactory.integrations.feishu._tools.drive import _DriveTools
from novelfactory.integrations.feishu._tools.event import _EventTools
from novelfactory.integrations.feishu._tools.im import _IMTools
from novelfactory.integrations.feishu._tools.mail import _MailTools
from novelfactory.integrations.feishu._tools.markdown import _MarkdownTools
from novelfactory.integrations.feishu._tools.minutes import _MinutesTools
from novelfactory.integrations.feishu._tools.note import _NoteTools
from novelfactory.integrations.feishu._tools.okr import _OKRTools
from novelfactory.integrations.feishu._tools.sheets import _SheetsTools
from novelfactory.integrations.feishu._tools.slides import _SlidesTools
from novelfactory.integrations.feishu._tools.task import _TaskTools
from novelfactory.integrations.feishu._tools.vc import _VCTools
from novelfactory.integrations.feishu._tools.whiteboard import _WhiteboardTools
from novelfactory.integrations.feishu._tools.wiki import _WikiTools

__all__ = [
    "_IMTools",
    "_DocsTools",
    "_DriveTools",
    "_CalendarTools",
    "_ContactTools",
    "_MailTools",
    "_SheetsTools",
    "_BaseTools",
    "_TaskTools",
    "_MinutesTools",
    "_VCTools",
    "_WikiTools",
    "_OKRTools",
    "_AppsTools",
    "_MarkdownTools",
    "_SlidesTools",
    "_WhiteboardTools",
    "_EventTools",
    "_NoteTools",
    "_ApprovalTools",
    "_AttendanceTools",
]
