"""飞书云盘操作 —— 通过 feishu_api（→ FeishuToolkit → httpx → tools-proxy）委派实现。

此文件保留公共接口兼容性，所有实际操作转发到 feishu_api.py，
后者委托 FeishuToolkit 通过 httpx 调用 tools-proxy HTTP API。

v6.5: 底层引擎改为 httpx HTTP 代理（tools-proxy v2.0.0），不再依赖 lark-cli subprocess。
"""

from __future__ import annotations

import logging
import os

from novelfactory.integrations.feishu.feishu_api import (  # noqa: F401
    create_feishu_doc,
    ensure_folder,
    ensure_project_folders_idempotent,
    ensure_volume_folder,
    upload_chapter_as_doc,
    upload_setup_docs,
)

logger = logging.getLogger(__name__)

# ── 兼容旧接口 ──────────────────────────────────────────


def create_folder(name: str, parent_token: str = "") -> str | None:
    """创建文件夹（委派到 feishu_api）。"""
    from novelfactory.integrations.feishu.feishu_api import _create_folder

    return _create_folder(name, parent_token)


def ensure_project_folders(
    project_name: str,
    root_folder_token: str = "",
) -> dict[str, str]:
    """兼容旧接口的目录树创建（委派到幂等版本）。"""
    return ensure_project_folders_idempotent(project_name)


def upload_chapter_to_drive(
    chapter_number: int,
    chapter_text: str,
    volume_number: int,
    chapters_folder_token: str,
    volume_folder_tokens: dict[int, str],
) -> str | None:
    """上传完成的章节到飞书（委派到 feishu_api）。

    保留兼容签名，内部调用 upload_chapter_as_doc。
    volume_folder_tokens 由 feishu_api 内部管理缓存。
    """
    # 构建最小 folder_tokens，兼容旧调用方
    folder_tokens = {
        "chapters": chapters_folder_token,
        "volume_folder_tokens": volume_folder_tokens or {},
    }
    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st

    project_name = os.environ.get("NOVEL_PROJECT_NAME", _st.PROJECT_NAME or "江寻录")
    return upload_chapter_as_doc(
        project_name=project_name,
        chapter_number=chapter_number,
        chapter_text=chapter_text,
        volume_number=volume_number,
        folder_tokens=folder_tokens,
    )
