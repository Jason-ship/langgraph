"""飞书操作 — 向后兼容封装层。

v6.5: 所有实现委托给 feishu_toolkit.FeishuToolkit。
公共接口完全向后兼容，下游模块无需修改。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from novelfactory.config.settings import settings as _st_lazy
from novelfactory.integrations.feishu.feishu_toolkit import FeishuToolkit, LarkResult

logger = logging.getLogger(__name__)

# 单例工具箱（延迟初始化，双重检查锁定）
_toolkit: FeishuToolkit | None = None
_toolkit_lock = threading.Lock()


def _get_toolkit() -> FeishuToolkit:
    """获取全局 FeishuToolkit 实例（线程安全）。"""
    global _toolkit
    if _toolkit is None:
        with _toolkit_lock:
            if _toolkit is None:
                _toolkit = FeishuToolkit()
    return _toolkit


# ── 常量（向后兼容）────────────────────────────────────────────
from novelfactory.config.constants import (  # noqa: E402
    FEISHU_LARK_DOC_TIMEOUT,
    FEISHU_LARK_TIMEOUT,
    FEISHU_MAX_DOC_CHARS,
)

_LARK_TIMEOUT = FEISHU_LARK_TIMEOUT  # 唯一来源: config.constants
_LARK_DOC_TIMEOUT = FEISHU_LARK_DOC_TIMEOUT  # 唯一来源: config.constants
_MAX_DOC_CHARS = FEISHU_MAX_DOC_CHARS  # 唯一来源: config.constants


# ── 底层调用（向后兼容）─────────────────────────────────────


def _run_lark_cli(
    args: list[str],
    timeout: int = _LARK_TIMEOUT,
    always_json: bool = True,
) -> dict:
    """执行 lark-cli 命令 — 兼容旧接口返回 dict。"""
    tk = _get_toolkit()
    r = tk.run_raw(args, timeout=timeout)
    return _lark_result_to_legacy(r)


def _lark_result_to_legacy(r: LarkResult) -> dict:
    """LarkResult → 旧版 dict 格式。"""
    return {
        "success": r.success,
        "data": r.data,
        "error": r.error,
    }


# ── 消息发送（IM）───────────────────────────────────────────


def send_lark_message(
    receive_id: str, text: str, receive_id_type: str = "open_id", timeout: int = 15
) -> bool:
    """发送飞书文本消息。"""
    tk = _get_toolkit()
    if receive_id_type == "chat_id":
        r = tk.im.send_text(text, chat_id=receive_id, timeout=timeout)
    else:
        r = tk.im.send_text(text, user_id=receive_id, timeout=timeout)
    if not r.success:
        logger.warning("[feishu_api] send_lark_message failed: %s", r.error)
    return r.success


def send_lark_card(
    receive_id: str,
    card: dict,
    receive_id_type: str = "open_id",
    timeout: int = 15,
) -> bool:
    """发送飞书交互卡片消息。"""
    tk = _get_toolkit()
    if receive_id_type == "chat_id":
        r = tk.im.send_card(card, chat_id=receive_id, timeout=timeout)
    else:
        r = tk.im.send_card(card, user_id=receive_id, timeout=timeout)
    if not r.success:
        logger.warning("[feishu_api] send_lark_card failed: %s", r.error)
    return r.success


# ── 文件夹操作（Drive）────────────────────────────────────────


def _find_folder(parent_folder_token: str, name: str) -> str | None:
    """在父目录下查找子文件夹。"""
    tk = _get_toolkit()
    return tk.drive.find_folder(parent_folder_token, name)


def _create_folder(name: str, parent_token: str = "") -> str | None:
    """创建文件夹。"""
    tk = _get_toolkit()
    r = tk.drive.create_folder(name, parent_token=parent_token)
    if r.success:
        data = r.data or {}
        inner = data.get("data", {}) if isinstance(data, dict) else {}
        return inner.get("folder_token", "") or inner.get("token", "")
    logger.warning("[feishu_api] 创建文件夹失败 '%s': %s", name, r.error)
    return None


def ensure_folder(name: str, parent_token: str) -> str | None:
    """幂等创建文件夹。"""
    tk = _get_toolkit()
    return tk.drive.ensure_folder(name, parent_token)


# ── 项目目录树（幂等）─────────────────────────────────────


def ensure_project_folders_idempotent(project_name: str) -> dict:
    """幂等创建项目标准目录树。

    bot 模式下：直接在 bot 根目录创建项目文件夹（飞书自动授予
    CLI 用户 full_access 权限，用户可在云盘中看到）。
    用户模式下：从 FEISHU_ROOT_FOLDER 开始创建。
    """
    tk = _get_toolkit()
    tokens: dict[str, Any] = {
        "project": "",
        "setup": "",
        "chapters": "",
        "volume_folder_tokens": {},
    }

    # ── 尝试在用户指定根目录创建（仅 user 身份有效）──
    # v6.1: 统一从 settings 读取
    root_token = _st_lazy.FEISHU_ROOT_FOLDER or os.environ.get("FEISHU_ROOT_FOLDER", "")
    if root_token:
        project_token = tk.drive.ensure_folder(project_name, root_token)
        if project_token:
            tokens["project"] = project_token
            tokens["setup"] = tk.drive.ensure_folder("设定文档", project_token) or ""
            tokens["chapters"] = tk.drive.ensure_folder("正文", project_token) or ""
            logger.info(
                "[feishu_api] 目录树已确保(用户空间): project=%s, setup=%s, chapters=%s",
                project_token,
                tokens["setup"],
                tokens["chapters"],
            )
            return tokens
        logger.info("[feishu_api] 用户根目录不可用(bot模式)，降级到bot根目录创建")

    # ── bot 模式下直接从 bot 根目录创建 ──
    project_token = tk.drive.ensure_folder(project_name, "")
    if not project_token:
        logger.warning("[feishu_api] 无法创建项目文件夹 '%s'", project_name)
        return tokens
    tokens["project"] = project_token
    tokens["setup"] = tk.drive.ensure_folder("设定文档", project_token) or ""
    tokens["chapters"] = tk.drive.ensure_folder("正文", project_token) or ""

    logger.info(
        "[feishu_api] 目录树已确保(bot空间): project=%s, setup=%s, chapters=%s",
        project_token,
        tokens["setup"],
        tokens["chapters"],
    )
    return tokens


def ensure_volume_folder(
    volume_number: int, chapters_folder_token: str, tokens: dict
) -> str | None:
    """幂等创建卷文件夹，缓存结果。"""
    if not chapters_folder_token:
        return None
    vol_key = f"卷{volume_number}"
    existing = tokens.get("volume_folder_tokens", {}).get(volume_number)
    if existing:
        return existing
    tk = _get_toolkit()
    vol_token = tk.drive.ensure_folder(vol_key, chapters_folder_token)
    if vol_token:
        tokens.setdefault("volume_folder_tokens", {})[volume_number] = vol_token
    return vol_token


# ── 文档创建（Docs）───────────────────────────────────────


def create_feishu_doc(
    title: str, content_markdown: str, folder_token: str
) -> str | None:
    """在指定目录创建飞书在线文档。folder_token 为空时在 Bot 根目录创建。"""
    if not content_markdown or not content_markdown.strip():
        logger.warning("[feishu_api] 内容为空，跳过创建 '%s'", title)
        return None

    tk = _get_toolkit()
    r = tk.docs.create(title, content_markdown, folder_token=folder_token)
    if r.success:
        data = r.data or {}
        if isinstance(data, dict):
            # 优先从 data.data.document.url 提取
            inner = data.get("data") or {}
            if isinstance(inner, dict):
                doc = inner.get("document") or {}
                url = doc.get("url", "")
                if url:
                    logger.info(
                        "[feishu_api] 创建文档 '%s' → %s%s",
                        title,
                        url,
                        " (Bot 根目录)" if not folder_token else "",
                    )
                    return url
            # 回退：直接从 data.document.url 提取
            doc = data.get("document") or {}
            url = doc.get("url", "")
            if url:
                logger.info("[feishu_api] 创建文档(回退) '%s' → %s", title, url)
                return url
        if isinstance(data, str) and data.startswith("http"):
            logger.info("[feishu_api] 创建文档(str) '%s' → %s", title, data)
            return data
        logger.warning(
            "[feishu_api] URL提取失败 '%s': data type=%s keys=%s raw=%s",
            title,
            type(data).__name__,
            list(data.keys()) if isinstance(data, dict) else "N/A",
            str(data)[:200],
        )
    logger.warning("[feishu_api] 创建文档失败 '%s': %s", title, r.error)
    return None


def upload_chapter_as_doc(
    project_name: str,
    chapter_number: int,
    chapter_text: str,
    volume_number: int,
    folder_tokens: dict,
    quality_score: float | None = None,
) -> str | None:
    """将章节作为飞书文档上传到正确的卷文件夹。
    当目录树不可用时降级到 Bot 根目录创建。

    Args:
        quality_score: 可选 — 最终通过质量评分，不为 None 时追加到文档标题。
    """
    score_suffix = f" (评分{quality_score:.0f})" if quality_score is not None else ""
    title = f"《{project_name}》第{chapter_number:03d}章{score_suffix}"
    content = chapter_text[:_MAX_DOC_CHARS] if chapter_text else "（正文待续）"

    chapters_token = folder_tokens.get("chapters", "")
    if chapters_token:
        vol_token = ensure_volume_folder(volume_number, chapters_token, folder_tokens)
        if vol_token:
            url = create_feishu_doc(title, content, vol_token)
            if url:
                return url
            logger.info("[feishu_api] vol_token=%s 可能失效，重建目录树重试", vol_token)

    # v6.1: 统一从 settings 读取
    root_token = _st_lazy.FEISHU_ROOT_FOLDER or os.environ.get("FEISHU_ROOT_FOLDER", "")
    if root_token:
        new_tokens = ensure_project_folders_idempotent(project_name)
        if new_tokens.get("chapters"):
            vol_token = ensure_volume_folder(
                volume_number, new_tokens["chapters"], new_tokens
            )
            if vol_token:
                folder_tokens.clear()
                folder_tokens.update(new_tokens)
                url = create_feishu_doc(title, content, vol_token)
                if url:
                    return url

    logger.info("[feishu_api] 目录树不可用，降级到 Bot 根目录创建 '%s'", title)
    return create_feishu_doc(title, content, "")


# ── 设定文档上传 ──────────────────────────────────────────


def upload_setup_docs(
    project_name: str,
    world_setting: str,
    character_setting: str,
    story_outline: str,
    folder_tokens: dict,
) -> dict[str, str]:
    """上传设定文档到飞书（世界观、角色、大纲）。"""
    urls: dict[str, str] = {"world": "", "character": "", "outline": ""}
    setup_token = folder_tokens.get("setup", "")
    if not setup_token:
        logger.warning("[feishu_api] 设定文档目录不存在，跳过")
        return urls

    docs = [
        ("世界观设定", world_setting[:_MAX_DOC_CHARS] if world_setting else "(无)"),
        (
            "角色设定",
            character_setting[:_MAX_DOC_CHARS] if character_setting else "(无)",
        ),
        ("故事大纲", story_outline[:_MAX_DOC_CHARS] if story_outline else "(无)"),
    ]
    key_map = {"世界观设定": "world", "角色设定": "character", "故事大纲": "outline"}
    for title, content in docs:
        doc_title = f"{project_name} - {title}"
        content_md = f"# {doc_title}\n\n{content}"
        urls[key_map.get(title, "")] = (
            create_feishu_doc(doc_title, content_md, setup_token) or ""
        )

    return urls
