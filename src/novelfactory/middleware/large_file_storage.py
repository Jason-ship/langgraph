"""LargeFileStorageMiddleware — 大章节草稿存入临时文件，避免 State 溢出。

当草稿超过 LARGE_FILE_THRESHOLD（默认 8000 字符）时，
自动写入 tmp/chapters/ 目录，State 中只存储文件路径和预览。
"""

from __future__ import annotations

import logging
import os

from novelfactory.middleware.base import Middleware

logger = logging.getLogger(__name__)

LARGE_FILE_THRESHOLD = 8000  # 字符数阈值
CHAPTERS_DIR = "tmp/chapters"


class LargeFileStorageMiddleware(Middleware):
    """草稿超阈值时写入临时文件，State 中只存路径。

    使用方式: 添加到 MiddlewareChain，在 writer/reviewer 节点执行后触发。
    """

    def __init__(self, threshold: int = LARGE_FILE_THRESHOLD):
        self.threshold = threshold

    def after_node(self, state: dict, result: dict, config: dict) -> dict | None:
        """检查节点输出，如果草稿超阈值则写入文件。"""
        chapter_draft = result.get("chapter_draft", "") or result.get(
            "refined_chapter", ""
        )
        if not chapter_draft or len(chapter_draft) <= self.threshold:
            return None

        ch_num = state.get("current_chapter", 1)
        os.makedirs(CHAPTERS_DIR, exist_ok=True)

        # 写入文件
        path = f"{CHAPTERS_DIR}/ch_{ch_num}_draft.txt"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(chapter_draft)
            logger.info(
                "[large_file] ch%d draft (%d chars) → %s",
                ch_num,
                len(chapter_draft),
                path,
            )
        except OSError as e:
            logger.warning("[large_file] 写入失败: %s", e)
            return None

        # 返回文件路径 + 预览
        preview = chapter_draft[:500]
        return {
            "chapter_draft": f"__FILE__:{path}",
            "chapter_preview": preview,
        }

    @staticmethod
    def read_chapter(state: dict) -> str:
        """从 state 中读取章节内容（兼容文件存储模式）。

        如果 chapter_draft 以 __FILE__: 开头，从文件读取；
        否则直接返回字符串内容。
        """
        draft = state.get("chapter_draft", "")
        if draft.startswith("__FILE__:"):
            path = draft[len("__FILE__:") :]
            try:
                with open(path, encoding="utf-8") as f:
                    return f.read()
            except (OSError, FileNotFoundError) as e:
                logger.warning("[large_file] 读取失败 %s: %s", path, e)
                return state.get("chapter_preview", "")
        return draft

    @staticmethod
    def cleanup(chapter_number: int | None = None):
        """清理临时文件。

        Args:
            chapter_number: 指定章节号时只清理该章节文件；None 时清理全部。
        """
        if chapter_number is not None:
            path = f"{CHAPTERS_DIR}/ch_{chapter_number}_draft.txt"
            if os.path.exists(path):
                os.remove(path)
                logger.info("[large_file] 已清理: %s", path)
        else:
            if os.path.exists(CHAPTERS_DIR):
                for fname in os.listdir(CHAPTERS_DIR):
                    fpath = os.path.join(CHAPTERS_DIR, fname)
                    if fname.startswith("ch_") and fname.endswith("_draft.txt"):
                        os.remove(fpath)
                logger.info("[large_file] 已清理全部临时文件")
