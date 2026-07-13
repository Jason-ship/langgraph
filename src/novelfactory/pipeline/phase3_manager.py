"""Phase 3 Manager — 卷完成检测 + 质量衰减检测 + 写入上下文管理器。

被 ScaleManager 和 phase_checks.py 依赖，提供：
  - volume 子管理器：检查卷完成、构建卷间过渡上下文
  - quality 子管理器：检测质量衰减趋势
  - build_writer_context(): 构建给 writer agent 的 Phase 3 上下文

重构要点（v2）：
  - 三 Manager 继承 BaseManager，消除 cursor 样板
  - 修复连接泄漏：懒加载连接通过 _ensure_conn + close() 生命周期管理
  - 修复括号不匹配：《」→ 《》
  - 修复 build_writer_context 中 start_chapter 始终为 ? 的 Bug
  - 移除 6 个未被调用的遗留方法（build_phase_chain / check_debate_convergence /
    get_parallel_defaults / handle_retry_exhaustion / get_state / get_genre）
  - 硬编码常量替换为 config.constants.QUALITY_DECAY_*
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.config.constants import (
    QUALITY_DECAY_MIN_SAMPLES,
    QUALITY_DECAY_RECENT_LIMIT,
    QUALITY_DECAY_THRESHOLD,
)
from novelfactory.pipeline.base import BaseManager

logger = logging.getLogger(__name__)


# ── 内部 Volume 管理器 ─────────────────────────────────────────────


class VolumeManager(BaseManager):
    """卷完成检测与过渡上下文生成器。"""

    def check_volume_completion(self, project_name: str, current_chapter: int) -> dict:
        """检查当前是否到达卷尾，返回卷完成状态。"""
        try:
            row = self._fetchone(
                """
                SELECT volume_number, title, start_chapter, end_chapter, status
                FROM novel_volumes
                WHERE project_name = %s AND start_chapter <= %s AND end_chapter >= %s
                ORDER BY volume_number DESC LIMIT 1
                """,
                (project_name, current_chapter, current_chapter),
            )

            if row:
                vol_number, title, start_ch, end_ch, status = row
                is_complete = current_chapter >= end_ch
                next_vol = vol_number + 1 if is_complete else None
                return {
                    "volume_complete": is_complete,
                    "volume_number": vol_number,
                    "volume_title": title,
                    "start_chapter": start_ch,
                    "current_chapter": current_chapter,
                    "end_chapter": end_ch,
                    "next_volume": next_vol,
                    "status": status,
                }

            # 没有找到匹配卷 — 返回安全默认
            row = self._fetchone(
                """
                SELECT COALESCE(MAX(volume_number), 0) + 1
                FROM novel_volumes
                WHERE project_name = %s
                """,
                (project_name,),
            )
            next_vol = row[0] if row else 1
            return {
                "volume_complete": False,
                "volume_number": 0,
                "next_volume": next_vol,
            }
        except Exception as e:
            logger.warning("[VolumeManager] check_volume_completion failed: %s", e)
            return {"volume_complete": False, "next_volume": None}

    def build_transition_context(self, project_name: str, current_chapter: int) -> str:
        """构建卷间过渡指导文本。"""
        try:
            row = self._fetchone(
                """
                SELECT volume_number, title, theme, summary
                FROM novel_volumes
                WHERE project_name = %s AND start_chapter <= %s AND end_chapter >= %s
                ORDER BY volume_number DESC LIMIT 1
                """,
                (project_name, current_chapter, current_chapter),
            )
            if row:
                vol_number, title, _theme, _summary = row
                next_vol = vol_number + 1
                next_row = self._fetchone(
                    """
                    SELECT title, theme, summary
                    FROM novel_volumes
                    WHERE project_name = %s AND volume_number = %s
                    """,
                    (project_name, next_vol),
                )
                # Bug fix: 《》配对（原代码 《」 不匹配）
                ctx = f"[卷{vol_number}《{title}》完成，准备进入下一卷"
                if next_row:
                    ctx += f"——《{next_row[0]}》"
                    if next_row[1]:
                        ctx += f"（主题：{next_row[1]}）"
                return ctx + "]"
        except Exception as e:
            logger.warning("[VolumeManager] build_transition_context failed: %s", e)
        return ""


# ── 内部 Quality 管理器 ────────────────────────────────────────────


class QualityManager(BaseManager):
    """质量衰减趋势检测器。"""

    def detect_decay(self, project_name: str) -> dict:
        """检测质量衰减趋势，返回衰减状态和告警。"""
        try:
            rows = self._fetchall(
                """
                SELECT quality_score, chapter_number
                FROM novel_chapters
                WHERE project_name = %s AND quality_score IS NOT NULL
                ORDER BY chapter_number DESC LIMIT %s
                """,
                (project_name, QUALITY_DECAY_RECENT_LIMIT),
            )

            if not rows or len(rows) < QUALITY_DECAY_MIN_SAMPLES:
                return {"decaying": False, "recent_avg": 0.0, "alerts": []}

            scores = [r[0] for r in rows]
            recent_avg = sum(scores) / len(scores)

            first_half = scores[: len(scores) // 2]
            second_half = scores[len(scores) // 2 :]
            decay_detected = (
                sum(second_half) / len(second_half)
                < sum(first_half) / len(first_half) - QUALITY_DECAY_THRESHOLD
            )

            if decay_detected:
                pattern = "持续下降" if len(scores) >= 6 else "近期下滑"
                return {
                    "decaying": True,
                    "recent_avg": round(recent_avg, 1),
                    "alerts": [
                        f"质量{pattern}（近{len(scores)}章均分{recent_avg:.1f}）"
                    ],
                    "pattern": pattern,
                    "scores": scores,
                }

            return {"decaying": False, "recent_avg": round(recent_avg, 1), "alerts": []}
        except Exception as e:
            logger.warning("[QualityManager] detect_decay failed: %s", e)
            return {"decaying": False, "recent_avg": 0.0, "alerts": []}


# ── Phase3Manager — 主入口 ─────────────────────────────────────────


class Phase3Manager(BaseManager):
    """Phase 3 统一管理器 — 卷管理 + 质量检测 + 写入上下文。

    用法::

        # 方式 1：调用方提供连接（推荐，连接生命周期由调用方管理）
        with DatabaseManager.get_instance().get_connection() as conn:
            m = Phase3Manager(conn, project_name)
            vol_status = m.volume.check_volume_completion(project_name, ch)

        # 方式 2：懒加载连接（Phase3Manager 自管理，需 close() 或 with）
        with Phase3Manager(project_name=project_name) as m:
            quality_info = m.quality.detect_decay(project_name)
    """

    def __init__(self, pg_conn: Any = None, project_name: str = "") -> None:
        super().__init__(pg_conn)
        self.name = "Phase3Manager"
        self._project = project_name
        self._volume_mgr: VolumeManager | None = None
        self._quality_mgr: QualityManager | None = None
        self._owns_conn: bool = pg_conn is None

    @property
    def volume(self) -> VolumeManager:
        if self._volume_mgr is None:
            self._ensure_conn()
            self._volume_mgr = VolumeManager(self._conn)
        return self._volume_mgr

    @property
    def quality(self) -> QualityManager:
        if self._quality_mgr is None:
            self._ensure_conn()
            self._quality_mgr = QualityManager(self._conn)
        return self._quality_mgr

    def _ensure_conn(self) -> None:
        """确保有可用连接；若调用方未提供则从池中获取。

        Bug fix: 原代码调用 ``__enter__()`` 但从不 ``__exit__()``，
        导致连接永不归还连接池。改为直接获取连接，
        由 ``close()`` 在 ``with`` 退出或显式调用时归还。
        """
        if self._conn is None:
            from novelfactory.config.database import DatabaseManager

            self._conn = DatabaseManager.get_instance().get_connection()

    def close(self) -> None:
        """归还自管理的连接到连接池（仅当 Phase3Manager 自己获取的连接）。"""
        if self._owns_conn and self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
            try:
                self._conn.close()  # _PooledConnection.close() → pool.putconn
            except Exception as e:
                logger.warning("[Phase3Manager] close connection failed: %s", e)
            finally:
                self._conn = None

    def __enter__(self) -> Phase3Manager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def build_writer_context(self, chapter_number: int) -> str:
        """构建 Phase 3 写入上下文（给 writer agent 的提示）。"""
        parts = []
        try:
            vol_info = self.volume.check_volume_completion(
                self._project, chapter_number
            )
            if vol_info and vol_info.get("volume_number", 0) > 0:
                start_ch = vol_info.get("start_chapter", "?")
                end_ch = vol_info.get("end_chapter", "?")
                parts.append(
                    f"当前卷：第{vol_info['volume_number']}卷"
                    f"（第{start_ch}—{end_ch}章）"
                )
        except Exception as e:
            logger.warning("[Phase3Manager] build_writer_context volume: %s", e)

        try:
            qual = self.quality.detect_decay(self._project)
            if qual and qual.get("decaying"):
                parts.append(f"⚠️ 质量预警：{qual.get('pattern', '下降中')}")
        except Exception as e:
            logger.warning("[Phase3Manager] build_writer_context quality: %s", e)

        return "\n".join(parts) if parts else ""

    def after_chapter(
        self,
        chapter_number: int,
    ) -> dict[str, Any]:
        """章节完成后的 Phase3 分析：卷完成检测 + 质量衰减趋势。

        Args:
            chapter_number: 章节编号

        Returns:
            包含卷完成状态和质量衰减信息的 dict
        """
        result: dict[str, Any] = {"chapter_number": chapter_number}

        # 卷完成检测
        try:
            vol_info = self.volume.check_volume_completion(
                self._project, chapter_number
            )
            if vol_info:
                result["volume"] = vol_info
        except Exception as e:
            logger.warning("[Phase3Manager] after_chapter volume: %s", e)

        # 质量衰减检测
        try:
            decay_info = self.quality.detect_decay(self._project)
            if decay_info:
                result["quality_decay"] = decay_info
        except Exception as e:
            logger.warning("[Phase3Manager] after_chapter quality: %s", e)

        return result

    def __repr__(self) -> str:
        return f"<{self.name}>"
