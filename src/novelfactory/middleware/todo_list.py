"""TodoListMiddleware — 项目启动时自动生成任务列表，阶段完成时更新状态。

任务列表:
  - 世界观设定 (setup)
  - 角色设计 (setup)
  - 故事大纲 (setup)
  - 章节写作 (writing)
  - 媒体生成 (media)
  - 飞书同步 (sync)
"""

from __future__ import annotations

import logging
from typing import Any

from novelfactory.middleware.base import Middleware

logger = logging.getLogger(__name__)

# 任务定义（按执行顺序）
TASKS = [
    {"id": "world_setting", "name": "世界观设定", "phase": "setup"},
    {"id": "character_design", "name": "角色设计", "phase": "setup"},
    {"id": "outline", "name": "故事大纲", "phase": "setup"},
    {"id": "chapter_writing", "name": "章节写作", "phase": "writing"},
    {"id": "media_generation", "name": "媒体生成", "phase": "media"},
    {"id": "feishu_sync", "name": "飞书同步", "phase": "sync"},
]


class TodoListMiddleware(Middleware):
    """在项目启动时生成任务列表，阶段切换时更新任务状态。"""

    def before_node(self, state: dict, config: dict) -> dict | None:
        updates: dict[str, Any] = {}

        # ── 首次进入：生成任务列表 ───────────────────────────────────────────
        if not state.get("todo_list"):
            total_chapters = state.get("target_chapters") or 10
            # 修正章节写作任务名称
            tasks = []
            for t in TASKS:
                task = dict(t)
                if task["id"] == "chapter_writing":
                    task["name"] = f"章节写作（{total_chapters}章）"
                task["status"] = "pending"
                tasks.append(task)
            updates["todo_list"] = tasks
            logger.info("[todo_list] 已生成 %d 项任务", len(tasks))

        # ── 阶段切换时更新任务状态 ──────────────────────────────────────────
        current_phase = state.get("current_phase", "")
        todo_list = list(state.get("todo_list", []) or updates.get("todo_list", []))
        if not todo_list:
            return updates if updates else None

        changed = False
        # 标记当前阶段的任务为 in_progress
        for task in todo_list:
            if task["phase"] == current_phase and task["status"] == "pending":
                task["status"] = "in_progress"
                changed = True
                break  # 只激活第一个匹配的任务

        # 标记已完成阶段的任务为 completed
        phase_order = ["setup", "writing", "media", "sync", "done"]
        if current_phase in phase_order:
            idx = phase_order.index(current_phase)
            completed_phases = set(phase_order[:idx])
            for task in todo_list:
                if task["phase"] in completed_phases and task["status"] != "completed":
                    task["status"] = "completed"
                    changed = True

        if changed:
            updates["todo_list"] = todo_list
            logger.info("[todo_list] 已更新任务状态: phase=%s", current_phase)

        return updates if updates else None
