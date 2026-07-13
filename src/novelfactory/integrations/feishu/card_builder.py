"""飞书交互卡片构建器 — 用于人工审核中断场景。

v6.0: 将 interrupt() 中断数据转换为飞书交互卡片，
用户可直接在飞书中点击按钮 approve/reject/modify，
回调通过 feishu_callback endpoint 恢复线程。
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# 卡片颜色映射
_STATUS_COLORS = {
    "approve": "green",
    "reject": "red",
    "modify": "orange",
    "pending": "blue",
}

# 审核类型中文映射
_REVIEW_TYPE_LABELS = {
    "kickoff": "开篇审核",
    "chapter": "章节终审",
    "quality": "质量审核",
    "unknown": "待审核",
}


def build_review_card(interrupt_data: dict) -> dict:
    """将 interrupt() 中断数据构建为飞书交互卡片。

    Args:
        interrupt_data: wait_for_review_node 传入 interrupt() 的数据字典
            {
                "review_type": str,
                "project_name": str,
                "chapter_id": int,
                "chapter_draft_preview": str,
                "quality_score": float,
                "suggested_actions": list[str],
                "thread_id": str,
            }

    Returns:
        飞书交互卡片 JSON（可直接用于 lark-cli im message send --card）
    """
    review_type = interrupt_data.get("review_type", "unknown")
    project_name = interrupt_data.get("project_name", "未命名项目")
    chapter_id = interrupt_data.get("chapter_id", 0)
    preview = interrupt_data.get("chapter_draft_preview", "")
    score = interrupt_data.get("quality_score", 0)
    thread_id = interrupt_data.get("thread_id", "")
    review_label = _REVIEW_TYPE_LABELS.get(review_type, review_type)

    # 截断预览文本
    preview_display = preview[:300] + "..." if len(preview) > 300 else preview

    # 构建飞书交互卡片
    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"📝 {review_label} — {project_name}",
            },
            "template": _STATUS_COLORS.get("pending", "blue"),
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**项目**: {project_name}\n"
                        f"**章节**: 第{chapter_id}章\n"
                        f"**审核类型**: {review_label}\n"
                        f"**质量评分**: {score:.1f}/100"
                    ),
                },
            },
            {"tag": "hr"},
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**章节预览**:\n{preview_display}"
                    if preview_display
                    else "（无预览）",
                },
            },
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✅ 通过"},
                        "type": "primary",
                        "value": json.dumps(
                            {
                                "action": "approve",
                                "thread_id": thread_id,
                                "chapter_id": chapter_id,
                            }
                        ),
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "🔄 重新生成"},
                        "type": "default",
                        "value": json.dumps(
                            {
                                "action": "reject",
                                "thread_id": thread_id,
                                "chapter_id": chapter_id,
                            }
                        ),
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "✏️ 修改建议"},
                        "type": "danger",
                        "value": json.dumps(
                            {
                                "action": "modify",
                                "thread_id": thread_id,
                                "chapter_id": chapter_id,
                            }
                        ),
                    },
                ],
            },
        ],
    }

    return card


def build_progress_card(
    project_name: str,
    chapter: int,
    total: int,
    status: str = "writing",
) -> dict:
    """构建进度通知卡片（非交互，仅展示）。

    Args:
        project_name: 项目名称
        chapter: 当前章节
        total: 总章节数
        status: 当前状态

    Returns:
        飞书消息卡片 JSON
    """
    progress_pct = (chapter / total * 100) if total > 0 else 0
    bar_length = 20
    filled = int(progress_pct / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)

    status_labels = {
        "writing": "写作中",
        "reviewing": "审核中",
        "syncing": "同步中",
        "completed": "已完成",
    }

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📊 {project_name} — 创作进度"},
            "template": "blue" if status != "completed" else "green",
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**状态**: {status_labels.get(status, status)}\n"
                        f"**进度**: {chapter}/{total} 章\n"
                        f"`{bar}` {progress_pct:.1f}%"
                    ),
                },
            },
        ],
    }


def parse_card_action_value(action_value: str) -> dict | None:
    """解析飞书卡片按钮回调的 value 字段。

    Args:
        action_value: 按钮 value 字段的 JSON 字符串

    Returns:
        解析后的字典，失败返回 None
    """
    try:
        return json.loads(action_value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[card_builder] Failed to parse action value: %s", action_value)
        return None
