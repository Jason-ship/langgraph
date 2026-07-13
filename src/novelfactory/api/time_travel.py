"""Time Travel — 章节回滚 API 端点。

利用已有 AsyncPostgresSaver checkpoint 历史，
支持回滚到任意已完成章节重新生成。

v6.1 P3-2: router 已注册到 app.py，rollback 使用 graph.aupdate_state 实现真正的状态回滚。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["time_travel"])


@router.post("/rollback")
async def rollback(thread_id: str, target_chapter: int):
    """回滚到指定章节。

    通过遍历 checkpoint 历史，找到目标章节的 checkpoint，
    然后使用 graph.aupdate_state 将状态回滚到该 checkpoint。

    Args:
        thread_id: 项目线程 ID
        target_chapter: 目标章节号

    Returns:
        {"success": bool, "message": str, "current_chapter": int}

    Raises:
        400: 目标章节不存在或当前有进行中的写作
        500: Checkpointer 未初始化或回滚失败
    """
    from novelfactory.server.app import get_app

    app = await get_app()
    graph = app.state.graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph 未初始化")

    checkpointer = getattr(graph, "checkpointer", None)
    if not checkpointer:
        raise HTTPException(status_code=500, detail="Checkpointer 未初始化")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 1. 遍历 checkpoint 历史找到目标章节
    target_checkpoint_id = None
    found_chapters: list[int] = []

    try:
        async for state_snapshot in graph.aget_state_history(config):
            current_ch = state_snapshot.values.get("current_chapter", 0)
            if current_ch > 0:
                found_chapters.append(current_ch)
            if current_ch == target_chapter:
                target_checkpoint_id = state_snapshot.config["configurable"].get(
                    "checkpoint_id"
                )
                break
    except Exception as e:
        logger.error("[rollback] 遍历 checkpoint 历史失败: %s", e)
        raise HTTPException(status_code=500, detail="获取 checkpoint 历史失败")

    if target_checkpoint_id is None:
        raise HTTPException(
            status_code=400,
            detail=f"第{target_chapter}章尚未完成，无法回滚。"
            f"已完成章节：{', '.join(str(c) for c in sorted(set(found_chapters))) or '无'}",
        )

    # 2. 获取当前最新状态
    current_state = await graph.aget_state(config)
    current_chapter = current_state.values.get("current_chapter", 1)

    # 3. 使用 aupdate_state 回滚到目标 checkpoint
    rollback_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": target_checkpoint_id,
        }
    }

    try:
        await graph.aupdate_state(
            rollback_config,
            {
                "current_chapter": target_chapter,
                "current_phase": "writing",
            },
        )
        logger.info(
            "[rollback] thread=%s 从第%d章回滚到第%d章 (checkpoint=%s)",
            thread_id,
            current_chapter,
            target_chapter,
            target_checkpoint_id[:12],
        )
    except Exception as e:
        logger.error("[rollback] 状态回滚失败: %s", e)
        raise HTTPException(status_code=500, detail=f"回滚失败: {e}")

    return {
        "success": True,
        "message": f"已回滚到第{target_chapter}章，将从该章重新开始创作",
        "current_chapter": target_chapter,
        "rollback_from": current_chapter,
        "checkpoint_id": target_checkpoint_id,
    }


@router.get("/checkpoints")
async def list_checkpoints(thread_id: str):
    """列出可回滚的 checkpoint 列表。

    Args:
        thread_id: 项目线程 ID

    Returns:
        {"checkpoints": [{"chapter": int, "timestamp": str, "summary": str}, ...]}
    """
    from novelfactory.server.app import get_app

    app = await get_app()
    graph = app.state.graph
    if not graph:
        raise HTTPException(status_code=500, detail="Graph 未初始化")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    result = []
    try:
        async for state_snapshot in graph.aget_state_history(config):
            ch = state_snapshot.values.get("current_chapter", 0)
            if ch > 0:
                ts = getattr(state_snapshot, "created_at", "")
                result.append(
                    {
                        "chapter": ch,
                        "timestamp": str(ts),
                        "checkpoint_id": state_snapshot.config.get(
                            "configurable", {}
                        ).get("checkpoint_id", ""),
                        "summary": f"第{ch}章",
                    }
                )
    except Exception as e:
        logger.error("[checkpoints] 获取列表失败: %s", e)
        raise HTTPException(status_code=500, detail="获取 checkpoint 列表失败")

    # 按章节号排序，去重
    seen = set()
    deduped = []
    for item in result:
        if item["chapter"] not in seen:
            seen.add(item["chapter"])
            deduped.append(item)
    deduped.sort(key=lambda x: x["chapter"])

    return {"checkpoints": deduped}
