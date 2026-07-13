"""Background schedulers: periodic GC and cron task execution.

Extracted from server/app.py (v6.1 P1-4) for single-responsibility separation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ── Scheduler Constants ────────────────────────────────────────────────────────

_GC_INTERVAL = 21600  # 6 hours — periodic GC cycle
_CRON_POLL_INTERVAL = 60  # 每 60 秒检查一次到期 cron
_CRON_MAX_RUNS = 1000  # 单个 cron 最大执行次数，超过自动暂停
_GC_RUN_RECORD_TTL = 86400  # 24 hours — ephemeral run record TTL


# ── Periodic GC ────────────────────────────────────────────────────────────────


async def periodic_gc(app: FastAPI) -> None:
    """Periodic GC (every 6 hours): 清理过期 run 记录，记录 checkpoint 状态。"""
    # Lazy import to avoid circular dependency with app.py
    from novelfactory.server.app import _run_store

    while True:
        await asyncio.sleep(_GC_INTERVAL)
        try:
            graph = getattr(app.state, "graph", None)
            if graph and getattr(graph, "checkpointer", None):
                cp = graph.checkpointer
                logger.info(
                    "[gc] Periodic GC cycle — checkpointer available (%s)",
                    type(cp).__name__,
                )
                now = time.time()
                cleaned = 0
                for tid in list(_run_store.keys()):
                    runs = _run_store[tid]
                    _run_store[tid] = [
                        r
                        for r in runs
                        if now - r.get("started_at", now) < _GC_RUN_RECORD_TTL
                    ]
                    if not _run_store[tid]:
                        del _run_store[tid]
                        cleaned += 1
                if cleaned:
                    logger.info("[gc] Cleaned %d stale thread run-records", cleaned)
            else:
                logger.warning("[gc] No checkpointer available — skipping GC cycle")
        except Exception:
            logger.warning("[gc] Periodic GC cycle failed", exc_info=True)


# ── Cron Scheduler ─────────────────────────────────────────────────────────────


async def cleanup_abnormal_crons(store: Any) -> None:
    """启动时清理异常 cron：自动暂停 total_runs > _CRON_MAX_RUNS 或仅有空 input 的 cron。

    防止数据库残留的异常 cron 在服务启动后自动执行。
    """
    try:
        items = await store.asearch(("cron",), limit=1000)
        now = datetime.now(timezone.utc)
        paused = 0
        for item in items:
            val = item.value
            if val.get("status") != "active":
                continue
            needs_pause = False
            reason = ""
            inp = val.get("input", {})
            if not inp or inp == {}:
                needs_pause = True
                reason = "空 input"
            elif val.get("total_runs", 0) >= _CRON_MAX_RUNS:
                needs_pause = True
                reason = f"total_runs={val['total_runs']} ≥ {_CRON_MAX_RUNS}"
            if needs_pause:
                val["status"] = "paused"
                val["updated_at"] = now.isoformat()
                await store.aput(("cron",), item.key, val)
                paused += 1
                logger.warning(
                    "[cron] 启动时清理 — cron=%s 已暂停（原因: %s）",
                    item.key,
                    reason,
                )
        if paused:
            logger.info("[cron] 启动时清理完成 — 共暂停 %d 个异常 cron", paused)
    except Exception as e:
        logger.warning("[cron] 启动时清理失败: %s", e)


async def cron_scheduler(app: FastAPI) -> None:
    """后台 Cron 调度器：每 60 秒检查一次到期任务并执行。

    安全守卫：
      - **空 input 守卫**：cron 的 input 为 ``{}`` 时跳过执行（防止误创建的空 cron 持续执行）
      - **异常高频检测**：total_runs > _CRON_MAX_RUNS 自动暂停并告警
    """
    while True:
        await asyncio.sleep(_CRON_POLL_INTERVAL)
        try:
            graph = getattr(app.state, "graph", None)
            store = getattr(graph, "store", None) if graph else None
            if store is None:
                continue

            items = await store.asearch(("cron",), limit=1000)
            now = datetime.now(timezone.utc)
            for item in items:
                try:
                    val = item.value
                    if val.get("status") != "active":
                        continue

                    # ── [Safety] 空 input 守卫 ──────────────────────────────
                    inp = val.get("input", {})
                    if not inp or inp == {}:
                        logger.warning(
                            "[cron] 空 input 守卫触发 — cron=%s 仅有空 input，已暂停",
                            item.key,
                        )
                        val["status"] = "paused"
                        val["updated_at"] = now.isoformat()
                        await store.aput(("cron",), item.key, val)
                        continue

                    # ── [Safety] 异常高频检测 ──────────────────────────────
                    total_runs = val.get("total_runs", 0)
                    if total_runs >= _CRON_MAX_RUNS:
                        logger.warning(
                            "[cron] 异常高频检测触发 — cron=%s runs=%d ≥ %d，已自动暂停",
                            item.key,
                            total_runs,
                            _CRON_MAX_RUNS,
                        )
                        val["status"] = "paused"
                        val["updated_at"] = now.isoformat()
                        await store.aput(("cron",), item.key, val)
                        continue

                    # ── Schedule 检查：按 cron 表达式判断是否到期 ──────────────
                    from novelfactory.server.routes.crons import should_run_now

                    schedule = val.get("schedule", "* * * * *")
                    if not should_run_now(schedule, now):
                        continue

                    # 避免同一分钟内重复执行（last_run_at 去重）
                    last_run_str = val.get("last_run_at", "")
                    if last_run_str:
                        try:
                            last_run_dt = datetime.fromisoformat(last_run_str)
                            if last_run_dt.tzinfo is None:
                                last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
                            if (now - last_run_dt).total_seconds() < 60:
                                continue
                        except (ValueError, TypeError):
                            pass  # 解析失败，继续执行

                    thread_id = val.get("thread_id", str(uuid.uuid4()))
                    from langchain_core.runnables import RunnableConfig

                    from novelfactory.state.novel_context import NovelContext

                    config: RunnableConfig = {
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 100,
                    }
                    context: NovelContext = {
                        "thread_id": thread_id,
                        "user_id": "cron",
                        "project_id": "",
                        "request_id": str(uuid.uuid4()),
                        "lark_config": None,
                    }
                    # 非阻塞执行（不等待结果）
                    asyncio.create_task(
                        graph.ainvoke(inp, config=config, context=context)
                    )
                    # 更新运行记录
                    val["last_run_at"] = now.isoformat()
                    val["total_runs"] = total_runs + 1
                    val["updated_at"] = now.isoformat()
                    await store.aput(("cron",), item.key, val)
                    logger.info(
                        "[cron] Executed cron %s (thread=%s, runs=%d)",
                        item.key,
                        thread_id,
                        total_runs + 1,
                    )
                except Exception as exc:
                    logger.warning(
                        "[cron] Failed to execute cron %s: %s", item.key, exc
                    )
        except Exception:
            logger.debug("[cron] Scheduler cycle failed", exc_info=True)
