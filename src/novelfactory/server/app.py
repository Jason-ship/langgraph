"""NovelFactory LangGraph API Server.

Production-grade FastAPI server with full LangGraph SDK compatibility:
  - All standard SDK endpoints (assistants, threads, runs, crons, store)
  - astream_events(version="v2") for real-time streaming
  - Standard interrupt() detection via stream.interrupts
  - AsyncPostgresSaver + AsyncPostgresStore
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import threading
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

# ── JSON encoder for AIMessage/BaseMessage serialization ──
# Uses CustomJSONResponse (below) instead of global monkey-patch.
from novelfactory.config.settings import settings
from novelfactory.server.serialization import _MessageJSONEncoder

# ── Custom JSONResponse ──────────────────────────────────────────────────────


class CustomJSONResponse(JSONResponse):
    """JSONResponse that uses _MessageJSONEncoder for AIMessage serialization.

    Replaces the previous global json.dumps monkey-patch. Only affects HTTP
    responses emitted by FastAPI — library-level json.dumps is unaffected.
    """

    def render(self, content: Any) -> bytes:
        return _json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=_MessageJSONEncoder,
        ).encode("utf-8")


from novelfactory.graph.checkpointer import (  # noqa: E402
    create_checkpointer,
    create_store,
    get_checkpointer_instance,
    set_checkpointer_instance,
)
from novelfactory.graph.new_builder import compile_app  # noqa: E402

logger = logging.getLogger(__name__)

# ── Suppress upstream LangChain deprecation warnings ──────────────────────────
warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
    category=DeprecationWarning,
    module="langchain",
)

# ── Constants ───────────────────────────────────────────────────
from novelfactory.server.error_handlers import register_error_handlers  # noqa: E402
from novelfactory.server.schedulers import (  # noqa: E402
    cleanup_abnormal_crons,
    cron_scheduler,
    periodic_gc,
)

# ── Globals ────────────────────────────────────────────────────

_app_instance: Any = None
_app_lock = threading.Lock()

# Thread Store — persisted via checkpointer (official LangGraph standard)
# Per the official langgraph-api architecture, threads are persisted in the
# checkpointer's underlying database.  We query the checkpointer directly for
# thread listing rather than maintaining a separate in-memory index.  Only
# run metadata (ephemeral per-invocation) is kept in memory.
_run_store: dict[str, list[dict]] = {}  # thread_id -> [run_info, ...] (ephemeral)

# v7.8: 上限守卫 — 防止 _run_store 无限增长
_MAX_RUN_STORE_ENTRIES = 1000  # 全局最大条目数
_MAX_RUNS_PER_THREAD = 100     # 单线程最大运行记录数


def _add_run_to_store(thread_id: str, run_info: dict) -> None:
    """添加运行记录到 _run_store，自动触发守卫清理。"""
    # 检查全局上限
    total = sum(len(v) for v in _run_store.values())
    if total >= _MAX_RUN_STORE_ENTRIES:
        # 清空所有已完成运行记录
        _run_store.clear()
    # 检查单线程上限
    runs = _run_store.setdefault(thread_id, [])
    if len(runs) >= _MAX_RUNS_PER_THREAD:
        runs.pop(0)  # FIFO — 丢弃最旧记录
    runs.append(run_info)


async def get_app() -> CompiledGraph:
    """Lazy-initialized compiled graph singleton (thread-safe)."""
    global _app_instance
    if _app_instance is None:
        with _app_lock:
            if _app_instance is None:
                checkpointer = await create_checkpointer()
                set_checkpointer_instance(checkpointer)
                store = await create_store()
                _app_instance = await compile_app(
                    checkpointer=checkpointer, store=store
                )
                logger.info("[server] App compiled and ready")
    return _app_instance


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: warm up the graph on startup, cleanup resources on shutdown."""
    # v5.4 fix: 确保 novelfactory 包内 logger 输出 INFO 级别日志，
    # 使评分过程（quality/composite）在容器日志中可见
    logging.getLogger("novelfactory").setLevel(logging.INFO)
    _nf_logger = logging.getLogger("novelfactory")
    if not _nf_logger.handlers:
        _nf_handler = logging.StreamHandler()
        _nf_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s +08:00: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _nf_logger.addHandler(_nf_handler)
    # 预热 graph 并挂载到 app.state，供 periodic_gc 访问 checkpointer
    app.state.graph = await get_app()
    # 启动时清理异常 cron（防止数据库残留的异常 cron 自动执行）
    store = getattr(app.state.graph, "store", None)
    if store:
        await cleanup_abnormal_crons(store)

    # ── 启动渠道服务 ──────────────────────────────────────────────────────
    channel_service = None
    try:
        from novelfactory.channels import start_channel_service

        channels_enabled = os.environ.get("CHANNELS_ENABLED", str(settings.CHANNELS_ENABLED)).lower() == "true"
        feishu_app_id = os.environ.get("LARK_APP_ID", settings.LARK_APP_ID or "")
        feishu_app_secret = os.environ.get("LARK_APP_SECRET", settings.LARK_APP_SECRET or "")

        if channels_enabled and feishu_app_id and feishu_app_secret:
            channels_config = {
                "feishu": {
                    "enabled": True,
                    "app_id": feishu_app_id,
                    "app_secret": feishu_app_secret,
                }
            }
            channel_service = await start_channel_service(
                channels_config=channels_config,
                get_graph=lambda: app.state.graph,
            )
            app.state.channel_service = channel_service
            logger.info("[server] Channel service started")
        elif feishu_app_id:
            logger.info("[server] Channels disabled, skipping channel service")
        else:
            logger.info("[server] Feishu not configured (LARK_APP_ID empty), skipping channel service")
    except Exception:
        logger.exception("[server] Failed to start channel service")

    gc_task = asyncio.create_task(periodic_gc(app))
    cron_task = asyncio.create_task(cron_scheduler(app))
    yield
    cron_task.cancel()
    gc_task.cancel()
    logger.info("[server] Shutting down — releasing resources")

    # ── 停止渠道服务 ──────────────────────────────────────────────────────
    if channel_service is not None:
        try:
            from novelfactory.channels import stop_channel_service

            await stop_channel_service()
            logger.info("[server] Channel service stopped")
        except Exception:
            logger.exception("[server] Error stopping channel service")

    # ── Close DatabaseManager ──────────────────────────────────────────────
    try:
        from novelfactory.config.database import DatabaseManager

        if DatabaseManager._instance is not None:
            DatabaseManager._instance.close()
            logger.info("[server] Database pool closed")
    except Exception as exc:
        logger.debug("[server] Database pool close skipped: %s", exc)

    # ── Close Checkpointer (Postgres connection pool) ──────────────────────
    cp = get_checkpointer_instance()
    if cp is not None:
        try:
            pool = getattr(cp, "conn", None)
            if pool is not None:
                await pool.close()
                logger.info("[server] Checkpointer connection pool closed")
            elif hasattr(cp, "aclose"):
                await cp.aclose()
                logger.info("[server] Checkpointer closed via aclose")
        except Exception as exc:
            logger.warning("[server] Error closing checkpointer: %s", exc)

    # ── Close Store (Postgres connection pool) ─────────────────────────────
    graph = getattr(app.state, "graph", None)
    if graph is not None:
        store = getattr(graph, "store", None)
        if store is not None:
            try:
                pool = getattr(store, "pool", None)
                if pool is not None:
                    await pool.close()
                    logger.info("[server] Store connection pool closed")
                elif hasattr(store, "aclose"):
                    await store.aclose()
                    logger.info("[server] Store closed via aclose")
            except Exception as exc:
                logger.warning("[server] Error closing store: %s", exc)


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NovelFactory LangGraph API",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    default_response_class=CustomJSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "assistants", "description": "LangGraph Assistant management"},
        {"name": "threads", "description": "Thread state management"},
        {"name": "runs", "description": "Graph execution (streaming + sync)"},
        {"name": "store", "description": "Key-value store"},
        {"name": "health", "description": "Health and readiness checks"},
        {"name": "console", "description": "Operations console & observability"},
        {"name": "memory", "description": "Global memory management"},
        {"name": "feedback", "description": "User feedback collection"},
        {"name": "input-polish", "description": "Input text polishing"},
        {"name": "suggestions", "description": "Follow-up suggestion generation"},
        {"name": "channels", "description": "IM channel management"},
    ],
)

# ── Trace Middleware ────────────────────────────────────────────────────────────
try:
    from novelfactory.middleware.trace_middleware import TraceMiddleware

    app.add_middleware(TraceMiddleware, enabled=True)
    logger.info("[server] Trace middleware enabled")
except Exception:
    logger.debug("[server] Trace middleware not available")

# ── CSRF Middleware ────────────────────────────────────────────────────────────
try:
    from novelfactory.middleware.csrf_middleware import CSRFMiddleware

    app.add_middleware(CSRFMiddleware, enabled=True)
    logger.info("[server] CSRF middleware enabled")
except Exception:
    logger.debug("[server] CSRF middleware not available")

# ── Request Logging Middleware (v7.8) ──────────────────────────────────────────
try:
    from novelfactory.middleware.request_logger import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)
    logger.info("[server] Request logging middleware enabled")
except Exception:
    logger.debug("[server] Request logging middleware not available")

# ── Rate Limit Middleware (v7.8) ───────────────────────────────────────────────
try:
    from novelfactory.middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware)
    logger.info("[server] Rate limit middleware enabled")
except Exception:
    logger.debug("[server] Rate limit middleware not available")

# v6.1: 统一从 settings 读取
# v7.8: 默认值从 "*" 改为 "http://localhost:3000"（生产环境应设置具体域名）
allowed_origins = os.environ.get(
    "CORS_ALLOWED_ORIGINS", settings.CORS_ALLOWED_ORIGINS or "http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=len(allowed_origins) == 1 and allowed_origins[0] != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Error Handlers ────────────────────────────────────────────────────────────

register_error_handlers(app)

# ── Route Registration ────────────────────────────────────────────────────────

# v6.1 P3-2: 注册时间旅行 API
from novelfactory.api.time_travel import router as time_travel_router  # noqa: E402
from novelfactory.server.routes.assistants import (  # noqa: E402
    router as assistants_router,
)
from novelfactory.server.routes.channel_connections import (  # noqa: E402
    router as channel_connections_router,
)
from novelfactory.server.routes.console import (  # noqa: E402
    router as console_router,
)
from novelfactory.server.routes.crons import router as crons_router  # noqa: E402
from novelfactory.server.routes.features import (  # noqa: E402
    router as features_router,
)
from novelfactory.server.routes.feedback import (  # noqa: E402
    router as feedback_router,
)
from novelfactory.server.routes.feishu_callback import (  # noqa: E402
    router as feishu_callback_router,
)
from novelfactory.server.routes.health import router as health_router  # noqa: E402
from novelfactory.server.routes.input_polish import (  # noqa: E402
    router as input_polish_router,
)
from novelfactory.server.routes.memory import (  # noqa: E402
    router as memory_router,
)
from novelfactory.server.routes.runs import router as runs_router  # noqa: E402
from novelfactory.server.routes.store import router as store_router  # noqa: E402
from novelfactory.server.routes.suggestions import (  # noqa: E402
    router as suggestions_router,
)
from novelfactory.server.routes.threads import router as threads_router  # noqa: E402

app.include_router(assistants_router)
app.include_router(channel_connections_router)
app.include_router(console_router)
app.include_router(crons_router)
app.include_router(features_router)
app.include_router(feedback_router)
app.include_router(feishu_callback_router)
app.include_router(health_router)
app.include_router(input_polish_router)
app.include_router(memory_router)
app.include_router(runs_router)
app.include_router(store_router)
app.include_router(suggestions_router)
app.include_router(threads_router)
app.include_router(time_travel_router)

# ── Static Files ──────────────────────────────────────────────────────────────
import os as _os
from fastapi.staticfiles import StaticFiles

_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── SDK: /ui ───────────────────────────────────────────────────────────────────


@app.post("/ui/{assistant_id}", include_in_schema=False)
async def get_ui(assistant_id: str) -> dict:
    """Get UI component (stub)."""
    return {"html": ""}


# ── Favicon ────────────────────────────────────────────────────────────────────


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon requests to avoid 404 in logs."""
    return JSONResponse(status_code=204)


# ── Prometheus Metrics (v7.8) ─────────────────────────────────────────────────
from fastapi.responses import PlainTextResponse  # noqa: E402


@app.get("/metrics", include_in_schema=False, tags=["health"])
async def metrics():
    """Prometheus 格式的指标数据。"""
    from novelfactory.server.metrics import generate_metrics

    return PlainTextResponse(content=generate_metrics(), media_type="text/plain; charset=utf-8")


# ── Main Entrypoint ────────────────────────────────────────────────────────────


def main() -> None:
    """启动 uvicorn 服务器。"""
    # v6.1: 统一从 settings 读取
    host = settings.HOST or os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", str(settings.PORT or "8000")))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "novelfactory.server.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
