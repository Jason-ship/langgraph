# ==============================================================================
# Health Check Endpoints
# ==============================================================================

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from novelfactory.config.settings import settings
from novelfactory.server.deps import get_graph

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", tags=["health"])
async def health() -> dict:
    """Health check endpoint — includes tools-proxy and channel status."""
    tp_status = await _check_tools_proxy()

    # Check channel service status
    channel_status = "not_configured"
    channel_info = {}
    try:
        from novelfactory.channels.service import get_channel_service

        service = get_channel_service()
        if service is not None:
            status = service.get_status()
            channel_status = "running" if status.get("service_running") else "stopped"
            channel_info = status
    except Exception:
        channel_status = "unavailable"

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "tools_proxy": tp_status,
        "channel_service": channel_status,
        "channel_info": channel_info,
        "lark_proxy_url": _get_lark_proxy_url(),
    }


def _get_lark_proxy_url() -> str:
    """获取 lark-proxy 实际地址（与 feishu_toolkit 保持一致）。

    v6.1: 统一从 settings 读取。
    """
    return (
        settings.lark_proxy_url
        or f"http://{settings.LARK_PROXY_HOST}:{settings.LARK_PROXY_PORT}"
    )


async def _check_tools_proxy() -> str:
    """Check tools-proxy health from inside the container."""
    proxy_url = _get_lark_proxy_url()
    try:
        import httpx as _httpx

        async with _httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{proxy_url}/health")
            if r.status_code == 200:
                return r.json().get("status", "ok")
            return f"degraded (HTTP {r.status_code})"
    except _httpx.ConnectError:
        return "unreachable"
    except Exception:
        return "unknown"


@router.get("/ready", tags=["health"])
async def ready() -> dict:
    """Readiness check — verifies graph is compiled and DB connections are alive."""
    try:
        graph = await get_graph()
    except (ValueError, OSError) as e:
        logger.exception("Readiness check failed")
        raise HTTPException(status_code=503, detail=f"Not ready: {e}") from e
    else:
        return {
            "status": "ready",
            "version": settings.APP_VERSION,
            "graph_compiled": graph is not None,
        }


@router.get("/info", tags=["health"])
async def info() -> dict:
    """Deployment info endpoint (used by SDK checkGraphStatus)."""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "assistant_id": "novelfactory",
        "deployment_type": "self_hosted",
    }


@router.get("/debug/config", tags=["health"])
async def debug_config() -> dict:
    """Return current effective config snapshot (sensitive fields masked)."""
    cfg = settings.model_dump()
    # Mask sensitive fields
    for key in list(cfg.keys()):
        if any(kw in key.lower() for kw in ["key", "secret", "password", "token"]):
            cfg[key] = "***masked***"
    return cfg


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus-compatible metrics endpoint."""
    import psutil

    process = psutil.Process()
    mem_info = process.memory_info()

    metrics_lines = [
        "# HELP novelfactory_info Application info",
        "# TYPE novelfactory_info gauge",
        f'novelfactory_info{{version="{settings.APP_VERSION}"}} 1',
        "",
        "# HELP novelfactory_memory_rss_bytes Process RSS memory in bytes",
        "# TYPE novelfactory_memory_rss_bytes gauge",
        f"novelfactory_memory_rss_bytes {mem_info.rss}",
        "",
        "# HELP novelfactory_memory_vms_bytes Process VMS memory in bytes",
        "# TYPE novelfactory_memory_vms_bytes gauge",
        f"novelfactory_memory_vms_bytes {mem_info.vms}",
        "",
        "# HELP novelfactory_cpu_percent Process CPU usage percent",
        "# TYPE novelfactory_cpu_percent gauge",
        f"novelfactory_cpu_percent {process.cpu_percent(interval=0.1):.2f}",
        "",
        "# HELP novelfactory_open_fds Process open file descriptors",
        "# TYPE novelfactory_open_fds gauge",
        f"novelfactory_open_fds {process.num_fds() if hasattr(process, 'num_fds') else -1}",
        "",
        "# HELP novelfactory_uptime_seconds Process uptime in seconds",
        "# TYPE novelfactory_uptime_seconds gauge",
        f"novelfactory_uptime_seconds {time.time() - process.create_time():.0f}",
        "",
    ]

    return Response(
        content="\n".join(metrics_lines),
        media_type="text/plain; version=0.0.4",
    )


@router.get("/params", tags=["health"])
async def llm_params() -> dict:
    """返回当前 LLM 参数配置（调优面板用）。"""
    from novelfactory.config.llm_params import center

    return center.list_params()
