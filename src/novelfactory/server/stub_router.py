"""DeerFlow 前端兼容性 Stub 路由 — 提供 P0 缺失端点。

这些端点的路由路径不带 /api 前缀，因为 Nginx 已将 /api/ → / 做前缀剥离。
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["deerflow-compat"])


# ── Auth (v1) ────────────────────────────────────────────────────────────────


@router.get("/v1/auth/me")
async def auth_me():
    """DeerFlow 前端认证检查 — 返回匿名用户。"""
    return {
        "id": "anonymous",
        "name": "NovelFactory User",
        "email": "user@novelfactory.local",
        "avatar_url": None,
        "role": "admin",
        "is_setup_complete": True,
    }


@router.get("/v1/auth/setup-status")
async def auth_setup_status():
    """DeerFlow 前端安装状态检查。"""
    return {"setup_complete": True, "has_admin": True, "has_users": True}


@router.post("/v1/auth/logout")
async def auth_logout():
    """DeerFlow 前端登出（无操作）。"""
    return {"ok": True}


# ── Agents ───────────────────────────────────────────────────────────────────


@router.get("/agents")
async def list_agents():
    """Agent 列表 — NovelFactory 使用 Assistant 概念，返回空列表。"""
    return {"agents": [
        {
            "name": "novelfactory-assistant",
            "display_name": "NovelFactory",
            "description": "小说创作智能体",
            "model_name": "deepseek-v4-flash",
            "is_default": True,
            "is_available": True,
        }
    ]}


@router.get("/agents/check")
async def check_agent_name(name: str = ""):
    """检查 Agent 名称可用性。"""
    return {"available": True}


# ── Models ────────────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models(request: Request):
    """模型列表 — 返回 NovelFactory 使用的模型。"""
    return {
        "models": [
            {
                "id": "deepseek-v4-flash",
                "name": "DeepSeek V4 Flash",
                "provider": "ark",
                "capabilities": ["chat", "streaming", "function_calling"],
                "is_default": True,
            }
        ],
        "token_usage": {"enabled": False},
    }


# ── Threads extension ────────────────────────────────────────────────────────


@router.post("/threads")
async def create_thread():
    """创建新 Thread — 实际由 LangGraph SDK 处理，这里返回占位。"""
    return {"ok": True}


# ── Scheduled Tasks (stub) ───────────────────────────────────────────────────


@router.get("/scheduled-tasks")
async def list_scheduled_tasks():
    """定时任务列表 — NovelFactory 使用 cron 机制，返回空。"""
    return {"tasks": [], "total": 0}


@router.post("/scheduled-tasks")
async def create_scheduled_task():
    """创建定时任务（stub）。"""
    return {"ok": True, "id": ""}


# ── Channels stub ────────────────────────────────────────────────────────────


@router.get("/channels/providers")
async def list_channel_providers():
    """渠道 Provider 列表。"""
    return {
        "providers": [
            {"type": "feishu", "name": "飞书", "available": True, "connected": False}
        ]
    }


@router.get("/channels/connections")
async def list_channel_connections():
    """渠道连接列表。"""
    return {"connections": []}
