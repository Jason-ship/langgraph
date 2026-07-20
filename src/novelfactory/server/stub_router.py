"""DeerFlow 前端兼容性 Stub 路由 — 提供前端需要的全部缺失端点。

这些端点的路由路径不带 /api 前缀，因为 Nginx 已将 /api/ → / 做前缀剥离。
每个端点返回前端能接受的 stub 响应，确保 UI 不报错。
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["deerflow-compat"])


# ═══════════════════════════════════════════════════════════════════════════════
# Auth (v1)
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/v1/auth/me")
async def auth_me():
    """DeerFlow 前端认证检查 — 返回匿名管理员用户。"""
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
    """系统安装状态检查。"""
    return {"setup_complete": True, "has_admin": True, "has_users": True}


@router.post("/v1/auth/logout")
async def auth_logout():
    """登出（无操作 — auth 已禁用）。"""
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Agents
# ═══════════════════════════════════════════════════════════════════════════════


_DEFAULT_AGENT = {
    "name": "novelfactory-assistant",
    "description": "小说创作智能体 — 基于 LangGraph 的多智能体编排",
    "model": "deepseek-v4-flash",
    "tool_groups": [],
    "skills": [],
    "soul": "你是一位专业的小说创作助手，擅长故事策划、章节写作和质量评审。",
}


@router.get("/agents")
async def list_agents():
    """Agent 列表。"""
    return {"agents": [_DEFAULT_AGENT]}


@router.get("/agents/check")
async def check_agent_name(name: str = ""):
    """检查 Agent 名称可用性。"""
    return {"available": True, "name": name}


@router.get("/agents/{agent_name}")
async def get_agent(agent_name: str):
    """获取单个 Agent 详情。"""
    if agent_name == "novelfactory-assistant":
        return _DEFAULT_AGENT
    return JSONResponse(status_code=404, content={"detail": f"Agent '{agent_name}' not found"})


@router.post("/agents")
async def create_agent():
    """创建新 Agent（stub — NovelFactory 使用单 Assistant 模式）。"""
    return _DEFAULT_AGENT


@router.put("/agents/{agent_name}")
async def update_agent(agent_name: str):
    """更新 Agent（stub）。"""
    return _DEFAULT_AGENT


@router.delete("/agents/{agent_name}")
async def delete_agent(agent_name: str):
    """删除 Agent（stub）。"""
    return JSONResponse(status_code=204, content=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/models")
async def list_models(request: Request):
    """可用模型列表。"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Threads — 扩展端点
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/threads/{thread_id}/token-usage")
async def get_thread_token_usage(thread_id: str):
    """线程 Token 用量统计。"""
    return {
        "thread_id": thread_id,
        "total_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_runs": 0,
        "by_model": {},
        "by_caller": {},
    }


@router.post("/threads/{thread_id}/branches")
async def branch_thread(thread_id: str):
    """从指定 turn 创建分支线程（stub）。"""
    return {
        "thread_id": f"{thread_id}-branch",
        "parent_thread_id": thread_id,
        "parent_checkpoint_id": "",
        "branched_from_message_id": "",
        "workspace_clone_mode": "none",
    }


@router.post("/threads/{thread_id}/compact")
async def compact_thread(thread_id: str):
    """压缩线程上下文（stub）。"""
    now = datetime.now(UTC).isoformat()
    return {
        "thread_id": thread_id,
        "compacted": True,
        "reason": None,
        "removed_message_count": 0,
        "preserved_message_count": 0,
        "summary_updated": False,
        "checkpoint_id": None,
        "total_tokens": 0,
    }


@router.get("/threads/{thread_id}/messages/page")
async def get_thread_messages_page(
    thread_id: str,
    before_seq: int | None = None,
    limit: int = 50,
):
    """分页加载线程消息历史（stub）。"""
    return {"data": [], "has_more": False, "next_before_seq": None}


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """获取线程可显示消息列表（stub）。"""
    return []


@router.post("/threads/{thread_id}/runs/regenerate/prepare")
async def prepare_regenerate(thread_id: str):
    """准备重新生成回复（stub）。"""
    return {
        "input": {"messages": []},
        "checkpoint": {"checkpoint_ns": "", "checkpoint_id": "", "checkpoint_map": {}},
        "metadata": {
            "regenerate_from_message_id": "",
            "regenerate_from_run_id": "",
            "regenerate_checkpoint_id": "",
        },
        "target_run_id": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Thread — Runs 扩展端点
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/threads/{thread_id}/runs/{run_id}/workspace-changes")
async def get_workspace_changes(thread_id: str, run_id: str):
    """工作区文件变更（stub — NovelFactory 不使用文件变更功能）。"""
    return {
        "available": False,
        "version": 1,
        "summary": {"created": 0, "modified": 0, "deleted": 0, "additions": 0, "deletions": 0, "truncated": False},
        "files": [],
        "limits": {"max_files": 0, "max_scanned_files": 0, "max_file_bytes_for_diff": 0, "max_total_diff_bytes": 0},
    }


@router.get("/threads/{thread_id}/runs/{run_id}/events")
async def get_run_events(thread_id: str, run_id: str):
    """子任务事件历史（stub — NovelFactory 不支持 subagent 事件追踪）。"""
    return []


@router.get("/threads/{thread_id}/runs/{run_id}/messages")
async def get_run_messages(thread_id: str, run_id: str):
    """运行消息历史（stub）。"""
    return {"data": [], "has_more": False}


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback — 仅补充 feedback.py 缺失的 HTTP 方法
# ═══════════════════════════════════════════════════════════════════════════════


@router.put("/threads/{thread_id}/runs/{run_id}/feedback")
async def upsert_feedback(thread_id: str, run_id: str):
    """幂等创建/更新反馈。"""
    return {
        "feedback_id": "",
        "run_id": run_id,
        "thread_id": thread_id,
        "user_id": "anonymous",
        "message_id": None,
        "rating": 0,
        "comment": None,
        "created_at": datetime.now(UTC).isoformat(),
    }


@router.delete("/threads/{thread_id}/runs/{run_id}/feedback")
async def delete_feedback(thread_id: str, run_id: str):
    """删除当前用户的反馈。"""
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Thread — Uploads
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/threads/{thread_id}/uploads")
async def upload_files(thread_id: str):
    """上传文件到线程（stub）。"""
    return {"success": True, "files": [], "message": "Upload stub — no-op", "skipped_files": []}


@router.get("/threads/{thread_id}/uploads/limits")
async def get_upload_limits(thread_id: str):
    """上传限制。"""
    return {"max_files": 0, "max_file_size": 0, "max_total_size": 0}


@router.get("/threads/{thread_id}/uploads/list")
async def list_uploads(thread_id: str):
    """列出已上传文件。"""
    return {"files": [], "count": 0}


@router.delete("/threads/{thread_id}/uploads/{filename:path}")
async def delete_upload(thread_id: str, filename: str):
    """删除上传文件。"""
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Thread — Artifacts
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/threads/{thread_id}/artifacts/{path:path}")
async def get_artifact(thread_id: str, path: str):
    """获取 artifact 文件（stub — NovelFactory 不使用文件 artifact）。"""
    return JSONResponse(status_code=404, content={"detail": "Artifacts not supported"})


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduled Tasks — 完整 CRUD
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/scheduled-tasks")
async def list_scheduled_tasks():
    """定时任务列表。"""
    return {"tasks": [], "total": 0}


@router.post("/scheduled-tasks")
async def create_scheduled_task():
    """创建定时任务（stub）。"""
    return {"ok": True, "id": ""}


@router.get("/scheduled-tasks/{task_id}")
async def get_scheduled_task(task_id: str):
    """获取单个定时任务详情。"""
    return JSONResponse(status_code=404, content={"detail": "Task not found"})


@router.patch("/scheduled-tasks/{task_id}")
async def update_scheduled_task(task_id: str):
    """更新定时任务（stub）。"""
    return {"ok": True}


@router.delete("/scheduled-tasks/{task_id}")
async def delete_scheduled_task(task_id: str):
    """删除定时任务（stub）。"""
    return {"success": True}


@router.post("/scheduled-tasks/{task_id}/pause")
async def pause_scheduled_task(task_id: str):
    """暂停定时任务。"""
    return {"ok": True}


@router.post("/scheduled-tasks/{task_id}/resume")
async def resume_scheduled_task(task_id: str):
    """恢复定时任务。"""
    return {"ok": True}


@router.post("/scheduled-tasks/{task_id}/trigger")
async def trigger_scheduled_task(task_id: str):
    """手动触发定时任务立即执行。"""
    return {"ok": True}


@router.get("/scheduled-tasks/{task_id}/runs")
async def get_scheduled_task_runs(task_id: str):
    """获取定时任务的运行记录。"""
    return {"runs": [], "total": 0}


@router.get("/threads/{thread_id}/scheduled-tasks")
async def get_thread_scheduled_tasks(thread_id: str):
    """获取指定线程的定时任务。"""
    return {"tasks": [], "total": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# Channels
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/channels/providers")
async def list_channel_providers():
    """渠道 Provider 列表。"""
    return {
        "enabled": False,
        "providers": [
            {
                "provider": "feishu",
                "display_name": "飞书",
                "enabled": False,
                "configured": False,
                "connectable": True,
                "unavailable_reason": None,
                "auth_mode": "oauth",
                "connection_status": "disconnected",
                "credential_fields": [],
                "credential_values": {},
            }
        ],
    }


@router.get("/channels/connections")
async def list_channel_connections():
    """渠道连接列表。"""
    return {"connections": []}


@router.post("/channels/{provider}/connect")
async def connect_channel(provider: str):
    """发起渠道 OAuth 连接。"""
    return {
        "provider": provider,
        "mode": "code",
        "url": "",
        "code": "stub-connect-code",
        "instruction": "渠道功能未启用，请配置 LARK_APP_ID 和 LARK_APP_SECRET",
        "expires_in": 600,
    }


@router.post("/channels/{provider}/runtime-config")
async def configure_channel_runtime(provider: str):
    """配置渠道运行时参数（stub）。"""
    return {
        "provider": provider,
        "display_name": provider,
        "enabled": False,
        "configured": False,
        "connectable": True,
        "unavailable_reason": "Runtime config not supported in stub mode",
        "auth_mode": "oauth",
        "connection_status": "disconnected",
        "credential_fields": [],
        "credential_values": {},
    }


@router.delete("/channels/{provider}/runtime-config")
async def delete_channel_runtime_config(provider: str):
    """删除渠道运行时配置（stub）。"""
    return {
        "provider": provider,
        "display_name": provider,
        "enabled": False,
        "configured": False,
        "connectable": True,
        "unavailable_reason": None,
        "auth_mode": "oauth",
        "connection_status": "disconnected",
        "credential_fields": [],
        "credential_values": {},
    }


@router.delete("/channels/connections/{connection_id}")
async def delete_channel_connection(connection_id: str):
    """断开渠道连接。"""
    return JSONResponse(status_code=204, content=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Skills
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/skills")
async def list_skills():
    """Skill 列表（stub — NovelFactory 不使用 DeerFlow Skill 系统）。"""
    return {"skills": []}


@router.get("/skills/{skill_name}")
async def get_skill(skill_name: str):
    """获取单个 Skill 详情。"""
    return JSONResponse(status_code=404, content={"detail": "Skills not supported"})


@router.put("/skills/{skill_name}")
async def toggle_skill(skill_name: str):
    """启用/禁用 Skill（stub）。"""
    return {
        "name": skill_name,
        "description": "",
        "license": None,
        "category": "custom",
        "enabled": False,
        "editable": False,
    }


@router.post("/skills/install")
async def install_skill():
    """安装新 Skill（stub）。"""
    return {"success": False, "skill_name": "", "message": "Skills not supported"}


# ═══════════════════════════════════════════════════════════════════════════════
# MCP
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/mcp/config")
async def get_mcp_config():
    """获取 MCP 配置（stub — NovelFactory 不使用 MCP）。"""
    return {"mcp_servers": {}}


@router.put("/mcp/config")
async def update_mcp_config():
    """更新 MCP 配置（stub）。"""
    return {"mcp_servers": {}}


# ═══════════════════════════════════════════════════════════════════════════════
# Goal (线程目标)
# ═══════════════════════════════════════════════════════════════════════════════


@router.get("/threads/{thread_id}/goal")
async def get_thread_goal(thread_id: str):
    """获取线程目标（stub）。"""
    return {"goal": None}


@router.put("/threads/{thread_id}/goal")
async def set_thread_goal(thread_id: str):
    """设置线程目标（stub）。"""
    return {"goal": {}}


@router.delete("/threads/{thread_id}/goal")
async def delete_thread_goal(thread_id: str):
    """删除线程目标（stub）。"""
    return {"goal": None}
