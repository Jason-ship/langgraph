"""
NovelFactory Tools Proxy — 轻量 lark-cli HTTP 代理

职责：
  接收 api 容器的 HTTP 请求，通过 subprocess 调用 lark-cli 执行飞书操作。
  _LarkCLIEngine 通过 HTTP 调用此服务，无需修改任何代码。

架构：
  api 容器 → HTTP POST /lark/run → tools_proxy 容器 → subprocess lark-cli
                                      ↑
                              共享卷 lark_cli_config (持久化 token)

v6.1: subprocess 改为 asyncio.create_subprocess_exec，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NovelFactory Tools Proxy",
    description="lark-cli HTTP 代理（被 FeishuToolkit 调用）",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v6.1: 并发限制，防止大量 lark-cli 调用耗尽进程数
_LARK_SEMAPHORE = asyncio.Semaphore(10)


async def _run_command_async(
    cmd: list[str], timeout: int = 60, cwd: str | None = None
) -> dict[str, Any]:
    """异步执行 CLI 命令并返回标准化结果。

    v6.1: 使用 asyncio.create_subprocess_exec 替代 subprocess.run，
    避免在 async 端点中阻塞事件循环。
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "timestamp": datetime.now().isoformat(),
        }
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return {
            "success": False,
            "error": "Command timeout",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def _resolve_lark_result(raw: dict) -> dict:
    """标准化 lark-cli 返回结果。

    lark-cli 有时 exit code 非零但操作已成功（stderr 有 info 日志）。
    通过解析 stdout JSON 中的 ok/code 字段兜底。
    """
    if raw.get("success"):
        stdout_str = raw.get("stdout", "")
        if stdout_str:
            try:
                return {"success": True, "data": json.loads(stdout_str), "raw": raw}
            except (json.JSONDecodeError, ValueError):
                return {"success": True, "data": stdout_str, "raw": raw}
        return {"success": True, "data": None, "raw": raw}

    stdout_str = raw.get("stdout", "")
    stderr_str = raw.get("stderr", "")
    if stdout_str:
        try:
            parsed = json.loads(stdout_str)
            if isinstance(parsed, dict):
                ok = parsed.get("ok") or parsed.get("code") == 0
                if ok:
                    return {"success": True, "data": parsed, "raw": raw}
                identities = parsed.get("identities", {})
                if isinstance(identities, dict) and any(
                    isinstance(v, dict) and v.get("available")
                    for v in identities.values()
                ):
                    return {"success": True, "data": parsed, "raw": raw}
        except (json.JSONDecodeError, ValueError):
            pass
        if '"ok": true' in stdout_str or '"code":0' in stdout_str.replace(" ", ""):
            try:
                return {"success": True, "data": json.loads(stdout_str), "raw": raw}
            except (json.JSONDecodeError, ValueError):
                pass

    return {
        "success": False,
        "data": None,
        "error": stderr_str
        or raw.get("error", f"exit code {raw.get('returncode', 'unknown')}")
        or stdout_str[:500],
        "raw": raw,
    }


@app.get("/health")
async def health_check() -> dict:
    """健康检查。"""
    return {
        "status": "ok",
        "service": "NovelFactory Tools Proxy",
        "version": "1.1.0",
        "timestamp": datetime.now().isoformat(),
    }


class LarkRunRequest(BaseModel):
    """通用 lark-cli 命令执行请求 — 覆盖全部 21 域。"""

    domain: str = Field(..., description="lark-cli 域: im/docs/drive/...")
    command: str = Field(..., description="子命令: +messages-send/+create/...")
    args: list[str] = Field(
        default_factory=list,
        description="参数列表: ['--chat-id', 'oc_xxx', ...]",
    )
    content: str | None = Field(
        default=None,
        description="大内容（替代临时文件模式）",
    )
    content_suffix: str = Field(
        default=".md",
        description="内容文件扩展名",
    )
    timeout: int = Field(default=60, description="超时秒数")
    format_json: bool = Field(
        default=True,
        description="是否追加 --format json",
    )


@app.post("/lark/run")
async def lark_run(req: LarkRunRequest) -> dict:
    """通用 lark-cli 命令执行端点。"""
    cmd = ["lark-cli", req.domain, req.command]
    cmd.extend(req.args)
    run_cwd = None
    tmp_path = ""

    if req.content:
        suffix = req.content_suffix if req.content_suffix.startswith(".") else f".{req.content_suffix}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False
        ) as f:
            f.write(req.content)
            tmp_path = f.name
        # lark-cli v2 --content @file 要求相对路径
        tmp_dir = os.path.dirname(tmp_path)
        tmp_name = os.path.basename(tmp_path)
        cmd.extend(["--content", f"@./{tmp_name}"])
        run_cwd = tmp_dir

    if req.format_json and "--format" not in req.args and "--json" not in req.args:
        cmd.extend(["--format", "json"])

    try:
        async with _LARK_SEMAPHORE:
            raw = await _run_command_async(cmd, timeout=req.timeout, cwd=run_cwd)
        result = _resolve_lark_result(raw)
        return result
    finally:
        if req.content and tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/lark/run-for-json")
async def lark_run_for_json(req: LarkRunRequest) -> dict:
    """执行并返回 JSON data。"""
    result = await lark_run(req)
    if result.get("success") and isinstance(result.get("data"), dict):
        return result["data"]
    return {}
