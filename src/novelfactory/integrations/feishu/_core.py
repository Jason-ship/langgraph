"""飞书工具箱核心层 — 常量、结果类型与底层 CLI 调用引擎。

本模块是 feishu 子包的基石，提供：
  - 模块常量: ``_LARK_PROXY_URL`` / ``_LARK_TIMEOUT`` / ``_LARK_DOC_TIMEOUT`` / ``_MAX_DOC_CHARS``
  - 结果类型: ``LarkResult`` / ``LarkListResult``
  - 调用引擎: ``_LarkCLIEngine``（通过 tools-proxy FastAPI 服务执行 lark-cli 命令）

注意：``_LarkCLIEngine.run()`` 引用本模块级 ``_LARK_PROXY_URL`` 全局变量，
``FeishuToolkit.__init__`` 通过 ``_core._LARK_PROXY_URL = ...`` 更新该变量。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────
from novelfactory.config.constants import (  # noqa: E402
    FEISHU_LARK_DOC_TIMEOUT,
    FEISHU_LARK_TIMEOUT,
    FEISHU_MAX_DOC_CHARS,
)

# v6.1: 统一从 settings 读取
try:
    from novelfactory.config.settings import settings as _st

    _LARK_PROXY_URL = _st.lark_proxy_url or os.environ.get(
        "LARK_PROXY_URL", "http://172.28.0.1:5004"
    )
except ImportError:
    _LARK_PROXY_URL = os.environ.get("LARK_PROXY_URL", "http://172.28.0.1:5004")
_LARK_TIMEOUT = FEISHU_LARK_TIMEOUT  # 默认超时（秒）— 唯一来源: config.constants
_LARK_DOC_TIMEOUT = FEISHU_LARK_DOC_TIMEOUT  # 文档操作超时 — 唯一来源: config.constants
_MAX_DOC_CHARS = (
    FEISHU_MAX_DOC_CHARS  # 文档单次上传最大字符数 — 唯一来源: config.constants
)


# ── 结果类型 ─────────────────────────────────────────────────────────────────


@dataclass
class LarkResult:
    """标准化的 lark-cli 命令执行结果。"""

    success: bool
    data: Any | None = None
    error: str = ""
    raw: dict | None = None

    def __bool__(self) -> bool:
        return self.success


@dataclass
class LarkListResult(LarkResult):
    """带分页信息的列表结果。"""

    items: list = field(default_factory=list)
    has_more: bool = False
    page_token: str = ""


# ── 底层 CLI 调用引擎 ────────────────────────────────────────────────────────


class _LarkCLIEngine:
    """lark-cli HTTP 代理调用引擎（通过 tools-proxy FastAPI 服务）。

    职责：
      - 统一的命令组装和 HTTP 调用
      - 自动 reauth
      - 错误标准化
    """

    _reauth_done = False  # 进程级标记，只自动 reauth 一次
    _client: httpx.Client | None = None

    @classmethod
    def _get_client(cls) -> httpx.Client:
        if cls._client is None:
            cls._client = httpx.Client(timeout=httpx.Timeout(120.0))
        return cls._client

    # ── 认证管理 ────────────────────────────────────────────────────────

    @classmethod
    def set_auth_token(cls, token: str) -> None:
        """设置用户/机器人凭证。"""
        cls._reauth_done = False

    @classmethod
    def _auto_reauth(cls) -> bool:
        """自动刷新飞书 token（仅限 user 身份设备流）。

        实际环境中使用 bot 模式（defaultAs: bot），
        bot 身份通过 app_id + app_secret 自动获取 tenant token，
        无需用户设备流授权。因此此方法为 no-op。
        """
        if cls._reauth_done:
            return False
        cls._reauth_done = True

        logger.info("[feishu] reauth: bot 模式下跳过 user 设备流授权")
        return False

    # ── 核心执行 ────────────────────────────────────────────────────────

    @classmethod
    def run(
        cls,
        args: list[str],
        *,
        timeout: int = _LARK_TIMEOUT,
        format_json: bool = True,
        max_retry: int = 1,
        raw_output: bool = False,
    ) -> LarkResult:
        """通过 tools-proxy 执行 lark-cli 命令。

        Args:
            args: 参数列表（不含前缀），例 ["im", "messages-send", "--receive-id", "oc_xxx"]
            timeout: 超时秒数
            format_json: 是否追加 --format json
            max_retry: token 过期后重试次数
            raw_output: 是否返回原始输出

        Returns:
            LarkResult
        """
        if len(args) < 2:
            return LarkResult(success=False, error=f"invalid args: {args}")

        domain = args[0]
        command = args[1]
        cli_args = list(args[2:])

        for attempt in range(max_retry + 1):
            try:
                client = cls._get_client()
                resp = client.post(
                    f"{_LARK_PROXY_URL}/lark/run",
                    json={
                        "domain": domain,
                        "command": command,
                        "args": cli_args,
                        "format_json": format_json,
                        "timeout": timeout,
                    },
                )
                proxy_result = resp.json()

                success = proxy_result.get("success", False)
                data = proxy_result.get("data")
                error = proxy_result.get("error", "")
                raw = proxy_result.get("raw")

                if raw_output:
                    if isinstance(data, str):
                        return LarkResult(success=success, data=data)
                    if raw and isinstance(raw, dict):
                        stdout = raw.get("stdout", "")
                        return LarkResult(
                            success=(raw.get("returncode", -1) == 0),
                            data=stdout,
                        )
                    return LarkResult(success=success, data=str(data or ""))

                parsed = LarkResult(
                    success=success,
                    data=data,
                    error=error,
                    raw=raw,
                )

                if not success and attempt < max_retry:
                    err_str = ""
                    if raw and isinstance(raw, dict):
                        err_str = str(raw.get("stderr", ""))
                    if isinstance(error, dict):
                        err_str += str(error.get("message", ""))
                    elif isinstance(error, str):
                        err_str = error
                    is_auth_err = any(
                        kw in err_str.lower()
                        for kw in (
                            "authentication",
                            "authorization",
                            "token_missing",
                            "app_scope_not_applied",
                            "access_denied",
                            "unauthorized",
                            "need_user_authorization",
                        )
                    )
                    if is_auth_err:
                        if cls._auto_reauth():
                            logger.info("[feishu] reauth 完成，重试命令...")
                            continue
                return parsed

            except httpx.ConnectError:
                return LarkResult(
                    success=False,
                    error=f"tools-proxy 不可达: {_LARK_PROXY_URL}",
                )
            except httpx.ReadTimeout:
                return LarkResult(
                    success=False,
                    error=f"timeout after {timeout}s",
                )
            except Exception as e:
                logger.warning("[feishu] HTTP 错误: %s", e)
                return LarkResult(success=False, error=str(e))

        return LarkResult(success=False, error="max retries exhausted")

    @classmethod
    def run_for_json(
        cls,
        args: list[str],
        *,
        timeout: int = _LARK_TIMEOUT,
    ) -> dict:
        """执行并返回 JSON data（通过 /lark/run-for-json 端点）。"""
        if len(args) < 2:
            return {}

        client = cls._get_client()
        try:
            resp = client.post(
                f"{_LARK_PROXY_URL}/lark/run-for-json",
                json={
                    "domain": args[0],
                    "command": args[1],
                    "args": list(args[2:]),
                    "timeout": timeout,
                },
            )
            proxy_result = resp.json()
            if proxy_result.get("success") and isinstance(
                proxy_result.get("data"), dict
            ):
                return proxy_result["data"]
            return {}
        except Exception as e:
            logger.warning("[feishu] run_for_json 错误: %s", e)
            return {}

    @classmethod
    def _run_with_tmpfile(
        cls,
        args: list[str],
        content: str,
        suffix: str = ".md",
        *,
        timeout: int = _LARK_TIMEOUT,
    ) -> LarkResult:
        """通过 HTTP body 传递大内容（替代临时文件模式）。"""
        if len(args) < 2:
            return LarkResult(success=False, error=f"invalid args: {args}")

        client = cls._get_client()
        try:
            resp = client.post(
                f"{_LARK_PROXY_URL}/lark/run",
                json={
                    "domain": args[0],
                    "command": args[1],
                    "args": list(args[2:]),
                    "content": content,
                    "content_suffix": suffix,
                    "format_json": True,
                    "timeout": timeout,
                },
            )
            proxy_result = resp.json()
            return LarkResult(
                success=proxy_result.get("success", False),
                data=proxy_result.get("data"),
                error=proxy_result.get("error", ""),
                raw=proxy_result.get("raw"),
            )
        except httpx.ConnectError:
            return LarkResult(
                success=False,
                error=f"tools-proxy 不可达: {_LARK_PROXY_URL}",
            )
        except Exception as e:
            return LarkResult(success=False, error=str(e))
