"""
NovelFactory 监控模块 — 结构化日志、审计日志、告警推送
=====================================================

功能:
  - StructuredLogger: 结构化 JSON 日志输出
  - AuditLogger: 审计日志持久化（操作记录、LLM 调用记录）
  - AlertManager: 告警推送（飞书 Webhook / 标准输出）
  - TraceContext: 全链路 trace_id 传播

用法:
    from novelfactory.utils.monitoring import (
        StructuredLogger, AuditLogger, AlertManager, TraceContext
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from novelfactory.config.settings import settings

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
_RETENTION_DAYS = 7
_FEISHU_WEBHOOK_TIMEOUT = 5  # 飞书 Webhook 超时（秒）


# ── Trace Context ──────────────────────────────────────────────────────────────

_trace_local = threading.local()


class TraceContext:
    """全链路 trace_id 上下文管理器。

    使用 thread-local 存储，支持嵌套上下文。

    用法:
        with TraceContext() as ctx:
            logger.info("processing", extra={"trace_id": ctx.trace_id})

        # 获取当前 trace_id
        trace_id = TraceContext.current()
    """

    @staticmethod
    def current() -> str | None:
        """获取当前线程的 trace_id。"""
        stack = getattr(_trace_local, "trace_stack", None)
        if stack:
            return stack[-1]
        return None

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self._prev_stack: list[str] | None = None

    def __enter__(self) -> TraceContext:
        self._prev_stack = getattr(_trace_local, "trace_stack", None)
        if self._prev_stack is None:
            _trace_local.trace_stack = [self.trace_id]
        else:
            _trace_local.trace_stack.append(self.trace_id)
        return self

    def __exit__(self, *args: Any) -> None:
        stack = getattr(_trace_local, "trace_stack", None)
        if stack:
            stack.pop()
        if not stack:
            try:
                del _trace_local.trace_stack
            except AttributeError:
                pass


# ── Structured JSON Logger ─────────────────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器。

    输出格式:
        {"timestamp": "2026-06-16T10:30:00.000Z", "level": "INFO",
         "logger": "novelfactory.xxx", "message": "...", "trace_id": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加 trace_id
        trace_id = TraceContext.current()
        if trace_id:
            log_entry["trace_id"] = trace_id

        # 添加额外字段
        if hasattr(record, "extra_fields") and record.extra_fields:
            log_entry.update(record.extra_fields)

        # 异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class StructuredLogger:
    """结构化日志管理器。

    支持两种输出模式:
      - json: 生产环境，输出 JSON 到 stdout
      - console: 开发环境，彩色输出到 stderr
    """

    _initialized = False

    @classmethod
    def setup(cls, level: str = "INFO", fmt: str = "console") -> None:
        """初始化全局日志配置。

        Args:
            level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
            fmt: 输出格式 (json/console)
        """
        if cls._initialized:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # 清除已有 handler
        root_logger.handlers.clear()

        if fmt == "json":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JsonFormatter())
        else:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        root_logger.addHandler(handler)

        # 降低第三方库日志级别
        for lib in ["httpx", "httpcore", "openai", "urllib3", "neo4j", "pymilvus"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

        cls._initialized = True


# ── Audit Logger ────────────────────────────────────────────────────────────────


class AuditLogger:
    """审计日志记录器。

    记录关键操作: LLM 调用、数据库写入、用户交互、系统事件。
    支持文件持久化和内存缓冲。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> AuditLogger:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._logger = logging.getLogger("novelfactory.audit")
        # v6.1: 从 settings 读取审计日志配置
        self._log_dir = settings.AUDIT_LOG_DIR
        self._enabled = settings.AUDIT_LOG_ENABLED

    def _rotate_log(self, log_path: Path) -> Path:
        """如果日志文件超过大小限制，自动轮转。"""
        if log_path.exists() and log_path.stat().st_size >= _MAX_FILE_SIZE:
            stem = log_path.stem  # audit_20260622
            counter = 1
            while True:
                new_path = log_path.with_name(f"{stem}_{counter:02d}{log_path.suffix}")
                if not new_path.exists():
                    return new_path
                counter += 1
        return log_path

    def _cleanup_old(self) -> None:
        """删除超过保留天数的旧日志文件。"""
        if not self._log_dir:
            return
        cutoff = datetime.now() - timedelta(days=_RETENTION_DAYS)
        log_dir = Path(self._log_dir)
        if not log_dir.exists():
            return
        for f in log_dir.glob("audit_*.jsonl*"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    self._logger.info("Removed old audit log: %s", f.name)
            except OSError:
                pass

    def _write(self, event_type: str, data: dict) -> None:
        """写入审计日志。"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "trace_id": TraceContext.current(),
            **data,
        }

        # 结构化日志输出
        self._logger.info(
            "AUDIT: %s | %s",
            event_type,
            json.dumps(data, ensure_ascii=False, default=str),
        )

        # 文件持久化（如果启用）
        if self._enabled and self._log_dir:
            try:
                log_dir = Path(self._log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                date_str = datetime.now().strftime("%Y%m%d")
                log_file = log_dir / f"audit_{date_str}.jsonl"
                log_file = self._rotate_log(log_file)
                with Path.open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                self._cleanup_old()
            except Exception as e:
                self._logger.warning("Audit file write failed: %s", e)

    def log_llm_call(
        self,
        model: str,
        phase: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        cost_cny: float,
    ) -> None:
        """记录 LLM 调用。"""
        self._write(
            "llm_call",
            {
                "model": model,
                "phase": phase,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": round(duration_ms, 2),
                "cost_cny": round(cost_cny, 6),
            },
        )

    def log_db_write(
        self, db: str, table: str, operation: str, record_count: int = 1
    ) -> None:
        """记录数据库写入。"""
        self._write(
            "db_write",
            {
                "database": db,
                "table": table,
                "operation": operation,
                "record_count": record_count,
            },
        )

    def log_user_action(
        self, action: str, thread_id: str = "", details: dict = None
    ) -> None:
        """记录用户操作。"""
        self._write(
            "user_action",
            {
                "action": action,
                "thread_id": thread_id,
                "details": details or {},
            },
        )

    def log_system_event(
        self, event: str, level: str = "info", details: dict = None
    ) -> None:
        """记录系统事件。"""
        self._write(
            "system_event",
            {
                "event": event,
                "level": level,
                "details": details or {},
            },
        )

    def log_error(self, error_type: str, message: str, context: dict = None) -> None:
        """记录错误。"""
        self._write(
            "error",
            {
                "error_type": error_type,
                "message": message,
                "context": context or {},
            },
        )


# ── Alert Manager ───────────────────────────────────────────────────────────────


class AlertManager:
    """告警管理器。

    支持多渠道推送: 飞书 Webhook、标准输出。
    告警级别: critical > error > warning > info
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> AlertManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._logger = logging.getLogger("novelfactory.alert")
        # v6.1: 从 settings 读取告警配置
        self._feishu_webhook = settings.FEISHU_ALERT_WEBHOOK or ""
        self._alert_level = settings.ALERT_MIN_LEVEL or "warning"
        self._http_client: Any = None

    def _should_alert(self, level: str) -> bool:
        """判断是否应该发送告警。"""
        levels = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}
        return levels.get(level, 0) >= levels.get(self._alert_level, 2)

    def _send_feishu(self, title: str, content: str, level: str) -> bool:
        """通过飞书 Webhook 发送告警。"""
        if not self._feishu_webhook:
            return False

        try:
            import urllib.request

            level_colors = {
                "critical": "red",
                "error": "red",
                "warning": "yellow",
                "info": "blue",
            }

            payload = json.dumps(
                {
                    "msg_type": "interactive",
                    "card": {
                        "header": {
                            "title": {
                                "tag": "plain_text",
                                "content": f"[{level.upper()}] {title}",
                            },
                            "template": level_colors.get(level, "blue"),
                        },
                        "elements": [
                            {"tag": "markdown", "content": content},
                            {
                                "tag": "note",
                                "elements": [
                                    {
                                        "tag": "plain_text",
                                        "content": f"NovelFactory | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                    }
                                ],
                            },
                        ],
                    },
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                self._feishu_webhook,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=_FEISHU_WEBHOOK_TIMEOUT)
            return True
        except Exception as e:
            self._logger.warning("Feishu alert send failed: %s", e)
            return False

    def alert(
        self,
        title: str,
        content: str = "",
        level: str = "warning",
        details: dict = None,
    ) -> None:
        """发送告警。

        Args:
            title: 告警标题
            content: 告警内容（支持 Markdown）
            level: 告警级别 (critical/error/warning/info)
            details: 附加详情
        """
        if not self._should_alert(level):
            return

        trace_id = TraceContext.current()
        full_content = content
        if trace_id:
            full_content += f"\n\nTrace ID: `{trace_id}`"
        if details:
            full_content += (
                f"\n\n详情: {json.dumps(details, ensure_ascii=False, default=str)}"
            )

        # 日志输出
        log_func = {
            "critical": self._logger.critical,
            "error": self._logger.error,
            "warning": self._logger.warning,
            "info": self._logger.info,
        }.get(level, self._logger.warning)
        log_func("ALERT [%s] %s: %s", level.upper(), title, content)

        # 飞书推送
        self._send_feishu(title, full_content, level)

    def alert_critical(
        self, title: str, content: str = "", details: dict = None
    ) -> None:
        """发送严重告警。"""
        self.alert(title, content, "critical", details)

    def alert_error(self, title: str, content: str = "", details: dict = None) -> None:
        """发送错误告警。"""
        self.alert(title, content, "error", details)


# ── Environment Detection ───────────────────────────────────────────────────────


def get_environment() -> str:
    """检测当前运行环境。

    Returns:
        "production" / "staging" / "development"
    """
    env = (settings.NOVELFACTORY_ENV or os.environ.get("NOVELFACTORY_ENV", "")).lower()
    if env in ("production", "prod"):
        return "production"
    if env in ("staging", "stage"):
        return "staging"
    return "development"


def is_production() -> bool:
    """是否为生产环境。"""
    return get_environment() == "production"


# ── Auto-setup ──────────────────────────────────────────────────────────────────


def setup_monitoring(
    log_level: str | None = None, log_format: str | None = None
) -> None:
    """一键初始化监控模块。

    根据环境自动选择日志格式:
      - production: JSON 格式
      - staging/development: console 格式
    """
    env = get_environment()

    if log_level is None:
        # v6.1: 从 settings 读取日志级别
        log_level: str = settings.LOG_LEVEL or os.environ.get("LOG_LEVEL", "INFO")

    if log_format is None:
        log_format = "json" if env == "production" else "console"

    StructuredLogger.setup(level=log_level, fmt=log_format)

    logger = logging.getLogger(__name__)
    logger.info(
        "Monitoring initialized: env=%s level=%s format=%s", env, log_level, log_format
    )
