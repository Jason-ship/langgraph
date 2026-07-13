"""Structured logger with optional file rotation and JSON output."""

from __future__ import annotations

import logging
import os

from novelfactory.config.constants import LOG_BACKUP_COUNT, LOG_MAX_BYTES

# Log rotation — 唯一来源: config.constants
_LOG_MAX_BYTES = LOG_MAX_BYTES
_LOG_BACKUP_COUNT = LOG_BACKUP_COUNT


def get_logger(name: str) -> logging.Logger:
    """Get a production-grade logger with JSON-like structured output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # v6.1: 统一从 settings 读取
    from novelfactory.config.settings import settings as _st

    level = getattr(
        logging,
        (_st.LOG_LEVEL or os.environ.get("LOG_LEVEL", "INFO")).upper(),
        logging.INFO,
    )
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = _st.LOG_FORMAT or os.environ.get("LOG_FORMAT", "text")
    if fmt == "json":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)-20s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_path = _st.NOVELFACTORY_LOG_PATH or os.environ.get("NOVELFACTORY_LOG_PATH", "")
    if log_path:
        from logging.handlers import RotatingFileHandler

        # Avoid duplicate RotatingFileHandler to the same file — multiple
        # handlers rotating the same log cause interleaved lines and truncated
        # records during rollover.
        already_handled = any(
            isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == log_path
            for h in logger.handlers
        )
        if not already_handled:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except (PermissionError, OSError):
                    # 非容器环境（如本地 CLI）无 /data 权限时跳过文件日志
                    return logger
            try:
                fh = RotatingFileHandler(
                    log_path, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT
                )
                fh.setLevel(level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except (OSError, FileNotFoundError, PermissionError):
                # 文件日志不可用时（非容器环境），仅返回 console logger
                pass

    return logger
