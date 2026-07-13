"""Timeout guard for LLM calls."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the timeout threshold."""


def with_timeout(
    seconds: float, default: T
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: run a function with a timeout, return default on timeout.

    Uses threading.Event on ALL platforms (Windows + Unix).
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import threading

            result = [default]
            exc_info = [None]
            stopped = threading.Event()

            # 线程目标函数需要捕获所有异常以传播到主线程；这是标准的安全模式
            def target() -> None:
                try:
                    # RunnableWithFallbacks（from with_fallbacks()）不支持 __call__，
                    # 需用 .invoke() 调用。兼容普通 callable 和 Runnable。
                    if hasattr(func, "invoke"):
                        result[0] = func.invoke(*args, **kwargs)
                    else:
                        result[0] = func(*args, **kwargs)
                except Exception as e:
                    exc_info[0] = e
                finally:
                    stopped.set()

            t = threading.Thread(target=target, daemon=True)
            t.start()
            if not stopped.wait(timeout=seconds):
                return default
            if exc_info[0] is not None:
                raise exc_info[0]
            return result[0]

        return wrapper

    return decorator
