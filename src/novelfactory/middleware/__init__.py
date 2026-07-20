"""NovelFactory Middleware 系统 — 可组合的横切关注点处理器。

使用方式:
    from novelfactory.middleware import get_middleware_chain, with_middleware

    chain = get_middleware_chain()
    graph.add_node("writing_crew", with_middleware(build_writing_crew(), chain))
"""

from __future__ import annotations

from novelfactory.middleware.base import MiddlewareChain
from novelfactory.middleware.context_compaction import (
    ThreadCompactionResult,
    compact_thread_context,
)
from novelfactory.middleware.guardrails import (
    BuiltinGuardrailProvider,
    GuardrailDecision,
    GuardrailProvider,
    GuardrailReason,
    GuardrailRequest,
    check_tool_guardrail,
)
from novelfactory.middleware.large_file_storage import LargeFileStorageMiddleware
from novelfactory.middleware.loop_detector import LoopDetector, check_loop
from novelfactory.middleware.safety_detectors import (
    AnthropicRefusalDetector,
    GeminiSafetyDetector,
    OpenAICompatibleContentFilterDetector,
    SafetyTermination,
    SafetyTerminationDetector,
    default_detectors,
)
from novelfactory.middleware.safety_middleware import check_safety_termination as check_safety
from novelfactory.middleware.skill_injection import SkillInjectionMiddleware
from novelfactory.middleware.summarization import SummarizationMiddleware
from novelfactory.middleware.token_usage import AggregatedUsage, TokenUsage, TokenUsageTracker
from novelfactory.middleware.todo_list import TodoListMiddleware
from novelfactory.middleware.wrapper import with_middleware

__all__ = [
    "get_middleware_chain",
    "with_middleware",
    "MiddlewareChain",
    "LoopDetector",
    "check_loop",
    "SafetyTermination",
    "SafetyTerminationDetector",
    "OpenAICompatibleContentFilterDetector",
    "AnthropicRefusalDetector",
    "GeminiSafetyDetector",
    "default_detectors",
    "check_safety",
    "ThreadCompactionResult",
    "compact_thread_context",
    "GuardrailRequest",
    "GuardrailReason",
    "GuardrailDecision",
    "GuardrailProvider",
    "BuiltinGuardrailProvider",
    "check_tool_guardrail",
    "TokenUsage",
    "AggregatedUsage",
    "TokenUsageTracker",
]

_middleware_chain: MiddlewareChain | None = None


def get_middleware_chain() -> MiddlewareChain:
    """获取全局中间件链（懒初始化单例）。

    v6.1: 将已实现的 4 个中间件挂载到链上。
    v7.0: 新增安全检测和循环检测中间件。
    """
    global _middleware_chain
    if _middleware_chain is None:
        from novelfactory.skills.loader import SkillLoader

        chain = MiddlewareChain()
        loader = SkillLoader()
        chain.add(SkillInjectionMiddleware(loader))
        chain.add(LargeFileStorageMiddleware())
        chain.add(SummarizationMiddleware())
        chain.add(TodoListMiddleware())
        _middleware_chain = chain
    return _middleware_chain
