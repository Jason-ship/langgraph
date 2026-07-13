"""NovelFactory Middleware 系统 — 可组合的横切关注点处理器。

使用方式:
    from novelfactory.middleware import get_middleware_chain, with_middleware

    chain = get_middleware_chain()
    graph.add_node("writing_crew", with_middleware(build_writing_crew(), chain))
"""

from __future__ import annotations

from novelfactory.middleware.base import MiddlewareChain
from novelfactory.middleware.large_file_storage import LargeFileStorageMiddleware
from novelfactory.middleware.skill_injection import SkillInjectionMiddleware
from novelfactory.middleware.summarization import SummarizationMiddleware
from novelfactory.middleware.todo_list import TodoListMiddleware
from novelfactory.middleware.wrapper import with_middleware

__all__ = [
    "get_middleware_chain",
    "with_middleware",
    "MiddlewareChain",
]

_middleware_chain: MiddlewareChain | None = None


def get_middleware_chain() -> MiddlewareChain:
    """获取全局中间件链（懒初始化单例）。

    v6.1: 将已实现的 4 个中间件挂载到链上。
    """
    global _middleware_chain
    if _middleware_chain is None:
        from novelfactory.skills.loader import SkillLoader

        chain = MiddlewareChain()
        # SkillInjectionMiddleware 需要 SkillLoader 实例
        loader = SkillLoader()
        chain.add(SkillInjectionMiddleware(loader))
        chain.add(LargeFileStorageMiddleware())
        chain.add(SummarizationMiddleware())
        chain.add(TodoListMiddleware())
        _middleware_chain = chain
    return _middleware_chain
