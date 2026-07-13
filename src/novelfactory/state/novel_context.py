"""运行时上下文定义 — 与图状态完全隔离。

NovelContext 作为 ``StateGraph(context_schema=NovelContext)`` 传入，
所有节点通过 ``runtime: Runtime[NovelContext]`` 参数注入访问。

包含不可变的运行元数据：
  - thread_id:   当前执行线程
  - user_id:     请求用户
  - project_id:  关联项目
  - request_id:  请求追踪 ID
  - lark_config: 飞书集成配置（可选）

与 ``NovelFactoryState`` 的区别：
  - State 是图状态，随时间变化、持久化到检查点
  - Context 是运行上下文，不可变、不持久化
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class NovelContext(TypedDict):
    """运行时上下文 — 与图状态完全隔离。

    属性均为只读（Runtime 保证不可变），节点可安全读取但不允许修改。
    """

    thread_id: str
    """当前 LangGraph 执行线程 ID。"""

    user_id: str
    """发起请求的用户标识。"""

    project_id: str
    """关联的小说项目 ID。"""

    request_id: str
    """请求追踪 ID（用于日志关联和审计）。"""

    lark_config: dict[str, Any] | None
    """飞书集成配置（可从环境变量或请求参数注入），None 表示不使用飞书。"""
