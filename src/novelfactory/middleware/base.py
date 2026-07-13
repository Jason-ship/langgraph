"""Middleware 基类 + MiddlewareChain 链管理器。

使用方式:
    chain = MiddlewareChain()
    chain.add(SkillInjectionMiddleware(loader))
    chain.add(SummarizationMiddleware())
    updates = chain.execute_before(state, config)
    prompt = chain.apply_prompt_modifiers(prompt, state, config)
"""

from __future__ import annotations

from abc import ABC


class Middleware(ABC):
    """Middleware 基类，提供三个钩子接口。

    子类按需重写以下方法：
      - before_node:  节点执行前，返回 dict 更新 state 或 None
      - after_node:   节点执行后，返回 dict 更新 state 或 None
      - modify_system_prompt: 修改 system prompt
    """

    def before_node(self, state: dict, config: dict) -> dict | None:
        """节点执行前钩子。

        Args:
            state: 当前 state
            config: 运行时配置

        Returns:
            dict: 合并到 state 的更新；None: 不操作；False: 短路后续中间件
        """
        return None

    def after_node(self, state: dict, result: dict, config: dict) -> dict | None:
        """节点执行后钩子。

        Args:
            state: 执行前 state
            result: 节点返回值
            config: 运行时配置

        Returns:
            dict: 合并到 state 的更新；None: 不操作
        """
        return None

    def modify_system_prompt(self, prompt: str, state: dict, config: dict) -> str:
        """修改 system prompt。

        Args:
            prompt: 原始 system prompt
            state: 当前 state
            config: 运行时配置

        Returns:
            修改后的 prompt
        """
        return prompt


class MiddlewareChain:
    """中间件链管理器。

    中间件按 add 顺序执行。before_node 支持短路（返回 False）。
    """

    def __init__(self):
        self._middlewares: list[Middleware] = []

    def add(self, middleware: Middleware):
        """添加中间件到链尾。"""
        self._middlewares.append(middleware)

    def execute_before(self, state: dict, config: dict | None = None) -> dict:
        """按序执行所有中间件的 before_node 钩子。

        Args:
            state: 当前 state
            config: 运行时配置

        Returns:
            所有中间件返回的更新合并后的 dict
        """
        if config is None:
            config = {}
        updates: dict = {}
        for m in self._middlewares:
            result = m.before_node(state, config)
            if result is False:
                break
            if result:
                updates.update(result)
        return updates

    def execute_after(
        self, state: dict, result: dict, config: dict | None = None
    ) -> dict:
        """按序执行所有中间件的 after_node 钩子。"""
        if config is None:
            config = {}
        updates: dict = {}
        for m in self._middlewares:
            r = m.after_node(state, result, config)
            if r:
                updates.update(r)
        return updates

    def apply_prompt_modifiers(
        self, prompt: str, state: dict, config: dict | None = None
    ) -> str:
        """按序应用所有中间件的 modify_system_prompt。"""
        if config is None:
            config = {}
        for m in self._middlewares:
            prompt = m.modify_system_prompt(prompt, state, config)
        return prompt

    @property
    def count(self) -> int:
        return len(self._middlewares)
