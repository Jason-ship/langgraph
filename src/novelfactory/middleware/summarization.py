"""SummarizationMiddleware — 智能上下文压缩中间件。

当消息数超过 trigger 阈值时，使用 LLM 摘要中间消息，
保留最近 keep 条原文，替代简单截断。

v6.1: _summarize 接入 LLM 生成语义摘要，失败时降级为字符串拼接。
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, SystemMessage

from novelfactory.middleware.base import Middleware

logger = logging.getLogger(__name__)


class SummarizationMiddleware(Middleware):
    """智能上下文压缩中间件。

    参考 DeepAgent 的 trigger/keep 配置模式。

    Attributes:
        trigger: 消息数阈值，超过时触发压缩（默认 100）
        keep: 保留的最新消息数（默认 20）
    """

    def __init__(self, trigger: int = 100, keep: int = 20):
        self.trigger = trigger
        self.keep = keep
        self._llm = None

    def _get_llm(self):
        """延迟获取 LLM 实例。"""
        if self._llm is None:
            try:
                from novelfactory.config.llm import get_reviewer_llm

                self._llm = get_reviewer_llm()
            except Exception as e:
                logger.warning("[summarization] LLM 初始化失败，降级为规则摘要: %s", e)
                self._llm = False  # 标记不可用
        return self._llm if self._llm is not False else None

    def before_node(self, state: dict, config: dict) -> dict | None:
        messages = state.get("messages", [])
        if len(messages) <= self.trigger:
            return None

        # 分离 system message
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= self.keep:
            return None

        # 需要压缩的消息
        to_summarize = other_msgs[: -self.keep]
        to_keep = other_msgs[-self.keep :]

        # 生成摘要
        summary_text = self._summarize(to_summarize)

        # 构造摘要消息
        summary_message = AIMessage(
            content=f"[上下文摘要] 以下是对前面 {len(to_summarize)} 条消息的摘要：\n{summary_text}",
            name="system",
        )

        # 重组: system + 摘要 + 最近 keep 条
        compressed = list(system_msgs) + [summary_message] + list(to_keep)

        logger.info(
            "[summarization] Compressed %d → %d messages (trigger=%d, keep=%d)",
            len(messages),
            len(compressed),
            self.trigger,
            self.keep,
        )

        return {"messages": compressed}

    def _summarize(self, messages: list) -> str:
        """使用 LLM 生成消息摘要，失败时降级为规则拼接。"""
        # 提取关键内容
        content_parts = []
        for m in messages[-10:]:  # 只摘要最近的部分
            if hasattr(m, "content") and m.content:
                text = str(m.content)[:200]
                name = getattr(m, "name", "")
                if name:
                    content_parts.append(f"[{name}]: {text}")
                else:
                    content_parts.append(text[:200])

        if not content_parts:
            return f"（{len(messages)} 条历史消息）"

        # v6.1: 尝试使用 LLM 生成语义摘要
        llm = self._get_llm()
        if llm:
            try:
                prompt = (
                    "请将以下写作过程中的对话摘要为简洁的上下文回顾（300字以内），"
                    "保留关键决策、角色状态和剧情进展：\n\n" + "\n".join(content_parts)
                )
                response = llm.invoke(prompt)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )
                summary = str(content).strip()
                if summary:
                    if len(summary) > 2000:
                        summary = summary[:2000] + "..."
                    logger.debug(
                        "[summarization] LLM 摘要生成成功 (%d chars)", len(summary)
                    )
                    return summary
            except Exception as e:
                logger.warning("[summarization] LLM 摘要失败，降级为规则: %s", e)

        # 降级：规则拼接
        summary = " | ".join(content_parts)
        if len(summary) > 2000:
            summary = summary[:2000] + "..."
        return summary
