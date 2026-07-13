"""Pipeline 基础管理器 — 提取三 Manager 共享的底层操作。

将以下重复模式统一至此：
  1. PG cursor 生命周期管理（execute / fetchone / fetchall，自动 close）
  2. LLM 懒加载（``_get_llm`` 统一入口）
  3. LLM JSON 提取（``_llm_invoke_json`` — invoke + regex + json.loads）

所有 pipeline 子管理器继承 ``BaseManager`` 即可获得这些能力，
无需各自重复 cursor 样板代码和 LLM 初始化逻辑。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


class BaseManager:
    """Pipeline 子管理器基类 — 封装 PG cursor 操作和 LLM 调用样板。

    子类只需实现业务逻辑，通过 ``_execute`` / ``_fetchone`` / ``_fetchall``
    操作数据库，通过 ``_llm_invoke_json`` 调用 LLM 并提取 JSON 结果。

    Example::

        class MyManager(BaseManager):
            def get_count(self, project: str) -> int:
                row = self._fetchone(
                    "SELECT COUNT(*) FROM my_table WHERE project_name = %s",
                    (project,),
                )
                return row[0] if row else 0
    """

    def __init__(self, pg_conn: Any) -> None:
        self._conn = pg_conn
        self._llm: Any = None

    # ── PG cursor helpers ───────────────────────────────────────────────

    def _execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        """执行 INSERT/UPDATE/DELETE，自动管理 cursor 生命周期。

        使用 try/finally 确保 cursor 在异常时也能被关闭，
        修复原有代码中 cursor 泄漏的隐患。
        """
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
        finally:
            cur.close()

    def _fetchone(self, sql: str, params: Sequence[Any] | None = None) -> tuple | None:
        """执行查询并返回单行（或 None），自动关闭 cursor。"""
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
            return cur.fetchone()
        finally:
            cur.close()

    def _fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
        """执行查询并返回所有行，自动关闭 cursor。"""
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
            return cur.fetchall()
        finally:
            cur.close()

    # ── LLM helpers ─────────────────────────────────────────────────────

    def _get_llm(self) -> Any:
        """获取 reviewer LLM（懒加载，统一到 config.llm.get_reviewer_llm 工厂）。

        所有子管理器共享此方法，无需各自重复定义。
        """
        if self._llm is None:
            from novelfactory.config.llm import get_reviewer_llm

            self._llm = get_reviewer_llm()
        return self._llm

    @staticmethod
    def _extract_json(text: str, is_array: bool = False) -> Any:
        """从 LLM 响应文本中提取 JSON（对象或数组）。

        Args:
            text: LLM 响应文本
            is_array: True 提取 JSON 数组 ``[...]``，False 提取 JSON 对象 ``{...}``

        Returns:
            解析后的 Python 对象，解析失败返回 None
        """
        pattern = _JSON_ARRAY_RE if is_array else _JSON_OBJECT_RE
        match = pattern.search(text)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def _llm_invoke_json(
        self,
        prompt: str,
        is_array: bool = False,
        llm: Any = None,
    ) -> Any:
        """调用 LLM 并从响应中提取 JSON。

        将 ``llm.invoke → resp.content → re.search → json.loads`` 四步合一，
        统一错误处理。失败时返回 None，由调用方决定降级策略。

        Args:
            prompt: 发送给 LLM 的完整 prompt
            is_array: 期望返回 JSON 数组还是对象
            llm: 可选的自定义 LLM 实例（默认使用 ``_get_llm()``）

        Returns:
            解析后的 JSON 对象/数组，失败返回 None
        """
        target_llm = llm or self._get_llm()
        try:
            resp = target_llm.invoke([("user", prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            return self._extract_json(text, is_array=is_array)
        except Exception as e:
            logger.warning("[%s] LLM JSON invoke failed: %s", type(self).__name__, e)
            return None
