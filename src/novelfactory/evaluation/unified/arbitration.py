"""分歧仲裁模块（v8.2）— 替代多轮辩论。

主评审输出五视角分歧点后，仅当存在严重分歧时触发 1 次轻量 LLM 仲裁调用：
三角色各 1 条反驳立场 → 裁决（采纳/折中）→ 分数修正。
保留"辩论"的多角色对抗与收敛裁决优点，消除固定多轮循环。
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra import async_llm_call_with_retry, get_logger
from novelfactory.evaluation.unified.prompts import build_arbitration_prompt

logger = get_logger(__name__)

_RE_NEW = re.compile(r"\[分数修正\]\s*final:\s*[\d.]+\s*->\s*([\d.]+)")
_RE_ACTION = re.compile(r"\[备注\]\s*(PASS|REFINE|REWRITE)", re.IGNORECASE)


def parse_arbitration(raw: str) -> dict | None:
    """解析仲裁输出；提取新分数与建议动作。"""
    if not raw:
        return None
    m = _RE_NEW.search(raw)
    if not m:
        return None
    m_action = _RE_ACTION.search(raw)
    return {
        "new_score": float(m.group(1)),
        "action": m_action.group(1).upper() if m_action else "REFINE",
    }


async def arbitrate(
    llm: BaseChatModel,
    *,
    disagreements: list[str],
    old_score: float,
) -> float:
    """分歧仲裁：输入分歧清单，返回仲裁后分数（失败返回原分）。"""
    if not disagreements:
        return old_score
    prompt = build_arbitration_prompt(disagreements=disagreements, old_score=old_score)
    try:
        response = await async_llm_call_with_retry(
            llm.ainvoke, prompt, step_name="unified_arbitration"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_arbitration(raw)
        if parsed:
            logger.info("[unified_arbitration] %s -> %s", old_score, parsed["new_score"])
            return parsed["new_score"]
    except Exception as e:
        logger.warning("[unified_arbitration] failed: %s", e)
    return old_score
