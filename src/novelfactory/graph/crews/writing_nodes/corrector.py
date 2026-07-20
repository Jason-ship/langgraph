"""纠偏器节点（v7.3 新增）— Bottom-up 事件注入。

当章节连续重写仍无法通过评审时，不走"强制 PASS"也不等人工介入，
而是生成意外事件修正后续大纲，打破 rewrite 死循环。

参考 StoryBox (AAAI 2026) 的 Abnormal Behavior 设计理念（沙盒角色持续概率异常），
但核心机制不同：
  - StoryBox: 每一步 30% 概率触发的持续性预防机制（独立角色级）
  - 本实现: rewrite 用尽后一次性注入的应急纠正机制（大纲级）
"""

from __future__ import annotations

import json
from typing import Any

from novelfactory.agents.infra import async_llm_call_with_retry, get_logger

logger = get_logger(__name__)


def _build_corrector_prompt(
    failed_chapter: int,
    failure_analysis: str,
    character_states: dict,
    world_setting: str,
) -> str:
    """构建纠偏器 prompt。"""
    char_summary = json.dumps(
        {k: str(v)[:100] for k, v in (character_states or {}).items()},
        ensure_ascii=False,
        indent=2,
    )

    return f"""第 {failed_chapter} 章连续重写未通过评审。

## 失败分析
{failure_analysis}

## 当前角色状态
{char_summary[:2000]}

## 世界观设定
{world_setting[:2000] if world_setting else "（无）"}

请生成 3 个意外事件来打破当前写作僵局，让故事从卡住的地方走出来：

要求：
- 至少 1 个是角色主动行为（角色主动改变策略/选择/认知）
- 至少 1 个是外部突发事件（环境变化/新角色介入/已有角色做出不可预知的行动）
- 所有事件必须符合已有世界观设定
- 事件要让故事产生新的可能性方向，而非只是"换种方式写同一件事"

输出格式（每行一个事件）：
事件 | 类型(主动/外部) | 简述(20字内) | 详细描述(50-100字) | 如何改变剧情走向
"""


async def corrector_node(state: dict[str, Any]) -> dict[str, Any]:
    """纠偏器节点 — 当 rewrite 次数耗尽且分数低时注入意外事件（async）。

    在 verdict_engine 路由到 REWRITE 前检查：如果已经 rewrite_exhausted，
    且分数低，则先生成意外事件修正后续大纲，再走 rewrite。

    Returns:
        dict 含 corrective_events（list[str]）+ corrector_applied（bool）
    """
    loop_count = int(state.get("loop_count", 0))
    max_rewrite = 5
    final_score = float(state.get("final_score", 0))

    # 只在 rewrite 次数用尽 + 低分时触发
    if loop_count < max_rewrite or final_score >= 55.0:
        return {"corrective_events": [], "corrector_applied": False}

    logger.info(
        "[纠偏器] 触发: ch%d loop=%d score=%.1f 已超过阈值",
        int(state.get("current_chapter", 0)),
        loop_count,
        final_score,
    )

    # 构建失败分析
    failure_analysis = (
        f"连续 {loop_count} 次重写未通过评审。"
        f"最终评分 {final_score:.0f}/100。"
        f"评审意见: {state.get('verdict_feedback', '')[:500]}"
    )

    prompt = _build_corrector_prompt(
        failed_chapter=int(state.get("chapter_number", 0)),
        failure_analysis=failure_analysis,
        character_states=state.get("character_states", {}),
        world_setting=state.get("story_outline", ""),
    )

    try:
        from novelfactory.config.llm import get_worker_llm

        llm = get_worker_llm()
        response = await async_llm_call_with_retry(llm, prompt, step_name="corrector")
        raw = response.content if hasattr(response, "content") else str(response)
        events = _parse_events(raw)
        logger.info("[纠偏器] 生成 %d 个意外事件", len(events))
        return {"corrective_events": events, "corrector_applied": True}
    except Exception as e:
        logger.warning("[纠偏器] 生成失败: %s", e)
        return {"corrective_events": [], "corrector_applied": False}


def _parse_events(raw: str) -> list[str]:
    """从 LLM 输出中解析事件列表。"""
    lines = raw.strip().split("\n")
    events = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if " | " in line:
            events.append(line)
        elif line.startswith("- ") or line.startswith("* "):
            events.append(line[2:])
    return events[:5]  # 最多 5 个
