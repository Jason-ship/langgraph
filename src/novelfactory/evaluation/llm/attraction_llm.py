"""LLM 吸引力专家团队评审模块（v1.0）。

三位 LLM 专家**并行独立**评审章节的吸引力/可读性（与 old_reader_llm / ai_style_llm 同构）：
  1. 番茄爆款编辑（FanqieEditor）— 节奏与爽点 / 开篇三阶段 / E-V-R 情绪节奏 / 章尾钩子（权重 0.4）
  2. 代入感专家（ImmersionExpert）— 视角贴紧 / 情绪身体化呈现 / 细节真实 / 心理共鸣（权重 0.3）
  3. 反AI味审计师（AIStyleAuditor）— 模板套语 / 句式对称度 / 情绪总结 / 对话口语化（权重 0.3）

三位专家共用同一 XML 输出格式（<评分结果> 内 <综合评分> / <问题列表> / <具体建议>），
通过 ``asyncio.gather`` 并行调用，任一专家失败不影响其他专家。

降级语义：
  - LLM 实例未配置 / 文本过短（<100 字符）→ 整体 failed=True，attraction_score=50.0
  - 单专家异常或解析失败 → 该专家按 50.0 参与加权（与降级语义一致），整体 failed=False
  - 三专家分数全部为默认降级值（50.0）→ 整体 failed=True，failure_reason="全部专家失败"
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra.async_retry import async_llm_call_with_retry
from novelfactory.evaluation.llm._shared import (
    clamp,
    extract_float,
    safe_match,
    trim_text,
)
from novelfactory.evaluation.llm.prompts import (
    AI_STYLE_AUDITOR_SYSTEM_PROMPT,
    FANQIE_EDITOR_SYSTEM_PROMPT,
    GENRE_AWARE_INSTRUCTION,
    IMMERSION_EXPERT_SYSTEM_PROMPT,
    RE_ATTRACTION_COMPOSITE_SCORE,
    RE_ATTRACTION_ISSUES,
    RE_ATTRACTION_SCORE_BLOCK,
    RE_ATTRACTION_SUGGESTIONS,
)
from novelfactory.evaluation.llm.schemas import LLMAttractionResult

logger = logging.getLogger(__name__)

# 综合分权重（与任务规格一致：0.4 × 节奏与爽点 + 0.3 × 代入感 + 0.3 × 文风）
_WEIGHT_FANQIE = 0.4
_WEIGHT_IMMERSION = 0.3
_WEIGHT_AI_STYLE = 0.3
# 默认降级分：单专家异常/解析失败时的兜底值（与"三专家全部失败"判定共用）
_FALLBACK_SCORE = 50.0


def _build_prompt(
    expert_prompt: str,
    chapter_text: str,
    genre: str | None = None,
    prev_summary: str = "",
) -> str:
    """构建单个专家的评审 prompt。

    结构（与 old_reader_llm._build_prompt 同构）：
      专家 system prompt → 待评审章节（trim_text 裁剪）→ 题材感知 → 前文摘要 → 输出格式重申
    """
    trimmed = trim_text(chapter_text)

    parts = [expert_prompt, "", "## 待评审章节", trimmed]

    if genre:
        parts.extend(["", GENRE_AWARE_INSTRUCTION.format(genre=genre)])

    if prev_summary:
        parts.extend(["", "## 前文摘要（供参考）", prev_summary[:2000]])

    # 输出格式重申（XML 标签，不要 markdown 包裹）
    parts.extend(
        [
            "",
            "## 输出格式（严格使用 XML 标签，不要 markdown 包裹）",
            "<评分结果>",
            "<综合评分>0-100</综合评分>",
            "<问题列表>问题1||问题2||问题3</问题列表>",
            "<具体建议>具体的修改建议</具体建议>",
            "</评分结果>",
        ]
    )
    return "\n".join(parts)


def _parse_response(raw: str) -> tuple[float, list[str], str]:
    """使用 RE_* 正则解析专家 LLM 响应（三位专家共用一套输出格式）。

    Returns:
        (综合评分, 问题列表, 具体建议)：评分 clamp 到 [0, 100]，
        解析失败时评分回落 50.0、问题为空列表。
    """
    score_match = RE_ATTRACTION_SCORE_BLOCK.search(raw)
    if not score_match:
        logger.warning("[LLM吸引力] 未找到 <评分结果> 标签，子分回落 50.0")
        return (_FALLBACK_SCORE, [], "")

    sb = score_match.group(1)
    score = clamp(
        extract_float(safe_match(RE_ATTRACTION_COMPOSITE_SCORE, sb), _FALLBACK_SCORE)
    )

    issues: list[str] = []
    issues_match = RE_ATTRACTION_ISSUES.search(sb)
    if issues_match:
        issues = [s.strip() for s in issues_match.group(1).split("||") if s.strip()]

    suggestion = safe_match(RE_ATTRACTION_SUGGESTIONS, sb) or ""
    return (score, issues, suggestion)


async def _run_expert(
    expert_name: str,
    system_prompt: str,
    prompt: str,
    caller_llm: BaseChatModel,
) -> tuple[float, list[str], str]:
    """独立调用单个专家 LLM 并解析其响应。

    每个专家有独立的异常保护：任何异常仅将该专家按 50.0 降级
    （记 warning 日志），不抛异常、不阻塞其他专家。

    Args:
        expert_name: 专家标识（用于 step_name 与日志）
        system_prompt: 专家 system prompt（已由 _build_prompt 拼入 prompt，
            保留参数以对齐调用签名，仅作调试上下文）
        prompt: 完整评审 prompt
        caller_llm: 实际调用的 LLM 实例

    Returns:
        (score, issues, fix)：异常时返回 (50.0, [], "") 降级值
    """
    step_name = f"attraction_{expert_name}"
    try:
        response = await async_llm_call_with_retry(
            caller_llm.ainvoke, prompt, step_name=step_name, retry_policy="reviewer"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        score, issues, fix = _parse_response(raw)
        logger.info("[%s] 完成 | score=%.1f issues=%d", step_name, score, len(issues))
        return (score, issues, fix)
    except Exception as e:
        logger.warning("[%s] 专家分析异常: %s", step_name, e)
        return (_FALLBACK_SCORE, [], "")


async def attraction_llm_analysis(
    chapter_text: str,
    genre: str | None = None,
    prev_summary: str = "",
    llm: BaseChatModel | None = None,
) -> LLMAttractionResult:
    """执行 LLM 吸引力专家团队评审。

    三位专家（番茄爆款编辑 / 代入感专家 / 反AI味审计师）通过 ``asyncio.gather``
    并行独立调用，任一专家失败不影响其他专家。

    Args:
        chapter_text: 章节正文
        genre: 题材（如「玄幻」「都市」）
        prev_summary: 前文摘要（注入 prompt 供上下文参考）
        llm: LLM 实例；为 None 时直接降级失败（不依赖全局注入实例）

    Returns:
        LLMAttractionResult（失败时 failed=True，含降级默认值）
    """
    if llm is None:
        logger.warning("[LLM吸引力] 未配置 LLM 实例，降级跳过")
        return LLMAttractionResult(
            failed=True, failure_reason="LLM 实例未配置", attraction_score=50.0
        )

    if not chapter_text or len(chapter_text.strip()) < 100:
        logger.info("[LLM吸引力] 文本过短 (%d chars)，跳过", len(chapter_text or ""))
        return LLMAttractionResult(
            failed=True, failure_reason="文本过短", attraction_score=50.0
        )

    experts = [
        ("fanqie_editor", "番茄爆款编辑", FANQIE_EDITOR_SYSTEM_PROMPT),
        ("immersion", "代入感专家", IMMERSION_EXPERT_SYSTEM_PROMPT),
        ("ai_style", "反AI味审计师", AI_STYLE_AUDITOR_SYSTEM_PROMPT),
    ]
    prompts = [
        _build_prompt(system_prompt, chapter_text, genre=genre, prev_summary=prev_summary)
        for _, _, system_prompt in experts
    ]

    # 三专家并行独立调用（各自内部 try/except，gather 不抛异常）
    outcomes = await asyncio.gather(
        *[
            _run_expert(name, system_prompt, prompt, llm)
            for (name, _, system_prompt), prompt in zip(experts, prompts)
        ]
    )

    fanqie_score, fanqie_issues, fanqie_fix = outcomes[0]
    immersion_score, immersion_issues, immersion_fix = outcomes[1]
    ai_style_score, ai_style_issues, ai_style_fix = outcomes[2]

    # 三专家全部失败判定：分数均为默认降级值（50.0）
    if (
        fanqie_score == _FALLBACK_SCORE
        and immersion_score == _FALLBACK_SCORE
        and ai_style_score == _FALLBACK_SCORE
    ):
        logger.warning("[LLM吸引力] 三位专家全部失败，降级")
        return LLMAttractionResult(
            failed=True, failure_reason="全部专家失败", attraction_score=50.0
        )

    # 综合分：0.4 × 节奏与爽点 + 0.3 × 代入感 + 0.3 × 文风
    # （部分失败时失败专家按 50.0 参与加权，与降级语义一致）
    attraction_score = round(
        _WEIGHT_FANQIE * fanqie_score
        + _WEIGHT_IMMERSION * immersion_score
        + _WEIGHT_AI_STYLE * ai_style_score,
        1,
    )

    # issues = 三专家合并非空问题（保持专家顺序）
    issues: list[str] = []
    issues.extend(fanqie_issues)
    issues.extend(immersion_issues)
    issues.extend(ai_style_issues)

    # fix 拼接："吸引力专家建议：\n- 专家：建议"
    fix_parts = [
        f"- {label}：{fix}"
        for (_, label, _), fix in zip(
            experts, (fanqie_fix, immersion_fix, ai_style_fix)
        )
        if fix
    ]
    fix = "吸引力专家建议：\n" + "\n".join(fix_parts) if fix_parts else ""

    result = LLMAttractionResult(
        attraction_score=attraction_score,
        fanqie_editor_score=fanqie_score,
        immersion_score=immersion_score,
        ai_style_score=ai_style_score,
        issues=issues,
        fix=fix,
        failed=False,
    )
    logger.info(
        "[LLM吸引力] 完成 | attraction=%.1f fanqie=%.1f immersion=%.1f "
        "ai_style=%.1f issues=%d failed=%s",
        result.attraction_score,
        result.fanqie_editor_score,
        result.immersion_score,
        result.ai_style_score,
        len(result.issues),
        result.failed,
    )
    return result
