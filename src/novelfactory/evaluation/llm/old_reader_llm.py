"""LLM 增强的老书虫评审模块（v7.2）。

v7.2 核心改进（基于论文原文）：
  1. 标签化输出格式 — XML 标签替代 JSON，正则解析成功率近 100%（WebNovelBench）
  2. COT 前置提取 — 先提取人物/冲突/情节再评分（WebNovelBench + WritingBench）
  3. 题材特异性毒点容忍度 — 基于 Fiction_Eval 豁免表
  4. 隐式规则发现 — 发现新毒点模式时可输出到规则库
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra.async_retry import async_llm_call_with_retry
from novelfactory.config.constants import resolve_genre
from novelfactory.evaluation.llm._shared import (
    clamp,
    extract_bool,
    extract_float,
    safe_match,
    trim_text,
)
from novelfactory.evaluation.llm.prompts import (
    GENRE_AWARE_INSTRUCTION,
    GENRE_TOXIC_TOLERANCE,
    OLD_READER_LLM_SYSTEM_PROMPT,
    RE_ABSORPTION_SCORE,
    RE_EMOTION_ARC_FLAG,
    RE_FRESHNESS_SCORE,
    RE_IMPLICIT_TOXIC,
    RE_LOGIC_SCORE,
    RE_SCORE_BLOCK,
    RE_SEMANTIC_SCORE,
    RE_SEVERE_TOXIC,
    RE_SHUANGDIAN_ITEM,
    RE_SHUANGDIAN_LIST,
    RE_STRENGTHS,
    RE_SUGGESTIONS,
    RE_TOXIC_ITEM,
    RE_TOXIC_LIST,
    RE_WEAKNESSES,
)
from novelfactory.evaluation.llm.schemas import (
    LLMOldReaderResult,
    LLMShuangdianDetail,
    LLMToxicDetail,
)

logger = logging.getLogger(__name__)

# 默认的 LLM 实例（注入评审 LLM）
_REVIEWER_LLM: BaseChatModel | None = None

# 运行时发现的隐式毒点规则（P3: 知识规则库迭代构建）
_DISCOVERED_TOXIC_RULES: dict[str, Any] = {}

# 严重毒点类型集合（与程序化 sensor 对齐，见 programmatic/runner.py）
# v7.6-fix: 移除 MORAL_WRONG — 三观不正不应归为严重毒点（各题材标准不同）
_SEVERE_TOXIC_TYPES = {"NTR", "NUE_ZHU", "SHENGMU"}


def set_reviewer_llm(llm: BaseChatModel) -> None:
    """设置全局评审 LLM 实例。"""
    global _REVIEWER_LLM  # noqa: PLW0603
    _REVIEWER_LLM = llm


def get_reviewer_llm() -> BaseChatModel | None:
    """获取全局评审 LLM 实例。"""
    return _REVIEWER_LLM


def get_discovered_toxic_rules() -> dict[str, dict[str, Any]]:
    """获取运行时发现的隐式毒点规则快照。"""
    return dict(_DISCOVERED_TOXIC_RULES)


def _build_prompt(
    chapter_text: str,
    genre: str | None = None,
    prev_summary: str = "",
) -> str:
    """构建老书虫 LLM 评审的 prompt。"""
    trimmed = trim_text(chapter_text)

    parts = [
        OLD_READER_LLM_SYSTEM_PROMPT,
        "",
        "## 待评审章节",
        trimmed,
    ]

    if genre:
        parts.extend(["", GENRE_AWARE_INSTRUCTION.format(genre=genre)])
        resolved = resolve_genre(genre)
        tolerance = GENRE_TOXIC_TOLERANCE.get(resolved, {})
        if tolerance:
            tol_str = "；".join(
                f"[{tp}]容忍系数={coef}" for tp, coef in tolerance.items()
            )
            parts.extend(
                [
                    "",
                    f"### 题材毒点容忍度（{resolved}）",
                    f"本题材对以下毒点的敏感度调整：{tol_str}",
                    "容忍系数接近 1.0=完全接受（读者预期），接近 0.0=零容忍。",
                ]
            )

    if prev_summary:
        parts.extend(["", "## 前文摘要（供参考）", prev_summary[:2000]])

    # 输出格式重申
    parts.extend(
        [
            "",
            "## 输出格式（严格使用 XML 标签，不要 markdown 包裹）",
            "<提取结果><主要人物>...</主要人物><核心冲突>...</核心冲突>"
            "<关键情节>...</关键情节><情感弧线>...</情感弧线></提取结果>",
            "<评分结果>",
            "<语义评分>0-100</语义评分>  <代入感评分>0-10</代入感评分>",
            "<逻辑自洽评分>0-10</逻辑自洽评分>  <新颖感评分>0-10</新颖感评分>",
            "<毒点列表>...</毒点列表>  <有严重毒点>true/false</有严重毒点>",
            "<有隐式毒点>true/false</有隐式毒点>  <爽点列表>...</爽点列表>",
            "<有情感弧线>true/false</有情感弧线>",
            "<亮点>亮点1||亮点2</亮点>  <问题>问题1||问题2</问题>",
            "<具体建议>...</具体建议>",
            "</评分结果>",
        ]
    )

    return "\n".join(parts)


def _parse_toxic_items(score_block: str) -> list[dict[str, str]]:
    """从评分块中解析毒点列表。"""
    list_match = RE_TOXIC_LIST.search(score_block)
    if not list_match:
        return []
    items: list[dict[str, str]] = []
    for m in RE_TOXIC_ITEM.finditer(list_match.group(1)):
        items.append(
            {
                "type": m.group(1).strip(),
                "severity": m.group(2).strip(),
                "description": m.group(3).strip(),
                "quote": m.group(4).strip(),
            }
        )
    return items


def _parse_shuangdian_items(score_block: str) -> list[dict[str, Any]]:
    """从评分块中解析爽点列表。"""
    list_match = RE_SHUANGDIAN_LIST.search(score_block)
    if not list_match:
        return []
    items: list[dict[str, Any]] = []
    for m in RE_SHUANGDIAN_ITEM.finditer(list_match.group(1)):
        items.append(
            {
                "type": m.group(1).strip(),
                "impact_rating": extract_float(m.group(2), 5.0),
                "description": m.group(3).strip(),
                "quote": m.group(4).strip(),
            }
        )
    return items


def _parse_separated_list(score_block: str, pattern: Any) -> list[str]:
    """解析用 || 分隔的列表。"""
    m = pattern.search(score_block)
    if not m or not m.group(1).strip():
        return []
    return [s.strip() for s in m.group(1).split("||") if s.strip()]


def _parse_response(raw: str) -> LLMOldReaderResult:
    """使用 XML 正则解析 LLM 响应（v7.2 标签化格式）。"""
    score_match = RE_SCORE_BLOCK.search(raw)
    if not score_match:
        logger.warning("[LLM老书虫] 未找到 <评分结果> 标签，降级")
        return LLMOldReaderResult(
            failed=True, failure_reason="未找到 <评分结果> 标签", semantic_score=50.0
        )

    sb = score_match.group(1)

    semantic_score = clamp(extract_float(safe_match(RE_SEMANTIC_SCORE, sb), 50.0))
    reading_absorption = clamp(
        extract_float(safe_match(RE_ABSORPTION_SCORE, sb), 5.0), 0.0, 10.0
    )
    logic_consistency = clamp(
        extract_float(safe_match(RE_LOGIC_SCORE, sb), 5.0), 0.0, 10.0
    )
    freshness = clamp(extract_float(safe_match(RE_FRESHNESS_SCORE, sb), 5.0), 0.0, 10.0)

    toxic_items = _parse_toxic_items(sb)
    has_severe = extract_bool(safe_match(RE_SEVERE_TOXIC, sb), False) or any(
        t["type"] in _SEVERE_TOXIC_TYPES for t in toxic_items
    )
    implicit_toxic = extract_bool(safe_match(RE_IMPLICIT_TOXIC, sb), False)

    # P3: 隐式毒点记录到运行时规则库
    if implicit_toxic and not any(t["type"] == "OTHER" for t in toxic_items):
        toxic_items.append(
            {
                "type": "OTHER",
                "severity": "medium",
                "description": "LLM 检测到隐式毒点（语义级，非关键词匹配）",
                "quote": "",
            }
        )
        _DISCOVERED_TOXIC_RULES["_implicit_count"] = (
            _DISCOVERED_TOXIC_RULES.get("_implicit_count", 0) + 1
        )

    sd_items = _parse_shuangdian_items(sb)
    emotion_arc = extract_bool(safe_match(RE_EMOTION_ARC_FLAG, sb), False)
    strengths = _parse_separated_list(sb, RE_STRENGTHS)
    weaknesses = _parse_separated_list(sb, RE_WEAKNESSES)
    suggestions = safe_match(RE_SUGGESTIONS, sb) or ""

    return LLMOldReaderResult(
        semantic_score=semantic_score,
        reading_absorption=reading_absorption,
        logic_consistency=logic_consistency,
        freshness=freshness,
        toxic_points=[LLMToxicDetail(**t) for t in toxic_items],
        has_severe_toxic=has_severe,
        implicit_toxic_found=implicit_toxic,
        shuangdian_points=[LLMShuangdianDetail(**s) for s in sd_items],
        emotional_arc_detected=emotion_arc,
        strengths=strengths,
        weaknesses=weaknesses,
        concrete_suggestions=suggestions,
        failed=False,
    )


async def llm_old_reader_analysis(
    chapter_text: str,
    genre: str | None = None,
    prev_summary: str = "",
    llm: BaseChatModel | None = None,
) -> LLMOldReaderResult:
    """执行 LLM 增强的老书虫评审。

    采用 WebNovelBench 验证的两步策略：
      1. COT 提取 — 先提取主要人物、核心冲突、关键情节
      2. 再评分 — 基于提取结果进行语义毒点/爽点/代入感评分

    Args:
        chapter_text: 章节正文
        genre: 题材（如「玄幻」「都市」）
        prev_summary: 前文摘要
        llm: LLM 实例（可选，默认使用全局注入的 LLM）

    Returns:
        LLMOldReaderResult（失败时 failed=True，含降级默认值）
    """
    caller_llm = llm or _REVIEWER_LLM
    if caller_llm is None:
        logger.warning("[LLM老书虫] 未配置 LLM 实例，降级跳过")
        return LLMOldReaderResult(
            failed=True, failure_reason="LLM 实例未配置", semantic_score=50.0
        )

    if not chapter_text or len(chapter_text.strip()) < 100:
        logger.info("[LLM老书虫] 文本过短 (%d chars)，跳过", len(chapter_text or ""))
        return LLMOldReaderResult(
            failed=True, failure_reason="文本过短", semantic_score=50.0
        )

    try:
        prompt = _build_prompt(chapter_text, genre=genre, prev_summary=prev_summary)
        response = await async_llm_call_with_retry(
            caller_llm, prompt, step_name="llm_old_reader", retry_policy="reviewer"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_response(raw)
        logger.info(
            "[LLM老书虫] 完成 | semantic=%.0f toxic=%d implicit=%s shuangdian=%d failed=%s",
            result.semantic_score,
            len(result.toxic_points),
            result.implicit_toxic_found,
            len(result.shuangdian_points),
            result.failed,
        )
        return result
    except Exception as e:
        logger.warning("[LLM老书虫] 分析异常: %s", e)
        return LLMOldReaderResult(
            failed=True, failure_reason=f"分析异常: {e}", semantic_score=50.0
        )
