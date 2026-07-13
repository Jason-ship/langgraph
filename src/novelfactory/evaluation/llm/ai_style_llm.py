"""LLM 增强的 AI 味语义分析模块（v7.2）。

v7.2 核心改进：
  1. 标签化输出格式 — XML 标签替代 JSON（WebNovelBench）
  2. COT 前置提取 — 先提取章节概要再评分（WritingBench）
  3. Antislop 负面示例注入 — 帮助 LLM 识别 AI 套话模式
  4. 共享工具函数 — 通过 _shared.py 消除重复
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra.retry import llm_call_with_retry
from novelfactory.evaluation.llm._shared import (
    clamp,
    extract_bool,
    extract_float,
    safe_match,
    trim_text,
)
from novelfactory.evaluation.llm.old_reader_llm import get_reviewer_llm
from novelfactory.evaluation.llm.prompts import (
    AI_STYLE_LLM_SYSTEM_PROMPT,
    GENRE_AWARE_INSTRUCTION,
    RE_AI_ISSUE_ITEM,
    RE_AI_ISSUE_LIST,
    RE_AI_SUMMARY,
    RE_EMOTION_AUTH_SCORE,
    RE_HAS_OBVIOUS_AI,
    RE_HUMAN_LIKE_SCORE,
    RE_NATURALNESS_SCORE,
    RE_SCORE_BLOCK,
    RE_VOICE_SCORE,
)
from novelfactory.evaluation.llm.schemas import LLMAIStyleIssue, LLMAIStyleResult

logger = logging.getLogger(__name__)

# Antislop 高频词列表（v7.2 注入负面示例，供 LLM 语义判断参考）
_ANTISLOP_PATTERNS: list[str] = [
    "某种说不清道不明的",
    "一种莫名的",
    "难以言喻的",
    "一种说不出的",
    "深吸一口气",
    "咬了咬嘴唇",
    "攥紧了拳头",
    "倒吸一口凉气",
    "嘴角微微上扬",
    "空气中弥漫着",
    "周围一片寂静",
    "落针可闻",
    "更大的风暴还在后面",
    "故事远未结束",
    "他沉声道",
    "她冷冷道",
    "他缓缓开口",
    "她轻声说道",
    "一股暖流涌上心头",
    "心中百感交集",
    "说不清是什么滋味",
    "某种复杂的情绪在心头蔓延",
]


def _build_prompt(
    chapter_text: str,
    genre: str | None = None,
    programmatic_metrics: str = "",
) -> str:
    """构建 AI 味 LLM 分析的 prompt。"""
    trimmed = trim_text(chapter_text)
    parts = [
        AI_STYLE_LLM_SYSTEM_PROMPT,
        "",
        "## 待分析章节",
        trimmed,
    ]
    if genre:
        parts.extend(["", GENRE_AWARE_INSTRUCTION.format(genre=genre)])
    if programmatic_metrics:
        parts.extend(
            [
                "",
                "## 程序化检测结果（仅供参考）",
                programmatic_metrics,
                "",
                "注意：以上是统计层面的检测结果。请从语义层面独立判断。",
            ]
        )
    # 输出格式
    parts.extend(
        [
            "",
            "## 输出格式（严格使用 XML 标签，不要 markdown 包裹）",
            "<提取结果><章节概要>...</章节概要><段落结构>...</段落结构></提取结果>",
            "<评分结果>",
            "<人类相似度>0-100</人类相似度>  <自然度评分>0-10</自然度评分>",
            "<叙事声音评分>0-10</叙事声音评分>  <情绪真实感评分>0-10</情绪真实感评分>",
            "<问题列表>...</问题列表>",
            "<有明显AI痕迹>true/false</有明显AI痕迹>",
            "<总体评估>一句话总结</总体评估>",
            "</评分结果>",
        ]
    )
    return "\n".join(parts)


def _parse_ai_issues(score_block: str) -> list[dict[str, str]]:
    """从评分块中解析 AI 味问题列表。"""
    list_match = RE_AI_ISSUE_LIST.search(score_block)
    if not list_match:
        return []
    items: list[dict[str, str]] = []
    for m in RE_AI_ISSUE_ITEM.finditer(list_match.group(1)):
        items.append(
            {
                "category": m.group(1).strip(),
                "description": m.group(2).strip(),
                "quote": m.group(3).strip(),
                "replacement_suggestion": m.group(4).strip(),
            }
        )
    return items


def _parse_response(raw: str) -> LLMAIStyleResult:
    """使用 XML 正则解析 AI 味 LLM 响应（v7.2 标签化格式）。"""
    score_match = RE_SCORE_BLOCK.search(raw)
    if not score_match:
        logger.warning("[LLM AI味] 未找到 <评分结果> 标签，降级")
        return LLMAIStyleResult(
            failed=True, failure_reason="未找到 <评分结果> 标签", human_like_score=50.0
        )

    sb = score_match.group(1)

    human_like = clamp(extract_float(safe_match(RE_HUMAN_LIKE_SCORE, sb), 50.0))
    naturalness = clamp(
        extract_float(safe_match(RE_NATURALNESS_SCORE, sb), 5.0), 0.0, 10.0
    )
    voice = clamp(extract_float(safe_match(RE_VOICE_SCORE, sb), 5.0), 0.0, 10.0)
    emotional = clamp(
        extract_float(safe_match(RE_EMOTION_AUTH_SCORE, sb), 5.0), 0.0, 10.0
    )
    semantic_issues = _parse_ai_issues(sb)
    has_obvious = extract_bool(safe_match(RE_HAS_OBVIOUS_AI, sb), False)
    summary = safe_match(RE_AI_SUMMARY, sb) or ""

    return LLMAIStyleResult(
        human_like_score=human_like,
        naturalness=naturalness,
        voice_consistency=voice,
        emotional_authenticity=emotional,
        semantic_issues=[LLMAIStyleIssue(**i) for i in semantic_issues],
        has_obvious_ai=has_obvious,
        summary=summary,
        failed=False,
    )


def llm_ai_style_analysis(
    chapter_text: str,
    genre: str | None = None,
    programmatic_metrics: str = "",
    llm: BaseChatModel | None = None,
) -> LLMAIStyleResult:
    """执行 LLM 增强的 AI 味语义分析。

    v7.2 改进：
      - COT 前置提取章节概要再评分
      - Antislop 负面示例注入帮助识别套话
      - 标签化格式提高解析成功率

    Args:
        chapter_text: 章节正文
        genre: 题材
        programmatic_metrics: 程序化检测结果摘要（可选）
        llm: LLM 实例（可选，默认全局注入）

    Returns:
        LLMAIStyleResult（失败时 failed=True）
    """
    caller_llm = llm or get_reviewer_llm()
    if caller_llm is None:
        logger.warning("[LLM AI味] 未配置 LLM 实例，降级跳过")
        return LLMAIStyleResult(
            failed=True, failure_reason="LLM 实例未配置", human_like_score=50.0
        )

    if not chapter_text or len(chapter_text.strip()) < 100:
        logger.info("[LLM AI味] 文本过短，跳过")
        return LLMAIStyleResult(
            failed=True, failure_reason="文本过短", human_like_score=50.0
        )

    try:
        prompt = _build_prompt(
            chapter_text, genre=genre, programmatic_metrics=programmatic_metrics
        )
        response = llm_call_with_retry(
            caller_llm, prompt, step_name="llm_ai_style", retry_policy="reviewer"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_response(raw)
        logger.info(
            "[LLM AI味] 完成 | human_like=%.0f natural=%.1f voice=%.1f "
            "emotional=%.1f issues=%d obv_ai=%s failed=%s",
            result.human_like_score,
            result.naturalness,
            result.voice_consistency,
            result.emotional_authenticity,
            len(result.semantic_issues),
            result.has_obvious_ai,
            result.failed,
        )
        return result
    except Exception as e:
        logger.warning("[LLM AI味] 分析异常: %s", e)
        return LLMAIStyleResult(
            failed=True, failure_reason=f"分析异常: {e}", human_like_score=50.0
        )
