"""Evaluation shared utilities — paragraph indexing for review↔refine alignment.

v7.0: 评审→润色段落编号统一。所有 LLM 看到的章节文本都按 [P0][P1][P2]... 编号，
确保评审意见和润色输出引用的是同一套段落索引，消除"第N段"→[P{N-1}] 映射误差。
"""

from __future__ import annotations

import json
import re
from typing import Any


def index_chapter_text(text: str) -> str:
    """Split chapter text into paragraphs and add [P0][P1][P2]... markers.

    Args:
        text: Raw chapter text.

    Returns:
        Paragraph-indexed text, each paragraph prefixed with [Pi].
    """
    paragraphs = split_paragraphs(text)
    return "\n\n".join(f"[P{i}] {p}" for i, p in enumerate(paragraphs))


def split_paragraphs(text: str) -> list[str]:
    """Split chapter text into paragraphs.

    Splits on double newlines; further splits very long (>600 chars) single-newline
    paragraphs into separate entries for more granular indexing.
    """
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    result: list[str] = []
    for p in raw:
        if len(p) > 600 and "\n" in p:
            sub = [s.strip() for s in p.split("\n") if s.strip()]
            result.extend(sub)
        else:
            result.append(p)
    return result


def normalize_paragraph_refs(text: str) -> str:
    """Normalize "第N段" / "第N-M段" references to [P{N-1}] format.

    Handles:
        "第3段"         → "[P2]"
        "第3-5段"       → "[P2]-[P4]"
        "第3段和第5段"  → "[P2]和[P4]"
        "第3、4段"       → "[P2]、[P3]"

    Args:
        text: Text that may contain Chinese paragraph references.

    Returns:
        Text with normalized [Pi] references.
    """
    # "第3-5段" → "[P2]-[P4]"
    text = re.sub(
        r"第(\d+)-(\d+)段",
        lambda m: f"[P{int(m.group(1)) - 1}]-[P{int(m.group(2)) - 1}]",
        text,
    )
    # "第3、4段" → "[P2]、[P3]"
    text = re.sub(
        r"第(\d+)、(\d+)段",
        lambda m: f"[P{int(m.group(1)) - 1}]、[P{int(m.group(2)) - 1}]",
        text,
    )
    # "第3段和第5段" → handle "和" separated (covered by the single case below
    # since re.sub processes left to right)
    # "第3段" → "[P2]"
    text = re.sub(r"第(\d+)段", lambda m: f"[P{int(m.group(1)) - 1}]", text)
    return text


def apply_paragraph_fixes(original: str, fixes: dict[int, str]) -> str:
    """Apply paragraph-level fixes to original text.

    Args:
        original: Original chapter text.
        fixes: Dict mapping paragraph index → replacement text.

    Returns:
        Patched chapter text.
    """
    paragraphs = split_paragraphs(original)
    for idx, replacement in fixes.items():
        if 0 <= idx < len(paragraphs):
            paragraphs[idx] = replacement
    return "\n\n".join(paragraphs)


# ── v8.2 自 debate 迁移：Markdown 分段解析（critic_pre 前置评估用） ──

# 收敛关键词
CONVERGENCE_KEYWORDS_NO = ("否", "无异议", "已认同", "认同", "收敛", "同意")
CONVERGENCE_KEYWORDS_YES = ("是", "仍有", "坚持", "不同意", "异议")


def parse_markdown_sections(text: str) -> dict[str, Any]:
    """解析 LLM 自由文本的 Markdown 分段输出。

    支持的 section 头：
      ## 评审意见 / ## 问题列表 / ## 亮点 / ## 改进建议
      ## 评审意见（读者视角） 等变体也能匹配
    """
    if not text or not isinstance(text, str):
        return {"review_comments": "", "issues": [], "strengths": [], "suggestions": ""}

    result: dict[str, Any] = {
        "review_comments": "",
        "issues": [],
        "strengths": [],
        "suggestions": "",
    }

    # 尝试 JSON 解析
    cleaned = text.strip()
    if cleaned.startswith("{"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in result:
                    if key in parsed:
                        result[key] = parsed[key]
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # 按 ## section 头分割
    sections = re.split(r"\n(?=##\s)", cleaned)

    raw_section: dict[str, str] = {}
    for section in sections:
        m = re.match(r"##\s*(.+?)\s*\n(.*)", section, re.DOTALL)
        if m:
            header = m.group(1).strip()
            body = m.group(2).strip()
            raw_section[header] = body
        else:
            raw_section.setdefault("_preamble", "")
            raw_section["_preamble"] += section

    # 映射 section 头到结果字段
    section_map: list[tuple[str, str, str]] = [
        ("评审意见", "review_comments", "text"),
        ("问题", "issues", "list"),
        ("问题列表", "issues", "list"),
        ("亮点", "strengths", "list"),
        ("改进建议", "suggestions", "text"),
        ("建议", "suggestions", "text"),
    ]

    for raw_header, body in raw_section.items():
        if raw_header == "_preamble":
            continue
        h_lower = raw_header.lower().replace("（", "(").replace("）", ")")
        for pattern, field, fmt in section_map:
            if pattern in h_lower:
                if fmt == "text":
                    result[field] = body
                elif fmt == "list":
                    result[field] = _extract_list_items(body)
                break

    # fallback
    if not result["review_comments"] and raw_section.get("_preamble"):
        result["review_comments"] = raw_section["_preamble"].strip()
    if not result["review_comments"]:
        result["review_comments"] = cleaned[:500]

    return result


def _extract_list_items(text: str) -> list[str]:
    """从文本中提取列表项（- 或 * 或 数字. 开头）。"""
    items: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        m = re.match(r"^[-*•]\s+(.+)$", stripped)
        if m:
            items.append(m.group(1).strip())
            continue
        m = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if m:
            items.append(m.group(1).strip())
    return items


