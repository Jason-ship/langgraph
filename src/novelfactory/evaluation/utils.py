"""Evaluation shared utilities — paragraph indexing for review↔refine alignment.

v7.0: 评审→润色段落编号统一。所有 LLM 看到的章节文本都按 [P0][P1][P2]... 编号，
确保评审意见和润色输出引用的是同一套段落索引，消除"第N段"→[P{N-1}] 映射误差。
"""

from __future__ import annotations

import re


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
