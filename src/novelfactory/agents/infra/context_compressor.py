"""上下文压缩层 — 按相关性筛选而非粗暴截断。

背景：writer prompt 全量注入 story_outline(20000字)/character_setting(15000字)，
加上 ContextBuilder 多源上下文，总注入量过大稀释核心信息（Lost in the Middle）。
本模块按当前章角色/关键词对上下文做段落级相关性筛选，保留高密度信息。

v9.1: 模型上下文窗口已扩容至 1M tokens（实际输入 60-80 万 tokens），
预算相应放宽（OUTLINE_BUDGET=50000 / CHARACTER_BUDGET=40000 / MEM_BUDGET=8000）。
压缩层仅做相关性筛选防止信息淹没，不再激进截断核心设定。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

OUTLINE_BUDGET = 50000     # v9.1: 大纲预算（1M 上下文放宽，50k 字符 ≈ 25k tokens）
CHARACTER_BUDGET = 40000   # v9.1: 角色设定预算（1M 上下文放宽）
MEM_BUDGET = 8000          # v9.1: 长期记忆预算（1M 上下文放宽）


def _split_paragraphs(text: str) -> list[str]:
    """按空行拆分段落。"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _relevance_score(para: str, terms: list[str]) -> int:
    """段落相关性得分：命中 term 的个数。"""
    if not terms:
        return 0
    return sum(1 for t in terms if t and t in para)


def _select_paragraphs(text: str, terms: list[str], budget: int) -> str:
    """按相关性+位置权重筛选段落，输出保持原文顺序。"""
    paras = _split_paragraphs(text)
    if not paras:
        return ""
    n = len(paras)
    # 每段打分：相关命中*100 + 位置权重（靠前段落基础分高，0-10）
    scored = []
    for i, p in enumerate(paras):
        pos_w = 10.0 * (n - i) / n if n else 0.0
        rel = _relevance_score(p, terms)
        scored.append((rel * 100.0 + pos_w, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    kept: set[int] = set()
    total = 0
    for _score, i in scored:
        # 预算按字符计，段落间分隔符 "\n\n" 计入开销，保证输出总长 <= budget
        sep = 2 if kept else 0
        if total + len(paras[i]) + sep > budget:
            continue
        kept.add(i)
        total += len(paras[i]) + sep
    if not kept and paras:  # 预算过小兜底：至少保留开头段
        kept = {0}
    return "\n\n".join(paras[i] for i in sorted(kept))


def compress_outline(
    outline: str, chapter_terms: Iterable[str] | None = None, budget: int = OUTLINE_BUDGET
) -> str:
    """压缩 story_outline：按本章角色/关键词相关性筛选段落。"""
    if not outline:
        return ""
    if len(outline) <= budget:
        return outline
    return _select_paragraphs(outline, [t for t in (chapter_terms or []) if t], budget)


def compress_character_setting(
    setting: str,
    chapter_characters: Iterable[str] | None = None,
    budget: int = CHARACTER_BUDGET,
) -> str:
    """压缩 character_setting：按本章出场角色优先保留设定段。"""
    if not setting:
        return ""
    if len(setting) <= budget:
        return setting
    return _select_paragraphs(setting, [c for c in (chapter_characters or []) if c], budget)


def compress_mem_text(mem_text: str, budget: int = MEM_BUDGET) -> str:
    """长期记忆结构化压缩：超长时保留要点（JSON dict 键值压缩，非 dict 头尾保留）。"""
    if not mem_text or len(mem_text) <= budget:
        return mem_text
    try:
        data = json.loads(mem_text)
    except Exception:
        data = None
    if isinstance(data, dict):
        items: list[str] = []
        for k, v in data.items():
            if not v:
                continue
            s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            items.append(f"{k}: {s[:200]}")
        result = "\n".join(items)
        return result[:budget]
    # 非 JSON 结构：保留头 60% + 尾 30%（头尾信息密度高）
    head = int(budget * 0.6)
    tail = int(budget * 0.3)
    return mem_text[:head] + "\n...(已压缩)...\n" + mem_text[-tail:]


def extract_chapter_characters(cr: dict[str, Any]) -> list[str]:
    """从 crew_result 提取本章出场角色（用于上下文相关性筛选）。

    优先级：chapter_plan.characters_involved > writer_context「出场角色：」行。
    """
    plan = cr.get("chapter_plan") or {}
    if isinstance(plan, dict):
        chars = plan.get("characters_involved") or []
        if chars:
            return list(chars)
    for key in ("writer_context", "cross_chapter_state"):
        ctx = cr.get(key) or ""
        if ctx:
            for line in ctx.splitlines():
                line = line.strip()
                if line.startswith("出场角色：") or line.startswith("涉及角色："):
                    names = [
                        n.strip()
                        for n in line.split("：", 1)[-1].replace("、", "，").split("，")
                        if n.strip()
                    ]
                    if names:
                        return names
    return []
