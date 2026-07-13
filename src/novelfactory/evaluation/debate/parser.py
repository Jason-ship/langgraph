"""Markdown 解析器 — 迁移自 quality_panel_agents.py。

解析 LLM 自由文本的 Markdown 分段输出，提取：
    - review_comments: 评审意见
    - issues: 问题列表
    - strengths: 亮点列表
    - suggestions: 改进建议
    - rebuttal_comments: 辩论意见
    - new_issues: 新增问题
    - revised_suggestions: 修正建议
    - has_dissent: 是否仍有异议
"""

from __future__ import annotations

import json
import re
from typing import Any

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


def parse_rebuttal(text: str) -> dict[str, Any]:
    """解析 rebuttal 自由文本输出。

    结构字段：
      - rebuttal_comments: 辩论意见正文
      - new_issues: 本轮新增问题
      - revised_suggestions: 基于辩论修正的建议
      - has_dissent: 是否仍有异议
    """
    if not text or not isinstance(text, str):
        return {
            "rebuttal_comments": "",
            "new_issues": [],
            "revised_suggestions": "",
            "has_dissent": True,
        }

    cleaned = text.strip()
    sections = re.split(r"\n(?=##\s)", cleaned)
    raw: dict[str, str] = {}
    for section in sections:
        m = re.match(r"##\s*(.+?)\s*\n(.*)", section, re.DOTALL)
        if m:
            raw[m.group(1).strip()] = m.group(2).strip()

    def _find(keys: list[str]) -> str:
        for header, body in raw.items():
            h = header.lower()
            if any(k in h for k in keys):
                return body
        return ""

    rebuttal_comments = _find(["辩论意见"])
    new_issues_text = _find(["新增问题"])
    revised_suggestions = _find(["修正建议"])
    dissent_text = _find(["是否仍有异议", "异议"])

    dissent_lower = dissent_text.lower()
    has_dissent = True
    if any(k in dissent_lower for k in CONVERGENCE_KEYWORDS_NO):
        has_dissent = False
    elif any(k in dissent_lower for k in CONVERGENCE_KEYWORDS_YES):
        has_dissent = True
    elif "否" in dissent_text:
        has_dissent = False

    if not rebuttal_comments:
        rebuttal_comments = cleaned[:500]

    return {
        "rebuttal_comments": rebuttal_comments,
        "new_issues": _extract_list_items(new_issues_text),
        "revised_suggestions": revised_suggestions,
        "has_dissent": has_dissent,
    }


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
