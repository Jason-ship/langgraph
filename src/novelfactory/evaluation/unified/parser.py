"""统一评审标签化解析器 + 自洽校验。

v8.2: 解析统一 LLM 评审输出（`[标签]` 正则提取，WebNovelBench 风格比 JSON 鲁棒），
并对 LLM 输出分数做自洽校验（只校验不评分，评分完全由 LLM 语义决定）。
"""

from __future__ import annotations

import re

from novelfactory.evaluation.unified.schemas import UnifiedReviewResult

_RE_FINAL = re.compile(r"\[评分\]\s*final\s*=\s*([\d.]+)")
_RE_DIM = re.compile(r"\[四维-([\u4e00-\u9fa5]+)\]\s*([\d.]+)")
_RE_TOXIC = re.compile(r"\[毒点\]\s*(.+)")
_RE_SHUANG = re.compile(r"\[爽点\]\s*(.+)")
_RE_AI = re.compile(r"\[AI味\]\s*([\d.]+)")
_RE_ATTR = re.compile(r"\[吸引力\]\s*([\d.]+)")
_RE_MM = re.compile(r"\[沉浸\]\s*([\d.]+)")
_RE_CROSS = re.compile(r"\[跨章\]\s*([\d.]+)")
_RE_DECAY = re.compile(r"\[衰减\]\s*(\S+)")
_RE_WATER = re.compile(r"\[废话段\]\s*(.+)")
_RE_JUMP = re.compile(r"\[跳跃\]\s*(.+)")
_RE_DISAGREE = re.compile(r"分歧点[：:]\s*(.+)")
_RE_SEVERE = re.compile(r"severe", re.IGNORECASE)

_DIM_KEY = {"剧情逻辑": "logic", "文笔": "writing", "人物": "character", "世界观": "world"}


def _first_float(text: str | None, default: float = 0.0) -> float:
    if not text:
        return default
    m = re.search(r"([\d.]+)", text)
    return float(m.group(1)) if m else default


def _split_semicolon(text: str | None) -> list[str]:
    if not text:
        return []
    if text.strip() in ("无", "none", "-"):
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def _parse_ints(text: str | None) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for token in re.split(r"[;,\s]+", text):
        if token.startswith("P") and token[1:].isdigit():
            out.append(int(token[1:]))
    return out


def parse_review_output(raw: str) -> UnifiedReviewResult | None:
    """标签化解析统一评审输出；无法提取综合分时返回 None。"""
    if not raw:
        return None
    m_final = _RE_FINAL.search(raw)
    if not m_final:
        return None
    result = UnifiedReviewResult(raw_text=raw)
    result.final_score = float(m_final.group(1))
    for m in _RE_DIM.finditer(raw):
        key = _DIM_KEY.get(m.group(1))
        if key:
            setattr(result.four_dim, key, float(m.group(2)))
    m_toxic = _RE_TOXIC.search(raw)
    if m_toxic:
        entries = []
        for item in _split_semicolon(m_toxic.group(1)):
            parts = item.split("|")
            paragraph = 0
            if len(parts) > 1 and re.search(r"\d", parts[1]):
                paragraph = int(re.sub(r"\D", "", parts[1]))
            entries.append({
                "type": parts[0].strip() if parts else "",
                "paragraph": paragraph,
                "severity": "severe" if any(_RE_SEVERE.search(p) for p in parts) else "normal",
                "reason": parts[2].strip() if len(parts) > 2 else "",
            })
        if entries:
            result.toxic_points = entries
            result.severe_toxic = any(e["severity"] == "severe" for e in entries)
    m_shuang = _RE_SHUANG.search(raw)
    if m_shuang:
        result.shuangdian_points = _split_semicolon(m_shuang.group(1))
        result.shuangdian_count = len(result.shuangdian_points)
    m_ai = _RE_AI.search(raw)
    if m_ai:
        result.human_like_score = _first_float(m_ai.group(1))
    m_attr = _RE_ATTR.search(raw)
    if m_attr:
        result.attraction_score = _first_float(m_attr.group(1))
    m_mm = _RE_MM.search(raw)
    if m_mm:
        result.immersion_score = _first_float(m_mm.group(1))
    m_cross = _RE_CROSS.search(raw)
    if m_cross:
        result.cross_chapter_score = _first_float(m_cross.group(1))
    m_decay = _RE_DECAY.search(raw)
    if m_decay:
        result.decay_hint = m_decay.group(1).lower()
    m_water = _RE_WATER.search(raw)
    if m_water:
        result.water_paragraphs = _parse_ints(m_water.group(1))
    m_jump = _RE_JUMP.search(raw)
    if m_jump:
        for item in _split_semicolon(m_jump.group(1)):
            parts = item.split("|")
            result.scene_transition_issues.append({
                "from": parts[0].strip() if parts else "",
                "to": parts[1].strip() if len(parts) > 1 else "",
                "reason": parts[2].strip() if len(parts) > 2 else "",
            })
    m_dsg = _RE_DISAGREE.search(raw)
    if m_dsg:
        result.perspective_disagreements = _split_semicolon(m_dsg.group(1))
    return result


def apply_consistency_check(result: UnifiedReviewResult) -> bool:
    """自洽校验（只校验不评分）：修正 final_score，返回是否修正。

    规则（硬约束优先于四维校准，防止"文笔分高但无爽点/有严重毒点"虚高）：
    1. 四维总分与 final 偏差 > 15 时以四维总分为准（先校准基准）
    2. severe 毒点存在时 final 不得 > 70（硬约束，降至 70）
    3. 无爽点时 final 不得 > 65（硬约束，降至 65）
    """
    changed = False
    if result.four_dim.total() > 0 and abs(result.four_dim.total() - result.final_score) > 15.0:
        result.final_score = round(result.four_dim.total(), 1)
        changed = True
    if result.severe_toxic and result.final_score > 70.0:
        result.final_score = 70.0
        result.consistency_fixed = True
        changed = True
    if result.shuangdian_count == 0 and result.final_score > 65.0:
        result.final_score = 65.0
        result.consistency_fixed = True
        changed = True
    return changed
