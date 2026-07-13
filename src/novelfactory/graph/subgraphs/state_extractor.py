"""
StateExtractor 子图 — 写后状态提取
===================================
将 NovelStateTracker.after_chapter() 中的 LLM 提取部分拆解为 LangGraph 子图节点。

节点:
  extract_characters   — LLM 提取角色状态变化
  extract_events       — LLM 提取关键事件
  detect_arcs          — LLM 检测弧线阶段推进
  run_audit            — LLM 一致性审计
  extract_foreshadowing — LLM 伏笔提取
  analyze_pacing       — LLM 节奏分析

所有 LLM 调用都是图节点，token 消耗和延迟可追踪。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH_FOR_EXTRACTION = 100  # 最小文本长度阈值（低于此值跳过处理）

# ── State ──────────────────────────────────────────────────────────────────────


class StateExtractorState(TypedDict):
    """StateExtractor 子图的状态。

    输入: chapter_text, chapter_number, project_name, world_setting, character_setting
    输出: extracted (包含所有提取结果的 dict)
    """

    project_name: str
    chapter_number: int
    chapter_text: str
    world_setting: str
    character_setting: str

    # ── 提取结果 ──
    extracted_characters: list[dict]  # 角色状态变化
    extracted_events: list[dict]  # 关键事件
    extracted_arc_transitions: list[dict]  # 弧线推进
    extracted_audit: dict  # 审计报告
    extracted_foreshadowing: list[dict]  # 伏笔
    extracted_pacing: dict  # 节奏分析

    # ── 汇总 ──
    extracted: dict  # 合并后的提取结果
    error: str


# ── LLM Helper ─────────────────────────────────────────────────────────────────


def _get_llm() -> Any:
    from novelfactory.config.llm import get_reviewer_llm

    return get_reviewer_llm()


# 并发限流 — 最多同时 3 个 LLM 调用，避免瞬间 6 个并发触发 API 429
_llm_semaphore = threading.Semaphore(3)


def _safe_llm_json(prompt: str, default: Any = None) -> Any:
    """同步调用 LLM 并解析 JSON 响应。

    使用 threading.Semaphore 限流，内置重试保护。
    """
    with _llm_semaphore:
        for attempt in range(3):
            try:
                llm = _get_llm()
                resp = llm.invoke([("user", prompt)])
                text = resp.content if hasattr(resp, "content") else str(resp)
                text = text.strip()
                # 优先尝试直接解析（如果以 { 或 [ 开头）
                if text.startswith("{") or text.startswith("["):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
                # 用贪婪正则匹配最外层 JSON 对象/数组
                match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                if attempt < 2:
                    continue
                break
            except Exception as e:
                logger.warning("LLM call attempt %d/3 failed: %s", attempt + 1, e)
                if attempt == 2:
                    return default
                import time

                time.sleep(1 * (attempt + 1))
        return default


# ── Node: extract_characters ───────────────────────────────────────────────────


def _extract_characters_node(state: StateExtractorState) -> dict:
    """LLM 提取角色状态变化。"""
    text: str = state.get("chapter_text", "")[:4000]

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {"extracted_characters": []}

    prompt = f"""从以下章节中提取角色状态变化。

章节内容：
{text}

输出JSON数组：
[
  {{
    "name": "角色名",
    "location": "当前位置",
    "mood": "当前心境",
    "power_level": "修为等级",
    "status": "健在/受伤/失踪/死亡",
    "relationships": {{"角色名": "关系描述"}},
    "knowledge": ["已知信息"],
    "items": ["持有物品"],
    "summary": "状态变化摘要（20字内）"
  }}
]

只输出有变化的角色。输出纯JSON数组。"""

    result = _safe_llm_json(prompt, [])
    return {"extracted_characters": result if isinstance(result, list) else []}


# ── Node: extract_events ───────────────────────────────────────────────────────


def _extract_events_node(state: StateExtractorState) -> dict:
    """LLM 提取关键事件。"""
    text: str = state.get("chapter_text", "")[:4000]

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {"extracted_events": []}

    prompt = f"""从以下章节中提取关键事件。

章节内容：
{text}

输出JSON数组：
[
  {{
    "event": "事件描述（20字内）",
    "characters": ["涉及角色"],
    "event_type": "plot_twist/character_growth/reveal/battle/transition",
    "importance": 7
  }}
]

importance: 10=主线重大转折, 8-9=重要推进, 6-7=常规, 1-5=日常。
输出纯JSON数组。"""

    result = _safe_llm_json(prompt, [])
    return {"extracted_events": result if isinstance(result, list) else []}


# ── Node: detect_arcs ──────────────────────────────────────────────────────────


def _detect_arcs_node(state: StateExtractorState) -> dict:
    """LLM 检测角色弧线阶段推进。"""
    text: str = state.get("chapter_text", "")[:3000]
    project: str = state.get("project_name", "")
    ch: int = state.get("chapter_number", 1)

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {"extracted_arc_transitions": []}

    tracker = None
    try:
        from novelfactory.store.tracker import NovelStateTracker

        tracker = NovelStateTracker(project)
        scale: Any = tracker.scale
        transitions = []

        if scale:
            arcs = scale.arcs.get_all_arcs(project)
            for arc in arcs:
                detection = scale.arcs.detect_stage_transition(
                    project, arc.character_name, ch, text, _get_llm()
                )
                if detection.get("should_advance"):
                    transitions.append(
                        {
                            "character": arc.character_name,
                            "from_stage": arc.current_stage,
                            "milestone": detection.get("milestone", ""),
                            "reason": detection.get("reason", ""),
                        }
                    )

        return {"extracted_arc_transitions": transitions}
    except Exception as e:
        logger.warning("detect_arcs failed: %s", e)
        return {"extracted_arc_transitions": []}
    finally:
        if tracker is not None:
            tracker.close()


# ── Node: run_audit ────────────────────────────────────────────────────────────


def _run_audit_node(state: StateExtractorState) -> dict:
    """LLM 一致性审计。"""
    text: str = state.get("chapter_text", "")[:3000]
    ws: str = state.get("world_setting", "")[:2000]
    cs: str = state.get("character_setting", "")[:2000]

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {"extracted_audit": {"score": 100, "findings": [], "summary": ""}}

    prompt = f"""你是一位小说一致性审计专家。快速检查本章是否与设定矛盾。

【世界观设定】
{ws}

【角色设定】
{cs}

【本章内容】
{text}

输出JSON：
{{
  "score": 95,
  "findings": [
    {{
      "severity": "critical/major/minor/info",
      "category": "character/world/plot/timeline/power_system",
      "description": "问题描述",
      "evidence": "原文引用",
      "suggestion": "修复建议"
    }}
  ],
  "summary": "一句话总结"
}}

critical=-20分, major=-10分, minor=-3分, info=不扣分。无问题则findings为空,score=100。
输出纯JSON。"""

    result = _safe_llm_json(prompt, {"score": 100, "findings": [], "summary": ""})
    return {
        "extracted_audit": result
        if isinstance(result, dict)
        else {"score": 100, "findings": [], "summary": ""}
    }


# ── Node: extract_foreshadowing ────────────────────────────────────────────────


def _extract_foreshadowing_node(state: StateExtractorState) -> dict:
    """LLM 伏笔提取。"""
    text: str = state.get("chapter_text", "")[:4000]

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {"extracted_foreshadowing": []}

    prompt = f"""从以下章节中提取伏笔信息。

章节内容：
{text}

输出JSON数组：
[
  {{
    "name": "伏笔名称（10字内）",
    "description": "伏笔描述（30字内）",
    "category": "plot/character/item/mystery/relationship",
    "priority": 7,
    "planned_resolve_chapter": 预计回收章节号（0=不确定）,
    "related_characters": ["相关角色"],
    "action": "planted/resolved"
  }}
]

priority: 9-10=主线核心, 7-8=重要, 5-6=次要, 1-4=小伏笔。
输出纯JSON数组。"""

    result = _safe_llm_json(prompt, [])
    return {"extracted_foreshadowing": result if isinstance(result, list) else []}


# ── Node: analyze_pacing ───────────────────────────────────────────────────────


def _analyze_pacing_node(state: StateExtractorState) -> dict:
    """LLM 节奏分析。"""
    text: str = state.get("chapter_text", "")[:3000]

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {
            "extracted_pacing": {
                "intensity": 5.0,
                "event_density": 5.0,
                "pacing_label": "balanced",
            }
        }

    prompt = f"""分析以下章节的节奏。

章节内容：
{text}

输出JSON：
{{
  "intensity": 6.5,
  "event_density": 7.0,
  "dialogue_ratio": 0.3,
  "action_ratio": 0.4,
  "description_ratio": 0.3,
  "pacing_label": "buildup"
}}

intensity: 1=极度舒缓, 5=正常, 10=极度紧张。
pacing_label: fast/balanced/slow/buildup/climax/cooldown。
输出纯JSON。"""

    result = _safe_llm_json(
        prompt, {"intensity": 5.0, "event_density": 5.0, "pacing_label": "balanced"}
    )
    return {
        "extracted_pacing": result
        if isinstance(result, dict)
        else {"intensity": 5.0, "event_density": 5.0, "pacing_label": "balanced"}
    }


# ── Node: aggregate ────────────────────────────────────────────────────────────


def _aggregate_extracted_node(state: StateExtractorState) -> dict:
    """合并所有提取结果。"""
    return {
        "extracted": {
            "characters": state.get("extracted_characters", []),
            "events": state.get("extracted_events", []),
            "arc_transitions": state.get("extracted_arc_transitions", []),
            "audit": state.get("extracted_audit", {}),
            "foreshadowing": state.get("extracted_foreshadowing", []),
            "pacing": state.get("extracted_pacing", {}),
        }
    }


# ── Graph Builder ──────────────────────────────────────────────────────────────


def build_state_extractor() -> CompiledStateGraph:
    """构建 StateExtractor 子图。

    6 个 LLM 节点并行执行（互不依赖），最后 aggregate 合并。
    使用 threading.Semaphore 限流，最多同时 3 个 LLM 调用。
    """
    builder = StateGraph(StateExtractorState)

    builder.add_node("extract_characters", _extract_characters_node)
    builder.add_node("extract_events", _extract_events_node)
    builder.add_node("detect_arcs", _detect_arcs_node)
    builder.add_node("run_audit", _run_audit_node)
    builder.add_node("extract_foreshadowing", _extract_foreshadowing_node)
    builder.add_node("analyze_pacing", _analyze_pacing_node)
    builder.add_node("aggregate", _aggregate_extracted_node)

    # 并行执行所有 LLM 提取节点
    builder.add_edge(START, "extract_characters")
    builder.add_edge(START, "extract_events")
    builder.add_edge(START, "detect_arcs")
    builder.add_edge(START, "run_audit")
    builder.add_edge(START, "extract_foreshadowing")
    builder.add_edge(START, "analyze_pacing")

    # 全部汇聚到 aggregate
    builder.add_edge("extract_characters", "aggregate")
    builder.add_edge("extract_events", "aggregate")
    builder.add_edge("detect_arcs", "aggregate")
    builder.add_edge("run_audit", "aggregate")
    builder.add_edge("extract_foreshadowing", "aggregate")
    builder.add_edge("analyze_pacing", "aggregate")

    builder.add_edge("aggregate", END)

    return builder.compile()
