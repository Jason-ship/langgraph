"""Prompt 模板管理器 — v5.4 集中化提示词管理。

用法:
    from novelfactory.config.prompts import get_prompt
    prompt = get_prompt("setup", "quality_gate")

模板文件位于 config/prompts/*.yaml，按场景组织。
每个模板支持 {variable} 占位符和 format() 动态填充。

回退策略: YAML 文件不可用时, 回退到模块内硬编码的 FALLBACK_PROMPTS。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent
_CACHE: dict[str, dict[str, str]] = {}


def _load_yaml_prompts(filename: str) -> dict[str, str]:
    """从 YAML 文件加载 prompt 模板。"""
    yaml_path = _PROMPTS_DIR / filename
    try:
        import yaml

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            logger.debug("[prompts] Loaded %d templates from %s", len(data), filename)
            return {k: str(v) for k, v in data.items()}
    except Exception as e:
        logger.debug("[prompts] Failed to load %s: %s", filename, e)
    return {}


def get_prompt(category: str, name: str) -> str:
    """获取指定 prompt 模板。

    Args:
        category: 场景分类 (setup, extraction, review, monitoring)
        name: 模板名称 (quality_gate, extract_characters, ...)

    Returns:
        prompt 模板字符串, 可用 {} 占位符 + format() 填充。

    回退: YAML 未加载时使用 FALLBACK_PROMPTS 中的硬编码模板。
    """
    if category not in _CACHE:
        filename_map = {
            "setup": "setup.yaml",
            "extraction": "extraction.yaml",
            "review": "review.yaml",
            "monitoring": "monitoring.yaml",
        }
        filename = filename_map.get(category, f"{category}.yaml")
        _CACHE[category] = _load_yaml_prompts(filename)

    prompts = _CACHE[category]
    if name in prompts:
        return prompts[name]

    # Fallback to hardcoded defaults
    return _FALLBACK_PROMPTS.get(category, {}).get(name, "")


# ═══════════════════════════════════════════════════════════════════════════
#  硬编码回退模板 (YAML 不可用时使用)
# ═══════════════════════════════════════════════════════════════════════════

_FALLBACK_PROMPTS: dict[str, dict[str, str]] = {
    "setup": {
        "quality_gate": """请审核以下 Setup 阶段产出，给出量化评分。

【世界观设定】（字数：{world_setting_len}）
{world_setting}

【角色设定】（字数：{character_setting_len}）
{character_setting}

【故事主线】（字数：{story_outline_len}）
{story_outline}

【章节大纲】（字数：{chapter_outlines_len}）
{chapter_outlines}

## 评分维度（总分100）
| 维度 | 满分 | 评分锚点 |
|------|------|----------|
| 世界观完整度 | 30分 | 地理/力量体系/社会结构/历史均有详细描述 |
| 角色立体度 | 25分 | 人物有明确性格、动机、成长弧线 |
| 大纲结构 | 25分 | 起承转合清晰，有明确冲突和悬念 |
| 设定自洽性 | 20分 | 世界观与角色行为完全自洽 |

## 字数门槛
- 世界观设定 ≥3000字（不足按比例扣分）
- 角色设定 ≥1500字（不足按比例扣分）
- 大纲 ≥2000字（不足按比例扣分）

请先在 <thinking> 标签内逐维分析，然后输出JSON。
你必须输出有效的JSON，不要输出其他内容。
JSON格式：{{"quality_score": <总分>, "review_comments": "<评分明细+改进建议>"}}
""",
    },
    "monitoring": {
        "analysis_system": """你是一名小说创作数据分析师，帮助分析 NovelFactory AI 小说的自动生成状态。

你收到的数据包括：
- 已完成章节数 / 目标总章数
- 最近章节的质量评分历史（滑动窗口）
- 累计 Token 消耗和估算费用
- 最近的错误日志（如有）
- 当前写作阶段

请输出简短（<200字）的分析报告，包含：
1. **状态摘要**：进度百分比，预计剩余时间
2. **质量趋势**：近几章评分是上升/下降/稳定，如有下降趋势给出预警
3. **成本提示**：当前总花费，如果有异常消耗提醒
4. **健康检查**：是否有任何异常（连续低分、Token 突增、错误）
5. **一句话建议**：当前状态需要关注什么

用中文回复，不要用 Markdown 格式，直接返回纯文本。""",
    },
    "extraction": {
        "extract_characters": """从以下章节中提取角色状态变化。

章节内容：
{chapter_text}

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

只输出有变化的角色。输出纯JSON数组。""",
        "extract_events": """从以下章节中提取关键事件。

章节内容：
{chapter_text}

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
输出纯JSON数组。""",
        "run_audit": """你是一位小说一致性审计专家。快速检查本章是否与设定矛盾。

【世界观设定】
{world_setting}

【角色设定】
{character_setting}

【本章内容】
{chapter_text}

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
输出纯JSON。""",
        "extract_foreshadowing": """从以下章节中提取伏笔信息。

章节内容：
{chapter_text}

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
输出纯JSON数组。""",
        "analyze_pacing": """分析以下章节的节奏。

章节内容：
{chapter_text}

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
输出纯JSON。""",
    },
}
