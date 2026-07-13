"""
Character State Tracker v2 — cross-chapter consistency for long-form novel writing.

v2 fixes:
  1. Pre-extract ALL character names from setup character_setting (no more missing characters)
  2. Force all known characters into state (even if not in current chapter)
  3. Better time inference with explicit last-chapter-summary enrichment
  4. Previous chapter summary now includes character locations/states

Usage:
    from novelfactory.state.chapter_state import (
        ChapterStateTracker,
        build_state_prompt_section,
    )

    tracker = ChapterStateTracker(world_setting, character_setting, chapter_outlines)
    # After each chapter:
    tracker.update(chapter_text, chapter_number)
    state_section = tracker.build_prompt_section()
"""

from __future__ import annotations

import json
import logging
import re

from novelfactory.config.llm import get_worker_llm

# ── Magic Number Constants ─────────────────────────────────────────────────────
DEFAULT_TRUNC_LEN = 8000  # 默认文本截断长度
EXTRACT_INPUT_MAX = 16000  # 章节状态提取输入最大长度
KNOWN_CHARS_PREVIEW = 5  # 已知角色预览数量
MAX_THREADS_DISPLAY = 8  # 待处理线索最大显示数
MAX_ACTIVE_CHARS_DISPLAY = 10  # 活跃角色最大显示数
MIN_CHAR_NAME_LEN_REGEX = 2  # 正则提取角色名最小长度
MAX_CHAR_NAME_LEN_REGEX = 4  # 正则提取角色名最大长度
CHAR_SETTING_EXTRACT_MAX = 3000  # 角色设定文本截断长度


class ChapterStateTracker:
    """Tracks character/plot state across chapters.

    Uses structured state injection to ensure:
    - No characters disappear between chapters
    - Unresolved plot threads are carried forward
    - Time/location progression is consistent
    - Word system (词条系统) state is tracked
    """

    def __init__(
        self,
        world_setting: str = "",
        character_setting: str = "",
        chapter_outlines: str = "",
    ) -> None:
        self.world_setting = world_setting
        self.character_setting = character_setting
        self.chapter_outlines = chapter_outlines

        # Internal state
        self._characters: dict[str, dict] = {}  # {name: {location, mood, ...}}
        self._unresolved_threads: list[str] = []
        self._current_location = ""
        self._time_since_start = "故事开始"
        self._last_chapter_number = 0
        self._last_chapter_summary = ""
        self._all_known_characters: list[str] = []

        # 词条系统状态 (Word System State)
        self._word_inventory: list[
            dict
        ] = []  # [{name, quality, level, source, effects}, ...]
        self._word_slots: int = 3  # 当前可镶嵌槽位数
        self._word_equipped: list[str] = []  # 当前已镶嵌的词条名称
        self._word_evolution_ready: list[str] = []  # 可进化的词条名称
        self._word_latest_action: str = ""  # 最近一次词条操作描述

        # 角色名在第一次 update() 时懒加载
        self._known_chars_extracted: bool = False

    def _extract_known_characters(self) -> None:
        """Extract ALL character names from character_setting.

        This ensures we never miss a character — even if they don't appear
        in a chapter, we still track that they exist.
        """
        llm = get_worker_llm()
        prompt = f"""从以下角色设定中，提取所有角色的姓名。

【角色设定】
{self.character_setting[:5000]}

请只输出一个JSON数组，包含所有角色名，不要其他内容。
格式：["角色名1", "角色名2", ...]"""

        try:
            resp = llm.invoke([("user", prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            arr_match = re.search(r"\[[\s\S]*\]", text)
            if arr_match:
                names = json.loads(arr_match.group())
                self._all_known_characters = [
                    n
                    for n in names
                    if isinstance(n, str) and len(n) >= MIN_CHAR_NAME_LEN_REGEX
                ]
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "[ChapterStateTracker] Failed to extract known characters from character_setting",
                exc_info=True,
            )

        if not self._all_known_characters:
            # Fallback: try simple regex extraction (Chinese names are 2-4 chars)
            names = re.findall(
                rf"(?:^|\n)\s*[-–—·•]\s*\*{{0,2}}([\u4e00-\u9fff]{{{MIN_CHAR_NAME_LEN_REGEX},{MAX_CHAR_NAME_LEN_REGEX}}})\*{{0,2}}[\s：:]",
                self.character_setting[:CHAR_SETTING_EXTRACT_MAX],
            )
            self._all_known_characters = list(
                set(n for n in names if len(n) >= MIN_CHAR_NAME_LEN_REGEX)
            )

        # Initialize all characters with "未出场" status
        for name in self._all_known_characters:
            if name not in self._characters:
                self._characters[name] = {
                    "status": "未出场",
                    "location": "未知",
                    "mood": "未知",
                    "power_level": "未知",
                    "chapter_active": [],
                }

    def _trunc(self, text: str, max_chars: int = EXTRACT_INPUT_MAX) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[...截断...]"

    def _build_word_state_text(self) -> str:
        """Build current word system state as readable text for prompts."""
        if not self._word_inventory:
            return "（尚未获得任何词条）"

        lines = [
            f"【当前词条背包】共{len(self._word_inventory)}个 | 槽位：{sum(1 for w in self._word_inventory if w['name'] in self._word_equipped)}/{self._word_slots}"
        ]
        for w in self._word_inventory:
            name = w.get("name", "?")
            quality = w.get("quality", "白")
            equipped = " [已镶嵌]" if name in self._word_equipped else ""
            effects = w.get("effects", "")
            line = f"  - {name} | {quality}品质{equipped}"
            if effects:
                line += f" | 效果：{effects}"
            lines.append(line)

        if self._word_evolution_ready:
            lines.append(f"【可进化词条】{'、'.join(self._word_evolution_ready)}")
        if self._word_latest_action:
            lines.append(f"【最新词条动态】{self._word_latest_action}")
        return "\n".join(lines)

    def update(self, chapter_text: str, chapter_number: int) -> None:
        """Update state after a chapter is completed.

        Args:
            chapter_text: Full chapter text.
            chapter_number: Current chapter number.
        """
        # 懒加载角色名：仅在首次 update() 时调用
        if not self._known_chars_extracted:
            self._extract_known_characters()
            self._known_chars_extracted = True

        llm = get_worker_llm()

        # Build known characters list for the prompt
        known_chars_str = json.dumps(self._all_known_characters, ensure_ascii=False)

        # Build previous state summary
        prev_state_lines = []
        for name, info in sorted(self._characters.items()):
            if info.get("status") != "未出场" or name in [
                c for c in self._all_known_characters[:KNOWN_CHARS_PREVIEW]
            ]:
                loc = info.get("location", "未知")
                status = info.get("status", "未出场")
                active = info.get("chapter_active", [])
                prev_state_lines.append(
                    f"  - {name}：位于{loc}，状态：{status}，最后出场于第{active[-1] if active else '?'}章"
                )

        prev_state_text = (
            "\n".join(prev_state_lines)
            if prev_state_lines
            else "（新故事，尚无历史状态）"
        )

        # Current word system state
        word_state_text = self._build_word_state_text()

        # 提升截断阈值以覆盖长章节(5000-10000字)
        prompt = f"""你是一位资深小说编辑。请分析以下章节正文，提取所有角色的当前状态以及词条系统状态。

【章节编号】第{chapter_number}章

【所有已知角色】（必须追踪这些角色，即使本章未出场也要保留上一章状态）
{known_chars_str}

【上一章状态（参考，不要丢失！）】
{prev_state_text}

【上一章时间】{self._time_since_start}
【上一章场景】{self._current_location or "未知"}

【上一章词条系统状态】
{word_state_text}

【章节正文】
{self._trunc(chapter_text, EXTRACT_INPUT_MAX)}

## 提取要求
请输出一个JSON对象，包含以下字段：

1. "characters": {{}}
   每个key是角色名。每个value包含：
   - "location": 当前位置（精确，如"数据坟场地下实验室"）
   - "mood": 心境（2-4字，如"警惕""迷茫""振奋"）
   - "power_level": 修为（如"筑基初期""金丹期"，未知填"未知"）
   - "status": "健在"/"受伤"/"失踪"/"死亡"/"未出场"
   - "knowledge": ["知道的事实1", ...]
   - "items": ["持有的道具", ...]

   注意：**陈铁骨、赵无极、苏晴、陈默、亚当·李** 等setup阶段定义的角色，即使本章只有少量出场也要追踪！

2. "unresolved_threads": []
   所有仍未解决的剧情线索。每条简明扼要。

3. "current_location": ""
   本章主要场景

4. "time_since_start": ""
   从故事开始累计到本章末尾的时间跨度（如"第3天""第1个月"）

5. "key_events": []
   本章关键事件

6. "word_system": {{}}
   词条系统状态（主角是唯一拥有者）：
   - "word_inventory": [{{"name": "词条名", "quality": "白/绿/蓝/紫/金/彩", "effects": "效果描述", "source": "来源"}}]
     当前背包中所有已获得的词条。新增词条追加，已有词条更新quality（合并升级后）。
   - "word_equipped": ["已镶嵌的词条名1", ...]
   - "word_slots": 当前可镶嵌槽位数
   - "word_evolution_ready": ["可进化的词条名1", ...]
   - "latest_action": "本章词条系统关键事件描述（如"成功抽取了'坚韧'词条""将[力量]+[体力]合并为蓝色[强韧]"）"

## 约束
- 必须包含所有已知角色（上面列表中的），未出场角色标注"status":"未出场"
- 上一章状态中可能出现的角色，如果本章未出现，保留上一章状态
- 词条系统状态必须追踪所有已出现词条，不要遗漏
- JSON必须严格有效，不要任何非JSON内容

JSON格式：
{{"characters": {{...}}, "unresolved_threads": [...], "current_location": "...", "time_since_start": "...", "key_events": [...], "word_system": {{...}}}}"""

        try:
            resp = llm.invoke([("user", prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "[ChapterStateTracker] LLM invoke failed during chapter %d state update",
                chapter_number,
                exc_info=True,
            )
            self._last_chapter_summary = chapter_text[:500]
            self._last_chapter_number = chapter_number
            return

        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            self._last_chapter_summary = chapter_text[:500]
            self._last_chapter_number = chapter_number
            return

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning(
                "[ChapterStateTracker] JSON parse failed for chapter %d state update — LLM response not valid JSON",
                chapter_number,
            )
            self._last_chapter_summary = chapter_text[:500]
            self._last_chapter_number = chapter_number
            return

        # Merge: update characters from this chapter, keep previous state for others
        new_chars = data.get("characters", {})
        for name, info in new_chars.items():
            if name not in self._characters:
                self._characters[name] = {}
            # Merge all fields
            for k, v in info.items():
                self._characters[name][k] = v
            # Track active chapters
            active = self._characters[name].get("chapter_active", [])
            if chapter_number not in active:
                active.append(chapter_number)
            self._characters[name]["chapter_active"] = active

        # Ensure ALL known characters exist
        for name in self._all_known_characters:
            if name not in self._characters:
                self._characters[name] = {
                    "status": "未出场",
                    "location": "未知",
                    "mood": "未知",
                    "chapter_active": [],
                }
            else:
                # Carry forward status for characters not in this chapter's output
                if name not in new_chars:
                    if self._characters[name].get("status") not in ("死亡", "失踪"):
                        pass  # keep previous state
                active = self._characters[name].get("chapter_active", [])
                if chapter_number not in active:
                    active.append(chapter_number) if self._characters[name].get(
                        "status"
                    ) != "未出场" else None

        # Update word system state
        word_sys = data.get("word_system")
        if word_sys:
            new_inventory = word_sys.get("word_inventory", [])
            if new_inventory:
                # Merge: replace matching names, append new ones
                existing_map = {w["name"]: w for w in self._word_inventory}
                for w in new_inventory:
                    existing_map[w["name"]] = w
                self._word_inventory = list(existing_map.values())
            if "word_equipped" in word_sys:
                self._word_equipped = word_sys["word_equipped"]
            if "word_slots" in word_sys:
                self._word_slots = word_sys["word_slots"]
            if "word_evolution_ready" in word_sys:
                self._word_evolution_ready = word_sys["word_evolution_ready"]
            if "latest_action" in word_sys:
                self._word_latest_action = word_sys["latest_action"]

        # Update global state
        if "unresolved_threads" in data:
            # Merge: keep old unresolved threads, add new ones
            old_threads = set(self._unresolved_threads)
            new_threads = data["unresolved_threads"]
            merged = list(old_threads)
            for t in new_threads:
                if t not in merged:
                    merged.append(t)
            self._unresolved_threads = merged[
                :50
            ]  # 最多保留50条活跃线索（防止无限增长）

        if data.get("current_location"):
            self._current_location = data["current_location"]
        if data.get("time_since_start"):
            self._time_since_start = data["time_since_start"]

        self._last_chapter_summary = chapter_text[:500]
        self._last_chapter_number = chapter_number

    def build_prompt_section(self) -> str:
        """Build a structured state section for injection into chapter writer prompt.

        Returns a formatted string that goes into the LLM prompt before writing
        the next chapter.
        """
        parts = []

        # Time & location
        ctx = "【当前剧情状态】"
        if self._time_since_start:
            ctx += f" 时间：{self._time_since_start}"
        if self._current_location:
            ctx += f" 地点：{self._current_location}"
        parts.append(ctx)

        # Character states — active characters first
        char_lines = ["【所有角色当前位置】"]
        # Sort: active characters first, then by name
        sorted_chars = sorted(
            self._characters.items(),
            key=lambda x: (
                0 if x[1].get("status") not in ("未出场",) else 1,
                x[0],
            ),
        )
        for name, info in sorted_chars:
            loc = info.get("location", "未知")
            mood = info.get("mood", "")
            power = info.get("power_level", "")
            status = info.get("status", "健在")
            items = info.get("items", [])

            line = f"  - {name}"
            if loc and loc != "未知":
                line += f" | {loc}"
            if status and status not in ("健在", "未出场"):
                line += f" | [{status}]"
            if power and power != "未知":
                line += f" | {power}期"
            if mood:
                line += f" | 心境：{mood}"
            if items:
                line += f" | 持有：{'、'.join(items[:3])}"
            if status == "未出场":
                line += " | （尚未登场）"
            char_lines.append(line)

        parts.append("\n".join(char_lines))

        # Unresolved threads
        if self._unresolved_threads:
            parts.append(
                "【待处理伏笔/线索】\n"
                + "\n".join(
                    f"  - {t}" for t in self._unresolved_threads[:MAX_THREADS_DISPLAY]
                )
            )

        # Critical: enforce character continuity
        # List all characters who MUST appear or be explained
        active_chars = [
            name
            for name, info in self._characters.items()
            if info.get("status") not in ("未出场", "死亡")
        ]
        if active_chars:
            parts.append(
                "【必须延续的角色】\n"
                f"  以下角色必须在当前章中出场，或有合理解释说明去向：{'、'.join(active_chars[:MAX_ACTIVE_CHARS_DISPLAY])}"
            )

        # Word system state
        word_state = self._build_word_state_text()
        if word_state:
            parts.append(f"【词条系统状态】\n{word_state}")

        return "\n\n".join(parts)

    def get_character(self, name: str) -> dict:
        return self._characters.get(name, {})

    def to_dict(self) -> dict:
        return {
            "characters": self._characters,
            "unresolved_threads": self._unresolved_threads,
            "current_location": self._current_location,
            "time_since_start": self._time_since_start,
            "last_chapter_number": self._last_chapter_number,
            "last_chapter_summary": self._last_chapter_summary,
            "word_inventory": self._word_inventory,
            "word_slots": self._word_slots,
            "word_equipped": self._word_equipped,
            "word_evolution_ready": self._word_evolution_ready,
            "word_latest_action": self._word_latest_action,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChapterStateTracker:
        tracker = cls()
        tracker._characters = data.get("characters", {})
        tracker._unresolved_threads = data.get("unresolved_threads", [])
        tracker._current_location = data.get("current_location", "")
        tracker._time_since_start = data.get("time_since_start", "故事开始")
        tracker._last_chapter_number = data.get("last_chapter_number", 0)
        tracker._last_chapter_summary = data.get("last_chapter_summary", "")
        tracker._all_known_characters = list(tracker._characters.keys())
        # Restore word system state
        tracker._word_inventory = data.get("word_inventory", [])
        tracker._word_slots = data.get("word_slots", 3)
        tracker._word_equipped = data.get("word_equipped", [])
        tracker._word_evolution_ready = data.get("word_evolution_ready", [])
        tracker._word_latest_action = data.get("word_latest_action", "")
        return tracker
