"""
NovelFactory 长篇小说扩展模块
===============================
为 300 万字（600-1000 章）长篇小说提供三大核心能力：

  1. 分层大纲管理 — 卷→章→节三级结构
  2. 滑动上下文窗口 — 摘要压缩 + 关键事件索引
  3. 角色弧线追踪 — 发展阶段 + 弧线完整性检测

集成方式：
  - 新建 PG 表：novel_volumes, novel_key_events, novel_character_arcs
  - 扩展 novel_chapters 表：新增 volume_number, section_number 字段
  - 在 writing_crew 中注入滑动窗口上下文
  - 在 NovelStateTracker 中嵌入弧线追踪
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from novelfactory.config.constants import (
    CONTEXT_EVENT_DEFAULT_IMPORTANCE,
    CONTEXT_EVENT_EXTRACT_INPUT_MAX,
    CONTEXT_KEY_EVENT_IMPORTANCE_THRESHOLD,
    CONTEXT_KEY_EVENTS_MAX,
    CONTEXT_LAYER1_COUNT,
    CONTEXT_PREV_SUMMARIES_COUNT,
    CONTEXT_RECENT_EVENTS_COUNT,
    OUTLINE_DEFAULT_WORD_COUNT,
)
from novelfactory.pipeline.base import BaseManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 分层大纲系统
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Volume:
    """卷 — 长篇小说的顶层结构单元（如"第一卷：废铁觉醒"）"""

    volume_number: int
    title: str
    summary: str = ""  # 卷概要（100-200字）
    chapter_range: tuple = (0, 0)  # (起始章, 结束章)
    theme: str = ""  # 卷主题（如"底层挣扎与觉醒"）
    status: str = "pending"  # pending / writing / completed


@dataclass
class ChapterOutline:
    """章大纲 — 每章的具体规划"""

    chapter_number: int
    volume_number: int
    title: str = ""
    goal: str = ""  # 本章目标（1句话）
    key_beats: list[str] = field(default_factory=list)  # 关键节拍（3-5个）
    pov_character: str = ""  # 主视角角色
    characters_involved: list[str] = field(default_factory=list)
    foreshadowing_plant: list[str] = field(default_factory=list)  # 本章埋下的伏笔
    foreshadowing_resolve: list[str] = field(default_factory=list)  # 本章回收的伏笔
    word_count_target: int = OUTLINE_DEFAULT_WORD_COUNT
    status: str = "pending"  # pending / writing / completed


class OutlineManager(BaseManager):
    """分层大纲管理器 — 卷→章→节三级结构。

    设计原则：
      - 卷是战略层（主题、情绪弧线）
      - 章是战术层（每章目标、节拍）
      - 节是执行层（由 LLM 在写作时自由发挥）

    300 万字规模：约 30 卷 × 20-30 章/卷 = 600-900 章
    """

    # ── Volume CRUD ──────────────────────────────────────────────────────

    def create_volume(
        self,
        project: str,
        volume_number: int,
        title: str,
        theme: str = "",
        summary: str = "",
        start_chapter: int = 0,
        end_chapter: int = 0,
    ) -> None:
        self._execute(
            """
            INSERT INTO novel_volumes
                (project_name, volume_number, title, theme, summary,
                 start_chapter, end_chapter, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
            ON CONFLICT (project_name, volume_number) DO UPDATE SET
                title = EXCLUDED.title, theme = EXCLUDED.theme,
                summary = EXCLUDED.summary,
                start_chapter = EXCLUDED.start_chapter,
                end_chapter = EXCLUDED.end_chapter
        """,
            (project, volume_number, title, theme, summary, start_chapter, end_chapter),
        )

    def get_volume(self, project: str, volume_number: int) -> Volume | None:
        row = self._fetchone(
            """
            SELECT volume_number, title, summary, start_chapter, end_chapter, theme, status
            FROM novel_volumes
            WHERE project_name = %s AND volume_number = %s
        """,
            (project, volume_number),
        )
        if row:
            return Volume(
                volume_number=row[0],
                title=row[1],
                summary=row[2] or "",
                chapter_range=(row[3] or 0, row[4] or 0),
                theme=row[5] or "",
                status=row[6] or "pending",
            )
        return None

    def get_current_volume(self, project: str, chapter_number: int) -> Volume | None:
        """根据章节号找到所属卷。"""
        row = self._fetchone(
            """
            SELECT volume_number, title, summary, start_chapter, end_chapter, theme, status
            FROM novel_volumes
            WHERE project_name = %s
              AND start_chapter <= %s
              AND (end_chapter >= %s OR end_chapter = 0)
            ORDER BY volume_number DESC
            LIMIT 1
        """,
            (project, chapter_number, chapter_number),
        )
        if row:
            return Volume(
                volume_number=row[0],
                title=row[1],
                summary=row[2] or "",
                chapter_range=(row[3] or 0, row[4] or 0),
                theme=row[5] or "",
                status=row[6] or "pending",
            )
        return None

    def get_all_volumes(self, project: str) -> list[Volume]:
        rows = self._fetchall(
            """
            SELECT volume_number, title, summary, start_chapter, end_chapter, theme, status
            FROM novel_volumes
            WHERE project_name = %s
            ORDER BY volume_number
        """,
            (project,),
        )
        return [
            Volume(
                volume_number=r[0],
                title=r[1],
                summary=r[2] or "",
                chapter_range=(r[3] or 0, r[4] or 0),
                theme=r[5] or "",
                status=r[6] or "pending",
            )
            for r in rows
        ]

    def update_volume_status(
        self, project: str, volume_number: int, status: str
    ) -> None:
        self._execute(
            """
            UPDATE novel_volumes SET status = %s, updated_at = NOW()
            WHERE project_name = %s AND volume_number = %s
        """,
            (status, project, volume_number),
        )

    # ── Chapter Outline CRUD ──────────────────────────────────────────────

    def save_chapter_outline(self, project: str, outline: ChapterOutline) -> None:
        self._execute(
            """
            INSERT INTO novel_chapter_outlines
                (project_name, chapter_number, volume_number, title, goal,
                 key_beats, pov_character, characters_involved,
                 foreshadowing_plant, foreshadowing_resolve,
                 word_count_target, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_name, chapter_number) DO UPDATE SET
                volume_number = EXCLUDED.volume_number,
                title = EXCLUDED.title, goal = EXCLUDED.goal,
                key_beats = EXCLUDED.key_beats,
                pov_character = EXCLUDED.pov_character,
                characters_involved = EXCLUDED.characters_involved,
                foreshadowing_plant = EXCLUDED.foreshadowing_plant,
                foreshadowing_resolve = EXCLUDED.foreshadowing_resolve,
                word_count_target = EXCLUDED.word_count_target,
                status = EXCLUDED.status
        """,
            (
                project,
                outline.chapter_number,
                outline.volume_number,
                outline.title,
                outline.goal,
                json.dumps(outline.key_beats, ensure_ascii=False),
                outline.pov_character,
                json.dumps(outline.characters_involved, ensure_ascii=False),
                json.dumps(outline.foreshadowing_plant, ensure_ascii=False),
                json.dumps(outline.foreshadowing_resolve, ensure_ascii=False),
                outline.word_count_target,
                outline.status,
            ),
        )

    def get_chapter_outline(
        self, project: str, chapter_number: int
    ) -> ChapterOutline | None:
        row = self._fetchone(
            """
            SELECT chapter_number, volume_number, title, goal, key_beats,
                   pov_character, characters_involved,
                   foreshadowing_plant, foreshadowing_resolve,
                   word_count_target, status
            FROM novel_chapter_outlines
            WHERE project_name = %s AND chapter_number = %s
        """,
            (project, chapter_number),
        )
        if row:
            return ChapterOutline(
                chapter_number=row[0],
                volume_number=row[1],
                title=row[2] or "",
                goal=row[3] or "",
                key_beats=json.loads(row[4])
                if isinstance(row[4], str)
                else (row[4] or []),
                pov_character=row[5] or "",
                characters_involved=json.loads(row[6])
                if isinstance(row[6], str)
                else (row[6] or []),
                foreshadowing_plant=json.loads(row[7])
                if isinstance(row[7], str)
                else (row[7] or []),
                foreshadowing_resolve=json.loads(row[8])
                if isinstance(row[8], str)
                else (row[8] or []),
                word_count_target=row[9] or OUTLINE_DEFAULT_WORD_COUNT,
                status=row[10] or "pending",
            )
        return None

    def get_chapter_outlines_in_volume(
        self, project: str, volume_number: int
    ) -> list[ChapterOutline]:
        rows = self._fetchall(
            """
            SELECT chapter_number, volume_number, title, goal, key_beats,
                   pov_character, characters_involved,
                   foreshadowing_plant, foreshadowing_resolve,
                   word_count_target, status
            FROM novel_chapter_outlines
            WHERE project_name = %s AND volume_number = %s
            ORDER BY chapter_number
        """,
            (project, volume_number),
        )
        return [
            ChapterOutline(
                chapter_number=r[0],
                volume_number=r[1],
                title=r[2] or "",
                goal=r[3] or "",
                key_beats=json.loads(r[4]) if isinstance(r[4], str) else (r[4] or []),
                pov_character=r[5] or "",
                characters_involved=json.loads(r[6])
                if isinstance(r[6], str)
                else (r[6] or []),
                foreshadowing_plant=json.loads(r[7])
                if isinstance(r[7], str)
                else (r[7] or []),
                foreshadowing_resolve=json.loads(r[8])
                if isinstance(r[8], str)
                else (r[8] or []),
                word_count_target=r[9] or OUTLINE_DEFAULT_WORD_COUNT,
                status=r[10] or "pending",
            )
            for r in rows
        ]

    def mark_chapter_complete(self, project: str, chapter_number: int) -> None:
        self._execute(
            """
            UPDATE novel_chapter_outlines SET status = 'completed', updated_at = NOW()
            WHERE project_name = %s AND chapter_number = %s
        """,
            (project, chapter_number),
        )

    # ── Batch outline generation ──────────────────────────────────────────

    def generate_volume_outlines(
        self, project: str, volume_number: int, chapter_count: int, llm: Any
    ) -> list[ChapterOutline]:
        """用 LLM 批量生成一卷内所有章的大纲。"""
        volume = self.get_volume(project, volume_number)
        vol_context = (
            f"卷{volume_number}《{volume.title}》主题：{volume.theme}"
            if volume
            else f"卷{volume_number}"
        )

        prompt = f"""你是小说大纲规划师。请为以下卷生成 {chapter_count} 章的大纲。

{vol_context}

每章大纲格式（JSON数组）：
[
  {{
    "chapter_number": 1,
    "title": "章节标题",
    "goal": "本章目标（1句话，30字内）",
    "key_beats": ["节拍1", "节拍2", "节拍3"],
    "pov_character": "主视角角色名",
    "characters_involved": ["出场角色1", "出场角色2"],
    "foreshadowing_plant": ["本章埋下的伏笔"],
    "foreshadowing_resolve": ["本章回收的伏笔"],
    "word_count_target": {OUTLINE_DEFAULT_WORD_COUNT}
  }}
]

要求：
- 章节间有因果链，不能跳跃
- 每章有明确的起承转合
- 伏笔埋设和回收要合理分布
- 输出纯JSON数组，不要其他内容"""

        outlines_data = self._llm_invoke_json(prompt, is_array=True, llm=llm)
        if not outlines_data:
            return []

        outlines: list[ChapterOutline] = []
        for od in outlines_data:
            ch_num = od.get("chapter_number", 0)
            outline = ChapterOutline(
                chapter_number=ch_num,
                volume_number=volume_number,
                title=od.get("title", f"第{ch_num}章"),
                goal=od.get("goal", ""),
                key_beats=od.get("key_beats", []),
                pov_character=od.get("pov_character", ""),
                characters_involved=od.get("characters_involved", []),
                foreshadowing_plant=od.get("foreshadowing_plant", []),
                foreshadowing_resolve=od.get("foreshadowing_resolve", []),
                word_count_target=od.get(
                    "word_count_target", OUTLINE_DEFAULT_WORD_COUNT
                ),
            )
            self.save_chapter_outline(project, outline)
            outlines.append(outline)
        return outlines


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 滑动上下文窗口
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class KeyEvent:
    """关键事件 — 跨章记忆的最小单元"""

    chapter: int
    event: str  # 事件描述（1句话）
    characters: list[str]  # 涉及角色
    event_type: str  # plot_twist / character_growth / reveal / battle / transition
    importance: int = 5  # 1-10，重要性评分


class SlidingContextWindow(BaseManager):
    """滑动上下文窗口 — 为每章写作提供精准的上下文。

    策略（三层上下文）：
      Layer 1 — 紧邻上下文：前 5 章完整摘要（衔接用）
      Layer 2 — 关键事件索引：前 50 章中 importance≥6 的事件（记忆锚点）
      Layer 3 — 卷级上下文：当前卷的概要 + 卷内已写章节摘要

    300 万字规模下，每章注入的上下文控制在 3000-5000 tokens。
    """

    # ── Key Events CRUD ──────────────────────────────────────────────────

    def save_key_event(self, project: str, event: KeyEvent) -> None:
        self._execute(
            """
            INSERT INTO novel_key_events
                (project_name, chapter_number, event_text, characters, event_type, importance)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_name, chapter_number, event_text) DO NOTHING
        """,
            (
                project,
                event.chapter,
                event.event,
                json.dumps(event.characters, ensure_ascii=False),
                event.event_type,
                event.importance,
            ),
        )

    def get_high_importance_events(
        self,
        project: str,
        before_chapter: int,
        max_events: int = CONTEXT_KEY_EVENTS_MAX,
    ) -> list[KeyEvent]:
        """获取 before_chapter 之前的高重要性事件。

        1000章规模需要至少30条关键事件，importance≥6 即纳入检索。
        """
        rows = self._fetchall(
            """
            SELECT chapter_number, event_text, characters, event_type, importance
            FROM novel_key_events
            WHERE project_name = %s
              AND chapter_number < %s
              AND importance >= %s
            ORDER BY importance DESC, chapter_number DESC
            LIMIT %s
        """,
            (
                project,
                before_chapter,
                CONTEXT_KEY_EVENT_IMPORTANCE_THRESHOLD,
                max_events,
            ),
        )
        return [
            KeyEvent(
                chapter=r[0],
                event=r[1],
                characters=json.loads(r[2]) if isinstance(r[2], str) else (r[2] or []),
                event_type=r[3] or "",
                importance=r[4] or 5,
            )
            for r in rows
        ]

    def get_recent_events(
        self,
        project: str,
        before_chapter: int,
        count: int = CONTEXT_RECENT_EVENTS_COUNT,
    ) -> list[KeyEvent]:
        """获取最近的 N 个关键事件（按章节倒序）。"""
        rows = self._fetchall(
            """
            SELECT chapter_number, event_text, characters, event_type, importance
            FROM novel_key_events
            WHERE project_name = %s AND chapter_number < %s
            ORDER BY chapter_number DESC
            LIMIT %s
        """,
            (project, before_chapter, count),
        )
        return [
            KeyEvent(
                chapter=r[0],
                event=r[1],
                characters=json.loads(r[2]) if isinstance(r[2], str) else (r[2] or []),
                event_type=r[3] or "",
                importance=r[4] or 5,
            )
            for r in rows
        ]

    # ── Chapter Summary Cache ────────────────────────────────────────────

    def get_chapter_summaries(
        self, project: str, chapters: list[int]
    ) -> dict[int, str]:
        """批量获取章节摘要。"""
        if not chapters:
            return {}
        placeholders = ",".join(["%s"] * len(chapters))
        rows = self._fetchall(
            f"""
            SELECT chapter_number, summary
            FROM novel_chapters
            WHERE project_name = %s AND chapter_number IN ({placeholders})
        """,
            [project] + chapters,
        )
        return {row[0]: row[1] or "" for row in rows}

    def get_previous_chapters_summary(
        self,
        project: str,
        before_chapter: int,
        count: int = CONTEXT_PREV_SUMMARIES_COUNT,
    ) -> list[str]:
        """获取 before_chapter 之前 count 章的摘要。"""
        start = max(1, before_chapter - count)
        chapters = list(range(start, before_chapter))
        summaries = self.get_chapter_summaries(project, chapters)
        return [summaries.get(ch, "") for ch in chapters if summaries.get(ch)]

    # ── Build Context Window ─────────────────────────────────────────────

    def build_context(
        self, project: str, chapter_number: int, outline_manager: OutlineManager = None
    ) -> str:
        """构建当前章的完整上下文窗口。

        返回格式化的字符串，可直接注入 writer prompt。
        """
        parts: list[str] = []

        # Layer 1: 紧邻上下文（前 5 章摘要）
        prev_summaries = self.get_previous_chapters_summary(
            project, chapter_number, CONTEXT_LAYER1_COUNT
        )
        if prev_summaries:
            parts.append("【前情提要】")
            for i, s in enumerate(prev_summaries):
                ch = chapter_number - len(prev_summaries) + i
                parts.append(f"  第{ch}章：{s[:200]}")

        # Layer 2: 卷级上下文
        if outline_manager:
            volume = outline_manager.get_current_volume(project, chapter_number)
            if volume:
                parts.append(f"\n【当前卷】{volume.title}")
                if volume.theme:
                    parts.append(f"  卷主题：{volume.theme}")
                if volume.summary:
                    parts.append(f"  卷概要：{volume.summary[:200]}")

                # 卷内已完成的章
                vol_outlines = outline_manager.get_chapter_outlines_in_volume(
                    project, volume.volume_number
                )
                completed = [
                    o
                    for o in vol_outlines
                    if o.chapter_number < chapter_number and o.status == "completed"
                ]
                if completed:
                    parts.append(
                        f"  本卷已完成 {len(completed)}/{len(vol_outlines)} 章"
                    )

        # Layer 3: 关键事件索引（高重要性）— 1000章规模需要至少30条关键事件
        key_events = self.get_high_importance_events(
            project, chapter_number, CONTEXT_KEY_EVENTS_MAX
        )
        if key_events:
            parts.append("\n【关键历史事件】")
            for e in key_events:
                chars = "、".join(e.characters[:3])
                parts.append(f"  [第{e.chapter}章] {e.event}（涉及：{chars}）")

        return "\n".join(parts)

    # ── Auto-extract key events from chapter ─────────────────────────────

    def extract_and_save_events(
        self, project: str, chapter_number: int, chapter_text: str, llm: Any
    ) -> list[KeyEvent]:
        """用 LLM 从章节中提取关键事件并保存。"""
        prompt = f"""从以下章节中提取关键事件。每个事件一句话描述。

章节内容：
{chapter_text[:CONTEXT_EVENT_EXTRACT_INPUT_MAX]}

输出JSON数组：
[
  {{
    "event": "事件描述（20字内）",
    "characters": ["涉及角色"],
    "event_type": "plot_twist/character_growth/reveal/battle/transition",
    "importance": 7
  }}
]

importance 评分标准：
  10 = 主线重大转折（主角死亡/觉醒/重大真相揭露）
  8-9 = 重要剧情推进（新角色登场/关键战斗/重要伏笔埋设）
  6-7 = 常规剧情推进（日常/过渡/次要战斗）
  1-5 = 日常/填充

输出纯JSON数组。"""

        events_data = self._llm_invoke_json(prompt, is_array=True, llm=llm)
        if not events_data:
            return []

        events: list[KeyEvent] = []
        for ed in events_data:
            event = KeyEvent(
                chapter=chapter_number,
                event=ed.get("event", ""),
                characters=ed.get("characters", []),
                event_type=ed.get("event_type", "transition"),
                importance=min(
                    10, max(1, ed.get("importance", CONTEXT_EVENT_DEFAULT_IMPORTANCE))
                ),
            )
            self.save_key_event(project, event)
            events.append(event)
        return events


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 角色弧线追踪
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArcStage:
    """弧线阶段"""

    stage_name: str  # 阶段名（如"凡人期"、"觉醒期"、"成长期"）
    start_chapter: int
    end_chapter: int = 0  # 0 表示未结束
    description: str = ""  # 阶段描述
    key_milestones: list[str] = field(default_factory=list)  # 关键里程碑


@dataclass
class CharacterArc:
    """角色弧线 — 一个角色的完整发展轨迹"""

    character_name: str
    arc_type: str = ""  # 成长型 / 堕落型 / 救赎型 / 稳定型 / 悲剧型
    stages: list[ArcStage] = field(default_factory=list)
    current_stage: str = ""  # 当前阶段名
    arc_completeness: float = 0.0  # 弧线完成度 0-100


class CharacterArcTracker(BaseManager):
    """角色弧线追踪器 — 追踪每个角色的发展阶段。

    核心能力：
      1. 定义弧线模板（成长型/堕落型/救赎型等）
      2. 追踪当前阶段
      3. 检测弧线完整性（是否缺阶段）
      4. 生成弧线状态提示（注入 writer prompt）
    """

    # 预设弧线模板 — 覆盖起点主流小说角色类型
    ARC_TEMPLATES = {
        # ── 主角常用 ──
        "成长型": [
            "平凡期",
            "触发期",
            "觉醒期",
            "试炼期",
            "突破期",
            "巅峰期",
            "传承期",
        ],
        "逆袭型": [
            "低谷期",
            "机缘期",
            "蛰伏期",
            "爆发期",
            "扬名期",
            "称霸期",
            "超脱期",
        ],
        "重生型": [
            "前世终局",
            "重生觉醒",
            "先知布局",
            "逆天改命",
            "势力崛起",
            "巅峰对决",
            "新纪元",
        ],
        "穿越型": [
            "穿越降临",
            "认知冲击",
            "适应融合",
            "降维打击",
            "势力经营",
            "文明引领",
            "传奇永恒",
        ],
        "废柴逆袭型": [
            "被弃期",
            "隐忍期",
            "觉醒期",
            "打脸期",
            "崛起期",
            "巅峰期",
            "逍遥期",
        ],
        # ── 反派/灰色角色 ──
        "堕落型": [
            "正直期",
            "诱惑期",
            "动摇期",
            "堕落期",
            "黑暗期",
            "悔悟期",
            "救赎期",
        ],
        "黑化型": [
            "纯真期",
            "背叛期",
            "绝望期",
            "黑化期",
            "复仇期",
            "空虚期",
            "毁灭/新生",
        ],
        "枭雄型": [
            "蛰伏期",
            "算计期",
            "夺权期",
            "称霸期",
            "守成期",
            "危机期",
            "落幕期",
        ],
        # ── 救赎/治愈类 ──
        "救赎型": [
            "堕落期",
            "低谷期",
            "触发期",
            "挣扎期",
            "觉醒期",
            "行动期",
            "新生期",
        ],
        "治愈型": [
            "创伤期",
            "相遇期",
            "打开期",
            "疗愈期",
            "成长期",
            "和解期",
            "圆满期",
        ],
        # ── 悲剧/牺牲类 ──
        "悲剧型": [
            "辉煌期",
            "隐患期",
            "转折期",
            "下滑期",
            "挣扎期",
            "毁灭期",
            "余韵期",
        ],
        "牺牲型": [
            "守护期",
            "危机期",
            "抉择期",
            "牺牲期",
            "传承期",
            "铭记期",
            "精神永存",
        ],
        "殉道型": [
            "信仰确立",
            "传道期",
            "受难期",
            "考验期",
            "殉道期",
            "遗产期",
            "永恒期",
        ],
        # ── 配角/功能性 ──
        "导师型": [
            "登场期",
            "传道期",
            "考验期",
            "托付期",
            "退场/牺牲",
            "精神传承",
        ],
        "战友型": [
            "相遇期",
            "磨合期",
            "信任期",
            "并肩期",
            "高光期",
            "沉淀期",
            "永恒羁绊",
        ],
        "对手型": [
            "初遇期",
            "较量期",
            "相惜期",
            "宿命对决",
            "超越/和解",
            "传承/告别",
        ],
        "红颜型": [
            "相遇期",
            "相知期",
            "暧昧期",
            "考验期",
            "定情/分离",
            "重逢/永别",
        ],
        # ── 特殊类型 ──
        "双面型": [
            "伪装期",
            "双面生活",
            "身份危机",
            "真相暴露",
            "抉择期",
            "融合/毁灭",
            "新生/终结",
        ],
        "继承型": [
            "继承前",
            "抗拒期",
            "接受期",
            "成长期",
            "超越期",
            "创新期",
            "传承期",
        ],
        "探索型": [
            "启程期",
            "发现期",
            "深入期",
            "危机期",
            "真相期",
            "抉择期",
            "归宿期",
        ],
        # ── 简单/稳定型 ──
        "稳定型": [
            "登场期",
            "协助期",
            "高光期",
            "沉淀期",
            "再起期",
        ],
        "功能型": [
            "出场期",
            "服务期",
            "退场期",
        ],
    }

    # 弧线类型 → 适用角色映射
    ARC_TYPE_GUIDE = {
        "成长型": "主角（传统升级流）",
        "逆袭型": "主角（废柴/底层逆袭）",
        "重生型": "主角（重生文）",
        "穿越型": "主角（穿越文）",
        "废柴逆袭型": "主角（退婚/废柴流）",
        "堕落型": "反派/灰色角色（可救赎）",
        "黑化型": "反派/受害者（不可逆）",
        "枭雄型": "野心家/帝王",
        "救赎型": "犯过错的主角/配角",
        "治愈型": "有心理创伤的角色",
        "悲剧型": "注定悲剧的英雄",
        "牺牲型": "为他人牺牲的角色",
        "殉道型": "为信仰献身的角色",
        "导师型": "主角的老师/引路人",
        "战友型": "主角的兄弟/搭档",
        "对手型": "宿敌/亦敌亦友",
        "红颜型": "女主角/感情线角色",
        "双面型": "卧底/双重身份",
        "继承型": "继承家业/传承",
        "探索型": "冒险/探索类主角",
        "稳定型": "功能性配角",
        "功能型": "龙套/工具人",
    }

    def define_arc(
        self,
        project: str,
        character_name: str,
        arc_type: str,
        custom_stages: list[str] = None,
    ) -> CharacterArc:
        """为角色定义弧线。"""
        stages = custom_stages or self.ARC_TEMPLATES.get(
            arc_type, ["登场期", "发展期", "高潮期", "收尾期"]
        )

        arc = CharacterArc(
            character_name=character_name,
            arc_type=arc_type,
            stages=[ArcStage(stage_name=s, start_chapter=0) for s in stages],
            current_stage=stages[0],
        )

        self._execute(
            """
            INSERT INTO novel_character_arcs
                (project_name, character_name, arc_type, stages_json, current_stage)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (project_name, character_name) DO UPDATE SET
                arc_type = EXCLUDED.arc_type,
                stages_json = EXCLUDED.stages_json,
                current_stage = EXCLUDED.current_stage
        """,
            (
                project,
                character_name,
                arc_type,
                json.dumps(
                    [
                        {
                            "stage_name": s.stage_name,
                            "start_chapter": s.start_chapter,
                            "end_chapter": s.end_chapter,
                            "description": s.description,
                            "key_milestones": s.key_milestones,
                        }
                        for s in arc.stages
                    ],
                    ensure_ascii=False,
                ),
                arc.current_stage,
            ),
        )
        return arc

    def get_arc(self, project: str, character_name: str) -> CharacterArc | None:
        row = self._fetchone(
            """
            SELECT character_name, arc_type, stages_json, current_stage
            FROM novel_character_arcs
            WHERE project_name = %s AND character_name = %s
        """,
            (project, character_name),
        )
        if row:
            stages_data = (
                json.loads(row[2]) if isinstance(row[2], str) else (row[2] or [])
            )
            stages = [
                ArcStage(
                    stage_name=s.get("stage_name", ""),
                    start_chapter=s.get("start_chapter", 0),
                    end_chapter=s.get("end_chapter", 0),
                    description=s.get("description", ""),
                    key_milestones=s.get("key_milestones", []),
                )
                for s in stages_data
            ]
            return CharacterArc(
                character_name=row[0],
                arc_type=row[1] or "",
                stages=stages,
                current_stage=row[3] or "",
            )
        return None

    def get_all_arcs(self, project: str) -> list[CharacterArc]:
        rows = self._fetchall(
            """
            SELECT character_name, arc_type, stages_json, current_stage
            FROM novel_character_arcs
            WHERE project_name = %s
        """,
            (project,),
        )
        arcs: list[CharacterArc] = []
        for row in rows:
            stages_data = (
                json.loads(row[2]) if isinstance(row[2], str) else (row[2] or [])
            )
            stages = [
                ArcStage(
                    stage_name=s.get("stage_name", ""),
                    start_chapter=s.get("start_chapter", 0),
                    end_chapter=s.get("end_chapter", 0),
                    description=s.get("description", ""),
                    key_milestones=s.get("key_milestones", []),
                )
                for s in stages_data
            ]
            arcs.append(
                CharacterArc(
                    character_name=row[0],
                    arc_type=row[1] or "",
                    stages=stages,
                    current_stage=row[3] or "",
                )
            )
        return arcs

    def advance_stage(
        self,
        project: str,
        character_name: str,
        chapter_number: int,
        milestone: str = "",
    ) -> bool:
        """推进角色到下一个弧线阶段。"""
        arc = self.get_arc(project, character_name)
        if not arc:
            return False

        # 找到当前阶段索引
        current_idx = None
        for i, s in enumerate(arc.stages):
            if s.stage_name == arc.current_stage:
                current_idx = i
                break

        if current_idx is None or current_idx >= len(arc.stages) - 1:
            return False  # 已是最后阶段

        # 标记当前阶段结束
        arc.stages[current_idx].end_chapter = chapter_number
        if milestone:
            arc.stages[current_idx].key_milestones.append(milestone)

        # 进入下一阶段
        next_stage = arc.stages[current_idx + 1]
        next_stage.start_chapter = chapter_number
        arc.current_stage = next_stage.stage_name

        # 计算完成度
        total = len(arc.stages)
        completed = sum(1 for s in arc.stages if s.end_chapter > 0)
        arc.arc_completeness = (completed / total) * 100

        # 保存
        self._execute(
            """
            UPDATE novel_character_arcs
            SET stages_json = %s, current_stage = %s, arc_completeness = %s
            WHERE project_name = %s AND character_name = %s
        """,
            (
                json.dumps(
                    [
                        {
                            "stage_name": s.stage_name,
                            "start_chapter": s.start_chapter,
                            "end_chapter": s.end_chapter,
                            "description": s.description,
                            "key_milestones": s.key_milestones,
                        }
                        for s in arc.stages
                    ],
                    ensure_ascii=False,
                ),
                arc.current_stage,
                arc.arc_completeness,
                project,
                character_name,
            ),
        )
        return True

    def detect_stage_transition(
        self,
        project: str,
        character_name: str,
        chapter_number: int,
        chapter_text: str,
        llm: Any,
    ) -> dict:
        """用 LLM 检测角色是否应该进入下一弧线阶段。"""
        arc = self.get_arc(project, character_name)
        if not arc:
            return {"should_advance": False, "reason": "无弧线定义"}

        # 计算下一阶段名（可读实现，替代原一行式嵌套 next()）
        current_idx = next(
            (i for i, s in enumerate(arc.stages) if s.stage_name == arc.current_stage),
            -1,
        )
        if 0 <= current_idx < len(arc.stages) - 1:
            next_stage_name = arc.stages[current_idx + 1].stage_name
        else:
            next_stage_name = "无"

        prompt = f"""你是角色弧线分析师。判断角色是否应该进入下一个发展阶段。

角色：{character_name}
弧线类型：{arc.arc_type}
当前阶段：{arc.current_stage}
下一阶段：{next_stage_name}

章节内容：
{chapter_text[:3000]}

输出JSON：
{{
  "should_advance": true/false,
  "reason": "判断理由（30字内）",
  "milestone": "里程碑事件描述（如有）"
}}"""

        result = self._llm_invoke_json(prompt, is_array=False, llm=llm)
        if result:
            return result
        return {"should_advance": False, "reason": "分析失败"}

    def build_arc_prompt(self, project: str) -> str:
        """生成弧线状态提示，注入 writer prompt。"""
        arcs = self.get_all_arcs(project)
        if not arcs:
            return ""

        lines = ["【角色弧线状态】"]
        for arc in arcs:
            total = len(arc.stages)
            completed = sum(1 for s in arc.stages if s.end_chapter > 0)
            pct = (completed / total * 100) if total > 0 else 0

            # 当前阶段在序列中的位置
            stage_names = [s.stage_name for s in arc.stages]
            current_pos = (
                stage_names.index(arc.current_stage)
                if arc.current_stage in stage_names
                else 0
            )

            stage_bar = ""
            for i, name in enumerate(stage_names):
                if i < current_pos:
                    stage_bar += f"[{name}]→"
                elif i == current_pos:
                    stage_bar += f"【{name}】→"
                else:
                    stage_bar += f"{name}→"
            stage_bar = stage_bar.rstrip("→")

            lines.append(
                f"  {arc.character_name} [{arc.arc_type}] {pct:.0f}%: {stage_bar}"
            )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 统一入口 — ScaleManager
# ═══════════════════════════════════════════════════════════════════════════════


class ScaleManager(BaseManager):
    """长篇小说扩展管理器 — 统一入口。

    在 writing_crew 中使用：
        scale = ScaleManager(pg_conn, project_name)

        # 写前
        context = scale.build_writer_context(chapter_number)

        # 写后
        scale.after_chapter(chapter_number, chapter_text)
    """

    def __init__(self, pg_conn: Any, project_name: str) -> None:
        super().__init__(pg_conn)
        self.project = project_name
        self.outline = OutlineManager(pg_conn)
        self.context = SlidingContextWindow(pg_conn)
        self.arcs = CharacterArcTracker(pg_conn)
        self._phase2: Any = None
        self._phase3: Any = None

    @property
    def phase2(self) -> Any:
        """Lazy-init Phase2Manager for audit + foreshadowing + pacing."""
        if self._phase2 is None:
            try:
                from novelfactory.pipeline.phase2_manager import Phase2Manager

                self._phase2 = Phase2Manager(self.outline._conn, self.project)
            except Exception as e:
                logger.warning("Phase2 init failed: %s", e)
                self._phase2 = None
        return self._phase2

    @property
    def phase3(self) -> Any:
        """Lazy-init Phase3Manager for checkpoint + cost + quality + volume."""
        if self._phase3 is None:
            try:
                from novelfactory.pipeline.phase3_manager import Phase3Manager

                self._phase3 = Phase3Manager(self.outline._conn, self.project)
            except Exception as e:
                logger.warning("Phase3 init failed: %s", e)
                self._phase3 = None
        return self._phase3

    def build_writer_context(self, chapter_number: int) -> str:
        """构建写入上下文 — 分层大纲 + 滑动窗口 + 弧线状态。"""
        parts: list[str] = []

        # 1. 分层大纲
        outline = self.outline.get_chapter_outline(self.project, chapter_number)
        if outline:
            parts.append(f"【本章大纲】第{chapter_number}章《{outline.title}》")
            parts.append(f"  目标：{outline.goal}")
            if outline.key_beats:
                parts.append(f"  节拍：{' → '.join(outline.key_beats)}")
            if outline.pov_character:
                parts.append(f"  主视角：{outline.pov_character}")
            if outline.characters_involved:
                parts.append(f"  出场角色：{'、'.join(outline.characters_involved)}")
            if outline.foreshadowing_plant:
                parts.append(f"  需埋伏笔：{'、'.join(outline.foreshadowing_plant)}")
            if outline.foreshadowing_resolve:
                parts.append(
                    f"  需回收伏笔：{'、'.join(outline.foreshadowing_resolve)}"
                )

        # 2. 滑动上下文窗口
        ctx = self.context.build_context(self.project, chapter_number, self.outline)
        if ctx:
            parts.append(ctx)

        # 3. 角色弧线
        arc_prompt = self.arcs.build_arc_prompt(self.project)
        if arc_prompt:
            parts.append(arc_prompt)

        # 4. Phase2: 审计 + 伏笔 + 节奏
        try:
            if self.phase2:
                p2_ctx = self.phase2.build_writer_context(chapter_number)
                if p2_ctx:
                    parts.append(p2_ctx)
        except Exception as e:
            logger.warning("Phase2 context build failed: %s", e)

        # 5. Phase3: 断点 + 成本 + 质量 + 多卷
        try:
            if self.phase3:
                p3_ctx = self.phase3.build_writer_context(chapter_number)
                if p3_ctx:
                    parts.append(p3_ctx)
        except Exception as e:
            logger.warning("Phase3 context build failed: %s", e)

        return "\n\n".join(parts)

    def after_chapter(
        self,
        chapter_number: int,
        chapter_text: str,
        world_setting: str = "",
        character_setting: str = "",
        quality_score: float = 0.0,
    ) -> dict:
        """章节完成后更新所有扩展系统。"""
        result: dict[str, Any] = {
            "outline": False,
            "events": 0,
            "arcs_checked": 0,
            "phase2": None,
        }

        # 1. 标记大纲完成
        try:
            self.outline.mark_chapter_complete(self.project, chapter_number)
            result["outline"] = True
        except Exception as e:
            logger.warning("Outline mark complete failed: %s", e)

        # 2. 提取关键事件
        try:
            events = self.context.extract_and_save_events(
                self.project, chapter_number, chapter_text, self._get_llm()
            )
            result["events"] = len(events)
        except Exception as e:
            logger.warning("Key events extraction failed: %s", e)

        # 3. 检查角色弧线推进
        try:
            arcs = self.arcs.get_all_arcs(self.project)
            for arc in arcs:
                detection = self.arcs.detect_stage_transition(
                    self.project,
                    arc.character_name,
                    chapter_number,
                    chapter_text,
                    self._get_llm(),
                )
                if detection.get("should_advance"):
                    self.arcs.advance_stage(
                        self.project,
                        arc.character_name,
                        chapter_number,
                        detection.get("milestone", ""),
                    )
                    result["arcs_checked"] += 1
        except Exception as e:
            logger.warning("Arc transition detection failed: %s", e)

        # 4. Phase2: 审计 + 伏笔 + 节奏
        try:
            if self.phase2:
                p2_result = self.phase2.after_chapter(
                    chapter_number,
                    chapter_text,
                    world_setting,
                    character_setting,
                )
                result["phase2"] = p2_result
        except Exception as e:
            logger.warning("Phase2 after_chapter failed: %s", e)

        # 5. Phase3: 断点 + 成本 + 质量 + 多卷
        try:
            if self.phase3:
                p3_result = self.phase3.after_chapter(chapter_number)
                result["phase3"] = p3_result
        except Exception:
            logger.exception("Phase3 after_chapter failed")

        return result

    def initialize_volume(
        self,
        volume_number: int,
        title: str,
        theme: str,
        chapter_count: int,
        start_chapter: int,
    ) -> list[ChapterOutline]:
        """初始化一卷：创建卷 + 生成章大纲。"""
        end_chapter = start_chapter + chapter_count - 1

        self.outline.create_volume(
            self.project,
            volume_number,
            title,
            theme,
            start_chapter=start_chapter,
            end_chapter=end_chapter,
        )

        return self.outline.generate_volume_outlines(
            self.project, volume_number, chapter_count, self._get_llm()
        )

    def define_character_arcs(self, characters: list[dict]) -> list[CharacterArc]:
        """批量定义角色弧线。

        characters: [{"name": "林北辰", "arc_type": "成长型"}, ...]
        """
        arcs: list[CharacterArc] = []
        for c in characters:
            arc = self.arcs.define_arc(
                self.project,
                c["name"],
                c.get("arc_type", "成长型"),
                c.get("custom_stages"),
            )
            arcs.append(arc)
        return arcs
