"""
NovelFactory Phase 2 — 一致性审计 + 伏笔管理 + 节奏控制
========================================================
为 300 万字长篇小说提供质量保障体系。

  1. 一致性审计 — 每 N 章自动检查设定偏离
  2. 伏笔管理系统 — 优先级 + 回收计划 + 到期提醒
  3. 节奏控制系统 — 张弛检测 + 节奏建议

集成方式：
  - 新建 PG 表：novel_audit_reports, novel_foreshadowing, novel_pacing_snapshots
  - 在 NovelStateTracker.after_chapter() 中触发审计/伏笔/节奏检查
  - 审计结果注入 before_chapter() 的 writer prompt
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from novelfactory.config.constants import (
    AUDIT_FULL_CHAPTERS_MAX,
    AUDIT_FULL_SETTING_MAX,
    AUDIT_FULL_WINDOW,
    AUDIT_INTERVAL,
    AUDIT_MAX_CRITICAL_DISPLAY,
    AUDIT_MAX_MAJOR_DISPLAY,
    AUDIT_MIN_TREND_POINTS,
    AUDIT_SCORE_LIMIT,
    AUDIT_TREND_DISPLAY_COUNT,
    FORESHADOW_AHEAD,
    FORESHADOW_DEFAULT_PRIORITY,
    FORESHADOW_INPUT_MAX,
    FORESHADOW_MAX_OVERDUE_DISPLAY,
    FORESHADOW_MAX_UPCOMING_DISPLAY,
    PACING_CLIMAX_STREAK,
    PACING_INPUT_MAX,
    PACING_INTENSITY_DIFF,
    PACING_LOW_INTENSITY,
    PACING_MIN_SAMPLES,
    PACING_SLOW_STREAK,
    PACING_TREND_WINDOW,
)
from novelfactory.pipeline.base import BaseManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 一致性审计系统
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AuditFinding:
    """审计发现的问题"""

    severity: str  # critical / major / minor / info
    category: str  # character / world / plot / timeline / power_system
    description: str  # 问题描述
    evidence: str  # 证据（引用原文）
    suggestion: str  # 修复建议
    chapter: int = 0


@dataclass
class AuditReport:
    """审计报告"""

    project: str
    chapter_range: tuple  # (起始章, 结束章)
    findings: list[AuditFinding] = field(default_factory=list)
    overall_score: float = 100.0  # 一致性评分 0-100
    summary: str = ""
    created_at: str = ""


class ConsistencyAuditor(BaseManager):
    """一致性审计器 — 定期检查小说是否偏离设定。

    审计维度：
      1. 角色一致性 — 性格/能力/关系是否突变
      2. 世界观一致性 — 力量体系/地理/历史是否矛盾
      3. 剧情一致性 — 时间线/因果链是否断裂
      4. 伏笔一致性 — 已埋伏笔是否遗忘

    审计频率：每 10 章一次全量审计，每章一次快速检查。
    """

    AUDIT_INTERVAL = AUDIT_INTERVAL  # 每 N 章全量审计

    def should_full_audit(self, chapter_number: int) -> bool:
        """判断是否应该进行全量审计。"""
        return chapter_number % self.AUDIT_INTERVAL == 0

    def run_quick_audit(
        self,
        project: str,
        chapter_number: int,
        chapter_text: str,
        world_setting: str,
        character_setting: str,
        llm: Any,
    ) -> AuditReport:
        """快速审计 — 每章执行，只检查当前章与设定的矛盾。"""
        prompt = f"""你是一位小说一致性审计专家。快速检查本章是否与设定矛盾。

【世界观设定】
{world_setting[:2000]}

【角色设定】
{character_setting[:2000]}

【本章内容】
{chapter_text[:3000]}

请输出JSON（仅JSON）：
{{
  "findings": [
    {{
      "severity": "critical/major/minor/info",
      "category": "character/world/plot/timeline/power_system",
      "description": "问题描述（20字内）",
      "evidence": "原文引用（15字内）",
      "suggestion": "修复建议（20字内）"
    }}
  ],
  "overall_score": 95,
  "summary": "一句话总结"
}}

评分标准：
  critical = 角色死亡/世界观崩塌级矛盾，-20分
  major = 性格突变/力量体系矛盾，-10分
  minor = 小细节不一致，-3分
  info = 建议性提醒，不扣分

如果没有问题，findings 为空数组，overall_score 为 100。"""

        data = self._llm_invoke_json(prompt, is_array=False, llm=llm)
        if data:
            findings = [
                AuditFinding(
                    severity=f.get("severity", "minor"),
                    category=f.get("category", "plot"),
                    description=f.get("description", ""),
                    evidence=f.get("evidence", ""),
                    suggestion=f.get("suggestion", ""),
                    chapter=chapter_number,
                )
                for f in data.get("findings", [])
            ]
            report = AuditReport(
                project=project,
                chapter_range=(chapter_number, chapter_number),
                findings=findings,
                overall_score=data.get("overall_score", 100),
                summary=data.get("summary", ""),
                created_at=datetime.now().isoformat(),
            )
            self._save_report(report)
            return report

        return AuditReport(
            project=project, chapter_range=(chapter_number, chapter_number)
        )

    def run_full_audit(
        self,
        project: str,
        chapter_number: int,
        world_setting: str,
        character_setting: str,
        recent_chapters: list[str],
        llm: Any,
    ) -> AuditReport:
        """全量审计 — 每 10 章执行，检查最近 10 章的累积一致性。"""
        chapters_text = "\n\n---\n\n".join(
            f"第{i + 1}章：{t[:1500]}"
            for i, t in enumerate(recent_chapters[-AUDIT_FULL_WINDOW:])
        )

        prompt = f"""你是一位小说一致性审计专家。请对最近 10 章进行全量审计。

【世界观设定】
{world_setting[:AUDIT_FULL_SETTING_MAX]}

【角色设定】
{character_setting[:AUDIT_FULL_SETTING_MAX]}

【最近 10 章内容】
{chapters_text[:AUDIT_FULL_CHAPTERS_MAX]}

请输出JSON（仅JSON）：
{{
  "findings": [
    {{
      "severity": "critical/major/minor/info",
      "category": "character/world/plot/timeline/power_system",
      "description": "问题描述",
      "evidence": "原文引用",
      "suggestion": "修复建议",
      "chapter": 章节号
    }}
  ],
  "overall_score": 85,
  "summary": "审计总结（50字内）"
}}

重点检查：
1. 角色性格是否一致（如第3章沉稳→第8章暴躁，中间是否有触发事件）
2. 力量体系是否一致（如设定最高元婴期，但出现化神期角色）
3. 时间线是否连贯（如第5章是冬天→第7章突然夏天）
4. 已埋伏笔是否被遗忘（如第2章埋下的道具到第10章仍未提及）
5. 地理/势力关系是否矛盾

评分标准同上。输出纯JSON。"""

        data = self._llm_invoke_json(prompt, is_array=False, llm=llm)
        if data:
            findings = [
                AuditFinding(
                    severity=f.get("severity", "minor"),
                    category=f.get("category", "plot"),
                    description=f.get("description", ""),
                    evidence=f.get("evidence", ""),
                    suggestion=f.get("suggestion", ""),
                    chapter=f.get("chapter", 0),
                )
                for f in data.get("findings", [])
            ]
            report = AuditReport(
                project=project,
                chapter_range=(chapter_number - 9, chapter_number),
                findings=findings,
                overall_score=data.get("overall_score", 100),
                summary=data.get("summary", ""),
                created_at=datetime.now().isoformat(),
            )
            self._save_report(report)
            return report

        return AuditReport(
            project=project, chapter_range=(chapter_number - 9, chapter_number)
        )

    def _save_report(self, report: AuditReport) -> None:
        self._execute(
            """
            INSERT INTO novel_audit_reports
                (project_name, chapter_start, chapter_end, findings_json,
                 overall_score, summary, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """,
            (
                report.project,
                report.chapter_range[0],
                report.chapter_range[1],
                json.dumps(
                    [
                        {
                            "severity": f.severity,
                            "category": f.category,
                            "description": f.description,
                            "evidence": f.evidence,
                            "suggestion": f.suggestion,
                            "chapter": f.chapter,
                        }
                        for f in report.findings
                    ],
                    ensure_ascii=False,
                ),
                report.overall_score,
                report.summary,
            ),
        )

    def get_latest_report(self, project: str) -> AuditReport | None:
        row = self._fetchone(
            """
            SELECT chapter_start, chapter_end, findings_json, overall_score, summary, created_at
            FROM novel_audit_reports
            WHERE project_name = %s
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (project,),
        )
        if row:
            findings_data = (
                json.loads(row[2]) if isinstance(row[2], str) else (row[2] or [])
            )
            return AuditReport(
                project=project,
                chapter_range=(row[0], row[1]),
                findings=[
                    AuditFinding(
                        severity=f.get("severity", ""),
                        category=f.get("category", ""),
                        description=f.get("description", ""),
                        evidence=f.get("evidence", ""),
                        suggestion=f.get("suggestion", ""),
                        chapter=f.get("chapter", 0),
                    )
                    for f in findings_data
                ],
                overall_score=row[3] or 100,
                summary=row[4] or "",
                created_at=str(row[5]) if row[5] else "",
            )
        return None

    def get_score_trend(
        self, project: str, limit: int = AUDIT_SCORE_LIMIT
    ) -> list[float]:
        """获取审计评分趋势。"""
        rows = self._fetchall(
            """
            SELECT overall_score FROM novel_audit_reports
            WHERE project_name = %s
            ORDER BY created_at DESC
            LIMIT %s
        """,
            (project, limit),
        )
        return [row[0] for row in rows]

    def build_audit_prompt(self, project: str) -> str:
        """生成审计状态提示，注入 writer prompt。"""
        report = self.get_latest_report(project)
        if not report:
            return ""

        parts = [f"【一致性审计】最近评分：{report.overall_score:.0f}/100"]

        if report.findings:
            critical = [f for f in report.findings if f.severity == "critical"]
            major = [f for f in report.findings if f.severity == "major"]
            if critical:
                parts.append(f"  🔴 严重问题 ({len(critical)}):")
                for f in critical[:AUDIT_MAX_CRITICAL_DISPLAY]:
                    parts.append(f"    - {f.description} → {f.suggestion}")
            if major:
                parts.append(f"  🟡 重要问题 ({len(major)}):")
                for f in major[:AUDIT_MAX_MAJOR_DISPLAY]:
                    parts.append(f"    - {f.description} → {f.suggestion}")

        if report.summary:
            parts.append(f"  总结：{report.summary}")

        # 评分趋势
        scores = self.get_score_trend(project, AUDIT_TREND_DISPLAY_COUNT)
        if len(scores) >= AUDIT_MIN_TREND_POINTS:
            trend = (
                "上升"
                if scores[0] > scores[-1]
                else "下降"
                if scores[0] < scores[-1]
                else "稳定"
            )
            parts.append(
                f"  趋势：{trend}（{', '.join(f'{s:.0f}' for s in scores[:AUDIT_TREND_DISPLAY_COUNT])}）"
            )

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 伏笔管理系统
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Foreshadowing:
    """伏笔"""

    name: str  # 伏笔名称
    description: str  # 伏笔描述
    planted_chapter: int  # 埋设章节
    planned_resolve_chapter: int = 0  # 计划回收章节
    actual_resolve_chapter: int = 0  # 实际回收章节
    priority: int = 5  # 1-10，优先级
    status: str = "planted"  # planted / developing / resolved / abandoned
    related_characters: list[str] = field(default_factory=list)
    category: str = "plot"  # plot / character / item / mystery / relationship
    notes: str = ""  # 备注


class ForeshadowingManager(BaseManager):
    """伏笔管理器 — 追踪所有伏笔的生命周期。

    核心能力：
      1. 伏笔注册与追踪
      2. 优先级排序
      3. 到期提醒（超过计划回收章仍未回收）
      4. 伏笔密度分析
    """

    def register(self, project: str, fs: Foreshadowing) -> None:
        self._execute(
            """
            INSERT INTO novel_foreshadowing
                (project_name, name, description, planted_chapter,
                 planned_resolve_chapter, priority, status,
                 related_characters, category, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_name, name) DO UPDATE SET
                description = EXCLUDED.description,
                planned_resolve_chapter = EXCLUDED.planned_resolve_chapter,
                priority = EXCLUDED.priority,
                status = EXCLUDED.status,
                related_characters = EXCLUDED.related_characters,
                category = EXCLUDED.category,
                notes = EXCLUDED.notes,
                updated_at = NOW()
        """,
            (
                project,
                fs.name,
                fs.description,
                fs.planted_chapter,
                fs.planned_resolve_chapter,
                fs.priority,
                fs.status,
                json.dumps(fs.related_characters, ensure_ascii=False),
                fs.category,
                fs.notes,
            ),
        )

    def resolve(self, project: str, name: str, chapter: int) -> None:
        """标记伏笔已回收。"""
        self._execute(
            """
            UPDATE novel_foreshadowing
            SET status = 'resolved', actual_resolve_chapter = %s, updated_at = NOW()
            WHERE project_name = %s AND name = %s
        """,
            (chapter, project, name),
        )

    def get_overdue(self, project: str, current_chapter: int) -> list[Foreshadowing]:
        """获取已过期的伏笔（超过计划回收章仍未回收）。"""
        rows = self._fetchall(
            """
            SELECT name, description, planted_chapter, planned_resolve_chapter,
                   actual_resolve_chapter, priority, status, related_characters, category, notes
            FROM novel_foreshadowing
            WHERE project_name = %s
              AND status = 'planted'
              AND planned_resolve_chapter > 0
              AND planned_resolve_chapter < %s
            ORDER BY priority DESC, planned_resolve_chapter ASC
        """,
            (project, current_chapter),
        )
        return [self._row_to_fs(r) for r in rows]

    def get_upcoming(
        self, project: str, current_chapter: int, ahead: int = FORESHADOW_AHEAD
    ) -> list[Foreshadowing]:
        """获取即将到期的伏笔（未来 ahead 章内应回收）。"""
        rows = self._fetchall(
            """
            SELECT name, description, planted_chapter, planned_resolve_chapter,
                   actual_resolve_chapter, priority, status, related_characters, category, notes
            FROM novel_foreshadowing
            WHERE project_name = %s
              AND status = 'planted'
              AND planned_resolve_chapter > 0
              AND planned_resolve_chapter BETWEEN %s AND %s
            ORDER BY planned_resolve_chapter ASC
        """,
            (project, current_chapter, current_chapter + ahead),
        )
        return [self._row_to_fs(r) for r in rows]

    def get_all_active(self, project: str) -> list[Foreshadowing]:
        """获取所有活跃伏笔。"""
        rows = self._fetchall(
            """
            SELECT name, description, planted_chapter, planned_resolve_chapter,
                   actual_resolve_chapter, priority, status, related_characters, category, notes
            FROM novel_foreshadowing
            WHERE project_name = %s AND status IN ('planted', 'developing')
            ORDER BY priority DESC, planted_chapter ASC
        """,
            (project,),
        )
        return [self._row_to_fs(r) for r in rows]

    def get_stats(self, project: str) -> dict:
        """伏笔统计。"""
        row = self._fetchone(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'planted' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
                SUM(CASE WHEN status = 'abandoned' THEN 1 ELSE 0 END) as abandoned,
                AVG(priority) as avg_priority
            FROM novel_foreshadowing
            WHERE project_name = %s
        """,
            (project,),
        )
        return {
            "total": row[0] or 0,
            "active": row[1] or 0,
            "resolved": row[2] or 0,
            "abandoned": row[3] or 0,
            "avg_priority": round(row[4] or 0, 1),
        }

    def _row_to_fs(self, row: Any) -> Foreshadowing:
        chars = json.loads(row[7]) if isinstance(row[7], str) else (row[7] or [])
        return Foreshadowing(
            name=row[0],
            description=row[1] or "",
            planted_chapter=row[2] or 0,
            planned_resolve_chapter=row[3] or 0,
            actual_resolve_chapter=row[4] or 0,
            priority=row[5] or 5,
            status=row[6] or "planted",
            related_characters=chars,
            category=row[8] or "plot",
            notes=row[9] or "",
        )

    def extract_from_chapter(
        self, project: str, chapter_number: int, chapter_text: str, llm: Any
    ) -> list[Foreshadowing]:
        """用 LLM 从章节中提取伏笔。"""
        prompt = f"""从以下章节中提取伏笔信息。

章节内容：
{chapter_text[:FORESHADOW_INPUT_MAX]}

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

priority 评分：
  9-10 = 主线核心伏笔（主角身世/终极真相）
  7-8 = 重要伏笔（关键道具/重要人物关系）
  5-6 = 次要伏笔（支线线索/配角背景）
  1-4 = 小伏笔（短期回收）

输出纯JSON数组。"""

        data = self._llm_invoke_json(prompt, is_array=True, llm=llm)
        if not data:
            return []

        foreshadowings: list[Foreshadowing] = []
        for d in data:
            action = d.get("action", "planted")
            fs = Foreshadowing(
                name=d.get("name", ""),
                description=d.get("description", ""),
                planted_chapter=chapter_number,
                planned_resolve_chapter=d.get("planned_resolve_chapter", 0),
                priority=min(
                    10, max(1, d.get("priority", FORESHADOW_DEFAULT_PRIORITY))
                ),
                status="resolved" if action == "resolved" else "planted",
                related_characters=d.get("related_characters", []),
                category=d.get("category", "plot"),
            )
            if action == "resolved":
                fs.actual_resolve_chapter = chapter_number
            self.register(project, fs)
            foreshadowings.append(fs)
        return foreshadowings

    def build_foreshadowing_prompt(self, project: str, current_chapter: int) -> str:
        """生成伏笔状态提示，注入 writer prompt。"""
        parts: list[str] = []

        # 过期伏笔
        overdue = self.get_overdue(project, current_chapter)
        if overdue:
            parts.append("【⚠️ 过期未回收伏笔 — 请尽快处理】")
            for fs in overdue[:FORESHADOW_MAX_OVERDUE_DISPLAY]:
                parts.append(
                    f"  🔴 [{fs.name}] 第{fs.planted_chapter}章埋设，"
                    f"计划第{fs.planned_resolve_chapter}章回收 → 已过期"
                )

        # 即将到期
        upcoming = self.get_upcoming(project, current_chapter, FORESHADOW_AHEAD)
        if upcoming:
            parts.append("【📋 即将到期伏笔 — 请在近期章节回收】")
            for fs in upcoming[:FORESHADOW_MAX_UPCOMING_DISPLAY]:
                parts.append(
                    f"  🟡 [{fs.name}] 第{fs.planted_chapter}章埋设，"
                    f"计划第{fs.planned_resolve_chapter}章回收"
                )

        # 统计
        stats = self.get_stats(project)
        if stats["total"] > 0:
            parts.append(
                f"【伏笔统计】共{stats['total']}个 | "
                f"活跃{stats['active']} | 已回收{stats['resolved']} | "
                f"废弃{stats['abandoned']}"
            )

        return "\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 节奏控制系统
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PacingSnapshot:
    """节奏快照 — 一章的节奏分析"""

    chapter: int
    intensity: float = 5.0  # 紧张度 1-10
    event_density: float = 5.0  # 事件密度 1-10
    dialogue_ratio: float = 0.3  # 对话占比
    action_ratio: float = 0.3  # 动作描写占比
    description_ratio: float = 0.3  # 场景描写占比
    pacing_label: str = (
        "balanced"  # fast / balanced / slow / buildup / climax / cooldown
    )


class PacingController(BaseManager):
    """节奏控制器 — 检测并建议章节节奏。

    核心能力：
      1. 单章节奏分析
      2. 连续章节节奏趋势检测
      3. 节奏异常警告（连续过快/过慢）
      4. 节奏建议（何时需要高潮/缓冲）
    """

    # 节奏模式模板
    PACING_PATTERNS = {
        "高潮卷": ["buildup", "buildup", "climax", "cooldown", "balanced"],
        "日常卷": ["balanced", "balanced", "slow", "balanced", "balanced"],
        "战斗卷": ["buildup", "climax", "climax", "cooldown", "balanced"],
        "揭秘卷": ["slow", "buildup", "climax", "cooldown", "slow"],
        "过渡卷": ["balanced", "balanced", "balanced", "balanced", "balanced"],
    }

    def analyze_chapter(
        self, project: str, chapter_number: int, chapter_text: str, llm: Any
    ) -> PacingSnapshot:
        """分析单章节奏。"""
        prompt = f"""分析以下章节的节奏。

章节内容：
{chapter_text[:PACING_INPUT_MAX]}

输出JSON：
{{
  "intensity": 6.5,
  "event_density": 7.0,
  "dialogue_ratio": 0.3,
  "action_ratio": 0.4,
  "description_ratio": 0.3,
  "pacing_label": "buildup"
}}

评分标准：
  intensity: 1=极度舒缓, 5=正常, 10=极度紧张
  event_density: 1=几乎没有事件, 5=正常, 10=事件密集
  pacing_label: fast/balanced/slow/buildup/climax/cooldown

输出纯JSON。"""

        data = self._llm_invoke_json(prompt, is_array=False, llm=llm)
        if data:
            snap = PacingSnapshot(
                chapter=chapter_number,
                intensity=data.get("intensity", 5.0),
                event_density=data.get("event_density", 5.0),
                dialogue_ratio=data.get("dialogue_ratio", 0.3),
                action_ratio=data.get("action_ratio", 0.3),
                description_ratio=data.get("description_ratio", 0.3),
                pacing_label=data.get("pacing_label", "balanced"),
            )
            self._save_snapshot(project, snap)
            return snap

        return PacingSnapshot(chapter=chapter_number)

    def _save_snapshot(self, project: str, snap: PacingSnapshot) -> None:
        self._execute(
            """
            INSERT INTO novel_pacing_snapshots
                (project_name, chapter_number, intensity, event_density,
                 dialogue_ratio, action_ratio, description_ratio, pacing_label)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (project_name, chapter_number) DO UPDATE SET
                intensity = EXCLUDED.intensity,
                event_density = EXCLUDED.event_density,
                dialogue_ratio = EXCLUDED.dialogue_ratio,
                action_ratio = EXCLUDED.action_ratio,
                description_ratio = EXCLUDED.description_ratio,
                pacing_label = EXCLUDED.pacing_label
        """,
            (
                project,
                snap.chapter,
                snap.intensity,
                snap.event_density,
                snap.dialogue_ratio,
                snap.action_ratio,
                snap.description_ratio,
                snap.pacing_label,
            ),
        )

    def get_recent_pacing(self, project: str, count: int = 10) -> list[PacingSnapshot]:
        """获取最近 N 章的节奏快照。"""
        rows = self._fetchall(
            """
            SELECT chapter_number, intensity, event_density,
                   dialogue_ratio, action_ratio, description_ratio, pacing_label
            FROM novel_pacing_snapshots
            WHERE project_name = %s
            ORDER BY chapter_number DESC
            LIMIT %s
        """,
            (project, count),
        )
        return [
            PacingSnapshot(
                chapter=r[0],
                intensity=r[1] or 5.0,
                event_density=r[2] or 5.0,
                dialogue_ratio=r[3] or 0.3,
                action_ratio=r[4] or 0.3,
                description_ratio=r[5] or 0.3,
                pacing_label=r[6] or "balanced",
            )
            for r in rows
        ]

    def detect_pacing_issues(self, project: str) -> list[str]:
        """检测节奏问题。"""
        recent = self.get_recent_pacing(project, PACING_TREND_WINDOW)
        if len(recent) < PACING_MIN_SAMPLES:
            return []

        issues: list[str] = []
        recent_asc = list(reversed(recent))  # 按章节升序

        # 1. 连续高潮（>3章连续 climax/fast）
        climax_streak = 0
        for snap in recent_asc:
            if snap.pacing_label in ("climax", "fast"):
                climax_streak += 1
            else:
                climax_streak = 0
            if climax_streak >= PACING_CLIMAX_STREAK:
                issues.append(
                    f"连续{climax_streak}章高潮/快节奏，读者可能疲劳，建议插入缓冲章"
                )
                break

        # 2. 连续平淡（>5章连续 slow/balanced 且 intensity<4）
        slow_streak = 0
        for snap in recent_asc:
            if (
                snap.pacing_label in ("slow", "balanced")
                and snap.intensity < PACING_LOW_INTENSITY
            ):
                slow_streak += 1
            else:
                slow_streak = 0
            if slow_streak >= PACING_SLOW_STREAK:
                issues.append(
                    f"连续{slow_streak}章低强度，读者可能流失，建议安排高潮事件"
                )
                break

        # 3. 强度波动过大（相邻章 intensity 差 >5）
        for i in range(1, len(recent_asc)):
            diff = abs(recent_asc[i].intensity - recent_asc[i - 1].intensity)
            if diff > PACING_INTENSITY_DIFF:
                issues.append(
                    f"第{recent_asc[i - 1].chapter}→{recent_asc[i].chapter}章 "
                    f"强度跳跃{int(diff)}点，过渡可能生硬"
                )
                break

        return issues

    def build_pacing_prompt(self, project: str) -> str:
        """生成节奏状态提示，注入 writer prompt。"""
        recent = self.get_recent_pacing(project, 5)
        if not recent:
            return ""

        recent_asc = list(reversed(recent))

        parts = ["【节奏分析】"]

        # 最近 5 章节奏趋势
        labels = {
            "fast": "⚡快",
            "climax": "🔥高潮",
            "buildup": "📈铺垫",
            "balanced": "➡️平稳",
            "slow": "🐢舒缓",
            "cooldown": "❄️缓冲",
        }
        trend = " → ".join(
            labels.get(s.pacing_label, s.pacing_label) for s in recent_asc
        )
        parts.append(f"  趋势：{trend}")

        # 平均强度
        avg_intensity = sum(s.intensity for s in recent) / len(recent)
        parts.append(f"  平均强度：{avg_intensity:.1f}/10")

        # 问题检测
        issues = self.detect_pacing_issues(project)
        if issues:
            parts.append("  ⚠️ 节奏警告：")
            for issue in issues:
                parts.append(f"    - {issue}")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Phase2 统一入口
# ═══════════════════════════════════════════════════════════════════════════════


class Phase2Manager(BaseManager):
    """Phase 2 统一管理器 — 一致性审计 + 伏笔管理 + 节奏控制。

    在 ScaleManager 中使用：
        phase2 = Phase2Manager(pg_conn, project_name)

        # 写后
        phase2.after_chapter(chapter_number, chapter_text, world_setting, character_setting)

        # 写前
        context = phase2.build_writer_context(chapter_number)
    """

    def __init__(self, pg_conn: Any, project_name: str) -> None:
        super().__init__(pg_conn)
        self.project = project_name
        self.auditor = ConsistencyAuditor(pg_conn)
        self.foreshadowing = ForeshadowingManager(pg_conn)
        self.pacing = PacingController(pg_conn)

    def after_chapter(
        self,
        chapter_number: int,
        chapter_text: str,
        world_setting: str = "",
        character_setting: str = "",
    ) -> dict:
        """章节完成后运行所有 Phase 2 检查。"""
        result: dict[str, Any] = {
            "audit_score": None,
            "foreshadowing_extracted": 0,
            "pacing_label": "",
            "issues": [],
        }

        llm = self._get_llm()

        # 1. 一致性审计
        try:
            if self.auditor.should_full_audit(chapter_number):
                # 全量审计（每 10 章）
                report = self.auditor.run_full_audit(
                    self.project,
                    chapter_number,
                    world_setting,
                    character_setting,
                    [chapter_text],
                    llm,
                )
            else:
                # 快速审计
                report = self.auditor.run_quick_audit(
                    self.project,
                    chapter_number,
                    chapter_text,
                    world_setting,
                    character_setting,
                    llm,
                )
            result["audit_score"] = report.overall_score
            if report.findings:
                result["issues"].extend(
                    [f"[{f.severity}] {f.description}" for f in report.findings[:3]]
                )
        except Exception as e:
            logger.warning("[Phase2Manager] audit error: %s", e)

        # 2. 伏笔提取
        try:
            foreshadowings = self.foreshadowing.extract_from_chapter(
                self.project,
                chapter_number,
                chapter_text,
                llm,
            )
            result["foreshadowing_extracted"] = len(foreshadowings)
        except Exception as e:
            logger.warning("[Phase2Manager] foreshadowing error: %s", e)

        # 3. 节奏分析
        try:
            snap = self.pacing.analyze_chapter(
                self.project,
                chapter_number,
                chapter_text,
                llm,
            )
            result["pacing_label"] = snap.pacing_label
        except Exception as e:
            logger.warning("[Phase2Manager] pacing error: %s", e)

        return result

    def build_writer_context(self, chapter_number: int) -> str:
        """构建 Phase 2 上下文，注入 writer prompt。"""
        parts: list[str] = []

        # 审计状态
        audit_prompt = self.auditor.build_audit_prompt(self.project)
        if audit_prompt:
            parts.append(audit_prompt)

        # 伏笔状态
        fs_prompt = self.foreshadowing.build_foreshadowing_prompt(
            self.project, chapter_number
        )
        if fs_prompt:
            parts.append(fs_prompt)

        # 节奏状态
        pacing_prompt = self.pacing.build_pacing_prompt(self.project)
        if pacing_prompt:
            parts.append(pacing_prompt)

        return "\n\n".join(parts)
