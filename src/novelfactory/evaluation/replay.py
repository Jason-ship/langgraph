"""检查点回溯重评服务。

利用 LangGraph 检查点历史，用新参数重新评审历史章节，
对比调参前后的 VerdictResult，验证参数变更效果。

流程：
  1. graph.aget_state_history() 遍历检查点
  2. 找到目标章节的检查点（含 chapter_draft）
  3. 提取 chapter_draft、genre、prev_summary、原 verdict_result
  4. 用当前 quality_center 参数调用 ReviewService.evaluate()
  5. 对比新旧 VerdictResult，生成差异报告

版本：v1.0.0
创建日期：2026-08-12
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from novelfactory.evaluation.schemas import VerdictResult
from novelfactory.evaluation.service import ReviewService

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  ReplayReport 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReplayReport:
    """单章重评对比报告。"""

    chapter_number: int
    success: bool
    error: str = ""
    old_score: float = 0.0
    new_score: float = 0.0
    delta: float = 0.0
    old_level: str = ""
    new_level: str = ""
    level_changed: bool = False
    dimension_diffs: dict[str, dict[str, float]] = field(default_factory=dict)
    toxic_added: list[str] = field(default_factory=list)
    toxic_removed: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def is_improved(self) -> bool:
        """参数变更是否带来了改善。"""
        if self.delta > 0:
            return True
        if self.level_changed:
            order = {"REWRITE": 0, "REFINE": 1, "PASS": 2}
            return order.get(self.new_level.upper(), 0) > order.get(self.old_level.upper(), 0)
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_number": self.chapter_number,
            "success": self.success,
            "error": self.error,
            "old_score": round(self.old_score, 2),
            "new_score": round(self.new_score, 2),
            "delta": round(self.delta, 2),
            "old_level": self.old_level,
            "new_level": self.new_level,
            "level_changed": self.level_changed,
            "dimension_diffs": self.dimension_diffs,
            "toxic_added": self.toxic_added,
            "toxic_removed": self.toxic_removed,
            "summary": self.summary,
            "is_improved": self.is_improved,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CheckpointReplayService
# ═══════════════════════════════════════════════════════════════════════════════


class CheckpointReplayService:
    """检查点回溯重评服务。

    利用 LangGraph 检查点历史，用新参数重新评审历史章节，
    对比调参前后的 VerdictResult，验证参数变更效果。
    """

    def __init__(self) -> None:
        self._review_service = ReviewService()

    async def replay_chapter(
        self,
        thread_id: str,
        chapter_number: int,
        config: RunnableConfig | None = None,
    ) -> ReplayReport:
        """回溯到指定章节的检查点，用当前参数重新评审。

        Args:
            thread_id: 项目线程 ID
            chapter_number: 要重评的章节号
            config: 可选的 RunnableConfig

        Returns:
            ReplayReport: 包含旧评分、新评分、差异分析
        """
        graph = await self._get_graph()
        if graph is None:
            return ReplayReport(
                chapter_number=chapter_number,
                success=False,
                error="Graph not available",
            )

        if config is None:
            config = {"configurable": {"thread_id": thread_id}}

        # 1. 遍历检查点历史，找到目标章节
        target_snapshot = None
        try:
            async for snapshot in graph.aget_state_history(config, limit=200):
                state = snapshot.values if hasattr(snapshot, "values") else {}
                if not isinstance(state, dict):
                    continue
                ch = state.get("current_chapter")
                if ch == chapter_number:
                    chapter_text = state.get("chapter_draft") or state.get("refined_chapter", "")
                    if chapter_text and len(str(chapter_text)) > 100:
                        target_snapshot = snapshot
                        break
        except Exception as e:
            logger.exception("[Replay] Failed to get state history: %s", e)
            return ReplayReport(
                chapter_number=chapter_number,
                success=False,
                error=f"检查点遍历失败: {e}",
            )

        if target_snapshot is None:
            return ReplayReport(
                chapter_number=chapter_number,
                success=False,
                error=f"未找到第{chapter_number}章含章节文本的检查点",
            )

        old_state = target_snapshot.values if hasattr(target_snapshot, "values") else {}

        # 2. 提取评审上下文
        chapter_text = old_state.get("chapter_draft") or old_state.get("refined_chapter", "")
        genre = old_state.get("genre", "")
        crew_result = old_state.get("crew_result", {})
        if isinstance(crew_result, dict):
            prev_summary = crew_result.get("prev_chapters_summary", "")
        else:
            prev_summary = ""

        # 3. 提取旧评审结果
        old_verdict_dict = old_state.get("verdict_result", {})
        old_verdict: VerdictResult | None = None
        if old_verdict_dict and isinstance(old_verdict_dict, dict):
            try:
                old_verdict = VerdictResult(**old_verdict_dict)
            except Exception as e:
                logger.warning("[Replay] Failed to reconstruct old VerdictResult: %s", e)

        # 4. 用当前参数重新评审
        try:
            new_verdict = await self._review_service.evaluate(
                chapter_text=str(chapter_text),
                genre=genre,
                prev_summary=prev_summary,
                chapter_index=chapter_number,
            )
        except Exception as e:
            logger.exception("[Replay] Re-evaluation failed for chapter %d: %s", chapter_number, e)
            return ReplayReport(
                chapter_number=chapter_number,
                success=False,
                error=f"重评失败: {e}",
            )

        # 5. 生成对比报告
        return self._build_report(chapter_number, old_verdict, new_verdict)

    async def replay_recent(
        self,
        thread_id: str,
        count: int = 3,
    ) -> list[ReplayReport]:
        """回溯最近 N 章的检查点，批量重评。

        Args:
            thread_id: 项目线程 ID
            count: 重评章节数（从最新章节往前数）

        Returns:
            多个 ReplayReport 列表（按章节号升序）
        """
        graph = await self._get_graph()
        if graph is None:
            return []

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # 找到最近 N 个有 chapter_draft 的章节
        chapters_to_replay: list[int] = []
        seen_chapters: set[int] = set()

        try:
            async for snapshot in graph.aget_state_history(config, limit=500):
                state = snapshot.values if hasattr(snapshot, "values") else {}
                if not isinstance(state, dict):
                    continue
                ch = state.get("current_chapter")
                if ch and ch not in seen_chapters:
                    chapter_text = state.get("chapter_draft") or state.get("refined_chapter", "")
                    if chapter_text and len(str(chapter_text)) > 100:
                        chapters_to_replay.append(ch)
                        seen_chapters.add(ch)
                        if len(chapters_to_replay) >= count:
                            break
        except Exception as e:
            logger.exception("[Replay] Failed to scan history: %s", e)
            return []

        # 逐章重评
        reports: list[ReplayReport] = []
        for ch in sorted(chapters_to_replay):
            report = await self.replay_chapter(thread_id, ch, config)
            reports.append(report)

        return reports

    # ── 内部方法 ──────────────────────────────────────────────────────────

    async def _get_graph(self) -> Any:
        """获取编译后的图实例。"""
        try:
            from novelfactory.server.deps import get_graph
            return await get_graph()
        except Exception as e:
            logger.warning("[Replay] Failed to get graph: %s", e)
            return None

    def _build_report(
        self,
        chapter_number: int,
        old: VerdictResult | None,
        new: VerdictResult,
    ) -> ReplayReport:
        """构建新旧评审对比报告。"""
        if old is None:
            summary = f"第{chapter_number}章：无历史评审数据，当前评分 {new.final_score:.1f}（{new.level.value}）"
            return ReplayReport(
                chapter_number=chapter_number,
                success=True,
                old_score=0.0,
                new_score=new.final_score,
                delta=new.final_score,
                old_level="N/A",
                new_level=new.level.value,
                summary=summary,
            )

        # 核心分数对比
        score_fields: dict[str, str] = {
            "final_score": "融合分",
            "quality_score": "四维评分",
            "programmatic_score": "程序化分",
            "cross_chapter_consistency": "跨章一致性",
            "ai_style_score": "AI味指数",
            "lao_shu_chong_score": "老书虫分",
            "llm_semantic_score": "LLM老书虫",
            "llm_human_like_score": "LLM人感",
            "llm_attraction_score": "吸引力分",
        }

        dimension_diffs: dict[str, dict[str, float]] = {}
        for field_name, label in score_fields.items():
            old_val = float(getattr(old, field_name, 0.0) or 0.0)
            new_val = float(getattr(new, field_name, 0.0) or 0.0)
            delta = new_val - old_val
            if abs(delta) > 0.01:
                dimension_diffs[label] = {
                    "old": round(old_val, 2),
                    "new": round(new_val, 2),
                    "delta": round(delta, 2),
                }

        # 级别变化
        level_changed = old.level != new.level

        # 毒点/爽点变化
        old_toxic: set[str] = set()
        new_toxic: set[str] = set()
        if old.feedback and hasattr(old.feedback, "toxic_points"):
            old_toxic = set(old.feedback.toxic_points or [])
        if new.feedback and hasattr(new.feedback, "toxic_points"):
            new_toxic = set(new.feedback.toxic_points or [])
        toxic_added = list(new_toxic - old_toxic)
        toxic_removed = list(old_toxic - new_toxic)

        summary = self._generate_summary(
            chapter_number, old, new, dimension_diffs,
            level_changed, toxic_added, toxic_removed,
        )

        return ReplayReport(
            chapter_number=chapter_number,
            success=True,
            old_score=old.final_score,
            new_score=new.final_score,
            delta=new.final_score - old.final_score,
            old_level=old.level.value,
            new_level=new.level.value,
            level_changed=level_changed,
            dimension_diffs=dimension_diffs,
            toxic_added=toxic_added,
            toxic_removed=toxic_removed,
            summary=summary,
        )

    def _generate_summary(
        self,
        chapter_number: int,
        old: VerdictResult,
        new: VerdictResult,
        diffs: dict[str, dict[str, float]],
        level_changed: bool,
        toxic_added: list[str],
        toxic_removed: list[str],
    ) -> str:
        """生成人类可读的对比摘要。"""
        lines: list[str] = []
        delta_str = f"+{new.final_score - old.final_score:.1f}" if new.final_score >= old.final_score else f"{new.final_score - old.final_score:.1f}"
        lines.append(
            f"第{chapter_number}章：融合分 {old.final_score:.1f} -> {new.final_score:.1f} ({delta_str})"
        )

        if level_changed:
            lines.append(f"  级别变化：{old.level.value} -> {new.level.value}")

        # 维度变化（只显示变化大的）
        for label, vals in diffs.items():
            d = vals["delta"]
            arrow = "↑" if d > 0 else "↓"
            lines.append(f"  {label}：{vals['old']:.1f} -> {vals['new']:.1f} ({arrow}{abs(d):.1f})")

        if toxic_added:
            lines.append(f"  新增检测毒点：{toxic_added}")
        if toxic_removed:
            lines.append(f"  消除毒点：{toxic_removed}")

        # 总体判断
        if new.final_score > old.final_score + 2:
            lines.append("  ✅ 参数变更效果正面，评分提升")
        elif new.final_score < old.final_score - 2:
            lines.append("  ⚠️ 参数变更效果负面，评分下降")
        else:
            lines.append("  ➡️ 参数变更影响较小，评分基本持平")

        return "\n".join(lines)
