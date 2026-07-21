"""知情辩论引擎 — 程序化分析结果注入辩论 prompt 的多轮辩论。

核心改进（v6.3）：
    1. 程序化报告 + 跨章信号注入辩论 prompt
    2. 辩论双方从"盲评"变为"知情辩论"
    3. 辩论结论回馈评分（debate_penalty）
    4. 完整 transcript 传递给翻修

普通函数编排，不是子图。VerdictEngine 内部调用。
"""

from __future__ import annotations

import json
import random

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra import async_llm_call_with_retry, get_logger
from novelfactory.evaluation.debate.parser import (
    parse_markdown_sections,
    parse_rebuttal,
)
from novelfactory.evaluation.debate.prompts import (
    CRITIC_REBUTTAL_PROMPT_INFORMED,
    CRITIC_REVIEW_PROMPT_INFORMED,
    EDITOR_REBUTTAL_PROMPT_INFORMED,
    EDITOR_REVIEW_PROMPT_INFORMED,
    READER_REBUTTAL_PROMPT_INFORMED,
    READER_REVIEW_PROMPT_INFORMED,
)
from novelfactory.evaluation.schemas import (
    CrossChapterSignals,
    DebateReport,
    PerspectiveReview,
    ProgrammaticReport,
    Rebuttal,
)
from novelfactory.evaluation.utils import (
    index_chapter_text,
    normalize_paragraph_refs,
)

logger = get_logger(__name__)

MAX_DEBATE_ROUNDS = 3  # v7.0: 1→3，支持多轮知情辩论

# 缺省收敛：连续2轮无新增问题即提前收敛
_CONVERGENCE_IDLE_ROUNDS = 2


class InformedDebateEngine:
    """知情辩论引擎 — 程序化结果注入的多轮辩论。

    普通函数编排，不是子图。VerdictEngine 内部调用。

    流程：
        1. 生成程序化分析摘要 + 跨章信号摘要
        2. 编辑评审（注入摘要）→ LLM ×1
        3. 读者评审（注入摘要 + 编辑意见）→ LLM ×1
        4. 多轮辩论（编辑反驳 → 读者反驳）→ LLM ×2/轮
        5. 融合产出 DebateReport
    """

    async def run(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prev_summary: str,
        programmatic: ProgrammaticReport,
        cross_chapter: CrossChapterSignals,
        llm: BaseChatModel,
    ) -> DebateReport:
        """执行知情辩论。

        Args:
            chapter_text: 章节文本
            genre: 题材
            genre_scoring_guide: 题材评分指南
            prev_summary: 前文摘要
            programmatic: 程序化分析报告
            cross_chapter: 跨章信号
            llm: LLM 实例

        Returns:
            DebateReport
        """
        # 1. 生成注入摘要
        prog_briefing = programmatic.to_debate_briefing()
        cross_briefing = cross_chapter.to_debate_briefing()
        prev_brief = prev_summary[:2000] if prev_summary else "（无前文）"

        # v7.3: Swap Operation — 随机化发言顺序消除 position bias
        # 参考 LLM-as-Judge Survey (ASU 2025, Section 4.2.1):
        #   pairwise 场景的做法是"同一个 Judge 判两次，交换顺序，不一致判 tie"。
        #   NovelFactory 是 3 角色辩论（非 pairwise），无法直接套用。
        #   替代方案：每轮随机化发言顺序 + 日志记录顺序供聚合审计。
        #   这能在统计层面消除顺序偏误，但单次无法标记 tie。
        speaker_order = ["editor", "reader", "critic"]
        random.shuffle(speaker_order)
        logger.info("[知情辩论] 发言顺序: %s", speaker_order)

        # 2. 首轮评审（按随机顺序）
        reviews: dict[str, PerspectiveReview] = {}
        for role in speaker_order:
            if role == "editor":
                reviews["editor"] = await self._editor_review(
                    chapter_text,
                    genre,
                    genre_scoring_guide,
                    prog_briefing,
                    cross_briefing,
                    prev_brief,
                    llm,
                )
            elif role == "reader":
                ed_review = reviews.get("editor")
                if ed_review is None:
                    ed_review = PerspectiveReview(
                        review_comments="",
                        issues=[],
                        strengths=[],
                        suggestions="",
                    )
                reviews["reader"] = await self._reader_review(
                    chapter_text,
                    genre,
                    genre_scoring_guide,
                    prog_briefing,
                    cross_briefing,
                    prev_brief,
                    ed_review,
                    llm,
                )
            elif role == "critic":
                ed_review = reviews.get("editor")
                rd_review = reviews.get("reader")
                reviews["critic"] = await self._critic_review(
                    chapter_text,
                    genre,
                    genre_scoring_guide,
                    prog_briefing,
                    cross_briefing,
                    prev_brief,
                    ed_review
                    or PerspectiveReview(
                        review_comments="",
                        issues=[],
                        strengths=[],
                        suggestions="",
                    ),
                    rd_review
                    or PerspectiveReview(
                        review_comments="",
                        issues=[],
                        strengths=[],
                        suggestions="",
                    ),
                    llm,
                )

        editor_review = reviews["editor"]
        reader_review = reviews["reader"]
        critic_review = reviews["critic"]

        # 4. 多轮辩论（3 角色）
        editor_rebuttals: list[Rebuttal] = []
        reader_rebuttals: list[Rebuttal] = []
        critic_rebuttals: list[Rebuttal] = []
        transcript_parts: list[str] = [
            f"[编辑·首轮] {editor_review.review_comments[:200]}",
            f"[读者·首轮] {reader_review.review_comments[:200]}",
            f"[评论员·首轮] {critic_review.review_comments[:200]}",
        ]

        convergence = False
        idle_rounds = 0  # 连续无新增问题轮次

        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            # v7.3: 反驳轮也随机化发言顺序
            rebuttal_order = random.sample(["editor", "reader", "critic"], 3)

            ed_rebuttal = rd_rebuttal = cr_rebuttal = None

            for role in rebuttal_order:
                if role == "editor":
                    ed_rebuttal = await self._editor_rebuttal(
                        round_num,
                        editor_review,
                        reader_review,
                        reader_rebuttals,
                        prog_briefing,
                        llm,
                    )
                    editor_rebuttals.append(ed_rebuttal)
                elif role == "reader":
                    rd_rebuttal = await self._reader_rebuttal(
                        round_num,
                        reader_review,
                        editor_rebuttals,
                        prog_briefing,
                        llm,
                    )
                    reader_rebuttals.append(rd_rebuttal)
                elif role == "critic":
                    cr_rebuttal = await self._critic_rebuttal(
                        round_num,
                        critic_review,
                        editor_rebuttals,
                        reader_rebuttals,
                        prog_briefing,
                        llm,
                    )
                    critic_rebuttals.append(cr_rebuttal)

            # 记录本轮所有反驳
            transcript_parts.append(f"[辩论·R{round_num}] 顺序: {rebuttal_order}")

            # 收敛判定 1: 三方无异议 → 提前收敛
            if (
                ed_rebuttal is not None
                and not ed_rebuttal.has_dissent
                and rd_rebuttal is not None
                and not rd_rebuttal.has_dissent
                and cr_rebuttal is not None
                and not cr_rebuttal.has_dissent
            ):
                convergence = True
                logger.info("[知情辩论] 第%d轮三方无异议，收敛", round_num)
                break

            # 收敛判定 2: 连续无新增问题 → 缺省收敛
            current_new = 0
            if ed_rebuttal is not None:
                current_new += len(ed_rebuttal.new_issues)
            if rd_rebuttal is not None:
                current_new += len(rd_rebuttal.new_issues)
            if cr_rebuttal is not None:
                current_new += len(cr_rebuttal.new_issues)
            if current_new == 0:
                idle_rounds += 1
            else:
                idle_rounds = 0
            if idle_rounds >= _CONVERGENCE_IDLE_ROUNDS:
                convergence = True
                logger.info(
                    "[知情辩论] 第%d轮连续%d轮无新增问题，缺省收敛",
                    round_num,
                    _CONVERGENCE_IDLE_ROUNDS,
                )
                break

        debate_rounds = len(editor_rebuttals)

        # 5. 融合产出
        return self._merge(
            editor_review,
            reader_review,
            critic_review,
            editor_rebuttals,
            reader_rebuttals,
            critic_rebuttals,
            debate_rounds,
            "\n".join(transcript_parts),
            convergence,
        )

    # ========== 内部方法 ==========

    async def _editor_review(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prog_briefing: str,
        cross_briefing: str,
        prev_brief: str,
        llm: BaseChatModel,
    ) -> PerspectiveReview:
        """编辑首轮评审 — 注入程序化摘要。"""
        # v7.0: 段落编号，统一评审↔润色引用格式
        indexed_text = index_chapter_text(chapter_text[:8000])
        prompt_parts = [
            EDITOR_REVIEW_PROMPT_INFORMED.format(
                programmatic_briefing=prog_briefing,
                cross_chapter_briefing=cross_briefing,
                prev_summary=prev_brief,
            ),
            "",
            f"体裁：{genre}",
        ]
        if genre_scoring_guide:
            prompt_parts.append(genre_scoring_guide)
        prompt_parts.extend(
            [
                "",
                "## 段落引用规则",
                "待评审章节已按 [P0][P1][P2]... 编号。请在问题列表和改进建议中",
                '引用段落编号，如 "[P3] 对话千人一言" 而非 "第4段"。',
                "",
                "## 待评审章节",
                indexed_text,
            ]
        )

        prompt = "\n".join(prompt_parts)
        response = await async_llm_call_with_retry(llm, prompt, step_name="editor_review_informed")
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_markdown_sections(raw)

        logger.info(
            "[editor_review_informed] 编辑评审完成 issues=%d strengths=%d",
            len(parsed.get("issues", [])),
            len(parsed.get("strengths", [])),
        )

        return PerspectiveReview(
            review_comments=normalize_paragraph_refs(parsed.get("review_comments", "")),
            issues=[normalize_paragraph_refs(i) for i in parsed.get("issues", [])],
            strengths=parsed.get("strengths", []),
            suggestions=normalize_paragraph_refs(parsed.get("suggestions", "")),
        )

    async def _reader_review(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prog_briefing: str,
        cross_briefing: str,
        prev_brief: str,
        editor_review: PerspectiveReview,
        llm: BaseChatModel,
    ) -> PerspectiveReview:
        """读者首轮评审 — 注入程序化摘要 + 编辑意见。"""
        editor_summary = json.dumps(
            {
                "comments": editor_review.review_comments[:300],
                "issues": editor_review.issues[:5],
                "strengths": editor_review.strengths[:5],
            },
            ensure_ascii=False,
        )

        # v7.0: 段落编号
        indexed_text = index_chapter_text(chapter_text[:8000])

        prompt_parts = [
            READER_REVIEW_PROMPT_INFORMED.format(
                programmatic_briefing=prog_briefing,
                cross_chapter_briefing=cross_briefing,
                prev_summary=prev_brief,
            ),
            "",
            f"体裁：{genre}",
        ]
        if genre_scoring_guide:
            prompt_parts.append(genre_scoring_guide)
        prompt_parts.extend(
            [
                f"编辑评审意见（供参考，请从读者角度补充或质疑）：\n{editor_summary}",
                "",
                "## 段落引用规则",
                "待评审章节已按 [P0][P1][P2]... 编号。请在问题列表和改进建议中",
                '引用段落编号，如 "[P3] AI味过重" 而非 "第4段"。',
                "",
                "## 待评审章节",
                indexed_text,
            ]
        )

        prompt = "\n".join(prompt_parts)
        response = await async_llm_call_with_retry(llm, prompt, step_name="reader_review_informed")
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_markdown_sections(raw)

        logger.info(
            "[reader_review_informed] 读者评审完成 issues=%d strengths=%d",
            len(parsed.get("issues", [])),
            len(parsed.get("strengths", [])),
        )

        return PerspectiveReview(
            review_comments=normalize_paragraph_refs(parsed.get("review_comments", "")),
            issues=[normalize_paragraph_refs(i) for i in parsed.get("issues", [])],
            strengths=parsed.get("strengths", []),
            suggestions=normalize_paragraph_refs(parsed.get("suggestions", "")),
        )

    # ── v7.3: Critic 评审 ─────────────────────────────────────────────

    async def _critic_review(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prog_briefing: str,
        cross_briefing: str,
        prev_brief: str,
        editor_review: PerspectiveReview,
        reader_review: PerspectiveReview,
        llm: BaseChatModel,
    ) -> PerspectiveReview:
        """评论员评审 — 对 Editor 和 Reader 进行二次验证。"""
        editor_summary = json.dumps(
            {
                "comments": editor_review.review_comments[:300],
                "issues": editor_review.issues[:5],
                "strengths": editor_review.strengths[:5],
            },
            ensure_ascii=False,
        )
        reader_summary = json.dumps(
            {
                "comments": reader_review.review_comments[:300],
                "issues": reader_review.issues[:5],
                "strengths": reader_review.strengths[:5],
            },
            ensure_ascii=False,
        )

        prompt_parts = [
            CRITIC_REVIEW_PROMPT_INFORMED.format(
                programmatic_briefing=prog_briefing,
                cross_chapter_briefing=cross_briefing,
                prev_summary=prev_brief,
            ),
            "",
            f"体裁：{genre}",
        ]
        if genre_scoring_guide:
            prompt_parts.append(genre_scoring_guide)
        prompt_parts.extend(
            [
                f"编辑评审意见：\n{editor_summary}",
                "",
                f"读者评审意见：\n{reader_summary}",
                "",
                "## 输出格式（必须使用 markdown 标题）",
                "## 数据验证\n...\n## 逻辑验证\n...\n## 维度混淆\n...\n## 遗漏检测\n...\n",
                "## 是否同意编辑\n...\n## 是否同意读者\n...\n## 新发现的问题列表\n- ...",
            ]
        )

        prompt = "\n".join(prompt_parts)
        response = await async_llm_call_with_retry(llm, prompt, step_name="critic_review_informed")
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_markdown_sections(raw)

        # Critic 的 issues 来自"新发现的问题列表"
        critic_issues = parsed.get("新发现的问题列表", [])
        if not critic_issues:
            critic_issues = parsed.get("issues", [])

        logger.info(
            "[critic_review_informed] Critic评审完成 issues=%d",
            len(critic_issues),
        )

        return PerspectiveReview(
            review_comments=parsed.get("review_comments", "")
            or f"数据验证: {parsed.get('数据验证', '无')}",
            issues=critic_issues,
            strengths=[],
            suggestions=parsed.get("改进建议", ""),
        )

    async def _critic_rebuttal(
        self,
        round_num: int,
        critic_review: PerspectiveReview,
        editor_rebuttals: list[Rebuttal],
        reader_rebuttals: list[Rebuttal],
        prog_briefing: str,
        llm: BaseChatModel,
    ) -> Rebuttal:
        """Critic 反驳。"""
        last_editor = editor_rebuttals[-1] if editor_rebuttals else None
        last_reader = reader_rebuttals[-1] if reader_rebuttals else None

        prompt_parts = [
            CRITIC_REBUTTAL_PROMPT_INFORMED.format(
                round=round_num,
                programmatic_briefing=prog_briefing,
            ),
            "",
            f"## 首轮 Critic 意见（你自己）\n{critic_review.review_comments[:300]}",
        ]
        if last_editor:
            prompt_parts.append(
                f"## 编辑最新反驳\n{last_editor.rebuttal_comments[:400]}"
            )
        if last_reader:
            prompt_parts.append(
                f"## 读者最新反驳\n{last_reader.rebuttal_comments[:400]}"
            )

        prompt = "\n".join(prompt_parts)
        response = await async_llm_call_with_retry(
            llm, prompt, step_name=f"critic_rebuttal_r{round_num}"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_rebuttal(raw)

        return Rebuttal(
            rebuttal_comments=normalize_paragraph_refs(
                parsed.get("rebuttal_comments", "")
            ),
            new_issues=[
                normalize_paragraph_refs(i) for i in parsed.get("new_issues", [])
            ],
            revised_suggestions=normalize_paragraph_refs(
                parsed.get("revised_suggestions", "")
            ),
            has_dissent=parsed.get("has_dissent", True),
        )

    async def _editor_rebuttal(
        self,
        round_num: int,
        editor_review: PerspectiveReview,
        reader_review: PerspectiveReview,
        reader_rebuttals: list[Rebuttal],
        prog_briefing: str,
        llm: BaseChatModel,
    ) -> Rebuttal:
        """编辑反驳。"""
        last_reader = reader_rebuttals[-1] if reader_rebuttals else None
        reader_summary = json.dumps(
            {
                "comments": reader_review.review_comments[:300],
                "issues": reader_review.issues[:5],
                "rebuttal": last_reader.rebuttal_comments[:400] if last_reader else "",
            },
            ensure_ascii=False,
        )

        prompt_parts = [
            EDITOR_REBUTTAL_PROMPT_INFORMED.format(
                round=round_num,
                programmatic_briefing=prog_briefing,
            ),
            "",
            f"## 首轮编辑意见（你自己）\n{editor_review.review_comments[:300]}",
            "",
            f"## 读者评审及辩论意见\n{reader_summary}",
        ]
        prompt = "\n".join(prompt_parts)

        response = await async_llm_call_with_retry(
            llm, prompt, step_name=f"editor_rebuttal_r{round_num}"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_rebuttal(raw)

        logger.info(
            "[editor_rebuttal_r%d] has_dissent=%s new_issues=%d",
            round_num,
            parsed["has_dissent"],
            len(parsed["new_issues"]),
        )

        return Rebuttal(
            rebuttal_comments=parsed["rebuttal_comments"],
            new_issues=parsed["new_issues"],
            revised_suggestions=parsed["revised_suggestions"],
            has_dissent=parsed["has_dissent"],
        )

    async def _reader_rebuttal(
        self,
        round_num: int,
        reader_review: PerspectiveReview,
        editor_rebuttals: list[Rebuttal],
        prog_briefing: str,
        llm: BaseChatModel,
    ) -> Rebuttal:
        """读者反驳。"""
        last_editor = editor_rebuttals[-1] if editor_rebuttals else None
        editor_summary = json.dumps(
            {
                "comments": last_editor.rebuttal_comments[:400] if last_editor else "",
                "new_issues": last_editor.new_issues[:3] if last_editor else [],
            },
            ensure_ascii=False,
        )

        prompt_parts = [
            READER_REBUTTAL_PROMPT_INFORMED.format(
                round=round_num,
                programmatic_briefing=prog_briefing,
            ),
            "",
            f"## 首轮读者意见（你自己）\n{reader_review.review_comments[:300]}",
            "",
            f"## 编辑反驳意见\n{editor_summary}",
        ]
        prompt = "\n".join(prompt_parts)

        response = await async_llm_call_with_retry(
            llm, prompt, step_name=f"reader_rebuttal_r{round_num}"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_rebuttal(raw)

        logger.info(
            "[reader_rebuttal_r%d] has_dissent=%s new_issues=%d",
            round_num,
            parsed["has_dissent"],
            len(parsed["new_issues"]),
        )

        return Rebuttal(
            rebuttal_comments=parsed["rebuttal_comments"],
            new_issues=parsed["new_issues"],
            revised_suggestions=parsed["revised_suggestions"],
            has_dissent=parsed["has_dissent"],
        )

    def _merge(
        self,
        editor_review: PerspectiveReview,
        reader_review: PerspectiveReview,
        critic_review: PerspectiveReview,
        editor_rebuttals: list[Rebuttal],
        reader_rebuttals: list[Rebuttal],
        critic_rebuttals: list[Rebuttal],
        debate_rounds: int,
        transcript: str,
        convergence: bool,
    ) -> DebateReport:
        """融合辩论产出 — 合并去重 issues/strengths/suggestions。

        v7.3: 新增 Critic 评审数据的合并。
        """
        # 合并 issues（首轮 + 各轮 new_issues，去重）
        all_issues: list[str] = []
        seen: set[str] = set()
        for issue in editor_review.issues + reader_review.issues + critic_review.issues:
            key = issue.strip()[:50]
            if key and key not in seen:
                all_issues.append(issue)
                seen.add(key)
        for rebuttal in editor_rebuttals + reader_rebuttals + critic_rebuttals:
            for issue in rebuttal.new_issues:
                key = issue.strip()[:50]
                if key and key not in seen:
                    all_issues.append(issue)
                    seen.add(key)

        # 合并 strengths（仅首轮，去重）
        all_strengths: list[str] = []
        seen_s: set[str] = set()
        for s in editor_review.strengths + reader_review.strengths:
            key = s.strip()[:50]
            if key and key not in seen_s:
                all_strengths.append(s)
                seen_s.add(key)

        # 合并 suggestions（首轮 + 各轮 revised_suggestions）
        suggestion_parts: list[str] = []
        if editor_review.suggestions:
            suggestion_parts.append(f"编辑视角·首轮：{editor_review.suggestions}")
        if reader_review.suggestions:
            suggestion_parts.append(f"读者视角·首轮：{reader_review.suggestions}")
        if critic_review.suggestions:
            suggestion_parts.append(f"评论员视角·首轮：{critic_review.suggestions}")
        for i, rebuttal in enumerate(editor_rebuttals):
            if rebuttal.revised_suggestions:
                suggestion_parts.append(
                    f"编辑视角·R{i + 1}：{rebuttal.revised_suggestions}"
                )
        for i, rebuttal in enumerate(reader_rebuttals):
            if rebuttal.revised_suggestions:
                suggestion_parts.append(
                    f"读者视角·R{i + 1}：{rebuttal.revised_suggestions}"
                )
        for i, rebuttal in enumerate(critic_rebuttals):
            if rebuttal.revised_suggestions:
                suggestion_parts.append(
                    f"评论员视角·R{i + 1}：{rebuttal.revised_suggestions}"
                )

        return DebateReport(
            editor_review=editor_review,
            reader_review=reader_review,
            editor_rebuttals=editor_rebuttals,
            reader_rebuttals=reader_rebuttals,
            debate_rounds=debate_rounds,
            debate_transcript=transcript,
            merged_issues=all_issues,
            merged_strengths=all_strengths,
            merged_suggestions="\n".join(suggestion_parts),
            convergence_achieved=convergence,
            debate_failed=False,
        )
