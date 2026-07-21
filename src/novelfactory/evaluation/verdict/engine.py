"""VerdictEngine — 评分融合引擎（核心 v7.1）。

将四维 LLM 评分、程序化分析、LLM 语义分析、知情辩论融合为统一 VerdictResult。

融合公式 v7.1（新增 LLM 语义分析维度）：
    final_score = quality_score × W_QUALITY
                + programmatic_normalized × W_PROGRAMMATIC
                + llm_old_reader_score × W_LLM_OLD_READER    ← NEW
                + llm_human_like_score × W_LLM_HUMAN_LIKE    ← NEW
                + cross_chapter_consistency × W_CROSS_CHAPTER
                - debate_penalty × W_DEBATE_PENALTY

设计参考：
    - WebNovelBench (ACL 2025)：LLM-as-Judge 的 8-dimension 叙事质量评估
    - WritingBench (NeurIPS 2025)：criteria-aware scoring 模式
    - ChineseHarm-Bench (2025)：语义级中文有害内容检测
    - EQ-Bench Creative Writing v3：voice consistency & emotional nuance

不是子图，不需要 LangGraph 编排。VerdictEngine 节点内部调用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from novelfactory.agents.infra import async_llm_call_with_retry, get_logger
from novelfactory.agents.infra.serialization import validate_json_output
from novelfactory.config.constants import (
    VERDICT_ITERATION_BONUS_MAX,
    VERDICT_ITERATION_BONUS_REFINE,
    VERDICT_ITERATION_BONUS_REWRITE,
    VERDICT_LENGTH_NORMALIZE,
    VERDICT_NORMALIZE_BASE,
    VERDICT_PASS_THRESHOLD,
    VERDICT_REFINE_THRESHOLD,
    VERDICT_WEIGHTS,
)
from novelfactory.evaluation.debate.engine import InformedDebateEngine
from novelfactory.evaluation.llm.ai_style_llm import llm_ai_style_analysis
from novelfactory.evaluation.llm.old_reader_llm import llm_old_reader_analysis
from novelfactory.evaluation.programmatic.runner import run_programmatic_analysis
from novelfactory.evaluation.schemas import (
    AttemptInfo,
    CrossChapterSignals,
    DebateReport,
    EvidenceItem,
    FeedbackBundle,
    FourDimReviewResult,
    ProgrammaticReport,
    VerdictLevel,
    VerdictResult,
)
from novelfactory.evaluation.utils import index_chapter_text, normalize_paragraph_refs
from novelfactory.evaluation.verdict.calibration import CalibrationModule, CalibrationResult
from novelfactory.evaluation.verdict.feedback import FeedbackBuilder
from novelfactory.schemas.review_schemas import FourDimScores

if TYPE_CHECKING:
    from novelfactory.evaluation.llm.schemas import LLMAIStyleResult, LLMOldReaderResult

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  四维评分上下文裁剪（私有常量）
# ═══════════════════════════════════════════════════════════════════════════════
_REVIEW_MAX_TOTAL = 8000
_REVIEW_HEAD_TAIL_SIZE = 1500
_REVIEW_SAMPLE_COUNT = 3
_REVIEW_SAMPLE_SIZE = 1000

_QUALITY_SCORE_MIN = 0.0
_QUALITY_SCORE_MAX = 100.0

# ═══════════════════════════════════════════════════════════════════════════════
#  质量衰减检测（v7.2 基于 Fiction_Eval "高开低走"模式）
# ═══════════════════════════════════════════════════════════════════════════════
_DECAY_HEAD_RATIO = 0.35  # 前段比例
_DECAY_TAIL_RATIO = 0.35  # 后段比例
_DECAY_PENALTY_PER_POINT = 0.5  # 每1分衰减扣分系数
_DECAY_MAX_PENALTY = 10.0  # 衰减惩罚上限


def _detect_quality_decay(chapter_text: str) -> float:
    """检测章节文本的"高开低走"质量衰减。

    基于 Fiction_Eval (ACL 2025) 的实证发现：
      LLM 生成小说的优质内容集中在前 40%-60%，
      后段质量显著下降。中文中文比英文更严重。

    使用程序化指标（句长标准差、模板词密度、词汇多样性）
    对前段和后段分别分析，计算衰减幅度。

    Returns:
        衰减惩罚 (0-10)，0=无衰减，10=严重衰减
    """
    if not chapter_text or len(chapter_text) < 500:
        return 0.0

    n = len(chapter_text)
    head_end = int(n * _DECAY_HEAD_RATIO)
    tail_start = int(n * (1.0 - _DECAY_TAIL_RATIO))

    head_text = chapter_text[:head_end]
    tail_text = chapter_text[tail_start:]

    if len(head_text) < 100 or len(tail_text) < 100:
        return 0.0

    try:
        from novelfactory.analysis.ai_style_analyzer import analyze_ai_style

        head_result = analyze_ai_style(head_text)
        tail_result = analyze_ai_style(tail_text)

        head_score = head_result["ai_style_score"]
        tail_score = tail_result["ai_style_score"]

        # 衰减 = 后段AI味 - 前段AI味（正值表示后段更AI）
        decay = max(0.0, tail_score - head_score - 0.05)  # 0.05 容差

        # 额外检测：后段词汇多样性下降
        head_diversity = head_result["metrics"].get("lexical_diversity", 0.5)
        tail_diversity = tail_result["metrics"].get("lexical_diversity", 0.5)
        diversity_decay = max(0.0, head_diversity - tail_diversity - 0.05)

        combined_decay = decay * 0.6 + diversity_decay * 0.4
        penalty = min(
            _DECAY_MAX_PENALTY, combined_decay * _DECAY_PENALTY_PER_POINT * 10
        )

        if penalty > 1.0:
            logger.info(
                "[质量衰减] 检测到高开低走: AI味前%.2f→后%.2f, "
                "多样性前%.2f→后%.2f, 惩罚%.1f",
                head_score,
                tail_score,
                head_diversity,
                tail_diversity,
                penalty,
            )
        return penalty

    except Exception as e:
        logger.debug("[质量衰减] 检测异常: %s", e)
        return 0.0


@dataclass(frozen=True, slots=True)
class _ToxicState:
    """毒点分析状态 — 一次性计算供 _fuse() 内多处引用。

    当 LLM 语义分析与程序化传感器对毒点判定矛盾时，
    动态调整权重以避免关键词误报拖累总分。
    """

    llm_severe_toxic: bool
    llm_analysis_healthy: bool
    prog_toxic_overridden: bool
    w_prog: float
    w_llm_or: float


class VerdictEngine:
    """评分融合引擎 — 将四维 LLM 评分、LLM 语义分析、程序化分析、知情辩论融合为统一决议。

    融合公式 v7.1（新增 LLM 老书虫 + LLM AI味语义分析）：
        final_score = quality_score × W_QUALITY
                    + programmatic_normalized × W_PROGRAMMATIC
                    + llm_old_reader_score × W_LLM_OLD_READER    ← LLM语义老书虫
                    + llm_human_like_score × W_LLM_HUMAN_LIKE    ← LLM语义AI味
                    + cross_chapter_consistency × W_CROSS_CHAPTER
                    - debate_penalty × W_DEBATE_PENALTY

    决策规则（3 级，替代 12 分支）：
        1. LLM严重毒点 + 未用尽重写 → REWRITE
        2. 程序化严重毒点 + 未用尽重写 → REWRITE
        3. final_score >= PASS_THRESHOLD → PASS
        4. final_score >= REFINE_THRESHOLD → REFINE
        5. final_score < REFINE_THRESHOLD → REWRITE
        6. 任何次数用尽 → PASS（防死循环兜底）
    """

    def __init__(self) -> None:
        self._calibration = CalibrationModule()
        self._feedback_builder = FeedbackBuilder()
        self._debate_engine = InformedDebateEngine()

    async def evaluate(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prev_summary: str,
        chapter_index: int,
        attempt_info: AttemptInfo,
        reviewer_llm: BaseChatModel,
        debate_llm: BaseChatModel,
    ) -> VerdictResult:
        """执行完整评审流程，返回统一决议（async）。

        v7.1：新增 LLM 语义分析步骤（老书虫 + AI味），
        与程序化分析并行互补。

        v7.8: 全 async — LLM 调用不再阻塞事件循环。

        Args:
            chapter_text: 章节文本
            genre: 题材
            genre_scoring_guide: 题材评分指南
            prev_summary: 前文摘要
            chapter_index: 章节序号
            attempt_info: 重写/润色次数追踪
            reviewer_llm: 四维评分 + LLM 语义分析 LLM
            debate_llm: 辩论 LLM

        Returns:
            VerdictResult
        """
        logger.info(
            "[VerdictEngine] ch%d 开始评审 | rewrite=%d/%d refine=%d/%d",
            chapter_index,
            attempt_info.loop_count,
            attempt_info.max_rewrite,
            attempt_info.refine_attempts,
            attempt_info.max_refine,
        )

        # 1. 程序化分析（纯代码，毫秒级 — 快速传感器）
        programmatic, cross_chapter = run_programmatic_analysis(
            chapter_text=chapter_text,
            chapter_index=chapter_index,
            genre=genre,
            prev_chapters_summary=prev_summary,
        )

        # 2. LLM 语义分析（并行轨道 — LLM 老书虫 + LLM AI味）
        # v7.2: 多模型分层支持 — 不同 LLM 执行不同维度的分析
        # 参考 Fiction_Eval 实证：Claude 擅长宏观, DeepSeek 擅长中观, GPT-4o 擅长微观
        # v7.8: async — await 所有 LLM 调用
        llm_or = await llm_old_reader_analysis(
            chapter_text=chapter_text,
            genre=genre,
            prev_summary=prev_summary,
            llm=reviewer_llm,
        )
        llm_ais = await llm_ai_style_analysis(
            chapter_text=chapter_text,
            genre=genre,
            programmatic_metrics=programmatic.ai_style_metrics.to_brief_string(),
            llm=reviewer_llm,
        )

        # v7.2: 质量衰减检测（Fiction_Eval"高开低走"模式）
        decay_penalty = _detect_quality_decay(chapter_text)

        # 3. 知情辩论（LLM，注入程序化结果 + LLM语义分析）
        debate = await self._run_debate(
            chapter_text,
            genre,
            genre_scoring_guide,
            prev_summary,
            programmatic,
            cross_chapter,
            debate_llm,
        )

        # 4. 四维 LLM 评分（只调一次！）
        four_dim = await self._run_four_dim_review(
            chapter_text,
            genre,
            genre_scoring_guide,
            prev_summary,
            cross_chapter,
            reviewer_llm,
        )

        verdict = self._fuse(
            four_dim,
            programmatic,
            debate,
            cross_chapter,
            attempt_info,
            llm_old_reader=llm_or,
            llm_ai_style=llm_ais,
            decay_penalty=decay_penalty,
            chapter_length=len(chapter_text),
        )

        logger.info(
            "[VerdictEngine] ch%d 评审完成 | level=%s final=%.1f "
            "quality=%.1f prog=%.3f llm_or=%.0f llm_ais=%.0f "
            "cross=%.1f debate_penalty=%.1f decay=%.1f",
            chapter_index,
            verdict.level.value,
            verdict.final_score,
            verdict.quality_score,
            verdict.programmatic_score,
            llm_or.semantic_score if not llm_or.failed else 0,
            llm_ais.human_like_score if not llm_ais.failed else 0,
            verdict.cross_chapter_consistency,
            verdict.debate_penalty,
            decay_penalty,
        )

        return verdict

    # ========== 内部方法 ==========

    async def _run_debate(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prev_summary: str,
        programmatic: ProgrammaticReport,
        cross_chapter: CrossChapterSignals,
        llm: BaseChatModel,
    ) -> DebateReport:
        """执行知情辩论（async）。"""
        try:
            return await self._debate_engine.run(
                chapter_text=chapter_text,
                genre=genre,
                genre_scoring_guide=genre_scoring_guide,
                prev_summary=prev_summary,
                programmatic=programmatic,
                cross_chapter=cross_chapter,
                llm=llm,
            )
        except Exception as e:
            logger.warning("[VerdictEngine] 辩论失败，降级: %s", e)
            return DebateReport(
                debate_failed=True,
                convergence_achieved=False,
                merged_issues=[],
                merged_strengths=[],
                merged_suggestions="",
                debate_transcript=f"辩论失败: {e}",
            )

    async def _run_four_dim_review(
        self,
        chapter_text: str,
        genre: str,
        genre_scoring_guide: str,
        prev_summary: str,
        cross_chapter: CrossChapterSignals,
        llm: BaseChatModel,
        use_close_prompt: bool = False,
    ) -> FourDimReviewResult:
        """执行四维 LLM 评分（async）。

        v7.3: 嵌入一致性检查清单（作为内部推理引导），输出证据链。
        v7.4: 新增 use_close_prompt 参数 — 两步模式：
          步骤 1（use_close_prompt=True）: SCORE_ONLY_PROMPT（仅 JSON 裸分）
          步骤 2（可选）: ANALYSIS_PROMPT（调试用）
        v7.8: async — LLM 调用不再阻塞事件循环。
        """
        from novelfactory.evaluation.llm.prompts import (
            CONSISTENCY_CHECKLIST,
            SCORE_ONLY_PROMPT,
        )

        trimmed = self._trim_for_review(chapter_text)
        indexed_trimmed = index_chapter_text(trimmed)

        # ── Close Prompt 模式: 先拿裸分（参考 Reference-Guided Verdict 2024）──
        if use_close_prompt:
            score_prompt = SCORE_ONLY_PROMPT
            score_response = await async_llm_call_with_retry(
                llm, score_prompt, step_name="four_dim_score_only"
            )
            score_raw = (
                score_response.content
                if hasattr(score_response, "content")
                else str(score_response)
            )
            score_result = self._parse_four_dim_response(score_raw)

            # 如果 close prompt 解析成功 → 直接返回裸分结果
            if not score_result.failed:
                return score_result

        # ── 详细分析模式（默认）: 评分+检查清单+证据链 ──
        prompt_parts: list[str] = [
            "你是专业小说编辑，对章节进行四维评分 + 跨章一致性评分。",
            "",
            f"## 前文上下文摘要\n{prev_summary[:2000] if prev_summary else '（无前文）'}",
            "",
            f"## 跨章统计信号（客观数据，需你判断含义）\n{cross_chapter.to_debate_briefing()}",
            "",
        ]
        if genre_scoring_guide:
            prompt_parts.append(f"## 题材感知评分指引\n{genre_scoring_guide}")
            prompt_parts.append("")

        # v7.3: 嵌入一致性检查清单（作为内部推理辅助，不要求输出）
        prompt_parts.extend(
            [
                CONSISTENCY_CHECKLIST,
                "",
            ]
        )

        prompt_parts.extend(
            [
                "## 评分维度（每维0-100）",
                "1. 剧情逻辑(30分)：矛盾/漏洞/逻辑硬伤",
                "2. 文笔表达(25分)：描写质量/感官细节/修辞",
                "3. 人物一致性(25分)：性格/对话/动机",
                "4. 世界观契合(20分)：设定违和",
                "5. 跨章一致性(100分制独立打分)：角色声音/文风/节奏/情节连贯/伏笔",
                "",
                "## 评分校准规则",
                "- AI初稿不可能完美，90+仅给极少数卓越章节",
                "- 流水账段落→总分≤88，感官空白→≤85，逻辑漏洞→≤70",
                "- 绝对禁止满分100",
                "",
                "## 跨章判断要求",
                "- 句长变化是角色成长还是声音漂移？",
                "- 文风偏移是否影响阅读体验？",
                "- 前文未解释元素本章是否需要呼应？",
                "- 与前文设定有无矛盾？",
                "",
                "## 段落引用规则（重要）",
                "待评审章节已按 [P0][P1][P2]... 编号。请在评审意见中引用段落编号，",
                '如 "[P3] 对话千人一言" 而非 "第4段"。',
                "",
                f"## 待评审章节\n{indexed_trimmed}",
                "",
                "## 输出JSON",
                '{"quality_score": <四维总分>, "review_comments": "<具体段落问题（引用 [Pi]）>", '
                '"cross_chapter_consistency": <0-100>, '
                '"cross_chapter_issues": ["<跨章问题1（引用 [Pi]）>", ...], '
                '"evidence_chain": ['
                '  {"issue": "<问题描述>", "span_a": "<矛盾片段A（含 [Pi]）>", '
                '   "span_b": "<矛盾片段B（含 [Pi]）>", "reasoning": "<推理说明>", '
                '   "error_type": "<时间矛盾|角色矛盾|世界观违反|细节不一致|风格偏离>"}'
                "]}",
            ]
        )

        prompt = "\n".join(prompt_parts)

        try:
            response = await async_llm_call_with_retry(llm, prompt, step_name="four_dim_review")
            raw = response.content if hasattr(response, "content") else str(response)
            result = self._parse_four_dim_response(raw)
            # v7.0: 归一化任何 "第N段" 引用为 [Pi]
            result.review_comments = normalize_paragraph_refs(result.review_comments)
            result.cross_chapter_issues = [
                normalize_paragraph_refs(i) for i in result.cross_chapter_issues
            ]
            return result
        except Exception as e:
            logger.warning("[VerdictEngine] 四维评分失败，降级: %s", e)
            return FourDimReviewResult(
                quality_score=VERDICT_REFINE_THRESHOLD,
                review_comments=f"四维评分失败: {e}",
                cross_chapter_consistency=VERDICT_REFINE_THRESHOLD,
                failed=True,
            )

    def _parse_four_dim_response(self, raw: str) -> FourDimReviewResult:
        """解析四维评分 LLM 响应。"""
        parsed, err = validate_json_output(
            raw,
            required_keys=["quality_score", "review_comments"],
            fail_closed=False,
        )

        if parsed:
            quality_score = max(
                _QUALITY_SCORE_MIN,
                min(_QUALITY_SCORE_MAX, float(parsed.get("quality_score", 70.0))),
            )
            review_comments = str(parsed.get("review_comments", ""))
            cross_consistency = float(
                parsed.get("cross_chapter_consistency", VERDICT_REFINE_THRESHOLD)
            )
            cross_issues = parsed.get("cross_chapter_issues", [])
            if not isinstance(cross_issues, list):
                cross_issues = []
        else:
            # 降级：尝试从文本提取（使用保守默认值，避免评分虚高）
            quality_score = VERDICT_REFINE_THRESHOLD
            review_comments = raw[:500] if raw else "评分解析失败"
            cross_consistency = VERDICT_REFINE_THRESHOLD
            cross_issues = []

        # 四维分项（如果 LLM 提供了）
        four_dim_scores = FourDimScores()
        if parsed and "four_dim_scores" in parsed:
            try:
                fd = parsed["four_dim_scores"]
                four_dim_scores = FourDimScores(
                    plot_logic=float(fd.get("plot_logic", 0)),
                    writing_style=float(fd.get("writing_style", 0)),
                    character_consistency=float(fd.get("character_consistency", 0)),
                    worldbuilding=float(fd.get("worldbuilding", 0)),
                )
            except (TypeError, ValueError):
                pass

        return FourDimReviewResult(
            quality_score=quality_score,
            four_dim_scores=four_dim_scores,
            review_comments=review_comments,
            cross_chapter_consistency=cross_consistency,
            cross_chapter_issues=cross_issues,
            evidence_chain=self._parse_evidence_chain(parsed),
            failed=not parsed,
        )

    def _parse_evidence_chain(self, parsed: dict | None) -> list[EvidenceItem]:
        """从解析后的 JSON 中提取证据链。"""
        if not parsed:
            return []
        raw_chain = parsed.get("evidence_chain", [])
        if not isinstance(raw_chain, list):
            return []
        result = []
        for item in raw_chain:
            if isinstance(item, dict) and item.get("issue"):
                result.append(
                    EvidenceItem(
                        issue=str(item.get("issue", "")),
                        span_a=str(item.get("span_a", "")),
                        span_b=str(item.get("span_b", "")),
                        reasoning=str(item.get("reasoning", "")),
                        error_type=str(item.get("error_type", "")),
                    )
                )
        return result

    # ── 评分融合子步骤 ──────────────────────────────────────────────────────

    def _resolve_llm_scores(
        self,
        llm_old_reader: LLMOldReaderResult | None,
        llm_ai_style: LLMAIStyleResult | None,
        programmatic: ProgrammaticReport,
    ) -> tuple[float, bool, float, bool]:
        """解析 LLM 语义评分，失败时降级为程序化评分。

        Returns:
            (llm_old_reader_score, llm_or_valid, llm_human_like_score, llm_ais_valid)
        """
        # LLM 老书虫语义评分（失败时降级为程序化评分）
        if llm_old_reader is not None and not llm_old_reader.failed:
            llm_old_reader_score = llm_old_reader.semantic_score
        else:
            llm_old_reader_score = programmatic.lao_shu_chong_score
        llm_or_valid = llm_old_reader is not None and not llm_old_reader.failed

        # LLM AI味语义评分（失败时降级为程序化评分的反向）
        llm_human_like_score = (
            llm_ai_style.human_like_score
            if llm_ai_style is not None and not llm_ai_style.failed
            else (1.0 - programmatic.ai_style_score) * 100.0
        )
        llm_ais_valid = llm_ai_style is not None and not llm_ai_style.failed

        return llm_old_reader_score, llm_or_valid, llm_human_like_score, llm_ais_valid

    def _analyze_toxic_state(
        self,
        llm_old_reader: LLMOldReaderResult | None,
        programmatic: ProgrammaticReport,
        programmatic_normalized: float,
    ) -> _ToxicState:
        """分析毒点状态并根据 LLM/程序化一致性动态调整权重。

        v7.8-fix: 毒点检测矛盾时动态调整权重。
        当 LLM 语义分析否认程序化毒点时，程序化评分不可靠（关键词误报），
        将其权重转移给 LLM 老书虫评分，避免程序化低分拖累总分。

        都市亲情/悬疑题材中"虐主"元素可能是剧情驱动的合理冲突，
        程序化传感器无法区分"剧情虐"和"恶意虐"。

        Returns:
            _ToxicState 数据类，包含毒点状态和调整后的权重。
        """
        llm_severe_toxic = (
            llm_old_reader is not None
            and not llm_old_reader.failed
            and llm_old_reader.has_severe_toxic
        )
        llm_analysis_healthy = (
            llm_old_reader is not None
            and not llm_old_reader.failed
        )
        # 程序化毒点被 LLM 语义分析否认时，不触发一票否决。
        prog_toxic_overridden = (
            programmatic.has_severe_toxic
            and llm_analysis_healthy
            and not llm_severe_toxic
        )

        w_prog = VERDICT_WEIGHTS["programmatic"]
        w_llm_or = VERDICT_WEIGHTS.get("llm_old_reader", 0.10)
        if prog_toxic_overridden and programmatic_normalized < 20.0:
            # 程序化评分极低但被 LLM 否认 → 转移权重给 LLM 老书虫
            transfer = w_prog * 0.5  # 转移 50% 的程序化权重
            w_prog -= transfer
            w_llm_or += transfer
            logger.info(
                "[VerdictEngine] 毒点矛盾权重调整: prog=%.3f→%.3f llm_or=%.3f→%.3f",
                VERDICT_WEIGHTS["programmatic"], w_prog,
                VERDICT_WEIGHTS.get("llm_old_reader", 0.10), w_llm_or,
            )

        return _ToxicState(
            llm_severe_toxic=llm_severe_toxic,
            llm_analysis_healthy=llm_analysis_healthy,
            prog_toxic_overridden=prog_toxic_overridden,
            w_prog=w_prog,
            w_llm_or=w_llm_or,
        )

    def _calculate_weighted_score(
        self,
        quality_score: float,
        programmatic_normalized: float,
        llm_old_reader_score: float,
        llm_human_like_score: float,
        cross_consistency: float,
        debate_penalty: float,
        decay_penalty: float,
        w_prog: float,
        w_llm_or: float,
    ) -> float:
        """计算加权融合分数。

        读取非调整权重（quality/llm_human_like/cross_chapter/debate_penalty）
        from VERDICT_WEIGHTS，使用经毒点分析调整后的 w_prog 和 w_llm_or。

        Returns:
            原始加权融合分数（未经校准/加分/归一化）。
        """
        w_quality = VERDICT_WEIGHTS["quality"]
        w_llm_ais = VERDICT_WEIGHTS.get("llm_human_like", 0.05)
        w_cross = VERDICT_WEIGHTS["cross_chapter"]
        w_debate = VERDICT_WEIGHTS["debate_penalty"]

        return (
            quality_score * w_quality
            + programmatic_normalized * w_prog
            + llm_old_reader_score * w_llm_or
            + llm_human_like_score * w_llm_ais
            + cross_consistency * w_cross
            - debate_penalty * w_debate
            - decay_penalty  # v7.2: 质量衰减惩罚（Fiction_Eval"高开低走"）
        )

    def _apply_bonuses(
        self,
        final_score: float,
        attempt_info: AttemptInfo,
        chapter_length: int,
    ) -> tuple[float, float, float]:
        """应用迭代宽松加分和长度归一化。

        v7.0: 迭代宽松加分 — 重写/润色次数越多，阈值越宽松。
        v7.3: CED 长度归一化 — 消除 Verbosity Bias。
        v7.6-fix: 仅对超过基准长度的文本做归一化，避免短文本被放大。
        参考 Lost in Stories (微软, 2026) 的 CED 归一化思路改编。

        Returns:
            (adjusted_score, iteration_bonus, length_factor)
        """
        # v7.0: 迭代宽松加分
        iteration_bonus = 0.0
        if attempt_info.loop_count > 0 or attempt_info.refine_attempts > 0:
            bonus = (
                attempt_info.loop_count * VERDICT_ITERATION_BONUS_REWRITE
                + attempt_info.refine_attempts * VERDICT_ITERATION_BONUS_REFINE
            )
            iteration_bonus = min(bonus, VERDICT_ITERATION_BONUS_MAX)
        final_score += iteration_bonus

        # v7.3: 长度归一化 — 消除 Verbosity Bias
        if (
            VERDICT_LENGTH_NORMALIZE
            and chapter_length > VERDICT_NORMALIZE_BASE
        ):
            length_factor = math.log2(chapter_length / VERDICT_NORMALIZE_BASE + 1)
            final_score = final_score / length_factor
        else:
            length_factor = 1.0

        return final_score, iteration_bonus, length_factor

    def _assemble_verdict_result(
        self,
        *,
        level: VerdictLevel,
        final_score: float,
        quality_score: float,
        programmatic: ProgrammaticReport,
        cross_chapter_consistency: float,
        debate_penalty: float,
        four_dim: FourDimReviewResult,
        llm_old_reader_score: float,
        llm_or_valid: bool,
        llm_human_like_score: float,
        llm_ais_valid: bool,
        llm_severe_toxic: bool,
        llm_old_reader: LLMOldReaderResult | None,
        llm_ai_style: LLMAIStyleResult | None,
        feedback: FeedbackBundle,
        cal_result: CalibrationResult,
        cal_reason: str,
        attempt_info: AttemptInfo,
        combined_severe_toxic: bool,
    ) -> VerdictResult:
        """组装最终 VerdictResult，汇总所有评审维度的输出。"""
        return VerdictResult(
            level=level,
            passed=level == VerdictLevel.PASS,
            final_score=final_score,
            quality_score=quality_score,
            programmatic_score=programmatic.programmatic_score,
            cross_chapter_consistency=cross_chapter_consistency,
            debate_penalty=debate_penalty,
            four_dim_scores=four_dim.four_dim_scores,
            ai_style_score=programmatic.ai_style_score,
            lao_shu_chong_score=programmatic.lao_shu_chong_score,
            # v7.1: LLM 语义分析追踪
            llm_semantic_score=llm_old_reader_score if llm_or_valid else 0.0,
            llm_human_like_score=llm_human_like_score if llm_ais_valid else 0.0,
            llm_severe_toxic_detected=llm_severe_toxic,
            llm_implicit_toxic_found=(
                llm_old_reader.implicit_toxic_found if llm_or_valid else False  # type: ignore[union-attr]
            ),
            llm_analysis_failed=(
                (llm_old_reader is not None and llm_old_reader.failed)
                or (llm_ai_style is not None and llm_ai_style.failed)
            ),
            feedback=feedback,
            is_short_text=programmatic.is_short_text,
            is_calibrated=cal_result.calibrated,
            calibration_reason=cal_reason,
            attempt_info=attempt_info,
            has_severe_toxic=combined_severe_toxic,
        )

    def _fuse(
        self,
        four_dim: FourDimReviewResult,
        programmatic: ProgrammaticReport,
        debate: DebateReport,
        cross_chapter: CrossChapterSignals,
        attempt_info: AttemptInfo,
        llm_old_reader: LLMOldReaderResult | None = None,
        llm_ai_style: LLMAIStyleResult | None = None,
        decay_penalty: float = 0.0,
        chapter_length: int = 0,
    ) -> VerdictResult:
        """融合计算 — 核心逻辑（v7.1 含 LLM 语义分析, v7.2 含质量衰减）。

        v7.3: 新增 CED 字数归一化，消除 Verbosity Bias。
        参考 Lost in Stories (微软, 2026) + Reference-Guided Verdict (2024)。
        """

        # --- 评分融合 ---
        quality_score = four_dim.quality_score

        programmatic_normalized = programmatic.lao_shu_chong_score * (
            1.0 - programmatic.ai_style_score
        )  # 0-100

        # 1. 解析 LLM 语义评分（失败时降级为程序化评分）
        llm_old_reader_score, llm_or_valid, llm_human_like_score, llm_ais_valid = (
            self._resolve_llm_scores(llm_old_reader, llm_ai_style, programmatic)
        )

        cross_consistency = four_dim.cross_chapter_consistency

        debate_penalty = debate.severity_weight

        # 2. 毒点分析 + 动态权重调整
        toxic_state = self._analyze_toxic_state(
            llm_old_reader, programmatic, programmatic_normalized
        )

        # 3. 加权融合计算
        final_score = self._calculate_weighted_score(
            quality_score,
            programmatic_normalized,
            llm_old_reader_score,
            llm_human_like_score,
            cross_consistency,
            debate_penalty,
            decay_penalty,
            toxic_state.w_prog,
            toxic_state.w_llm_or,
        )

        # 4. 迭代宽松加分 + 长度归一化
        final_score, iteration_bonus, length_factor = self._apply_bonuses(
            final_score, attempt_info, chapter_length
        )

        # --- 校准 ---
        cal_result = self._calibration.calibrate(
            final_score,
            quality_score,
            programmatic,
            debate,
            cross_chapter,
            attempt_info,
            llm_severe_toxic=toxic_state.llm_severe_toxic,
            llm_analysis_healthy=toxic_state.llm_analysis_healthy,
        )
        final_score = cal_result.score

        # --- 校准后级别决策 ---
        level = self._decide_level(
            final_score,
            programmatic,
            attempt_info,
            llm_severe_toxic=toxic_state.llm_severe_toxic,
            four_dim_failed=four_dim.failed,  # v7.3-fix: 传参防 REFINE 死循环
            prog_toxic_overridden=toxic_state.prog_toxic_overridden,
        )

        # --- 构建反馈包（含 LLM 分析结果）---
        feedback = self._feedback_builder.build(
            four_dim,
            programmatic,
            debate,
            cross_chapter,
            final_score,
            debate_penalty,
            llm_old_reader=llm_old_reader,
            llm_ai_style=llm_ai_style,
        )

        # 校准原因
        cal_reason = cal_result.reason
        if iteration_bonus > 0:
            cal_reason = (
                (f"{cal_reason}; " if cal_reason else "")
                + f"迭代宽松+{iteration_bonus:.0f}分(R{attempt_info.loop_count}/F{attempt_info.refine_attempts})"
            )

        # v7.8-fix: 仅双向用尽才标记 force-pass 到校准原因
        both_exhausted = attempt_info.rewrite_exhausted and attempt_info.refine_exhausted
        if level == VerdictLevel.PASS and both_exhausted:
            cal_reason = (
                f"{cal_reason}; " if cal_reason else "评分校准: "
            ) + f"双向次数用尽强制通过(final={final_score:.0f}分)"

        # 融合 LLM 严重毒点标记
        combined_severe_toxic = (
            toxic_state.llm_severe_toxic
            or (programmatic.has_severe_toxic and not toxic_state.prog_toxic_overridden)
        )

        # 5. 组装最终结果
        return self._assemble_verdict_result(
            level=level,
            final_score=final_score,
            quality_score=quality_score,
            programmatic=programmatic,
            cross_chapter_consistency=cross_consistency,
            debate_penalty=debate_penalty,
            four_dim=four_dim,
            llm_old_reader_score=llm_old_reader_score,
            llm_or_valid=llm_or_valid,
            llm_human_like_score=llm_human_like_score,
            llm_ais_valid=llm_ais_valid,
            llm_severe_toxic=toxic_state.llm_severe_toxic,
            llm_old_reader=llm_old_reader,
            llm_ai_style=llm_ai_style,
            feedback=feedback,
            cal_result=cal_result,
            cal_reason=cal_reason,
            attempt_info=attempt_info,
            combined_severe_toxic=combined_severe_toxic,
        )

    def _decide_level(
        self,
        final_score: float,
        programmatic: ProgrammaticReport,
        attempt_info: AttemptInfo,
        llm_severe_toxic: bool = False,
        four_dim_failed: bool = False,
        prog_toxic_overridden: bool = False,
    ) -> VerdictLevel:
        """三级决议 — 替代 12 分支路由。

        v7.1：LLM 语义级严重毒点参与重写判定。
        v7.3-fix: four_dim_failed 时跳过 REFINE（fallback 质量分=55.0 正撞 REFINE 阈值导致死循环）
        v7.7-fix: 程序化毒点被 LLM 语义分析否认时，不触发强制重写。
        v7.8-fix: 四维失败时降级为 REWRITE（非 PASS），避免垃圾章假通过。
                 次数用尽改为双向 and 才强制 PASS，单向用尽时降级走另一路径。
        决策规则：
        1. 双方一致确认毒点（LLM+程序化）+ 未用尽重写 → REWRITE
        2. 仅程序化毒点 + LLM 分析失败（无法验证）→ REWRITE（安全兜底）
        3. 仅程序化毒点 + LLM 明确否认 → 走分数路由（信任 LLM 语义理解）
        4. 四维评分全部失败（fallback）→ REWRITE（非 PASS，防垃圾章假通过）
               rewrite_exhausted 时降级为 REFINE
        5. 单向次数用尽 → 降级走另一路径（rewrite 用尽→尝试 REFINE，反之亦然）
        6. 双向次数用尽 → PASS（防死循环兜底）
        7. final_score >= PASS_THRESHOLD → PASS
        8. final_score >= REFINE_THRESHOLD → REFINE
        9. final_score < REFINE_THRESHOLD → REWRITE
        """
        # ── 兜底：双向用尽强制通过 ──
        both_exhausted = attempt_info.rewrite_exhausted and attempt_info.refine_exhausted
        if both_exhausted:
            logger.warning(
                "[VerdictEngine] 双向次数用尽强制通过 rewrite=%d/%d refine=%d/%d final=%.1f",
                attempt_info.loop_count,
                attempt_info.max_rewrite,
                attempt_info.refine_attempts,
                attempt_info.max_refine,
                final_score,
            )
            return VerdictLevel.PASS

        # ── 单向用尽：降级走另一路径（红绿灯机制） ──
        rewrite_spent = attempt_info.rewrite_exhausted and not attempt_info.refine_exhausted
        refine_spent = attempt_info.refine_exhausted and not attempt_info.rewrite_exhausted

        # 严重毒点强制重写（仅当 LLM 也确认，或 LLM 分析失败无法验证时）
        # v7.7-fix: 程序化毒点传感器是关键词匹配，缺乏上下文理解。
        # 当 LLM 语义分析功能正常且明确否认毒点时，不触发强制重写。
        if prog_toxic_overridden:
            logger.info(
                "[VerdictEngine] 程序化毒点被LLM语义分析否认，跳过强制重写 "
                "(severe_toxic=%s, llm_severe_toxic=%s)→走评分路由",
                programmatic.has_severe_toxic,
                llm_severe_toxic,
            )
        elif llm_severe_toxic and not attempt_info.rewrite_exhausted:
            logger.info("[VerdictEngine] LLM确认严重毒点 → REWRITE")
            return VerdictLevel.REWRITE
        elif programmatic.has_severe_toxic and not attempt_info.rewrite_exhausted:
            logger.info("[VerdictEngine] 程序化严重毒点(LLM分析不可用) → REWRITE")
            return VerdictLevel.REWRITE

        # ── 四维评分失败：降级为重写（非PASS），防垃圾章假通过 ──
        # v7.8-fix: 原逻辑直接 PASS 导致 ch90 42分假通过。
        # 改为 REWRITE 尝试修复内容，rewrite 用尽时再降级为 REFINE。
        if four_dim_failed:
            if rewrite_spent:
                # rewrite 已用尽 → 尝试润色
                logger.info(
                    "[VerdictEngine] 四维评分失败+rewrite用尽，降级REFINE final=%.1f",
                    final_score,
                )
                return VerdictLevel.REFINE
            logger.info(
                "[VerdictEngine] 四维评分全部失败(fallback)，降级REWRITE final=%.1f",
                final_score,
            )
            return VerdictLevel.REWRITE

        # ── 单向用尽降级路由 ──
        if rewrite_spent:
            # rewrite 用尽但 refine 还有 → 尝试 REFINE
            logger.info(
                "[VerdictEngine] rewrite用尽，降级REFINE final=%.1f",
                final_score,
            )
            return VerdictLevel.REFINE
        if refine_spent and final_score < VERDICT_PASS_THRESHOLD:
            # refine 用尽但 rewrite 还有 → 尝试 REWRITE
            logger.info(
                "[VerdictEngine] refine用尽，降级REWRITE final=%.1f",
                final_score,
            )
            return VerdictLevel.REWRITE
        if refine_spent:
            # refine 用尽但分数已达 pass 线 → 直接 PASS
            return VerdictLevel.PASS

        # ── 正常分数路由 ──
        if final_score >= VERDICT_PASS_THRESHOLD:
            return VerdictLevel.PASS
        if final_score >= VERDICT_REFINE_THRESHOLD:
            return VerdictLevel.REFINE
        return VerdictLevel.REWRITE

    def _trim_for_review(self, text: str) -> str:
        """裁剪章节文本用于 LLM 评审。

        保留开头 + 转折点 + 结尾，约 75% 情节信息。
        """
        if not text or len(text) <= _REVIEW_MAX_TOTAL:
            return text

        head = text[:_REVIEW_HEAD_TAIL_SIZE]
        tail = text[-_REVIEW_HEAD_TAIL_SIZE:]
        mid_len = len(text) - len(head) - len(tail)
        if mid_len <= 0:
            return head + tail

        step = max(mid_len // 4, 1)
        mid_samples: list[str] = []
        for i in range(1, _REVIEW_SAMPLE_COUNT + 1):
            start = len(head) + i * step
            end = start + _REVIEW_SAMPLE_SIZE
            if start < len(text):
                mid_samples.append(text[start : min(end, len(text) - len(tail))])

        return head + "\n[...] " + " ".join(mid_samples) + "\n[...] \n" + tail
