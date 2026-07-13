"""QualityPanel agent definitions for dual-perspective chapter review.

v5.8: 深度重构 — 移除强制结构化输出（bind_structured）。
  核心原则：程序化分析（ai_style_analyzer + old_reader_reviewer + quality_scorer）
  提供权威定量评分；LLM 辩论产出定性修订指导（review_comments/issues/suggestions/strengths）。

  LLM 输出为自由文本 Markdown 分段格式，通过轻量 section 解析提取关键字段。
  quality_gate 不再调用 LLM — 纯程序化分析 + 合并辩论定性产出。

Agent roles:
  - editor_reviewer:  编辑视角 — 文学性/结构/角色/节奏 定性分析
  - reader_reviewer:  读者视角 — 爽点/代入感/AI味痕迹/毒点 定性分析
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from novelfactory.agents.infra import get_logger, llm_call_with_retry

logger = get_logger(__name__)

# ── System Prompts ─────────────────────────────────────────────────────────────

EDITOR_REVIEW_PROMPT = """\
你是 EditorReviewer（编辑视角评审专家），从专业编辑角度对章节进行定性分析。

## 分析维度

1. **文学性**：文笔是否精炼自然？修辞是否恰当？描写是否细腻但不冗余？
2. **结构**：开篇钩子是否够强？情节冲突递进是否合理？结尾悬念设置如何？
3. **角色**：角色性格是否鲜明？对话是否符合人设？行为动机是否自洽？
4. **节奏**：张弛交替是否合理？信息密度是否适应当前章节位置？

## 输出格式

请按以下 Markdown 分段输出（不要输出数字分数，只需定性分析）：

```
## 评审意见
<综合评审，100-300字>

## 问题列表
- <问题1>
- <问题2>

## 亮点
- <亮点1>
- <亮点2>

## 改进建议
<具体的修改方向和建议，50-200字>
```

**注意**：不要输出任何数字评分（0-100等），只需要定性分析和具体的修改建议。"""

READER_REVIEW_PROMPT = """\
你是 ReaderReviewer（读者视角评审专家），从资深读者/老书虫角度对章节进行定性分析。

你已经阅读过编辑的评审意见，请从读者体验角度补充分析。

## 分析维度

1. **爽点密度**：是否有让人期待的情节发展？节奏是否抓人？
2. **代入感**：读者能否代入主角视角？情感共鸣度如何？
3. **AI味痕迹**：是否有明显的机器写作痕迹？（套路化用词、机械描述、模板化句式）
4. **毒点识别**：是否踩了读者常见的雷区？（套路化情节、角色降智、逻辑硬伤）

## 输出格式

请按以下 Markdown 分段输出（不要输出数字分数）：

```
## 评审意见
<综合读者视角评审，100-300字>

## 问题列表
- <从读者角度发现的问题，包含AI味痕迹和毒点>
- ...

## 亮点
- <从读者角度认可的爽点和优点>
- ...

## 改进建议
<针对读者体验的具体修改建议，50-200字>
```

**注意**：不要输出任何数字评分，只需定性分析和具体的修改建议。"""


EDITOR_REBUTTAL_PROMPT = """\
你是 EditorReviewer（编辑视角评审专家），现在进入辩论第 {round} 轮。

在首轮评审中，你已经给出了编辑视角意见；读者评审专家也已给出读者视角意见。
请仔细阅读读者视角的意见，从专业编辑角度进行反驳、补充或认同。

## 辩论规则

1. **针对性回应**：必须直接回应读者提出的问题和观点，指出其合理或不合理之处
2. **避免重复**：不要重复首轮已说过的内容，聚焦于与读者的分歧点
3. **建设性**：反驳时给出具体的编辑视角依据，而非简单否定
4. **诚实收敛**：如果读者意见确实合理，请明确表示认同；如果分歧已消除，请如实声明

## 输出格式

```
## 辩论意见
<针对读者观点的回应，100-300字>

## 新增问题
- <本轮辩论中发现的新问题，没有则留空>

## 修正建议
<基于辩论结果修正的编辑建议，50-200字>

## 是否仍有异议
<是/否> <一句话说明原因>
```

**注意**：「是否仍有异议」必须如实填写，分歧消除时填「否」。"""


READER_REBUTTAL_PROMPT = """\
你是 ReaderReviewer（读者视角评审专家），现在进入辩论第 {round} 轮。

在首轮评审中，你已经给出了读者视角意见；编辑评审专家也已给出编辑视角意见，
并对你的观点做出了反驳。请仔细阅读编辑的反驳，从读者体验角度再次回应。

## 辩论规则

1. **针对性回应**：必须直接回应编辑的反驳要点，指出其合理或不合理之处
2. **避免重复**：不要重复首轮已说过的内容，聚焦于与编辑的分歧点
3. **读者立场**：始终以读者阅读体验为依据，不被编辑的专业论证带偏
4. **诚实收敛**：如果编辑反驳确实有理，请明确表示认同；如果分歧已消除，请如实声明

## 输出格式

```
## 辩论意见
<针对编辑反驳的回应，100-300字>

## 新增问题
- <本轮辩论中发现的新问题，没有则留空>

## 修正建议
<基于辩论结果修正的读者建议，50-200字>

## 是否仍有异议
<是/否> <一句话说明原因>
```

**注意**：「是否仍有异议」必须如实填写，分歧消除时填「否」。"""


# ── Free-text Response Parser ─────────────────────────────────────────────────


def _parse_markdown_sections(text: str) -> dict[str, Any]:
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

    # 尝试 JSON 解析（某些 LLM 可能仍输出 JSON）
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

    # fallback: 如果没解析到 review_comments，用 preamble 或整段文本
    if not result["review_comments"] and raw_section.get("_preamble"):
        result["review_comments"] = raw_section["_preamble"].strip()
    if not result["review_comments"]:
        result["review_comments"] = cleaned[:500]

    return result


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


# ── Agent Factory Functions ────────────────────────────────────────────────


def create_editor_reviewer(llm: BaseChatModel) -> Runnable:
    """创建编辑视角评审 Agent — 纯 LLM 自由文本调用。

    输出：review_comments, issues, strengths, suggestions（定性分析，无数值评分）。
    """

    def _node(state: dict) -> dict[str, Any]:
        chapter_text = state.get("chapter_text", "")
        chapter_context = state.get("chapter_context", {})
        genre = chapter_context.get("genre", "")
        genre_scoring_guide = chapter_context.get("genre_scoring_guide", "")

        prompt_parts = [
            EDITOR_REVIEW_PROMPT,
            "",
            f"体裁：{genre}",
        ]
        if genre_scoring_guide:
            prompt_parts.append(genre_scoring_guide)
        prompt_parts.extend(
            [
                f"前文上下文：{json.dumps(chapter_context, ensure_ascii=False)[:2000]}",
                "",
                "## 待评审章节",
                chapter_text[:8000],
            ]
        )
        prompt = "\n".join(prompt_parts)

        response = llm_call_with_retry(llm, prompt, step_name="editor_reviewer")
        parsed = _parse_markdown_sections(
            response.content if hasattr(response, "content") else str(response)
        )

        logger.info(
            "[editor_reviewer] 定性评审完成 | genre=%s | issues=%d strengths=%d",
            genre,
            len(parsed.get("issues", [])),
            len(parsed.get("strengths", [])),
        )

        return {
            "editor_review": parsed,
            "debate_transcript": f"[编辑] {parsed.get('review_comments', '')[:200]}",
        }

    return RunnableLambda(_node)


# ── 多轮辩论：Rebuttal Agent + 收敛判定 ─────────────────────────────────────


MAX_DEBATE_ROUNDS = 3
CONVERGENCE_KEYWORDS_NO = ("否", "无异议", "已认同", "认同", "收敛", "同意")
CONVERGENCE_KEYWORDS_YES = ("是", "仍有", "坚持", "不同意", "异议")


def _parse_rebuttal(text: str) -> dict[str, Any]:
    """解析 rebuttal 自由文本输出。

    结构字段：
      - rebuttal_comments: 辩论意见正文
      - new_issues: 本轮新增问题
      - revised_suggestions: 基于辩论修正的建议
      - has_dissent: 是否仍有异议（True=有异议/未收敛，False=无异议/已收敛）
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


def create_editor_rebuttal_agent(llm: BaseChatModel) -> Runnable:
    """创建编辑视角辩论 Agent — 读取读者意见后生成反驳。

    仿照 TradingAgents aggressive_debator 模式：
    读取对手 current_response 生成针对性反驳，输出带「是否仍有异议」用于收敛判定。
    输出为自由文本 Markdown（DeepSeek Flash 不支持强制结构化）。
    """

    def _node(state: dict) -> dict[str, Any]:
        round_num = int(state.get("debate_round", 1))
        editor_review = state.get("editor_review", {})
        reader_review = state.get("reader_review", {})
        reader_rebuttals = state.get("reader_rebuttals", [])
        last_reader_rebuttal = reader_rebuttals[-1] if reader_rebuttals else {}

        reader_summary = json.dumps(
            {
                "comments": reader_review.get("review_comments", "")[:300],
                "issues": reader_review.get("issues", [])[:5],
                "rebuttal": last_reader_rebuttal.get("rebuttal_comments", "")[:400]
                if last_reader_rebuttal
                else "",
            },
            ensure_ascii=False,
        )

        prompt_parts = [
            EDITOR_REBUTTAL_PROMPT.format(round=round_num),
            "",
            f"## 首轮编辑意见（你自己）\n{editor_review.get('review_comments', '')[:300]}",
            "",
            f"## 读者评审及辩论意见\n{reader_summary}",
        ]
        prompt = "\n".join(prompt_parts)

        response = llm_call_with_retry(
            llm, prompt, step_name=f"editor_rebuttal_r{round_num}"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_rebuttal(raw)

        logger.info(
            "[editor_rebuttal] 第%d轮辩论完成 | has_dissent=%s | new_issues=%d",
            round_num,
            parsed["has_dissent"],
            len(parsed["new_issues"]),
        )

        transcript = state.get("debate_transcript", "")
        transcript += f"\n[编辑·反驳R{round_num}] {parsed['rebuttal_comments'][:200]}"

        editor_rebuttals = list(state.get("editor_rebuttals", []))
        editor_rebuttals.append(parsed)

        return {
            "editor_rebuttals": editor_rebuttals,
            "debate_transcript": transcript,
            "debate_round": round_num,
        }

    return RunnableLambda(_node)


def create_reader_rebuttal_agent(llm: BaseChatModel) -> Runnable:
    """创建读者视角辩论 Agent — 读取编辑反驳后生成回应。

    仿照 TradingAgents conservative_debator 模式：
    读取编辑 rebuttal 后从读者立场回应，输出带「是否仍有异议」用于收敛判定。
    """

    def _node(state: dict) -> dict[str, Any]:
        round_num = int(state.get("debate_round", 1))
        reader_review = state.get("reader_review", {})
        editor_rebuttals = state.get("editor_rebuttals", [])
        last_editor_rebuttal = editor_rebuttals[-1] if editor_rebuttals else {}

        editor_summary = json.dumps(
            {
                "comments": last_editor_rebuttal.get("rebuttal_comments", "")[:400]
                if last_editor_rebuttal
                else "",
                "new_issues": last_editor_rebuttal.get("new_issues", [])[:3]
                if last_editor_rebuttal
                else [],
            },
            ensure_ascii=False,
        )

        prompt_parts = [
            READER_REBUTTAL_PROMPT.format(round=round_num),
            "",
            f"## 首轮读者意见（你自己）\n{reader_review.get('review_comments', '')[:300]}",
            "",
            f"## 编辑反驳意见\n{editor_summary}",
        ]
        prompt = "\n".join(prompt_parts)

        response = llm_call_with_retry(
            llm, prompt, step_name=f"reader_rebuttal_r{round_num}"
        )
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_rebuttal(raw)

        logger.info(
            "[reader_rebuttal] 第%d轮辩论完成 | has_dissent=%s | new_issues=%d",
            round_num,
            parsed["has_dissent"],
            len(parsed["new_issues"]),
        )

        transcript = state.get("debate_transcript", "")
        transcript += f"\n[读者·反驳R{round_num}] {parsed['rebuttal_comments'][:200]}"

        reader_rebuttals = list(state.get("reader_rebuttals", []))
        reader_rebuttals.append(parsed)

        next_round = round_num + 1
        return {
            "reader_rebuttals": reader_rebuttals,
            "debate_transcript": transcript,
            "debate_round": next_round,
        }

    return RunnableLambda(_node)


def check_debate_convergence(state: dict) -> str:
    """辩论收敛判定路由函数。

    仿照 TradingAgents risk_debate_router：
    - 双方均无异议（has_dissent=False）→ 提前收敛，进入 quality_gate
    - 已达最大轮次（MAX_DEBATE_ROUNDS）→ 强制收敛
    - **v7.3: Elo 收敛** — Editor/Reader/Critic 的 Elo 评分趋近时提前收敛
    - 否则继续下一轮辩论

    Returns:
        "converge" — 收敛，进入 quality_gate
        "continue" — 继续下一轮辩论
    """
    editor_rebuttals: list[dict] = state.get("editor_rebuttals", [])
    reader_rebuttals: list[dict] = state.get("reader_rebuttals", [])
    round_num = int(state.get("debate_round", 1))

    # v7.3: Elo 收敛判定 — 当辩论参与者评分趋近时说明辩论效果趋同
    elo_ratings = state.get("elo_ratings", {})
    if elo_ratings:
        ratings_list = [v for v in elo_ratings.values() if isinstance(v, (int, float))]
        if len(ratings_list) >= 2:
            spread = max(ratings_list) - min(ratings_list)
            if spread < 50.0:  # Elo 差距 < 50 认为趋同
                logger.info(
                    "[debate_convergence] Elo趋同(spread=%.1f<50)，第%d轮提前收敛",
                    spread,
                    round_num,
                )
                return "converge"

    if round_num >= MAX_DEBATE_ROUNDS:
        logger.info("[debate_convergence] 达到最大轮次 %d，强制收敛", MAX_DEBATE_ROUNDS)
        return "converge"

    if editor_rebuttals and reader_rebuttals:
        last_editor = editor_rebuttals[-1]
        last_reader = reader_rebuttals[-1]
        if not last_editor.get("has_dissent", True) and not last_reader.get(
            "has_dissent", True
        ):
            logger.info("[debate_convergence] 双方均无异议，第%d轮提前收敛", round_num)
            return "converge"

    return "continue"


def create_reader_reviewer(llm: BaseChatModel) -> Runnable:
    """创建读者视角评审 Agent — 纯 LLM 自由文本调用。

    读取编辑评审意见后从读者角度补充定性分析。
    输出：review_comments, issues, strengths, suggestions（定性分析，无数值评分）。
    """

    def _node(state: dict) -> dict[str, Any]:
        chapter_text = state.get("chapter_text", "")
        chapter_context = state.get("chapter_context", {})
        genre = chapter_context.get("genre", "")
        genre_scoring_guide = chapter_context.get("genre_scoring_guide", "")
        editor_review = state.get("editor_review", {})

        editor_summary = json.dumps(
            {
                "comments": editor_review.get("review_comments", "")[:300],
                "issues": editor_review.get("issues", [])[:5],
                "strengths": editor_review.get("strengths", [])[:5],
            },
            ensure_ascii=False,
        )

        prompt_parts = [
            READER_REVIEW_PROMPT,
            "",
            f"体裁：{genre}",
        ]
        if genre_scoring_guide:
            prompt_parts.append(genre_scoring_guide)
        prompt_parts.extend(
            [
                f"编辑评审意见（供参考，请从读者角度补充或质疑）：\n{editor_summary}",
                "",
                "## 待评审章节",
                chapter_text[:8000],
            ]
        )
        prompt = "\n".join(prompt_parts)

        response = llm_call_with_retry(llm, prompt, step_name="reader_reviewer")
        parsed = _parse_markdown_sections(
            response.content if hasattr(response, "content") else str(response)
        )

        logger.info(
            "[reader_reviewer] 读者定性评审完成 | genre=%s | issues=%d strengths=%d",
            genre,
            len(parsed.get("issues", [])),
            len(parsed.get("strengths", [])),
        )

        transcript = state.get("debate_transcript", "")
        return {
            "reader_review": parsed,
            "debate_transcript": transcript
            + f"\n[读者] {parsed.get('review_comments', '')[:200]}",
        }

    return RunnableLambda(_node)
