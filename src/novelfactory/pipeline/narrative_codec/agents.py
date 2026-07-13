"""Narrative Codec Engine Agent 工厂函数。

每个 Agent 采用 create_react_agent + RunnableLambda 模式构建，
通过 _retry_agent_invoke 进行生产级调用（超时 + 指数退避重试 + 降级）。

参考:
- Beyond LLMs (ACL 2025) §3.1 — Vertices Extraction (句子精炼)
- Beyond LLMs (ACL 2025) §3.2 — STAC Categorization (叙事四分类)
- Beyond LLMs (ACL 2025) §3.4 — Graph Construction Iter3 (反事实剪枝)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent

from novelfactory.agents.infra import (
    extract_ai_message_text,
    validate_json_output,
)
from novelfactory.agents.infra.helpers import make_retry_agent_invoke
from novelfactory.config.llm import get_reviewer_llm
from novelfactory.pipeline.narrative_codec.prompts import (
    COUNTERFACTUAL_SYSTEM,
    COUNTERFACTUAL_USER,
    SENTENCE_REFINER_SYSTEM,
    SENTENCE_REFINER_USER,
    STAC_CLASSIFIER_SYSTEM,
    STAC_CLASSIFIER_USER,
)

logger = logging.getLogger(__name__)

# ── 模块级 _retry_agent_invoke 工厂 ──────────────────────────────────────
_retry_agent_invoke = make_retry_agent_invoke("narrative_codec")


# ═══════════════════════════════════════════════════════════════════════
# Agent 1: 句子精炼 Agent
# ═══════════════════════════════════════════════════════════════════════


def create_sentence_refiner_agent(llm=None) -> RunnableLambda:
    """构建句子精炼 Agent。

    参考: Beyond LLMs §3.1 — Vertices Extraction
    将原始句子精炼为适合因果分析的标准化形式：
      1. Summarization（摘要精简）
      2. Pronoun Substitution（代词替换）
      3. Clause Simplification（从句简化）
      4. Active Voice + Continuous Flow（主动语态 + 连续流）

    每个句子有独立 try/except 保护，单句失败不阻塞整体。

    Returns:
        RunnableLambda: _node(state) -> dict
    """
    if llm is None:
        llm = get_reviewer_llm()

    agent = create_react_agent(
        llm,
        tools=[],
        prompt=SENTENCE_REFINER_SYSTEM,
    )

    def _node(state: dict) -> dict[str, Any]:
        text = state.get("raw_text", "")
        context = ""  # TODO: extract from state if needed

        input_text = SENTENCE_REFINER_USER.format(context=context, text=text)

        try:
            result = _retry_agent_invoke(
                agent, {"messages": [("user", input_text)]}, "sentence_refiner"
            )
            response = extract_ai_message_text(result)
        except Exception as exc:
            logger.warning("[sentence_refiner] invoke failed: %s", exc)
            response = ""

        # 解析 JSON，失败时降级使用原文
        try:
            parsed = json.loads(response) if response.strip() else None
            if isinstance(parsed, dict):
                refined = [parsed.get("refined", text)]
            elif isinstance(parsed, list):
                refined = [item.get("refined", text) for item in parsed]
            else:
                refined = [text]
        except (json.JSONDecodeError, TypeError, AttributeError):
            refined = text.split("\n") if isinstance(text, str) else [str(text)]

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {**existing_cr, "refined_sentences": refined},
            "refined_sentences": refined,
        }

    return RunnableLambda(_node)


# ═══════════════════════════════════════════════════════════════════════
# Agent 2: STAC 分类 Agent
# ═══════════════════════════════════════════════════════════════════════


def create_stac_classifier_agent(llm=None) -> RunnableLambda:
    """构建 STAC 分类 Agent。

    参考: Beyond LLMs §3.2 — STAC Categorization
    对句子进行 {Situation, Task, Action, Consequence} 四分类。
    每个句子独立调用 LLM，单句失败以降级值替代，不阻塞整体。

    Returns:
        RunnableLambda: _node(state) -> dict
    """
    if llm is None:
        llm = get_reviewer_llm()

    agent = create_react_agent(
        llm,
        tools=[],
        prompt=STAC_CLASSIFIER_SYSTEM,
    )

    def _node(state: dict) -> dict[str, Any]:
        sentences = state.get(
            "refined_sentences",
            state.get("_input_text", []),
        )
        if isinstance(sentences, str):
            sentences = [s for s in sentences.split("\n") if s.strip()]

        results = []
        for s in sentences:
            try:
                input_text = STAC_CLASSIFIER_USER.format(text=s)
                result = _retry_agent_invoke(
                    agent, {"messages": [("user", input_text)]}, "stac_classifier"
                )
                response = extract_ai_message_text(result)

                parsed, err = validate_json_output(response, ["label"])
                if parsed is None:
                    logger.warning("[stac_classifier] parse failed for: %s", err)
                    item = {
                        "label": "situation",
                        "confidence": 0.3,
                        "reasoning": "parse fallback",
                    }
                else:
                    item = {
                        "label": parsed.get("label", "situation"),
                        "confidence": parsed.get("confidence", 0.3),
                        "reasoning": parsed.get("reasoning", ""),
                    }
            except Exception as exc:
                logger.warning("[stac_classifier] invoke failed for sentence: %s", exc)
                item = {
                    "label": "situation",
                    "confidence": 0.3,
                    "reasoning": "invoke fallback",
                }

            results.append(
                {
                    "text": s,
                    **item,
                }
            )

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {**existing_cr, "stac_labels": results},
            "stac_labels_agent": results,
        }

    return RunnableLambda(_node)


# ═══════════════════════════════════════════════════════════════════════
# Agent 3: 反事实剪枝 Agent
# ═══════════════════════════════════════════════════════════════════════


def create_counterfactual_agent(llm=None) -> RunnableLambda:
    """构建反事实剪枝 Agent。

    参考: Beyond LLMs §3.4 — Graph Construction Iter3
    判断"如果 A 不发生，B 是否仍会发生？"，
    用于因果图构建阶段的边剪枝。
    每对因果假设独立调用 LLM，单对失败不阻塞整体。

    Returns:
        RunnableLambda: _node(state) -> dict
    """
    if llm is None:
        llm = get_reviewer_llm()

    agent = create_react_agent(
        llm,
        tools=[],
        prompt=COUNTERFACTUAL_SYSTEM,
    )

    def _node(state: dict) -> dict[str, Any]:
        pairs = state.get("counterfactual_pairs", [])
        if not pairs:
            existing_cr = state.get("crew_result", {})
            return {"crew_result": existing_cr}

        judgments = []
        for pair in pairs:
            try:
                cause_text = pair.get("cause", "")
                effect_text = pair.get("effect", "")
                input_text = COUNTERFACTUAL_USER.format(
                    cause_text=cause_text,
                    effect_text=effect_text,
                )
                result = _retry_agent_invoke(
                    agent, {"messages": [("user", input_text)]}, "counterfactual"
                )
                response = extract_ai_message_text(result)

                parsed, err = validate_json_output(response, ["answer"])
                if parsed is None:
                    logger.warning("[counterfactual] parse failed: %s", err)
                    judged = {
                        "answer": "MAYBE",
                        "confidence": 0.3,
                        "reasoning": "parse fallback",
                    }
                else:
                    judged = {
                        "answer": parsed.get("answer", "MAYBE"),
                        "confidence": parsed.get("confidence", 0.3),
                        "reasoning": parsed.get("reasoning", ""),
                    }
            except Exception as exc:
                logger.warning("[counterfactual] invoke failed for pair: %s", exc)
                judged = {
                    "answer": "MAYBE",
                    "confidence": 0.3,
                    "reasoning": "invoke fallback",
                }

            judgments.append(
                {
                    "cause": pair.get("cause", ""),
                    "effect": pair.get("effect", ""),
                    **judged,
                }
            )

        existing_cr = state.get("crew_result", {})
        return {
            "crew_result": {**existing_cr, "counterfactual_judgments": judgments},
            "counterfactual_judgments": judgments,
        }

    return RunnableLambda(_node)
