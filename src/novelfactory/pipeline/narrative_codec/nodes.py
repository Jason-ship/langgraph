"""Narrative Codec Engine 的 LangGraph Node 函数。

每个 node 签名 (state: CodecCrewLocalState) -> dict，遵循项目标准模式：
- from __future__ import annotations
- 独立 try/except 保护
- LLM 调用失败时保留前一步结果 (codec_stage 不变)
- 纯规则 tool 调用失败时标记为 error

参考:
- Beyond LLMs (ACL 2025) — STAC + Expert Index + Causal Graph
- Shadow-Loom (arXiv 2026) — WorldStateV1 + 双时间轴
"""

from __future__ import annotations

import json
import logging
from typing import Any

import networkx as nx

from novelfactory.pipeline.narrative_codec.schemas import (
    VALID_STAC_BONDS,
    CausalGraph,
    EmotionArc,
    EventNode,
    ExpertIndex,
    Scene,
    STACLabel,
    STACLabeledSentence,
)
from novelfactory.pipeline.narrative_codec.state import CodecCrewLocalState
from novelfactory.pipeline.narrative_codec.tools import (
    compute_expert_index,
    extract_emotion_arc,
    find_scene_boundaries,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Node 1: 场景分割
# ═══════════════════════════════════════════════════════════════════════


def scene_splitter_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """场景分割 node — 调用 find_scene_boundaries tool。

    零 LLM，纯规则。基于实体密度突变 + 时间/地点关键词的规则分割。
    解析失败时标记为 error 而非阻塞整体流程。

    Args:
        state: CodecCrewLocalState，需包含 raw_text

    Returns:
        dict: 更新 scenes / codec_stage 或 codec_error
    """
    raw_text = state.get("raw_text", "")
    if not raw_text:
        logger.warning("[scene_splitter] empty input")
        return {"codec_stage": "error", "codec_error": "empty input"}

    try:
        result_str = find_scene_boundaries.invoke({"text": raw_text})
        scenes_data = json.loads(result_str)
        if isinstance(scenes_data, dict) and "error" in scenes_data:
            logger.error("[scene_splitter] tool error: %s", scenes_data["error"])
            return {"codec_stage": "error", "codec_error": scenes_data["error"]}

        scenes = []
        for i, s in enumerate(scenes_data):
            scenes.append(
                Scene(
                    scene_id=i + 1,
                    start_char=s.get("start_char", 0),
                    end_char=s.get("end_char", 0),
                    characters=s.get("characters", []),
                    location=s.get("location", ""),
                    time_marker=s.get("time_marker", ""),
                    text=s.get("text", ""),
                )
            )
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error("[scene_splitter] parse error: %s", e)
        return {"codec_stage": "error", "codec_error": f"parse error: {e}"}

    return {
        "scenes": scenes,
        "codec_stage": "refine",
        # 保持 total_cost 透传
        "total_cost": state.get("total_cost", 0.0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 2: 句子精炼
# ═══════════════════════════════════════════════════════════════════════


def sentence_refiner_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """句子精炼 node — 调用 SentenceRefinerAgent。

    LLM 驱动，每个句子独立保护。调用失败时降级使用原文。
    参考: Beyond LLMs §3.1 — Vertices Extraction

    Args:
        state: CodecCrewLocalState，需包含 raw_text

    Returns:
        dict: 更新 refined_sentences / codec_stage
    """
    try:
        agent = create_sentence_refiner_agent()
        result = agent.invoke(state)
        refined = result.get("refined_sentences", [])
    except Exception as e:
        logger.warning("[sentence_refiner] agent invoke failed: %s", e)
        # 降级：使用 raw_text 按句分割
        raw = state.get("raw_text", "")
        refined = [s.strip() for s in raw.split("\n") if s.strip()] if raw else []

    return {
        "refined_sentences": refined if isinstance(refined, list) else [],
        "codec_stage": "stac",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 3: STAC 分类
# ═══════════════════════════════════════════════════════════════════════


def stac_classifier_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """STAC 分类 node — 规则优先，Agent 降级。

    先规则分类（基于动词关键词 + 句法特征），置信度 >= 0.7 直接采用。
    低于 0.7 的调用 STAC Agent 精修。
    参考: Beyond LLMs §3.2 — STAC Categorization

    容错: 规则分类/LLM 任一失败不阻塞整体，降级为 situation(0.3)。

    Args:
        state: CodecCrewLocalState，需包含 refined_sentences

    Returns:
        dict: 更新 stac_labels / codec_stage
    """
    sentences = state.get("refined_sentences", [])
    if not sentences:
        logger.info("[stac_classifier] no sentences to classify")
        return {"codec_stage": "stac", "stac_labels": []}

    # ── 规则分类 (基于简单动词关键词匹配) ──
    try:
        from novelfactory.pipeline.narrative_codec.tools import apply_rule_stac

        rule_result_str = apply_rule_stac.invoke({"texts": sentences})
        rule_data = json.loads(rule_result_str)
        rule_results = []
        for item in rule_data:
            rule_results.append(
                STACLabeledSentence(
                    text=item.get("text", ""),
                    stac_label=STACLabel(item.get("label", "situation")),
                    confidence=item.get("confidence", 0.3),
                )
            )
    except Exception as e:
        logger.warning("[stac_classifier] rule classifier failed: %s", e)
        rule_results = [
            STACLabeledSentence(text=s, stac_label=STACLabel.SITUATION, confidence=0.3)
            for s in sentences
        ]

    # ── 低置信度 -> LLM 精修 ──
    llm_needed = [s for s in rule_results if s.confidence < 0.7]
    llm_fixes: dict[int, STACLabeledSentence] = {}

    if llm_needed:
        try:
            agent = create_stac_classifier_agent()
            agent_result = agent.invoke(
                {
                    **state,
                    "refined_sentences": [s.text for s in llm_needed],
                }
            )
            stac_data = agent_result.get("stac_labels_agent", [])
            for item in stac_data:
                item_text = item.get("text", "")
                for idx, s in enumerate(sentences):
                    if s == item_text:
                        llm_fixes[idx] = STACLabeledSentence(
                            text=item_text,
                            stac_label=STACLabel(item.get("label", "situation")),
                            confidence=item.get("confidence", 0.5),
                        )
                        break
        except Exception as e:
            logger.warning("[stac_classifier] LLM fallback failed: %s", e)

    # ── 合并结果 ──
    # rule_results 的顺序与 sentences 一致
    final_labels: list[STACLabeledSentence] = []
    for i, s in enumerate(rule_results):
        if i in llm_fixes:
            final_labels.append(llm_fixes[i])
        else:
            final_labels.append(s)

    return {
        "stac_labels": final_labels,
        "codec_stage": "expert",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 4: Expert Index
# ═══════════════════════════════════════════════════════════════════════


def expert_index_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """Expert Index node — 调用 compute_expert_index tool。

    零 LLM，纯规则。7 维语言学特征 + 15 维 one-hot 编码。
    参考: Beyond LLMs §3.3 — Expert Index
    每个 sentence 的 7 维特征独立提取，单句失败不阻塞。

    Args:
        state: CodecCrewLocalState，需包含 refined_sentences

    Returns:
        dict: 更新 expert_indices / codec_stage
    """
    sentences = state.get("refined_sentences", [])
    if not sentences:
        logger.info("[expert_index] no sentences to index")
        return {"codec_stage": "expert", "expert_indices": []}

    try:
        result_str = compute_expert_index.invoke({"texts": sentences})
        indices_data = json.loads(result_str)
        if isinstance(indices_data, dict) and "error" in indices_data:
            logger.error("[expert_index] tool error: %s", indices_data["error"])
            return {"codec_stage": "error", "codec_error": indices_data["error"]}

        indices = []
        for item in indices_data:
            idx = ExpertIndex(
                genericity=item.get("genericity", "specific"),
                eventivity=item.get("eventivity", "dynamic"),
                boundedness=item.get("boundedness", "episodic"),
                initiativity=item.get("initiativity", "initiate"),
                time_start=item.get("time_start", "past"),
                time_end=item.get("time_end", "current"),
                impact=item.get("impact", "impactful"),
            )
            indices.append(idx)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error("[expert_index] parse error: %s", e)
        return {"codec_stage": "error", "codec_error": f"parse error: {e}"}

    return {
        "expert_indices": indices,
        "codec_stage": "graph",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 5: 因果图构建
# ═══════════════════════════════════════════════════════════════════════


def causal_graph_builder_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """因果图构建 node — 3 轮迭代 DAG 构建。

    Iter1: STAC Bond 学习 — 相邻句子间的合法 Bond 映射为候选因果边
    Iter2: 长程候选边生成 + 反事实剪枝(LLM) — 只处理前 50 对避免耗时过长
    Iter3: DAG 化 + 环检测 — 使用 networkx 确保 DAG 约束

    参考:
    - Beyond LLMs §3.4 — Graph Construction (Iter1-Iter3)
    - Shadow-Loom §5 — Causal Reasoning & Graph Construction

    容错: LLM 反事实剪枝失败时不阻塞，保留全部候选边。
    STAC Bond 验证失败也使用全部候选边。

    Args:
        state: CodecCrewLocalState，需包含 stac_labels

    Returns:
        dict: 更新 causal_graph / codec_stage
    """
    labels = state.get("stac_labels", [])
    if not labels:
        logger.info("[causal_graph] no stac labels, skipping graph construction")
        return {"codec_stage": "graph", "causal_graph": None}

    # ── Iter1: STAC Bond 学习 ──
    g: nx.DiGraph = nx.DiGraph()
    for i, lbl in enumerate(labels):
        event_id = f"event_{i}"
        g.add_node(event_id, label=lbl.text[:50], stac=lbl.stac_label)

    # 相邻句子 Bond
    edges_data: list[tuple[str, str, str]] = []
    for i in range(len(labels) - 1):
        src_label = labels[i].stac_label
        tgt_label = labels[i + 1].stac_label
        if src_label and tgt_label:
            key = (src_label, tgt_label)
            if key in VALID_STAC_BONDS:
                edges_data.append((f"event_{i}", f"event_{i + 1}", "causal"))

    # ── Iter2: 长程候选边 + 反事实剪枝 ──
    counterfactual_pairs: list[dict[str, str]] = []
    for i in range(len(labels)):
        for j in range(i + 2, len(labels)):
            src_label = labels[i].stac_label
            tgt_label = labels[j].stac_label
            if src_label and tgt_label:
                key = (src_label, tgt_label)
                if key in VALID_STAC_BONDS:
                    edges_data.append((f"event_{i}", f"event_{j}", "causal"))
                    counterfactual_pairs.append(
                        {
                            "cause": labels[i].text[:80],
                            "effect": labels[j].text[:80],
                        }
                    )

    # 反事实剪枝（只处理前 50 对避免耗时过长）
    if counterfactual_pairs:
        try:
            cf_agent = create_counterfactual_agent()
            cf_result = cf_agent.invoke(
                {
                    **state,
                    "counterfactual_pairs": counterfactual_pairs[:50],
                }
            )
            judgments = cf_result.get("counterfactual_judgments", [])

            # 只有长程边需要剪枝（短程边从相邻 Bond 得来，不剪）
            # 短程边数 = len(labels) - 1
            short_edge_count = max(0, len(labels) - 1)
            long_edges = edges_data[short_edge_count:]
            for jg, (_pair, edge) in zip(judgments, long_edges):
                if jg.get("answer") == "NO" and edge in edges_data:
                    edges_data.remove(edge)
        except Exception as e:
            logger.warning("[causal_graph] counterfactual pruning failed: %s", e)

    # ── Iter3: DAG 化 ──
    g.clear_edges()
    for src, tgt, rel in edges_data:
        g.add_edge(src, tgt, relation=rel)

    # 环检测与移除
    while not nx.is_directed_acyclic_graph(g):
        try:
            cycle = nx.find_cycle(g)
            g.remove_edge(cycle[0][0], cycle[0][1])
        except nx.NetworkXNoCycle:
            break

    # ── 转 CausalGraph ──
    events: list[EventNode] = []
    for node in g.nodes():
        parts = node.split("_")
        idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        events.append(
            EventNode(
                event_id=node,
                label=str(g.nodes[node].get("label", "")),
                fabula_time=idx,
                syuzhet_index=idx,
            )
        )
    events.sort(key=lambda e: e.syuzhet_index)
    edges = [(u, v, str(d.get("relation", "causal"))) for u, v, d in g.edges(data=True)]

    causal_graph = CausalGraph(events=events, edges=edges)

    return {
        "causal_graph": causal_graph,
        "codec_stage": "arc",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 6: 情绪曲线
# ═══════════════════════════════════════════════════════════════════════


def emotion_arc_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """情绪曲线 node — 调用 extract_emotion_arc tool。

    零 LLM，纯词典。使用情感词典 + 滑动窗口分析。
    参考: Shadow-Loom §4.4 — Emotion Arc Framework

    Args:
        state: CodecCrewLocalState，需包含 raw_text

    Returns:
        dict: 更新 emotion_arc / codec_stage
    """
    raw_text = state.get("raw_text", "")
    if not raw_text:
        logger.info("[emotion_arc] empty input, skipping")
        return {"codec_stage": "arc"}

    try:
        result_str = extract_emotion_arc.invoke({"text": raw_text})
        arc_data = json.loads(result_str)
        if isinstance(arc_data, dict) and "error" in arc_data:
            logger.error("[emotion_arc] tool error: %s", arc_data["error"])
            return {"codec_stage": "error", "codec_error": arc_data["error"]}

        arc = EmotionArc(
            valence_sequence=arc_data.get("valence_sequence", []),
            arousal_sequence=arc_data.get("arousal_sequence", []),
            window_positions=arc_data.get("window_positions", []),
            stages=arc_data.get("stages", []),
        )
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error("[emotion_arc] parse error: %s", e)
        return {"codec_stage": "error", "codec_error": f"parse error: {e}"}

    return {
        "emotion_arc": arc,
        "codec_stage": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 7: DAG 验证
# ═══════════════════════════════════════════════════════════════════════


def dag_validator_node(state: CodecCrewLocalState) -> dict[str, Any]:
    """DAG 验证 node — 验证因果图 DAG 约束 + 更新 crew_result。

    使用 networkx 验证因果图是否满足 DAG 约束，
    将验证结果写入 crew_result 供父图消费。

    容错: causal_graph 为 None 时默认验证失败，不抛出异常。

    Args:
        state: CodecCrewLocalState，需包含 causal_graph

    Returns:
        dict: 更新 crew_result.codec_* 字段
    """
    causal_graph = state.get("causal_graph")
    is_valid = False
    event_count = 0
    edge_count = 0

    if causal_graph and causal_graph.events:
        try:
            g: nx.DiGraph = nx.DiGraph()
            for e in causal_graph.events:
                g.add_node(e.event_id)
            for src, tgt, _ in causal_graph.edges:
                g.add_edge(src, tgt)
            is_valid = nx.is_directed_acyclic_graph(g)
            event_count = len(causal_graph.events)
            edge_count = len(causal_graph.edges)
        except Exception as e:
            logger.warning("[dag_validator] validation error: %s", e)
            is_valid = False

    existing_cr = state.get("crew_result", {})
    return {
        "crew_result": {
            **existing_cr,
            "codec_dag_valid": is_valid,
            "codec_events": event_count,
            "codec_edges": edge_count,
        },
        "codec_stage": "done",
    }


# ═══════════════════════════════════════════════════════════════════════
# Node 8: 退出
# ═══════════════════════════════════════════════════════════════════════


def _exit_for_codec(state: CodecCrewLocalState) -> dict[str, Any]:
    """Codec 子图结束节点。

    将 codec_stage 标记为 done，LangGraph 的 END 边据此判定子图结束。
    不执行任何业务逻辑。

    Args:
        state: CodecCrewLocalState

    Returns:
        dict: codec_stage = "done"
    """
    return {"codec_stage": "done"}


# ═══════════════════════════════════════════════════════════════════════
# 延迟导入辅助（避免模块级循环依赖）
# ═══════════════════════════════════════════════════════════════════════


def create_sentence_refiner_agent():
    """延迟导入 SentenceRefinerAgent 工厂。

    在函数内部导入以避免模块级循环依赖。
    遵循项目 delay-import 模式（参考 prepare_writing.py 的 SkillLoader 导入）。
    """
    from novelfactory.pipeline.narrative_codec.agents import (
        create_sentence_refiner_agent as _factory,
    )

    return _factory()


def create_stac_classifier_agent():
    """延迟导入 STACClassifierAgent 工厂。"""
    from novelfactory.pipeline.narrative_codec.agents import (
        create_stac_classifier_agent as _factory,
    )

    return _factory()


def create_counterfactual_agent():
    """延迟导入 CounterfactualAgent 工厂。"""
    from novelfactory.pipeline.narrative_codec.agents import (
        create_counterfactual_agent as _factory,
    )

    return _factory()
