"""Codec Crew — 编解码子图构建。

7 个 node 串联流水线:
  scene_splitter → sentence_refiner → stac_classifier
  → expert_index → causal_graph_builder → emotion_arc → dag_validator → exit

遵循 build_writing_crew() 标准模式:
  - StateGraph(CodecCrewLocalState)
  - 编译不传 checkpointer（父图统一持久化）
  - recursion_limit = SUBGRAPH_RECURSION_LIMIT (200)

参考:
  - Beyond LLMs (ACL 2025) — STAC + Expert Index + Causal Graph
  - Shadow-Loom (arXiv 2026) — WorldStateV1 + 双时间轴
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from novelfactory.config.constants import SUBGRAPH_RECURSION_LIMIT
from novelfactory.pipeline.narrative_codec.nodes import (
    _exit_for_codec,
    causal_graph_builder_node,
    dag_validator_node,
    emotion_arc_node,
    expert_index_node,
    scene_splitter_node,
    sentence_refiner_node,
    stac_classifier_node,
)
from novelfactory.pipeline.narrative_codec.state import CodecCrewLocalState


def build_codec_crew() -> CompiledStateGraph:
    """构建编解码子图。

    7 个 node 串联流水线:
      scene_splitter → sentence_refiner → stac_classifier
      → expert_index → causal_graph_builder → emotion_arc → dag_validator → exit

    每个 node 都有独立 try/except 保护（在 nodes.py 中实现）。
    LLM 调用失败时保留前一步结果（codec_stage 不变）。
    因果图构建通过 networkx 环检测确保 DAG 约束。

    使用方式:
        graph.add_node("codec_crew", build_codec_crew())

    Returns:
        CompiledStateGraph: 编解码子图编译结果
    """
    graph = StateGraph(CodecCrewLocalState)

    # ── 添加节点 ──
    graph.add_node("scene_splitter", scene_splitter_node)
    graph.add_node("sentence_refiner", sentence_refiner_node)
    graph.add_node("stac_classifier", stac_classifier_node)
    graph.add_node("expert_index", expert_index_node)
    graph.add_node("causal_graph_builder", causal_graph_builder_node)
    graph.add_node("emotion_arc", emotion_arc_node)
    graph.add_node("dag_validator", dag_validator_node)
    graph.add_node("_exit_for_codec", _exit_for_codec)

    # ── 串联：START → 流水线 → END ──
    graph.add_edge(START, "scene_splitter")
    graph.add_edge("scene_splitter", "sentence_refiner")
    graph.add_edge("sentence_refiner", "stac_classifier")
    graph.add_edge("stac_classifier", "expert_index")
    graph.add_edge("expert_index", "causal_graph_builder")
    graph.add_edge("causal_graph_builder", "emotion_arc")
    graph.add_edge("emotion_arc", "dag_validator")
    graph.add_edge("dag_validator", "_exit_for_codec")
    graph.add_edge("_exit_for_codec", END)

    # ── 编译（不传 checkpointer — 父图统一持久化） ──
    compiled = graph.compile()
    compiled.recursion_limit = SUBGRAPH_RECURSION_LIMIT
    return compiled
