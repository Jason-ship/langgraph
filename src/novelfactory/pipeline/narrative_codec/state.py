"""Narrative Codec Engine 子图局部状态定义。

继承 BaseCrewState (含 messages/crew_result/crew_error)。
所有编解码中间结果通过 crew_result 或顶层字段传递。

设计原则：
- 所有字段使用 ``Annotated[T, _last_value]`` 防多节点同一 tick 并发写入冲突
- messages 继承自 BaseCrewState，自带 add_messages reducer
- 与 NovelFactoryState 重叠的 key（如 messages）自动传播到父图
"""

from __future__ import annotations

from typing import Annotated

from novelfactory.pipeline.narrative_codec.schemas import (
    CausalGraph,
    EmotionArc,
    ExpertIndex,
    Scene,
    STACLabeledSentence,
)
from novelfactory.state.crew_state import BaseCrewState
from novelfactory.state.reducers import _last_value


class CodecCrewLocalState(BaseCrewState):
    """编解码子图局部状态。

    继承 BaseCrewState（含 messages/crew_result/crew_error）。
    所有编解码中间结果通过 crew_result 或顶层字段传递。
    """

    # ── 编入引擎中间结果 ──
    raw_text: Annotated[str, _last_value]
    """输入原始文本"""

    scenes: Annotated[list[Scene], _last_value]
    """场景分割结果"""

    refined_sentences: Annotated[list[str], _last_value]
    """LLM 精炼后句子"""

    stac_labels: Annotated[list[STACLabeledSentence], _last_value]
    """STAC 分类结果"""

    expert_indices: Annotated[list[ExpertIndex], _last_value]
    """ExpertIndex 提取结果"""

    causal_graph: Annotated[CausalGraph | None, _last_value]
    """因果图（可能为 None）"""

    emotion_arc: Annotated[EmotionArc | None, _last_value]
    """情绪曲线（可能为 None）"""

    # ── 控制字段 ──
    codec_stage: Annotated[str, _last_value]
    """当前处理阶段: split / refine / stac / expert / graph / arc / done"""

    codec_error: Annotated[str | None, _last_value]
    """错误信息（可选）"""

    total_cost: Annotated[float, _last_value]
    """LLM 调用累计成本（元）"""
