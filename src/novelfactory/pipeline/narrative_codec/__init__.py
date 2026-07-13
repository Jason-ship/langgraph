"""Narrative Codec Engine — 编解码引擎 LangGraph子图。

本模块实现小说级文本的"编入→编出→校验"流水线。
遵循项目LangGraph标准模式: tools→agents→nodes→crew。

使用:
    from novelfactory.pipeline.narrative_codec import build_codec_crew
    codec_crew = build_codec_crew()
    # 在根图中: graph.add_node("codec_crew", codec_crew)

参考论文:
- LitVISTA (ACL 2026): VISTA Space叙事编排框架
- Beyond LLMs (ACL 2025): STAC四分类+Expert Index
- Shadow-Loom (arXiv 2026): WorldStateV1+双时间轴+因果推理
- NK Weaver (arXiv 2026): 多Agent叙事图谱构建
- LLM×MapReduce (ACL 2025): 结构化协议+置信度校准
- ConStory-Bench (arXiv 2026): 5×19一致性错误分类
"""

from novelfactory.pipeline.narrative_codec.crew import build_codec_crew

__all__ = ["build_codec_crew"]
