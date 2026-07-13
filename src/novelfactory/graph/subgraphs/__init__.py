"""NovelFactory Subgraphs — LangGraph 子图模块。

每个子图封装一组相关节点，通过 state 通道通信。
所有逻辑都在图节点中，checkpoint 可追踪。

子图:
  context_builder  — 写前上下文构建（load_state → build_context → aggregate）
  state_extractor  — 写后状态提取（6 个 LLM 节点并行 → aggregate）
  database_writer  — 写后持久化（5 个 DB 节点并行 → aggregate）
"""

from novelfactory.graph.subgraphs.context_builder import build_context_builder
from novelfactory.graph.subgraphs.database_writer import build_database_writer
from novelfactory.graph.subgraphs.state_extractor import build_state_extractor

__all__ = [
    "build_context_builder",
    "build_state_extractor",
    "build_database_writer",
]
