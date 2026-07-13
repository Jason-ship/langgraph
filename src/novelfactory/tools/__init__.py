"""NovelFactory Tools — LangGraph @tool 注册模块。

所有工具通过 @tool 装饰器定义，可直接传入 create_react_agent 的 tools 参数。
工具按域分文件组织：
  neo4j_tools  — 人物关系图谱查询（Neo4j）
  milvus_tools — 向量语义检索（Milvus）
  feishu_tools — 飞书消息/文档/通知（httpx → tools-proxy）
"""

from novelfactory.tools.feishu_tools import get_feishu_tools
from novelfactory.tools.milvus_tools import get_milvus_tools
from novelfactory.tools.neo4j_tools import get_neo4j_tools

__all__ = [
    "get_neo4j_tools",
    "get_milvus_tools",
    "get_feishu_tools",
]
