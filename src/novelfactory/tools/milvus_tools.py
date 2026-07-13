"""Milvus 工具集 — 向量语义检索工具。

通过 @tool 装饰器封装 MilvusStore 的检索方法，
让 LLM Agent 能自主决定何时检索相似章节和写作指南。

使用方式：
    tools = get_milvus_tools()
    agent = create_react_agent(llm, tools=tools, prompt=...)
"""

from __future__ import annotations

import json
import logging
import threading

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 模块级单例（线程安全懒加载）──────────────────────────────────────────
_milvus_store = None
_embedding_service = None
_guide_store = None
_lock = threading.Lock()


def _get_milvus():
    """懒加载 MilvusStore 单例。"""
    global _milvus_store
    if _milvus_store is None:
        with _lock:
            if _milvus_store is None:
                from novelfactory.config.settings import settings
                from novelfactory.store.milvus_store import MilvusStore

                _milvus_store = MilvusStore(settings)
                if not _milvus_store.is_connected():
                    logger.warning("[milvus_tools] Milvus 连接失败，工具将返回空结果")
    return _milvus_store


def _get_embedding():
    """懒加载 EmbeddingService 单例。"""
    global _embedding_service
    if _embedding_service is None:
        with _lock:
            if _embedding_service is None:
                from novelfactory.store.embedding import EmbeddingService

                _embedding_service = EmbeddingService()
    return _embedding_service


def _get_guide_store():
    """懒加载 WritingGuideStore 单例。"""
    global _guide_store
    if _guide_store is None:
        with _lock:
            if _guide_store is None:
                from novelfactory.store.guide_store import WritingGuideStore

                _guide_store = WritingGuideStore()
    return _guide_store


# ── @tool 定义 ──────────────────────────────────────────────────────────────


@tool
def search_similar_chapters(query: str, top_k: int = 3, project: str = "") -> str:
    """语义搜索与查询内容相似的历史章节（Milvus 向量检索）。

    将查询文本向量化后，在已写章节的向量库中检索最相似的章节。
    适用于：写作前回顾相似场景的处理方式、检查是否与已有章节重复、
    寻找类似情节的写法参考。

    Args:
        query: 要搜索的内容描述（自然语言），如"主角与反派第一次对决的场景"
        top_k: 返回最相似的章节数量，默认 3，最大 10
        project: 项目名称（限定搜索范围），如"我的小说"。空字符串表示搜索全部项目。
    """
    store = _get_milvus()
    embedding_svc = _get_embedding()
    if not store or not store.is_connected():
        return json.dumps({"error": "Milvus 未连接"}, ensure_ascii=False)
    if not embedding_svc:
        return json.dumps({"error": "Embedding 服务不可用"}, ensure_ascii=False)
    try:
        query_vec = embedding_svc.embed(query[:8000])
        if not query_vec or all(v == 0 for v in query_vec):
            return json.dumps(
                {"error": "Embedding 生成失败（API Key 可能未配置）"},
                ensure_ascii=False,
            )
        results = store.search_similar(query_vec, top_k=min(top_k, 10), project=project)
        return json.dumps(
            {"query": query, "results": results, "count": len(results)},
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        logger.error("[milvus_tools] search_similar_chapters error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def search_writing_guides(query: str, top_k: int = 3) -> str:
    """语义搜索写作指南（Milvus 向量检索）。

    在写作指南知识库中检索与查询最相关的指南内容。
    适用于：写作技巧查询、风格指导、题材规范查询。

    Args:
        query: 要搜索的写作问题或需求（自然语言），如"如何写好悬疑氛围"
        top_k: 返回最相关的指南数量，默认 3，最大 5
    """
    guide_store = _get_guide_store()
    embedding_svc = _get_embedding()
    if not guide_store:
        return json.dumps({"error": "WritingGuideStore 不可用"}, ensure_ascii=False)
    if not embedding_svc:
        return json.dumps({"error": "Embedding 服务不可用"}, ensure_ascii=False)
    try:
        results = guide_store.search(query, top_k=min(top_k, 5))
        return json.dumps(
            {"query": query, "guides": results, "count": len(results)},
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        logger.error("[milvus_tools] search_writing_guides error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def get_chapter_summary(chapter_number: int, project: str = "") -> str:
    """获取指定章节的摘要（从 Milvus 向量库）。

    从已存储的章节摘要中获取指定章节的内容概要。
    适用于：快速回顾某一章的剧情、检查连续性。

    Args:
        chapter_number: 章节号
        project: 项目名称（限定搜索范围），如"我的小说"。空字符串表示搜索全部项目。
    """
    store = _get_milvus()
    if not store or not store.is_connected():
        return json.dumps({"error": "Milvus 未连接"}, ensure_ascii=False)
    try:
        filters = [f"chapter_number == {chapter_number}"]
        if project:
            filters.append(f'project_name == "{project}"')
        results = store._client.query(
            collection_name=store.COLLECTION_NAME,
            filter=" and ".join(filters),
            output_fields=["chapter_number", "summary", "project_name"],
            limit=1,
        )
        if results:
            return json.dumps(results[0], ensure_ascii=False, default=str)
        return json.dumps(
            {"message": f"未找到第 {chapter_number} 章的摘要"}, ensure_ascii=False
        )
    except Exception as e:
        logger.error("[milvus_tools] get_chapter_summary error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── 工具集导出 ──────────────────────────────────────────────────────────────


def get_milvus_tools() -> list:
    """返回 Milvus 工具列表，可直接传入 create_react_agent。"""
    return [
        search_similar_chapters,
        search_writing_guides,
        get_chapter_summary,
    ]
