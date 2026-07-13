"""Neo4j 工具集 — 人物关系图谱查询工具。

通过 @tool 装饰器封装 Neo4jStore 的查询方法，
让 LLM Agent 能自主决定何时查询角色关系网络。

使用方式：
    tools = get_neo4j_tools()
    agent = create_react_agent(llm, tools=tools, prompt=...)
"""

from __future__ import annotations

import json
import logging
import threading

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 模块级单例（线程安全懒加载）──────────────────────────────────────────
_neo4j_store = None
_lock = threading.Lock()


def _get_store():
    """懒加载 Neo4jStore 单例。"""
    global _neo4j_store
    if _neo4j_store is None:
        with _lock:
            if _neo4j_store is None:
                from novelfactory.config.settings import settings
                from novelfactory.store.neo4j_store import Neo4jStore

                _neo4j_store = Neo4jStore(settings)
                if not _neo4j_store.is_connected():
                    logger.warning("[neo4j_tools] Neo4j 连接失败，工具将返回空结果")
    return _neo4j_store


# ── @tool 定义 ──────────────────────────────────────────────────────────────


@tool
def get_character_network(character_name: str, max_depth: int = 2) -> str:
    """查询指定角色的关系网络（Neo4j 图遍历）。

    返回该角色的所有关系（朋友、敌人、保护者等）以及关系网络中的其他角色。
    适用于：了解角色之间的关系、发现隐藏的关系链、检查角色状态。

    Args:
        character_name: 角色名称（精确匹配）
        max_depth: 关系遍历深度，默认 2（即朋友的朋友），最大 3
    """
    store = _get_store()
    if not store or not store.is_connected():
        return json.dumps({"error": "Neo4j 未连接"}, ensure_ascii=False)
    try:
        results = store.get_character_network(character_name, min(max_depth, 3))
        if not results:
            return json.dumps(
                {"message": f"未找到角色 '{character_name}' 或其关系网络为空"},
                ensure_ascii=False,
            )
        return json.dumps(results, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error("[neo4j_tools] get_character_network error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def get_all_characters() -> str:
    """获取所有已登记的角色列表。

    返回当前小说中所有角色的名称列表。
    适用于：快速了解有哪些角色可用、检查角色是否已存在。
    """
    store = _get_store()
    if not store or not store.is_connected():
        return json.dumps({"error": "Neo4j 未连接"}, ensure_ascii=False)
    try:
        characters = store.get_all_characters()
        return json.dumps(
            {"characters": characters, "count": len(characters)},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("[neo4j_tools] get_all_characters error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def get_character_info(character_name: str) -> str:
    """获取指定角色的详细信息（属性 + 所有关系）。

    返回角色的属性（位置、情绪、能力、状态等）以及所有关系。
    适用于：写作前了解角色当前状态、检查角色发展轨迹。

    Args:
        character_name: 角色名称（精确匹配）
    """
    store = _get_store()
    if not store or not store.is_connected():
        return json.dumps({"error": "Neo4j 未连接"}, ensure_ascii=False)
    try:
        network = store.get_character_network(character_name, max_depth=1)
        # store.get_character_network 返回该角色的 1 跳邻居，结构为
        # {character: 邻居名, relation: 关系类型, distance: 跳数}
        # max_depth=1 时所有返回项都是直接关系，直接透传即可
        direct_relations = [
            {
                "neighbor": rel.get("character", ""),
                "relation": rel.get("relation", ""),
                "distance": rel.get("distance", 0),
            }
            for rel in network
        ]
        return json.dumps(
            {
                "character": character_name,
                "relations": direct_relations,
                "relation_count": len(direct_relations),
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        logger.error("[neo4j_tools] get_character_info error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def get_plot_threads(status_filter: str = "open") -> str:
    """获取剧情线索列表。

    返回当前活跃的剧情线索（名称、描述、起始章节、状态）。
    适用于：检查有哪些未完结的伏笔、确认剧情线索发展状态。

    Args:
        status_filter: 状态过滤，可选值：'open'（未完结，默认）、'closed'（已完结）、'all'（全部）
    """
    store = _get_store()
    if not store or not store.is_connected():
        return json.dumps({"error": "Neo4j 未连接"}, ensure_ascii=False)
    try:
        with store._driver.session() as session:
            check = session.run("MATCH (t:PlotThread) RETURN count(t) AS cnt").single()
            if not check or check["cnt"] == 0:
                return json.dumps({"plot_threads": [], "count": 0}, ensure_ascii=False)
            result = session.run(
                "MATCH (t:PlotThread) RETURN t.name AS name, properties(t) AS props"
            )
            threads = []
            for record in result:
                props = record["props"] or {}
                status = props.get("status", "open")
                if status_filter != "all" and status != status_filter:
                    continue
                threads.append(
                    {
                        "name": record["name"],
                        "desc": props.get("description", ""),
                        "chapter": props.get("chapter", 0),
                        "status": status,
                    }
                )
        return json.dumps(
            {"plot_threads": threads, "count": len(threads)},
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        logger.error("[neo4j_tools] get_plot_threads error: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ── 工具集导出 ──────────────────────────────────────────────────────────────


def get_neo4j_tools() -> list:
    """返回 Neo4j 工具列表，可直接传入 create_react_agent。"""
    return [
        get_character_network,
        get_all_characters,
        get_character_info,
        get_plot_threads,
    ]
