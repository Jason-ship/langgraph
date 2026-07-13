"""Leiden 社区检测（v7.3 新增）。

参考 GraphRAG (微软 2024) 的社区发现设计。
对 Neo4j 角色关系图做 Leiden 社区检测，自动发现"宗门/家族/阵营"等角色社群。

依赖: graspologic (pip install graspologic)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


def detect_communities(
    neo4j_driver,
    min_community_size: int = 3,
    max_cluster_size: int = 30,
    use_lcc: bool = True,
    randomness: float = 0.001,
    seed: int = 42,
) -> dict[str, list[str]]:
    """对 Neo4j 角色关系图做 Leiden 社区检测。

    Args:
        neo4j_driver: Neo4j GraphDatabase.driver 实例
        min_community_size: 最小社区大小（小于此的合并到最近的大社区）
        max_cluster_size: 最大集群大小（超出此的继续递归分割）

    Returns:
        dict: {community_name: [角色名列表]}
    """

    try:
        from graspologic.partition import hierarchical_leiden
    except ImportError:
        logger.warning(
            "graspologic 未安装，社区检测不可用。安装: pip install graspologic"
        )
        return _fallback_neighborhood(neo4j_driver)

    # 1. 从 Neo4j 导出角色关系边
    edges = _export_edges(neo4j_driver)
    if not edges:
        logger.info("[CommunityDetection] Neo4j 无边数据，跳过")
        return {}

    # 2. 构建 NetworkX 图（含边方向归一化 + 去重）
    # 参考 GraphRAG cluster_graph.py
    g = nx.Graph()
    edge_pairs: set[tuple[str, str]] = set()
    for a, b, w in edges:
        # 方向归一化: 保证 source < target（字典序）
        lo, hi = (a, b) if a <= b else (b, a)
        if (lo, hi) not in edge_pairs:
            edge_pairs.add((lo, hi))
            g.add_edge(lo, hi, weight=w or 0.1)

    if len(g.nodes) < min_community_size:
        logger.info("[CommunityDetection] 节点数不足%d，跳过", min_community_size)
        return {}

    # 3. Leiden 社区检测（参考 GraphRAG 的 cluster_graph.py）
    try:
        communities = hierarchical_leiden(
            g,
            max_cluster_size=max_cluster_size,
            randomness=randomness,
        )
    except Exception as e:
        logger.warning("Leiden 检测失败: %s，回退到 neighborhood", e)
        return _fallback_neighborhood(neo4j_driver)

    # 4. 按社区分组
    result: dict[str, list[str]] = defaultdict(list)
    for node, community_id, _ in communities:
        if community_id is not None:
            result[f"community_{community_id}"].append(node)

    # 5. 过滤小社区
    result = {
        name: members
        for name, members in result.items()
        if len(members) >= min_community_size
    }

    logger.info(
        "[CommunityDetection] 发现 %d 个角色社群: %s",
        len(result),
        {k: len(v) for k, v in result.items()},
    )

    return dict(result)


def _export_edges(neo4j_driver) -> list[tuple[str, str, float]]:
    """从 Neo4j 导出角色关系边。"""
    if not neo4j_driver:
        return []
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (a)-[r]-(b) "
                "WHERE (a:Character OR a:Entity) AND (b:Character OR b:Entity) "
                "RETURN a.name AS source, b.name AS target, "
                "CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1.0 END AS weight"
            )
            return [(r["source"], r["target"], r["weight"]) for r in result]
    except Exception as e:
        logger.warning("Neo4j export edges failed: %s", e)
        return []


def _fallback_neighborhood(neo4j_driver) -> dict[str, list[str]]:
    """回退方法：按角色间的直接关系做简单邻居分组。"""
    edges = _export_edges(neo4j_driver)
    if not edges:
        return {}

    adjacency: dict[str, set[str]] = defaultdict(set)
    for a, b, _ in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited: set[str] = set()
    communities: dict[str, list[str]] = {}

    for node in list(adjacency.keys()):
        if node in visited:
            continue

        # BFS 找连通分量
        group = []
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            group.append(curr)
            stack.extend(n for n in adjacency[curr] if n not in visited)

        if len(group) >= 3:
            communities[f"group_{len(communities)}"] = group

    return communities
