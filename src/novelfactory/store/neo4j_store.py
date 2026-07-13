"""Neo4j graph storage for character relationships and plot threads."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 关系类型白名单模式：仅允许 ASCII 字母/数字/下划线（Neo4j 关系类型不能参数化，
# 必须通过字符串拼接注入查询，因此白名单必须严格确保无注入风险）
_REL_TYPE_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_REL_TYPE_CLEANUP = re.compile(r"[^a-zA-Z0-9_]")


class Neo4jStore:
    """Neo4j graph storage for character relationships and plot threads."""

    def __init__(self, config) -> None:
        from neo4j import GraphDatabase

        self._driver = None
        host = getattr(config, "NEO4J_HOST", getattr(config, "neo4j_host", "localhost"))
        port = getattr(config, "NEO4J_PORT", getattr(config, "neo4j_port", "7687"))
        user = getattr(config, "NEO4J_USER", getattr(config, "neo4j_user", "neo4j"))
        password = getattr(
            config, "NEO4J_PASSWORD", getattr(config, "neo4j_password", "")
        )
        try:
            self._driver = GraphDatabase.driver(
                f"bolt://{host}:{port}",
                auth=(user, password),
            )
            self._driver.verify_connectivity()
            self._init_schema()
            logger.info("Neo4jStore connected to %s:%s", host, port)
        except Exception as e:
            logger.warning("Neo4jStore init failed: %s", e)

    def _init_schema(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Place) REQUIRE p.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:PlotThread) REQUIRE t.name IS UNIQUE"
            )

    def is_connected(self) -> bool:
        return self._driver is not None

    def upsert_character(self, name: str, properties: dict) -> None:
        """Create or update a character node.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug("[Neo4j] upsert_character name=%s", name)
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:Character {name: $name})
                SET c += $props
            """,
                name=name,
                props=properties,
            )

    def upsert_place(self, name: str, properties: dict = None) -> None:
        """Create or update a place node.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug("[Neo4j] upsert_place name=%s", name)
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (p:Place {name: $name})
                SET p += COALESCE($props, {})
            """,
                name=name,
                props=properties or {},
            )

    @staticmethod
    def _sanitize_rel_type(rel_type: str) -> str | None:
        """Validate and sanitize relationship type (Cypher injection prevention).

        Returns sanitized type string, or None if the input fails validation.

        Neo4j 关系类型不能参数化（仅支持字符串拼接），因此 sanitize 必须确保
        输出仅含 ASCII 字母、数字和下划线，且不以数字开头。
        """
        if not isinstance(rel_type, str) or not rel_type:
            logger.warning("Skipping empty/non-string relationship type: %r", rel_type)
            return None

        # 1) 替换所有不安全字符为下划线
        sanitized = _REL_TYPE_CLEANUP.sub("_", rel_type)

        # 2) 压缩连续下划线
        sanitized = re.sub(r"_+", "_", sanitized)

        # 3) 去除首尾下划线
        sanitized = sanitized.strip("_")

        # 4) 确保以字母或下划线开头（Neo4j 禁止以数字开头的关系类型名）
        if sanitized and sanitized[0].isdigit():
            sanitized = f"REL_{sanitized}"

        # 5) 如果为空或过长，拒绝
        if not sanitized or len(sanitized) > 64:
            logger.warning(
                "Skipping invalid relationship type: %r → sanitized=%r",
                rel_type,
                sanitized,
            )
            return None

        # 6) 最终白名单验证
        if not _REL_TYPE_PATTERN.match(sanitized):
            logger.warning(
                "Skipping relationship type (failed final validation): %r", rel_type
            )
            return None

        return sanitized

    def create_relationship(
        self, char1: str, rel_type: str, char2: str, properties: dict = None
    ) -> None:
        """Create a relationship between two characters.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug(
            "[Neo4j] create_relationship %s --[%s]--> %s", char1, rel_type, char2
        )
        if not self._driver:
            return
        safe_type = self._sanitize_rel_type(rel_type)
        if safe_type is None:
            return
        with self._driver.session() as session:
            session.run(
                f"""
                MATCH (a:Character {{name: $c1}})
                MATCH (b:Character {{name: $c2}})
                MERGE (a)-[r:{safe_type}]->(b)
                SET r += COALESCE($props, {{}})
            """,
                c1=char1,
                c2=char2,
                props=properties or {},
            )

    def create_relationships_batch(
        self, rel_type: str, pairs: list[dict], properties: dict = None
    ) -> None:
        """Batch create relationships between multiple character pairs.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug(
            "[Neo4j] create_relationships_batch rel_type=%s pairs=%d",
            rel_type,
            len(pairs),
        )
        if not self._driver or not pairs:
            return
        safe_type = self._sanitize_rel_type(rel_type)
        if safe_type is None:
            return
        with self._driver.session() as session:
            session.run(
                f"""
                UNWIND $pairs AS pair
                MATCH (a:Character {{name: pair.c1}})
                MATCH (b:Character {{name: pair.c2}})
                MERGE (a)-[r:{safe_type}]->(b)
                SET r += COALESCE($props, {{}})
            """,
                pairs=pairs,
                props=properties or {},
            )

    def create_location_relationship(self, char_name: str, location: str) -> None:
        """Create a relationship between a character and a location.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug(
            "[Neo4j] create_location_relationship char=%s location=%s",
            char_name,
            location,
        )
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (p:Place {name: $loc})
                WITH p
                MATCH (c:Character {name: $char})
                MERGE (c)-[:IS_AT]->(p)
            """,
                char=char_name,
                loc=location,
            )

    def upsert_plot_thread(
        self, name: str, description: str, chapter: int, status: str = "open"
    ) -> None:
        """Create or update a plot thread node.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug(
            "[Neo4j] upsert_plot_thread name=%s chapter=%s status=%s",
            name,
            chapter,
            status,
        )
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MERGE (t:PlotThread {name: $name})
                SET t.description = $desc, t.chapter = $ch, t.status = $status
            """,
                name=name,
                desc=description,
                ch=chapter,
                status=status,
            )

    def link_character_to_thread(self, char_name: str, thread_name: str) -> None:
        """Link a character to a plot thread.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug(
            "[Neo4j] link_character_to_thread char=%s thread=%s", char_name, thread_name
        )
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                """
                MATCH (c:Character {name: $c})
                MATCH (t:PlotThread {name: $t})
                MERGE (c)-[:INVOLVED_IN]->(t)
            """,
                c=char_name,
                t=thread_name,
            )

    def get_character_network(self, char_name: str, max_depth: int = 2) -> list[dict]:
        """查询角色关系网络。

        返回与该角色通过任意关系（包括 INVOLVED_IN/IS_AT 及自定义关系）
        间接相连的其他角色节点，最深 max_depth 跳。过滤关系类型会让角色
        通过剧情线/地点形成的间接连接丢失，因此这里不限制关系类型，
        仅约束终点节点必须是 :Character 以保持「角色网络」语义。
        """
        logger.debug(
            "[Neo4j] get_character_network name=%s depth=%s", char_name, max_depth
        )
        if not self._driver:
            return []
        depth = max(1, min(int(max_depth), 5))
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (c:Character {{name: $name}})-[*1..{depth}]-(other:Character)
                WHERE other.name <> $name
                RETURN other.name as character,
                       coalesce(
                           [r IN relationships(path) WHERE startNode(r) = c | type(r)][0],
                           type(last(relationships(path)))
                       ) as relation,
                       length(path) as distance
                ORDER BY distance ASC, character ASC
                LIMIT 50
            """,
                name=char_name,
            )
            return [dict(r) for r in result]

    def get_all_characters(self) -> list[str]:
        """Get all character names.

        v6.1: 添加运行时调用计数日志。
        """
        logger.debug("[Neo4j] get_all_characters")
        if not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(
                "MATCH (c:Character) RETURN c.name as name ORDER BY c.name"
            )
            return [r["name"] for r in result]

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    # ── v7.3: 通用实体操作（Neo4jUpdater 使用）────────────────────────────

    def upsert_entity(self, name: str, entity_type: str, description: str = "") -> bool:
        """创建或更新通用实体节点。"""
        from neo4j.exceptions import ClientError

        label_map = {"character": "Character", "location": "Place", "object": "Object"}
        label = label_map.get(entity_type.lower(), "Entity")
        try:
            with self._driver.session() as session:
                session.run(
                    f"MERGE (e:{label} {{name: $name}}) "
                    "SET e.description = $description, e.updated_at = timestamp()",
                    name=name,
                    description=description,
                )
                return True
        except ClientError as e:
            logger.warning("Neo4j upsert_entity failed: %s", e)
            return False

    def upsert_relationship(
        self,
        from_name: str,
        to_name: str,
        rel_type: str,
        properties: dict | None = None,
    ) -> bool:
        """创建或更新实体间关系。"""
        from neo4j.exceptions import ClientError

        sanitized = Neo4jStore._sanitize_rel_type(rel_type)
        if not sanitized:
            return False
        try:
            # 自动检测两端标签
            with self._driver.session() as session:
                query = (
                    f"MATCH (a) WHERE a.name = $from_name "
                    f"MATCH (b) WHERE b.name = $to_name "
                    f"MERGE (a)-[r:{sanitized}]->(b) "
                    f"SET r.updated_at = timestamp()"
                )
                session.run(query, from_name=from_name, to_name=to_name)
                return True
        except ClientError as e:
            logger.warning("Neo4j upsert_relationship failed: %s", e)
            return False

    def set_entity_property(self, name: str, key: str, value: str) -> bool:
        """设置实体节点的属性。"""
        try:
            with self._driver.session() as session:
                query = "MATCH (e) WHERE e.name = $name SET e[$key] = $value"
                session.run(query, name=name, key=key, value=value)
                return True
        except Exception as e:
            logger.warning("Neo4j set_entity_property failed: %s", e)
            return False

    # ── v7.3: 场景级子图检索（context_builder 使用）───────────────────────

    def query_subgraph(
        self, keywords: list[str], depth: int = 2, limit: int = 50
    ) -> str:
        """根据关键词查询相关子图，返回文本描述。

        Args:
            keywords: 关键词列表
            depth: 遍历深度
            limit: 最大返回条数

        Returns:
            子图文本描述（供 LLM prompt 注入）
        """
        if not keywords or not self.is_connected():
            return ""
        try:
            with self._driver.session() as session:
                query = (
                    "MATCH (n)-[r*1..$depth]-(m) "
                    "WHERE any(kw IN $keywords WHERE n.name CONTAINS kw OR m.name CONTAINS kw) "
                    "RETURN n.name as source, type(r[0]) as rel, m.name as target "
                    "LIMIT $limit"
                )
                result = session.run(query, keywords=keywords, depth=depth, limit=limit)
                lines = []
                seen = set()
                for record in result:
                    key = f"{record['source']}-{record['rel']}-{record['target']}"
                    if key not in seen:
                        seen.add(key)
                        lines.append(
                            f"{record['source']} —[{record['rel']}]→ {record['target']}"
                        )
                return "\n".join(lines) if lines else ""
        except Exception as e:
            logger.warning("Neo4j query_subgraph failed: %s", e)
            return ""
