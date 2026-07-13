"""Neo4j 增量更新模块（v7.3 新增）。

参考 GraphRAG (微软 2024) 的实体提取 + 关系提取 prompt 设计。
每章完成后增量更新 Neo4j — 新增实体、更新关系、同步世界状态。

用法:
    updater = Neo4jUpdater(neo4j_store)
    updater.update_from_chapter(chapter_text, chapter_number)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jUpdater:
    """每章完成后增量更新 Neo4j。

    职责:
        1. 提取本章新出现的实体（角色、地点、物品）
        2. 提取实体间的新关系
        3. 更新世界状态（如角色位置变化）
    """

    def __init__(self, neo4j_store) -> None:
        self._store = neo4j_store

    def update_from_chapter(
        self,
        chapter_text: str,
        chapter_number: int,
        existing_entity_names: list[str] | None = None,
    ) -> dict:
        """解析本章文本，增量更新 Neo4j。

        Args:
            chapter_text: 当前章节文本
            chapter_number: 当前章节序号
            existing_entity_names: 已有实体名列表（跳过已知实体）

        Returns:
            dict: {entities_added: int, relations_added: int, state_updates: list[str]}
        """
        if not chapter_text or not self._store.is_connected():
            return {"entities_added": 0, "relations_added": 0, "state_updates": []}

        result: dict[str, Any] = {
            "entities_added": 0,
            "relations_added": 0,
            "state_updates": [],
        }

        # 1. 提取实体
        entities: list[dict[str, Any]] = self._extract_entities(
            chapter_text[:6000], existing_entity_names or []
        )
        for ent in entities:
            try:
                self._store.upsert_entity(
                    name=ent["name"],
                    entity_type=ent["type"],
                    description=ent.get("description", ""),
                )
                result["entities_added"] += 1
            except Exception as e:
                logger.warning("Neo4jUpdater upsert_entity failed: %s", e)

        # 2. 提取关系
        relations: list[dict[str, Any]] = self._extract_relations(chapter_text[:6000])
        for rel in relations:
            try:
                self._store.upsert_relationship(
                    from_name=rel["from"],
                    to_name=rel["to"],
                    rel_type=rel["type"],
                    properties=rel.get("properties", {}),
                )
                result["relations_added"] += 1
            except Exception as e:
                logger.warning("Neo4jUpdater upsert_relationship failed: %s", e)

        # 3. 更新世界状态（角色位置等）
        state_changes: list[dict[str, Any]] = self._extract_state_changes(
            chapter_text[:6000]
        )
        for change in state_changes:
            try:
                self._store.set_entity_property(
                    name=change["entity"],
                    key=change["key"],
                    value=change["value"],
                )
                result["state_updates"].append(
                    f"{change['entity']}.{change['key']} = {change['value']}"
                )
            except Exception as e:
                logger.warning("Neo4jUpdater state update failed: %s", e)

        if result["entities_added"] > 0 or result["relations_added"] > 0:
            logger.info(
                "[Neo4jUpdater] ch%d: +%d entities, +%d relations, %d state changes",
                chapter_number,
                result["entities_added"],
                result["relations_added"],
                len(result["state_updates"]),
            )

        return result

    def _extract_entities(self, text: str, existing: list[str]) -> list[dict]:
        """用 LLM 提取本章新出现的实体。

        Prompt 参考 GraphRAG (微软 2024) graph_extractor.py 的实体提取设计。
        """
        from novelfactory.agents.infra.retry import llm_call_with_retry

        prompt = (
            f"从以下章节文本中提取新出现的重要实体。\n"
            f"不包含已存在的实体: {existing[:50]}\n\n"
            f"每行格式: 实体名 | 类型(character/location/object) | 简短描述\n\n"
            f"文本:\n{text[:4000]}"
        )
        try:
            from novelfactory.config.llm import get_worker_llm

            llm = get_worker_llm()
            response = llm_call_with_retry(
                llm, prompt, step_name="neo4j_extract_entities"
            )
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_entity_lines(raw)
        except Exception as e:
            logger.warning("Neo4jUpdater extract_entities failed: %s", e)
            return []

    def _parse_entity_lines(self, raw: str) -> list[dict]:
        """解析实体行。"""
        entities = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if (
                not line
                or line.startswith("#")
                or line.startswith("```")
                or line.startswith("---")
            ):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                entities.append(
                    {
                        "name": parts[0],
                        "type": parts[1] if len(parts) > 1 else "object",
                        "description": parts[2] if len(parts) > 2 else "",
                    }
                )
        return entities

    def _extract_relations(self, text: str) -> list[dict]:
        """提取实体间的新关系。"""
        from novelfactory.agents.infra.retry import llm_call_with_retry

        prompt = (
            f"从以下章节文本中提取实体间的核心关系。\n\n"
            f"每行格式: 实体A | 关系类型(英文大写如KNOWS/FAMILY/ALLIANCE/HOSTILE) | 实体B\n\n"
            f"文本:\n{text[:4000]}"
        )
        try:
            from novelfactory.config.llm import get_worker_llm

            llm = get_worker_llm()
            response = llm_call_with_retry(
                llm, prompt, step_name="neo4j_extract_relations"
            )
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_relation_lines(raw)
        except Exception as e:
            logger.warning("Neo4jUpdater extract_relations failed: %s", e)
            return []

    def _parse_relation_lines(self, raw: str) -> list[dict]:
        """解析关系行。"""
        relations = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                rel_type = parts[1].upper().replace(" ", "_")
                relations.append(
                    {
                        "from": parts[0],
                        "to": parts[2],
                        "type": rel_type,
                        "properties": {},
                    }
                )
        return relations

    def _extract_state_changes(self, text: str) -> list[dict]:
        """提取世界状态变化。"""
        from novelfactory.agents.infra.retry import llm_call_with_retry

        prompt = (
            f"从以下章节文本中提取角色的位置变化和状态变化。\n\n"
            f"每行格式: 实体名 | 属性名(英文如location/status) | 新值\n\n"
            f"文本:\n{text[:4000]}"
        )
        try:
            from novelfactory.config.llm import get_worker_llm

            llm = get_worker_llm()
            response = llm_call_with_retry(
                llm, prompt, step_name="neo4j_extract_states"
            )
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_state_lines(raw)
        except Exception as e:
            logger.warning("Neo4jUpdater extract_states failed: %s", e)
            return []

    def _parse_state_lines(self, raw: str) -> list[dict]:
        """解析状态行。"""
        changes = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                changes.append(
                    {
                        "entity": parts[0],
                        "key": parts[1].lower().replace(" ", "_"),
                        "value": parts[2],
                    }
                )
        return changes
