"""
DatabaseWriter 子图 — 写后持久化
=================================
将 NovelStateTracker.after_chapter() 中的 DB 写入部分拆解为 LangGraph 子图节点。

节点:
  save_to_pg      — 角色状态 + 章节 + 剧情线索 → PostgreSQL
  save_to_milvus  — 章节向量 → Milvus
  save_to_neo4j   — 角色关系 → Neo4j
  save_phase2     — 审计 + 伏笔 + 节奏 → PostgreSQL
  save_phase3     — 断点 + 成本 + 质量 + 卷 → PostgreSQL

所有 DB 写入都是图节点，checkpoint 可追踪。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import defaultdict
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH_FOR_EXTRACTION = 100  # 最小文本长度阈值（低于此值跳过处理）

# ── Module-level singletons (v5.1.1: 避免每章创建新连接) ──────────────────────
# MilvusClient, EmbeddingService, Neo4j driver 作为模块级懒加载单例，避免每个章节
# 都重新创建 TCP 连接和 embedding 客户端。
# v5.2: 增加 threading.Lock 保护懒加载，避免多线程竞态重复创建连接。
_milvus_client: Any = None
_milvus_embedding_service: Any = None
_neo4j_driver: Any = None
_module_lock = threading.Lock()


def _get_milvus_client() -> Any:
    """获取模块级 MilvusClient 单例（线程安全）。"""
    global _milvus_client
    if _milvus_client is None:
        with _module_lock:
            # Double-checked locking
            if _milvus_client is None:
                from pymilvus import MilvusClient

                from novelfactory.config.settings import settings

                _milvus_client = MilvusClient(
                    uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                    timeout=10,
                    db_name="default",
                )
    return _milvus_client


def _get_milvus_embedding_service() -> Any:
    """获取模块级 EmbeddingService 单例（线程安全）。"""
    global _milvus_embedding_service
    if _milvus_embedding_service is None:
        with _module_lock:
            if _milvus_embedding_service is None:
                from novelfactory.store.tracker import EmbeddingService

                _milvus_embedding_service = EmbeddingService()
    return _milvus_embedding_service


def _get_neo4j_driver() -> Any:
    """获取模块级 Neo4j driver 单例（线程安全）。"""
    global _neo4j_driver
    if _neo4j_driver is None:
        with _module_lock:
            if _neo4j_driver is None:
                from neo4j import GraphDatabase

                from novelfactory.config.settings import settings

                _neo4j_driver = GraphDatabase.driver(
                    f"bolt://{settings.NEO4J_HOST}:{settings.NEO4J_PORT}",
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                )
    return _neo4j_driver


# ── State ──────────────────────────────────────────────────────────────────────


class DatabaseWriterState(TypedDict):
    """DatabaseWriter 子图的状态。

    输入: project_name, chapter_number, chapter_text, chapter_title,
          quality_score, extracted (来自 StateExtractor)
    输出: results (各 DB 写入结果)
    """

    project_name: str
    chapter_number: int
    chapter_text: str
    chapter_title: str
    quality_score: float
    extracted: dict  # 来自 StateExtractor 的提取结果
    actual_usage: NotRequired[dict]  # 实际 token 使用量（由 writing_crew 传入）

    # ── 写入结果 ──
    pg_result: dict
    milvus_result: dict
    neo4j_result: dict
    phase2_result: dict
    phase3_result: dict

    # ── 汇总 ──
    results: dict
    error: str


# ── Node: save_to_pg ───────────────────────────────────────────────────────────


def _save_to_pg_node(state: DatabaseWriterState) -> dict:
    """写入 PostgreSQL: 角色状态 + 章节 + 剧情线索。"""
    project = state.get("project_name", "")
    ch = state.get("chapter_number", 1)
    text = state.get("chapter_text", "")
    title = state.get("chapter_title", f"第{ch}章")
    score = state.get("quality_score", 0.0)
    extracted = state.get("extracted", {})

    result: dict[str, Any] = {
        "characters_saved": 0,
        "chapter_saved": False,
        "threads_saved": 0,
    }

    try:
        from novelfactory.config.database import DatabaseManager

        with DatabaseManager.get_instance().get_connection() as conn:
            conn.autocommit = True
            cur = conn.cursor()

            # 角色状态
            chars = extracted.get("characters", [])
            for c in chars:
                try:
                    cur.execute(
                        """
                        INSERT INTO novel_character_states
                            (project_name, chapter_number, character_name, location, mood,
                             power_level, status, relationships, knowledge, items, raw_state)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            project,
                            ch,
                            c.get("name", "?"),
                            c.get("location", ""),
                            c.get("mood", ""),
                            c.get("power_level", ""),
                            c.get("status", "健在"),
                            json.dumps(c.get("relationships", {}), ensure_ascii=False),
                            json.dumps(c.get("knowledge", []), ensure_ascii=False),
                            json.dumps(c.get("items", []), ensure_ascii=False),
                            json.dumps(c, ensure_ascii=False),
                        ),
                    )
                    result["characters_saved"] += 1
                except Exception as e:
                    logger.warning("save character %s failed: %s", c.get("name"), e)

            # 章节记录
            try:
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                cur.execute(
                    """
                    INSERT INTO novel_chapters
                        (project_name, chapter_number, title, word_count, quality_score, summary, full_text_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_name, chapter_number) DO UPDATE SET
                        title = EXCLUDED.title, word_count = EXCLUDED.word_count,
                        quality_score = EXCLUDED.quality_score, summary = EXCLUDED.summary,
                        full_text_hash = EXCLUDED.full_text_hash
                """,
                    (
                        project,
                        ch,
                        title,
                        len(text),
                        score,
                        text[:1000].replace("\n", " ")[:2000],
                        text_hash,
                    ),
                )
                result["chapter_saved"] = True
            except Exception as e:
                logger.warning("save chapter failed: %s", e)

            # 剧情线索
            events = extracted.get("events", [])
            for ev in events:
                try:
                    cur.execute(
                        """
                        INSERT INTO novel_plot_threads
                            (project_name, thread_name, status, created_chapter, description, related_characters)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_name, thread_name) DO UPDATE SET
                            status = EXCLUDED.status,
                            description = EXCLUDED.description,
                            related_characters = EXCLUDED.related_characters
                    """,
                        (
                            project,
                            ev.get("event", "")[:100],
                            "open",
                            ch,
                            ev.get("event", ""),
                            json.dumps(ev.get("characters", []), ensure_ascii=False),
                        ),
                    )
                    result["threads_saved"] += 1
                except Exception as e:
                    logger.warning("save thread failed: %s", e)

            cur.close()
    except Exception as e:
        logger.error("save_to_pg failed: %s", e)
        result["error"] = str(e)

    return {"pg_result": result}


# ── Node: save_to_milvus ───────────────────────────────────────────────────────


def _save_to_milvus_node(state: DatabaseWriterState) -> dict:
    """写入 Milvus: 章节向量嵌入 (v5.1.1: 复用模块级单例, 避免每章创建新连接)。"""
    project = state.get("project_name", "")
    ch = state.get("chapter_number", 1)
    text = state.get("chapter_text", "")

    result = {"embedding_saved": False}

    if not text or len(text) < _MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {"milvus_result": result}

    try:
        client = _get_milvus_client()
        emb = _get_milvus_embedding_service()
        embedding = emb.embed(text[:2000])
        client.insert(
            collection_name="novel_chapters",
            data=[
                {
                    "chapter_number": ch,
                    "project_name": project,
                    "vector": embedding,
                    "summary": text[:1000].replace("\n", " ")[:8000],
                }
            ],
        )
        result["embedding_saved"] = True
    except Exception as e:
        logger.warning("save_to_milvus failed: %s", e)

    return {"milvus_result": result}


# ── Node: save_to_neo4j ────────────────────────────────────────────────────────


def _save_to_neo4j_node(state: DatabaseWriterState) -> dict:
    """写入 Neo4j: 角色节点和关系 (v5.1.1: UNWIND 批量写入)。

    优化:
      - 角色节点: 使用 UNWIND 批量 MERGE, 一次事务完成所有角色 upsert
      - 角色关系: 收集后使用 UNWIND 批量创建, 15 角色从 105 次独立查询 → 1 次批量
      - 单次 session + 单次事务, 大幅减少往返开销
      - v5.2: 复用模块级 Neo4j driver 连接池，避免每章创建新连接
    """
    extracted = state.get("extracted", {})

    result = {"characters_upserted": 0, "relationships_created": 0}

    try:
        from novelfactory.graph.subgraphs.database_writer import _get_neo4j_driver

        driver = _get_neo4j_driver()

        chars = extracted.get("characters", [])
        if not chars:
            return {"neo4j_result": result}

        # ── 批量收集角色节点数据 ──
        char_params = []
        for c in chars:
            char_params.append(
                {
                    "name": c.get("name", "?"),
                    "status": c.get("status", "健在"),
                    "location": c.get("location", ""),
                    "power_level": c.get("power_level", ""),
                }
            )

        # ── 批量收集关系数据 ──
        # rel_type 在 Neo4j 中不能参数化，必须字符串拼接，因此先做严格白名单清洗
        # 复用 Neo4jStore._sanitize_rel_type 保证与底层写入逻辑一致
        from novelfactory.store.neo4j_store import Neo4jStore

        rel_params: list[dict[str, Any]] = []
        for c in chars:
            relationships = c.get("relationships", {})
            if not isinstance(relationships, dict):
                continue
            for target, rel_type in relationships.items():
                safe_type = Neo4jStore._sanitize_rel_type(str(rel_type))
                if safe_type is None:
                    logger.warning("Skipping invalid relationship type: %r", rel_type)
                    continue
                rel_params.append(
                    {
                        "c1": c.get("name", "?"),
                        "c2": str(target),
                        "rel_type": safe_type,
                    }
                )

        # ── 单次事务批量写入 ──
        with driver.session() as session:
            # 批量 upsert 角色节点
            if char_params:
                try:
                    session.run(
                        """
                        UNWIND $chars AS c
                        MERGE (ch:Character {name: c.name})
                        SET ch.status = c.status,
                            ch.location = c.location,
                            ch.power_level = c.power_level
                    """,
                        chars=char_params,
                    )
                    result["characters_upserted"] = len(char_params)
                except Exception as e:
                    logger.warning("neo4j batch upsert failed: %s", e)

            # 批量创建关系
            # 主路径：APOC 可用时单次 UNWIND + apoc.merge.relationship 支持动态类型
            # Fallback：按 sanitize 后的 rel_type 分组，纯 Cypher 字符串拼接 + UNWIND MERGE
            # 两层防御，避免 APOC 不可用时关系静默丢失
            if rel_params:
                created_count = _create_relationships_safely(session, rel_params)
                result["relationships_created"] = created_count

    except Exception as e:
        logger.warning("save_to_neo4j failed: %s", e)

    return {"neo4j_result": result}


def _create_relationships_safely(session: Any, rel_params: list[dict[str, Any]]) -> int:
    """批量写入关系，主路径走 APOC，失败则回退纯 Cypher 分组 MERGE。

    APOC 的 apoc.merge.relationship 支持动态 rel type 参数，但部署环境
    若未安装 APOC，主路径会抛错。回退路径按 sanitize 后的 rel_type 分组，
    每组用字符串拼接 + UNWIND MERGE 一次性写入（Neo4j 关系类型无法参数化，
    sanitize 已保证仅含 ASCII 字母数字下划线，无注入风险）。
    """
    created = 0

    # 主路径：APOC 批量
    try:
        result = session.run(
            """
            UNWIND $rels AS r
            MATCH (a:Character {name: r.c1})
            MATCH (b:Character {name: r.c2})
            CALL apoc.merge.relationship(a, r.rel_type, {}, {}, b, {})
            YIELD rel
            RETURN count(rel) AS created
        """,
            rels=rel_params,
        )
        record = result.single()
        created = int(record["created"]) if record else len(rel_params)
        return created
    except Exception:
        logger.debug(
            "apoc.merge.relationship unavailable, falling back to grouped MERGE"
        )

    # Fallback：按 rel_type 分组纯 Cypher 写入
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rel_params:
        grouped[r["rel_type"]].append({"c1": r["c1"], "c2": r["c2"]})

    for safe_type, pairs in grouped.items():
        try:
            session.run(
                f"""
                UNWIND $pairs AS pair
                MATCH (a:Character {{name: pair.c1}})
                MATCH (b:Character {{name: pair.c2}})
                MERGE (a)-[:{safe_type}]->(b)
            """,
                pairs=pairs,
            )
            created += len(pairs)
        except Exception as e:
            logger.warning("neo4j batch MERGE for rel_type=%s failed: %s", safe_type, e)

    return created


# ── Node: save_phase2 ──────────────────────────────────────────────────────────


def _save_phase2_node(state: DatabaseWriterState) -> dict:
    """写入 Phase2 数据: 审计报告 + 伏笔 + 节奏快照。"""
    project = state.get("project_name", "")
    ch = state.get("chapter_number", 1)
    extracted = state.get("extracted", {})

    result = {"audit_saved": False, "foreshadowing_saved": 0, "pacing_saved": False}

    try:
        from novelfactory.config.database import DatabaseManager

        with DatabaseManager.get_instance().get_connection() as conn:
            conn.autocommit = True
            cur = conn.cursor()

            # 审计报告
            audit = extracted.get("audit", {})
            if audit and audit.get("score") is not None:
                try:
                    findings_json = json.dumps(
                        [
                            {
                                "severity": f.get("severity", "minor"),
                                "category": f.get("category", "plot"),
                                "description": f.get("description", ""),
                                "evidence": f.get("evidence", ""),
                                "suggestion": f.get("suggestion", ""),
                                "chapter": ch,
                            }
                            for f in audit.get("findings", [])
                        ],
                        ensure_ascii=False,
                    )
                    cur.execute(
                        """
                        INSERT INTO novel_audit_reports
                            (project_name, chapter_start, chapter_end, findings_json,
                             overall_score, summary, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                        (
                            project,
                            ch,
                            ch,
                            findings_json,
                            audit.get("score", 100),
                            audit.get("summary", ""),
                        ),
                    )
                    result["audit_saved"] = True
                except Exception as e:
                    logger.warning("save audit failed: %s", e)

            # 伏笔
            foreshadowing = extracted.get("foreshadowing", [])
            for fs in foreshadowing:
                try:
                    status = "resolved" if fs.get("action") == "resolved" else "planted"
                    cur.execute(
                        """
                        INSERT INTO novel_foreshadowing
                            (project_name, name, description, planted_chapter,
                             planned_resolve_chapter, priority, status,
                             related_characters, category, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_name, name) DO UPDATE SET
                            description = EXCLUDED.description,
                            planned_resolve_chapter = EXCLUDED.planned_resolve_chapter,
                            priority = EXCLUDED.priority,
                            status = EXCLUDED.status,
                            related_characters = EXCLUDED.related_characters,
                            category = EXCLUDED.category,
                            notes = EXCLUDED.notes,
                            updated_at = NOW()
                    """,
                        (
                            project,
                            fs.get("name", ""),
                            fs.get("description", ""),
                            ch,
                            fs.get("planned_resolve_chapter", 0),
                            fs.get("priority", 5),
                            status,
                            json.dumps(
                                fs.get("related_characters", []), ensure_ascii=False
                            ),
                            fs.get("category", "plot"),
                            "",
                        ),
                    )
                    if fs.get("action") == "resolved":
                        cur.execute(
                            """
                            UPDATE novel_foreshadowing
                            SET actual_resolve_chapter = %s, updated_at = NOW()
                            WHERE project_name = %s AND name = %s
                        """,
                            (ch, project, fs.get("name", "")),
                        )
                    result["foreshadowing_saved"] += 1
                except Exception as e:
                    logger.warning("save foreshadowing failed: %s", e)

            # 节奏快照
            pacing = extracted.get("pacing", {})
            if pacing:
                try:
                    cur.execute(
                        """
                        INSERT INTO novel_pacing_snapshots
                            (project_name, chapter_number, intensity, event_density,
                             dialogue_ratio, action_ratio, description_ratio, pacing_label)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_name, chapter_number) DO UPDATE SET
                            intensity = EXCLUDED.intensity,
                            event_density = EXCLUDED.event_density,
                            dialogue_ratio = EXCLUDED.dialogue_ratio,
                            action_ratio = EXCLUDED.action_ratio,
                            description_ratio = EXCLUDED.description_ratio,
                            pacing_label = EXCLUDED.pacing_label
                    """,
                        (
                            project,
                            ch,
                            pacing.get("intensity", 5.0),
                            pacing.get("event_density", 5.0),
                            pacing.get("dialogue_ratio", 0.3),
                            pacing.get("action_ratio", 0.3),
                            pacing.get("description_ratio", 0.3),
                            pacing.get("pacing_label", "balanced"),
                        ),
                    )
                    result["pacing_saved"] = True
                except Exception as e:
                    logger.warning("save pacing failed: %s", e)

            cur.close()
    except Exception as e:
        logger.warning("save_phase2 failed: %s", e)

    return {"phase2_result": result}


# ── Node: save_phase3 ──────────────────────────────────────────────────────────


def _save_phase3_node(state: DatabaseWriterState) -> dict:
    """写入 Phase3 数据: 断点日志 + 成本 + 质量 + 卷检测。"""
    project = state.get("project_name", "")
    ch = state.get("chapter_number", 1)
    text = state.get("chapter_text", "")
    score = state.get("quality_score", 0.0)

    result = {
        "checkpoint_logged": False,
        "cost_recorded": False,
        "quality_recorded": False,
        "volume_checked": False,
    }

    try:
        from novelfactory.config.database import DatabaseManager

        with DatabaseManager.get_instance().get_connection() as conn:
            conn.autocommit = True
            cur = conn.cursor()

            # 断点日志
            try:
                cur.execute(
                    """
                    INSERT INTO novel_checkpoint_log
                        (project_name, chapter_number, status, details)
                    VALUES (%s, %s, %s, %s)
                """,
                    (project, ch, "ok", ""),
                )
                result["checkpoint_logged"] = True
            except Exception as e:
                logger.warning("log checkpoint failed: %s", e)

            # 成本记录：使用实际 token 使用量（优先使用传入的 actual_usage，回退到 read_usage_tracking）
            try:
                from novelfactory.agents.infra import read_usage_tracking
                from novelfactory.config.pricing import calc_cost

                actual_usage = state.get("actual_usage") or read_usage_tracking()
                input_tokens = actual_usage.get("prompt_tokens", 0)
                output_tokens = actual_usage.get("completion_tokens", 0)
                cost = calc_cost(input_tokens, output_tokens, "deepseek-v4-flash")
                cur.execute(
                    """
                    INSERT INTO novel_cost_records
                        (project_name, chapter_number, model, input_tokens, output_tokens, cost_rmb, phase)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        project,
                        ch,
                        "deepseek-v4-flash",
                        input_tokens,
                        output_tokens,
                        round(cost, 6),
                        "writing",
                    ),
                )
                result["cost_recorded"] = True
            except Exception as e:
                logger.warning("record cost failed: %s", e)

            # 质量记录
            try:
                if score > 0:
                    cur.execute(
                        """
                        INSERT INTO novel_quality_trends
                            (project_name, chapter_number, quality_score, word_count,
                             rewrite_count, audit_score)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_name, chapter_number) DO UPDATE SET
                            quality_score = EXCLUDED.quality_score,
                            word_count = EXCLUDED.word_count,
                            rewrite_count = EXCLUDED.rewrite_count,
                            audit_score = EXCLUDED.audit_score
                    """,
                        (project, ch, score, len(text), 0, 100.0),
                    )
                    result["quality_recorded"] = True
            except Exception as e:
                logger.warning("record quality failed: %s", e)

            # 卷完成检测
            try:
                cur.execute(
                    """
                    SELECT volume_number, title, end_chapter, status
                    FROM novel_volumes
                    WHERE project_name = %s
                      AND start_chapter <= %s
                      AND (end_chapter >= %s OR end_chapter = 0)
                    ORDER BY volume_number DESC
                    LIMIT 1
                """,
                    (project, ch, ch),
                )
                row = cur.fetchone()
                if row:
                    vol_num, vol_title, end_ch, vol_status = row
                    if end_ch > 0 and ch >= end_ch and vol_status != "completed":
                        cur.execute(
                            """
                            UPDATE novel_volumes SET status = 'completed', updated_at = NOW()
                            WHERE project_name = %s AND volume_number = %s
                        """,
                            (project, vol_num),
                        )
                    result["volume_checked"] = True
            except Exception as e:
                logger.warning("check volume failed: %s", e)

            cur.close()
    except Exception as e:
        logger.warning("save_phase3 failed: %s", e)

    return {"phase3_result": result}


# ── Node: aggregate ────────────────────────────────────────────────────────────


def _aggregate_results_node(state: DatabaseWriterState) -> dict:
    """合并所有写入结果 + v5.12 精确告警（区分真实错误与空数据跳过）。"""
    pg_result = state.get("pg_result", {})
    milvus_result = state.get("milvus_result", {})
    neo4j_result = state.get("neo4j_result", {})
    phase2_result = state.get("phase2_result", {})
    phase3_result = state.get("phase3_result", {})
    extracted = state.get("extracted", {})

    # v5.12: 区分真实 DB 错误与空数据跳过（空数据不告警）。
    # Neo4j 和 audit 在 extracted 为空时跳跳过是正常行为。
    has_characters = bool(extracted.get("characters"))
    has_audit = bool(extracted.get("audit"))

    db_failures: list[str] = []
    if pg_result.get("error"):
        db_failures.append(f"PG: {pg_result['error'][:80]}")
    if pg_result.get("chapter_saved") is False and pg_result.get("error"):
        db_failures.append("PG: chapter save failed")
    if milvus_result.get("embedding_saved") is False and milvus_result.get("error"):
        db_failures.append("Milvus: embedding not saved")
    if neo4j_result.get("characters_upserted", 0) == 0 and has_characters:
        db_failures.append("Neo4j: no characters upserted")
    if phase2_result.get("audit_saved") is False and has_audit:
        db_failures.append("PG: audit report not saved")

    if db_failures:
        logger.warning(
            "[database_writer] %d DB 写入失败: %s",
            len(db_failures),
            "; ".join(db_failures),
        )
    elif not has_characters:
        logger.info(
            "[database_writer] extracted 为空，角色/审计/伏笔等写入已跳过（预期行为）"
        )

    return {
        "results": {
            "pg": pg_result,
            "milvus": milvus_result,
            "neo4j": neo4j_result,
            "phase2": phase2_result,
            "phase3": phase3_result,
            "db_failures": db_failures,
            "db_failure_count": len(db_failures),
        }
    }


# ── Graph Builder ──────────────────────────────────────────────────────────────


def build_database_writer() -> CompiledStateGraph:
    """构建 DatabaseWriter 子图。

    5 个 DB 写入节点并行执行，最后 aggregate 合并结果。
    """
    builder = StateGraph(DatabaseWriterState)

    builder.add_node("save_to_pg", _save_to_pg_node)
    builder.add_node("save_to_milvus", _save_to_milvus_node)
    builder.add_node("save_to_neo4j", _save_to_neo4j_node)
    builder.add_node("save_phase2", _save_phase2_node)
    builder.add_node("save_phase3", _save_phase3_node)
    builder.add_node("aggregate", _aggregate_results_node)

    # 并行写入
    builder.add_edge(START, "save_to_pg")
    builder.add_edge(START, "save_to_milvus")
    builder.add_edge(START, "save_to_neo4j")
    builder.add_edge(START, "save_phase2")
    builder.add_edge(START, "save_phase3")

    # 汇聚
    builder.add_edge("save_to_pg", "aggregate")
    builder.add_edge("save_to_milvus", "aggregate")
    builder.add_edge("save_to_neo4j", "aggregate")
    builder.add_edge("save_phase2", "aggregate")
    builder.add_edge("save_phase3", "aggregate")

    builder.add_edge("aggregate", END)

    return builder.compile()
