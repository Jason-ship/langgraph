"""database_writer 批量写入验收测试。

验证架构优化中引入的批量写入路径（executemany / UNWIND）：
    - PGStore.save_character_states_batch  → executemany 批量角色状态
    - PGStore.save_plot_threads_batch      → executemany 批量剧情线索
    - Neo4jStore.upsert_characters_batch   → UNWIND 批量角色节点
    - Neo4jStore.create_relationships_hetero_batch → UNWIND 批量关系（按类型分组）
    - database_writer._save_to_pg_node     → 节点内角色/线索均走 executemany

所有数据库连接均以 mock 隔离，不依赖真实 PostgreSQL / Neo4j 服务。
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from novelfactory.graph.subgraphs.database_writer import _save_to_pg_node
from novelfactory.store.neo4j_store import Neo4jStore
from novelfactory.store.postgres_store import DBConfig, PGStore

# ═══════════════════════════════════════════════════════════════════════════════
#  PGStore 批量方法
# ═══════════════════════════════════════════════════════════════════════════════


class TestPgSaveCharacterStatesBatch:
    """角色状态批量写入 — executemany 验证。"""

    def _store_with_mock_cursor(self, monkeypatch: pytest.MonkeyPatch) -> tuple[PGStore, mock.MagicMock]:
        """构造 PGStore（db_manager 隔离真实连接）并 mock _get_cursor。"""
        store = PGStore(config=DBConfig(), db_manager=mock.MagicMock())
        cur = mock.MagicMock()
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = cur
        monkeypatch.setattr(store, "_get_cursor", lambda: ctx)
        return store, cur

    def test_executemany_called_with_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """executemany 被调用且 rows 数量/结构正确。"""
        store, cur = self._store_with_mock_cursor(monkeypatch)

        characters = {
            "张三": {"location": "京城", "mood": "平静"},
            "李四": {"location": "边关", "power_level": "炼气三层"},
        }
        store.save_character_states_batch(
            project="测试项目", chapter=3, characters=characters
        )

        cur.executemany.assert_called_once()
        sql, rows = cur.executemany.call_args.args
        assert "novel_character_states" in sql
        assert len(rows) == 2

        # 行结构: (project, chapter, name, location, mood, power_level,
        #          status, relationships, knowledge, items, raw_state)
        row0 = rows[0]
        assert row0[0] == "测试项目"
        assert row0[1] == 3
        assert row0[2] == "张三"
        assert row0[3] == "京城"
        assert row0[4] == "平静"
        assert json.loads(row0[7]) == {}  # relationships JSON
        assert json.loads(row0[10]) == {"location": "京城", "mood": "平静"}  # raw_state

    def test_empty_characters_no_executemany(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """空 characters 直接返回，不触发 executemany。"""
        store, cur = self._store_with_mock_cursor(monkeypatch)

        store.save_character_states_batch(project="测试项目", chapter=1, characters={})

        cur.executemany.assert_not_called()


class TestPgSavePlotThreadsBatch:
    """剧情线索批量写入 — executemany 验证。"""

    def _store_with_mock_cursor(self, monkeypatch: pytest.MonkeyPatch) -> tuple[PGStore, mock.MagicMock]:
        store = PGStore(config=DBConfig(), db_manager=mock.MagicMock())
        cur = mock.MagicMock()
        ctx = mock.MagicMock()
        ctx.__enter__.return_value = cur
        monkeypatch.setattr(store, "_get_cursor", lambda: ctx)
        return store, cur

    def test_executemany_called_with_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """executemany 被调用且 rows 数量/结构正确。"""
        store, cur = self._store_with_mock_cursor(monkeypatch)

        threads = [
            {
                "thread_name": "主角复仇线",
                "description": "主角踏上复仇之路",
                "chapter": 3,
                "related_chars": ["张三"],
            },
            {
                "thread_name": "门派危机",
                "description": "门派遭遇外敌",
                "chapter": 4,
                "status": "open",
            },
        ]
        store.save_plot_threads_batch(project="测试项目", threads=threads)

        cur.executemany.assert_called_once()
        sql, rows = cur.executemany.call_args.args
        assert "novel_plot_threads" in sql
        assert len(rows) == 2

        # 行结构: (project, thread_name, status, chapter, description, related_chars_json)
        row0 = rows[0]
        assert row0[0] == "测试项目"
        assert row0[1] == "主角复仇线"
        assert row0[2] == "open"  # 默认状态
        assert row0[3] == 3
        assert row0[4] == "主角踏上复仇之路"
        assert json.loads(row0[5]) == ["张三"]

    def test_empty_threads_no_executemany(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """空 threads 直接返回，不触发 executemany。"""
        store, cur = self._store_with_mock_cursor(monkeypatch)

        store.save_plot_threads_batch(project="测试项目", threads=[])

        cur.executemany.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
#  Neo4jStore 批量方法
# ═══════════════════════════════════════════════════════════════════════════════


class TestNeo4jBatch:
    """Neo4j 批量写入 — UNWIND session.run 验证（driver 全 mock）。"""

    @pytest.fixture
    def mock_store(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Neo4jStore, mock.MagicMock]:
        """构造 Neo4jStore 并注入 mock driver，返回 (store, session_mock)。"""
        # patch neo4j 包的 GraphDatabase，避免真实 TCP 连接
        monkeypatch.setattr(
            "neo4j.GraphDatabase", mock.MagicMock(name="GraphDatabase")
        )
        store = Neo4jStore(config=DBConfig())
        assert store._driver is not None, "mock driver 未注入"
        session = mock.MagicMock(name="session")
        store._driver.session.return_value.__enter__.return_value = session
        return store, session

    def test_upsert_characters_batch(self, mock_store: tuple[Neo4jStore, mock.MagicMock]) -> None:
        """角色节点批量 upsert 走 UNWIND 单次 session.run。"""
        store, session = mock_store

        store.upsert_characters_batch(
            [
                {"name": "张三", "properties": {"status": "健在", "location": "京城"}},
                {"name": "李四", "properties": {"status": "重伤"}},
            ]
        )

        session.run.assert_called_once()
        cypher = session.run.call_args.args[0]
        params = session.run.call_args.kwargs
        assert "UNWIND" in cypher
        assert "MERGE" in cypher
        assert len(params["chars"]) == 2

    def test_upsert_characters_batch_empty_no_run(
        self, mock_store: tuple[Neo4jStore, mock.MagicMock]
    ) -> None:
        """空角色列表不触发任何查询。"""
        store, session = mock_store

        store.upsert_characters_batch([])

        session.run.assert_not_called()

    def test_create_relationships_hetero_batch(
        self, mock_store: tuple[Neo4jStore, mock.MagicMock]
    ) -> None:
        """异构关系按 sanitize 后类型分组，每类型一次 UNWIND。"""
        store, session = mock_store

        store.create_relationships_hetero_batch(
            [
                {
                    "source": "张三",
                    "target": "李四",
                    "rel_type": "is_friend",
                    "properties": {"since": 1},
                },
                {
                    "source": "李四",
                    "target": "王五",
                    "rel_type": "is_enemy",
                    "properties": {},
                },
            ]
        )

        # 2 个不同类型 → 2 组 → 2 次 session.run
        assert session.run.call_count == 2
        for call in session.run.call_args_list:
            cypher = call.args[0]
            params = call.kwargs
            assert "UNWIND" in cypher
            assert len(params["pairs"]) == 1

    def test_sanitize_invalid_rel_type_skipped(
        self, mock_store: tuple[Neo4jStore, mock.MagicMock]
    ) -> None:
        """非法关系类型被 sanitize 拒绝，不影响其余分组。"""
        store, session = mock_store

        store.create_relationships_hetero_batch(
            [
                {"source": "张三", "target": "李四", "rel_type": "is_friend"},
                {"source": "王五", "target": "赵六", "rel_type": "!! 非法类型 !!"},
            ]
        )

        assert session.run.call_count == 1  # 仅合法分组执行


# ═══════════════════════════════════════════════════════════════════════════════
#  database_writer._save_to_pg_node 批量调用
# ═══════════════════════════════════════════════════════════════════════════════


class TestDatabaseWriterSaveToPg:
    """_save_to_pg_node 节点 — 角色/剧情线索均走 executemany。"""

    @pytest.fixture
    def mock_db(self, monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
        """mock DatabaseManager，使节点函数不连接真实数据库。"""
        cur = mock.MagicMock()
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        manager = mock.MagicMock()
        manager.get_connection.return_value.__enter__.return_value = conn

        db_cls = mock.MagicMock()
        db_cls.get_instance.return_value = manager
        monkeypatch.setattr("novelfactory.config.database.DatabaseManager", db_cls)
        return cur

    def _make_state(self) -> dict:
        return {
            "project_name": "测试项目",
            "chapter_number": 2,
            "chapter_text": "测试文本。" * 300,
            "chapter_title": "第二章",
            "quality_score": 82.0,
            "extracted": {
                "characters": [
                    {"name": "张三", "location": "京城", "mood": "冷静"},
                    {"name": "李四", "location": "边关", "mood": "愤怒"},
                ],
                "events": [
                    {"event": "主角获得神剑", "characters": ["张三"]},
                    {"event": "门派遭遇袭击", "characters": ["李四"]},
                ],
            },
        }

    def test_characters_and_threads_use_executemany(self, mock_db: mock.MagicMock) -> None:
        """角色状态 + 剧情线索分别通过 executemany 批量写入。"""
        cur = mock_db

        result = _save_to_pg_node(self._make_state())

        executemany_calls = cur.executemany.call_args_list
        assert len(executemany_calls) == 2, "应恰好 2 次 executemany（角色 + 线索）"

        # 角色批量
        sql_chars, rows_chars = executemany_calls[0].args
        assert "novel_character_states" in sql_chars
        assert "ON CONFLICT" in sql_chars
        assert len(rows_chars) == 2

        # 线索批量
        sql_threads, rows_threads = executemany_calls[1].args
        assert "novel_plot_threads" in sql_threads
        assert "ON CONFLICT" in sql_threads
        assert len(rows_threads) == 2

        # 结果计数
        assert result["pg_result"]["characters_saved"] == 2
        assert result["pg_result"]["threads_saved"] == 2

    def test_empty_extracted_no_batch(self, mock_db: mock.MagicMock) -> None:
        """无角色/线索时跳过 executemany，仅章节单条写入。"""
        cur = mock_db
        state = self._make_state()
        state["extracted"] = {}

        result = _save_to_pg_node(state)

        cur.executemany.assert_not_called()
        assert result["pg_result"]["characters_saved"] == 0
        assert result["pg_result"]["threads_saved"] == 0
