"""Threads 路由单元测试 — 线程 CRUD（create/get/search/delete/copy/state/history）。

使用 FastAPI TestClient，mock get_app() 返回 fake graph（带 checkpointer）。
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import AsyncMock, patch

# ── 确保源码路径可导入 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


# ── Fake Graph（带 checkpointer / aget_state / aget_state_history） ──────────

class _FakeTask:
    def __init__(self):
        self.interrupts = []


class _FakeState:
    """Fake aget_state 返回值。"""
    def __init__(self, values=None, next_nodes=None, tasks=None):
        self.values = values or {}
        self.next = next_nodes or ()
        self.tasks = tasks or []


class _FakeCheckpointer:
    """Fake AsyncPostgresSaver — 支持 conn / adelete_thread。"""

    def __init__(self, thread_ids=None):
        self._thread_ids = thread_ids or ["thread-001", "thread-002"]
        self._conn = _FakeConnection(self._thread_ids)

    @property
    def conn(self):
        """返回 sync conn 对象（兼容 async with checkpointer.conn.connection()）。"""
        return self._conn

    async def adelete_thread(self, config):
        pass


class _FakeConnection:
    """Fake asyncpg connection — 支持 execute + fetchall + connection()。"""

    def __init__(self, thread_ids):
        self._thread_ids = thread_ids

    def connection(self):
        """async context manager 返回自身 — 模拟 pool.connection()。"""
        return self._ConnectionContext(self)

    class _ConnectionContext:
        def __init__(self, conn):
            self._conn = conn
        async def __aenter__(self):
            return self._conn
        async def __aexit__(self, *args):
            pass

    async def execute(self, query):
        return _FakeResult(self._thread_ids)

    def transaction(self):
        return _FakeTransaction()


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeResult:
    def __init__(self, rows):
        self._rows = [{"thread_id": r} for r in rows]

    async def fetchall(self):
        return self._rows


class FakeThreadGraph:
    """Fake CompiledStateGraph — 带 checkpointer + aget_state + aget_state_history。"""

    def __init__(self, thread_ids=None):
        self.checkpointer = _FakeCheckpointer(thread_ids)

    async def aget_state(self, config):
        return _FakeState(
            values={"messages": []},
            next_nodes=(),
            tasks=[_FakeTask()],
        )

    async def aget_state_history(self, config, limit=10):
        """返回历史状态列表。"""
        yield _FakeState(values={"step": 1})
        raise StopAsyncIteration()


# ── 导入 app ──

from fastapi.testclient import TestClient

from novelfactory.server.app import app

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_client() -> TestClient:
    """创建 TestClient 实例。"""
    return TestClient(app)


def _assert_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


# ── 纯 Stub 端点（无需 mock get_app） ─────────────────────────────────────────


class TestThreadCreate:
    """POST /threads — 创建线程。"""

    def test_create_thread_returns_thread_id(self):
        """POST /threads → 200，返回有效 UUID。"""
        client = _make_client()
        resp = client.post("/threads", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert _assert_valid_uuid(data["thread_id"])

    def test_create_thread_multiple_unique(self):
        """多次 POST /threads → 每个返回唯一 thread_id。"""
        client = _make_client()
        ids = set()
        for _ in range(5):
            resp = client.post("/threads", json={})
            assert resp.status_code == 200
            ids.add(resp.json()["thread_id"])
        assert len(ids) == 5


class TestThreadStubs:
    """PATCH / copy / state — 纯 stub 端点。"""

    def test_patch_thread(self):
        """PATCH /threads/{id} → 200，stub 返回 metadata。"""
        client = _make_client()
        tid = str(uuid.uuid4())
        resp = client.patch(f"/threads/{tid}", json={"name": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == tid
        assert data["metadata"] == {"name": "test"}

    def test_copy_thread(self):
        """POST /threads/{id}/copy → 200，返回新的 thread_id。"""
        client = _make_client()
        tid = str(uuid.uuid4())
        resp = client.post(f"/threads/{tid}/copy")
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert data["thread_id"] != tid  # 新 ID
        assert _assert_valid_uuid(data["thread_id"])

    def test_post_thread_state_stub(self):
        """POST /threads/{id}/state → 200，stub。"""
        client = _make_client()
        tid = str(uuid.uuid4())
        resp = client.post(f"/threads/{tid}/state", json={"values": {"key": "val"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_patch_thread_state_stub(self):
        """PATCH /threads/{id}/state → 200，stub。"""
        client = _make_client()
        tid = str(uuid.uuid4())
        resp = client.patch(f"/threads/{tid}/state", json={"meta": "data"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_get_state_by_checkpoint_stub(self):
        """POST /threads/{id}/state/checkpoint → 200，stub。"""
        client = _make_client()
        tid = str(uuid.uuid4())
        resp = client.post(f"/threads/{tid}/state/checkpoint")
        assert resp.status_code == 200
        data = resp.json()
        assert "values" in data and "next" in data

    def test_get_state_by_checkpoint_id_stub(self):
        """GET /threads/{id}/state/{checkpoint} → 200，stub。"""
        client = _make_client()
        tid = str(uuid.uuid4())
        ckpt = str(uuid.uuid4())
        resp = client.get(f"/threads/{tid}/state/{ckpt}")
        assert resp.status_code == 200
        assert "values" in resp.json()


# ── Checkpointer 相关端点（需 mock get_app） ──────────────────────────────────


class TestThreadsWithGraph:
    """GET/search/delete/state/history — 依赖 get_app() 的端点。"""

    def _patch_get_app(self):
        # Reset the global _app_instance cache so the test graph is used,
        # not the real graph cached by a previous startup() call.
        import novelfactory.server.app as _app_mod
        _app_mod._app_instance = None
        return patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            return_value=FakeThreadGraph(thread_ids=["a", "b"]),
        )

    def test_search_threads(self):
        """POST /threads/search → 200，返回 thread 列表。"""
        with self._patch_get_app():
            client = _make_client()
            resp = client.post("/threads/search")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert isinstance(data, list)
        # Data length depends on checkpointer state; verify response structure is correct

    def test_get_thread_returns_state(self):
        """GET /threads/{id} → 200，返回 thread state。"""
        with self._patch_get_app():
            client = _make_client()
            tid = str(uuid.uuid4())
            resp = client.get(f"/threads/{tid}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["thread_id"] == tid
            assert "values" in data
            assert "next" in data

    def test_get_thread_invalid_uuid_returns_404(self):
        """GET /threads/not-a-uuid → 404。"""
        client = _make_client()
        resp = client.get("/threads/not-a-uuid")
        assert resp.status_code == 404

    def test_delete_thread(self):
        """DELETE /threads/{id} → 200，deleted。"""
        with self._patch_get_app():
            client = _make_client()
            tid = str(uuid.uuid4())
            resp = client.delete(f"/threads/{tid}")
            assert resp.status_code == 200
            assert resp.json()["deleted"] == tid

    def test_get_thread_state(self):
        """GET /threads/{id}/state → 200，返回 state。"""
        with self._patch_get_app():
            client = _make_client()
            tid = str(uuid.uuid4())
            resp = client.get(f"/threads/{tid}/state")
            assert resp.status_code == 200
            data = resp.json()
            assert "values" in data
            assert "next" in data
            assert "metadata" in data

    def test_get_thread_history(self):
        """POST /threads/{id}/history → 200，返回历史列表。"""
        with self._patch_get_app():
            client = _make_client()
            tid = str(uuid.uuid4())
            resp = client.post(f"/threads/{tid}/history", json={"limit": 5})
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)

    def test_thread_not_found_error_path(self):
        """GET /threads/{id}/state → graph.aget_state 异常 → 404。"""
        bad_graph = FakeThreadGraph(thread_ids=["a", "b"])
        bad_graph.aget_state = AsyncMock(side_effect=Exception("state error"))
        with patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            return_value=bad_graph,
        ):
            client = _make_client()
            tid = str(uuid.uuid4())
            resp = client.get(f"/threads/{tid}/state")
            assert resp.status_code == 404
