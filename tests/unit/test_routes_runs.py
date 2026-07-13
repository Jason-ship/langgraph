"""Runs 路由单元测试 — 关键端点（batch/list/get/delete/cancel/stream/wait）。

使用 FastAPI TestClient，mock get_app() 返回 fake graph 对象。
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import AsyncMock, patch

import pytest

# ── 确保源码路径可导入 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


# ── Fake Graph（带 astream_events / ainvoke / aget_state） ─────────────────

class _FakeState:
    """Fake aget_state 返回值 — next 为空表示线程已完成（无中断）。"""
    def __init__(self):
        self.next = ()
        self.tasks = []
        self.values = {}


class FakeGraph:
    """Fake CompiledStateGraph — 支持 ainvoke、astream_events、aget_state、aget_state_history。"""

    async def ainvoke(self, input_data, config=None, **kwargs):
        return {"status": "completed", "output": "test output"}

    async def astream_events(self, input_data, config=None, version="v2", **kwargs):
        """返回少量 SSE 事件模拟 stream。"""
        yield {"type": "values", "data": {"messages": [{"content": "hello", "type": "ai"}]}}
        yield {"type": "end", "data": {}}
        raise StopAsyncIteration()

    async def aget_state(self, config):
        return _FakeState()

    async def aget_state_history(self, config, limit=10):
        """用于 /runs/{id}/stream 的 stream_run_history。"""
        yield _FakeState()
        raise StopAsyncIteration()

    def getattr(self, name, default=None):
        return getattr(self, name, default)


# ── 导入 app ──

from fastapi.testclient import TestClient

from novelfactory.server.app import app

# ── Helpers ────────────────────────────────────────────────────────────────────


def _create_fake_thread(client: TestClient) -> str:
    """创建虚拟线程并返回 thread_id。"""
    resp = client.post("/threads", json={})
    assert resp.status_code == 200
    return resp.json()["thread_id"]


# ── 纯 Stub 端点（无需 mock get_app） ─────────────────────────────────────────


class TestRunsBatch:
    """POST /runs/batch — 批量创建 run（纯 stub）。"""

    def test_batch_returns_list(self):
        client = TestClient(app)
        payload = [
            {"input": {"action": "start"}, "stream": False},
            {"input": {"action": "continue"}, "stream": False},
        ]
        resp = client.post("/runs/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        for item in data:
            assert "run_id" in item
            assert item["status"] == "completed"


class TestRunsCRUD:
    """GET / DELETE / POST cancel — 纯 stub 端点。"""

    def test_list_runs_returns_list(self):
        """GET /threads/{id}/runs → 200, 返回 list。"""
        client = TestClient(app)
        thread_id = _create_fake_thread(client)
        resp = client.get(f"/threads/{thread_id}/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_run_returns_details(self):
        """GET /threads/{id}/runs/{run_id} → 200, 返回 run 详情。"""
        client = TestClient(app)
        thread_id = _create_fake_thread(client)
        run_id = str(uuid.uuid4())
        resp = client.get(f"/threads/{thread_id}/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == "completed"

    def test_delete_run_returns_deleted(self):
        """DELETE /threads/{id}/runs/{run_id} → 200, deleted。"""
        client = TestClient(app)
        thread_id = _create_fake_thread(client)
        run_id = str(uuid.uuid4())
        resp = client.delete(f"/threads/{thread_id}/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == run_id

    def test_cancel_run_returns_cancelled(self):
        """POST /threads/{id}/runs/{run_id}/cancel → 200, cancelled。"""
        client = TestClient(app)
        thread_id = _create_fake_thread(client)
        run_id = str(uuid.uuid4())
        resp = client.post(f"/threads/{thread_id}/runs/{run_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cancelled"] == run_id

    def test_join_run_returns_completed(self):
        """GET /threads/{id}/runs/{run_id}/join → 200, status=completed。"""
        client = TestClient(app)
        thread_id = _create_fake_thread(client)
        run_id = str(uuid.uuid4())
        resp = client.get(f"/threads/{thread_id}/runs/{run_id}/join")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"


# ── Run 创建端点（需 mock get_app） ───────────────────────────────────────────


class TestRunCreate:
    """POST /threads/{id}/runs 系列 — 创建 run（需 fake graph）。"""

    def _patch_get_app(self):
        """返回一个 mock get_app 的 patch 上下文。"""
        return patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            return_value=FakeGraph(),
        )

    def test_create_sync_run_returns_result(self):
        """POST /threads/{id}/runs (stream=False) → 200, 返回 run_id + result。"""
        with self._patch_get_app():
            client = TestClient(app)
            thread_id = _create_fake_thread(client)
            resp = client.post(
                f"/threads/{thread_id}/runs",
                json={"input": {"action": "ping"}, "stream": False},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "run_id" in data
            assert data["thread_id"] == thread_id
            assert data["status"] == "completed"
            assert "result" in data

    def test_create_streaming_run_returns_sse(self):
        """POST /threads/{id}/runs (stream=True) → 200, Content-Type=text/event-stream。"""
        with self._patch_get_app():
            client = TestClient(app)
            thread_id = _create_fake_thread(client)
            resp = client.post(
                f"/threads/{thread_id}/runs",
                json={"input": {"action": "ping"}, "stream": True},
            )
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type

    @pytest.mark.skip(reason="TestClient event loop 与 EventSourceResponse 不兼容")
    def test_create_run_stream_no_thread(self):
        """POST /runs/stream (无 thread) → 自动创建 thread 并 stream。"""
        with self._patch_get_app():
            client = TestClient(app)
            resp = client.post(
                "/runs/stream",
                json={"input": {"action": "start"}, "stream": True},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_create_run_wait_no_thread(self):
        """POST /runs/wait (无 thread) → 自动创建 thread 并返回 result。"""
        with self._patch_get_app():
            client = TestClient(app)
            resp = client.post(
                "/runs/wait",
                json={"input": {"action": "start"}, "stream": False},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"

    def test_create_run_wait_with_thread(self):
        """POST /threads/{id}/runs/wait → stream=False 自动设置。"""
        with self._patch_get_app():
            client = TestClient(app)
            thread_id = _create_fake_thread(client)
            resp = client.post(
                f"/threads/{thread_id}/runs/wait",
                json={"input": {"action": "ping"}},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"

    @pytest.mark.skip(reason="TestClient event loop 与 EventSourceResponse 不兼容")
    def test_stream_run_history(self):
        """GET /threads/{id}/runs/{run_id}/stream → SSE 历史流。"""
        with self._patch_get_app():
            client = TestClient(app)
            thread_id = _create_fake_thread(client)
            run_id = str(uuid.uuid4())
            resp = client.get(f"/threads/{thread_id}/runs/{run_id}/stream")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_run_invoke_failure_returns_500(self):
        """POST /threads/{id}/runs (non-stream) → graph.ainvoke 失败返回 500。"""
        bad_graph = FakeGraph()
        bad_graph.ainvoke = AsyncMock(side_effect=ValueError("invoke error"))
        with patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            return_value=bad_graph,
        ):
            client = TestClient(app)
            thread_id = _create_fake_thread(client)
            resp = client.post(
                f"/threads/{thread_id}/runs",
                json={"input": {"action": "fail"}, "stream": False},
            )
            assert resp.status_code == 500
