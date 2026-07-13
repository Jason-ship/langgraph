"""Assistants 路由单元测试 — Assistant 管理（list/get/create/update/delete/graph/schemas）。

使用 FastAPI TestClient，mock get_app() 返回 fake graph（仅 /graph 端点需要）。
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

# ── 确保源码路径可导入 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


# ── Fake Graph（仅用于 /assistants/{id}/graph 端点） ─────────────────────────

class _FakeDrawableGraph:
    """Fake graph.get_graph() 返回值 — 带 to_json。"""
    def to_json(self):
        return {"nodes": [{"id": "root"}], "edges": []}


class FakeAssistantGraph:
    """Fake CompiledStateGraph — 仅带 get_graph 方法。"""
    def get_graph(self, xray=0):
        return _FakeDrawableGraph()


# ── 导入 app ──

from fastapi.testclient import TestClient

from novelfactory.server.app import app

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_client() -> TestClient:
    return TestClient(app)


# ── 纯 Stub 端点（无需 mock get_app） ─────────────────────────────────────────


class TestListAssistants:
    """GET /assistants — 列出 Assistant。"""

    def test_list_assistants(self):
        """GET /assistants → 200，返回 novelfactory assistant。"""
        client = _make_client()
        resp = client.get("/assistants")
        assert resp.status_code == 200
        data = resp.json()
        assert "assistants" in data
        assert len(data["assistants"]) >= 1
        ast = data["assistants"][0]
        assert ast["assistant_id"] == "novelfactory"
        assert ast["name"] == "NovelFactory"

    def test_search_assistants(self):
        """POST /assistants/search → 200，返回与 list 一致的数据。"""
        client = _make_client()
        resp = client.post("/assistants/search")
        assert resp.status_code == 200
        data = resp.json()
        assert "assistants" in data


class TestGetAssistant:
    """GET /assistants/{id} — 获取指定 Assistant。"""

    def test_get_novelfactory(self):
        """GET /assistants/novelfactory → 200。"""
        client = _make_client()
        resp = client.get("/assistants/novelfactory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "novelfactory"

    def test_get_nonexistent_returns_404(self):
        """GET /assistants/nonexistent → 404。"""
        client = _make_client()
        resp = client.get("/assistants/nonexistent")
        assert resp.status_code == 404


class TestAssistantCRUD:
    """POST / PATCH / DELETE — CRUD 操作。"""

    def test_create_assistant(self):
        """POST /assistants → 200，返回 assistant 数据。"""
        client = _make_client()
        resp = client.post("/assistants")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "novelfactory"

    def test_update_assistant(self):
        """PATCH /assistants/novelfactory → 200。"""
        client = _make_client()
        resp = client.patch("/assistants/novelfactory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "novelfactory"

    def test_update_nonexistent_returns_404(self):
        """PATCH /assistants/nonexistent → 404。"""
        client = _make_client()
        resp = client.patch("/assistants/nonexistent")
        assert resp.status_code == 404

    def test_delete_assistant(self):
        """DELETE /assistants/{id} → 200, deleted。"""
        client = _make_client()
        resp = client.delete("/assistants/novelfactory")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "novelfactory"


class TestAssistantMeta:
    """GET schemas / subgraphs / versions — 元数据端点。"""

    def test_get_schemas(self):
        """GET /assistants/{id}/schemas → 200。"""
        client = _make_client()
        resp = client.get("/assistants/novelfactory/schemas")
        assert resp.status_code == 200
        data = resp.json()
        assert "graph" in data
        assert "config" in data

    def test_get_subgraphs(self):
        """GET /assistants/{id}/subgraphs → 200，返回 list。"""
        client = _make_client()
        resp = client.get("/assistants/novelfactory/subgraphs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_subgraph_by_namespace(self):
        """GET /assistants/{id}/subgraphs/{ns} → 200。"""
        client = _make_client()
        resp = client.get("/assistants/novelfactory/subgraphs/writing_crew")
        assert resp.status_code == 200

    def test_list_versions(self):
        """POST /assistants/{id}/versions → 200，返回 list。"""
        client = _make_client()
        resp = client.post("/assistants/novelfactory/versions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_set_latest(self):
        """POST /assistants/{id}/latest → 200，返回 assistant。"""
        client = _make_client()
        resp = client.post("/assistants/novelfactory/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "novelfactory"


# ── /graph 端点（需 mock get_app） ────────────────────────────────────────────


class TestAssistantGraph:
    """GET /assistants/{id}/graph — 获取图结构（需 fake graph）。"""

    def _patch_get_app(self):
        return patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            return_value=FakeAssistantGraph(),
        )

    def test_get_graph_returns_structure(self):
        """GET /assistants/novelfactory/graph → 200，返回 nodes + edges。"""
        with self._patch_get_app():
            client = _make_client()
            resp = client.get("/assistants/novelfactory/graph")
            assert resp.status_code == 200
            data = resp.json()
            assert "nodes" in data
            assert "edges" in data

    def test_get_graph_for_nonexistent_assistant(self):
        """GET /assistants/nonexistent/graph → 404。"""
        client = _make_client()
        resp = client.get("/assistants/nonexistent/graph")
        assert resp.status_code == 404
