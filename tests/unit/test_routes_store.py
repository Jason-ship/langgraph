"""Store 路由单元测试 — 持久化 Key-Value 存储（put/get/delete/search/namespaces）。

使用 FastAPI TestClient，mock get_app() 返回带 AsyncPostgresStore 的 fake graph。
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

# ── 确保源码路径可导入 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


# ── Fake Store ──────────────────────────────────────────────────────────────────

class _FakeStoreItem:
    """Fake store.aget / store.asearch 返回的 item。"""
    def __init__(self, namespace, key, value):
        self.namespace = namespace
        self.key = key
        self.value = value


class FakeStore:
    """Fake AsyncPostgresStore — 支持 aput / aget / adelete / asearch / alist_namespaces。"""

    def __init__(self):
        self._data: dict[str, dict] = {}

    def _make_key(self, namespace, key):
        ns_key = "/".join(namespace) if isinstance(namespace, (list, tuple)) else str(namespace)
        return f"{ns_key}:{key}"

    async def aput(self, namespace, key, value):
        self._data[self._make_key(namespace, key)] = value

    async def aget(self, namespace, key):
        val = self._data.get(self._make_key(namespace, key))
        if val is not None:
            return _FakeStoreItem(namespace, key, val)
        return None

    async def adelete(self, namespace, key):
        self._data.pop(self._make_key(namespace, key), None)

    async def asearch(self, namespace_prefix, limit=10, offset=0):
        ns_prefix = "/".join(namespace_prefix) if namespace_prefix else ""
        items = []
        for k, v in self._data.items():
            if k.startswith(ns_prefix):
                parts = k.rsplit(":", 1)
                items.append(_FakeStoreItem(list(namespace_prefix), parts[1] if len(parts) > 1 else k, v))
        return items[offset:offset + limit]

    async def alist_namespaces(self):
        return [["test_ns"]]


class FakeStoreGraph:
    """Fake CompiledStateGraph — 带 store 属性。"""

    def __init__(self):
        self.store = FakeStore()


# ── 导入 app ──

from fastapi.testclient import TestClient

from novelfactory.server.app import app

# ── Helpers ────────────────────────────────────────────────────────────────────

def _patch_store():
    """Mock get_app 返回带 FakeStore 的 graph。"""
    return patch(
        "novelfactory.server.app.get_app",
        new_callable=AsyncMock,
        return_value=FakeStoreGraph(),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestStorePutAndGet:
    """PUT /store/items + GET /store/items — 写入与读取。"""

    def test_put_item_returns_ok(self):
        """PUT /store/items → 200, status=ok。"""
        with _patch_store():
            client = TestClient(app)
            resp = client.put("/store/items", json={
                "namespace": ["test", "v1"],
                "key": "config",
                "value": {"theme": "dark"},
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_get_item_after_put(self):
        """PUT → GET 可正确读取写入的值。"""
        with _patch_store():
            client = TestClient(app)
            client.put("/store/items", json={
                "namespace": ["test", "v1"],
                "key": "settings",
                "value": {"lang": "zh"},
            })
            resp = client.get("/store/items", params={
                "namespace": ["test", "v1"],
                "key": "settings",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["namespace"] == ["test", "v1"]
            assert data["key"] == "settings"
            assert data["value"] == {"lang": "zh"}

    def test_get_nonexistent_item(self):
        """GET /store/items 查询不存在的 key → 返回空 value。"""
        with _patch_store():
            client = TestClient(app)
            resp = client.get("/store/items", params={
                "namespace": ["nonexistent"],
                "key": "missing",
            })
            assert resp.status_code == 200
            assert resp.json()["value"] == {}


class TestStoreDelete:
    """DELETE /store/items — 删除条目。"""

    def test_delete_item_returns_deleted(self):
        """DELETE /store/items → 200, deleted 包含 namespace:key。"""
        with _patch_store():
            client = TestClient(app)
            client.put("/store/items", json={
                "namespace": ["test"],
                "key": "to_delete",
                "value": {"data": 1},
            })
            resp = client.delete("/store/items", params={
                "namespace": ["test"],
                "key": "to_delete",
            })
            assert resp.status_code == 200
            assert "deleted" in resp.json()
            assert "test:to_delete" in resp.json()["deleted"]

    def test_get_after_delete_returns_empty(self):
        """DELETE 后 GET 同 key → 返回空 value。"""
        with _patch_store():
            client = TestClient(app)
            client.put("/store/items", json={
                "namespace": ["temp"],
                "key": "ephemeral",
                "value": {"x": 1},
            })
            client.delete("/store/items", params={
                "namespace": ["temp"],
                "key": "ephemeral",
            })
            resp = client.get("/store/items", params={
                "namespace": ["temp"],
                "key": "ephemeral",
            })
            assert resp.status_code == 200
            assert resp.json()["value"] == {}


class TestStoreSearch:
    """POST /store/items/search — 搜索。"""

    def test_search_returns_list(self):
        """POST /store/items/search → 200，返回 item 列表。"""
        with _patch_store():
            client = TestClient(app)
            client.put("/store/items", json={
                "namespace": ["search", "ns1"],
                "key": "item1",
                "value": {"a": 1},
            })
            client.put("/store/items", json={
                "namespace": ["search", "ns1"],
                "key": "item2",
                "value": {"b": 2},
            })
            resp = client.post("/store/items/search", json={
                "namespace_prefix": ["search"],
                "limit": 10,
                "offset": 0,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)


class TestStoreNamespaces:
    """POST /store/namespaces — 列出命名空间。"""

    def test_list_namespaces_returns_list(self):
        """POST /store/namespaces → 200，返回 namespace 列表。"""
        with _patch_store():
            client = TestClient(app)
            resp = client.post("/store/namespaces")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert ["test_ns"] in data
