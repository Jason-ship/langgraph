"""Health 路由单元测试 — 4 个端点（/health、/ready、/info、/metrics）。

使用 FastAPI TestClient，mock 外部依赖（tools-proxy、get_app、psutil）。
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── 确保源码路径可导入 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# ── Mock 避免 DB/外部依赖触发 ──
# health.py 中 /health 会调用 _check_tools_proxy()，该函数会 HTTP 请求 tools-proxy。
# /ready 会调用 get_app()，需要编译 graph 并连接 PG。
# /metrics 会 import psutil，CI 环境中可能未安装。
# 以下 mock 在 app 导入前注入，确保路由函数执行时不触发真实的外部调用。

_fake_graph = MagicMock()  # 一个非 None fake graph 对象
_fake_graph.__bool__.return_value = True

# 预注入 psutil 到 sys.modules 避免 /metrics 导入失败
_fake_psutil = MagicMock()
_fake_process = MagicMock()
_fake_process.memory_info.return_value = MagicMock(rss=100_000_000, vms=200_000_000)
_fake_process.cpu_percent.return_value = 5.0
_fake_process.num_fds.return_value = 42
_fake_process.create_time.return_value = 1000000.0
_fake_psutil.Process.return_value = _fake_process
sys.modules["psutil"] = _fake_psutil

from novelfactory.server.app import app

client = pytest.mark.anyio  # 标记异步支持（通过 pytest-asyncio）


class TestHealthEndpoint:
    """GET /health — 健康检查端点。"""

    def test_health_returns_ok(self):
        """GET /health → 200, status="ok", version="6.1.0"。"""
        with patch(
            "novelfactory.server.routes.health._check_tools_proxy",
            new_callable=AsyncMock,
            return_value="ok",
        ):
            from fastapi.testclient import TestClient
            c = TestClient(app)
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["version"] and "." in data["version"]
            assert "tools_proxy" in data

    def test_health_tools_proxy_unreachable(self):
        """GET /health → tools_proxy 不可达时仍返回 200。"""
        with patch(
            "novelfactory.server.routes.health._check_tools_proxy",
            new_callable=AsyncMock,
            return_value="unreachable",
        ):
            from fastapi.testclient import TestClient
            c = TestClient(app)
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["tools_proxy"] == "unreachable"


class TestReadyEndpoint:
    """GET /ready — 就绪检查端点。"""

    def test_ready_returns_ready(self):
        """GET /ready → 200, status="ready", graph_compiled=True。"""
        with patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            return_value=_fake_graph,
        ):
            from fastapi.testclient import TestClient
            c = TestClient(app)
            resp = c.get("/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ready"
            assert data["graph_compiled"] is True
            assert data["version"] and "." in data["version"]

    def test_ready_graph_failure_returns_503(self):
        """GET /ready → graph 编译失败时返回 503。"""
        with patch(
            "novelfactory.server.app.get_app",
            new_callable=AsyncMock,
            side_effect=ValueError("DB connection failed"),
        ):
            from fastapi.testclient import TestClient
            c = TestClient(app)
            resp = c.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert "detail" in data


class TestInfoEndpoint:
    """GET /info — 部署信息端点（纯 stub，无需 mock）。"""

    def test_info_returns_correct_data(self):
        """GET /info → 200, version="6.1.0", assistant_id="novelfactory"。"""
        from fastapi.testclient import TestClient
        c = TestClient(app)
        resp = c.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] and "." in data["version"]
        assert data["assistant_id"] == "novelfactory"
        assert "deployment_type" in data


class TestMetricsEndpoint:
    """GET /metrics — Prometheus 指标端点。"""

    def test_metrics_returns_plain_text(self):
        """GET /metrics → 200, Content-Type 包含 text/plain。"""
        from fastapi.testclient import TestClient
        c = TestClient(app)
        resp = c.get("/metrics")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_contains_prometheus_format(self):
        """GET /metrics → 返回 Prometheus 格式的指标行。"""
        from fastapi.testclient import TestClient
        c = TestClient(app)
        resp = c.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "# HELP" in body
        assert "# TYPE" in body
        assert "novelfactory_info" in body
        assert "novelfactory_memory_rss_bytes" in body
        assert "version=" in body and "novelfactory_info" in body
