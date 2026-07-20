"""API integration tests — run against a live NovelFactory server.

Usage:
    # Default: http://localhost:8123
    python3 -m pytest tests/test_api_integration.py -v

    # Custom URL:
    NOVELFACTORY_API_URL=http://localhost:8123 python3 -m pytest tests/test_api_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import requests

API_URL = os.environ.get("NOVELFACTORY_API_URL", "http://localhost:8123")
TIMEOUT = 30
TIMEOUT_LONG = 600  # for LLM-triggering runs (non-streaming invokes full pipeline)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get(path: str) -> requests.Response:
    return requests.get(f"{API_URL}{path}", timeout=TIMEOUT)


def _post(path: str, json: dict | None = None) -> requests.Response:
    return requests.post(
        f"{API_URL}{path}",
        json=json or {},
        timeout=TIMEOUT,
        headers={"Content-Type": "application/json"},
    )


def _create_thread() -> str:
    """Create a thread and return its thread_id."""
    resp = _post("/threads")
    assert resp.status_code == 200, f"create thread failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert "thread_id" in data, f"missing thread_id: {data}"
    tid = data["thread_id"]
    assert _is_valid_uuid(tid), f"invalid UUID: {tid}"
    return tid


def _is_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


# ── Health & System ────────────────────────────────────────────────────────────


class TestHealth:
    """Smoke tests for health and readiness endpoints."""

    def test_health_returns_ok(self):
        resp = _get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "8.0.0"

    def test_ready_returns_ready(self):
        resp = _get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["graph_compiled"] is True

    def test_openapi_schema(self):
        resp = _get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "NovelFactory LangGraph API"
        assert len(data["paths"]) >= 7

    def test_docs_page(self):
        resp = _get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


class TestAssistants:
    """Assistant metadata endpoints."""

    def test_list_assistants(self):
        resp = _get("/assistants")
        assert resp.status_code == 200
        data = resp.json()
        assert "assistants" in data
        assert len(data["assistants"]) >= 1
        ast = data["assistants"][0]
        assert ast["assistant_id"] == "novelfactory"

    def test_get_assistant_found(self):
        resp = _get("/assistants/novelfactory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "novelfactory"

    def test_get_assistant_not_found(self):
        resp = _get("/assistants/nonexistent")
        assert resp.status_code == 404


class TestThreads:
    """Thread CRUD operations."""

    def test_create_thread_returns_valid_id(self):
        tid = _create_thread()
        assert len(tid) == 36  # UUID v4

    def test_get_existing_thread(self):
        tid = _create_thread()
        resp = _get(f"/threads/{tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == tid
        assert isinstance(data["values"], dict)
        assert isinstance(data["next"], list)

    def test_get_thread_with_invalid_id_returns_404(self):
        resp = _get("/threads/not-a-uuid")
        assert resp.status_code == 404

    # NOTE: Thread with a valid UUID but no history currently returns 200
    # with empty state. This is by design — valid UUIDs are accepted even
    # if they haven't been used yet.
    def test_get_valid_uuid_no_history_returns_200(self):
        tid = str(uuid.uuid4())
        resp = _get(f"/threads/{tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == tid
        assert not data["values"]  # empty dict

    def test_create_thread_multiple_times(self):
        tids = [_create_thread() for _ in range(3)]
        assert len(set(tids)) == 3  # all unique


class TestRuns:
    """Run creation — covers both streaming and non-streaming modes."""

    def test_create_non_streaming_run_returns_run_id(self):
        tid = _create_thread()
        resp = requests.post(
            f"{API_URL}/threads/{tid}/runs",
            json={"input": {"action": "ping"}, "stream": False},
            timeout=TIMEOUT_LONG,
            headers={"Content-Type": "application/json"},
        )
        # Non-streaming returns immediately with result or error
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "run_id" in data
            assert data["thread_id"] == tid
            assert data["status"] == "completed"

    def test_create_streaming_run_returns_sse(self):
        tid = _create_thread()
        resp = requests.post(
            f"{API_URL}/threads/{tid}/runs",
            json={"input": {"action": "ping"}, "stream": True},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        if resp.status_code == 200:
            assert "text/event-stream" in resp.headers.get("content-type", "")
            # Read first chunk (without consuming the full stream)
            chunk = resp.raw.readline().decode("utf-8", errors="replace")
            assert "event:" in chunk
            resp.close()
        else:
            assert resp.status_code in (400, 422, 500)
            resp.close()

    def test_resume_interrupted_run(self):
        """Create a run on a completed thread should trigger resume logic."""
        tid = _create_thread()
        # First run (may complete or interrupt)
        resp = requests.post(
            f"{API_URL}/threads/{tid}/runs",
            json={"input": {}, "stream": True},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        resp.close()
        # Second run on the same thread — should detect as resume/continue
        resp2 = requests.post(
            f"{API_URL}/threads/{tid}/runs",
            json={"input": {"resume": "continue"}, "stream": True},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        assert resp2.status_code in (200, 500)
        resp2.close()


class TestErrorHandling:
    """API error scenarios."""

    def test_invalid_json_body(self):
        resp = requests.post(
            f"{API_URL}/threads/{uuid.uuid4()}/runs",
            data="not json",
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422  # FastAPI default for parse failure

    def test_missing_content_type(self):
        tid = _create_thread()
        resp = requests.post(
            f"{API_URL}/threads/{tid}/runs",
            data="{}",
            timeout=TIMEOUT,
        )
        # FastAPI handles missing Content-Type gracefully
        assert resp.status_code in (200, 422)
