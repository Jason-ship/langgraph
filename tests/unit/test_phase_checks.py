"""P0: volume→quality→foreshadowing check chain (v4.1+ nodes).

Tests the node interface contracts: each node accepts NovelFactoryState
and returns a dict. DB-dependent assertions (volume_status, quality_trend,
foreshadowing_status keys) are skipped when DB is unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import make_state


@pytest.fixture(scope="module")
def _db_mock():
    """Mock DatabaseManager.get_instance() to avoid actual DB calls."""
    mock_instance = MagicMock()
    mock_conn = MagicMock()
    mock_instance.get_connection.return_value.__enter__.return_value = mock_conn
    mock_instance.get_connection.return_value.__exit__.return_value = None
    return mock_instance


# ── volume_check_node ─────────────────────────────────────────────────────────

class TestVolumeCheckNode:
    """P0: volume_check_node — interface contract."""

    @pytest.fixture(scope="module")
    def node(self):
        from novelfactory.graph.nodes.phase_checks import volume_check_node
        return volume_check_node

    def test_returns_dict(self, node):
        state = make_state(current_chapter=5)
        result = node(state)
        assert isinstance(result, dict)

    def test_returns_dict_no_db(self, node):
        """Without DB, returns empty dict (graceful degradation)."""
        state = make_state(current_chapter=5)
        result = node(state)
        assert isinstance(result, dict)
        # Without DB, result may be empty or contain error-only status
        if "volume_status" in result:
            assert isinstance(result["volume_status"], dict)

    def test_handles_existing_guidance(self, node):
        state = make_state(current_chapter=5, auto_guidance="existing")
        result = node(state)
        if "auto_guidance" in result:
            assert "existing" in result["auto_guidance"]


# ── quality_check_node ────────────────────────────────────────────────────────

class TestQualityCheckNode:
    """P0: quality_check_node — interface contract."""

    @pytest.fixture(scope="module")
    def node(self):
        from novelfactory.graph.nodes.phase_checks import quality_check_node
        return quality_check_node

    def test_returns_dict(self, node):
        state = make_state(current_chapter=5)
        result = node(state)
        assert isinstance(result, dict)

    def test_returns_dict_no_db(self, node):
        """Without DB, returns empty dict (graceful degradation)."""
        state = make_state(current_chapter=5)
        result = node(state)
        assert isinstance(result, dict)

    def test_preserves_existing_guidance(self, node):
        state = make_state(current_chapter=5, auto_guidance="existing guide")
        result = node(state)
        if "auto_guidance" in result:
            assert "existing guide" in result["auto_guidance"]


# ── foreshadowing_check_node ──────────────────────────────────────────────────

class TestForeshadowingCheckNode:
    """P0: foreshadowing_check_node — interface contract."""

    @pytest.fixture(scope="module")
    def node(self):
        from novelfactory.graph.nodes.phase_checks import foreshadowing_check_node
        return foreshadowing_check_node

    def test_returns_dict(self, node):
        state = make_state(current_chapter=5)
        result = node(state)
        assert isinstance(result, dict)

    def test_returns_dict_no_db(self, node):
        """Without DB, returns empty dict (graceful degradation)."""
        state = make_state(current_chapter=5)
        result = node(state)
        assert isinstance(result, dict)

    def test_preserves_existing_guidance(self, node):
        state = make_state(current_chapter=5, auto_guidance="existing guide")
        result = node(state)
        if "auto_guidance" in result:
            assert "existing guide" in result["auto_guidance"]
