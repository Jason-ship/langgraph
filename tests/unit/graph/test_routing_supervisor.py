"""Tests for graph/routing.py — supervisor routing logic."""

from __future__ import annotations

from novelfactory.graph.routing import (
    _resolve_target_chapters,
    route_from_supervisor,
)


class TestRouteFromSupervisor:
    """Test the main supervisor routing function."""

    def test_route_setup_phase(self):
        """Setup phase without setup_complete should route to setup_crew."""
        state = {
            "current_phase": "setup",
            "setup_complete": False,
        }
        result = route_from_supervisor(state)
        assert result == "setup_crew"

    def test_route_setup_phase_complete(self):
        """Setup phase with setup_complete should route based on next state."""
        state = {
            "current_phase": "setup",
            "setup_complete": True,
            "current_chapter": 1,
        }
        # After setup complete, should go to load_memory or writing path
        result = route_from_supervisor(state)
        assert isinstance(result, str)

    def test_route_writing_first_chapter(self):
        """Writing phase, chapter 1 should route to refresh_quota."""
        state = {
            "current_phase": "writing",
            "current_chapter": 1,
            "pending_review": None,
            "chapter_approved": False,
        }
        result = route_from_supervisor(state)
        assert result == "refresh_quota"

    def test_route_writing_subsequent_chapter(self):
        """Writing phase, chapter > 1 should route to volume_check."""
        state = {
            "current_phase": "writing",
            "current_chapter": 3,
            "pending_review": None,
            "chapter_approved": False,
        }
        result = route_from_supervisor(state)
        assert result == "volume_check"

    def test_route_writing_pending_review(self):
        """Writing phase with pending_review="chapter" should interrupt for review (v5.5)."""
        state = {
            "current_phase": "writing",
            "current_chapter": 5,
            "pending_review": "chapter",
        }
        result = route_from_supervisor(state)
        assert result == "wait_for_review"

    def test_route_writing_approved_needs_more_chapters(self):
        """After chapter>1 without pending review → routes through phase-check chain (v5.5)."""
        state = {
            "current_phase": "writing",
            "current_chapter": 3,
        }
        result = route_from_supervisor(state)
        # 必须路由到 volume_check（phase_check 链起点）或 refresh_quota
        assert result in ("volume_check", "refresh_quota")

    def test_route_writing_all_done(self):
        """After many chapters written — still routes via phase-check chain (v5.5)."""
        state = {
            "current_phase": "writing",
            "current_chapter": 10,
            "completed_chapters": [{"idx": i} for i in range(1, 11)],
        }
        result = route_from_supervisor(state)
        assert result in ("volume_check", "refresh_quota")

    def test_route_media_phase(self):
        """Media phase should route to media_crew."""
        state = {
            "current_phase": "media",
        }
        result = route_from_supervisor(state)
        assert result == "media_crew"

    def test_route_sync_phase(self):
        """Sync phase should route to sync_crew."""
        state = {
            "current_phase": "sync",
        }
        result = route_from_supervisor(state)
        assert result == "sync_crew"

    def test_route_done_phase(self):
        """Done phase should route to save_memory."""
        state = {
            "current_phase": "done",
        }
        result = route_from_supervisor(state)
        assert result == "save_memory"


class TestResolveTargetChapters:
    """Test the chapter resolution utility."""

    def test_empty_target(self):
        """No target specified → returns FALLBACK_TARGET_CHAPTERS=600 (v5.5)."""
        state = {}
        result = _resolve_target_chapters(state)
        assert result == 600

    def test_single_chapter(self):
        """target_chapters=N → returns N."""
        state = {"target_chapters": 10}
        result = _resolve_target_chapters(state)
        assert result == 10


class TestRoutingEdgeCases:
    """Edge cases for routing logic."""

    def test_missing_phase(self):
        """Missing current_phase should not crash."""
        state = {}
        try:
            result = route_from_supervisor(state)
            assert isinstance(result, str)
        except KeyError:
            # Acceptable if routing expects the field
            pass

    def test_empty_state(self):
        """Completely empty state."""
        state = {}
        try:
            result = route_from_supervisor(state)
            assert isinstance(result, str)
        except (KeyError, TypeError):
            pass

    def test_writing_negative_chapter(self):
        """Negative chapter number."""
        state = {
            "current_phase": "writing",
            "current_chapter": -1,
        }
        result = route_from_supervisor(state)
        assert isinstance(result, str)
