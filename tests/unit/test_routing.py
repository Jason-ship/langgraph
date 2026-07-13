"""P0: route_from_supervisor + main_supervisor_node routing (10+ branches).

Tests every phase transition and routing decision in the root supervisor.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_state

# ── route_from_supervisor — 5 phases, 10+ branches ────────────────────────────

class TestRouteFromSupervisor:
    """P0: route_from_supervisor — every phase → target node mapping."""

    @pytest.fixture(scope="module")
    def router(self):
        from novelfactory.graph.routing import route_from_supervisor
        return route_from_supervisor

    # Setup phase
    def test_setup_not_complete_routes_to_setup_crew(self, router):
        state = make_state(current_phase="setup", setup_complete=False)
        assert router(state) == "setup_crew"

    def test_setup_complete_kickoff_review_routes_to_wait(self, router):
        state = make_state(current_phase="setup", setup_complete=True,
                           pending_review="kickoff")
        assert router(state) == "wait_for_review"

    def test_setup_complete_no_review_routes_to_load_memory(self, router):
        state = make_state(current_phase="setup", setup_complete=True)
        assert router(state) == "load_memory"

    # Writing phase
    def test_writing_needs_guidance_routes_to_human_guidance(self, router):
        state = make_state(current_phase="writing", chapter_needs_guidance=True,
                           guidance_complete=False)
        assert router(state) == "chapter_human_guidance"

    def test_writing_pending_chapter_review_routes_to_wait(self, router):
        state = make_state(current_phase="writing", pending_review="chapter")
        assert router(state) == "wait_for_review"

    def test_writing_chapter_above_1_routes_to_volume_check(self, router):
        state = make_state(current_phase="writing", current_chapter=5)
        assert router(state) == "volume_check"

    def test_writing_first_chapter_routes_to_refresh_quota(self, router):
        state = make_state(current_phase="writing", current_chapter=1)
        assert router(state) == "refresh_quota"

    # Media phase
    def test_media_routes_to_media_crew(self, router):
        state = make_state(current_phase="media", media_complete=True)
        assert router(state) == "media_crew"

    # Sync phase
    def test_sync_routes_to_sync_crew(self, router):
        state = make_state(current_phase="sync")
        assert router(state) == "sync_crew"

    # Done phase
    def test_done_routes_to_save_memory(self, router):
        state = make_state(current_phase="done")
        assert router(state) == "save_memory"


# ── main_supervisor_node — phase machine with phase transitions ────────────────

class TestMainSupervisorNode:
    """P0: main_supervisor_node — phase transition logic."""

    @pytest.fixture(scope="module")
    def supervisor(self):
        from novelfactory.graph.nodes.supervisor import main_supervisor_node
        return main_supervisor_node

    # Setup phase transitions
    def test_setup_not_complete_stays_setup(self, supervisor):
        state = make_state(current_phase="setup", setup_complete=False)
        result = supervisor(state)
        assert result.get("current_phase") == "setup"

    def test_setup_complete_kickoff_review_unchanged(self, supervisor):
        state = make_state(current_phase="setup", setup_complete=True,
                           pending_review="kickoff")
        result = supervisor(state)
        assert result.get("current_phase") is None  # no phase change

    def test_setup_complete_transitions_to_writing(self, supervisor):
        state = make_state(current_phase="setup", setup_complete=True)
        result = supervisor(state)
        assert result.get("current_phase") == "writing"

    # Writing phase transitions
    def test_writing_no_review_no_approval_sets_sync(self, supervisor):
        state = make_state(current_phase="writing", current_chapter=5,
                           target_chapters=100)
        result = supervisor(state)
        # All chapters not done → still transitions to sync
        assert result.get("current_phase") == "sync"

    def test_writing_chapter_approved_sets_sync(self, supervisor):
        state = make_state(current_phase="writing", chapter_approved=True)
        result = supervisor(state)
        assert result.get("current_phase") == "sync"

    def test_writing_pending_review_unchanged(self, supervisor):
        state = make_state(current_phase="writing", pending_review="chapter")
        result = supervisor(state)
        assert "current_phase" not in result  # no change

    # Media phase
    def test_media_transitions_to_sync(self, supervisor):
        state = make_state(current_phase="media", media_complete=True)
        result = supervisor(state)
        assert result.get("current_phase") == "sync"

    # Sync phase
    def test_sync_all_done_transitions_to_done(self, supervisor):
        state = make_state(current_phase="sync", current_chapter=100,
                           target_chapters=100)
        result = supervisor(state)
        assert result.get("current_phase") == "done"

    def test_sync_more_chapters_transitions_to_writing(self, supervisor):
        state = make_state(current_phase="sync", current_chapter=50,
                           target_chapters=100)
        result = supervisor(state)
        assert result.get("current_phase") == "writing"

    # Done phase
    def test_done_all_complete_stays_done(self, supervisor):
        state = make_state(current_phase="done", current_chapter=101,
                           target_chapters=100)
        result = supervisor(state)
        assert "current_phase" not in result  # no change

    def test_done_more_chapters_resumes(self, supervisor):
        state = make_state(current_phase="done", current_chapter=80,
                           target_chapters=100)
        result = supervisor(state)
        assert result.get("current_phase") == "sync"

    # Phase transition messages
    def test_phase_transition_emits_message(self, supervisor):
        state = make_state(current_phase="setup", setup_complete=True,
                           _last_supervisor_phase="")
        result = supervisor(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "写作" in result["messages"][0].content

    def test_same_phase_no_message(self, supervisor):
        state = make_state(current_phase="setup", setup_complete=False,
                           _last_supervisor_phase="setup")
        result = supervisor(state)
        messages = result.get("messages", [])
        assert len(messages) == 0
