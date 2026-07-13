"""Tests for StreamStateTracker SSE progress events."""

from novelfactory.server.streaming import StreamStateTracker


def test_tracker_initial_state():
    tracker = StreamStateTracker()
    assert tracker.phase == ""
    assert tracker.current_chapter == 0
    assert tracker.current_agent == ""
    assert tracker.agent_status == "pending"


def test_tracker_update_phase():
    tracker = StreamStateTracker()
    changed = tracker.update_from_state({"current_phase": "writing", "current_chapter": 1})
    assert changed
    assert tracker.phase == "writing"
    assert tracker.current_chapter == 1


def test_tracker_no_change():
    tracker = StreamStateTracker()
    tracker.update_from_state({"current_phase": "writing", "current_chapter": 1})
    changed = tracker.update_from_state({"current_phase": "writing", "current_chapter": 1})
    assert not changed


def test_tracker_update_agent():
    tracker = StreamStateTracker()
    changed = tracker.update_from_state({"next": ["writing_crew"]})
    assert changed
    assert tracker.current_agent == "writing_crew"
    assert tracker.agent_status == "in_progress"


def test_tracker_progress_event_format():
    tracker = StreamStateTracker()
    tracker.update_from_state({"current_phase": "writing", "current_chapter": 3})
    ev = tracker.to_progress_event()
    assert ev["phase"] == "writing"
    assert ev["chapter"] == 3
    assert "agent" in ev
    assert "agent_status" in ev


def test_tracker_chapter_update_only():
    tracker = StreamStateTracker()
    tracker.update_from_state({"current_phase": "writing"})
    changed = tracker.update_from_state({"current_chapter": 5})
    assert changed
    assert tracker.current_chapter == 5


def test_tracker_empty_state():
    tracker = StreamStateTracker()
    changed = tracker.update_from_state({})
    assert not changed
