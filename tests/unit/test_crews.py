"""P2: Crew orchestration — supervisor factory + review queue."""

from __future__ import annotations

import pytest


class TestCrewSupervisorFactory:
    """P2: create_crew_supervisor — builds a compiled StateGraph."""

    @pytest.fixture(scope="module")
    def mock_agents(self):
        def agent_a(state):
            return {"crew_result": {"output": "a"}}

        def agent_b(state):
            return {"crew_result": {"output": "b"}}

        return {"agent_a": agent_a, "agent_b": agent_b}

    def test_creates_supervisor_graph(self, mock_agents):
        from novelfactory.crews.supervisor import create_crew_supervisor
        from tests.conftest import MockLLM

        model = MockLLM("agent_a")
        graph = create_crew_supervisor(
            crew_name="test_crew",
            agents=mock_agents,
            supervisor_prompt="Choose the best agent.",
            model=model,
        )
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "supervisor" in nodes
        assert "agent_a" in nodes
        assert "agent_b" in nodes

    def test_empty_agents_raises(self):
        from novelfactory.crews.supervisor import create_crew_supervisor
        from tests.conftest import MockLLM

        with pytest.raises(ValueError, match="must have at least one agent"):
            create_crew_supervisor(
                crew_name="empty",
                agents={},
                supervisor_prompt="test",
                model=MockLLM(),
            )

    def test_single_agent_graph(self):
        from novelfactory.crews.supervisor import create_crew_supervisor
        from tests.conftest import MockLLM

        def solo_agent(state):
            return {"crew_result": {}}

        graph = create_crew_supervisor(
            crew_name="solo",
            agents={"solo": solo_agent},
            supervisor_prompt="test",
            model=MockLLM(),
        )
        nodes = set(graph.get_graph().nodes.keys())
        assert "solo" in nodes
        assert "supervisor" in nodes


class TestCrewHandoff:
    """P2: crew_handoff — returns Command with graph=Command.PARENT."""

    def test_handoff_returns_command(self):
        from langgraph.types import Command

        from novelfactory.crews.supervisor import crew_handoff

        cmd = crew_handoff(
            goto="writing_crew",
            crew_name="setup",
            updates={"setup_complete": True},
        )
        assert isinstance(cmd, Command)
        assert cmd.goto == "writing_crew"
        assert cmd.graph == Command.PARENT
        assert cmd.update["crew_result"]["setup_complete"] is True


class TestReviewQueue:
    """P2: UnifiedReviewQueue — add/decide/get_pending lifecycle."""

    @pytest.fixture
    def queue(self):
        import os
        import tempfile

        from novelfactory.crews.review_queue import UnifiedReviewQueue
        path = os.path.join(tempfile.mkdtemp(), "test_queue.json")
        return UnifiedReviewQueue(db_path=path)

    def test_add_review_item(self, queue):
        from novelfactory.crews.review_queue import ReviewItem
        item = ReviewItem(
            thread_id="tid-1",
            review_type="chapter",
            project_name="test",
            current_chapter=3,
            content_summary="summary",
        )
        result = queue.add(item)
        assert result.thread_id == "tid-1"
        assert result.status.value == "pending"

    def test_get_pending_returns_item(self, queue):
        from novelfactory.crews.review_queue import ReviewItem
        queue.add(ReviewItem(
            thread_id="tid-2", review_type="chapter",
            project_name="test", current_chapter=5,
            content_summary="pending summary",
        ))
        pending = queue.get_pending()
        assert len(pending) >= 1
        assert any(p.thread_id == "tid-2" for p in pending)

    def test_get_pending_by_thread_id(self, queue):
        from novelfactory.crews.review_queue import ReviewItem
        queue.add(ReviewItem(
            thread_id="tid-a", review_type="chapter",
            project_name="test", current_chapter=1,
            content_summary="a",
        ))
        queue.add(ReviewItem(
            thread_id="tid-b", review_type="chapter",
            project_name="test", current_chapter=2,
            content_summary="b",
        ))
        pending_a = queue.get_pending(thread_id="tid-a")
        assert len(pending_a) == 1
        assert pending_a[0].thread_id == "tid-a"

    def test_decide_moves_to_completed(self, queue):
        from novelfactory.crews.review_queue import ReviewItem
        queue.add(ReviewItem(
            thread_id="tid-3", review_type="chapter",
            project_name="test", current_chapter=10,
            content_summary="decide test",
        ))
        result = queue.decide(thread_id="tid-3", decision="approve")
        assert result is True
        pending = queue.get_pending(thread_id="tid-3")
        assert len(pending) == 0
        completed = queue.get_completed()
        assert any(c.thread_id == "tid-3" for c in completed)

    def test_get_decision_sync(self, queue):
        from novelfactory.crews.review_queue import ReviewItem
        queue.add(ReviewItem(
            thread_id="tid-4", review_type="kickoff",
            project_name="test", current_chapter=0,
            content_summary="kickoff",
        ))
        queue.decide(thread_id="tid-4", decision="reject", comment="not good")
        decision = queue.get_decision_sync("tid-4")
        assert decision is not None
        assert decision["decision"] == "reject"
        assert decision["comment"] == "not good"

    def test_get_completed_respects_limit(self, queue):
        completed = queue.get_completed(limit=10)
        assert isinstance(completed, list)

    def test_clear_completed(self, queue):
        from novelfactory.crews.review_queue import ReviewItem
        queue.add(ReviewItem(
            thread_id="tid-5", review_type="chapter",
            project_name="test", current_chapter=1,
            content_summary="to clear",
        ))
        queue.decide(thread_id="tid-5", decision="approve")
        removed = queue.clear_completed()
        assert removed >= 1


class TestReviewItemDataclass:
    """P2: ReviewItem dataclass fields and defaults."""

    def test_create_review_item(self):
        from novelfactory.crews.review_queue import ReviewItem, ReviewStatus
        item = ReviewItem(
            thread_id="tid",
            review_type="milestone",
            project_name="p",
            current_chapter=50,
            content_summary="milestone reached",
        )
        assert item.status == ReviewStatus.PENDING
        assert item.decision is None
        assert item.comment is None

    def test_review_status_enum(self):
        from novelfactory.crews.review_queue import ReviewStatus
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.APPROVED.value == "approved"
        assert ReviewStatus.REJECTED.value == "rejected"
        assert ReviewStatus.MODIFIED.value == "modified"
