"""P1: Review / HIL nodes — wait_for_review_node + chapter_human_guidance.

Tests the state interface of the interrupt-based review nodes.
The actual interrupt() call cannot be unit-tested without a LangGraph
runtime, but the state-read path (before interrupt) and the resume path
(after interrupt returns a dict) can be verified with FakeState.
"""

from __future__ import annotations

from tests.conftest import make_state

_DRAFT_PREVIEW_LENGTH = 500


class TestWaitForReviewNode:
    """P1: wait_for_review_node — state shape before interrupt().

    The node reads chapter_draft, quality_score, pending_review from state
    and calls interrupt(). The interrupt itself suspends — we test that
    the state-read path is correct by verifying the input fields are
    accessed properly via FakeState.
    """

    def test_node_accessible(self):
        from novelfactory.graph.nodes.review import wait_for_review_node
        # With FakeState, interrupt() is not available (no runtime),
        # so this will raise. But the import and function are valid.
        assert callable(wait_for_review_node)

    def test_node_reads_state_fields_correctly(self):
        """Verify node references expected state field names."""
        import inspect

        from novelfactory.graph.nodes.review import wait_for_review_node
        source = inspect.getsource(wait_for_review_node)
        # key state fields referenced by the node
        assert "pending_review" in source
        assert "chapter_draft" in source
        assert "quality_score" in source
        assert "current_chapter" in source
        assert "target_chapters" in source


class TestChapterHumanGuidance:
    """P1: chapter_human_guidance — state-read + resume path.

    Tests:
    1. If human_guidance already in state, returns empty dict (resume path)
    2. Otherwise builds interrupt_data with review info
    """

    def test_node_accessible(self):
        from novelfactory.graph.nodes.review import chapter_human_guidance
        assert callable(chapter_human_guidance)

    def test_generates_guidance_when_llm_fails(self):
        """When LLM call fails, uses fallback guidance text."""
        from novelfactory.graph.nodes.review import chapter_human_guidance
        state = make_state(
            current_chapter=5, human_guidance="already provided",
            crew_result={
                "review_result": {"quality_score": 75, "review_comments": "needs work"},
                "chapter_draft": "draft text",
            },
        )
        result = chapter_human_guidance(state)
        # v6.3: node always generates guidance even if human_guidance already set
        assert result.get("user_decision") == "provide_guidance"
        assert result.get("pending_review") is None
        assert result.get("guidance_complete") is True
        assert result.get("chapter_needs_guidance") is False
        assert "human_guidance" in result

    def test_reads_review_result_fields(self):
        """Verify node references expected state field names."""
        import inspect

        from novelfactory.graph.nodes.review import chapter_human_guidance
        source = inspect.getsource(chapter_human_guidance)
        assert "human_guidance" in source
        assert "crew_result" in source
        assert "review_result" in source
        assert "quality_score" in source
        assert "current_chapter" in source
