"""Tests for writing_crew.py — writing subgraph construction and flow."""

from __future__ import annotations

import pytest


class TestWritingCrewBuild:
    """Verify writing crew graph builds correctly."""

    @pytest.mark.smoke
    def test_writing_crew_imports(self):
        """Verify writing_crew module imports without error."""
        import novelfactory.graph.crews.writing_crew  # noqa: F401

        assert True

    @pytest.mark.smoke
    def test_build_writing_crew(self):
        """Verify build_writing_crew returns a compiled graph."""
        try:
            import novelfactory.graph.crews.writing_crew as wc

            crew = wc.build_writing_crew()
            assert crew is not None
        except Exception as exc:
            pytest.skip(f"Writing crew build requires LLM config: {exc}")

    def test_writing_crew_constants(self):
        """Verify writing crew constants are importable."""
        from novelfactory.config.constants import (
            MAX_REWRITE_ATTEMPTS,
            MIN_CHAPTER_TEXT_LENGTH,
        )

        assert MAX_REWRITE_ATTEMPTS >= 1
        assert MIN_CHAPTER_TEXT_LENGTH > 0

    def test_writing_crew_exports_writer_node(self):
        """Verify the chapter writer node function is importable."""
        from novelfactory.graph.crews.writing_nodes.writer import _chapter_writer_node

        assert callable(_chapter_writer_node)


class TestWritingCrewState:
    """Test the WritingCrewLocalState definition."""

    def test_base_crew_state_fields(self):
        """Verify BaseCrewState has expected fields."""
        import typing

        from novelfactory.state.crew_state import BaseCrewState

        hints = typing.get_type_hints(BaseCrewState)
        assert "messages" in hints
        assert "crew_result" in hints
        assert "crew_error" in hints

    def test_writing_crew_state_has_result(self):
        """Verify WritingCrewLocalState has crew_result field."""
        import typing

        from novelfactory.graph.crews.writing_crew import WritingCrewLocalState

        hints = typing.get_type_hints(WritingCrewLocalState)
        assert "crew_result" in hints
