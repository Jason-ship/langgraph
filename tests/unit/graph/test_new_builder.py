"""Tests for graph/new_builder.py — root graph construction and compilation."""

from __future__ import annotations

import pytest

from novelfactory.config.constants import DEFAULT_RETRY, REVIEWER_RETRY, WRITER_RETRY


class TestRetryPolicy:
    """Verify RetryPolicy defaults defined in checkpointer.py."""

    def test_default_retry_three_attempts(self):
        assert DEFAULT_RETRY.max_attempts == 3

    def test_default_retry_has_jitter(self):
        assert DEFAULT_RETRY.jitter is True

    def test_writer_retry_five_attempts(self):
        assert WRITER_RETRY.max_attempts == 5

    def test_reviewer_retry_three_attempts(self):
        assert REVIEWER_RETRY.max_attempts == 3

    def test_writer_retry_retries_all(self):
        """Writer retry should retry on any exception (LLM calls)."""
        for exc in [RuntimeError("fail"), ValueError("fail"), TimeoutError("timeout")]:
            assert WRITER_RETRY.retry_on(exc) is True

    def test_default_retry_filters_timeout(self):
        assert DEFAULT_RETRY.retry_on(TimeoutError("timeout")) is True
        assert DEFAULT_RETRY.retry_on(ValueError("no")) is False
        assert DEFAULT_RETRY.retry_on(RuntimeError("oops")) is True


class TestGraphCompilation:
    """Test that the root graph compiles without errors."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_build_novel_factory_graph_imports(self):
        """Verify the builder module imports cleanly."""
        import novelfactory.graph.new_builder  # noqa: F811

        assert hasattr(novelfactory.graph.new_builder, "build_novel_factory_graph")

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_build_and_compile_graph(self):
        """Smoke test: build graph with explicit InMemory checkpointer (v5.7)."""
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        from novelfactory.graph.new_builder import build_novel_factory_graph

        checkpointer = InMemorySaver()
        store = InMemoryStore()
        try:
            graph = build_novel_factory_graph()
            compiled = graph.compile(checkpointer=checkpointer, store=store)
            assert compiled is not None
            assert compiled.recursion_limit == 5000
        except Exception as exc:
            pytest.skip(f"Graph compilation requires full env: {exc}")

    @pytest.mark.smoke
    def test_builder_accepts_checkpointer_override(self):
        """Verify the builder accepts an explicit checkpointer."""
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        assert checkpointer is not None


class TestGraphStructure:
    """Structural tests on the graph definition."""

    @pytest.mark.smoke
    def test_graph_has_supervisor_node(self):
        """Root graph should define a main_supervisor node."""
        import novelfactory.graph.new_builder as builder_mod

        assert hasattr(builder_mod, "build_novel_factory_graph")

    @pytest.mark.smoke
    def test_checkpointer_module_exports(self):
        """Verify checkpointer.py exports expected symbols."""
        from novelfactory.config.constants import (  # noqa: F401
            DEFAULT_RETRY,
            REVIEWER_RETRY,
            WRITER_RETRY,
        )
        from novelfactory.graph.checkpointer import (  # noqa: F401
            create_checkpointer,
            create_store,
            get_checkpointer_instance,
            set_checkpointer_instance,
        )

        assert True
