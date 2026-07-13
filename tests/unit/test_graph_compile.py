"""P1: Graph compile and structure verification tests.

Validates that the root graph and writing crew subgraph can be built
without errors and contain the expected nodes.
"""

from __future__ import annotations

import pytest


class TestRootGraphCompile:
    """P1: Root StateGraph compiles and contains expected core nodes."""

    @pytest.fixture(scope="module")
    def graph(self):
        from novelfactory.graph.new_builder import build_novel_factory_graph
        return build_novel_factory_graph()

    def test_graph_builds_without_error(self, graph):
        assert graph is not None

    def test_graph_has_compile_method(self, graph):
        assert hasattr(graph, "compile")

    def test_compiled_graph_has_core_nodes(self, graph):
        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        expected = {
            "main_supervisor",
            "setup_crew",
            "writing_crew",
            "media_crew",
            "sync_crew",
            "prepare_writing",
            "save_memory",
        }
        missing = expected - node_names
        assert not missing, f"Missing core nodes: {missing}"

    def test_compiled_graph_has_phase_check_nodes(self, graph):
        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        phase_nodes = {"volume_check", "quality_check", "foreshadowing_check"}
        present = phase_nodes & node_names
        assert len(present) > 0

    def test_compiled_graph_has_interrupt_nodes(self, graph):
        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        interrupt_nodes = {"wait_for_review", "chapter_human_guidance"}
        present = interrupt_nodes & node_names
        assert len(present) > 0

    def test_compiled_graph_has_monitor_node(self, graph):
        compiled = graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        assert "intelligent_monitor" in node_names


class TestWritingCrewCompile:
    """P1: Writing crew subgraph compiles correctly."""

    @pytest.fixture(scope="module")
    def crew(self):
        from novelfactory.graph.crews.writing_crew import build_writing_crew
        return build_writing_crew()

    def test_crew_builds_without_error(self, crew):
        assert crew is not None

    def test_crew_has_recursion_limit(self, crew):
        assert hasattr(crew, "recursion_limit")
        assert crew.recursion_limit == 200

    def test_crew_contains_writing_nodes(self, crew):
        nodes = set(crew.get_graph().nodes.keys())
        expected = {
            "context_builder_node",
            "chapter_writer",
            "verdict_engine",
            "chapter_refiner",
            "state_extractor_node",
            "database_writer_node",
        }
        missing = expected - nodes
        assert not missing, f"Missing writing crew nodes: {missing}"
