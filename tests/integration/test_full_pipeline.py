"""P2: End-to-end pipeline structure test — all subgraphs compile + connect.

Validates that the full pipeline can be assembled: root graph + all four
crew subgraphs (setup/writing/media/sync) compile and connect via edges.
"""

from __future__ import annotations

import pytest


class TestFullPipelineAssembly:
    """P2: All subgraphs compile and integrate into the root graph."""

    @pytest.fixture(scope="module")
    def root_graph(self):
        from novelfactory.graph.new_builder import build_novel_factory_graph
        return build_novel_factory_graph()

    def test_setup_crew_compiles(self):
        from novelfactory.graph.lightweight_setup import build_setup_crew
        crew = build_setup_crew()
        assert crew is not None
        nodes = set(crew.get_graph().nodes.keys())
        assert "init_setup" in nodes
        assert "world_builder" in nodes
        assert crew.recursion_limit == 200

    def test_writing_crew_compiles(self):
        from novelfactory.graph.crews.writing_crew import build_writing_crew
        crew = build_writing_crew()
        assert crew is not None
        nodes = set(crew.get_graph().nodes.keys())
        assert "chapter_writer" in nodes
        assert "quality_panel" in nodes  # v5.5: 辩论评审替代 chapter_reviewer
        assert crew.recursion_limit == 200

    def test_media_crew_compiles(self):
        from novelfactory.graph.crews.media_crew import build_media_crew
        crew = build_media_crew()
        assert crew is not None
        assert crew.recursion_limit == 200

    def test_sync_crew_compiles(self):
        from novelfactory.graph.crews.sync_crew import build_sync_crew
        crew = build_sync_crew()
        assert crew is not None
        assert crew.recursion_limit == 200

    def test_root_graph_contains_all_crews(self, root_graph):
        compiled = root_graph.compile()
        node_names = set(compiled.get_graph().nodes.keys())
        crews = {"setup_crew", "writing_crew", "media_crew", "sync_crew"}
        present = crews & node_names
        assert len(present) == 4, f"Missing crews: {crews - present}"

    def test_setup_edged_to_main_supervisor(self, root_graph):
        """setup_crew → main_supervisor edge exists."""
        compiled = root_graph.compile()
        edges = list(compiled.get_graph().edges)
        edge_src_target = [(e.source, e.target) for e in edges]
        assert ("setup_crew", "main_supervisor") in edge_src_target

    def test_prepare_writing_edged_to_writing_crew(self, root_graph):
        compiled = root_graph.compile()
        edges = list(compiled.get_graph().edges)
        edge_src_target = [(e.source, e.target) for e in edges]
        assert ("prepare_writing", "writing_crew") in edge_src_target

    def test_phase_checks_form_chain(self, root_graph):
        """volume → quality → foreshadowing must form a chain."""
        compiled = root_graph.compile()
        edges = list(compiled.get_graph().edges)
        edge_src_target = [(e.source, e.target) for e in edges]
        vol_to_quality = ("volume_check", "quality_check") in edge_src_target
        quality_to_fore = ("quality_check", "foreshadowing_check") in edge_src_target
        assert vol_to_quality, "Missing volume_check → quality_check edge"
        assert quality_to_fore, "Missing quality_check → foreshadowing_check edge"


class TestNodePipelineIntegration:
    """P2: Individual nodes interface compatibility with FakeState."""

    def test_supervisor_node_accepts_fake_state(self):
        from novelfactory.graph.nodes.supervisor import main_supervisor_node
        from tests.conftest import make_state
        state = make_state(current_phase="setup")
        result = main_supervisor_node(state)
        assert isinstance(result, dict)

    def test_route_from_supervisor_accepts_fake_state(self):
        from novelfactory.graph.routing import route_from_supervisor
        from tests.conftest import make_state
        state = make_state(current_phase="setup")
        result = route_from_supervisor(state)
        assert isinstance(result, str)
        assert result in (
            "setup_crew", "load_memory", "save_memory", "refresh_quota",
            "writing_crew", "media_crew", "sync_crew", "wait_for_review",
            "chapter_human_guidance", "volume_check",
        )

    def test_score_router_accepts_fake_state(self):
        from novelfactory.graph.crews.writing_nodes.routing import _score_router
        state = type("obj", (), {
            "quality_score": 95.0, "composite_score": 0.8,
            "loop_count": 0, "refine_attempts": 0,
            "get": lambda self, k, d=0: getattr(self, k, d),
        })()
        result = _score_router(state)
        assert result in ("__exit_for_chapter__", "chapter_writer", "chapter_refiner")


class TestStateReducersIntegration:
    """P2: Reducers compose correctly across multiple chapters."""

    def test_multi_chapter_accumulation(self):
        from novelfactory.state.novel_state import _add_usage

        ch1 = {
            "chapter_usages": [
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 100, "completion_tokens": 50},
            ],
        }
        ch2 = {
            "chapter_usages": [
                {"chapter_number": 2, "phase": "writing",
                 "prompt_tokens": 200, "completion_tokens": 100},
            ],
        }
        ch3 = {
            "chapter_usages": [
                {"chapter_number": 3, "phase": "writing",
                 "prompt_tokens": 300, "completion_tokens": 150},
            ],
        }

        result = _add_usage({}, ch1)
        result = _add_usage(result, ch2)
        result = _add_usage(result, ch3)

        assert len(result["chapter_usages"]) == 3
        assert result["total_tokens"] == 900  # 150 + 300 + 450

    def test_rewrite_then_continue(self):
        from novelfactory.state.novel_state import _add_usage

        ch1 = {
            "chapter_usages": [
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 100, "completion_tokens": 50},
            ],
        }
        ch1_rewrite = {
            "chapter_usages": [
                {"chapter_number": 1, "phase": "writing",
                 "prompt_tokens": 200, "completion_tokens": 100},
            ],
        }
        ch2 = {
            "chapter_usages": [
                {"chapter_number": 2, "phase": "writing",
                 "prompt_tokens": 300, "completion_tokens": 150},
            ],
        }

        result = _add_usage({}, ch1)
        result = _add_usage(result, ch1_rewrite)
        result = _add_usage(result, ch2)

        assert len(result["chapter_usages"]) == 2
        # ch1 replaced by rewrite: 200+100=300, ch2: 300+150=450
        assert result["total_tokens"] == 750
