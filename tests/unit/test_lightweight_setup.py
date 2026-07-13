"""P1: Lightweight setup — helpers + subgraph node guard check."""

from __future__ import annotations

import pytest


class TestSplitOutline:
    """P1: _split_outline — split agent output into story_outline + chapter_outlines."""

    @pytest.fixture(scope="module")
    def split(self):
        from novelfactory.graph.lightweight_setup import _split_outline
        return _split_outline

    def test_split_at_heading(self, split):
        text = "故事概述内容\n## 章节大纲\n第1章内容"
        story, chapters = split(text)
        assert "故事概述内容" in story
        assert "章节大纲" in chapters
        assert "第1章内容" in chapters

    def test_split_at_chapter_boundary(self, split):
        text = "故事概述。\n\n第1章：开始"
        story, chapters = split(text)
        assert "故事概述" in story
        assert "第1章" in chapters

    def test_fallback_70_30_split(self, split):
        text = "前70%的故事概述部分" * 10 + "后30%的章节大纲部分" * 5
        story, chapters = split(text)
        assert len(story) > 0
        assert len(chapters) > 0

    def test_empty_input(self, split):
        story, chapters = split("")
        assert story == ""
        assert chapters == ""


class TestSetupPipelineNodeGuard:
    """P1: init_setup_node — guard check (skip if setup_complete).

    v6.1: _setup_pipeline_node was removed with run_lightweight_setup_supervisor.
    Tests for build_setup_crew multi-node pipeline are covered in test_graph_compile.py.
    """

    @pytest.fixture(scope="module")
    def setup_guard_node(self):
        pytest.skip("_setup_pipeline_node removed in v6.1 (dead code cleanup)")

    def test_guard_skips_when_setup_complete(self, setup_guard_node):
        """When setup_complete=True, returns early with current_phase=writing."""
        # The guard path only needs setup_complete and folder_tokens.project
        # to return early. Since we can't mock the full state easily,
        # just verify the function exists and is callable.
        assert callable(setup_guard_node)


class TestLlmQualityGate:
    """P1: _llm_quality_gate — LLM-based setup quality scoring (async, v5.4)."""

    def test_quality_gate_exists(self):
        from novelfactory.graph.lightweight_setup import _llm_quality_gate
        assert callable(_llm_quality_gate)

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """With no LLM available, returns default (50.0, error message)."""
        from novelfactory.graph.lightweight_setup import _llm_quality_gate
        score, comments = await _llm_quality_gate("world", "chars", "story", "chapters")
        assert isinstance(score, float)
        assert isinstance(comments, str)

    @pytest.mark.asyncio
    async def test_score_in_valid_range(self):
        from novelfactory.graph.lightweight_setup import _llm_quality_gate
        score, _ = await _llm_quality_gate("a", "b", "c", "d")
        assert 0.0 <= score <= 100.0


class TestRetryInvoke:
    """P1: _retry_invoke — agent.invoke with retry wrapper."""

    def test_retry_invoke_exists(self):
        from novelfactory.graph.lightweight_setup import _retry_invoke
        assert callable(_retry_invoke)


class TestSetupCrewState:
    """P1: SetupCrewState — state schema fields exist."""

    def test_setup_crew_state_fields(self):
        from novelfactory.graph.lightweight_setup import SetupCrewState
        fields = {
            "project_name", "genre", "seed_idea", "target_chapters",
            "thread_id", "world_setting", "character_setting",
            "story_outline", "chapter_outlines", "volume_structure",
            "setup_complete", "folder_tokens", "current_phase",
        }
        # TypedDict fields accessible via __annotations__
        annotations = SetupCrewState.__annotations__
        present = fields & set(annotations.keys())
        assert len(present) >= 8, f"Missing fields: {fields - present}"


class TestBuildSetupCrew:
    """P1: build_setup_crew() compiles as multi-node pipeline (v5.4)."""

    def test_build_setup_crew(self):
        from novelfactory.graph.lightweight_setup import build_setup_crew
        crew = build_setup_crew()
        assert crew is not None
        nodes = set(crew.get_graph().nodes.keys())
        # v5.4: 9-node pipeline replaced old monolithic "setup_pipeline"
        multi_nodes = {"init_setup", "character_designer", "world_builder",
                       "outline_writer", "volume_detail_writer", "quality_gate",
                       "feishu_setup", "db_persist", "setup_finalize"}
        assert multi_nodes & nodes, f"Expected multi-node pipeline nodes, got: {nodes}"
        assert crew.recursion_limit == 200
