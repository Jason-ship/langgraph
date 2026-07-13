"""Shared test fixtures — mock LLM, mock DB, fake states."""

from __future__ import annotations

# ── Fake State Helpers ─────────────────────────────────────────────────────────


class FakeState(dict):
    """A dict subclass that supports .get() for LangGraph state simulation."""

    def get(self, key, default=None):
        return super().get(key, default)


def make_state(**overrides) -> FakeState:
    """Build a minimal FakeState resembling NovelFactoryState."""
    base = {
        "project_name": "测试项目",
        "genre": "仙侠",
        "seed_idea": "一个少年获得绝世神剑的故事",
        "target_chapters": 100,
        "thread_id": "test-tid-001",
        "current_chapter": 1,
        "current_phase": "setup",
        "setup_complete": False,
        "pending_review": None,
        "chapter_approved": False,
        "chapter_needs_guidance": False,
        "guidance_complete": False,
        "world_setting": "",
        "character_setting": "",
        "story_outline": "",
        "auto_guidance": "",
        "volume_status": {},
        "quality_trend": {},
        "foreshadowing_status": {},
        "messages": [],
    }
    base.update(overrides)
    return FakeState(base)


class FakeWritingState:
    """Fake state that mimics WritingCrewLocalState for _score_router testing."""

    def __init__(self, quality_score=0.0, composite_score=0.0,
                 loop_count=0, refine_attempts=0):
        self.quality_score = quality_score
        self.composite_score = composite_score
        self.loop_count = loop_count
        self.refine_attempts = refine_attempts

    def get(self, key, default=0):
        return getattr(self, key, default)


# ── Mock LLM ──────────────────────────────────────────────────────────────────


class MockLLMResponse:
    """Simulate a LangChain LLM response."""

    def __init__(self, content="", response_metadata=None):
        self.content = content
        self.response_metadata = response_metadata or {}

    def __str__(self):
        return self.content


class MockLLM:
    """Mock LLM that returns a fixed response."""

    def __init__(self, response: MockLLMResponse | str = ""):
        self._response = response
        self.model = "deepseek-chat"

    def invoke(self, messages, **kwargs):
        if isinstance(self._response, str):
            return MockLLMResponse(content=self._response)
        return self._response

    async def ainvoke(self, messages, **kwargs):
        """v5.4: 异步 invoke — 用于测试异步 LLM 路径。"""
        return self.invoke(messages, **kwargs)


class MockLLMFail:
    """Mock LLM that always raises an exception."""

    model = "deepseek-chat"

    def invoke(self, messages, **kwargs):
        raise RuntimeError("mock LLM failure")

    async def ainvoke(self, messages, **kwargs):
        """v5.4: 异步 invoke — 用于测试异步 LLM 错误路径。"""
        raise RuntimeError("mock LLM failure")
