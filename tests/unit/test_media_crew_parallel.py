"""media_crew 并行媒体生成验收测试。

覆盖 _parallel_media_node 的 ThreadPoolExecutor + wait 并行收集：
    - 插图/配音均成功 → illustration_url + audio_url 正确返回
    - 单维度失败（插图异常）不影响配音结果（独立 try/except 保护）

外部依赖（agent 工厂 / LLM / 流写入）全部 mock，不触发真实生成。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from novelfactory.graph.crews.media_crew import (
    MediaCrewLocalState,
    _parallel_media_node,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  测试辅助
# ═══════════════════════════════════════════════════════════════════════════════


class FakeStreamWriter:
    """模拟 StreamWriter — 记录 section/write 调用，无需真实文件写入。"""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def section(self, text: str) -> None:
        self.lines.append(("section", text))

    def write(self, text: str) -> None:
        self.lines.append(("write", text))


def _make_state() -> MediaCrewLocalState:
    """构建 MediaCrewLocalState 最小 state（LangGraph 节点以 dict 传参）。"""
    return cast(
        MediaCrewLocalState,
        {
            "crew_result": {"current_chapter_number": 1},
            "messages": [],
            "crew_error": None,
            "illustration_url": "",
            "illustration_prompt": "",
            "audio_url": "",
            "media_retry_count": 0,
            "illustration_error": "",
            "audio_error": "",
        },
    )


def _patch_media_deps(
    monkeypatch: pytest.MonkeyPatch,
    illustrator_fn: object,
    tts_fn: object,
) -> FakeStreamWriter:
    """mock media_crew 的外部依赖，返回 FakeStreamWriter 用于断言。

    Args:
        illustrator_fn: 可调用对象，签名 (inputs: dict) -> dict
        tts_fn: 可调用对象，签名 (inputs: dict) -> dict
    """
    sw = FakeStreamWriter()
    monkeypatch.setattr(
        "novelfactory.graph.crews.media_crew.get_crew_stream",
        lambda crew_name, prefix: sw,
    )
    monkeypatch.setattr(
        "novelfactory.graph.crews.media_crew.cleanup_crew_stream",
        lambda crew_name, prefix: None,
    )
    monkeypatch.setattr(
        "novelfactory.graph.crews.media_crew.get_worker_llm",
        lambda: SimpleNamespace(model="mock-llm"),
    )
    monkeypatch.setattr(
        "novelfactory.graph.crews.media_crew.create_illustrator_agent",
        lambda llm: illustrator_fn,
    )
    monkeypatch.setattr(
        "novelfactory.graph.crews.media_crew.create_tts_generator_agent",
        lambda llm: tts_fn,
    )
    return sw


# ═══════════════════════════════════════════════════════════════════════════════
#  并行收集验证
# ═══════════════════════════════════════════════════════════════════════════════


class TestParallelMediaNode:
    """_parallel_media_node wait 并行收集测试。"""

    def test_both_success_returns_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """插图/配音均成功 → 正确返回 illustration_url + audio_url。"""
        illustrator = lambda inputs: {  # noqa: E731
            "illustration_url": "http://mock/img/1.png",
            "illustration_prompt": "少年持剑而立",
        }
        tts = lambda inputs: {"audio_url": "http://mock/audio/1.mp3"}  # noqa: E731
        _patch_media_deps(monkeypatch, illustrator, tts)

        result = _parallel_media_node(_make_state())

        cr = result["crew_result"]
        assert cr["illustration_url"] == "http://mock/img/1.png"
        assert cr["illustration_prompt"] == "少年持剑而立"
        assert cr["audio_url"] == "http://mock/audio/1.mp3"
        # 返回的 crew_result 应保留原字段（current_chapter_number）
        assert cr["current_chapter_number"] == 1

    def test_illustrator_failure_keeps_audio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """插图异常不阻塞配音结果（独立 try/except 保护）。"""
        def illustrator(inputs: dict) -> dict:
            raise RuntimeError("mock 插图生成失败")

        tts = lambda inputs: {"audio_url": "http://mock/audio/2.mp3"}  # noqa: E731
        _patch_media_deps(monkeypatch, illustrator, tts)

        result = _parallel_media_node(_make_state())

        cr = result["crew_result"]
        assert cr["audio_url"] == "http://mock/audio/2.mp3"  # 配音不受影响
        assert cr["illustration_url"] == ""  # 插图失败 → 空 URL

    def test_tts_failure_keeps_illustration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """配音异常不影响插图结果。"""
        illustrator = lambda inputs: {  # noqa: E731
            "illustration_url": "http://mock/img/3.png",
            "illustration_prompt": "雨夜孤城",
        }
        def tts(inputs: dict) -> dict:
            raise RuntimeError("mock 配音生成失败")

        _patch_media_deps(monkeypatch, illustrator, tts)

        result = _parallel_media_node(_make_state())

        cr = result["crew_result"]
        assert cr["illustration_url"] == "http://mock/img/3.png"
        assert cr["audio_url"] == ""

    def test_both_fail_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """双维度均失败 → 两个 URL 均为空，不抛异常。"""
        def illustrator(inputs: dict) -> dict:
            raise RuntimeError("插图失败")

        def tts(inputs: dict) -> dict:
            raise RuntimeError("配音失败")

        _patch_media_deps(monkeypatch, illustrator, tts)

        result = _parallel_media_node(_make_state())

        cr = result["crew_result"]
        assert cr["illustration_url"] == ""
        assert cr["audio_url"] == ""

    def test_stream_writer_receives_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """流式进度写入 StreamWriter（section/write 被调用）。"""
        illustrator = lambda inputs: {  # noqa: E731
            "illustration_url": "http://mock/img/4.png",
            "illustration_prompt": "",
        }
        tts = lambda inputs: {"audio_url": "http://mock/audio/4.mp3"}  # noqa: E731
        sw = _patch_media_deps(monkeypatch, illustrator, tts)

        _parallel_media_node(_make_state())

        sections = [text for kind, text in sw.lines if kind == "section"]
        writes = [text for kind, text in sw.lines if kind == "write"]
        assert any("媒体生成" in s for s in sections)
        assert writes, "应至少写入一条进度"
