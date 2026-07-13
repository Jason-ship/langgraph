"""P1: StreamWriter + crew stream manager."""

from __future__ import annotations

import os

import pytest

from novelfactory.agents.infra.stream import (
    StreamWriter,
    cleanup_all_crew_streams,
    cleanup_crew_stream,
    get_crew_stream,
)


class TestStreamWriter:
    """P1: StreamWriter — buffered output to temp file."""

    @pytest.fixture
    def stream(self):
        writer = StreamWriter(prefix="test", project_name="testproj")
        yield writer
        try:
            writer.close()
        except Exception:
            pass

    def test_creates_temp_file(self, stream):
        assert os.path.exists(stream.path)

    def test_writes_content(self, stream):
        stream.write("hello world")
        stream.close()
        with open(stream.path, encoding="utf-8") as f:
            content = f.read()
        assert "hello world" in content

    def test_section_header(self, stream):
        stream.section("测试章节")
        stream.close()
        with open(stream.path, encoding="utf-8") as f:
            content = f.read()
        assert "## 测试章节" in content

    def test_buffer_flushes_on_close(self, stream):
        stream.write("buffered content")
        stream.close()
        with open(stream.path, encoding="utf-8") as f:
            content = f.read()
        assert "buffered content" in content

    def test_closed_writer_noop(self, stream):
        stream.close()
        stream.write("after close")  # should not raise

    def test_context_manager(self):
        with StreamWriter(prefix="ctx", project_name="p") as sw:
            sw.write("ctx write")
        assert os.path.exists(sw.path)
        with open(sw.path, encoding="utf-8") as f:
            assert "ctx write" in f.read()


class TestCrewStreamManager:
    """P1: get_crew_stream / cleanup_crew_stream lifecycle."""

    def test_get_and_cleanup(self):
        sw = get_crew_stream("test_crew", "ch1")
        assert isinstance(sw, StreamWriter)
        cleanup_crew_stream("test_crew", "ch1")

    def test_same_key_returns_same_instance(self):
        sw1 = get_crew_stream("shared", "ch1")
        sw2 = get_crew_stream("shared", "ch1")
        assert sw1 is sw2
        cleanup_crew_stream("shared", "ch1")

    def test_different_prefix_different_instance(self):
        sw1 = get_crew_stream("crew", "ch1")
        sw2 = get_crew_stream("crew", "ch2")
        assert sw1 is not sw2
        cleanup_crew_stream("crew", "ch1")
        cleanup_crew_stream("crew", "ch2")

    def test_cleanup_all(self):
        get_crew_stream("ca1", "p1")
        get_crew_stream("ca2", "p2")
        cleanup_all_crew_streams()
