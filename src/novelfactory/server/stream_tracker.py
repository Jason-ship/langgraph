"""SSE 流式进度追踪器 — 从 streaming.py 拆出。

``StreamStateTracker`` 从 LangGraph stream 事件的 state dict 中提取进度信息
（phase / chapter / agent / quality_score / chapter_preview），维护消息去重集合，
并格式化为 progress event dict 供 SSE 推送。

借鉴 TradingAgents MessageBuffer 模式：
  - 累积状态 from stream chunks
  - 发射 progress events（含章节预览、质量数据）
  - 消息去重（防止 LangGraph stream 重复推送同一 message）
"""

from __future__ import annotations


class StreamStateTracker:
    """Tracks agent progress across SSE stream events with message dedup.

    Similar to TradingAgents' MessageBuffer pattern:
    accumulates state from stream chunks and emits progress events
    with chapter preview, quality data, and message deduplication.
    """

    def __init__(self) -> None:
        self.phase: str = ""
        self.current_chapter: int = 0
        self.total_chapters: int = 0
        self.current_agent: str = ""
        self.agent_status: str = "pending"
        self.chapter_preview: str | None = None
        self.quality_score: float | None = None
        self.composite_score: float | None = None
        self._processed_message_ids: set[str] = set()

    def update_from_state(self, state: dict) -> bool:
        """Extract progress from a state dict. Returns True if anything changed."""
        changed = False
        phase = state.get("current_phase", "")
        if phase and phase != self.phase:
            self.phase = phase
            changed = True
        chapter = state.get("current_chapter", 0)
        if chapter and chapter != self.current_chapter:
            self.current_chapter = chapter
            changed = True
        next_nodes = state.get("next", [])
        if next_nodes:
            agent = next_nodes[0] if isinstance(next_nodes, list) else str(next_nodes)
            if agent != self.current_agent:
                self.current_agent = agent
                self.agent_status = "in_progress"
                changed = True

        content = state.get("chapter_content") or state.get("content")
        if isinstance(content, str) and len(content) > 80:
            preview = content[:800]
            if preview != self.chapter_preview:
                self.chapter_preview = preview
                changed = True

        quality = state.get("quality_score")
        if quality is not None and quality != self.quality_score:
            self.quality_score = quality
            # v6.1: 从 verdict_result 读取 programmatic_score
            verdict = state.get("verdict_result", {})
            self.composite_score = verdict.get("programmatic_score") or state.get(
                "composite_score"
            )
            changed = True

        outline = state.get("outline") or state.get("chapter_outline") or {}
        tc = outline.get("total_chapters") or len(outline.get("chapters", [])) or 0
        if tc and tc > self.total_chapters:
            self.total_chapters = tc
            changed = True

        return changed

    def is_message_duplicate(self, msg_id: str) -> bool:
        if msg_id and msg_id in self._processed_message_ids:
            return True
        if msg_id:
            self._processed_message_ids.add(msg_id)
        return False

    def to_progress_event(self) -> dict:
        event: dict = {
            "phase": self.phase,
            "chapter": self.current_chapter,
            "agent": self.current_agent,
            "agent_status": self.agent_status,
        }
        if self.chapter_preview:
            event["chapter_preview"] = self.chapter_preview
        if self.quality_score is not None:
            event["quality_score"] = self.quality_score
            event["composite_score"] = self.composite_score
        if self.total_chapters:
            event["total_chapters"] = self.total_chapters
        return event
