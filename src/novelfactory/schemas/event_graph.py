"""事件图结构（v7.3 新增）。

参考 STORYWRITER (清华 2025) 的 EventSeed/EventValidator 事件图设计。
用于结构化的事件图大纲，替代纯文本大纲，便于一致性校验和伏笔追踪。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StoryEvent(BaseModel):
    """事件图节点。"""

    event_id: str = Field(description="事件唯一标识，如 'E001'")
    sequence: int = Field(description="事件时序序号")
    title: str = Field(description="事件标题")
    time: str = Field(description="事件发生时间，如 '第3天/秋季/三年前'")
    location: str = Field(description="事件地点")
    characters: list[str] = Field(default_factory=list, description="涉及的角色名列表")
    goal: str = Field(default="", description="角色在此事件中的核心目标")
    conflict: str = Field(default="", description="核心冲突描述")
    plot_twist: str = Field(default="", description="情节转折点")
    causal_parents: list[str] = Field(
        default_factory=list, description="前驱事件ID列表（因果链）"
    )
    foreshadowing_plant: list[str] = Field(
        default_factory=list, description="本事件埋下的伏笔"
    )
    foreshadowing_resolve: list[str] = Field(
        default_factory=list, description="本事件回收的伏笔"
    )


class EventGraph(BaseModel):
    """完整事件图 — 覆盖一卷或一部小说的所有事件。"""

    events: list[StoryEvent] = Field(default_factory=list, description="所有事件")
    chapters: list[list[str]] = Field(
        default_factory=list, description="每章包含的事件ID列表"
    )

    def get_chapter_events(self, chapter_index: int) -> list[StoryEvent]:
        """获取指定章节的事件列表。"""
        if chapter_index < 0 or chapter_index >= len(self.chapters):
            return []
        event_ids = set(self.chapters[chapter_index])
        return [e for e in self.events if e.event_id in event_ids]

    def get_causal_chain(self, event_id: str) -> list[StoryEvent]:
        """追踪事件的因果链（递归找所有前驱事件）。"""
        event_map = {e.event_id: e for e in self.events}
        chain = []
        visited = set()
        stack = [event_id]
        while stack:
            eid = stack.pop()
            if eid in visited or eid not in event_map:
                continue
            visited.add(eid)
            event = event_map[eid]
            chain.append(event)
            stack.extend(event.causal_parents)
        return chain

    def add_event(self, event: StoryEvent) -> None:
        """添加事件。"""
        self.events.append(event)

    def assign_to_chapter(self, event_id: str, chapter_index: int) -> None:
        """将事件分配到章节。"""
        while len(self.chapters) <= chapter_index:
            self.chapters.append([])
        if event_id not in self.chapters[chapter_index]:
            self.chapters[chapter_index].append(event_id)
