# ==============================================================================
# NovelFactory Rich CLI Dashboard — Real-time Writing Progress Monitor
#
# 借鉴 TradingAgents cli/main.py (1291 行) Rich Live 布局模式:
#   - MessageBuffer: 消息/工具/状态/报告累积
#   - Layout: header / upper(progress+messages) / analysis / footer
#   - Live 刷新: 4 fps 实时更新终端界面
#   - SSE 流: 连接 API stream 端点，解析 progress/updates/messages 事件
# ==============================================================================

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import threading
import time
from collections import deque
from typing import Any

import httpx
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


class MessageBuffer:
    """累积 SSE 流中的消息、工具调用、状态、报告。"""

    AGENT_TEAMS = {
        "Setup": ["setup_crew"],
        "Memory": ["load_memory", "save_memory"],
        "Writing": [
            "writing_crew",
            "quality_panel",
            "chapter_writer",
            "chapter_refiner",
        ],
        "Review": ["wait_for_review"],
        "Media": ["media_crew"],
        "Sync": ["sync_crew"],
        "Checks": ["volume_check", "quality_check", "foreshadowing_check"],
        "System": ["stream"],
    }

    TEAM_ORDER = ["Setup", "Memory", "Writing", "Review", "Media", "Sync", "Checks"]

    def __init__(self, max_length: int = 100) -> None:
        self.messages: deque[tuple[str, str, str]] = deque(maxlen=max_length)
        self.tool_calls: deque[tuple[str, str, str]] = deque(maxlen=max_length)
        self.current_report: str | None = None
        self.final_report: str | None = None
        self.agent_status: dict[str, str] = {}
        self.current_agent: str | None = None
        self.report_sections: dict[str, str | None] = {
            "chapter_preview": None,
            "quality_review": None,
        }
        self._processed_message_ids: set[str] = set()
        self.current_chapter: int = 0
        self.total_chapters: int = 0
        self.phase: str = ""
        self.project_name: str = ""

    def init_for_run(self, project_name: str = "", total_chapters: int = 0) -> None:
        self.project_name = project_name
        self.total_chapters = total_chapters
        self.agent_status = {}
        self.current_agent = None
        self.current_chapter = 0
        self.phase = ""
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()
        self.current_report = None
        self.final_report = None
        for section in self.report_sections:
            self.report_sections[section] = None

    def add_message(self, message_type: str, content: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, str(content)[:500]))

    def add_tool_call(self, tool_name: str, args: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, str(args)[:300]))

    def update_agent_status(self, agent: str, status: str) -> None:
        self.agent_status[agent] = status
        self.current_agent = agent

    def update_report_section(self, section_name: str, content: str) -> None:
        if section_name in self.report_sections:
            self.report_sections[section_name] = str(content)[:2000]
            self._update_current_report()

    def _update_current_report(self) -> None:
        for section, content in self.report_sections.items():
            if content:
                titles = {
                    "chapter_preview": f"Chapter {self.current_chapter} Preview",
                    "quality_review": f"Chapter {self.current_chapter} Quality Review",
                }
                title = titles.get(section, section)
                self.current_report = f"### {title}\n{content}"
                return

    def get_completed_agents(self) -> int:
        return sum(1 for s in self.agent_status.values() if s == "completed")

    def get_total_agents(self) -> int:
        return len(self.agent_status)

    def get_active_teams(self) -> dict[str, list[str]]:
        teams: dict[str, list[str]] = {}
        for team_name in self.TEAM_ORDER:
            agents_in_team = self.AGENT_TEAMS.get(team_name, [])
            active = [a for a in agents_in_team if a in self.agent_status]
            if active:
                teams[team_name] = active
        return teams


def format_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def format_elapsed(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def create_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3),
        Layout(name="analysis", ratio=5),
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="messages", ratio=3),
    )
    return layout


def render_layout(
    layout: Layout,
    buffer: MessageBuffer,
    stats_handler: StreamStats | None = None,
    start_time: float | None = None,
    thread_id: str = "",
) -> None:
    _render_header(layout, buffer, thread_id)
    _render_progress(layout, buffer)
    _render_messages(layout, buffer)
    _render_analysis(layout, buffer)
    _render_footer(layout, buffer, stats_handler, start_time)


def _render_header(layout: Layout, buffer: MessageBuffer, thread_id: str) -> None:
    project = buffer.project_name or "(unnamed)"
    from novelfactory.config.settings import settings as _settings

    _ver = _settings.APP_VERSION
    header_text = f"[bold cyan]NovelFactory CLI[/bold cyan]  v{_ver}  |  Thread: [dim]{thread_id[:16]}...[/dim]  |  《{project}》"
    layout["header"].update(
        Panel(
            Align.center(header_text),
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _render_progress(layout: Layout, buffer: MessageBuffer) -> None:
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAD,
        padding=(0, 2),
        expand=True,
    )
    table.add_column("Team", style="cyan", justify="center", width=12)
    table.add_column("Agent", style="green", justify="center", width=18)
    table.add_column("Status", style="yellow", justify="center", width=16)

    teams = buffer.get_active_teams()
    for team_name, agents in teams.items():
        first = agents[0]
        status = buffer.agent_status.get(first, "pending")
        status_cell = _render_status(status)
        table.add_row(team_name, first, status_cell)
        for agent in agents[1:]:
            status = buffer.agent_status.get(agent, "pending")
            table.add_row("", agent, _render_status(status))
        table.add_row("─" * 12, "─" * 18, "─" * 16, style="dim")

    layout["progress"].update(
        Panel(table, title="Progress", border_style="cyan", padding=(1, 2))
    )


def _render_status(status: str) -> Spinner | str:
    if status == "in_progress":
        return Spinner("dots", text="[blue]in_progress[/blue]", style="bold cyan")
    color = {"pending": "yellow", "completed": "green", "error": "red"}.get(
        status, "white"
    )
    return f"[{color}]{status}[/{color}]"


def _render_messages(layout: Layout, buffer: MessageBuffer) -> None:
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.MINIMAL,
        expand=True,
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("Time", style="cyan", width=8, justify="center")
    table.add_column("Type", style="green", width=10, justify="center")
    table.add_column("Content", style="white", no_wrap=False, ratio=1)

    all_msgs: list[tuple[str, str, str]] = []
    for timestamp, tool_name, args in buffer.tool_calls:
        all_msgs.append((timestamp, "Tool", f"{tool_name}"))
    for timestamp, msg_type, content in buffer.messages:
        content_str = str(content)
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_msgs.append((timestamp, msg_type, content_str))

    all_msgs.sort(key=lambda x: x[0], reverse=True)
    for timestamp, msg_type, content in all_msgs[:12]:
        wrapped = Text(content, overflow="fold")
        table.add_row(timestamp, msg_type, wrapped)

    layout["messages"].update(
        Panel(table, title="Messages & Tools", border_style="blue", padding=(1, 2))
    )


def _render_analysis(layout: Layout, buffer: MessageBuffer) -> None:
    if buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(buffer.current_report),
                title=f"Current — Phase: {buffer.phase} | Ch: {buffer.current_chapter}/{buffer.total_chapters}",
                border_style="green",
                padding=(1, 2),
            )
        )
    elif buffer.agent_status.get("stream") == "connected":
        layout["analysis"].update(
            Panel(
                "[yellow]SSE 流已连接，等待 LLM 输出...[/yellow]\n\n"
                "[dim]首次 LLM 调用约需 20-60 秒，请耐心等待。[/dim]",
                title="Connecting",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for novel writing progress...[/italic]\n\n"
                "[dim]使用 --attach 附加到已有线程，或提供 --seed 开始新创作。[/dim]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )


def _render_footer(
    layout: Layout,
    buffer: MessageBuffer,
    stats_handler: StreamStats | None,
    start_time: float | None,
) -> None:
    parts: list[str] = []
    parts.append(f"Ch: {buffer.current_chapter}/{buffer.total_chapters}")
    parts.append(f"Agents: {buffer.get_completed_agents()}/{buffer.get_total_agents()}")

    if stats_handler:
        s = stats_handler.get_stats()
        parts.append(f"LLM: {s['llm_calls']}")
        parts.append(f"Tools: {s['tool_calls']}")
        if s["tokens_in"] > 0 or s["tokens_out"] > 0:
            parts.append(
                f"Tokens: {format_tokens(s['tokens_in'])}\u2191 {format_tokens(s['tokens_out'])}\u2193"
            )

    if start_time:
        parts.append(f"\u23f1 {format_elapsed(time.time() - start_time)}")

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(parts))
    layout["footer"].update(Panel(stats_table, border_style="grey50"))


class StreamStats:
    """轻量级回调处理器 — 追踪 LLM/工具/Token 统计。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def record_llm(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out

    def record_tool(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            }


def classify_message_type(message: dict) -> tuple[str, str | None]:
    """将 LangChain 消息分类为显示类型 (User/Agent/Data/Control/System)。"""
    msg_type = message.get("type", "unknown")
    content = message.get("content", "")

    mapping = {
        "human": ("User", content),
        "ai": ("Agent", content),
        "tool": ("Data", content),
        "system": ("System", content),
    }
    return mapping.get(msg_type, ("Info", str(content)[:200]))


def parse_sse_line(line: str) -> dict | None:
    """解析单行 SSE 数据。"""
    if line.startswith("data: "):
        data_str = line[6:].strip()
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None
    return None


def process_stream_event(
    event_type: str,
    data: dict,
    buffer: MessageBuffer,
    stats: StreamStats,
) -> None:
    """处理单个流事件，更新缓冲区。"""
    if event_type == "metadata":
        run_id = data.get("run_id", "")[:8]
        buffer.add_message("System", f"Run started: {run_id}")
        buffer.update_agent_status("stream", "connected")

    elif event_type == "updates":
        for node_name, node_update in data.items():
            if isinstance(node_update, dict):
                _process_node_update(node_name, node_update, buffer, stats)
        buffer.update_agent_status("stream", "streaming")

    elif event_type == "values":
        _process_values(data, buffer)

    elif event_type == "progress":
        phase = data.get("phase", "")
        chapter = data.get("chapter", 0)
        agent = data.get("agent", "")
        agent_status = data.get("agent_status", "")
        if phase:
            buffer.phase = phase
        if chapter:
            buffer.current_chapter = chapter
        if agent:
            buffer.update_agent_status(agent, agent_status)
            buffer.add_message("Progress", f"{agent} → {agent_status}")

    elif event_type == "messages":
        if isinstance(data, list) and len(data) >= 1:
            msg = data[0]
            if isinstance(msg, dict):
                msg_type, content = classify_message_type(msg)
                if content:
                    msg_id = msg.get("id", "")
                    if msg_id and msg_id in buffer._processed_message_ids:
                        return
                    if msg_id:
                        buffer._processed_message_ids.add(msg_id)
                    buffer.add_message(msg_type, content)

    elif event_type == "interrupt":
        buffer.add_message("Control", "INTERRUPT — waiting for review input")


def _process_node_update(
    node_name: str,
    update: dict,
    buffer: MessageBuffer,
    stats: StreamStats,
) -> None:
    """处理节点状态更新。"""
    buffer.update_agent_status(node_name, "in_progress")

    if node_name == "chapter_writer":
        content = update.get("content", update.get("chapter_content", ""))
        if content:
            preview = str(content)[:800]
            buffer.update_report_section("chapter_preview", preview)
            buffer.add_message("Agent", f"Writer produced {len(str(content))} chars")

    elif node_name in ("quality_panel", "chapter_reviewer"):
        quality = update.get("quality_score")
        if quality is not None:
            # v6.1: 从 verdict_result 读取 programmatic_score
            verdict = update.get("verdict_result", {})
            prog_score = verdict.get("programmatic_score") or update.get(
                "composite_score", "?"
            )
            buffer.update_report_section(
                "quality_review",
                f"Quality: {quality}/100 | Programmatic: {prog_score}",
            )
            buffer.add_message("Agent", f"Review: quality={quality}")

    buffer.update_agent_status(node_name, "completed")


def _process_values(values: dict, buffer: MessageBuffer) -> None:
    """处理完整状态快照。"""
    phase = values.get("current_phase", "")
    chapter = values.get("current_chapter", 0)
    project = values.get("project_context", {}).get("title", "")

    if phase:
        buffer.phase = phase
    if chapter:
        buffer.current_chapter = chapter
    if project:
        buffer.project_name = project

    outline = values.get("outline", values.get("chapter_outline", {}))
    total = outline.get("total_chapters", outline.get("chapters", 0))
    if isinstance(total, list):
        total = len(total)
    if total and total > buffer.total_chapters:
        buffer.total_chapters = total


async def stream_sse_events(
    url: str,
    thread_id: str,
    seed: dict | None,
    buffer: MessageBuffer,
    stats: StreamStats,
) -> None:
    """连接 API SSE 流端点并解析事件。"""
    api_url = f"{url.rstrip('/')}/threads/{thread_id}/runs"
    payload: dict[str, Any] = {"stream": True}
    if seed:
        payload["input"] = seed

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        async with client.stream("POST", api_url, json=payload) as response:
            response.raise_for_status()
            current_event: str | None = None
            data_lines: list[str] = []

            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data_lines.append(line)
                elif line == "" and current_event:
                    data_str = "".join(
                        dl[6:] if dl.startswith("data: ") else dl for dl in data_lines
                    )
                    try:
                        parsed = json.loads(data_str)
                    except json.JSONDecodeError:
                        parsed = {}
                    process_stream_event(current_event, parsed, buffer, stats)
                    current_event = None
                    data_lines = []


def run_dashboard(
    api_url: str,
    thread_id: str,
    seed: dict | None = None,
    project_name: str = "",
    total_chapters: int = 0,
) -> None:
    """主入口 — 启动 Rich Live 终端面板。"""
    buffer = MessageBuffer()
    buffer.init_for_run(project_name=project_name, total_chapters=total_chapters)
    stats = StreamStats()
    start_time = time.time()

    try:
        project_ctx = seed.get("project_context", {}) if seed else {}
        title = (
            project_ctx.get("title", "")
            if isinstance(project_ctx, dict)
            else str(project_ctx)
        )
        if title:
            buffer.project_name = title
        outline = seed.get("outline", {}) if seed else {}
        ch_count = outline.get("total_chapters", 0)
        if isinstance(ch_count, int) and ch_count > 0:
            buffer.total_chapters = ch_count
    except (TypeError, KeyError, AttributeError):
        pass

    layout = create_layout()
    buffer.add_message("System", f"Connecting to {api_url} | thread={thread_id[:16]}")
    render_layout(layout, buffer, stats, start_time, thread_id)

    async def stream_loop() -> None:
        try:
            await stream_sse_events(api_url, thread_id, seed, buffer, stats)
        except (httpx.HTTPError, OSError, ConnectionError) as exc:
            buffer.add_message("Error", f"Connection failed: {exc}")
        except Exception as exc:
            buffer.add_message("Error", f"Stream error: {exc}")
        finally:
            for agent in list(buffer.agent_status.keys()):
                buffer.update_agent_status(agent, "completed")
            buffer.add_message("System", "Stream ended")

    def sync_stream_runner() -> None:
        asyncio.run(stream_loop())

    stream_thread = threading.Thread(target=sync_stream_runner, daemon=True)
    stream_thread.start()

    try:
        with Live(layout, refresh_per_second=4, console=console) as live:
            while stream_thread.is_alive():
                render_layout(layout, buffer, stats, start_time, thread_id)
                live.refresh()
                time.sleep(0.25)
            render_layout(layout, buffer, stats, start_time, thread_id)
            live.refresh()
    except KeyboardInterrupt:
        buffer.add_message("System", "Dashboard stopped by user")
        render_layout(layout, buffer, stats, start_time, thread_id)
