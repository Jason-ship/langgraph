import json
import pathlib
import sys
import time

import click
import requests
from rich.console import Console

from novelfactory import __version__

console = Console()


def _load_seed_file(seed_path: str) -> dict | None:
    fp = pathlib.Path(seed_path).expanduser().resolve()
    if not fp.exists():
        console.print(f"[red]Seed file not found:[/red] {fp}")
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


@click.group()
@click.version_option(__version__, "-V", "--version")
def cli() -> None:
    """NovelFactory CLI — Multi-Agent Novel Creation Server."""


@cli.command()
@click.option(
    "--url", default="http://localhost:8123", help="API base URL", show_default=True
)
@click.option("--thread-id", "-t", required=True, help="Thread ID for the novel run")
@click.option(
    "--seed",
    "-s",
    default=None,
    help="Path to seed JSON file (project_context + outline)",
)
def run(url: str, thread_id: str, seed: str | None) -> None:
    """Start a novel writing run via the API."""
    payload: dict = {"input": {}}
    if seed:
        seed_data = _load_seed_file(seed)
        if seed_data is None:
            sys.exit(1)
        payload["input"] = seed_data

    api = f"{url.rstrip('/')}/threads/{thread_id}/runs"
    console.print(f"[cyan]Starting run...[/cyan] thread={thread_id}")
    try:
        resp = requests.post(api, json=payload, timeout=10)
        resp.raise_for_status()
        console.print(f"[green]Run started:[/green] {resp.json()}")
    except requests.RequestException as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.option(
    "--url", default="http://localhost:8123", help="API base URL", show_default=True
)
def health(url: str) -> None:
    """Check API health status."""
    try:
        resp = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        resp.raise_for_status()
        console.print_json(data=resp.json())
    except requests.RequestException as exc:
        console.print(f"[red]Health check failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Print NovelFactory version."""
    console.print(f"[bold cyan]NovelFactory[/bold cyan] v{__version__}")


@cli.command()
@click.option(
    "--url", default="http://localhost:8123", help="API base URL", show_default=True
)
@click.option("--thread-id", "-t", required=True, help="Thread ID")
@click.option("--seed", "-s", default=None, help="Path to seed JSON file")
@click.option(
    "--attach",
    "-a",
    is_flag=True,
    default=False,
    help="Attach to existing run (do not send seed input)",
)
def dashboard(url: str, thread_id: str, seed: str | None, attach: bool) -> None:
    """Launch Rich terminal dashboard for real-time novel writing progress.

    Connects to API SSE streaming endpoint and renders a live terminal UI showing:
      - Agent/team progress status
      - Real-time messages and tool call logs
      - Current chapter preview and quality review
      - Statistics (LLM calls, tokens, elapsed time)

    Press Ctrl+C to exit the dashboard.
    """
    from novelfactory.cli.dashboard import run_dashboard

    seed_data: dict | None = None
    if seed and not attach:
        seed_data = _load_seed_file(seed)
        if seed_data is None:
            sys.exit(1)

    if attach:
        console.print(f"[cyan]Attaching to thread[/cyan] {thread_id} ...")
    else:
        console.print(f"[cyan]Launching dashboard[/cyan] thread={thread_id} ...")
        time.sleep(0.5)

    run_dashboard(
        api_url=url,
        thread_id=thread_id,
        seed=seed_data,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  参数调优 CLI
# ═══════════════════════════════════════════════════════════════════════════════


@cli.group()
def params() -> None:
    """LLM 参数管理中心 — 查看和调优运行参数。"""


@params.command("list")
def params_list() -> None:
    """列出所有 LLM 参数。"""
    from novelfactory.config.llm_params import center

    data = center.list_params()
    console.print("[bold]Tier 参数:[/bold]")
    for name, cfg in sorted(data["tiers"].items()):
        console.print(f"  [cyan]{name}[/cyan]:")
        for key, val in cfg.items():
            console.print(f"    {key}: {val}")
    console.print("")
    console.print("[bold]Agent 参数:[/bold]")
    for key, cfg in sorted(data["agents"].items()):
        console.print(f"  [cyan]{key}[/cyan]:")
        for k, v in cfg.items():
            console.print(f"    {k}: {v}")
    if data["env_overrides"]:
        console.print("")
        console.print(f"[yellow]环境变量覆盖: {data['env_overrides']}[/yellow]")


@params.command("set-tier")
@click.argument("name")
@click.option("--temperature", type=float, help="新 temperature")
@click.option("--max-tokens", type=int, help="新 max_tokens")
@click.option("--timeout", type=float, help="新 timeout_seconds")
@click.option("--max-retries", type=int, help="新 max_retries")
def params_set_tier(
    name: str,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float | None,
    max_retries: int | None,
) -> None:
    """更新 Tier 参数（运行时生效）。"""
    from novelfactory.config.llm_params import center

    overrides: dict = {}
    if temperature is not None:
        overrides["temperature"] = temperature
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens
    if timeout is not None:
        overrides["timeout_seconds"] = timeout
    if max_retries is not None:
        overrides["max_retries"] = max_retries

    if not overrides:
        console.print("[yellow]未指定任何参数。[/yellow]")
        return

    updated = center.update_tier(name, **overrides)
    if updated is None:
        console.print(f"[red]Tier '{name}' 不存在。[/red]")
        return
    console.print(
        f"[green]Tier '{name}' 已更新: temp={updated.temperature} "
        f"timeout={updated.timeout_seconds}s retries={updated.max_retries}[/green]"
    )


@params.command("set-agent")
@click.argument("tier")
@click.argument("agent")
@click.option("--temperature", type=float, help="新 temperature")
@click.option("--timeout", type=float, help="新 timeout_seconds")
def params_set_agent(
    tier: str,
    agent: str,
    temperature: float | None,
    timeout: float | None,
) -> None:
    """更新 Agent 参数（运行时生效）。"""
    from novelfactory.config.llm_params import center

    overrides: dict = {}
    if temperature is not None:
        overrides["temperature"] = temperature
    if timeout is not None:
        overrides["timeout_seconds"] = timeout

    if not overrides:
        console.print("[yellow]未指定任何参数。[/yellow]")
        return

    updated = center.update_agent(tier, agent, **overrides)
    if updated is None:
        console.print(f"[red]Agent '{tier}/{agent}' 不存在。[/red]")
        return
    console.print(
        f"[green]Agent '{tier}/{agent}' 已更新: temp={updated.temperature} "
        f"timeout={updated.timeout_seconds}s[/green]"
    )
