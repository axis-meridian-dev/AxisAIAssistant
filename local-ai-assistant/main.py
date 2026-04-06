#!/usr/bin/env python3
"""
Local AI Assistant — Main Agent Loop
Runs a tool-calling LLM agent with file, search, desktop, and voice capabilities.
"""

import json
import sys
import os
import signal
import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent import Agent
from config import load_config
from log import setup_logging

console = Console()
logger = setup_logging()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]LOCAL AI ASSISTANT[/bold cyan]\n"
        "[dim]Powered by Ollama · Fully Local · Fully Yours[/dim]",
        border_style="cyan"
    ))
    console.print()


def print_inquiry_stats(agent: Agent):
    """Print performance stats for the last inquiry."""
    stats = agent.last_inquiry_stats
    if not stats:
        return

    console.print(Panel(
        agent.session_stats.format_inquiry_stats(stats),
        title="[dim]Performance[/dim]",
        border_style="dim",
        padding=(0, 1)
    ))


def print_help():
    """Print available commands."""
    console.print(Panel(
        "[bold]Chat Commands:[/bold]\n"
        "  new                   Start a new conversation\n"
        "  history               List saved sessions\n"
        "  load <id>             Resume a previous session\n"
        "  rename <title>        Rename current session\n"
        "  delete <id>           Delete a saved session\n"
        "\n"
        "[bold]Model Commands:[/bold]\n"
        "  model                 Show current models and balances\n"
        "  model list            List all available cloud models\n"
        "  model use <id>        Set cloud model for this session\n"
        "  model auto            Return to auto model selection\n"
        "  balance               Show provider balances\n"
        "\n"
        "[bold]Mode Commands:[/bold]\n"
        "  mode                  Show current mode\n"
        "  mode <name>           Set mode (research|analysis|argument|write|general)\n"
        "\n"
        "[bold]Other:[/bold]\n"
        "  tools                 List available tools\n"
        "  stats                 Session statistics\n"
        "  perf                  Model performance history\n"
        "  clear                 Clear history and reset mode\n"
        "  help                  Show this help\n"
        "  quit                  Exit",
        title="[bold cyan]Commands[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    ))


def print_model_list(agent: Agent):
    """Print all available cloud models grouped by provider."""
    from cloud_reasoning import MODELS
    console.print()
    for provider in ("anthropic", "openai"):
        console.print(f"[bold cyan]{provider.title()} Models:[/bold cyan]")
        for model_id, info in MODELS.items():
            if info["provider"] != provider:
                continue
            affordable = agent.cloud.can_afford(model_id)
            is_active = model_id == agent.cloud_model_override
            is_default = model_id in (agent.cloud.default_anthropic, agent.cloud.default_openai)

            tag = ""
            if is_active:
                tag = " [bold green]<< ACTIVE[/bold green]"
            elif is_default:
                tag = " [dim](default)[/dim]"

            cost_indicator = "[green]$[/green]" if affordable else "[red]$[/red]"
            console.print(
                f"  {cost_indicator} [bold]{model_id}[/bold] — {info['name']}  "
                f"[dim]in=${info['input']}/M out=${info['output']}/M  "
                f"{info['speed']}  {info['context']//1000}K ctx[/dim]{tag}"
            )
        console.print()


async def cli_loop(agent: Agent):
    """Standard CLI interaction loop."""
    print_banner()

    cfg = agent.config
    bal = agent.cloud.get_balances()
    console.print(f"[dim]Local model: {cfg['llm']['primary_model']}[/dim]")
    console.print(f"[dim]Cloud: Anthropic ${bal.get('anthropic', 0):.2f} · OpenAI ${bal.get('openai', 0):.2f}[/dim]")
    console.print(f"[dim]Session: {agent.session_stats.session_id}[/dim]")
    console.print(f"[dim]Type 'help' for commands, 'quit' to exit[/dim]\n")

    while True:
        try:
            mode_tag = f" [{agent.mode}]" if agent.mode != "general" else ""
            model_tag = f" @{agent.cloud_model_override}" if agent.cloud_model_override else ""
            user_input = Prompt.ask(f"[bold green]You{mode_tag}{model_tag}[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()

        # ── Exit ──────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        # ── Help ──────────────────────────────────────────────────────
        elif cmd == "help":
            print_help()
            continue

        # ── New session ───────────────────────────────────────────────
        elif cmd == "new":
            agent.new_session()
            console.print(f"[dim]New session: {agent.session_stats.session_id}[/dim]")
            continue

        # ── Tools ─────────────────────────────────────────────────────
        elif cmd == "tools":
            agent.print_tools()
            continue

        # ── Clear ─────────────────────────────────────────────────────
        elif cmd == "clear":
            agent.clear_history()
            console.print("[dim]Conversation cleared. Mode reset to general.[/dim]")
            continue

        # ── Balance ───────────────────────────────────────────────────
        elif cmd == "balance":
            bal = agent.cloud.get_balances()
            sp = agent.cloud.provider_spend
            console.print(f"[bold cyan]Account Balances:[/bold cyan]")
            console.print(f"  Anthropic: [green]${bal.get('anthropic', 0):.2f}[/green] remaining  (${sp.get('anthropic', 0):.2f} spent)")
            console.print(f"  OpenAI:    [green]${bal.get('openai', 0):.2f}[/green] remaining  (${sp.get('openai', 0):.2f} spent)")
            console.print(f"  Total spent this month: ${agent.cloud.monthly_spend:.2f}")
            continue

        # ── Model commands ────────────────────────────────────────────
        elif cmd == "model" or cmd == "model list" or cmd.startswith("model "):
            parts = user_input.strip().split(maxsplit=2)
            subcmd = parts[1] if len(parts) > 1 else ""

            if subcmd == "list":
                print_model_list(agent)

            elif subcmd == "use" and len(parts) > 2:
                from cloud_reasoning import MODELS
                model_id = parts[2].strip()
                if model_id in MODELS:
                    agent.cloud_model_override = model_id
                    info = MODELS[model_id]
                    console.print(f"[dim]Cloud model set to: {model_id} ({info['name']}) via {info['provider']}[/dim]")
                else:
                    console.print(f"[red]Unknown model: {model_id}[/red]")
                    console.print(f"[dim]Use 'model list' to see available models[/dim]")

            elif subcmd == "auto":
                agent.cloud_model_override = None
                console.print("[dim]Cloud model: auto (smart routing)[/dim]")

            else:
                # Show current
                console.print(f"[dim]Local:  {cfg['llm']['primary_model']}[/dim]")
                if agent.cloud_model_override:
                    console.print(f"[dim]Cloud:  {agent.cloud_model_override} (manual)[/dim]")
                else:
                    console.print(f"[dim]Cloud:  auto-routed (default: {agent.cloud.default_anthropic} / {agent.cloud.default_openai})[/dim]")
                bal = agent.cloud.get_balances()
                console.print(f"[dim]Anthropic balance: ${bal.get('anthropic', 0):.2f}  ·  OpenAI balance: ${bal.get('openai', 0):.2f}[/dim]")
                console.print(f"[dim]Use 'model list' for all models, 'model use <id>' to pick one[/dim]")
            continue

        # ── Stats ─────────────────────────────────────────────────────
        elif cmd == "stats":
            console.print(Panel(
                agent.session_stats.format_session_summary(),
                title="[bold cyan]Session Stats[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))
            continue
        elif cmd == "perf":
            console.print(Panel(
                agent.session_stats.format_model_stats(),
                title="[bold cyan]Model Performance[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))
            continue

        # ── Chat history commands ─────────────────────────────────────
        elif cmd == "history":
            sessions = agent.session_stats.list_chat_sessions()
            if not sessions:
                console.print("[dim]No chat history found.[/dim]")
            else:
                console.print("[bold cyan]Saved Sessions:[/bold cyan]")
                for i, s in enumerate(sessions, 1):
                    is_current = s["session_id"] == agent.session_stats.session_id
                    marker = " [green]<< current[/green]" if is_current else ""
                    title = s.get("title", "Untitled")
                    if len(title) > 60:
                        title = title[:57] + "..."
                    console.print(
                        f"  {i:>2}. [dim]{s['session_id']}[/dim]  "
                        f"{s['started_at'][:16]}  "
                        f"[bold]{title}[/bold]  "
                        f"[dim]{s['messages']} msgs[/dim]{marker}"
                    )
                console.print(f"\n[dim]'load <id>' to resume · 'delete <id>' to remove · 'new' for fresh chat[/dim]")
            continue

        elif cmd.startswith("load "):
            session_id = cmd.split(maxsplit=1)[1].strip()
            if agent.load_session(session_id):
                title = "?"
                data = agent.session_stats.load_chat_session(agent.session_stats.session_id)
                if data:
                    title = data.get("title", "Untitled")
                console.print(f"[dim]Resumed session: {agent.session_stats.session_id}[/dim]")
                console.print(f"[dim]Title: {title}[/dim]")
                console.print(f"[dim]{len(agent.history)} messages loaded. Continue the conversation:[/dim]")
                # Show last few messages for context
                recent = agent.history[-4:]
                for msg in recent:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")[:200]
                    if role == "user":
                        console.print(f"  [green]You:[/green] {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
                    elif role == "assistant":
                        console.print(f"  [cyan]AI:[/cyan] {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
                console.print()
            else:
                console.print(f"[red]Session '{session_id}' not found.[/red]")
            continue

        elif cmd.startswith("rename "):
            new_title = user_input.strip()[7:].strip()
            if new_title:
                agent.session_stats.rename_session(new_title)
                console.print(f"[dim]Session renamed to: {new_title}[/dim]")
            else:
                console.print("[dim]Usage: rename <title>[/dim]")
            continue

        elif cmd.startswith("delete "):
            session_id = cmd.split(maxsplit=1)[1].strip()
            if agent.session_stats.delete_chat_session(session_id):
                console.print(f"[dim]Session {session_id} deleted.[/dim]")
            else:
                console.print(f"[red]Session '{session_id}' not found.[/red]")
            continue

        # ── Mode ──────────────────────────────────────────────────────
        elif cmd.startswith("mode"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                console.print(f"[dim]Current mode: {agent.mode}[/dim]")
                console.print(f"[dim]Available: research, analysis, argument, write, general[/dim]")
            else:
                result = agent.set_mode(parts[1])
                console.print(f"[dim]{result}[/dim]")
            continue

        # ── Process query ─────────────────────────────────────────────
        console.print()
        console.print("[bold cyan]Processing...[/bold cyan]")
        response = await agent.process(user_input)

        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold cyan]Assistant[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))

        # Show performance stats for this inquiry
        print_inquiry_stats(agent)
        console.print()


def verify_dependencies(config: dict) -> bool:
    """Check that critical services are available before starting."""
    ok = True

    # Check Ollama connectivity
    try:
        import ollama as _ollama
        models = _ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, "models") else []
        primary = config["llm"]["primary_model"]
        if model_names and not any(primary in m for m in model_names):
            console.print(f"[yellow]Warning: Primary model '{primary}' not found in Ollama. "
                          f"Available: {', '.join(model_names[:5])}[/yellow]")
        else:
            console.print(f"[green]Ollama: OK ({len(model_names)} models)[/green]")
    except Exception as e:
        console.print(f"[red]Ollama: UNAVAILABLE — {e}[/red]")
        console.print("[red]  Start with: ollama serve[/red]")
        ok = False

    # Check cloud API keys (non-fatal)
    cloud_cfg = config.get("cloud", {})
    if cloud_cfg.get("enabled", False):
        has_anthropic = bool(cloud_cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(cloud_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY"))
        provider = cloud_cfg.get("provider", "anthropic")
        if provider == "anthropic" and not has_anthropic:
            console.print("[yellow]Warning: Cloud enabled but no ANTHROPIC_API_KEY set[/yellow]")
        elif provider == "openai" and not has_openai:
            console.print("[yellow]Warning: Cloud enabled but no OPENAI_API_KEY set[/yellow]")
        else:
            console.print(f"[green]Cloud: OK (provider={provider})[/green]")

    # Check knowledge base path
    kb_path = Path(config.get("knowledge_base", {}).get("db_path", "")).expanduser()
    if kb_path.exists():
        console.print(f"[green]Knowledge base: OK ({kb_path})[/green]")
    else:
        console.print("[dim]Knowledge base: not initialized (run ingest to create)[/dim]")

    console.print()
    return ok


async def main():
    config = load_config()

    if not verify_dependencies(config):
        console.print("[red]Critical dependency missing. Fix the above and try again.[/red]")
        sys.exit(1)

    agent = Agent(config)

    # Check if voice mode requested
    if "--voice" in sys.argv:
        try:
            from voice import VoiceInterface
            voice = VoiceInterface(agent, config)
            await voice.run()
        except ImportError:
            console.print("[red]Voice dependencies not installed. Run setup.sh first.[/red]")
            sys.exit(1)
    else:
        await cli_loop(agent)

    # Final save on exit (session auto-saves after each exchange, this catches edge cases)
    if agent.history:
        agent.session_stats.save_chat_session(agent.history)
        console.print(f"[dim]Session saved: {agent.session_stats.session_id}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
