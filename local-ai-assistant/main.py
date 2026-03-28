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

console = Console()


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


async def cli_loop(agent: Agent):
    """Standard CLI interaction loop."""
    print_banner()

    cfg = agent.config
    console.print(f"[dim]Model: {cfg['llm']['primary_model']}[/dim]")
    console.print(f"[dim]Tools: {', '.join(agent.list_tools())}[/dim]")
    console.print(f"[dim]Type 'quit' to exit, 'tools' to list capabilities[/dim]")
    console.print(f"[dim]Modes: 'mode <research|analysis|argument|write|general>'[/dim]")
    console.print(f"[dim]Stats: 'stats' | 'perf' | 'history' | 'history <id>'[/dim]\n")

    while True:
        try:
            mode_tag = f" [{agent.mode}]" if agent.mode != "general" else ""
            user_input = Prompt.ask(f"[bold green]You{mode_tag}[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        elif cmd == "tools":
            agent.print_tools()
            continue
        elif cmd == "clear":
            agent.clear_history()
            console.print("[dim]Conversation cleared. Legal lock released. Mode reset to general.[/dim]")
            continue
        elif cmd == "model":
            console.print(f"[dim]Primary: {cfg['llm']['primary_model']}[/dim]")
            console.print(f"[dim]Fast:    {cfg['llm']['fast_model']}[/dim]")
            continue
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
        elif cmd.startswith("history"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                # List recent sessions
                sessions = agent.session_stats.list_chat_sessions()
                if not sessions:
                    console.print("[dim]No chat history found.[/dim]")
                else:
                    console.print("[bold cyan]Recent Sessions:[/bold cyan]")
                    for s in sessions:
                        total = s.get("total_time", 0)
                        console.print(
                            f"  [dim]{s['session_id']}[/dim]  "
                            f"{s['started_at'][:16]}  "
                            f"{s['messages']} msgs  "
                            f"{s['inquiries']} queries  "
                            f"{total:.1f}s total"
                        )
                    console.print("[dim]Use 'history <session_id>' to load a session.[/dim]")
            else:
                session_id = parts[1]
                session = agent.session_stats.load_chat_session(session_id)
                if not session:
                    console.print(f"[red]Session '{session_id}' not found.[/red]")
                else:
                    console.print(f"[bold cyan]Session: {session.get('session_id', '?')}[/bold cyan]")
                    console.print(f"[dim]Started: {session.get('started_at', '?')}[/dim]")
                    console.print(f"[dim]Queries: {session.get('inquiry_count', 0)} | "
                                  f"Total time: {session.get('total_time', 0):.1f}s[/dim]\n")
                    for msg in session.get("messages", []):
                        role = msg.get("role", "?")
                        content = msg.get("content", "")[:300]
                        if role == "user":
                            console.print(f"[green]You:[/green] {content}")
                        elif role == "assistant":
                            console.print(f"[cyan]AI:[/cyan] {content}")
                        console.print()
            continue
        elif cmd.startswith("mode"):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 1:
                console.print(f"[dim]Current mode: {agent.mode}[/dim]")
                console.print(f"[dim]Available: research, analysis, argument, write, general[/dim]")
            else:
                result = agent.set_mode(parts[1])
                console.print(f"[dim]{result}[/dim]")
            continue

        # Process through agent — no spinner so tool calls print in real time
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


async def main():
    config = load_config()
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

    # Save chat history on exit
    if agent.history:
        agent.session_stats.save_chat_session(agent.history)
        console.print(f"[dim]Chat saved: session {agent.session_stats.session_id}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
