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


async def cli_loop(agent: Agent):
    """Standard CLI interaction loop."""
    print_banner()
    
    cfg = agent.config
    console.print(f"[dim]Model: {cfg['llm']['primary_model']}[/dim]")
    console.print(f"[dim]Tools: {', '.join(agent.list_tools())}[/dim]")
    console.print(f"[dim]Type 'quit' to exit, 'tools' to list capabilities[/dim]")
    console.print(f"[dim]Modes: 'mode <research|analysis|argument|write|general>'[/dim]\n")

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
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        elif cmd == "model":
            console.print(f"[dim]Primary: {cfg['llm']['primary_model']}[/dim]")
            console.print(f"[dim]Fast:    {cfg['llm']['fast_model']}[/dim]")
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


if __name__ == "__main__":
    asyncio.run(main())
