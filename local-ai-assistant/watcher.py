"""
Knowledge Base File Watcher — auto-ingest new/changed files.

Runs in the background and watches specified directories for changes.
When files are created or modified, they're automatically re-ingested
into the knowledge base so the AI always has current context.

Usage:
    python watcher.py                    # Watch ~/Documents, ~/Projects, ~/Desktop
    python watcher.py /path/to/watch     # Watch a specific directory
"""

import sys
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

from rich.console import Console
from config import load_config
from tools.knowledge_base import KnowledgeBaseTool, TEXT_EXTENSIONS

console = Console()


class KBEventHandler(FileSystemEventHandler):
    """Watches for file changes and triggers re-ingestion."""
    
    def __init__(self, kb: KnowledgeBaseTool):
        self.kb = kb
        self.valid_ext = TEXT_EXTENSIONS | {".pdf"}
        self._debounce = {}
        self._debounce_seconds = 5  # Wait 5s after last change before ingesting
    
    def _should_process(self, path: str) -> bool:
        p = Path(path)
        if p.suffix.lower() not in self.valid_ext:
            return False
        # Skip hidden, build artifacts, etc.
        skip = {".git", "node_modules", "__pycache__", ".cache", ".venv",
                "venv", "dist", "build", "target", ".eggs"}
        return not any(part in skip for part in p.parts)
    
    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            self._queue_ingest(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            self._queue_ingest(event.src_path)
    
    def _queue_ingest(self, path: str):
        """Debounced ingestion — wait for file to settle before processing."""
        import threading
        
        # Cancel previous timer for this file
        if path in self._debounce:
            self._debounce[path].cancel()
        
        def do_ingest():
            try:
                result = self.kb.ingest_file(path)
                console.print(f"[dim]Auto-ingested: {Path(path).name}[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to ingest {path}: {e}[/yellow]")
            finally:
                self._debounce.pop(path, None)
        
        timer = threading.Timer(self._debounce_seconds, do_ingest)
        self._debounce[path] = timer
        timer.start()


def main():
    config = load_config()
    kb = KnowledgeBaseTool(config)
    
    # Determine watch directories
    if len(sys.argv) > 1:
        watch_dirs = [Path(p).expanduser().resolve() for p in sys.argv[1:]]
    else:
        home = Path.home()
        watch_dirs = [
            home / "Documents",
            home / "LegalResearch",
            home / "Desktop",
            home / "Development",
            home / "Documents",
            home / "Pictures",
        ]
    
    # Filter to only directories that actually exist and are accessible
    missing = [d for d in watch_dirs if not d.exists() or not d.is_dir()]
    watch_dirs = [d for d in watch_dirs if d.exists() and d.is_dir()]
    
    for d in missing:
        console.print(f"[yellow]Skipping (not found or not a directory): {d}[/yellow]")
    
    if not watch_dirs:
        console.print("[red]No valid directories to watch.[/red]")
        console.print("[dim]Provide existing directories: python watcher.py ~/Documents ~/Desktop[/dim]")
        sys.exit(1)
    
    handler = KBEventHandler(kb)
    observer = Observer()
    
    for d in watch_dirs:
        try:
            observer.schedule(handler, str(d), recursive=True)
            console.print(f"[green]Watching:[/green] {d}")
        except Exception as e:
            console.print(f"[yellow]Failed to watch {d}: {e}[/yellow]")
    
    console.print(f"\n[dim]Files will be auto-ingested into the knowledge base.[/dim]")
    console.print(f"[dim]Press Ctrl+C to stop.[/dim]\n")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]Watcher stopped.[/dim]")
    observer.join()


if __name__ == "__main__":
    main()