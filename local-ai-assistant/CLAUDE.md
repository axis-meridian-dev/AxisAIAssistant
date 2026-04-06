# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A local-first AI assistant for Linux combining Ollama (local LLM) with optional cloud reasoning (Claude/GPT). Specialized for legal research but general-purpose. Provides CLI, voice, and web dashboard interfaces.

## Running

```bash
source .venv/bin/activate
python main.py              # CLI mode
python main.py --voice      # Voice mode
python server.py            # Web dashboard (localhost:5000)
python watcher.py ~/Documents  # Auto-ingest file watcher
```

No automated tests exist. Test manually via the CLI.

## Architecture

**Hybrid execution flow:** User query → intent detection → local Ollama routes tools → tools execute → cloud/local generates final response → validation/patching → output.

Key files:
- `agent.py` — Core orchestrator: intent detection, mode system, tool dispatch, hybrid cloud/local routing, response validation
- `cloud_reasoning.py` — Multi-provider routing (Anthropic/OpenAI), model selection by complexity, cost tracking against monthly budget
- `config.py` — Loads defaults, merges `config/settings.json`, injects `.env` secrets at runtime
- `config_utils.py` — Safe config writing (strips API keys before saving)
- `main.py` — Entry point, CLI input loop
- `voice.py` — Whisper STT + Piper TTS + wake word detection
- `server.py` — Flask web dashboard (serves `templates/dashboard.html`)
- `watcher.py` — watchdog-based file system watcher for auto-ingestion
- `stats.py` — Per-inquiry and session performance metrics
- `templates/dashboard.html` — Web dashboard front-end (HTML/CSS/JS)

**Tools** (in `tools/`, all inherit from `tools/base.py:BaseTool`):
- `file_manager.py` — File ops with security restrictions (allowed roots, excluded dirs)
- `web_search.py` — SearXNG + DuckDuckGo fallback
- `desktop_control.py` — xdotool/wmctrl-based app launching, window management, clipboard, screenshots, command execution with tiered security
- `system_info.py` — CPU/RAM/GPU/disk stats via psutil + nvidia-smi
- `knowledge_base.py` — ChromaDB RAG with legal-aware chunking (statutes on § boundaries, cases on structural markers)
- `legal_research.py` — CourtListener case law, House.gov/Cornell statutes, DOJ stats, news, research projects
- `document_writer.py` — Markdown document generation with YAML frontmatter

## Adding a New Tool

1. Create `tools/my_tool.py` inheriting from `BaseTool`
2. Implement `get_tool_definitions()` (Ollama JSON Schema format) and `get_handlers()` (name→method map)
3. Register in `agent.py` `__init__`: `self.tool_instances["my_tool"] = MyTool(config)`

The agent auto-discovers definitions and handlers from registered tool instances.

## Configuration

- `config/settings.json` — User-facing config (models, search, voice, features, cloud settings)
- `.env` — API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — never committed
- Config loads: hardcoded defaults → settings.json overrides → .env secrets injected

## Utility Scripts (scripts/)

- `scripts/setup.sh` — Full environment install (Ollama, Python packages, system tools)
- `scripts/setup_knowledge_base.sh` — Knowledge base initialization
- `scripts/install.sh` — File placement helper for fresh installs
- `scripts/local-ai-assistant.service` — systemd service unit file
- `scripts/bulk_download.py` — Bulk legal content downloader (v1)
- `scripts/bulk_download_v2.py` — Bulk legal content downloader (v2, fixed URLs)
- `scripts/fine_tune_generator.py` — OpenAI fine-tuning JSONL generator
- `scripts/patches/` — One-shot migration scripts (already applied, archived for reference)

## Legal Mode Specifics

When intent is "legal", the agent enforces strict mode: mandatory tool calls before answering, citation requirements, confidence scoring, and auto-appended disclaimers. Responses follow a structured format (APPLICABLE LAW → CASE LAW → APPLICATION → CONFIDENCE → REASONING). The `validate_and_patch()` method in `agent.py` enforces this.

## Dependencies

- **System:** xdotool, wmctrl, xclip, xdg-utils, ffmpeg, scrot (X11 only — Wayland needs ydotool)
- **Python:** ollama, httpx, rich, chromadb, flask, anthropic, openai, duckduckgo-search, beautifulsoup4, PyMuPDF, watchdog, psutil, pynput, python-xlib
- **Services:** Ollama (required), SearXNG via Docker (optional, DDG fallback available)
