# Local AI Assistant

A fully local, privacy-first AI assistant for Linux with legal research, file management, web research, desktop control, and voice interaction. Everything runs on your machine — no data leaves.

---

## What This Is

This is a local AI-powered workstation that combines:

- **LLM reasoning** (Ollama — Llama 3.1 70B or 8B)
- **Vector knowledge base** (ChromaDB + nomic-embed-text)
- **Legal research tools** (statute lookup, case law search, news monitoring)
- **Full desktop control** (apps, windows, clipboard, commands)
- **Voice interface** (Whisper STT + Piper TTS + wake word)

Think of it as a local, customizable version of Westlaw + Alexa + a research assistant, running entirely on your hardware.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         YOUR LINUX DESKTOP       │
                        └──────────────┬──────────────────┘
                                       │
              Microphone ──► Whisper (STT) ──┐
                                             ▼
                                    ┌────────────────┐
                                    │   Ollama LLM    │
                                    │  (Tool Router)  │
                                    └───────┬────────┘
                                            │
                    ┌───────────┬───────────┼───────────┬───────────┐
                    ▼           ▼           ▼           ▼           ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
              │  Files    │ │  Web   │ │ Desktop  │ │ Legal  │ │Knowledge │
              │ Manager   │ │Search  │ │ Control  │ │Research│ │  Base    │
              └──────────┘ └────────┘ └──────────┘ └────────┘ └──────────┘
                                                        │           │
                                                   ┌────┴────┐ ┌───┴───┐
                                                   │Projects │ │ChromaDB│
                                                   │ Writer  │ │Vectors │
                                                   └─────────┘ └───────┘
                                            │
                                            ▼
                                    Piper (TTS) ──► Speaker
```

---

## Requirements

- **Linux** (Ubuntu 22.04+, Debian 12+, Fedora 38+)
- **NVIDIA GPU** with 24GB+ VRAM for 70B models (8GB+ for 8B models)
- **Python 3.10+**
- **Ollama** installed and running
- **Docker** (optional, for SearXNG self-hosted search)

---

## Quick Start

```bash
# 1. Create and activate virtual environment
cd local-ai-assistant
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install \
    ollama httpx rich pydantic watchdog \
    duckduckgo-search readability-lxml beautifulsoup4 lxml \
    pynput pyperclip psutil python-xlib \
    sounddevice numpy \
    chromadb PyMuPDF

# 3. Install system tools
sudo apt install xdotool wmctrl xclip xdg-utils ffmpeg scrot

# 4. Install Ollama (if not already)
curl -fsSL https://ollama.ai/install.sh | sh

# 5. Pull models
ollama pull llama3.1:8b           # Fast model (~4GB)
ollama pull nomic-embed-text      # Embedding model (~275MB)
# ollama pull llama3.1:70b        # Full model (~40GB, optional)

# 6. Run
python main.py
```

---

## Usage

### CLI Mode (default)

```bash
python main.py
```

### Voice Mode

```bash
python main.py --voice
```

Requires additional packages: `pip install faster-whisper piper-tts openwakeword`

### File Watcher (auto-ingest on changes)

```bash
# In a separate terminal
python watcher.py ~/Documents ~/Desktop ~/LegalResearch
```

### As a System Service

```bash
sudo cp scripts/local-ai-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now local-ai-assistant
```

---

## Tool Categories (61 Functions)

### 1. File Management (10 tools)

```
list_directory     — List files with sizes and dates
read_file          — Read text file contents
write_file         — Create or overwrite files
search_files       — Find files by name or content (grep)
move_file          — Move or rename
copy_file          — Copy files or directories
delete_file        — Delete (with safety checks)
make_directory     — Create directories
file_info          — Detailed file metadata
disk_usage         — Disk space stats
```

### 2. Web Search (2 tools)

```
web_search         — Search via SearXNG or DuckDuckGo
fetch_webpage      — Fetch and extract text from any URL
```

### 3. Desktop Control (10 tools)

```
launch_app         — Open applications or URLs
list_windows       — Show all open windows
focus_window       — Bring window to front
close_window       — Close a window
run_command        — Execute shell commands (security-filtered)
clipboard_read     — Read clipboard
clipboard_write    — Copy to clipboard
screenshot         — Capture screen or window
send_keys          — Send keyboard shortcuts
type_text          — Type text into focused window
get_active_window  — Get active window info
```

**Security:** `run_command` uses a 3-tier filter:
- **Safe** (allowlisted): ls, cat, grep, git, python, firefox, etc. — run freely
- **Blocked**: rm -rf /, mkfs, curl|bash, reverse shells, privilege escalation — hard reject
- **Unknown**: anything else — runs with a warning

### 4. System Info (3 tools)

```
system_stats       — CPU, RAM, GPU, disk usage
list_processes     — Top processes by CPU/memory
network_info       — Interfaces, IPs, traffic stats
```

### 5. Knowledge Base (6 tools)

```
ingest_directory   — Scan and embed files into vector DB
ingest_file        — Add a single file
query_knowledge    — Semantic search with filters and re-ranking
knowledge_stats    — Database statistics
remove_source      — Remove files from the DB
list_sources       — List all indexed files
```

**Legal-aware features:**
- Statutes chunk on `§` section boundaries
- Case opinions chunk on structural markers (OPINION, DISSENT, FACTS)
- Auto-classifies document type: statute, case_law, legal_brief, news_article, statistics, code
- Auto-tags legal topics: civil_rights, excessive_force, qualified_immunity, search_seizure, etc.
- Auto-detects jurisdiction: federal, scotus, connecticut, federal_appellate
- Extracts citations from text
- Re-ranking option uses LLM to reorder results by actual relevance

**Filters in queries:**
```
> search knowledge base for excessive force, filter by statutes only
> query knowledge about qualified immunity, jurisdiction federal, rerank
```

### 6. Legal Research (16 tools)

```
lookup_statute             — Fetch statute text (federal or CT) from Cornell LII
search_case_law            — Search court opinions via CourtListener
fetch_court_opinion        — Download full opinion text
search_legal_statistics    — Find DOJ/FBI/BJS crime data
search_legal_news          — Search recent legal news
clip_article               — Save article with metadata and tags
download_resource          — Download PDFs/reports to library
download_statute_collection — Bulk download statute sets
generate_research_brief    — Create structured brief template
list_civil_rights_statutes — Quick reference list
list_research_files        — Browse your research library
compare_cases              — Side-by-side case comparison
create_project             — Start a legal research project
update_project             — Add notes, laws, facts to a project
list_projects              — View all projects
get_project                — Full project details
```

**Available statute collections:**
- `civil_rights` — 42 USC 1983, 1981, 1985, 18 USC 241/242, etc.
- `criminal_procedure` — Speedy Trial Act, Bail Reform, Sentencing, Habeas
- `evidence` — Federal Rules of Evidence key sections
- `police_accountability` — Section 1983, Pattern/Practice, 18 USC 242

### 7. Document Writer (4 tools)

```
write_document       — Create articles, essays, briefs, memos, reports
write_debate_prep    — Structured trial/debate preparation document
append_to_document   — Add content to existing documents
list_documents       — Browse generated documents
```

---

## Legal Research Workflow

### Starting a New Case

```
> create project traffic stop excessive force
> download the civil rights statute collection
> download the police accountability statute collection
> ingest directory ~/LegalResearch/federal_statutes
```

### Research Phase

```
> lookup statute 42 USC 1983
> search case law for excessive force traffic stop fourth amendment
> compare cases Graham v. Connor and Tennessee v. Garner
> search legal statistics on police use of force
> search legal news on police brutality settlements Connecticut
```

### Analysis Phase

```
> Does excessive force during a traffic stop violate constitutional rights?
```

The assistant will:
1. Look up relevant statutes (42 USC 1983, 4th/14th Amendments)
2. Search case law (Graham v. Connor, Terry v. Ohio)
3. Query the knowledge base for any ingested research
4. Respond with mandatory citation format:

```
APPLICABLE LAW:
- 42 U.S.C. § 1983 — Civil action for deprivation of rights
- Fourth Amendment — Unreasonable search and seizure

CASE LAW:
- Graham v. Connor, 490 U.S. 386 (1989) — Objective reasonableness standard
- Tennessee v. Garner, 471 U.S. 1 (1985) — Deadly force limits

APPLICATION:
[Analysis tying law + cases to the facts]

SOURCE FILES:
~/LegalResearch/federal_statutes/42_USC_1983.txt

Note: This is AI-generated legal research, not legal advice.
Verify all citations independently.
```

### Writing Phase

```
> argument mode
> build arguments for and against qualified immunity in this traffic stop case
> writing mode
> write an essay on Fourth Amendment violations in traffic stop encounters
> create a debate prep document for challenging qualified immunity
```

### Legal Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| Research | "research mode" | Raw sources only, no interpretation |
| Analysis | default | Apply law to facts, cite everything |
| Argument | "argument mode" | Plaintiff + defense arguments, cited |
| Writing | "writing mode" | Essays/briefs with Sources section |

---

## Hallucination Guardrails

This system enforces strict rules for legal output:

1. **Retrieval before reasoning** — Must search for sources before answering any legal question
2. **Mandatory citations** — Every legal claim must cite a statute or case
3. **Refusal on insufficient authority** — If no sources found, responds: *"Insufficient legal authority found to support a conclusion."*
4. **Disclaimer on every response** — Reminds the user this is research, not legal advice
5. **Source file references** — Points to local files where full text can be read

---

## File Structure

```
local-ai-assistant/
├── main.py                  # Entry point — CLI and voice loops
├── agent.py                 # Core agent — LLM + tool dispatch + system prompt
├── server.py                # Flask web dashboard server
├── config.py                # Configuration loader with defaults
├── config_utils.py          # Safe config writing (strips API keys)
├── cloud_reasoning.py       # Multi-provider cloud routing (Anthropic/OpenAI)
├── watcher.py               # File watcher — auto-ingest on changes
├── stats.py                 # Per-inquiry and session performance metrics
├── voice.py                 # Whisper STT + Piper TTS + wake word
├── config/
│   └── settings.json        # User configuration
├── templates/
│   └── dashboard.html       # Web dashboard front-end
├── tools/
│   ├── base.py              # Abstract base tool class
│   ├── file_manager.py      # File operations (10 tools)
│   ├── web_search.py        # Web search + page fetching (2 tools)
│   ├── desktop_control.py   # Desktop/app/command control (10 tools)
│   ├── system_info.py       # Hardware/process monitoring (3 tools)
│   ├── knowledge_base.py    # RAG vector database (6 tools)
│   ├── legal_research.py    # Legal research engine (16 tools)
│   └── document_writer.py   # Document generation (4 tools)
├── scripts/
│   ├── setup.sh             # Full install script
│   ├── setup_knowledge_base.sh
│   ├── install.sh           # File placement helper for fresh installs
│   ├── local-ai-assistant.service  # systemd service unit
│   ├── bulk_download.py     # Bulk legal content downloader (v1)
│   ├── bulk_download_v2.py  # Bulk legal content downloader (v2, fixed URLs)
│   ├── fine_tune_generator.py      # OpenAI fine-tuning JSONL generator
│   └── patches/             # One-shot migration scripts (already applied)
│       ├── patch_cloud_chat.py
│       └── patch_cloud_dashboard.py
└── searxng/
    ├── docker-compose.yml
    └── settings.yml
```

### Data Directories (created automatically)

```
~/LegalResearch/
├── federal_statutes/        # Downloaded US Code sections
├── state_statutes/          # Connecticut General Statutes
├── case_law/                # Court opinions
├── statistics/              # DOJ/FBI/BJS data
├── news_clips/              # Clipped articles with metadata
├── research_briefs/         # Generated briefs and comparisons
├── projects/                # Ongoing research projects
│   └── [project_name]/
│       ├── project.json     # Facts, laws, notes, status
│       └── notes.md         # Running notes log
└── documents/               # Generated essays, articles, debate prep

~/.local/share/ai-assistant/
└── knowledge_db/
    ├── chroma/              # ChromaDB vector storage
    └── manifest.json        # Ingestion tracking
```

---

## Configuration

Edit `config/settings.json`:

```json
{
    "llm": {
        "primary_model": "llama3.1:70b",
        "fast_model": "llama3.1:8b",
        "ollama_host": "http://localhost:11434",
        "temperature": 0.3,
        "context_window": 8192
    },
    "search": {
        "searxng_url": "http://localhost:8888",
        "fallback_to_ddg": true,
        "max_results": 5
    },
    "voice": {
        "enabled": false,
        "wake_word": "computer",
        "stt_model": "base.en",
        "tts_voice": "en_US-lessac-medium",
        "push_to_talk_key": "ctrl+space"
    },
    "knowledge_base": {
        "db_path": "~/.local/share/ai-assistant/knowledge_db",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embed_model": "nomic-embed-text",
        "max_results": 5
    }
}
```

## Model Recommendations

| VRAM | Model | Speed | Quality |
|------|-------|-------|---------|
| 24GB+ | `llama3.1:70b` (Q4) | Slow | Excellent reasoning and tool use |
| 24GB | `qwen2.5:32b` | Medium | Great tool calling |
| 16GB | `llama3.1:8b` | Fast | Good for most tasks |
| 12GB | `mistral:7b` | Fast | Lightweight option |

---

## Adding New Tools

Create a file in `tools/` inheriting from `BaseTool`:

```python
from tools.base import BaseTool
from typing import Callable

class MyTool(BaseTool):
    def get_tool_definitions(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "my_function",
                "description": "What this does — be specific so the LLM knows when to use it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "..."}
                    },
                    "required": ["arg1"]
                }
            }
        }]

    def get_handlers(self) -> dict[str, Callable]:
        return {"my_function": self.my_function}

    def my_function(self, arg1: str) -> str:
        return "result"
```

Register in `agent.py`:
```python
from tools.my_tool import MyTool
# In __init__:
self.tool_instances["my_tool"] = MyTool(config)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama` not responding | `sudo systemctl restart ollama` or `ollama serve` in another terminal |
| Out of VRAM | Change `primary_model` to `llama3.1:8b` in config |
| "Thinking..." hangs | Ingestion running on too many files — Ctrl+C and target specific directories |
| SearXNG not running | `cd searxng && docker-compose up -d` (or just use DDG fallback) |
| No sound / TTS | `aplay -l` to check audio devices, install `espeak` as fallback |
| xdotool errors | Only works on X11. For Wayland, replace with `ydotool` |
| Watcher crashes | Pass only existing directories: `python watcher.py ~/Documents` |
| apt mirror errors | Ubuntu mirror syncing — wait an hour or use `--fix-missing` |
| Docker conflicts | `sudo apt remove containerd && sudo apt install docker.io` |

---

## Roadmap

- [x] Core agent with tool-calling loop
- [x] File management (10 tools)
- [x] Web search with SearXNG + DDG fallback
- [x] Desktop control with security layer
- [x] RAG knowledge base with ChromaDB
- [x] Legal-aware chunking and auto-tagging
- [x] Legal research engine (statutes, case law, news, stats)
- [x] Case comparison engine
- [x] Project management for ongoing cases
- [x] Document writer (essays, briefs, debate prep)
- [x] Citation enforcement and hallucination guardrails
- [x] LLM re-ranking for search precision
- [x] Voice interface (Whisper + Piper + wake word)
- [ ] Auto-ingest legal updates on a daily schedule
- [ ] Argument tree builder (pro vs. defense with cited authority)
- [ ] Timeline reconstruction for cases
- [ ] Deployable product packaging under AXMH

---

*Built for [Axis Meridian Holdings](https://axismh.com) — fully local, fully yours.*