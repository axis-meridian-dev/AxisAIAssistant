# Local AI Assistant

A fully local, privacy-first AI assistant for Linux with file management, web research, desktop control, and voice interaction.

## Architecture

```
Microphone → Whisper (STT) → Ollama (LLM + tool routing) → Action Layer → Piper (TTS) → Speaker
                                                              ↓
                                              ┌───────────────┼───────────────┐
                                              │               │               │
                                         Desktop Ctrl    File System     Web Research
                                        (xdotool/dbus)  (read/write)   (searxng/ddg)
```

## Requirements

- **Linux** (Ubuntu 22.04+, Debian 12+, Fedora 38+)
- **NVIDIA GPU** with 24GB+ VRAM (for 70B models)
- **Docker** (for SearXNG self-hosted search)
- **Python 3.10+**

## Quick Start

```bash
# 1. Clone or copy this project
cd local-ai-assistant

# 2. Run setup (installs everything)
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Activate environment and run
source .venv/bin/activate
python main.py
```

## Usage

### CLI Mode (default)
```bash
python main.py
```

Example commands:
- `list files in ~/Documents sorted by size`
- `search for python files containing "async"`
- `what's using the most CPU right now?`
- `search the web for latest linux kernel release`
- `open firefox and go to github.com`
- `take a screenshot`
- `copy this text to clipboard: hello world`

### Voice Mode
```bash
python main.py --voice
```

- **Wake word**: Say "computer" to activate
- **Push-to-talk**: Hold `Ctrl+Space`
- Configure in `config/settings.json`

### As a System Service
```bash
sudo cp scripts/local-ai-assistant.service /etc/systemd/system/
sudo systemctl enable --now local-ai-assistant
```

## Configuration

Edit `config/settings.json`:

```json
{
    "llm": {
        "primary_model": "llama3.1:70b",    // Main reasoning model
        "fast_model": "llama3.1:8b",         // Quick tasks / routing
        "temperature": 0.3
    },
    "voice": {
        "enabled": false,
        "wake_word": "computer",
        "push_to_talk_key": "ctrl+space"
    }
}
```

## Project Structure

```
local-ai-assistant/
├── main.py              # Entry point + CLI loop
├── agent.py             # Core agent (LLM + tool dispatch)
├── config.py            # Config loader
├── voice.py             # Voice interface (Whisper + Piper)
├── config/
│   └── settings.json    # User configuration
├── tools/
│   ├── base.py          # Base tool class
│   ├── file_manager.py  # File ops (read/write/search/organize)
│   ├── web_search.py    # Web search (SearXNG + DDG)
│   ├── desktop_control.py  # App/window/clipboard/command control
│   └── system_info.py   # Hardware stats, processes, network
├── searxng/
│   ├── docker-compose.yml
│   └── settings.yml
└── scripts/
    ├── setup.sh         # Full install script
    └── local-ai-assistant.service  # systemd unit
```

## Adding New Tools

Create a new file in `tools/` inheriting from `BaseTool`:

```python
from tools.base import BaseTool

class MyTool(BaseTool):
    def get_tool_definitions(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "my_function",
                "description": "What this does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "..."}
                    },
                    "required": ["arg1"]
                }
            }
        }]

    def get_handlers(self) -> dict:
        return {"my_function": self.my_function}

    def my_function(self, arg1: str) -> str:
        return "result"
```

Then register it in `agent.py`:
```python
from tools.my_tool import MyTool
# In __init__:
self.tool_instances["my_tool"] = MyTool(config)
```

## Model Recommendations

| VRAM | Model | Notes |
|------|-------|-------|
| 24GB | `llama3.1:70b` (Q4) | Best reasoning, tight fit |
| 24GB | `qwen2.5:32b` | Great tool calling |
| 16GB | `llama3.1:8b` | Fast, capable |
| 12GB | `mistral:7b` | Lightweight |
| 8GB  | `phi3:3.8b` | Minimum viable |

## Troubleshooting

**Ollama not responding**: `sudo systemctl restart ollama`
**Out of VRAM**: Switch to 8B model in config
**SearXNG not running**: `cd searxng && docker-compose up -d`
**No sound**: Check `aplay -l` for audio devices, install `pulseaudio-utils`
**xdotool errors**: Only works on X11, not Wayland. For Wayland, use `ydotool`
