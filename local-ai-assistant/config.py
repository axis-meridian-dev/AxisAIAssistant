""""Configuration loader."""

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "llm": {
        "primary_model": "llama3.1:70b",
        "fast_model": "llama3.1:8b",
        "ollama_host": "http://localhost:11434",
        "temperature": 0.3,
        "context_window": 8192
    },
    "search": {
        "searxng_url": "http://localhost:8888",
        "fallback_to_ddg": True,
        "max_results": 5
    },
    "voice": {
        "enabled": False,
        "wake_word": "computer",
        "stt_model": "base.en",
        "tts_voice": "en_US-lessac-medium",
        "push_to_talk_key": "ctrl+space"
    },
    "files": {
        "allowed_roots": ["~"],
        "excluded_dirs": [".git", "node_modules", "__pycache__", ".cache"],
        "max_file_size_mb": 50
    },
    "desktop": {
        "screenshot_enabled": True,
        "app_launch_enabled": True,
        "clipboard_enabled": True
    },
    "knowledge_base": {
        "db_path": "~/.local/share/ai-assistant/knowledge_db",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embed_model": "nomic-embed-text",
        "max_results": 5,
        "auto_watch_dirs": ["~/Documents", "~/Projects", "~/Desktop"]
    }
}


def load_config() -> dict:
    config_path = Path(__file__).parent / "config" / "settings.json"
    if config_path.exists():
        with open(config_path) as f:
            user_cfg = json.load(f)
        # Merge with defaults
        merged = DEFAULT_CONFIG.copy()
        for section, values in user_cfg.items():
            if section in merged and isinstance(merged[section], dict):
                merged[section].update(values)
            else:
                merged[section] = values
        return merged
    return DEFAULT_CONFIG.copy()