"""Configuration loader — loads settings.json + .env for secrets."""

import json
import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed — fall back to os.environ

DEFAULT_CONFIG = {
    "llm": {
        "primary_model": "qwen2.5:14b",
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
    },
    "features": {
        "autonomous_collection": False,
        "auto_ingest": False,
        "strict_legal_mode": True,
        "allow_external_downloads": False
    }
}


def _inject_env_secrets(config: dict) -> dict:
    """
    Inject API keys from environment variables into the config.
    Environment variables ALWAYS override settings.json values for secrets.
    This prevents keys from ever needing to be in settings.json.
    """
    if "cloud" not in config:
        config["cloud"] = {}

    cloud = config["cloud"]

    # API keys — env vars override config file
    env_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
    env_openai = os.environ.get("OPENAI_API_KEY", "")

    if env_anthropic:
        cloud["anthropic_api_key"] = env_anthropic
    if env_openai:
        cloud["openai_api_key"] = env_openai

    # Optional overrides from env
    env_overrides = {
        "ANTHROPIC_MODEL": "anthropic_model",
        "OPENAI_MODEL": "openai_model",
        "CLOUD_PROVIDER": "provider",
    }
    for env_key, config_key in env_overrides.items():
        val = os.environ.get(env_key)
        if val:
            cloud[config_key] = val

    # Boolean overrides
    for env_key, config_key in [("CLOUD_ENABLED", "enabled"), ("CLOUD_AUTO_ROUTE", "auto_route")]:
        val = os.environ.get(env_key)
        if val is not None:
            cloud[config_key] = val.lower() in ("true", "1", "yes")

    # Numeric overrides
    budget = os.environ.get("CLOUD_MONTHLY_BUDGET")
    if budget:
        try:
            cloud["max_monthly_budget"] = float(budget)
        except ValueError:
            pass

    # Set defaults for cloud config if not present
    cloud.setdefault("provider", "anthropic")
    cloud.setdefault("anthropic_model", "claude-opus-4-6")
    cloud.setdefault("openai_model", "gpt-4o")
    cloud.setdefault("enabled", True)
    cloud.setdefault("auto_route", True)
    cloud.setdefault("cloud_first", True)
    cloud.setdefault("max_monthly_budget", 60.0)
    cloud.setdefault("monthly_spend", 0.0)

    return config


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
    else:
        merged = DEFAULT_CONFIG.copy()

    # Inject secrets from .env / environment variables
    merged = _inject_env_secrets(merged)

    return merged
