"""
Utility: Strip API keys from config dict before writing to settings.json.
Import and use in server.py anywhere config is saved to disk.
"""

import json
from pathlib import Path

# Keys that should NEVER be written to settings.json
SECRET_KEYS = {"anthropic_api_key", "openai_api_key"}


def save_config_safe(config: dict, config_path: Path = None):
    """
    Save config to settings.json with all API keys stripped out.
    Keys live in .env only — they get injected at runtime by config.py.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "settings.json"

    config_path.parent.mkdir(exist_ok=True)

    # Deep copy the cloud section and strip secrets
    safe_config = {}
    for section, values in config.items():
        if section == "cloud" and isinstance(values, dict):
            safe_cloud = {k: v for k, v in values.items() if k not in SECRET_KEYS}
            safe_config["cloud"] = safe_cloud
        else:
            safe_config[section] = values

    with open(config_path, "w") as f:
        json.dump(safe_config, f, indent=4)
