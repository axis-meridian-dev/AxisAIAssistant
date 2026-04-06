"""Centralized logging setup for the AI assistant."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path.home() / ".local" / "share" / "ai-assistant" / "logs"
LOG_FILE = LOG_DIR / "assistant.log"
MAX_BYTES = 10_000_000  # 10 MB
BACKUP_COUNT = 5


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root application logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_assistant")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stderr handler for warnings and above only (don't pollute Rich output)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
