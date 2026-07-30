"""
app/logger.py
─────────────
Centralized structured logging configuration for AgriPrice Sentinel.
Replaces ad-hoc `logging.basicConfig()` calls throughout the codebase.

Usage:
    from app.logger import get_logger
    logger = get_logger(__name__)
"""

import sys
import logging
from app.config import get_settings


_LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging() -> None:
    """Configure root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpcore", "httpx", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring the root config has been applied."""
    setup_logging()
    return logging.getLogger(name)
