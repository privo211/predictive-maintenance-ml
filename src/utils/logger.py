"""Structured logging for the Predictive Maintenance Platform.

Configures JSON-formatted logs for production and colorised console output for
local development.  All platform code should obtain loggers through
:func:`get_logger` to pick up the configured format automatically.
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _StructuredFormatter(logging.Formatter):
    """Emits log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_CONSOLE_FMT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    *,
    log_format: str = "json",
) -> None:
    """Configure the root logger for the entire platform.

    Parameters
    ----------
    level:
        Python log-level name (``DEBUG``, ``INFO``, ...).
    log_file:
        When set, a rotating file handler is attached in addition to the
        console handler.
    log_format:
        ``"json"`` for production / structured-logging ingest; ``"console"``
        for human-readable colourised output.
    """

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # --- Console handler ------------------------------------------------
    console = logging.StreamHandler(sys.stderr)
    if log_format == "json":
        console.setFormatter(_StructuredFormatter())
    else:
        console.setFormatter(
            logging.Formatter(_CONSOLE_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        )
    root.addHandler(console)

    # --- Optional rotating file handler ---------------------------------
    if log_file:
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MiB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(_StructuredFormatter())
        root.addHandler(handler)

    # Suppress overly-verbose third-party loggers
    for noisy in ("uvicorn", "uvicorn.access", "sqlalchemy.engine", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the root with the configured format.

    Use in every module::

        from src.utils.logger import get_logger

        _logger = get_logger(__name__)
    """

    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Auto-configure on first import when settings are available
# ---------------------------------------------------------------------------

_configured = False


def _auto_configure() -> None:
    global _configured

    if _configured:
        return

    try:
        from config.settings import settings

        level = settings.log_level
        fmt = settings.log_format
    except (ImportError, AttributeError):
        level = os.getenv("PMP_LOG_LEVEL", "INFO")
        fmt = os.getenv("PMP_LOG_FORMAT", "json")

    setup_logging(level=level, log_format=fmt)
    _configured = True
