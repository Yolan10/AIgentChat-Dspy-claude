"""Structured JSON logging utilities with performance tracking."""
from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

import utils


class StructuredLogger:
    """A simple structured logger that writes JSON lines."""

    def __init__(self, logfile: str = "logs/system.log", max_bytes: int = 1048576, backup_count: int = 5) -> None:
        utils.ensure_logs_dir()
        self.logger = logging.getLogger("structured")
        self.logger.handlers.clear()
        handler = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(self, name: str, **data: Any) -> None:
        entry: Dict[str, Any] = {"event": name, "ts": time.time()}
        entry.update(data)
        self.logger.info(json.dumps(entry))
