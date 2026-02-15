"""
logging_utils.py

Centralized logging setup for this project.

Goals:
- Use stdlib `logging` (no extra deps).
- Keep tqdm progress bars clean by routing console logs via `tqdm.write(...)` when enabled.
- Allow controlling verbosity via env vars.

Env vars:
- LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default: INFO)
- LOG_FORMAT: plain|json (default: plain)
- LOG_FILE: optional file path to also write logs to
- LOG_USE_TQDM: 1|0 (default: 1)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_log_level(value: str) -> int:
    if not value:
        return logging.INFO
    upper = value.strip().upper()
    return getattr(logging, upper, logging.INFO)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TqdmLoggingHandler(logging.StreamHandler):
    """A console handler that writes via tqdm to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm  # Imported lazily to keep optional.

                tqdm.write(msg, file=self.stream)
            except Exception:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    use_tqdm: Optional[bool] = None,
    force: bool = False,
) -> logging.Logger:
    """
    Configure root logger.

    This function is safe to call multiple times. Use `force=True` to replace existing handlers.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_format = (log_format or os.getenv("LOG_FORMAT", "plain")).strip().lower()
    log_file = log_file or os.getenv("LOG_FILE")
    if use_tqdm is None:
        use_tqdm = _parse_bool_env("LOG_USE_TQDM", True)

    root = logging.getLogger()
    if getattr(root, "_magic_split_logging_configured", False) and not force:
        return root

    if force:
        for handler in list(root.handlers):
            # Close handlers to avoid leaking file descriptors in long-lived processes.
            handler.close()
            root.removeHandler(handler)

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if use_tqdm:
        console_handler: logging.Handler = TqdmLoggingHandler(stream=sys.stdout)
    else:
        console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(_parse_log_level(level))

    root.setLevel(_parse_log_level(level))
    root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_parse_log_level(level))
        root.addHandler(file_handler)

    root._magic_split_logging_configured = True  # type: ignore[attr-defined]
    return root
