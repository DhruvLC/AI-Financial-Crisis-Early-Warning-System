"""Centralized logging setup for the ingestion module."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO", logfile: str | None = None) -> None:
    """Idempotently configure root logging to console + optional rotating file."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
