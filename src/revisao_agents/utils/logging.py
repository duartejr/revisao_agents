"""
logging.py - Structured logging configuration for revisao_agents.

Uses Python's standard logging with rich formatting for the terminal.
Import `get_logger` wherever you need a logger:

    from ..utils.logging import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


@lru_cache(maxsize=None)
def get_logger(name: str = "revisao_agents", level: str = "INFO") -> logging.Logger:
    """
    Return a configured logger. Results are cached so the same logger
    is returned for repeated calls with the same *name*.

    Args:
        name:  Logger name (typically ``__name__`` of the caller).
        level: Minimum log level string ("DEBUG", "INFO", "WARNING", …).

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — return as-is
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger
