"""
Logging system for Grid Trading Bot.

Provides colored terminal output and rotating file logs.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    GRAY = "\033[90m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"


# Color mapping for log levels
LEVEL_COLORS = {
    logging.DEBUG: Colors.GRAY,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.RED + Colors.BOLD,
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to terminal output."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Get color for this level
        color = LEVEL_COLORS.get(record.levelno, Colors.RESET)

        # Format the message
        message = super().format(record)

        # Add color
        return f"{color}{message}{Colors.RESET}"


class PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no colors)."""
    pass


def setup_logger(
    name: str,
    level: int | str | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name, typically module name like "exchange.binance"
        level: Log level. Defaults to LOG_LEVEL env var or INFO
        log_file: Log file path. Defaults to logs/trading.log

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("exchange.binance")
        >>> logger.info("Connected to Binance")
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Determine log level
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Terminal handler (colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(log_format, date_format))
    logger.addHandler(console_handler)

    # File handler (rotating)
    if log_file is None:
        # Default log file path
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "trading.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(PlainFormatter(log_format, date_format))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    If the logger doesn't exist, creates a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("trading.strategy")
        >>> logger.info("Strategy initialized")
    """
    logger = logging.getLogger(name)

    # If no handlers, set up the logger
    if not logger.handlers:
        return setup_logger(name)

    return logger
