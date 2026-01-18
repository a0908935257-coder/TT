"""
Core module for Trading Master.

Provides logging utilities.
"""

from .logger import setup_logger, get_logger

__all__ = [
    "setup_logger",
    "get_logger",
]
