"""
Fund Manager Storage.

Persistence layer for fund allocation records.
"""

from .repository import AllocationRepository, DEFAULT_DB_PATH

__all__ = ["AllocationRepository", "DEFAULT_DB_PATH"]
