"""
Fund Manager Module.

Provides centralized fund allocation and distribution across trading bots.
Monitors total capital pool and automatically allocates funds based on
configured strategies.
"""

from .manager import FundManager

__all__ = ["FundManager"]
