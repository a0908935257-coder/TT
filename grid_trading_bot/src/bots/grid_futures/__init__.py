"""
Grid Futures Bot Module.

Provides futures-based grid trading with leverage and trend-following.
"""

from .bot import GridFuturesBot
from .models import (
    GridFuturesConfig,
    GridDirection,
    GridLevel,
    GridLevelState,
    GridSetup,
    FuturesPosition,
    PositionSide,
    GridTrade,
    GridFuturesStats,
    ExitReason,
)

__all__ = [
    "GridFuturesBot",
    "GridFuturesConfig",
    "GridDirection",
    "GridLevel",
    "GridLevelState",
    "GridSetup",
    "FuturesPosition",
    "PositionSide",
    "GridTrade",
    "GridFuturesStats",
    "ExitReason",
]
