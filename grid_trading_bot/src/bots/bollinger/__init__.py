"""
Bollinger Band Mean Reversion Bot.

A futures trading bot that uses Bollinger Bands for mean reversion strategy.
Enters positions when price touches bands and exits when price returns to middle.

Features:
- Bollinger Bands calculation with configurable period and multiplier
- BBW (Bollinger Band Width) filter to avoid squeeze conditions
- Futures trading with leverage support
- Limit orders for entry/take-profit (lower fees)
- Stop market orders for stop-loss (guaranteed execution)
"""

from .models import (
    BBWData,
    BollingerBands,
    BollingerBotStats,
    BollingerConfig,
    ExitReason,
    Position,
    PositionSide,
    Signal,
    SignalType,
    TradeRecord,
)
from .indicators import (
    BandPosition,
    BollingerCalculator,
    InsufficientDataError,
)
from .signal_generator import SignalGenerator

__all__ = [
    # Models
    "BollingerConfig",
    "BollingerBands",
    "BBWData",
    "SignalType",
    "Signal",
    "PositionSide",
    "Position",
    "TradeRecord",
    "BollingerBotStats",
    "ExitReason",
    # Indicators
    "BollingerCalculator",
    "BandPosition",
    "InsufficientDataError",
    # Signal Generator
    "SignalGenerator",
]
