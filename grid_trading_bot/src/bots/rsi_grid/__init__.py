"""
RSI-Grid Hybrid Bot Module.

A futures trading bot combining RSI zone filtering with Grid entry mechanism.

Strategy Logic:
- RSI Zone Filter: Oversold=LONG only, Overbought=SHORT only, Neutral=follow trend
- Trend Filter: SMA-based trend direction
- Grid Entry: ATR-based dynamic grid levels
- Risk Management: ATR-based stop loss, grid-based take profit

Design Goals:
- Target Sharpe > 3.0
- Walk-Forward Consistency > 90%
- Win Rate > 70%
- Max Drawdown < 5%
"""

from .bot import RSIGridBot
from .indicators import RSICalculator, ATRCalculator, SMACalculator, RSIResult, ATRResult
from .models import (
    RSIGridConfig,
    RSIZone,
    PositionSide,
    GridLevel,
    GridLevelState,
    GridSetup,
    RSIGridPosition,
    RSIGridTrade,
    RSIGridStats,
    ExitReason,
)

__all__ = [
    # Bot
    "RSIGridBot",
    # Indicators
    "RSICalculator",
    "ATRCalculator",
    "SMACalculator",
    "RSIResult",
    "ATRResult",
    # Models
    "RSIGridConfig",
    "RSIZone",
    "PositionSide",
    "GridLevel",
    "GridLevelState",
    "GridSetup",
    "RSIGridPosition",
    "RSIGridTrade",
    "RSIGridStats",
    "ExitReason",
]
