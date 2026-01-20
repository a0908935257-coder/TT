"""
Bollinger Band Trading Bot.

A futures trading bot that uses Bollinger Bands for trading.
Supports two strategy modes:
- MEAN_REVERSION: Enters positions when price touches bands, exits when returns to middle
- BREAKOUT: Enters positions when price breaks through bands, follows the trend

Features:
- Bollinger Bands calculation with configurable period and multiplier
- BBW (Bollinger Band Width) filter
- Futures trading with leverage support
- Trailing stop for breakout mode
- ATR-based dynamic stop loss
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
    StrategyMode,
    TradeRecord,
)
from .indicators import (
    BandPosition,
    BollingerCalculator,
    InsufficientDataError,
)
from .signal_generator import SignalGenerator
from .position_manager import PositionManager, PositionExistsError, NoPositionError
from .order_executor import OrderExecutor, OrderNotFoundError
from .bot import BollingerBot

__all__ = [
    # Models
    "BollingerConfig",
    "BollingerBands",
    "BBWData",
    "SignalType",
    "StrategyMode",
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
    # Position Manager
    "PositionManager",
    "PositionExistsError",
    "NoPositionError",
    # Order Executor
    "OrderExecutor",
    "OrderNotFoundError",
    # Bot
    "BollingerBot",
]
