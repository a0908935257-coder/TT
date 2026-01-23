"""
Strategy Module.

Provides base classes and adapters for backtest strategies.
"""

from .base import BacktestStrategy, BacktestContext
from .bollinger import BollingerBacktestStrategy, BollingerStrategyConfig

__all__ = [
    "BacktestStrategy",
    "BacktestContext",
    "BollingerBacktestStrategy",
    "BollingerStrategyConfig",
]
