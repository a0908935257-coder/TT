"""
Unified Backtest Framework.

A modular, extensible backtesting framework that consolidates
scattered backtesting logic across strategies.

Core Components:
- BacktestEngine: Main orchestrator for running backtests
- BacktestConfig: Configuration for backtest parameters
- BacktestResult: Comprehensive result model with metrics
- BacktestStrategy: Abstract base class for strategy adapters
- PositionManager: Position and trade tracking
- OrderSimulator: Order matching simulation
- MetricsCalculator: Performance metrics calculation
"""

from .config import BacktestConfig
from .result import BacktestResult, Trade, WalkForwardResult, WalkForwardPeriod
from .position import PositionManager, Position
from .order import OrderSimulator, Signal, SignalType
from .metrics import MetricsCalculator
from .strategy.base import BacktestStrategy, BacktestContext
from .engine import BacktestEngine

__all__ = [
    # Config
    "BacktestConfig",
    # Results
    "BacktestResult",
    "Trade",
    "WalkForwardResult",
    "WalkForwardPeriod",
    # Position
    "PositionManager",
    "Position",
    # Order
    "OrderSimulator",
    "Signal",
    "SignalType",
    # Metrics
    "MetricsCalculator",
    # Strategy
    "BacktestStrategy",
    "BacktestContext",
    # Engine
    "BacktestEngine",
]
