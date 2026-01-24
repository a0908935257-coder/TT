"""
Strategy Module.

Provides base classes and adapters for backtest strategies.
"""

from .base import BacktestStrategy, BacktestContext
from .bollinger import BollingerBacktestStrategy, BollingerStrategyConfig
from .supertrend import SupertrendBacktestStrategy, SupertrendStrategyConfig
from .grid import GridBacktestStrategy, GridStrategyConfig, MultiLevelGridStrategy
from .grid_futures import GridFuturesBacktestStrategy, GridFuturesStrategyConfig, GridDirection

# Multi-Timeframe
from .multi_tf_trend import MultiTFTrendStrategy, MultiTFTrendConfig

# DSL Strategy Language
from .dsl import (
    DSLParser,
    DSLParseError,
    DSLValidator,
    DSLValidationError,
    DSLStrategyGenerator,
    DSLGeneratedStrategy,
    StrategyDefinition,
)

__all__ = [
    # Base
    "BacktestStrategy",
    "BacktestContext",
    # Bollinger
    "BollingerBacktestStrategy",
    "BollingerStrategyConfig",
    # Supertrend
    "SupertrendBacktestStrategy",
    "SupertrendStrategyConfig",
    # Grid (Spot)
    "GridBacktestStrategy",
    "GridStrategyConfig",
    "MultiLevelGridStrategy",
    # Grid Futures
    "GridFuturesBacktestStrategy",
    "GridFuturesStrategyConfig",
    "GridDirection",
    # Multi-Timeframe
    "MultiTFTrendStrategy",
    "MultiTFTrendConfig",
    # DSL
    "DSLParser",
    "DSLParseError",
    "DSLValidator",
    "DSLValidationError",
    "DSLStrategyGenerator",
    "DSLGeneratedStrategy",
    "StrategyDefinition",
]
