"""
Strategy Module.

Provides base classes and adapters for backtest strategies.
"""

from .base import BacktestStrategy, BacktestContext
from .bollinger import BollingerBacktestStrategy, BollingerStrategyConfig
from .supertrend import SupertrendBacktestStrategy, SupertrendStrategyConfig
from .grid import GridBacktestStrategy, GridStrategyConfig, MultiLevelGridStrategy

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
    # Grid
    "GridBacktestStrategy",
    "GridStrategyConfig",
    "MultiLevelGridStrategy",
    # DSL
    "DSLParser",
    "DSLParseError",
    "DSLValidator",
    "DSLValidationError",
    "DSLStrategyGenerator",
    "DSLGeneratedStrategy",
    "StrategyDefinition",
]
