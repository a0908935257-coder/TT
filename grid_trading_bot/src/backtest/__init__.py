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

Enhanced Components (v2.0):
- Slippage Models: Fixed, Volatility-based, Market Impact
- Fee Calculators: Fixed, Maker/Taker, Tiered
- Order Book: Limit orders, Stop orders, Partial fills
- Market Microstructure: Spread simulation, Gap handling

Strategy Layer Components (v2.1):
- Parameter Optimization: Grid, Random, Bayesian (Optuna), Genetic Algorithm
- Walk-Forward Optimization: Overfitting detection
- Strategy Versioning: Serialization, metadata, snapshots
- Experiment Tracking: Result persistence, comparison reports
- Monte Carlo Simulation: Robustness validation, confidence intervals
"""

# Config
from .config import (
    BacktestConfig,
    SlippageModelType,
    FeeModelType,
    IntraBarSequence,
)

# Results
from .result import BacktestResult, Trade, WalkForwardResult, WalkForwardPeriod

# Position
from .position import PositionManager, Position

# Order (core)
from .order import (
    OrderSimulator,
    Signal,
    SignalType,
    # New order types
    OrderType,
    OrderSide,
    OrderTimeInForce,
    OrderStatus,
    PendingOrder,
    Fill,
    OrderBook,
)

# Slippage Models
from .slippage import (
    SlippageModel,
    SlippageContext,
    FixedSlippage,
    VolatilityBasedSlippage,
    MarketImpactSlippage,
    create_slippage_model,
)

# Fee Calculators
from .fees import (
    FeeCalculator,
    FeeContext,
    FeeTier,
    FixedFeeCalculator,
    MakerTakerFeeCalculator,
    TieredFeeCalculator,
    create_fee_calculator,
)

# Market Microstructure
from .microstructure import (
    MarketMicrostructure,
    SpreadContext,
    PriceSequence,
    create_microstructure_from_config,
)

# Optimizer
from .optimizer import (
    Optimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    WalkForwardOptimizer,
    ParameterSpace,
    OptimizationTrial,
    OptimizationResult,
    OptimizationMethod,
    OptimizationDirection,
    create_optimizer,
    is_optuna_available,
)

# Versioning
from .versioning import (
    StrategySerializer,
    VersionManager,
    StrategyMetadata,
    StrategySnapshot,
    SerializationFormat,
    is_yaml_available,
)

# Git Integration
from .git_integration import (
    GitVersionManager,
    GitClient,
    GitCommit,
    GitTag,
    GitBranch,
    GitStatus,
    GitError,
    create_git_version_manager,
)

# Experiment Tracking
from .experiment import (
    ExperimentTracker,
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
)

# Metrics
from .metrics import MetricsCalculator

# Monte Carlo
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloMethod,
    MonteCarloDistribution,
    ConfidenceInterval,
    run_monte_carlo_validation,
    generate_monte_carlo_report,
)

# Strategy
from .strategy.base import BacktestStrategy, BacktestContext

# Engine
from .engine import BacktestEngine

# Multi-Timeframe
from .multi_timeframe import (
    MultiTimeframeEngine,
    MultiTimeframeStrategy,
    MultiTimeframeContext,
    TimeframeData,
    TimeframeResampler,
    Timeframe,
    get_timeframe_minutes,
    can_resample,
)

__all__ = [
    # Config
    "BacktestConfig",
    "SlippageModelType",
    "FeeModelType",
    "IntraBarSequence",
    # Results
    "BacktestResult",
    "Trade",
    "WalkForwardResult",
    "WalkForwardPeriod",
    # Position
    "PositionManager",
    "Position",
    # Order (core)
    "OrderSimulator",
    "Signal",
    "SignalType",
    # Order (advanced)
    "OrderType",
    "OrderSide",
    "OrderTimeInForce",
    "OrderStatus",
    "PendingOrder",
    "Fill",
    "OrderBook",
    # Slippage Models
    "SlippageModel",
    "SlippageContext",
    "FixedSlippage",
    "VolatilityBasedSlippage",
    "MarketImpactSlippage",
    "create_slippage_model",
    # Fee Calculators
    "FeeCalculator",
    "FeeContext",
    "FeeTier",
    "FixedFeeCalculator",
    "MakerTakerFeeCalculator",
    "TieredFeeCalculator",
    "create_fee_calculator",
    # Market Microstructure
    "MarketMicrostructure",
    "SpreadContext",
    "PriceSequence",
    "create_microstructure_from_config",
    # Optimizer
    "Optimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "WalkForwardOptimizer",
    "ParameterSpace",
    "OptimizationTrial",
    "OptimizationResult",
    "OptimizationMethod",
    "OptimizationDirection",
    "create_optimizer",
    "is_optuna_available",
    # Versioning
    "StrategySerializer",
    "VersionManager",
    "StrategyMetadata",
    "StrategySnapshot",
    "SerializationFormat",
    "is_yaml_available",
    # Git Integration
    "GitVersionManager",
    "GitClient",
    "GitCommit",
    "GitTag",
    "GitBranch",
    "GitStatus",
    "GitError",
    "create_git_version_manager",
    # Experiment Tracking
    "ExperimentTracker",
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    # Metrics
    "MetricsCalculator",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MonteCarloMethod",
    "MonteCarloDistribution",
    "ConfidenceInterval",
    "run_monte_carlo_validation",
    "generate_monte_carlo_report",
    # Strategy
    "BacktestStrategy",
    "BacktestContext",
    # Engine
    "BacktestEngine",
    # Multi-Timeframe
    "MultiTimeframeEngine",
    "MultiTimeframeStrategy",
    "MultiTimeframeContext",
    "TimeframeData",
    "TimeframeResampler",
    "Timeframe",
    "get_timeframe_minutes",
    "can_resample",
]
