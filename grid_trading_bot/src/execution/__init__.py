"""
Execution Layer Module.

Provides order execution infrastructure:
- Order routing and algorithm selection
- Order splitting (TWAP, VWAP, Iceberg)
- Execution quality tracking
- Exchange adapter integration

Usage:
    >>> from src.execution import (
    ...     OrderRouter,
    ...     RouterConfig,
    ...     ExecutionRequest,
    ...     ExecutionResult,
    ... )
    >>>
    >>> config = RouterConfig()
    >>> router = OrderRouter(config, executor=exchange)
    >>>
    >>> request = ExecutionRequest(
    ...     symbol="BTCUSDT",
    ...     side=OrderSide.BUY,
    ...     quantity=Decimal("1.5"),
    ...     price=Decimal("50000"),
    ... )
    >>>
    >>> result = await router.execute(request)
"""

from src.execution.models import (
    # Enums
    ExecutionAlgorithm,
    ExecutionUrgency,
    OrderSide,
    OrderStatus,
    OrderType,
    SplitStrategy,
    TimeInForce,
    # Data classes
    ChildOrder,
    DepthLevel,
    ExecutionPlan,
    ExecutionRequest,
    ExecutionResult,
    MarketDepthAnalysis,
    RouterConfig,
    # Callback types
    OnExecutionComplete,
    OnOrderCancelled,
    OnOrderError,
    OnOrderFilled,
    OnOrderSent,
)
from src.execution.router import (
    ExchangeExecutor,
    MarketDataProvider,
    OrderRouter,
    SymbolInfoProvider,
)
from src.execution.algorithms import (
    # Base
    AlgorithmProgress,
    AlgorithmState,
    BaseExecutionAlgorithm,
    # TWAP
    TWAPAlgorithm,
    TWAPConfig,
    # VWAP
    VWAPAlgorithm,
    VWAPConfig,
    # Iceberg
    IcebergAlgorithm,
    IcebergConfig,
    # Sniper
    SniperAlgorithm,
    SniperConfig,
    # POV
    POVAlgorithm,
    POVConfig,
    # Protocols
    OrderBookProvider,
    OrderExecutor,
    PriceProvider,
    VolumeProvider,
    # Factory
    create_algorithm,
)

__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "ExecutionAlgorithm",
    "ExecutionUrgency",
    "SplitStrategy",
    # Data classes
    "ExecutionRequest",
    "ChildOrder",
    "ExecutionPlan",
    "ExecutionResult",
    "DepthLevel",
    "MarketDepthAnalysis",
    "RouterConfig",
    # Main components
    "OrderRouter",
    # Protocols
    "ExchangeExecutor",
    "MarketDataProvider",
    "SymbolInfoProvider",
    # Callbacks
    "OnOrderSent",
    "OnOrderFilled",
    "OnOrderCancelled",
    "OnOrderError",
    "OnExecutionComplete",
    # Algorithm base
    "AlgorithmState",
    "AlgorithmProgress",
    "BaseExecutionAlgorithm",
    # TWAP
    "TWAPAlgorithm",
    "TWAPConfig",
    # VWAP
    "VWAPAlgorithm",
    "VWAPConfig",
    # Iceberg
    "IcebergAlgorithm",
    "IcebergConfig",
    # Sniper
    "SniperAlgorithm",
    "SniperConfig",
    # POV
    "POVAlgorithm",
    "POVConfig",
    # Algorithm protocols
    "OrderBookProvider",
    "OrderExecutor",
    "PriceProvider",
    "VolumeProvider",
    # Factory
    "create_algorithm",
]
