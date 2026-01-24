"""
Execution Layer Models.

Provides data models for order routing, execution algorithms,
and order management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.models import MarketType


# =============================================================================
# Enums
# =============================================================================


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    CREATED = "created"           # Order created locally
    PENDING = "pending"           # Waiting to be sent
    SUBMITTED = "submitted"       # Sent to exchange
    ACCEPTED = "accepted"         # Accepted by exchange
    PARTIALLY_FILLED = "partially_filled"  # Partially filled
    FILLED = "filled"             # Fully filled
    CANCELLED = "cancelled"       # Cancelled
    REJECTED = "rejected"         # Rejected by exchange
    FAILED = "failed"             # Failed to send
    EXPIRED = "expired"           # Order expired


class TimeInForce(str, Enum):
    """Time in force options."""
    GTC = "GTC"  # Good 'til cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill
    GTD = "GTD"  # Good 'til date


class ExecutionAlgorithm(str, Enum):
    """Execution algorithm types."""
    DIRECT = "direct"           # Direct execution (no splitting)
    TWAP = "twap"               # Time-weighted average price
    VWAP = "vwap"               # Volume-weighted average price
    ICEBERG = "iceberg"         # Iceberg order (hidden quantity)
    SNIPER = "sniper"           # Sniper (capture liquidity)
    POV = "pov"                 # Percentage of volume
    SMART = "smart"             # Smart order routing (auto-select)


class ExecutionUrgency(str, Enum):
    """Execution urgency level."""
    LOW = "low"           # Can wait, minimize market impact
    MEDIUM = "medium"     # Balance speed and impact
    HIGH = "high"         # Execute quickly, accept some impact
    IMMEDIATE = "immediate"  # Execute now, market orders


class SplitStrategy(str, Enum):
    """Order splitting strategy."""
    NONE = "none"           # No splitting
    FIXED_SIZE = "fixed_size"      # Fixed size per child order
    FIXED_COUNT = "fixed_count"    # Fixed number of child orders
    LIQUIDITY_BASED = "liquidity_based"  # Based on market depth
    TIME_BASED = "time_based"      # Based on time intervals


# =============================================================================
# Order Request
# =============================================================================


@dataclass
class ExecutionRequest:
    """
    Request for order execution.

    This is the input to the OrderRouter, containing all information
    needed to determine how to execute an order.

    Attributes:
        symbol: Trading symbol
        side: Buy or sell
        quantity: Total quantity to execute
        order_type: Type of order
        price: Limit price (for limit orders)
        market_type: Spot or futures
        urgency: How quickly to execute
        algorithm: Preferred execution algorithm (or SMART for auto)
        max_slippage_pct: Maximum acceptable slippage percentage
        strategy_id: ID of the strategy requesting execution
        client_order_id: Client-provided order ID
        metadata: Additional metadata
    """

    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.LIMIT
    price: Optional[Decimal] = None
    market_type: MarketType = MarketType.SPOT
    urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART
    max_slippage_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Algorithm-specific parameters
    twap_duration_minutes: int = 30  # For TWAP
    twap_intervals: int = 10         # Number of slices
    iceberg_visible_pct: Decimal = field(default_factory=lambda: Decimal("0.10"))  # 10%
    pov_participation_rate: Decimal = field(default_factory=lambda: Decimal("0.10"))  # 10%

    def __post_init__(self):
        """Ensure proper types."""
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.market_type, str):
            self.market_type = MarketType(self.market_type)
        if isinstance(self.urgency, str):
            self.urgency = ExecutionUrgency(self.urgency)
        if isinstance(self.algorithm, str):
            self.algorithm = ExecutionAlgorithm(self.algorithm)

        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if self.price is not None and not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.max_slippage_pct, Decimal):
            self.max_slippage_pct = Decimal(str(self.max_slippage_pct))

    @property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value if price is set."""
        if self.price:
            return self.quantity * self.price
        return None


# =============================================================================
# Child Order
# =============================================================================


@dataclass
class ChildOrder:
    """
    A child order created by splitting a parent order.

    Attributes:
        parent_id: Parent execution ID
        child_index: Index of this child (0-based)
        symbol: Trading symbol
        side: Order side
        quantity: Quantity for this child
        price: Limit price
        order_type: Order type
        status: Current status
        exchange_order_id: Order ID from exchange
        client_order_id: Client order ID
        filled_quantity: Filled quantity
        average_price: Average fill price
        fee: Total fees paid
        fee_asset: Fee asset
        scheduled_at: When to send this order
        sent_at: When order was sent
        filled_at: When order was filled
    """

    parent_id: str
    child_index: int
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.CREATED
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    average_price: Optional[Decimal] = None
    fee: Decimal = field(default_factory=lambda: Decimal("0"))
    fee_asset: str = ""
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
            OrderStatus.EXPIRED,
        )

    @property
    def fill_pct(self) -> Decimal:
        """Get fill percentage."""
        if self.quantity <= 0:
            return Decimal("0")
        return (self.filled_quantity / self.quantity) * Decimal("100")


# =============================================================================
# Execution Plan
# =============================================================================


@dataclass
class ExecutionPlan:
    """
    Execution plan created by the OrderRouter.

    Contains the algorithm selection, split strategy, and child orders.

    Attributes:
        execution_id: Unique execution ID
        request: Original execution request
        algorithm: Selected algorithm
        split_strategy: How orders are split
        child_orders: List of child orders
        total_quantity: Total quantity across all children
        estimated_duration_seconds: Estimated execution time
        created_at: When plan was created
    """

    execution_id: str
    request: ExecutionRequest
    algorithm: ExecutionAlgorithm
    split_strategy: SplitStrategy
    child_orders: List[ChildOrder] = field(default_factory=list)
    estimated_duration_seconds: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_quantity(self) -> Decimal:
        """Get total quantity."""
        return sum(c.quantity for c in self.child_orders)

    @property
    def total_filled(self) -> Decimal:
        """Get total filled quantity."""
        return sum(c.filled_quantity for c in self.child_orders)

    @property
    def fill_pct(self) -> Decimal:
        """Get overall fill percentage."""
        if self.total_quantity <= 0:
            return Decimal("0")
        return (self.total_filled / self.total_quantity) * Decimal("100")

    @property
    def is_complete(self) -> bool:
        """Check if all child orders are complete."""
        return all(c.is_complete for c in self.child_orders)

    @property
    def pending_children(self) -> List[ChildOrder]:
        """Get pending child orders."""
        return [c for c in self.child_orders if not c.is_complete]

    @property
    def average_fill_price(self) -> Optional[Decimal]:
        """Calculate volume-weighted average fill price."""
        filled = [(c.filled_quantity, c.average_price)
                  for c in self.child_orders
                  if c.filled_quantity > 0 and c.average_price]
        if not filled:
            return None
        total_qty = sum(qty for qty, _ in filled)
        if total_qty <= 0:
            return None
        weighted_sum = sum(qty * price for qty, price in filled)
        return weighted_sum / total_qty

    @property
    def total_fees(self) -> Decimal:
        """Get total fees paid."""
        return sum(c.fee for c in self.child_orders)


# =============================================================================
# Execution Result
# =============================================================================


@dataclass
class ExecutionResult:
    """
    Result of an execution.

    Returned by the OrderRouter after execution completes.

    Attributes:
        execution_id: Execution ID
        success: Whether execution succeeded
        plan: The execution plan
        final_quantity: Total filled quantity
        average_price: Volume-weighted average price
        total_fee: Total fees paid
        slippage_pct: Actual slippage percentage
        duration_seconds: Execution duration
        error_message: Error message if failed
    """

    execution_id: str
    success: bool
    plan: ExecutionPlan
    final_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    average_price: Optional[Decimal] = None
    total_fee: Decimal = field(default_factory=lambda: Decimal("0"))
    slippage_pct: Optional[Decimal] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def filled_pct(self) -> Decimal:
        """Get fill percentage."""
        if self.plan.request.quantity <= 0:
            return Decimal("0")
        return (self.final_quantity / self.plan.request.quantity) * Decimal("100")


# =============================================================================
# Market Depth Analysis
# =============================================================================


@dataclass
class DepthLevel:
    """Single level in order book."""
    price: Decimal
    quantity: Decimal
    cumulative_quantity: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class MarketDepthAnalysis:
    """
    Analysis of market depth for execution planning.

    Attributes:
        symbol: Trading symbol
        bid_levels: Bid side depth
        ask_levels: Ask side depth
        best_bid: Best bid price
        best_ask: Best ask price
        spread: Bid-ask spread
        spread_pct: Spread as percentage
        total_bid_volume: Total bid volume (within N levels)
        total_ask_volume: Total ask volume (within N levels)
        imbalance: Order book imbalance (-1 to 1)
        analyzed_at: Analysis timestamp
    """

    symbol: str
    bid_levels: List[DepthLevel] = field(default_factory=list)
    ask_levels: List[DepthLevel] = field(default_factory=list)
    best_bid: Optional[Decimal] = None
    best_ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    spread_pct: Optional[Decimal] = None
    total_bid_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    total_ask_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    imbalance: Decimal = field(default_factory=lambda: Decimal("0"))
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def get_fill_price_estimate(
        self,
        side: OrderSide,
        quantity: Decimal,
    ) -> Optional[Decimal]:
        """
        Estimate fill price for a given quantity.

        Args:
            side: Order side (BUY uses asks, SELL uses bids)
            quantity: Quantity to fill

        Returns:
            Estimated volume-weighted average price, or None
        """
        levels = self.ask_levels if side == OrderSide.BUY else self.bid_levels

        if not levels:
            return None

        remaining = quantity
        total_cost = Decimal("0")
        total_filled = Decimal("0")

        for level in levels:
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            total_filled += fill_qty
            remaining -= fill_qty
            if remaining <= 0:
                break

        if total_filled <= 0:
            return None

        return total_cost / total_filled

    def get_market_impact_pct(
        self,
        side: OrderSide,
        quantity: Decimal,
    ) -> Optional[Decimal]:
        """
        Estimate market impact as percentage.

        Args:
            side: Order side
            quantity: Quantity

        Returns:
            Estimated impact percentage
        """
        if not self.mid_price:
            return None

        fill_price = self.get_fill_price_estimate(side, quantity)
        if not fill_price:
            return None

        if side == OrderSide.BUY:
            impact = (fill_price - self.mid_price) / self.mid_price
        else:
            impact = (self.mid_price - fill_price) / self.mid_price

        return impact * Decimal("100")


# =============================================================================
# Router Configuration
# =============================================================================


@dataclass
class RouterConfig:
    """
    Configuration for OrderRouter.

    Attributes:
        small_order_threshold_pct: Orders below this % of avg volume are "small"
        large_order_threshold_pct: Orders above this % of avg volume are "large"
        default_split_size: Default size for split orders
        max_child_orders: Maximum number of child orders
        min_child_order_value: Minimum value per child order
        enable_smart_routing: Enable smart algorithm selection
        enable_liquidity_check: Check liquidity before execution
        default_urgency: Default execution urgency
        max_parallel_children: Max parallel child order execution
    """

    small_order_threshold_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))
    large_order_threshold_pct: Decimal = field(default_factory=lambda: Decimal("5.0"))
    default_split_size: Decimal = field(default_factory=lambda: Decimal("100"))
    max_child_orders: int = 50
    min_child_order_value: Decimal = field(default_factory=lambda: Decimal("10"))
    enable_smart_routing: bool = True
    enable_liquidity_check: bool = True
    default_urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    max_parallel_children: int = 5

    # Algorithm-specific defaults
    twap_default_intervals: int = 10
    iceberg_default_visible_pct: Decimal = field(default_factory=lambda: Decimal("0.10"))
    pov_default_rate: Decimal = field(default_factory=lambda: Decimal("0.10"))


# =============================================================================
# Callbacks
# =============================================================================


# Type aliases for callbacks
OnOrderSent = Callable[[ChildOrder], None]
OnOrderFilled = Callable[[ChildOrder], None]
OnOrderCancelled = Callable[[ChildOrder], None]
OnOrderError = Callable[[ChildOrder, str], None]
OnExecutionComplete = Callable[[ExecutionResult], None]
