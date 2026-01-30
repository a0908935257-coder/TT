"""
Order Simulation Module.

Handles order matching simulation using kline high/low prices.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Optional
import uuid

from ..core.models import Kline
from .config import BacktestConfig
from .fees import FeeCalculator, FeeContext
from .position import Position
from .slippage import SlippageContext, SlippageModel

if TYPE_CHECKING:
    pass


# =============================================================================
# Order Types and Enums
# =============================================================================


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderTimeInForce(str, Enum):
    """Time in force enumeration."""

    GTC = "gtc"  # Good Till Cancelled
    GTD = "gtd"  # Good Till Date (expiry bar)
    IOC = "ioc"  # Immediate Or Cancel (partial fill OK, cancel rest)
    FOK = "fok"  # Fill Or Kill (all or nothing)


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# =============================================================================
# Pending Order and Fill
# =============================================================================


@dataclass
class PendingOrder:
    """
    Pending order waiting for execution.

    Attributes:
        order_id: Unique order identifier
        order_type: Type of order (MARKET, LIMIT, etc.)
        side: Order side (BUY/SELL)
        quantity: Order quantity
        limit_price: Limit price (for LIMIT and STOP_LIMIT orders)
        stop_price: Stop trigger price (for STOP orders)
        time_in_force: Time in force policy
        created_bar: Bar index when order was created
        expiry_bar: Bar index when order expires (for GTD)
        filled_quantity: Quantity already filled
        stop_loss: Stop loss price for resulting position
        take_profit: Take profit price for resulting position
    """

    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    created_bar: int = 0
    expiry_bar: Optional[int] = None
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY

    @property
    def is_limit(self) -> bool:
        """Check if this is a limit order."""
        return self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)

    def is_expired(self, current_bar: int) -> bool:
        """Check if order has expired."""
        if self.time_in_force == OrderTimeInForce.GTD and self.expiry_bar is not None:
            return current_bar >= self.expiry_bar
        return False


@dataclass
class Fill:
    """
    Order fill record.

    Attributes:
        order_id: ID of the filled order
        fill_price: Execution price
        fill_quantity: Quantity filled
        fill_bar: Bar index when filled
        fill_time: Timestamp of fill
        fee: Transaction fee
        is_maker: Whether this was a maker fill
    """

    order_id: str
    fill_price: Decimal
    fill_quantity: Decimal
    fill_bar: int
    fill_time: datetime
    fee: Decimal = field(default_factory=lambda: Decimal("0"))
    is_maker: bool = False


# =============================================================================
# Order Book
# =============================================================================


class OrderBook:
    """
    Manages pending orders and their execution.

    Simulates order book behavior with limit orders, partial fills,
    and various time-in-force policies.

    Example:
        book = OrderBook(config)
        order_id = book.add_order(PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            limit_price=Decimal("100"),
        ))
        fills = book.process_bar(kline, bar_idx)
    """

    def __init__(
        self,
        config: BacktestConfig,
        enable_partial_fills: bool = False,
    ) -> None:
        """
        Initialize order book.

        Args:
            config: Backtest configuration
            enable_partial_fills: Whether to simulate partial fills
        """
        self._config = config
        self._enable_partial_fills = enable_partial_fills
        self._orders: dict[str, PendingOrder] = {}
        self._fee_calculator: FeeCalculator = config.create_fee_calculator()

    @property
    def pending_orders(self) -> list[PendingOrder]:
        """Get all pending orders."""
        return [
            o for o in self._orders.values() if o.status == OrderStatus.PENDING
        ]

    @property
    def order_count(self) -> int:
        """Get count of pending orders."""
        return len(self.pending_orders)

    def add_order(self, order: PendingOrder) -> str:
        """
        Add a new order to the book.

        Args:
            order: Order to add

        Returns:
            Order ID
        """
        self._orders[order.order_id] = order
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled, False if not found or already filled
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False

        order.status = OrderStatus.CANCELLED
        return True

    def get_order(self, order_id: str) -> Optional[PendingOrder]:
        """Get an order by ID."""
        return self._orders.get(order_id)

    def process_bar(
        self,
        kline: Kline,
        bar_idx: int,
        avg_volume: Optional[Decimal] = None,
    ) -> list[Fill]:
        """
        Process all pending orders against a kline.

        Checks if orders can be filled based on kline prices,
        handles partial fills, and manages order expiration.

        Args:
            kline: Current kline data
            bar_idx: Current bar index
            avg_volume: Average volume for partial fill calculation

        Returns:
            List of fills executed this bar
        """
        fills: list[Fill] = []

        for order in list(self.pending_orders):
            # Check expiration
            if order.is_expired(bar_idx):
                order.status = OrderStatus.EXPIRED
                continue

            # Try to fill the order
            fill = self._try_fill_order(order, kline, bar_idx, avg_volume)
            if fill:
                fills.append(fill)

        return fills

    def _try_fill_order(
        self,
        order: PendingOrder,
        kline: Kline,
        bar_idx: int,
        avg_volume: Optional[Decimal] = None,
    ) -> Optional[Fill]:
        """
        Try to fill a single order.

        Args:
            order: Order to fill
            kline: Current kline
            bar_idx: Current bar index
            avg_volume: Average volume for partial fill calculation

        Returns:
            Fill if executed, None otherwise
        """
        # Check if order can be triggered/filled
        can_fill, fill_price = self._check_fill_conditions(order, kline)
        if not can_fill or fill_price is None:
            return None

        # Calculate fill quantity (partial fill logic)
        fill_qty = self._calculate_fill_quantity(order, avg_volume, kline)
        if fill_qty <= 0:
            return None

        # Handle IOC/FOK
        if order.time_in_force == OrderTimeInForce.FOK:
            if fill_qty < order.remaining_quantity:
                order.status = OrderStatus.CANCELLED
                return None
        elif order.time_in_force == OrderTimeInForce.IOC:
            # IOC: fill what we can, cancel the rest
            pass

        # Execute fill
        order.filled_quantity += fill_qty
        is_maker = order.is_limit

        # Calculate fee
        fee = self._fee_calculator.calculate_fee(
            fill_price,
            fill_qty,
            FeeContext(is_maker=is_maker),
        )

        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            # IOC: cancel remaining after partial fill
            if order.time_in_force == OrderTimeInForce.IOC:
                order.status = OrderStatus.CANCELLED

        return Fill(
            order_id=order.order_id,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            fill_bar=bar_idx,
            fill_time=kline.close_time,
            fee=fee,
            is_maker=is_maker,
        )

    def _check_fill_conditions(
        self,
        order: PendingOrder,
        kline: Kline,
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if order can be filled and determine fill price.

        Args:
            order: Order to check
            kline: Current kline

        Returns:
            Tuple of (can_fill, fill_price)
        """
        if order.order_type == OrderType.MARKET:
            # Market orders fill at close (worst case assumption)
            return True, kline.close

        elif order.order_type == OrderType.LIMIT:
            if order.is_buy:
                # Buy limit: price must touch or go below limit
                if kline.low <= order.limit_price:
                    # Gap handling for limit orders:
                    # If price gaps down past the limit price, fill at the better
                    # price (open price). This simulates favorable gap fills where
                    # the order gets filled at a better price than requested.
                    # Example: limit buy at 100, open gaps to 98 -> fill at 98
                    fill_price = min(order.limit_price, kline.open)
                    return True, fill_price
            else:
                # Sell limit: price must touch or go above limit
                if kline.high >= order.limit_price:
                    # Gap handling for limit orders:
                    # If price gaps up past the limit price, fill at the better
                    # price (open price). This simulates favorable gap fills.
                    # Example: limit sell at 100, open gaps to 102 -> fill at 102
                    fill_price = max(order.limit_price, kline.open)
                    return True, fill_price

        elif order.order_type == OrderType.STOP_MARKET:
            if order.is_buy:
                # Buy stop: triggered when price goes above stop
                if kline.high >= order.stop_price:
                    # Fill at stop or worse (market after trigger)
                    fill_price = max(order.stop_price, kline.open)
                    fill_price = min(fill_price, kline.high)
                    return True, fill_price
            else:
                # Sell stop: triggered when price goes below stop
                if kline.low <= order.stop_price:
                    # Fill at stop or worse
                    fill_price = min(order.stop_price, kline.open)
                    fill_price = max(fill_price, kline.low)
                    return True, fill_price

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.is_buy:
                # Buy stop-limit: triggered when price goes above stop
                if kline.high >= order.stop_price:
                    # Then check limit
                    if kline.low <= order.limit_price:
                        return True, order.limit_price
            else:
                # Sell stop-limit: triggered when price goes below stop
                if kline.low <= order.stop_price:
                    # Then check limit
                    if kline.high >= order.limit_price:
                        return True, order.limit_price

        return False, None

    def _calculate_fill_quantity(
        self,
        order: PendingOrder,
        avg_volume: Optional[Decimal],
        kline: Kline,
    ) -> Decimal:
        """
        Calculate fill quantity based on volume and partial fill settings.

        Partial fill logic:
        - Order < 1% of volume: 100% fill
        - Order 1-10% of volume: 50-100% fill (linear interpolation)
        - Order > 10% of volume: minimum 10% fill

        Args:
            order: Order to fill
            avg_volume: Average trading volume
            kline: Current kline

        Returns:
            Quantity to fill
        """
        remaining = order.remaining_quantity

        if not self._enable_partial_fills or avg_volume is None or avg_volume <= 0:
            # Full fill
            return remaining

        # Calculate order as percentage of volume
        order_notional = remaining * (order.limit_price or kline.close)
        volume_notional = avg_volume * kline.close
        order_pct = order_notional / volume_notional

        # Determine fill percentage
        if order_pct <= Decimal("0.01"):
            fill_pct = Decimal("1.0")
        elif order_pct <= Decimal("0.1"):
            # Linear interpolation: 1% -> 100%, 10% -> 50%
            # fill_pct = 1.0 - (order_pct - 0.01) / 0.09 * 0.5
            normalized = (order_pct - Decimal("0.01")) / Decimal("0.09")
            fill_pct = Decimal("1.0") - normalized * Decimal("0.5")
        else:
            fill_pct = Decimal("0.1")

        # Apply fill percentage to remaining quantity
        fill_qty = remaining * fill_pct

        # Ensure minimum fill (at least 1 unit or remaining)
        min_fill = min(Decimal("0.001"), remaining)
        fill_qty = max(fill_qty, min_fill)

        return min(fill_qty, remaining)

    def reset(self) -> None:
        """Reset order book state."""
        self._orders.clear()


class SignalType(str, Enum):
    """Trading signal type."""

    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    CLOSE_ALL = "close_all"


@dataclass
class Signal:
    """
    Trading signal from strategy.

    Attributes:
        signal_type: Type of signal
        price: Target price (None for market orders)
        stop_loss: Stop loss price (for entries)
        take_profit: Take profit price (for entries)
        reason: Optional reason/tag for the signal
    """

    signal_type: SignalType
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    reason: Optional[str] = None

    @classmethod
    def long_entry(
        cls,
        price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        reason: Optional[str] = None,
    ) -> "Signal":
        """Create a long entry signal."""
        return cls(
            signal_type=SignalType.LONG_ENTRY,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    @classmethod
    def short_entry(
        cls,
        price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        reason: Optional[str] = None,
    ) -> "Signal":
        """Create a short entry signal."""
        return cls(
            signal_type=SignalType.SHORT_ENTRY,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    @classmethod
    def close_all(cls, reason: Optional[str] = None) -> "Signal":
        """Create a close all positions signal."""
        return cls(signal_type=SignalType.CLOSE_ALL, reason=reason)


class OrderSimulator:
    """
    Simulates order matching using kline data.

    Uses kline high/low prices to simulate order fills,
    applying configurable slippage and fees via pluggable models.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize order simulator.

        Args:
            config: Backtest configuration
        """
        self._config = config
        # Create models from config factory methods
        self._slippage_model: SlippageModel = config.create_slippage_model()
        self._fee_calculator: FeeCalculator = config.create_fee_calculator()
        # Context for advanced slippage/fee calculation
        self._recent_klines: list[Kline] = []
        self._cumulative_volume: Decimal = Decimal("0")

    @property
    def fee_rate(self) -> Decimal:
        """Get base fee rate (for backward compatibility)."""
        return self._fee_calculator.base_rate

    @property
    def slippage_pct(self) -> Decimal:
        """Get slippage percentage (for backward compatibility)."""
        return self._config.slippage_pct

    @property
    def slippage_model(self) -> SlippageModel:
        """Get the slippage model."""
        return self._slippage_model

    @property
    def fee_calculator(self) -> FeeCalculator:
        """Get the fee calculator."""
        return self._fee_calculator

    def update_context(
        self,
        kline: Kline,
        max_klines: int = 20,
    ) -> None:
        """
        Update context with recent kline data for advanced models.

        Args:
            kline: Current kline to add
            max_klines: Maximum klines to keep for ATR calculation
        """
        self._recent_klines.append(kline)
        if len(self._recent_klines) > max_klines:
            self._recent_klines.pop(0)

    def update_cumulative_volume(self, volume: Decimal) -> None:
        """
        Update cumulative trading volume for tiered fee calculation.

        Args:
            volume: Volume to add
        """
        self._cumulative_volume += volume

    def calculate_fill_price(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        order_size: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate the actual fill price with slippage.

        Args:
            target_price: Target/signal price
            is_buy: Whether this is a buy order
            kline: Current kline for price bounds
            order_size: Order size for market impact calculation
            avg_volume: Average volume for market impact calculation

        Returns:
            Adjusted fill price
        """
        # Build slippage context
        context = SlippageContext(
            order_size=order_size if order_size else Decimal("0"),
            avg_volume=avg_volume,
            recent_klines=self._recent_klines if self._recent_klines else None,
        )

        # Use slippage model to calculate fill price
        return self._slippage_model.apply_slippage(target_price, is_buy, kline, context)

    def override_notional(self, notional: Optional[Decimal]) -> None:
        """Override notional for next trade (equity-based sizing)."""
        self._override_notional = notional

    def calculate_quantity(self, price: Decimal) -> Decimal:
        """
        Calculate position quantity based on config.

        Args:
            price: Entry price

        Returns:
            Position quantity (0 if price is invalid)
        """
        if price <= 0:
            return Decimal("0")
        notional = getattr(self, '_override_notional', None) or self._config.notional_per_trade
        self._override_notional = None  # Reset after use
        return notional / price

    def calculate_fee(
        self,
        price: Decimal,
        quantity: Decimal,
        is_maker: bool = False,
    ) -> Decimal:
        """
        Calculate trading fee.

        Args:
            price: Fill price
            quantity: Order quantity
            is_maker: Whether this is a maker order (limit order)

        Returns:
            Fee amount
        """
        context = FeeContext(
            is_maker=is_maker,
            cumulative_volume=self._cumulative_volume,
        )
        fee = self._fee_calculator.calculate_fee(price, quantity, context)
        # Binance charges fees on leveraged notional (qty × price × leverage)
        if self._config.use_margin and self._config.leverage > 1:
            fee = fee * Decimal(self._config.leverage)
        return fee

    def create_position(
        self,
        side: str,
        kline: Kline,
        bar_index: int,
        target_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Position:
        """
        Create a new position from a fill.

        Note on entry price consistency:
        The actual entry_price may differ from target_price due to slippage
        simulation. This is expected behavior that models real market execution.
        If stop_loss/take_profit are provided as absolute prices, they remain
        unchanged. Strategies should be aware that the risk/reward ratio may
        shift slightly due to slippage.

        Args:
            side: 'LONG' or 'SHORT'
            kline: Current kline
            bar_index: Current bar index
            target_price: Target entry price (or kline.close for market)
            stop_loss: Stop loss price (absolute, not adjusted for slippage)
            take_profit: Take profit price (absolute, not adjusted for slippage)

        Returns:
            New Position object
        """
        # Determine entry price
        if target_price is None:
            entry_price = kline.close
        else:
            is_buy = side == "LONG"
            entry_price = self.calculate_fill_price(target_price, is_buy, kline)

        # Calculate quantity and fee
        quantity = self.calculate_quantity(entry_price)
        entry_fee = self.calculate_fee(entry_price, quantity)

        return Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=kline.close_time,
            entry_bar=bar_index,
            entry_fee=entry_fee,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def check_stop_loss(
        self, position: Position, kline: Kline
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if stop loss was hit.

        Args:
            position: Current position
            kline: Current kline

        Returns:
            Tuple of (triggered, fill_price)
        """
        if position.stop_loss is None:
            return False, None

        if position.side == "LONG":
            if kline.low <= position.stop_loss:
                # Fill at stop or worse (kline low if gapped through)
                fill_price = min(position.stop_loss, kline.open)
                fill_price = max(fill_price, kline.low)
                return True, fill_price
        else:  # SHORT
            if kline.high >= position.stop_loss:
                # Fill at stop or worse (kline high if gapped through)
                fill_price = max(position.stop_loss, kline.open)
                fill_price = min(fill_price, kline.high)
                return True, fill_price

        return False, None

    def check_take_profit(
        self, position: Position, kline: Kline
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if take profit was hit.

        Args:
            position: Current position
            kline: Current kline

        Returns:
            Tuple of (triggered, fill_price)
        """
        if position.take_profit is None:
            return False, None

        if position.side == "LONG":
            if kline.high >= position.take_profit:
                return True, position.take_profit
        else:  # SHORT
            if kline.low <= position.take_profit:
                return True, position.take_profit

        return False, None

    def simulate_exit_order(
        self, position: Position, kline: Kline, exit_price: Optional[Decimal] = None
    ) -> tuple[Decimal, Decimal]:
        """
        Simulate an exit order fill.

        Args:
            position: Position to exit
            kline: Current kline
            exit_price: Target exit price (or kline.close for market)

        Returns:
            Tuple of (fill_price, fee)
        """
        if exit_price is None:
            fill_price = kline.close
        else:
            is_buy = position.side == "SHORT"  # Exit short = buy
            fill_price = self.calculate_fill_price(exit_price, is_buy, kline)

        fee = self.calculate_fee(fill_price, position.quantity)
        return fill_price, fee
