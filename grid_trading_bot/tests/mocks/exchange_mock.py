"""
Mock Exchange Client for testing.

Simulates exchange behavior without real API calls.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

from src.core.models import Balance, Kline, MarketType, Order, OrderSide, OrderStatus, OrderType, SymbolInfo


@dataclass
class MockOrderBook:
    """Simulated order book for the mock exchange."""

    bids: list[tuple[Decimal, Decimal]] = field(default_factory=list)
    asks: list[tuple[Decimal, Decimal]] = field(default_factory=list)


class MockExchangeClient:
    """
    Mock Exchange Client.

    Simulates exchange functionality for testing:
    - Price control and updates
    - Order creation, cancellation, and fill simulation
    - Balance tracking
    - Order fill callbacks

    Example:
        >>> mock = MockExchangeClient()
        >>> mock.set_price(50000)
        >>> mock.set_balance("USDT", 10000)
        >>> order = await mock.limit_buy("BTCUSDT", 0.1, 49000)
        >>> mock.move_price(48000)  # Triggers buy order fill
    """

    def __init__(self):
        """Initialize mock exchange client."""
        # Current prices per symbol
        self._prices: dict[str, Decimal] = {}

        # Balances per asset
        self._balances: dict[str, Decimal] = {}

        # Open orders: order_id -> Order
        self._open_orders: dict[str, Order] = {}

        # Filled orders history
        self._filled_orders: list[Order] = []

        # Order fill callback
        self._on_fill_callback: Optional[Callable[[Order], Any]] = None

        # Kline data storage
        self._klines: dict[str, list[Kline]] = {}

        # Connection status
        self._connected: bool = True

        # Symbol precision settings
        self._price_precision: dict[str, int] = {}
        self._quantity_precision: dict[str, int] = {}

        # Default fee rate
        self._fee_rate: Decimal = Decimal("0.001")  # 0.1%

        # Simulated latency (seconds)
        self._latency: float = 0.01

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        return self._connected

    # =========================================================================
    # Price Control
    # =========================================================================

    def set_price(self, price: Decimal | float | str, symbol: str = "BTCUSDT") -> None:
        """
        Set current price for a symbol.

        Args:
            price: The price to set
            symbol: Trading pair symbol
        """
        self._prices[symbol] = Decimal(str(price))

    def get_price_sync(self, symbol: str = "BTCUSDT") -> Optional[Decimal]:
        """
        Get current price synchronously.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price or None if not set
        """
        return self._prices.get(symbol)

    async def get_price(
        self,
        symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Decimal]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair symbol
            market_type: Market type (ignored in mock)

        Returns:
            Current price
        """
        await asyncio.sleep(self._latency)
        return self._prices.get(symbol)

    def move_price(
        self,
        new_price: Decimal | float | str,
        symbol: str = "BTCUSDT",
    ) -> list[Order]:
        """
        Move price and trigger order fills.

        When price moves:
        - BUY LIMIT orders fill if price <= order price
        - SELL LIMIT orders fill if price >= order price

        Args:
            new_price: New price to set
            symbol: Trading pair symbol

        Returns:
            List of orders that were filled
        """
        new_price = Decimal(str(new_price))
        old_price = self._prices.get(symbol, new_price)
        self._prices[symbol] = new_price

        filled_orders = []

        # Check all open orders for potential fills
        orders_to_fill = []
        for order_id, order in list(self._open_orders.items()):
            if order.symbol != symbol:
                continue

            should_fill = False

            if order.side == OrderSide.BUY:
                # BUY LIMIT fills when price drops to or below order price
                if new_price <= order.price:
                    should_fill = True
            else:  # SELL
                # SELL LIMIT fills when price rises to or above order price
                if new_price >= order.price:
                    should_fill = True

            if should_fill:
                orders_to_fill.append(order_id)

        # Fill orders
        for order_id in orders_to_fill:
            filled = self._fill_order(order_id, fill_price=new_price)
            if filled:
                filled_orders.append(filled)

        return filled_orders

    async def move_price_async(
        self,
        new_price: Decimal | float | str,
        symbol: str = "BTCUSDT",
    ) -> list[Order]:
        """
        Async version of move_price.

        Args:
            new_price: New price to set
            symbol: Trading pair symbol

        Returns:
            List of orders that were filled
        """
        await asyncio.sleep(self._latency)
        filled_orders = self.move_price(new_price, symbol)

        # Trigger callbacks for filled orders
        if self._on_fill_callback:
            for order in filled_orders:
                result = self._on_fill_callback(order)
                if asyncio.iscoroutine(result):
                    await result

        return filled_orders

    # =========================================================================
    # Order Management
    # =========================================================================

    async def limit_buy(
        self,
        symbol: str,
        quantity: Decimal | float | str,
        price: Decimal | float | str,
        market_type: MarketType = MarketType.SPOT,
        **kwargs,
    ) -> Order:
        """
        Place a limit buy order.

        Args:
            symbol: Trading pair symbol
            quantity: Order quantity
            price: Order price
            market_type: Market type

        Returns:
            Created Order
        """
        await asyncio.sleep(self._latency)

        order = self._create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
        )

        # Check for immediate fill
        current_price = self._prices.get(symbol)
        if current_price and current_price <= order.price:
            self._fill_order(order.order_id, fill_price=current_price)
            return self._filled_orders[-1]

        return order

    async def limit_sell(
        self,
        symbol: str,
        quantity: Decimal | float | str,
        price: Decimal | float | str,
        market_type: MarketType = MarketType.SPOT,
        **kwargs,
    ) -> Order:
        """
        Place a limit sell order.

        Args:
            symbol: Trading pair symbol
            quantity: Order quantity
            price: Order price
            market_type: Market type

        Returns:
            Created Order
        """
        await asyncio.sleep(self._latency)

        order = self._create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
        )

        # Check for immediate fill
        current_price = self._prices.get(symbol)
        if current_price and current_price >= order.price:
            self._fill_order(order.order_id, fill_price=current_price)
            return self._filled_orders[-1]

        return order

    async def market_buy(
        self,
        symbol: str,
        quantity: Decimal | float | str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Place a market buy order (immediately fills).

        Args:
            symbol: Trading pair symbol
            quantity: Order quantity
            market_type: Market type

        Returns:
            Filled Order
        """
        await asyncio.sleep(self._latency)

        current_price = self._prices.get(symbol, Decimal("0"))

        order = self._create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(quantity)),
            price=current_price,
        )

        # Immediately fill market orders
        self._fill_order(order.order_id, fill_price=current_price)
        return self._filled_orders[-1]

    async def market_sell(
        self,
        symbol: str,
        quantity: Decimal | float | str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Place a market sell order (immediately fills).

        Args:
            symbol: Trading pair symbol
            quantity: Order quantity
            market_type: Market type

        Returns:
            Filled Order
        """
        await asyncio.sleep(self._latency)

        current_price = self._prices.get(symbol, Decimal("0"))

        order = self._create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(quantity)),
            price=current_price,
        )

        # Immediately fill market orders
        self._fill_order(order.order_id, fill_price=current_price)
        return self._filled_orders[-1]

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Cancel an open order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            market_type: Market type

        Returns:
            Cancelled Order
        """
        await asyncio.sleep(self._latency)

        if order_id not in self._open_orders:
            raise ValueError(f"Order {order_id} not found")

        order = self._open_orders.pop(order_id)
        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.now(timezone.utc)

        return order

    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            market_type: Market type

        Returns:
            Order if found
        """
        await asyncio.sleep(self._latency)

        # Check open orders
        if order_id in self._open_orders:
            return self._open_orders[order_id]

        # Check filled orders
        for order in self._filled_orders:
            if order.order_id == order_id:
                return order

        return None

    async def get_open_orders(
        self,
        symbol: str = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> list[Order]:
        """
        Get all open orders for a symbol.

        Args:
            symbol: Trading pair symbol (None for all)
            market_type: Market type

        Returns:
            List of open orders
        """
        await asyncio.sleep(self._latency)

        if symbol is None:
            return list(self._open_orders.values())
        return [
            order for order in self._open_orders.values()
            if order.symbol == symbol
        ]

    async def get_open_orders_for_bot(
        self,
        bot_id: str,
        symbol: str = None,
        market: MarketType = MarketType.SPOT,
        **kwargs,
    ) -> list[Order]:
        """
        Get open orders belonging to a specific bot.

        Args:
            bot_id: Bot identifier to filter by
            symbol: Trading pair symbol (None for all)
            market_type: Market type

        Returns:
            List of open orders belonging to the specified bot
        """
        all_orders = await self.get_open_orders(symbol, market)
        return [
            order for order in all_orders
            if order.client_order_id and order.client_order_id.startswith(bot_id)
        ]

    def simulate_fill(self, order_id: str, fill_price: Optional[Decimal] = None) -> Optional[Order]:
        """
        Manually trigger an order fill.

        Args:
            order_id: Order ID to fill
            fill_price: Price to fill at (uses order price if None)

        Returns:
            Filled Order or None if not found
        """
        return self._fill_order(order_id, fill_price)

    # =========================================================================
    # Balance Management
    # =========================================================================

    def set_balance(self, asset: str, amount: Decimal | float | str) -> None:
        """
        Set balance for an asset.

        Args:
            asset: Asset symbol (e.g., "USDT", "BTC")
            amount: Balance amount
        """
        self._balances[asset] = Decimal(str(amount))

    async def get_balance(
        self,
        asset: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Balance:
        """
        Get balance for an asset.

        Args:
            asset: Asset symbol
            market_type: Market type (ignored in mock)

        Returns:
            Balance object
        """
        await asyncio.sleep(self._latency)
        amount = self._balances.get(asset, Decimal("0"))
        return Balance(
            asset=asset,
            free=amount,
            locked=Decimal("0"),
        )

    async def get_balances(self) -> dict[str, Decimal]:
        """
        Get all balances.

        Returns:
            Dict of asset -> balance
        """
        await asyncio.sleep(self._latency)
        return self._balances.copy()

    # =========================================================================
    # Ticker and Klines
    # =========================================================================

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get ticker info for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker dictionary
        """
        await asyncio.sleep(self._latency)

        price = self._prices.get(symbol, Decimal("0"))

        return {
            "symbol": symbol,
            "price": price,
            "bid": price * Decimal("0.999"),
            "ask": price * Decimal("1.001"),
            "volume": Decimal("1000000"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def set_klines(self, symbol: str, klines: list[Kline]) -> None:
        """
        Set kline data for a symbol.

        Args:
            symbol: Trading pair symbol
            klines: List of Kline objects
        """
        self._klines[symbol] = klines

    async def get_klines(
        self,
        symbol: str,
        interval: str = "4h",
        limit: int = 100,
        market: MarketType = MarketType.SPOT,
    ) -> list[Kline]:
        """
        Get kline data for a symbol.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            limit: Number of klines
            market: Market type

        Returns:
            List of Kline objects
        """
        await asyncio.sleep(self._latency)

        klines = self._klines.get(symbol, [])
        return klines[-limit:] if len(klines) > limit else klines

    # =========================================================================
    # Precision and Rounding
    # =========================================================================

    def set_precision(
        self,
        symbol: str,
        price_precision: int = 2,
        quantity_precision: int = 6,
    ) -> None:
        """
        Set precision for a symbol.

        Args:
            symbol: Trading pair symbol
            price_precision: Decimal places for price
            quantity_precision: Decimal places for quantity
        """
        self._price_precision[symbol] = price_precision
        self._quantity_precision[symbol] = quantity_precision

    async def get_symbol_info(
        self,
        symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> SymbolInfo:
        """
        Get symbol trading info.

        Args:
            symbol: Trading pair symbol
            market_type: Market type

        Returns:
            SymbolInfo with trading constraints
        """
        await asyncio.sleep(self._latency)

        price_precision = self._price_precision.get(symbol, 2)
        quantity_precision = self._quantity_precision.get(symbol, 6)

        tick_size = Decimal(10) ** -price_precision
        step_size = Decimal(10) ** -quantity_precision

        base_asset = self._get_base_asset(symbol)
        quote_asset = self._get_quote_asset(symbol)

        return SymbolInfo(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            price_precision=price_precision,
            quantity_precision=quantity_precision,
            min_quantity=Decimal("0.00001"),
            min_notional=Decimal("10"),
            tick_size=tick_size,
            step_size=step_size,
        )

    def round_price(
        self,
        symbol: str,
        price: Decimal,
        market_type: MarketType = MarketType.SPOT,
    ) -> Decimal:
        """
        Round price to exchange precision.

        Args:
            symbol: Trading pair symbol
            price: Price to round
            market_type: Market type

        Returns:
            Rounded price
        """
        precision = self._price_precision.get(symbol, 2)
        factor = Decimal(10) ** precision
        return (price * factor).to_integral_value() / factor

    def round_quantity(
        self,
        symbol: str,
        quantity: Decimal,
        market_type: MarketType = MarketType.SPOT,
    ) -> Decimal:
        """
        Round quantity to exchange precision.

        Args:
            symbol: Trading pair symbol
            quantity: Quantity to round
            market_type: Market type

        Returns:
            Rounded quantity
        """
        precision = self._quantity_precision.get(symbol, 6)
        factor = Decimal(10) ** precision
        return (quantity * factor).to_integral_value() / factor

    # =========================================================================
    # User Data Stream
    # =========================================================================

    def set_on_fill_callback(self, callback: Callable[[Order], Any]) -> None:
        """
        Register callback for order fills.

        Args:
            callback: Function to call when order fills
        """
        self._on_fill_callback = callback

    async def subscribe_user_data(self, callback: Callable) -> None:
        """Subscribe to user data stream (no-op in mock)."""
        pass

    async def unsubscribe_user_data(self) -> None:
        """Unsubscribe from user data stream (no-op in mock)."""
        pass

    # =========================================================================
    # Connection Control
    # =========================================================================

    def set_connected(self, connected: bool) -> None:
        """
        Set connection status.

        Args:
            connected: Connection status
        """
        self._connected = connected

    def set_latency(self, latency: float) -> None:
        """
        Set simulated latency in seconds.

        Args:
            latency: Latency in seconds
        """
        self._latency = latency

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal,
    ) -> Order:
        """Create a new order."""
        order_id = str(uuid.uuid4())[:8]

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            price=price,
            quantity=quantity,
            filled_qty=Decimal("0"),
            avg_price=None,
            fee=Decimal("0"),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        self._open_orders[order_id] = order
        return order

    def _fill_order(
        self,
        order_id: str,
        fill_price: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """Fill an open order."""
        if order_id not in self._open_orders:
            return None

        order = self._open_orders.pop(order_id)

        if fill_price is None:
            fill_price = order.price

        # Calculate fee
        fee = order.quantity * fill_price * self._fee_rate

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        order.avg_price = fill_price
        order.fee = fee
        order.updated_at = datetime.now(timezone.utc)

        # Update balances
        if order.side == OrderSide.BUY:
            # Deduct quote currency, add base currency
            base = self._get_base_asset(order.symbol)
            quote = self._get_quote_asset(order.symbol)
            cost = order.quantity * fill_price + fee
            self._balances[quote] = self._balances.get(quote, Decimal("0")) - cost
            self._balances[base] = self._balances.get(base, Decimal("0")) + order.quantity
        else:  # SELL
            # Deduct base currency, add quote currency
            base = self._get_base_asset(order.symbol)
            quote = self._get_quote_asset(order.symbol)
            revenue = order.quantity * fill_price - fee
            self._balances[base] = self._balances.get(base, Decimal("0")) - order.quantity
            self._balances[quote] = self._balances.get(quote, Decimal("0")) + revenue

        self._filled_orders.append(order)

        return order

    def _get_base_asset(self, symbol: str) -> str:
        """Get base asset from symbol."""
        # Simple heuristic for common pairs
        if symbol.endswith("USDT"):
            return symbol[:-4]
        if symbol.endswith("USD"):
            return symbol[:-3]
        if symbol.endswith("BTC"):
            return symbol[:-3]
        return symbol[:3]

    def _get_quote_asset(self, symbol: str) -> str:
        """Get quote asset from symbol."""
        if symbol.endswith("USDT"):
            return "USDT"
        if symbol.endswith("USD"):
            return "USD"
        if symbol.endswith("BTC"):
            return "BTC"
        return symbol[-3:]

    # =========================================================================
    # Test Helpers
    # =========================================================================

    def reset(self) -> None:
        """Reset all mock state."""
        self._prices.clear()
        self._balances.clear()
        self._open_orders.clear()
        self._filled_orders.clear()
        self._klines.clear()
        self._on_fill_callback = None
        self._connected = True

    def get_filled_orders(self) -> list[Order]:
        """Get all filled orders."""
        return self._filled_orders.copy()

    def get_open_orders_count(self) -> int:
        """Get count of open orders."""
        return len(self._open_orders)
