"""
Grid Order Manager.

Manages grid orders including initial placement, level-order mapping,
and synchronization with the exchange.
"""

import asyncio
from decimal import Decimal
from typing import Optional

from core import get_logger
from core.models import MarketType, Order, OrderSide, OrderStatus
from data import MarketDataManager
from exchange import ExchangeClient
from notification import NotificationManager

from .models import GridLevel, GridSetup, LevelSide, LevelState

logger = get_logger(__name__)


class GridOrderManager:
    """
    Grid Order Manager.

    Manages grid level orders including:
    - Initial order placement based on current price
    - Level-to-order and order-to-level mappings
    - Order cancellation at specific levels
    - Synchronization with exchange order state

    Example:
        >>> manager = GridOrderManager(
        ...     exchange=exchange,
        ...     data_manager=data_manager,
        ...     notifier=notifier,
        ...     bot_id="grid_bot_1",
        ...     symbol="BTCUSDT",
        ... )
        >>> manager.initialize(grid_setup)
        >>> placed = await manager.place_initial_orders()
        >>> print(f"Placed {placed} orders")
    """

    # Default tolerance for price comparison (0.1%)
    DEFAULT_PRICE_TOLERANCE = Decimal("0.001")

    # Default interval between batch orders (100ms)
    DEFAULT_ORDER_INTERVAL = 0.1

    def __init__(
        self,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: NotificationManager,
        bot_id: str,
        symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ):
        """
        Initialize GridOrderManager.

        Args:
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance
            bot_id: Unique bot identifier
            symbol: Trading pair (e.g., "BTCUSDT")
            market_type: Market type (SPOT or FUTURES)
        """
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier
        self._bot_id = bot_id
        self._symbol = symbol
        self._market_type = market_type

        # Grid setup (set via initialize)
        self._setup: Optional[GridSetup] = None

        # Mappings: level_index <-> order_id
        self._level_order_map: dict[int, str] = {}
        self._order_level_map: dict[str, int] = {}

        # Order cache: order_id -> Order
        self._orders: dict[str, Order] = {}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def setup(self) -> Optional[GridSetup]:
        """Get current grid setup."""
        return self._setup

    @property
    def symbol(self) -> str:
        """Get trading symbol."""
        return self._symbol

    @property
    def bot_id(self) -> str:
        """Get bot identifier."""
        return self._bot_id

    @property
    def active_order_count(self) -> int:
        """Get count of active orders."""
        return len(self._level_order_map)

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self, setup: GridSetup) -> None:
        """
        Initialize with grid setup.

        Args:
            setup: GridSetup with calculated grid levels
        """
        self._setup = setup

        # Clear mappings
        self._level_order_map.clear()
        self._order_level_map.clear()
        self._orders.clear()

        logger.info(
            f"GridOrderManager initialized: {self._symbol}, "
            f"{len(setup.levels)} levels, "
            f"range {setup.lower_price}-{setup.upper_price}"
        )

    # =========================================================================
    # Initial Order Placement
    # =========================================================================

    async def place_initial_orders(self) -> int:
        """
        Place initial orders for all grid levels.

        Orders are placed based on current price:
        - Levels above current price -> SELL LIMIT
        - Levels below current price -> BUY LIMIT
        - Levels near current price (within tolerance) -> Skip

        Returns:
            Number of orders successfully placed

        Raises:
            RuntimeError: If setup is not initialized
        """
        if self._setup is None:
            raise RuntimeError("GridOrderManager not initialized. Call initialize() first.")

        # Get current price
        current_price = await self._data_manager.get_price(
            self._symbol,
            self._market_type,
        )
        if current_price is None:
            current_price = await self._exchange.get_price(
                self._symbol,
                self._market_type,
            )

        logger.info(f"Placing initial orders at current price: {current_price}")

        # Prepare orders to place
        orders_to_place: list[dict] = []

        for level in self._setup.levels:
            # Skip if level already has order
            if level.index in self._level_order_map:
                continue

            # Determine side based on price relationship
            if self._is_price_near(level.price, current_price):
                # Skip levels near current price
                logger.debug(f"Skipping level {level.index} - near current price")
                continue

            if level.price > current_price:
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY

            # Calculate quantity from allocated amount
            quantity = level.allocated_amount / level.price

            orders_to_place.append({
                "level_index": level.index,
                "side": side,
                "price": level.price,
                "quantity": quantity,
            })

        # Batch place orders
        success_count = await self._batch_place_orders(orders_to_place)

        # Count buy/sell orders
        buy_count = sum(1 for o in orders_to_place if o["side"] == OrderSide.BUY)
        sell_count = sum(1 for o in orders_to_place if o["side"] == OrderSide.SELL)

        # Send notification
        await self._notify_initial_orders_placed(buy_count, sell_count, success_count)

        logger.info(
            f"Initial orders placed: {success_count} total "
            f"({buy_count} BUY, {sell_count} SELL)"
        )

        return success_count

    # =========================================================================
    # Single Level Order Operations
    # =========================================================================

    async def place_order_at_level(
        self,
        level_index: int,
        side: OrderSide | str,
    ) -> Optional[Order]:
        """
        Place an order at a specific grid level.

        If the level already has an order, it will be cancelled first.

        Args:
            level_index: Grid level index
            side: Order side (BUY or SELL)

        Returns:
            Order if successful, None otherwise

        Raises:
            RuntimeError: If setup is not initialized
            ValueError: If level_index is invalid
        """
        if self._setup is None:
            raise RuntimeError("GridOrderManager not initialized. Call initialize() first.")

        if level_index < 0 or level_index >= len(self._setup.levels):
            raise ValueError(f"Invalid level index: {level_index}")

        # Convert side to enum if string
        if isinstance(side, str):
            side = OrderSide(side.upper())

        level = self._setup.levels[level_index]

        # Cancel existing order if present
        if level_index in self._level_order_map:
            logger.debug(f"Cancelling existing order at level {level_index}")
            await self.cancel_order_at_level(level_index)

        # Calculate quantity from allocated amount
        quantity = level.allocated_amount / level.price

        # Round to exchange precision
        rounded_quantity = self._exchange.round_quantity(
            self._symbol,
            quantity,
            self._market_type,
        )
        rounded_price = self._exchange.round_price(
            self._symbol,
            level.price,
            self._market_type,
        )

        try:
            # Place order via exchange
            if side == OrderSide.BUY:
                order = await self._exchange.limit_buy(
                    self._symbol,
                    rounded_quantity,
                    rounded_price,
                    self._market_type,
                )
            else:
                order = await self._exchange.limit_sell(
                    self._symbol,
                    rounded_quantity,
                    rounded_price,
                    self._market_type,
                )

            # Update mappings
            self._level_order_map[level_index] = order.order_id
            self._order_level_map[order.order_id] = level_index
            self._orders[order.order_id] = order

            # Update level state
            if side == OrderSide.BUY:
                level.state = LevelState.PENDING_BUY
            else:
                level.state = LevelState.PENDING_SELL
            level.order_id = order.order_id

            # Save to database
            await self._data_manager.save_order(
                order,
                bot_id=self._bot_id,
                market_type=self._market_type,
            )

            logger.info(
                f"Order placed at level {level_index}: "
                f"{side.value} {rounded_quantity} @ {rounded_price}"
            )

            return order

        except Exception as e:
            logger.error(f"Failed to place order at level {level_index}: {e}")
            return None

    async def cancel_order_at_level(self, level_index: int) -> bool:
        """
        Cancel order at a specific grid level.

        Args:
            level_index: Grid level index

        Returns:
            True if cancelled successfully, False otherwise

        Raises:
            RuntimeError: If setup is not initialized
        """
        if self._setup is None:
            raise RuntimeError("GridOrderManager not initialized. Call initialize() first.")

        # Check if level has order
        if level_index not in self._level_order_map:
            logger.debug(f"No order at level {level_index}")
            return False

        order_id = self._level_order_map[level_index]

        try:
            # Cancel via exchange
            cancelled_order = await self._exchange.cancel_order(
                self._symbol,
                order_id,
                self._market_type,
            )

            # Update mappings
            del self._level_order_map[level_index]
            del self._order_level_map[order_id]
            self._orders[order_id] = cancelled_order

            # Update level state
            level = self._setup.levels[level_index]
            level.state = LevelState.EMPTY
            level.order_id = None

            # Update in database
            await self._data_manager.update_order(cancelled_order, bot_id=self._bot_id)

            logger.info(f"Order cancelled at level {level_index}: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order at level {level_index}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """
        Cancel all active grid orders.

        Returns:
            Number of orders successfully cancelled
        """
        if self._setup is None:
            return 0

        cancelled_count = 0
        level_indices = list(self._level_order_map.keys())

        for level_index in level_indices:
            if await self.cancel_order_at_level(level_index):
                cancelled_count += 1
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.05)

        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count

    # =========================================================================
    # Lookup Methods
    # =========================================================================

    def get_level_by_order_id(self, order_id: str) -> Optional[int]:
        """
        Get level index by order ID.

        Args:
            order_id: Exchange order ID

        Returns:
            Level index if found, None otherwise
        """
        return self._order_level_map.get(order_id)

    def get_order_by_level(self, level_index: int) -> Optional[Order]:
        """
        Get order by level index.

        Args:
            level_index: Grid level index

        Returns:
            Order if found, None otherwise
        """
        order_id = self._level_order_map.get(level_index)
        if order_id:
            return self._orders.get(order_id)
        return None

    def get_level(self, level_index: int) -> Optional[GridLevel]:
        """
        Get grid level by index.

        Args:
            level_index: Grid level index

        Returns:
            GridLevel if found, None otherwise
        """
        if self._setup is None:
            return None
        if 0 <= level_index < len(self._setup.levels):
            return self._setup.levels[level_index]
        return None

    # =========================================================================
    # Order Synchronization
    # =========================================================================

    async def sync_orders(self) -> dict[str, int]:
        """
        Synchronize local order state with exchange.

        Compares local orders with exchange open orders:
        - Exchange has, local missing: External order (ignored or warned)
        - Local has, exchange missing: Order filled/cancelled (mark for processing)

        Returns:
            Dict with sync statistics:
                - synced: Number of orders still active
                - filled: Number of orders no longer on exchange
                - external: Number of external orders detected
        """
        if self._setup is None:
            return {"synced": 0, "filled": 0, "external": 0}

        # Get open orders from exchange
        exchange_orders = await self._exchange.get_open_orders(
            self._symbol,
            self._market_type,
        )
        exchange_order_ids = {o.order_id for o in exchange_orders}

        # Local order IDs
        local_order_ids = set(self._level_order_map.values())

        # Find discrepancies
        still_active = local_order_ids & exchange_order_ids
        no_longer_active = local_order_ids - exchange_order_ids
        external_orders = exchange_order_ids - local_order_ids

        # Process orders no longer active (filled or cancelled)
        filled_levels: list[int] = []
        for order_id in no_longer_active:
            level_index = self._order_level_map.get(order_id)
            if level_index is not None:
                filled_levels.append(level_index)

                # Get latest order status from exchange
                try:
                    order = await self._exchange.get_order(
                        self._symbol,
                        order_id,
                        self._market_type,
                    )
                    self._orders[order_id] = order

                    # Update level state based on order status
                    level = self._setup.levels[level_index]
                    if order.status == OrderStatus.FILLED:
                        level.state = LevelState.FILLED
                        level.filled_quantity = order.filled_qty
                        level.filled_price = order.avg_price
                    else:
                        # Cancelled or other status
                        level.state = LevelState.EMPTY

                    level.order_id = None

                    # Remove from mappings
                    del self._level_order_map[level_index]
                    del self._order_level_map[order_id]

                    # Update in database
                    await self._data_manager.update_order(order, bot_id=self._bot_id)

                except Exception as e:
                    logger.warning(f"Failed to get order status for {order_id}: {e}")

        # Warn about external orders
        if external_orders:
            logger.warning(
                f"Detected {len(external_orders)} external orders for {self._symbol}. "
                f"These may interfere with grid operation."
            )

        stats = {
            "synced": len(still_active),
            "filled": len(filled_levels),
            "external": len(external_orders),
        }

        logger.info(
            f"Order sync complete: {stats['synced']} active, "
            f"{stats['filled']} filled, {stats['external']} external"
        )

        return stats

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _is_price_near(
        self,
        price1: Decimal,
        price2: Decimal,
        tolerance: Optional[Decimal] = None,
    ) -> bool:
        """
        Check if two prices are within tolerance.

        Args:
            price1: First price
            price2: Second price
            tolerance: Tolerance as decimal (default 0.001 = 0.1%)

        Returns:
            True if prices are within tolerance
        """
        if tolerance is None:
            tolerance = self.DEFAULT_PRICE_TOLERANCE

        if price2 == 0:
            return price1 == 0

        diff_ratio = abs(price1 - price2) / price2
        return diff_ratio <= tolerance

    async def _batch_place_orders(
        self,
        orders_to_place: list[dict],
        interval: float = DEFAULT_ORDER_INTERVAL,
    ) -> int:
        """
        Place multiple orders with interval between each.

        Args:
            orders_to_place: List of order specs with keys:
                - level_index: Grid level index
                - side: OrderSide (BUY or SELL)
                - price: Order price
                - quantity: Order quantity
            interval: Seconds between orders (default 0.1)

        Returns:
            Number of orders successfully placed
        """
        success_count = 0

        for order_spec in orders_to_place:
            level_index = order_spec["level_index"]
            side = order_spec["side"]
            price = order_spec["price"]
            quantity = order_spec["quantity"]

            # Round to exchange precision
            rounded_quantity = self._exchange.round_quantity(
                self._symbol,
                quantity,
                self._market_type,
            )
            rounded_price = self._exchange.round_price(
                self._symbol,
                price,
                self._market_type,
            )

            try:
                # Place order
                if side == OrderSide.BUY:
                    order = await self._exchange.limit_buy(
                        self._symbol,
                        rounded_quantity,
                        rounded_price,
                        self._market_type,
                    )
                else:
                    order = await self._exchange.limit_sell(
                        self._symbol,
                        rounded_quantity,
                        rounded_price,
                        self._market_type,
                    )

                # Update mappings
                self._level_order_map[level_index] = order.order_id
                self._order_level_map[order.order_id] = level_index
                self._orders[order.order_id] = order

                # Update level state
                level = self._setup.levels[level_index]
                if side == OrderSide.BUY:
                    level.state = LevelState.PENDING_BUY
                else:
                    level.state = LevelState.PENDING_SELL
                level.order_id = order.order_id

                # Save to database
                await self._data_manager.save_order(
                    order,
                    bot_id=self._bot_id,
                    market_type=self._market_type,
                )

                success_count += 1
                logger.debug(
                    f"Order placed: level={level_index}, "
                    f"side={side.value}, price={rounded_price}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to place order at level {level_index}: {e}"
                )

            # Wait between orders to avoid rate limiting
            if interval > 0:
                await asyncio.sleep(interval)

        return success_count

    async def _notify_initial_orders_placed(
        self,
        buy_count: int,
        sell_count: int,
        success_count: int,
    ) -> None:
        """
        Send notification for initial orders placed.

        Args:
            buy_count: Number of buy orders
            sell_count: Number of sell orders
            success_count: Number of successful orders
        """
        try:
            # Use notifier's generic send method
            message = (
                f"Grid initialized for {self._symbol}\n"
                f"Placed {success_count} orders: "
                f"{buy_count} BUY, {sell_count} SELL"
            )
            await self._notifier.send_info(
                title="Grid Orders Placed",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
