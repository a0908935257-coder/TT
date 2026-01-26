"""
Grid Order Manager.

Manages grid orders including initial placement, level-order mapping,
fill handling, reverse order placement, and profit tracking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from src.core import get_logger
from src.core.models import MarketType, Order, OrderSide, OrderStatus
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.notification import NotificationManager

from .models import GridLevel, GridSetup, LevelSide, LevelState

logger = get_logger(__name__)


@dataclass
class FilledRecord:
    """
    Record of a filled order.

    Tracks fill details for profit calculation and history.

    Example:
        >>> record = FilledRecord(
        ...     level_index=5,
        ...     side=OrderSide.BUY,
        ...     price=Decimal("50000"),
        ...     quantity=Decimal("0.1"),
        ...     fee=Decimal("5"),
        ...     timestamp=datetime.now(timezone.utc),
        ...     order_id="12345",
        ... )
    """

    level_index: int
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fee: Decimal
    timestamp: datetime
    order_id: str

    # Optional: linked record for profit calculation
    paired_record: Optional["FilledRecord"] = field(default=None, repr=False)

    @property
    def value(self) -> Decimal:
        """Total value of the fill (price * quantity)."""
        return self.price * self.quantity

    @property
    def net_value(self) -> Decimal:
        """Net value after fees."""
        if self.side == OrderSide.BUY:
            return self.value + self.fee  # Cost to buy
        else:
            return self.value - self.fee  # Revenue from sell


class GridOrderManager:
    """
    Grid Order Manager.

    Manages grid level orders including:
    - Initial order placement based on current price
    - Level-to-order and order-to-level mappings
    - Order cancellation at specific levels
    - Synchronization with exchange order state
    - Fill handling and reverse order placement
    - Profit tracking and statistics

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
        >>>
        >>> # Handle order fill event
        >>> await manager.on_order_filled(filled_order)
        >>> stats = manager.get_statistics()
    """

    # Default tolerance for price comparison (0.1%)
    DEFAULT_PRICE_TOLERANCE = Decimal("0.001")

    # Default interval between batch orders (100ms)
    DEFAULT_ORDER_INTERVAL = 0.1

    # Default fee rate (0.1%)
    DEFAULT_FEE_RATE = Decimal("0.001")

    # Default max order age in seconds (24 hours)
    DEFAULT_MAX_ORDER_AGE = 86400

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

        # Fill tracking (Prompt 20)
        self._filled_history: list[FilledRecord] = []
        self._total_profit: Decimal = Decimal("0")
        self._trade_count: int = 0

        # Track pending buy fills waiting for sell (for profit calculation)
        # Key: level_index, Value: FilledRecord
        self._pending_buy_fills: dict[int, FilledRecord] = {}

        # Lock for atomic buy-sell matching (prevents race conditions)
        # Initialized early to avoid race condition in lazy initialization
        self._match_lock: asyncio.Lock = asyncio.Lock()

        # Track order creation times for timeout handling
        # Key: order_id, Value: creation timestamp
        self._order_created_times: dict[str, datetime] = {}

        # Max order age for timeout (can be configured)
        self._max_order_age: int = self.DEFAULT_MAX_ORDER_AGE

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

        # Clear fill tracking
        self._filled_history.clear()
        self._total_profit = Decimal("0")
        self._trade_count = 0
        self._pending_buy_fills.clear()

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

        # Load symbol info for proper precision
        await self._exchange.get_symbol_info(self._symbol, self._market_type)

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

        # Get symbol info for validation
        symbol_info = await self._exchange.get_symbol_info(self._symbol, self._market_type)
        min_notional = symbol_info.min_notional
        min_quantity = symbol_info.min_quantity

        # Check base asset balance for SELL orders (SPOT market only)
        base_asset_balance = Decimal("0")
        quote_asset_balance = Decimal("0")
        if self._market_type == MarketType.SPOT:
            base_asset = symbol_info.base_asset
            quote_asset = symbol_info.quote_asset

            # Get base asset balance (for SELL orders)
            balance = await self._exchange.get_balance(base_asset, self._market_type)
            if balance:
                base_asset_balance = balance.free
            logger.info(f"Base asset ({base_asset}) balance: {base_asset_balance}")

            # Get quote asset balance (for BUY orders)
            balance = await self._exchange.get_balance(quote_asset, self._market_type)
            if balance:
                quote_asset_balance = balance.free
            logger.info(f"Quote asset ({quote_asset}) balance: {quote_asset_balance}")

        # Prepare orders to place
        orders_to_place: list[dict] = []
        skipped_sell_no_balance = 0
        skipped_buy_no_balance = 0
        skipped_low_notional = 0

        # Track cumulative required quote asset for BUY orders
        cumulative_buy_cost = Decimal("0")

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

            # Calculate quantity from allocated amount (with zero-division protection)
            if level.price <= 0:
                logger.error(f"Invalid level price at index {level.index}: {level.price}")
                continue
            quantity = level.allocated_amount / level.price

            # Round quantity for validation
            rounded_quantity = self._exchange.round_quantity(
                self._symbol, quantity, self._market_type
            )

            # Validate minimum quantity
            if rounded_quantity < min_quantity:
                logger.warning(
                    f"Skipping level {level.index} - quantity {rounded_quantity} "
                    f"below minimum {min_quantity}"
                )
                skipped_low_notional += 1
                continue

            # Validate minimum notional value
            notional = rounded_quantity * level.price
            if notional < min_notional:
                logger.warning(
                    f"Skipping level {level.index} - notional {notional:.4f} USDT "
                    f"below minimum {min_notional}"
                )
                skipped_low_notional += 1
                continue

            # For SELL orders in SPOT, check if we have enough base asset
            if side == OrderSide.SELL and self._market_type == MarketType.SPOT:
                if base_asset_balance < rounded_quantity:
                    logger.debug(
                        f"Skipping SELL at level {level.index} - "
                        f"insufficient {symbol_info.base_asset} balance"
                    )
                    skipped_sell_no_balance += 1
                    continue

            # For BUY orders in SPOT, check if we have enough quote asset
            if side == OrderSide.BUY and self._market_type == MarketType.SPOT:
                order_cost = rounded_quantity * level.price
                if cumulative_buy_cost + order_cost > quote_asset_balance:
                    logger.warning(
                        f"Skipping BUY at level {level.index} - "
                        f"insufficient {symbol_info.quote_asset} balance "
                        f"(need {cumulative_buy_cost + order_cost:.2f}, have {quote_asset_balance:.2f})"
                    )
                    skipped_buy_no_balance += 1
                    continue
                cumulative_buy_cost += order_cost

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

        # Log results
        logger.info(
            f"Initial orders placed: {success_count} total "
            f"({buy_count} BUY, {sell_count} SELL)"
        )
        if skipped_sell_no_balance > 0:
            logger.info(
                f"Skipped {skipped_sell_no_balance} SELL orders "
                f"(insufficient base asset balance)"
            )
        if skipped_buy_no_balance > 0:
            logger.info(
                f"Skipped {skipped_buy_no_balance} BUY orders "
                f"(insufficient quote asset balance, available: {quote_asset_balance:.2f} USDT)"
            )
        if skipped_low_notional > 0:
            logger.info(
                f"Skipped {skipped_low_notional} orders "
                f"(below minimum notional {min_notional} USDT)"
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

        # Calculate quantity from allocated amount (with zero-division protection)
        if level.price <= 0:
            raise ValueError(f"Invalid level price at index {level_index}: {level.price}")
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
            # Place order via exchange with bot_id for tracking
            if side == OrderSide.BUY:
                order = await self._exchange.limit_buy(
                    self._symbol,
                    rounded_quantity,
                    rounded_price,
                    self._market_type,
                    bot_id=self._bot_id,
                )
            else:
                order = await self._exchange.limit_sell(
                    self._symbol,
                    rounded_quantity,
                    rounded_price,
                    self._market_type,
                    bot_id=self._bot_id,
                )

            # Update mappings
            self._level_order_map[level_index] = order.order_id
            self._order_level_map[order.order_id] = level_index
            self._orders[order.order_id] = order
            self._order_created_times[order.order_id] = datetime.now(timezone.utc)

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

            # Update mappings (use pop to avoid KeyError in concurrent scenarios)
            self._level_order_map.pop(level_index, None)
            self._order_level_map.pop(order_id, None)
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

    def get_order_mapping(self) -> dict[str, int]:
        """
        Get current order-to-level mapping for persistence.

        Returns:
            Dict of order_id -> level_index
        """
        return self._order_level_map.copy()

    def restore_order_mapping(self, mapping: dict[str, int]) -> int:
        """
        Restore order-to-level mapping from saved state.

        This allows the bot to recognize its own orders after restart.

        Args:
            mapping: Dict of order_id -> level_index

        Returns:
            Number of orders restored
        """
        restored = 0
        for order_id, level_index in mapping.items():
            # Validate level index bounds
            if self._setup and 0 <= level_index < len(self._setup.levels):
                # Check if level already has a different order mapped
                if level_index in self._level_order_map:
                    old_order_id = self._level_order_map[level_index]
                    if old_order_id != order_id:
                        logger.warning(
                            f"Level {level_index} already has order {old_order_id[:8]}..., "
                            f"replacing with {order_id[:8]}..."
                        )
                        # Remove old mapping and cached order
                        self._order_level_map.pop(old_order_id, None)
                        self._orders.pop(old_order_id, None)

                self._order_level_map[order_id] = level_index
                self._level_order_map[level_index] = order_id
                restored += 1
                logger.debug(f"Restored order mapping: {order_id[:8]}... -> level {level_index}")

        logger.info(f"Restored {restored} order mappings from saved state")
        return restored

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

        # Get open orders from exchange - ONLY this bot's orders
        exchange_orders = await self._exchange.get_open_orders_for_bot(
            bot_id=self._bot_id,
            symbol=self._symbol,
            market=self._market_type,
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
                        # Process as filled order to place reverse orders
                        # This handles orders that filled while bot was offline
                        await self.on_order_filled(order)
                        logger.info(
                            f"Processed filled order during sync: {order_id} at level {level_index}"
                        )
                    else:
                        # Cancelled or other status
                        level.state = LevelState.EMPTY
                        level.order_id = None

                        # Remove from mappings
                        if level_index in self._level_order_map:
                            del self._level_order_map[level_index]
                        if order_id in self._order_level_map:
                            del self._order_level_map[order_id]

                        # Update in database
                        await self._data_manager.update_order(order, bot_id=self._bot_id)

                except Exception as e:
                    logger.error(
                        f"Failed to get order status for {order_id}: {e}. "
                        "Order state may be inconsistent - manual check recommended."
                    )
                    # Mark level as needing attention but don't corrupt state
                    if level_index is not None and level_index < len(self._setup.levels):
                        level = self._setup.levels[level_index]
                        level.state = LevelState.EMPTY
                        level.order_id = None
                        # Clean up mappings to allow retry
                        self._level_order_map.pop(level_index, None)
                        self._order_level_map.pop(order_id, None)

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
                # Place order with bot_id for tracking
                if side == OrderSide.BUY:
                    order = await self._exchange.limit_buy(
                        self._symbol,
                        rounded_quantity,
                        rounded_price,
                        self._market_type,
                        bot_id=self._bot_id,
                    )
                else:
                    order = await self._exchange.limit_sell(
                        self._symbol,
                        rounded_quantity,
                        rounded_price,
                        self._market_type,
                        bot_id=self._bot_id,
                    )

                # Update mappings
                self._level_order_map[level_index] = order.order_id
                self._order_level_map[order.order_id] = level_index
                self._orders[order.order_id] = order
                self._order_created_times[order.order_id] = datetime.now(timezone.utc)

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

    # =========================================================================
    # Fill Handling (Prompt 20)
    # =========================================================================

    async def on_order_filled(self, order: Order) -> Optional[Order]:
        """
        Handle order fill event.

        When an order is filled:
        1. Record the fill in history
        2. Update level state
        3. Place reverse order at appropriate level
        4. Calculate profit if completing a round trip
        5. Send notification
        6. Save to database

        Args:
            order: The filled Order

        Returns:
            The reverse order if placed, None otherwise
        """
        if self._setup is None:
            logger.warning("Cannot handle fill - not initialized")
            return None

        # Find level index for this order
        level_index = self.get_level_by_order_id(order.order_id)
        if level_index is None:
            logger.warning(f"Order {order.order_id} not found in level mappings")
            return None

        # Validate level index bounds
        if level_index < 0 or level_index >= len(self._setup.levels):
            logger.error(f"Invalid level index {level_index} for order {order.order_id}")
            return None

        level = self._setup.levels[level_index]

        # Determine fill price and quantity with validation
        fill_price = order.avg_price if order.avg_price else order.price
        if fill_price is None or fill_price <= 0:
            logger.error(f"Order {order.order_id} has invalid fill price: {fill_price}")
            return None

        fill_qty = order.filled_qty
        if fill_qty is None or fill_qty <= 0:
            logger.error(f"Order {order.order_id} has invalid fill quantity: {fill_qty}")
            return None

        fee = order.fee if order.fee else fill_price * fill_qty * self.DEFAULT_FEE_RATE

        # Create fill record
        fill_record = FilledRecord(
            level_index=level_index,
            side=OrderSide(order.side) if isinstance(order.side, str) else order.side,
            price=fill_price,
            quantity=fill_qty,
            fee=fee,
            timestamp=datetime.now(timezone.utc),
            order_id=order.order_id,
        )

        # Add to history
        self._filled_history.append(fill_record)

        # Update level state
        level.state = LevelState.FILLED
        level.filled_quantity = fill_qty
        level.filled_price = fill_price

        # Remove from active mappings (use pop to avoid KeyError in concurrent scenarios)
        self._level_order_map.pop(level_index, None)
        self._order_level_map.pop(order.order_id, None)

        # Update order cache
        self._orders[order.order_id] = order

        # Calculate profit if completing a round trip
        profit = Decimal("0")
        filled_side = OrderSide(order.side) if isinstance(order.side, str) else order.side

        if filled_side == OrderSide.BUY:
            # Store buy fill for later profit calculation (protected by lock)
            async with self._match_lock:
                self._pending_buy_fills[level_index] = fill_record
        elif filled_side == OrderSide.SELL:
            # Check if we have a matching buy fill to calculate profit
            # Look for buy fills at lower levels
            # Note: _find_matching_buy_fill uses lock to atomically pair records
            matching_buy = await self._find_matching_buy_fill(level_index, fill_record)
            if matching_buy:
                profit = self.calculate_profit(matching_buy, fill_record)
                self._total_profit += profit
                self._trade_count += 1

                logger.info(
                    f"Trade completed: profit={profit:.4f}, "
                    f"total_profit={self._total_profit:.4f}, "
                    f"trade_count={self._trade_count}"
                )

        # Update order in database
        await self._data_manager.update_order(order, bot_id=self._bot_id)

        # Place reverse order
        reverse_order = await self.place_reverse_order(level_index, filled_side)

        # Send notification
        await self._notify_order_filled(fill_record, profit, reverse_order)

        logger.info(
            f"Order filled at level {level_index}: "
            f"{filled_side.value} {fill_qty} @ {fill_price}"
        )

        return reverse_order

    async def place_reverse_order(
        self,
        level_index: int,
        filled_side: OrderSide | str,
    ) -> Optional[Order]:
        """
        Place reverse order after a fill.

        BUY filled -> Place SELL at level_index + 1 (upper level)
        SELL filled -> Place BUY at level_index - 1 (lower level)

        Args:
            level_index: Index of the filled level
            filled_side: Side of the filled order

        Returns:
            The reverse Order if placed, None otherwise
        """
        if self._setup is None:
            return None

        # Convert side to enum if string
        if isinstance(filled_side, str):
            filled_side = OrderSide(filled_side.upper())

        # Determine target level and side for reverse order
        if filled_side == OrderSide.BUY:
            # BUY filled -> place SELL at upper level
            target_level_index = level_index + 1
            reverse_side = OrderSide.SELL

            # Check boundary
            if target_level_index >= len(self._setup.levels):
                logger.info(
                    f"Level {level_index} is highest level - no reverse order placed"
                )
                return None

        else:  # SELL filled
            # SELL filled -> place BUY at lower level
            target_level_index = level_index - 1
            reverse_side = OrderSide.BUY

            # Check boundary
            if target_level_index < 0:
                logger.info(
                    f"Level {level_index} is lowest level - no reverse order placed"
                )
                return None

        # Place the reverse order
        logger.info(
            f"Placing reverse order: {reverse_side.value} at level {target_level_index}"
        )
        return await self.place_order_at_level(target_level_index, reverse_side)

    def calculate_profit(
        self,
        buy_record: FilledRecord,
        sell_record: FilledRecord,
    ) -> Decimal:
        """
        Calculate profit from a buy-sell round trip.

        Profit = (sell_price - buy_price) × quantity - fees

        Args:
            buy_record: The buy fill record
            sell_record: The sell fill record

        Returns:
            Net profit after fees
        """
        # Use the smaller quantity (in case of partial fills)
        quantity = min(buy_record.quantity, sell_record.quantity)

        # Guard against zero quantity
        if quantity <= 0:
            logger.warning(
                f"Zero quantity in profit calculation: buy={buy_record.quantity}, "
                f"sell={sell_record.quantity}"
            )
            return Decimal("0")

        # Gross profit
        gross_profit = (sell_record.price - buy_record.price) * quantity

        # Total fees (proportional to quantity used)
        buy_fee = buy_record.fee * (quantity / buy_record.quantity) if buy_record.quantity > 0 else Decimal("0")
        sell_fee = sell_record.fee * (quantity / sell_record.quantity) if sell_record.quantity > 0 else Decimal("0")
        total_fees = buy_fee + sell_fee

        # Net profit
        net_profit = gross_profit - total_fees

        logger.debug(
            f"Profit calculation: ({sell_record.price} - {buy_record.price}) × {quantity} "
            f"- {total_fees} = {net_profit}"
        )

        return net_profit

    def get_statistics(self) -> dict:
        """
        Get grid trading statistics.

        Returns:
            Dict with:
                - total_profit: Cumulative profit
                - trade_count: Completed round trips
                - buy_filled_count: Total buy fills
                - sell_filled_count: Total sell fills
                - pending_buy_count: Active buy orders
                - pending_sell_count: Active sell orders
                - avg_profit_per_trade: Average profit per trade
                - total_fees: Total fees paid
        """
        # Count fills by side
        buy_filled_count = sum(
            1 for r in self._filled_history if r.side == OrderSide.BUY
        )
        sell_filled_count = sum(
            1 for r in self._filled_history if r.side == OrderSide.SELL
        )

        # Count pending orders by side
        pending_buy_count = 0
        pending_sell_count = 0
        if self._setup:
            for level in self._setup.levels:
                if level.state == LevelState.PENDING_BUY:
                    pending_buy_count += 1
                elif level.state == LevelState.PENDING_SELL:
                    pending_sell_count += 1

        # Calculate total fees
        total_fees = sum(r.fee for r in self._filled_history)

        # Average profit per trade
        avg_profit = (
            self._total_profit / Decimal(self._trade_count)
            if self._trade_count > 0
            else Decimal("0")
        )

        return {
            "total_profit": self._total_profit,
            "trade_count": self._trade_count,
            "buy_filled_count": buy_filled_count,
            "sell_filled_count": sell_filled_count,
            "pending_buy_count": pending_buy_count,
            "pending_sell_count": pending_sell_count,
            "avg_profit_per_trade": avg_profit,
            "total_fees": total_fees,
        }

    # =========================================================================
    # WebSocket Event Handlers (Prompt 20)
    # =========================================================================

    async def handle_order_update(self, order: Order) -> None:
        """
        Handle WebSocket order update event.

        This should be registered as a callback for user data stream.

        Args:
            order: Updated Order from WebSocket
        """
        # Only process orders we're tracking
        if order.order_id not in self._orders:
            return

        # Update cached order
        self._orders[order.order_id] = order

        # Handle based on status
        if order.status == OrderStatus.FILLED:
            await self.on_order_filled(order)
        elif order.status == OrderStatus.CANCELED:
            await self._handle_order_canceled(order)
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            # Update order state for partial fills
            level_index = self.get_level_by_order_id(order.order_id)
            if level_index is not None and self._setup:
                level = self._setup.levels[level_index]
                level.filled_quantity = order.filled_qty
                level.filled_price = order.avg_price if order.avg_price else order.price
                level.state = LevelState.PARTIAL
            logger.info(
                f"Order {order.order_id} partially filled: {order.filled_qty}/{order.quantity} "
                f"at level {level_index}"
            )

    async def _handle_order_canceled(self, order: Order) -> None:
        """
        Handle order cancellation event.

        Args:
            order: Cancelled Order
        """
        level_index = self.get_level_by_order_id(order.order_id)
        if level_index is None:
            return

        # Update level state
        if self._setup:
            level = self._setup.levels[level_index]
            level.state = LevelState.EMPTY
            level.order_id = None

        # Remove from mappings (use pop to avoid KeyError in concurrent scenarios)
        self._level_order_map.pop(level_index, None)
        self._order_level_map.pop(order.order_id, None)

        # Update in database
        await self._data_manager.update_order(order, bot_id=self._bot_id)

        logger.info(f"Order cancelled at level {level_index}: {order.order_id}")

    # =========================================================================
    # Fill Handling Helpers
    # =========================================================================

    async def _find_matching_buy_fill(self, sell_level_index: int, sell_record: FilledRecord) -> Optional[FilledRecord]:
        """
        Find a matching buy fill for profit calculation.

        When a SELL is filled at level N, look for a BUY fill at level N-1
        (the level below, where buy should have been placed).

        This method uses a lock to atomically pair the matched buy with the sell,
        preventing race conditions where multiple sells match the same buy.

        Args:
            sell_level_index: Level index where sell was filled
            sell_record: The sell FilledRecord to pair with

        Returns:
            Matching buy FilledRecord if found, None otherwise
        """
        async with self._match_lock:
            # Strategy 1: Look for pending buy fill at the level below (most common case)
            buy_level_index = sell_level_index - 1
            if buy_level_index >= 0 and buy_level_index in self._pending_buy_fills:
                buy_record = self._pending_buy_fills.pop(buy_level_index)
                # Check if already paired (safety check)
                if buy_record.paired_record is not None:
                    logger.warning(
                        f"Buy record at level {buy_level_index} already paired, skipping"
                    )
                else:
                    # Atomically pair both records (under lock protection)
                    buy_record.paired_record = sell_record
                    sell_record.paired_record = buy_record
                    return buy_record

            # Strategy 2: Search nearby levels (for dynamic grid adjustments)
            # Check levels within a range below the sell level
            for offset in range(2, min(5, sell_level_index + 1)):
                nearby_level = sell_level_index - offset
                if nearby_level >= 0 and nearby_level in self._pending_buy_fills:
                    buy_record = self._pending_buy_fills.pop(nearby_level)
                    # Check if already paired (safety check)
                    if buy_record.paired_record is not None:
                        logger.warning(
                            f"Buy record at level {nearby_level} already paired, skipping"
                        )
                        continue
                    buy_record.paired_record = sell_record
                    sell_record.paired_record = buy_record
                    logger.debug(
                        f"Matched buy at level {nearby_level} with sell at level {sell_level_index}"
                    )
                    return buy_record

            # Fallback: find any unpaired buy fill at a lower price
            for record in reversed(self._filled_history):
                if (
                    record.side == OrderSide.BUY
                    and record.paired_record is None
                    and record.level_index < sell_level_index
                ):
                    # Atomically pair (under lock protection)
                    record.paired_record = sell_record
                    sell_record.paired_record = record
                    # Also remove from _pending_buy_fills if present to prevent double pairing
                    self._pending_buy_fills.pop(record.level_index, None)
                    return record

            return None

    async def _notify_order_filled(
        self,
        fill_record: FilledRecord,
        profit: Decimal,
        reverse_order: Optional[Order],
    ) -> None:
        """
        Send notification for order fill.

        Args:
            fill_record: The fill record
            profit: Profit from this trade (0 if not completing round trip)
            reverse_order: The reverse order placed (if any)
        """
        try:
            side_str = "BUY" if fill_record.side == OrderSide.BUY else "SELL"
            message = (
                f"{self._symbol} {side_str} filled\n"
                f"Price: {fill_record.price}\n"
                f"Qty: {fill_record.quantity}\n"
                f"Level: {fill_record.level_index}"
            )

            if profit != 0:
                message += f"\nProfit: {profit:.4f}"
                message += f"\nTotal: {self._total_profit:.4f}"

            if reverse_order:
                reverse_side = "SELL" if fill_record.side == OrderSide.BUY else "BUY"
                message += f"\nReverse {reverse_side} order placed"

            await self._notifier.send_success(
                title="Order Filled",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send fill notification: {e}")

    # =========================================================================
    # Order Timeout Handling
    # =========================================================================

    async def check_stale_orders(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Check for and cancel stale orders that have exceeded the maximum age.

        This helps prevent old orders from executing at outdated prices
        after significant market movements.

        Args:
            max_age_seconds: Maximum order age in seconds (uses default if None)

        Returns:
            Number of stale orders cancelled
        """
        if not self._setup:
            return 0

        max_age = max_age_seconds or self._max_order_age
        now = datetime.now(timezone.utc)
        cancelled_count = 0
        stale_orders: list[tuple[int, str]] = []

        # Find stale orders
        for order_id, created_time in list(self._order_created_times.items()):
            age_seconds = (now - created_time).total_seconds()
            if age_seconds > max_age:
                level_index = self._order_level_map.get(order_id)
                if level_index is not None:
                    stale_orders.append((level_index, order_id))
                    logger.warning(
                        f"Order {order_id} at level {level_index} is stale "
                        f"(age: {age_seconds:.0f}s > {max_age}s)"
                    )

        # Cancel stale orders
        for level_index, order_id in stale_orders:
            try:
                if await self.cancel_order_at_level(level_index):
                    cancelled_count += 1
                    # Clean up timestamp
                    self._order_created_times.pop(order_id, None)
                    logger.info(f"Cancelled stale order {order_id} at level {level_index}")
            except Exception as e:
                logger.error(f"Failed to cancel stale order {order_id}: {e}")

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} stale orders")

        return cancelled_count

    def set_max_order_age(self, max_age_seconds: int) -> None:
        """
        Set the maximum order age for timeout detection.

        Args:
            max_age_seconds: Maximum age in seconds
        """
        self._max_order_age = max_age_seconds
        logger.info(f"Max order age set to {max_age_seconds} seconds")

    def _cleanup_order_timestamp(self, order_id: str) -> None:
        """
        Clean up order timestamp when order is removed.

        Args:
            order_id: Order ID to clean up
        """
        self._order_created_times.pop(order_id, None)
