"""
Grid Trading Bot.

GridBot implements grid trading strategy by inheriting from BaseBot.
It places buy orders below current price and sell orders above,
profiting from price oscillations within the grid range.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.core.models import Kline, MarketType, Order, OrderSide, OrderType, OrderStatus
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.master.models import BotState
from src.notification import NotificationManager

from .calculator import SmartGridCalculator
from .models import ATRConfig, DynamicAdjustConfig, GridConfig, GridLevel, GridSetup, GridType, RiskLevel
from .order_manager import FilledRecord, GridOrderManager
from .risk_manager import GridRiskManager, RiskConfig

logger = get_logger(__name__)


@dataclass
class GridBotConfig:
    """
    Grid Bot configuration.

    Example:
        >>> config = GridBotConfig(
        ...     symbol="BTCUSDT",
        ...     market_type=MarketType.SPOT,
        ...     total_investment=Decimal("10000"),
        ...     risk_level=RiskLevel.MODERATE,
        ... )
    """

    # Required
    symbol: str
    market_type: MarketType = MarketType.SPOT
    total_investment: Decimal = field(default_factory=lambda: Decimal("1000"))

    # Grid calculation
    risk_level: RiskLevel = RiskLevel.MODERATE
    grid_type: GridType = GridType.ARITHMETIC

    # Manual overrides (optional)
    manual_upper: Optional[Decimal] = None
    manual_lower: Optional[Decimal] = None
    manual_grid_count: Optional[int] = None

    # Grid limits
    min_order_value: Decimal = field(default_factory=lambda: Decimal("10"))
    min_grid_count: int = 5
    max_grid_count: int = 50

    # Risk configuration
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # Dynamic adjustment configuration
    dynamic_adjust: DynamicAdjustConfig = field(default_factory=DynamicAdjustConfig)

    # ATR configuration (unified)
    atr_config: ATRConfig = field(default_factory=ATRConfig)

    # Kline settings for data fetching
    kline_limit: int = 100

    # State persistence
    save_interval_minutes: int = 5

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.total_investment, Decimal):
            self.total_investment = Decimal(str(self.total_investment))
        if self.manual_upper and not isinstance(self.manual_upper, Decimal):
            self.manual_upper = Decimal(str(self.manual_upper))
        if self.manual_lower and not isinstance(self.manual_lower, Decimal):
            self.manual_lower = Decimal(str(self.manual_lower))

    @property
    def has_manual_range(self) -> bool:
        """Check if manual price range is set."""
        return self.manual_upper is not None and self.manual_lower is not None


class GridBot(BaseBot):
    """
    Grid Trading Bot.

    Inherits from BaseBot and implements grid trading strategy.

    Components:
    - SmartGridCalculator: Calculate grid levels based on ATR
    - GridOrderManager: Manage orders and fills
    - GridRiskManager: Monitor risk and control bot state

    Example:
        >>> config = GridBotConfig(
        ...     symbol="BTCUSDT",
        ...     total_investment=Decimal("10000"),
        ... )
        >>> bot = GridBot(
        ...     bot_id="grid_btc_001",
        ...     config=config,
        ...     exchange=exchange,
        ...     data_manager=data_manager,
        ...     notifier=notifier,
        ... )
        >>> await bot.start()
        >>> status = bot.get_status()
        >>> await bot.stop()
    """

    def __init__(
        self,
        bot_id: str,
        config: GridBotConfig,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: NotificationManager,
        heartbeat_callback: Optional[Callable] = None,
    ):
        """
        Initialize GridBot.

        Args:
            bot_id: Unique bot identifier
            config: GridBotConfig instance
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance
            heartbeat_callback: Optional callback for sending heartbeats to Master
        """
        # Call parent constructor
        super().__init__(
            bot_id=bot_id,
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
            heartbeat_callback=heartbeat_callback,
        )

        # Grid-specific components (initialized in _do_start)
        self._calculator: Optional[SmartGridCalculator] = None
        self._order_manager: Optional[GridOrderManager] = None
        self._risk_manager: Optional[GridRiskManager] = None

        # Grid setup
        self._setup: Optional[GridSetup] = None

        # Persistence task
        self._save_task: Optional[asyncio.Task] = None

        # Last trade profit (for risk manager callback)
        self._last_trade_profit: Decimal = Decimal("0")
        self._previous_profit: Decimal = Decimal("0")

        # Lock for profit tracking to prevent race conditions
        self._profit_lock: asyncio.Lock = asyncio.Lock()

    # =========================================================================
    # Abstract Properties Implementation
    # =========================================================================

    @property
    def bot_type(self) -> str:
        """Return bot type identifier."""
        return "grid"

    @property
    def symbol(self) -> str:
        """Return trading symbol."""
        return self._config.symbol

    # =========================================================================
    # Additional Properties
    # =========================================================================

    @property
    def setup(self) -> Optional[GridSetup]:
        """Get grid setup."""
        return self._setup

    @property
    def calculator(self) -> Optional[SmartGridCalculator]:
        """Get grid calculator."""
        return self._calculator

    @property
    def order_manager(self) -> Optional[GridOrderManager]:
        """Get order manager."""
        return self._order_manager

    @property
    def risk_manager(self) -> Optional[GridRiskManager]:
        """Get risk manager."""
        return self._risk_manager

    # =========================================================================
    # Abstract Lifecycle Methods Implementation
    # =========================================================================

    async def _do_start(self) -> None:
        """
        Grid Bot start logic.

        Flow:
        1. Validate configuration
        2. Fetch kline data for ATR calculation
        3. Initialize calculator and compute grid
        4. Initialize order manager
        5. Initialize risk manager
        6. Place initial orders
        7. Subscribe to user data stream
        8. Start risk monitoring
        """
        # Step 1: Validate config
        self._validate_config()

        # Step 2: Fetch kline data for ATR calculation
        klines = await self._fetch_klines()
        if not klines:
            raise RuntimeError("Failed to fetch kline data")

        # Step 3: Initialize calculator
        grid_config = self._create_grid_config()
        self._calculator = SmartGridCalculator(
            config=grid_config,
            klines=klines,
        )

        # Step 4: Calculate grid
        self._setup = self._calculator.calculate()
        logger.info(
            f"Grid calculated: {self._setup.grid_count} levels, "
            f"range {self._setup.lower_price}-{self._setup.upper_price}"
        )

        # Step 5: Initialize order manager
        self._order_manager = GridOrderManager(
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=self._notifier,
            bot_id=self._bot_id,
            symbol=self._config.symbol,
            market_type=self._config.market_type,
        )
        self._order_manager.initialize(self._setup)

        # Step 6: Initialize risk manager
        self._risk_manager = GridRiskManager(
            order_manager=self._order_manager,
            notifier=self._notifier,
            config=self._config.risk_config,
        )

        # Configure dynamic adjustment
        if self._config.dynamic_adjust.enabled:
            self._risk_manager.set_dynamic_adjust_config(self._config.dynamic_adjust)
            self._risk_manager.set_rebuild_callback(self._rebuild_grid)
            logger.info("Dynamic grid adjustment enabled")

        # Step 7: Subscribe to user data stream BEFORE placing orders
        # This ensures we don't miss any fill events
        await self._subscribe_user_data()

        # Step 8: Try to restore previous order mapping
        restored_orders = await self._try_restore_orders()

        # Step 9: Place initial orders (only if no orders restored)
        if restored_orders == 0:
            placed = await self._order_manager.place_initial_orders()
            logger.info(f"Placed {placed} initial orders")
        else:
            logger.info(f"Restored {restored_orders} orders from previous session")

        # Step 10: Subscribe to K-line close events for dynamic adjustment
        if self._config.dynamic_adjust.enabled:
            await self._subscribe_kline_close()

        # Step 11: Start risk monitoring
        await self._risk_manager.start_monitoring()

        # Step 12: Start persistence task
        self._start_save_task()

        # Step 13: Send notification
        await self._notify_bot_started()

    async def _do_stop(self, clear_position: bool = False) -> None:
        """
        Grid Bot stop logic.

        Args:
            clear_position: If True, market sell all positions
        """
        # Stop persistence task
        await self._stop_save_task()

        # Stop risk monitoring
        if self._risk_manager:
            await self._risk_manager.stop_monitoring()

        # Cancel all orders
        cancelled = 0
        if self._order_manager:
            cancelled = await self._order_manager.cancel_all_orders()
            logger.info(f"Cancelled {cancelled} orders")

        # Clear position if requested
        if clear_position and self._order_manager:
            await self._clear_position()

        # Unsubscribe from streams
        await self._unsubscribe_user_data()
        if self._config.dynamic_adjust.enabled:
            await self._unsubscribe_kline_close()

        # Get final statistics
        stats = self.get_statistics()

        # Save final state
        await self._save_state()

        # Send notification
        await self._notify_bot_stopped(stats, clear_position)

    async def _do_pause(self) -> None:
        """
        Grid Bot pause logic.

        Cancels all pending orders but keeps positions.
        """
        if self._order_manager:
            cancelled = await self._order_manager.cancel_all_orders()
            logger.info(f"Paused: cancelled {cancelled} orders")

    async def _do_resume(self) -> None:
        """
        Grid Bot resume logic.

        Re-places orders based on current grid state.
        """
        if self._order_manager:
            placed = await self._order_manager.place_initial_orders()
            logger.info(f"Resumed: placed {placed} orders")

    def _get_extra_status(self) -> Dict[str, Any]:
        """
        Return grid-specific status fields.

        Returns:
            Dictionary with grid-specific metrics
        """
        extra: Dict[str, Any] = {}

        # Grid info
        if self._setup:
            extra.update({
                "upper_price": str(self._setup.upper_price),
                "lower_price": str(self._setup.lower_price),
                "current_price": str(self._setup.current_price),
                "grid_count": self._setup.grid_count,
                "grid_spacing": f"{self._setup.grid_spacing_percent:.2f}%",
                "grid_version": self._setup.version,
                "atr_multiplier": self._setup.atr_data.multiplier,
                "atr_period": self._setup.atr_data.period,
                "atr_timeframe": self._setup.atr_data.timeframe,
            })

        # Order info
        if self._order_manager:
            stats = self._order_manager.get_statistics()
            extra.update({
                "pending_buy_orders": stats.get("pending_buy_count", 0),
                "pending_sell_orders": stats.get("pending_sell_count", 0),
                "total_orders_placed": self._order_manager.active_order_count,
            })

        # Risk status
        if self._risk_manager:
            extra.update({
                "daily_pnl": str(self._risk_manager.daily_pnl),
                "consecutive_losses": self._risk_manager.consecutive_losses,
                "last_breakout": (
                    self._risk_manager.last_breakout.direction.value
                    if self._risk_manager.last_breakout
                    else "none"
                ),
            })

            # Dynamic adjustment status
            rebuilds_used = self._risk_manager.rebuilds_in_cooldown_period
            max_rebuilds = self._config.dynamic_adjust.max_rebuilds
            rebuilds_remaining = max(0, max_rebuilds - rebuilds_used)
            is_in_cooldown = rebuilds_used >= max_rebuilds

            rebuild_history = self._risk_manager.rebuild_history
            last_rebuild_time = (
                rebuild_history[-1].timestamp if rebuild_history else None
            )

            extra.update({
                "auto_rebuild_enabled": self._config.dynamic_adjust.enabled,
                "rebuilds_used": rebuilds_used,
                "rebuilds_remaining": rebuilds_remaining,
                "cooldown_days": self._config.dynamic_adjust.cooldown_days,
                "is_in_cooldown": is_in_cooldown,
                "last_rebuild_time": last_rebuild_time.isoformat() if last_rebuild_time else None,
                "next_rebuild_available": (
                    self._risk_manager.next_rebuild_available.isoformat()
                    if self._risk_manager.next_rebuild_available
                    else None
                ),
            })

        return extra

    async def _extra_health_checks(self) -> Dict[str, bool]:
        """
        Perform grid-specific health checks.

        Returns:
            Dictionary of check results
        """
        checks = {}

        # Check orders are synced with exchange
        if self._order_manager:
            try:
                sync_result = await self._order_manager.sync_orders()
                checks["orders_synced"] = sync_result.get("synced", 0) >= 0
            except Exception:
                checks["orders_synced"] = False

        # Check price is within grid range
        if self._setup:
            try:
                ticker = await self._exchange.get_ticker(self._config.symbol)
                current_price = Decimal(str(ticker.get("price", 0)))
                checks["within_range"] = (
                    self._setup.lower_price <= current_price <= self._setup.upper_price
                )
            except Exception:
                checks["within_range"] = False

        return checks

    # =========================================================================
    # Override heartbeat metrics for grid-specific data
    # =========================================================================

    def _get_heartbeat_metrics(self) -> Dict[str, Any]:
        """Get metrics to include in heartbeat."""
        metrics = {
            "uptime_seconds": self._get_uptime_seconds(),
            "total_trades": self._stats.total_trades,
        }

        # Use order_manager as single source of truth for profit (avoid double counting)
        if self._order_manager:
            stats = self._order_manager.get_statistics()
            metrics.update({
                "total_profit": float(stats.get("total_profit", 0)),
                "pending_buy_orders": stats.get("pending_buy_count", 0),
                "pending_sell_orders": stats.get("pending_sell_count", 0),
            })
        else:
            metrics["total_profit"] = 0.0

        return metrics

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def on_order_filled(self, order: Order) -> None:
        """
        Handle order fill event.

        Called when an order is filled via WebSocket.

        Args:
            order: Filled order
        """
        if not self._order_manager:
            return

        # Use lock to prevent race conditions in profit tracking
        # on_order_filled must be inside lock so concurrent fills don't corrupt delta calculation
        async with self._profit_lock:
            # Delegate to order manager
            await self._order_manager.on_order_filled(order)
            # Get profit from last trade (if completed)
            stats = self._order_manager.get_statistics()
            current_profit = stats.get("total_profit", Decimal("0"))

            # Calculate profit from this trade
            trade_profit = current_profit - self._previous_profit
            self._last_trade_profit = trade_profit

            # Record trade in stats
            if trade_profit != 0:
                fee = stats.get("last_fee", Decimal("0"))
                self._stats.record_trade(trade_profit, fee)

            # Notify risk manager
            if self._risk_manager and trade_profit != 0:
                self._risk_manager.on_trade_completed(trade_profit)

            self._previous_profit = current_profit

    async def on_price_update(self, price: Decimal) -> None:
        """
        Handle price update event.

        Args:
            price: Current price
        """
        if not self._risk_manager:
            return

        # Check breakout first (for tracking/notification)
        direction = self._risk_manager.check_breakout(price)

        # Check dynamic adjustment (may rebuild grid)
        if self._config.dynamic_adjust.enabled:
            adjusted = await self._risk_manager.check_and_execute_dynamic_adjust(price)
            if adjusted:
                if direction.value != "none":
                    self._risk_manager.record_breakout(direction, price)
                return

        # Handle breakout if detected
        if direction.value != "none":
            await self._risk_manager.handle_breakout(direction, price)

    async def on_kline_close(self, kline: Kline) -> None:
        """
        Handle K-line close event.

        Args:
            kline: Closed Kline data
        """
        if not self._config.dynamic_adjust.enabled:
            return

        if not self._risk_manager:
            return

        if self._state != BotState.RUNNING:
            return

        current_price = kline.close
        rebuilt = await self._risk_manager.check_and_execute_dynamic_adjust(current_price)

        if rebuilt:
            if self._order_manager and self._order_manager.setup:
                self._setup = self._order_manager.setup
                logger.info(f"Grid rebuilt via K-line close, version {self._setup.version}")

    async def _rebuild_grid(self, new_center_price: Decimal) -> None:
        """
        Rebuild grid around new center price.

        Uses transactional approach: saves old state before rebuild,
        and rolls back if rebuild fails.

        Args:
            new_center_price: New center price for grid calculation
        """
        logger.info(f"Rebuilding grid around {new_center_price}")

        # Save old state for rollback
        old_setup = self._setup
        old_calculator = self._calculator

        try:
            klines = await self._fetch_klines()
            if not klines:
                raise RuntimeError("Failed to fetch kline data for rebuild")

            grid_config = self._create_grid_config()
            new_calculator = SmartGridCalculator(
                config=grid_config,
                klines=klines,
                current_price=new_center_price,
            )

            new_setup = new_calculator.calculate()
            logger.info(
                f"Grid calculated: {new_setup.grid_count} levels, "
                f"range {new_setup.lower_price}-{new_setup.upper_price}"
            )

            # Cancel all existing orders before reinitializing (prevent duplicate orders)
            try:
                await self._order_manager.cancel_all_orders()
            except Exception as cancel_err:
                logger.warning(f"Error cancelling orders before rebuild: {cancel_err}")

            # Initialize order manager with new setup
            self._order_manager.initialize(new_setup)

            # Try to place orders - this is the critical step
            placed = await self._order_manager.place_initial_orders()

            if placed == 0:
                # No orders placed - this is a failure, rollback
                raise RuntimeError("No orders were placed after grid rebuild")

            # Success - commit the new state
            self._setup = new_setup
            self._calculator = new_calculator
            logger.info(f"Placed {placed} orders after grid rebuild")

            await self._save_state()

        except Exception as e:
            logger.error(f"Failed to rebuild grid: {e}")

            # Rollback to old state
            if old_setup is not None:
                logger.info("Rolling back to previous grid state")
                self._setup = old_setup
                self._calculator = old_calculator
                self._order_manager.initialize(old_setup)
                # Note: old orders were cancelled, need to re-place them
                try:
                    restored = await self._order_manager.place_initial_orders()
                    logger.info(f"Restored {restored} orders after rollback")
                except Exception as restore_error:
                    logger.error(f"Failed to restore orders after rollback: {restore_error}")

            await self._notify_error(f"Grid rebuild failed: {e}")
            # Re-raise with context to distinguish from initialization errors
            raise RuntimeError(f"Grid rebuild failed at price {new_center_price}") from e

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._order_manager:
            return {}
        return self._order_manager.get_statistics()

    def get_orders(self) -> List[Order]:
        """Get all pending orders."""
        if not self._order_manager or not self._setup:
            return []

        orders = []
        # Use setup levels instead of directly accessing private _level_order_map
        for i, level in enumerate(self._setup.levels):
            order = self._order_manager.get_order_by_level(i)
            if order:
                orders.append(order)
        return orders

    def get_levels(self) -> List[GridLevel]:
        """Get all grid levels (returns a copy to prevent mutation)."""
        if not self._setup:
            return []
        return list(self._setup.levels)

    def get_history(self) -> List[FilledRecord]:
        """Get fill history (returns a copy to prevent mutation)."""
        if not self._order_manager:
            return []
        return list(self._order_manager._filled_history)

    # =========================================================================
    # Persistence
    # =========================================================================

    async def _try_restore_orders(self) -> int:
        """Try to restore order mapping from saved state."""
        try:
            saved_state = await self._data_manager.get_bot_state(self._bot_id)
            if not saved_state:
                logger.debug("No saved state found, starting fresh")
                return 0

            order_mapping = saved_state.get("order_mapping") or saved_state.get("state_data", {}).get("order_mapping")
            if not order_mapping:
                logger.debug("No order mapping in saved state")
                return 0

            restored = self._order_manager.restore_order_mapping(order_mapping)

            if restored > 0:
                sync_result = await self._order_manager.sync_orders()
                active = sync_result.get("synced", 0)
                logger.info(f"Order restoration: {restored} mapped, {active} still active on exchange")
                return active

            return 0

        except Exception as e:
            logger.warning(f"Failed to restore orders: {e}")
            return 0

    def _convert_decimals(self, obj: Any) -> Any:
        """Recursively convert Decimal values to strings for JSON serialization."""
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals(item) for item in obj]
        return obj

    async def _save_state(self) -> None:
        """Save bot state to database."""
        try:
            config = {
                "symbol": self._config.symbol,
                "market_type": self._config.market_type.value,
                "total_investment": str(self._config.total_investment),
                "risk_level": self._config.risk_level.value,
                "grid_type": self._config.grid_type.value,
            }

            statistics = self._convert_decimals(self.get_statistics())
            state_data = {
                "start_time": self._stats.start_time.isoformat() if self._stats.start_time else None,
                "statistics": statistics,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            if self._setup:
                state_data["setup"] = {
                    "upper_price": str(self._setup.upper_price),
                    "lower_price": str(self._setup.lower_price),
                    "grid_count": self._setup.grid_count,
                }

            if self._order_manager:
                state_data["order_mapping"] = self._order_manager.get_order_mapping()

            await self._data_manager.save_bot_state(
                bot_id=self._bot_id,
                bot_type="grid",
                status=self._state.value,
                config=config,
                state_data=state_data,
            )

            logger.debug(f"Bot state saved: {self._bot_id}")

        except Exception as e:
            logger.warning(f"Failed to save bot state: {e}")

    def _start_save_task(self) -> None:
        """Start periodic save task."""
        if self._save_task is not None:
            return

        async def save_loop():
            while self._running:
                await asyncio.sleep(self._config.save_interval_minutes * 60)
                if self._running:
                    await self._save_state()

        self._save_task = asyncio.create_task(save_loop())

    async def _stop_save_task(self) -> None:
        """Stop periodic save task and wait for cleanup."""
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
            self._save_task = None

    @classmethod
    async def restore(
        cls,
        bot_id: str,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: NotificationManager,
    ) -> Optional["GridBot"]:
        """
        Restore a GridBot from saved state.

        Args:
            bot_id: Bot ID to restore
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance

        Returns:
            Restored GridBot or None if not found
        """
        try:
            state_data = await data_manager.get_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

            config_data = state_data.get("config", {})
            config = GridBotConfig(
                symbol=config_data.get("symbol", ""),
                market_type=MarketType(config_data.get("market_type", "spot")),
                total_investment=Decimal(config_data.get("total_investment", "1000")),
                risk_level=RiskLevel(config_data.get("risk_level", "medium")),
                grid_type=GridType(config_data.get("grid_type", "arithmetic")),
            )

            bot = cls(
                bot_id=bot_id,
                config=config,
                exchange=exchange,
                data_manager=data_manager,
                notifier=notifier,
            )

            await bot.start()

            if bot._order_manager:
                await bot._order_manager.sync_orders()

            logger.info(f"GridBot {bot_id} restored successfully")
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot {bot_id}: {e}")
            return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _validate_config(self) -> None:
        """Validate bot configuration."""
        if not self._config.symbol:
            raise ValueError("Symbol is required")

        if self._config.total_investment <= 0:
            raise ValueError("Total investment must be positive")

        if self._config.has_manual_range:
            if self._config.manual_upper <= self._config.manual_lower:
                raise ValueError("Upper price must be greater than lower price")

    async def _fetch_klines(self) -> List[Kline]:
        """Fetch kline data for ATR calculation."""
        klines = await self._data_manager.get_klines(
            symbol=self._config.symbol,
            interval=self._config.atr_config.timeframe,
            limit=self._config.kline_limit,
        )

        if not klines:
            klines = await self._exchange.get_klines(
                symbol=self._config.symbol,
                interval=self._config.atr_config.timeframe,
                limit=self._config.kline_limit,
                market=self._config.market_type,
            )

        return klines

    def _create_grid_config(self) -> GridConfig:
        """Create GridConfig from bot config."""
        return GridConfig(
            symbol=self._config.symbol,
            total_investment=self._config.total_investment,
            risk_level=self._config.risk_level,
            grid_type=self._config.grid_type,
            manual_upper_price=self._config.manual_upper,
            manual_lower_price=self._config.manual_lower,
            manual_grid_count=self._config.manual_grid_count,
            atr_config=self._config.atr_config,
            min_order_value=self._config.min_order_value,
            min_grid_count=self._config.min_grid_count,
            max_grid_count=self._config.max_grid_count,
        )

    async def _subscribe_user_data(self) -> None:
        """Subscribe to user data stream for order updates."""
        try:
            await self._exchange.subscribe_user_data(
                callback=self._handle_user_data,
            )
            logger.info("Subscribed to user data stream")
        except Exception as e:
            logger.warning(f"Failed to subscribe to user data: {e}")

    async def _unsubscribe_user_data(self) -> None:
        """Unsubscribe from user data stream."""
        try:
            await self._exchange.unsubscribe_user_data()
            logger.info("Unsubscribed from user data stream")
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from user data: {e}")

    async def _subscribe_kline_close(self) -> None:
        """Subscribe to K-line close events for dynamic adjustment."""
        try:
            await self._exchange.subscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.atr_config.timeframe,
                callback=self._handle_kline_close,
            )
            logger.info(
                f"Subscribed to K-line close events: {self._config.symbol} "
                f"interval={self._config.atr_config.timeframe}"
            )
        except Exception as e:
            logger.warning(f"Failed to subscribe to K-line close events: {e}")

    async def _unsubscribe_kline_close(self) -> None:
        """Unsubscribe from K-line close events."""
        try:
            await self._exchange.unsubscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.atr_config.timeframe,
            )
            logger.info("Unsubscribed from K-line close events")
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from K-line close: {e}")

    async def _handle_kline_close(self, kline: Kline) -> None:
        """Handle K-line close event."""
        try:
            await self.on_kline_close(kline)
        except Exception as e:
            logger.error(f"Error handling kline close: {e}")

    async def _handle_user_data(self, data: dict) -> None:
        """Handle user data stream events."""
        event_type = data.get("e")

        if event_type == "executionReport":
            order = self._parse_order_update(data)
            if order:
                if order.status == OrderStatus.FILLED:
                    # Route fills through bot.on_order_filled for profit tracking + risk mgmt
                    # on_order_filled internally calls order_manager.on_order_filled
                    await self.on_order_filled(order)
                    # Also update order_manager cache (without re-calling on_order_filled)
                    if order.order_id in self._order_manager._orders or order.order_id in self._order_manager._order_level_map:
                        self._order_manager._orders[order.order_id] = order
                else:
                    # Non-fill updates (cancel, partial, etc.) go to order_manager directly
                    await self._order_manager.handle_order_update(order)

    def _parse_order_update(self, data: dict) -> Optional[Order]:
        """Parse order update from WebSocket executionReport data.

        Binance WebSocket executionReport fields:
        - s: symbol
        - S: side (BUY/SELL)
        - o: order type
        - X: order status
        - i: order id
        - c: client order id
        - q: original quantity
        - p: price
        - z: cumulative filled quantity
        - Z: cumulative quote asset transacted
        - n: commission amount
        - N: commission asset
        - T: transaction time
        - L: last filled price
        """
        try:
            # Validate required fields
            order_id = data.get("i")
            symbol = data.get("s")
            if not order_id or not symbol:
                logger.warning(f"Missing order_id or symbol in execution report: {data}")
                return None

            # Parse side
            side_str = data.get("S", "BUY")
            try:
                side = OrderSide(side_str)
            except ValueError:
                logger.warning(f"Invalid order side: {side_str}")
                return None

            # Parse order type
            type_str = data.get("o", "LIMIT")
            try:
                order_type = OrderType(type_str)
            except ValueError:
                order_type = OrderType.LIMIT  # Default to LIMIT

            # Parse status
            status_str = data.get("X", "NEW")
            try:
                status = OrderStatus(status_str)
            except ValueError:
                logger.warning(f"Invalid order status: {status_str}")
                return None

            # Parse quantities and prices
            quantity = Decimal(str(data.get("q", "0")))
            price = Decimal(str(data.get("p", "0")))
            filled_qty = Decimal(str(data.get("z", "0")))

            # Calculate average price from cumulative quote qty / filled qty (Z/z)
            cumulative_quote = Decimal(str(data.get("Z", "0")))
            avg_price = (cumulative_quote / filled_qty) if filled_qty > 0 else (price if price > 0 else None)

            # Parse commission
            fee = Decimal(str(data.get("n", "0")))
            fee_asset = data.get("N")

            # Parse timestamps
            transaction_time = data.get("T", 0)
            created_at = datetime.fromtimestamp(
                int(transaction_time) / 1000,
                tz=timezone.utc
            ) if transaction_time else datetime.now(timezone.utc)

            return Order(
                order_id=str(order_id),
                client_order_id=data.get("c"),
                symbol=symbol,
                side=side,
                order_type=order_type,
                status=status,
                price=price if price > 0 else None,
                quantity=quantity,
                filled_qty=filled_qty,
                avg_price=avg_price,
                fee=fee,
                fee_asset=fee_asset,
                created_at=created_at,
                updated_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Failed to parse order update: {e}, data: {data}")
            return None

    async def _clear_position(self) -> None:
        """Market sell all positions."""
        if not self._order_manager:
            return

        total_qty = self._calculate_position()

        if total_qty > 0:
            try:
                await self._exchange.market_sell(
                    self._config.symbol,
                    total_qty,
                    self._config.market_type,
                )
                logger.info(f"Cleared position: sold {total_qty}")
            except Exception as e:
                logger.error(f"Failed to clear position: {e}")

    def _calculate_position(self) -> Decimal:
        """Calculate current position quantity."""
        if not self._order_manager:
            return Decimal("0")

        total = Decimal("0")
        for record in self._order_manager._filled_history:
            if record.side == OrderSide.BUY and record.paired_record is None:
                total += record.quantity

        return total

    def _calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L."""
        if not self._order_manager:
            return Decimal("0")

        total_pnl = Decimal("0")
        for record in self._order_manager._filled_history:
            if record.side == OrderSide.BUY and record.paired_record is None:
                pnl = (current_price - record.price) * record.quantity
                total_pnl += pnl

        return total_pnl

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _notify_bot_started(self) -> None:
        """Send bot started notification."""
        try:
            config_summary = {
                "symbol": self._config.symbol,
                "investment": str(self._config.total_investment),
                "grid_count": self._setup.grid_count if self._setup else 0,
                "price_range": f"{self._setup.lower_price}-{self._setup.upper_price}" if self._setup else "N/A",
                "risk_level": self._config.risk_level.value,
            }
            await self._notifier.notify_bot_started(
                bot_name=self._bot_id,
                bot_type="Grid Trading",
                config_summary=config_summary,
            )
        except Exception as e:
            logger.warning(f"Failed to send start notification: {e}")

    async def _notify_bot_stopped(self, stats: dict, cleared_position: bool) -> None:
        """Send bot stopped notification with stats."""
        try:
            runtime = 0
            if self._stats.start_time:
                delta = datetime.now(timezone.utc) - self._stats.start_time
                runtime = delta.total_seconds()

            reason = "Position cleared" if cleared_position else "Normal stop"
            total_pnl = stats.get('total_profit', Decimal("0"))

            await self._notifier.notify_bot_stopped(
                bot_name=self._bot_id,
                reason=reason,
                runtime=runtime,
                total_pnl=total_pnl,
            )
        except Exception as e:
            logger.warning(f"Failed to send stop notification: {e}")

    async def _notify_error(self, error: str) -> None:
        """Send error notification."""
        try:
            await self._notifier.notify_error(
                bot_name=self._bot_id,
                error_type="Runtime Error",
                error_message=error,
            )
        except Exception as e:
            logger.warning(f"Failed to send error notification: {e}")
