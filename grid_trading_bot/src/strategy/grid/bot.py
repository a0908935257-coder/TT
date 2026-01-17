"""
Grid Trading Bot.

Main GridBot class that integrates calculator, order manager, and risk manager
to provide a complete grid trading solution.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from core import get_logger
from core.models import Kline, MarketType, Order
from data import MarketDataManager
from exchange import ExchangeClient
from notification import NotificationManager

from .calculator import SmartGridCalculator
from .models import GridConfig, GridLevel, GridSetup, GridType, RiskLevel
from .order_manager import FilledRecord, GridOrderManager
from .risk_manager import BotState, GridRiskManager, RiskConfig

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
        ...     risk_level=RiskLevel.MEDIUM,
        ... )
    """

    # Required
    symbol: str
    market_type: MarketType = MarketType.SPOT
    total_investment: Decimal = field(default_factory=lambda: Decimal("1000"))

    # Grid calculation
    risk_level: RiskLevel = RiskLevel.MEDIUM
    grid_type: GridType = GridType.ARITHMETIC

    # Manual overrides (optional)
    manual_upper: Optional[Decimal] = None
    manual_lower: Optional[Decimal] = None
    manual_grid_count: Optional[int] = None

    # Risk configuration
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # ATR settings
    atr_period: int = 14
    kline_timeframe: str = "4h"
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


class GridBot:
    """
    Grid Trading Bot.

    Integrates all grid trading components:
    - SmartGridCalculator: Calculate grid levels
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
    ):
        """
        Initialize GridBot.

        Args:
            bot_id: Unique bot identifier
            config: GridBotConfig instance
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance
        """
        self._bot_id = bot_id
        self._config = config
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier

        # Components (initialized in start())
        self._calculator: Optional[SmartGridCalculator] = None
        self._order_manager: Optional[GridOrderManager] = None
        self._risk_manager: Optional[GridRiskManager] = None

        # Grid setup
        self._setup: Optional[GridSetup] = None

        # State
        self._state: BotState = BotState.INITIALIZING
        self._running: bool = False
        self._start_time: Optional[datetime] = None

        # Persistence task
        self._save_task: Optional[asyncio.Task] = None

        # Last trade profit (for risk manager callback)
        self._last_trade_profit: Decimal = Decimal("0")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def bot_id(self) -> str:
        """Get bot ID."""
        return self._bot_id

    @property
    def state(self) -> BotState:
        """Get current bot state."""
        return self._state

    @property
    def config(self) -> GridBotConfig:
        """Get bot configuration."""
        return self._config

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

    @property
    def start_time(self) -> Optional[datetime]:
        """Get start time."""
        return self._start_time

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running and self._state == BotState.RUNNING

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the GridBot.

        Flow:
        1. Set state to INITIALIZING
        2. Validate configuration
        3. Fetch kline data
        4. Initialize calculator and compute grid
        5. Initialize order manager
        6. Initialize risk manager
        7. Place initial orders
        8. Subscribe to user data stream
        9. Start risk monitoring
        10. Set state to RUNNING

        Returns:
            True if started successfully
        """
        try:
            logger.info(f"Starting GridBot {self._bot_id}")
            self._state = BotState.INITIALIZING

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

            # Register order manager callback for fills
            # (This would require extending order_manager to support callbacks)

            # Step 7: Place initial orders
            placed = await self._order_manager.place_initial_orders()
            logger.info(f"Placed {placed} initial orders")

            # Step 8: Subscribe to user data stream
            await self._subscribe_user_data()

            # Step 9: Start risk monitoring
            await self._risk_manager.start_monitoring()

            # Step 10: Update state
            self._state = BotState.RUNNING
            self._running = True
            self._start_time = datetime.now(timezone.utc)

            # Start persistence task
            self._start_save_task()

            # Send notification
            await self._notify_bot_started()

            logger.info(f"GridBot {self._bot_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start GridBot: {e}")
            self._state = BotState.ERROR
            await self._notify_error(f"Failed to start: {e}")
            return False

    async def stop(self, clear_position: bool = False) -> bool:
        """
        Stop the GridBot.

        Args:
            clear_position: If True, market sell all positions

        Returns:
            True if stopped successfully
        """
        if self._state in (BotState.STOPPED, BotState.STOPPING):
            logger.warning(f"Bot already stopping/stopped: {self._state.value}")
            return False

        try:
            logger.info(f"Stopping GridBot {self._bot_id}")
            self._state = BotState.STOPPING
            self._running = False

            # Stop persistence task
            self._stop_save_task()

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

            # Unsubscribe from user data
            await self._unsubscribe_user_data()

            # Get final statistics
            stats = self.get_statistics()

            # Update state
            self._state = BotState.STOPPED

            # Save final state
            await self._save_state()

            # Send notification
            await self._notify_bot_stopped(stats, clear_position)

            logger.info(f"GridBot {self._bot_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping GridBot: {e}")
            self._state = BotState.ERROR
            return False

    async def pause(self, reason: str = "Manual pause") -> bool:
        """
        Pause the GridBot.

        Delegates to risk_manager.pause().

        Args:
            reason: Pause reason

        Returns:
            True if paused successfully
        """
        if not self._risk_manager:
            return False

        result = await self._risk_manager.pause(reason)
        if result:
            self._state = BotState.PAUSED

        return result

    async def resume(self) -> bool:
        """
        Resume the GridBot from paused state.

        Delegates to risk_manager.resume().

        Returns:
            True if resumed successfully
        """
        if not self._risk_manager:
            return False

        result = await self._risk_manager.resume()
        if result:
            self._state = BotState.RUNNING

        return result

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

        # Delegate to order manager
        reverse_order = await self._order_manager.on_order_filled(order)

        # Get profit from last trade (if completed)
        stats = self._order_manager.get_statistics()
        current_profit = stats.get("total_profit", Decimal("0"))

        # Calculate profit from this trade
        if hasattr(self, "_previous_profit"):
            trade_profit = current_profit - self._previous_profit
            self._last_trade_profit = trade_profit

            # Notify risk manager
            if self._risk_manager and trade_profit != 0:
                self._risk_manager.on_trade_completed(trade_profit)

        self._previous_profit = current_profit

    async def on_price_update(self, price: Decimal) -> None:
        """
        Handle price update event.

        Optional: Subscribe to price updates for real-time risk checking.

        Args:
            price: Current price
        """
        if not self._risk_manager:
            return

        # Check breakout
        direction = self._risk_manager.check_breakout(price)
        if direction.value != "none":
            await self._risk_manager.handle_breakout(direction, price)

    # =========================================================================
    # Status Query Methods
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get comprehensive bot status.

        Returns:
            Dict with bot status, grid info, orders, performance, positions
        """
        status: dict[str, Any] = {
            "bot_id": self._bot_id,
            "state": self._state.value,
            "symbol": self._config.symbol,
            "market_type": self._config.market_type.value,
            "start_time": self._start_time,
            "running_duration": self._get_running_duration(),
        }

        # Grid info
        if self._setup:
            status.update({
                "upper_price": self._setup.upper_price,
                "lower_price": self._setup.lower_price,
                "current_price": self._setup.current_price,
                "grid_count": self._setup.grid_count,
                "grid_spacing": f"{self._setup.grid_spacing_percent:.2f}%",
            })

        # Order info
        if self._order_manager:
            stats = self._order_manager.get_statistics()
            status.update({
                "pending_buy_orders": stats.get("pending_buy_count", 0),
                "pending_sell_orders": stats.get("pending_sell_count", 0),
                "total_orders_placed": self._order_manager.active_order_count,
            })

            # Performance
            status.update({
                "total_profit": stats.get("total_profit", Decimal("0")),
                "total_trades": stats.get("trade_count", 0),
                "avg_profit_per_trade": stats.get("avg_profit_per_trade", Decimal("0")),
                "total_fees": stats.get("total_fees", Decimal("0")),
                "net_profit": stats.get("total_profit", Decimal("0")) - stats.get("total_fees", Decimal("0")),
            })

            # Position info
            position_qty = self._calculate_position()
            current_price = self._setup.current_price if self._setup else Decimal("0")
            status.update({
                "current_position": position_qty,
                "position_value": position_qty * current_price,
                "unrealized_pnl": self._calculate_unrealized_pnl(current_price),
            })

        # Risk status
        if self._risk_manager:
            status.update({
                "daily_pnl": self._risk_manager.daily_pnl,
                "consecutive_losses": self._risk_manager.consecutive_losses,
                "last_breakout": (
                    self._risk_manager.last_breakout.direction.value
                    if self._risk_manager.last_breakout
                    else "none"
                ),
            })

        return status

    def get_statistics(self) -> dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dict with trading statistics
        """
        if not self._order_manager:
            return {}

        return self._order_manager.get_statistics()

    def get_orders(self) -> list[Order]:
        """
        Get all pending orders.

        Returns:
            List of pending Order objects
        """
        if not self._order_manager:
            return []

        orders = []
        for level_index in self._order_manager._level_order_map:
            order = self._order_manager.get_order_by_level(level_index)
            if order:
                orders.append(order)

        return orders

    def get_levels(self) -> list[GridLevel]:
        """
        Get all grid levels.

        Returns:
            List of GridLevel objects
        """
        if not self._setup:
            return []

        return self._setup.levels

    def get_history(self) -> list[FilledRecord]:
        """
        Get fill history.

        Returns:
            List of FilledRecord objects
        """
        if not self._order_manager:
            return []

        return self._order_manager._filled_history

    # =========================================================================
    # Persistence
    # =========================================================================

    async def _save_state(self) -> None:
        """Save bot state to database."""
        try:
            # Create state snapshot
            state_data = {
                "bot_id": self._bot_id,
                "state": self._state.value,
                "config": {
                    "symbol": self._config.symbol,
                    "market_type": self._config.market_type.value,
                    "total_investment": str(self._config.total_investment),
                    "risk_level": self._config.risk_level.value,
                    "grid_type": self._config.grid_type.value,
                },
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "statistics": self.get_statistics(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            if self._setup:
                state_data["setup"] = {
                    "upper_price": str(self._setup.upper_price),
                    "lower_price": str(self._setup.lower_price),
                    "grid_count": self._setup.grid_count,
                }

            # Save to data manager
            await self._data_manager.save_bot_state(self._bot_id, state_data)

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

    def _stop_save_task(self) -> None:
        """Stop periodic save task."""
        if self._save_task:
            self._save_task.cancel()
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
            # Load state from database
            state_data = await data_manager.load_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

            # Recreate config
            config_data = state_data.get("config", {})
            config = GridBotConfig(
                symbol=config_data.get("symbol", ""),
                market_type=MarketType(config_data.get("market_type", "spot")),
                total_investment=Decimal(config_data.get("total_investment", "1000")),
                risk_level=RiskLevel(config_data.get("risk_level", "medium")),
                grid_type=GridType(config_data.get("grid_type", "arithmetic")),
            )

            # Create bot instance
            bot = cls(
                bot_id=bot_id,
                config=config,
                exchange=exchange,
                data_manager=data_manager,
                notifier=notifier,
            )

            # Restore start time
            if state_data.get("start_time"):
                bot._start_time = datetime.fromisoformat(state_data["start_time"])

            # Start the bot (will sync orders)
            await bot.start()

            # Sync orders with exchange
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

    async def _fetch_klines(self) -> list[Kline]:
        """Fetch kline data for ATR calculation."""
        klines = await self._data_manager.get_klines(
            symbol=self._config.symbol,
            timeframe=self._config.kline_timeframe,
            limit=self._config.kline_limit,
            market_type=self._config.market_type,
        )

        if not klines:
            # Try exchange directly
            klines = await self._exchange.get_klines(
                symbol=self._config.symbol,
                interval=self._config.kline_timeframe,
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
            atr_period=self._config.atr_period,
        )

    async def _subscribe_user_data(self) -> None:
        """Subscribe to user data stream for order updates."""
        try:
            # Register order update callback
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

    async def _handle_user_data(self, data: dict) -> None:
        """Handle user data stream events."""
        event_type = data.get("e")

        if event_type == "executionReport":
            # Order update
            order = self._parse_order_update(data)
            if order:
                await self._order_manager.handle_order_update(order)

    def _parse_order_update(self, data: dict) -> Optional[Order]:
        """Parse order update from WebSocket data."""
        # This would depend on the specific exchange format
        # Simplified implementation
        return None

    async def _clear_position(self) -> None:
        """Market sell all positions."""
        if not self._order_manager:
            return

        # Calculate total position
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
            if record.side.value == "BUY" and record.paired_record is None:
                total += record.quantity

        return total

    def _calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L."""
        if not self._order_manager:
            return Decimal("0")

        total_pnl = Decimal("0")
        for record in self._order_manager._filled_history:
            if record.side.value == "BUY" and record.paired_record is None:
                pnl = (current_price - record.price) * record.quantity
                total_pnl += pnl

        return total_pnl

    def _get_running_duration(self) -> str:
        """Get human-readable running duration."""
        if not self._start_time:
            return "0m"

        delta = datetime.now(timezone.utc) - self._start_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")

        return " ".join(parts)

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _notify_bot_started(self) -> None:
        """Send bot started notification."""
        try:
            message = (
                f"GridBot {self._bot_id} Started\n"
                f"Symbol: {self._config.symbol}\n"
                f"Investment: {self._config.total_investment}\n"
                f"Grid Count: {self._setup.grid_count if self._setup else 0}\n"
                f"Range: {self._setup.lower_price}-{self._setup.upper_price if self._setup else 'N/A'}"
            )

            await self._notifier.send_success(
                title="Grid Bot Started",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send start notification: {e}")

    async def _notify_bot_stopped(
        self,
        stats: dict,
        cleared_position: bool,
    ) -> None:
        """Send bot stopped notification with stats."""
        try:
            duration = self._get_running_duration()

            message = (
                f"GridBot {self._bot_id} Stopped\n"
                f"Duration: {duration}\n"
                f"Total Profit: {stats.get('total_profit', 0):.4f}\n"
                f"Total Trades: {stats.get('trade_count', 0)}\n"
                f"Position Cleared: {'Yes' if cleared_position else 'No'}"
            )

            await self._notifier.send_info(
                title="Grid Bot Stopped",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send stop notification: {e}")

    async def _notify_error(self, error: str) -> None:
        """Send error notification."""
        try:
            message = (
                f"GridBot {self._bot_id} Error\n"
                f"Error: {error}"
            )

            await self._notifier.send_error(
                title="Grid Bot Error",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send error notification: {e}")
