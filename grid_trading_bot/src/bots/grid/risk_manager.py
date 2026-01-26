"""
Grid Risk Manager.

Handles price breakout detection, risk actions, grid recovery,
continuous monitoring, and bot state management.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from src.core import get_logger
from src.core.models import Kline, MarketType, OrderSide
from src.master.models import BotState
from src.notification import NotificationManager

from .models import DynamicAdjustConfig
from .order_manager import GridOrderManager

logger = get_logger(__name__)


@dataclass
class RebuildRecord:
    """Record of a grid rebuild event."""

    timestamp: datetime
    reason: str
    old_upper: Decimal
    old_lower: Decimal
    new_upper: Decimal
    new_lower: Decimal
    trigger_price: Decimal


class BreakoutDirection(str, Enum):
    """Direction of price breakout."""

    NONE = "none"      # No breakout - price within grid range
    UPPER = "upper"    # Upward breakout - price above upper bound
    LOWER = "lower"    # Downward breakout - price below lower bound


class BreakoutAction(str, Enum):
    """Action to take on breakout."""

    HOLD = "hold"           # Hold and wait for price return
    PAUSE = "pause"         # Pause trading, cancel orders
    STOP_LOSS = "stop_loss" # Stop loss - market sell all
    RESET_GRID = "reset"    # Reset grid around current price (manual)
    EXPAND_GRID = "expand"  # Expand grid range in breakout direction
    AUTO_REBUILD = "auto"   # Auto rebuild using dynamic adjuster


class RiskState(str, Enum):
    """Current risk management state."""

    NORMAL = "normal"           # Normal operation
    BREAKOUT_UPPER = "breakout_upper"  # Upper breakout detected
    BREAKOUT_LOWER = "breakout_lower"  # Lower breakout detected
    PAUSED = "paused"           # Trading paused
    STOPPED = "stopped"         # Bot stopped (stop loss triggered)


# BotState is imported from src.master.models for consistency
# Valid state transitions for GridRiskManager (subset of master.models)
VALID_STATE_TRANSITIONS: dict[BotState, set[BotState]] = {
    BotState.INITIALIZING: {BotState.RUNNING, BotState.ERROR, BotState.STOPPED},
    BotState.RUNNING: {BotState.PAUSED, BotState.STOPPING, BotState.ERROR},
    BotState.PAUSED: {BotState.RUNNING, BotState.STOPPING, BotState.STOPPED},
    BotState.STOPPING: {BotState.STOPPED},
    BotState.STOPPED: set(),  # Terminal state
    BotState.ERROR: {BotState.STOPPED, BotState.PAUSED},
    BotState.UNKNOWN: {BotState.STOPPED, BotState.ERROR},  # UNKNOWN can transition to recovery states
}


@dataclass
class RiskConfig:
    """
    Risk management configuration.

    Attributes:
        upper_breakout_action: Action for upper breakout (default: AUTO_REBUILD)
        lower_breakout_action: Action for lower breakout (default: AUTO_REBUILD)
        stop_loss_percent: Stop loss threshold percentage (default: 20)
        breakout_buffer: Buffer percentage before triggering breakout (default: 0.5)
        auto_rebuild_enabled: Enable automatic grid rebuild (default: True)
        breakout_threshold: Price movement % to trigger rebuild (default: 4.0)
        cooldown_days: Days between allowed rebuilds (default: 7)
        max_rebuilds_in_period: Max rebuilds within cooldown period (default: 3)

        Price-based stop loss (new):
        price_stop_loss_enabled: Enable price-based stop loss (default: False)
        price_stop_loss: Absolute price level for stop loss (optional)
        range_stop_loss_percent: Stop loss as % below grid lower bound (optional)

    Example:
        >>> config = RiskConfig(
        ...     upper_breakout_action=BreakoutAction.AUTO_REBUILD,
        ...     lower_breakout_action=BreakoutAction.STOP_LOSS,
        ...     stop_loss_percent=Decimal("15"),
        ...     daily_loss_limit=Decimal("5"),
        ... )

        >>> # Price-based stop loss example
        >>> config = RiskConfig(
        ...     price_stop_loss_enabled=True,
        ...     price_stop_loss=Decimal("42000"),  # Stop if price < 42000
        ... )

        >>> # Range-based stop loss example
        >>> config = RiskConfig(
        ...     price_stop_loss_enabled=True,
        ...     range_stop_loss_percent=Decimal("10"),  # Stop if price < lower_bound * 0.9
        ... )
    """

    # Breakout actions (Prompt 21: default to AUTO_REBUILD)
    upper_breakout_action: BreakoutAction = BreakoutAction.AUTO_REBUILD
    lower_breakout_action: BreakoutAction = BreakoutAction.AUTO_REBUILD

    # Stop loss threshold (percentage-based, for unrealized P&L)
    stop_loss_percent: Decimal = field(default_factory=lambda: Decimal("20"))

    # Breakout buffer (percentage beyond grid bounds)
    breakout_buffer: Decimal = field(default_factory=lambda: Decimal("0.5"))

    # Auto rebuild settings (Prompt 21)
    auto_rebuild_enabled: bool = True  # Enable automatic grid rebuild
    breakout_threshold: Decimal = field(default_factory=lambda: Decimal("4.0"))  # Trigger threshold %
    cooldown_days: int = 7  # Cooldown period in days
    max_rebuilds_in_period: int = 3  # Max rebuilds within cooldown period

    # Expand grid settings
    expand_atr_multiplier: Decimal = field(default_factory=lambda: Decimal("1.5"))

    # Price-based stop loss settings (new)
    price_stop_loss_enabled: bool = False  # Enable price-based stop loss
    price_stop_loss: Optional[Decimal] = None  # Absolute price level
    range_stop_loss_percent: Optional[Decimal] = None  # % below grid lower bound

    # Monitoring settings (Prompt 22)
    daily_loss_limit: Decimal = field(default_factory=lambda: Decimal("5"))  # Daily loss limit %
    max_consecutive_losses: int = 5  # Max consecutive losing trades
    volatility_threshold: Decimal = field(default_factory=lambda: Decimal("10"))  # Single candle volatility %
    order_failure_threshold: int = 3  # Order failures before pause
    health_check_interval: int = 30  # Health check interval seconds
    monitoring_interval: int = 5  # Price monitoring interval seconds

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.stop_loss_percent, Decimal):
            self.stop_loss_percent = Decimal(str(self.stop_loss_percent))
        if not isinstance(self.breakout_buffer, Decimal):
            self.breakout_buffer = Decimal(str(self.breakout_buffer))
        if not isinstance(self.breakout_threshold, Decimal):
            self.breakout_threshold = Decimal(str(self.breakout_threshold))
        if not isinstance(self.expand_atr_multiplier, Decimal):
            self.expand_atr_multiplier = Decimal(str(self.expand_atr_multiplier))
        if not isinstance(self.daily_loss_limit, Decimal):
            self.daily_loss_limit = Decimal(str(self.daily_loss_limit))
        if not isinstance(self.volatility_threshold, Decimal):
            self.volatility_threshold = Decimal(str(self.volatility_threshold))
        if self.price_stop_loss is not None and not isinstance(self.price_stop_loss, Decimal):
            self.price_stop_loss = Decimal(str(self.price_stop_loss))
        if self.range_stop_loss_percent is not None and not isinstance(self.range_stop_loss_percent, Decimal):
            self.range_stop_loss_percent = Decimal(str(self.range_stop_loss_percent))


@dataclass
class BreakoutEvent:
    """Record of a breakout event."""

    direction: BreakoutDirection
    price: Decimal
    timestamp: datetime
    action_taken: BreakoutAction
    upper_price: Decimal
    lower_price: Decimal


class GridRiskManager:
    """
    Grid Risk Manager.

    Monitors price movements and handles breakout situations:
    - Detects when price breaks out of grid range
    - Executes configured actions (hold, pause, stop loss, reset, expand)
    - Tracks unrealized P&L for stop loss
    - Handles price return to grid range

    Example:
        >>> risk_manager = GridRiskManager(
        ...     order_manager=order_manager,
        ...     notifier=notifier,
        ...     config=RiskConfig(),
        ... )
        >>> direction = risk_manager.check_breakout(current_price)
        >>> if direction != BreakoutDirection.NONE:
        ...     await risk_manager.handle_breakout(direction)
    """

    def __init__(
        self,
        order_manager: GridOrderManager,
        notifier: NotificationManager,
        config: Optional[RiskConfig] = None,
    ):
        """
        Initialize GridRiskManager.

        Args:
            order_manager: GridOrderManager instance
            notifier: NotificationManager instance
            config: RiskConfig (uses defaults if None)
        """
        self._order_manager = order_manager
        self._notifier = notifier
        self._config = config or RiskConfig()

        # State tracking
        self._state: RiskState = RiskState.NORMAL
        self._last_breakout: Optional[BreakoutEvent] = None
        self._breakout_history: list[BreakoutEvent] = []
        self._last_reset_time: Optional[datetime] = None

        # Position tracking for stop loss
        self._entry_prices: dict[int, Decimal] = {}  # level_index -> entry_price
        self._position_quantities: dict[int, Decimal] = {}  # level_index -> quantity

        # Bot state management (Prompt 22)
        self._bot_state: BotState = BotState.INITIALIZING
        self._pause_reason: str = ""

        # Daily statistics (Prompt 22)
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_start_time: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Consecutive loss tracking
        self._consecutive_losses: int = 0
        self._last_trade_profitable: bool = True

        # Order failure tracking
        self._order_failures: int = 0

        # Health check tracking
        self._last_health_check: Optional[datetime] = None
        self._consecutive_health_failures: int = 0

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Background tasks tracking (fire-and-forget tasks that need cleanup)
        self._pending_tasks: set[asyncio.Task] = set()

        # Lock for state transitions to prevent TOCTOU vulnerabilities
        self._state_lock: asyncio.Lock = asyncio.Lock()

        # State change callbacks
        self._state_change_callbacks: list[Callable[[BotState, BotState, str], Any]] = []

        # Dynamic adjustment tracking
        self._rebuild_history: list[RebuildRecord] = []
        self._dynamic_adjust_config: Optional[DynamicAdjustConfig] = None
        self._on_rebuild_callback: Optional[Callable[[Decimal], Any]] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> RiskState:
        """Get current risk state."""
        return self._state

    @property
    def config(self) -> RiskConfig:
        """Get risk configuration."""
        return self._config

    @property
    def is_paused(self) -> bool:
        """Check if trading is paused."""
        return self._state == RiskState.PAUSED

    @property
    def is_stopped(self) -> bool:
        """Check if bot is stopped."""
        return self._state == RiskState.STOPPED

    @property
    def last_breakout(self) -> Optional[BreakoutEvent]:
        """Get last breakout event."""
        return self._last_breakout

    @property
    def bot_state(self) -> BotState:
        """Get current bot state."""
        return self._bot_state

    @property
    def pause_reason(self) -> str:
        """Get pause reason if paused."""
        return self._pause_reason

    @property
    def daily_pnl(self) -> Decimal:
        """Get daily P&L."""
        return self._daily_pnl

    @property
    def consecutive_losses(self) -> int:
        """Get consecutive loss count."""
        return self._consecutive_losses

    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._running and self._monitoring_task is not None

    # =========================================================================
    # Breakout Detection
    # =========================================================================

    def check_breakout(self, current_price: Decimal) -> BreakoutDirection:
        """
        Check if price has broken out of grid range.

        Uses breakout_buffer to add tolerance before triggering.

        Args:
            current_price: Current market price

        Returns:
            BreakoutDirection indicating breakout status
        """
        setup = self._order_manager.setup
        if setup is None:
            return BreakoutDirection.NONE

        # Calculate thresholds with buffer
        buffer_rate = self._config.breakout_buffer / Decimal("100")
        upper_threshold = setup.upper_price * (Decimal("1") + buffer_rate)
        lower_threshold = setup.lower_price * (Decimal("1") - buffer_rate)

        if current_price > upper_threshold:
            logger.info(
                f"Upper breakout detected: {current_price} > {upper_threshold}"
            )
            return BreakoutDirection.UPPER
        elif current_price < lower_threshold:
            logger.info(
                f"Lower breakout detected: {current_price} < {lower_threshold}"
            )
            return BreakoutDirection.LOWER
        else:
            return BreakoutDirection.NONE

    def check_price_return(self, current_price: Decimal) -> bool:
        """
        Check if price has returned to grid range after breakout.

        Args:
            current_price: Current market price

        Returns:
            True if price has returned to range
        """
        setup = self._order_manager.setup
        if setup is None:
            return False

        # Only relevant if we're in breakout state
        if self._state == RiskState.BREAKOUT_UPPER:
            # Price returned below upper bound
            return current_price <= setup.upper_price
        elif self._state == RiskState.BREAKOUT_LOWER:
            # Price returned above lower bound
            return current_price >= setup.lower_price
        elif self._state == RiskState.PAUSED and self._last_breakout:
            # Check return based on last breakout direction
            if self._last_breakout.direction == BreakoutDirection.UPPER:
                return current_price <= setup.upper_price
            elif self._last_breakout.direction == BreakoutDirection.LOWER:
                return current_price >= setup.lower_price

        return False

    # =========================================================================
    # Breakout Handling
    # =========================================================================

    async def handle_breakout(
        self,
        direction: BreakoutDirection,
        current_price: Decimal,
    ) -> None:
        """
        Handle a breakout event.

        Args:
            direction: Breakout direction
            current_price: Current market price
        """
        if direction == BreakoutDirection.NONE:
            return

        setup = self._order_manager.setup
        if setup is None:
            return

        # Determine action based on direction
        if direction == BreakoutDirection.UPPER:
            action = self._config.upper_breakout_action
        else:
            action = self._config.lower_breakout_action

        # Note: _state will be set by execute_action based on the action taken
        # This avoids the issue where BREAKOUT_* state is immediately overwritten

        # Record breakout event
        event = BreakoutEvent(
            direction=direction,
            price=current_price,
            timestamp=datetime.now(timezone.utc),
            action_taken=action,
            upper_price=setup.upper_price,
            lower_price=setup.lower_price,
        )
        self._last_breakout = event
        self._breakout_history.append(event)

        # Send alert notification
        await self._notify_breakout(direction, current_price, action)

        # Execute the action
        await self.execute_action(action, direction, current_price)

        logger.warning(
            f"Breakout handled: {direction.value} at {current_price}, "
            f"action={action.value}"
        )

    def record_breakout(
        self,
        direction: BreakoutDirection,
        current_price: Decimal,
    ) -> None:
        """
        Record a breakout event without taking action.

        Used when dynamic adjustment handles the breakout instead
        of traditional breakout handling.

        Args:
            direction: Breakout direction (UPPER or LOWER)
            current_price: Price at breakout
        """
        setup = self._order_manager.setup
        if setup is None:
            return

        event = BreakoutEvent(
            direction=direction,
            price=current_price,
            timestamp=datetime.now(timezone.utc),
            action_taken=BreakoutAction.HOLD,  # No action, grid was rebuilt
            upper_price=setup.upper_price,
            lower_price=setup.lower_price,
        )
        self._last_breakout = event
        self._breakout_history.append(event)

        logger.info(
            f"Breakout recorded (grid rebuilt): {direction.value} at {current_price}"
        )

    async def execute_action(
        self,
        action: BreakoutAction,
        direction: BreakoutDirection,
        current_price: Decimal,
    ) -> None:
        """
        Execute breakout action.

        Args:
            action: Action to execute
            direction: Breakout direction
            current_price: Current market price
        """
        if action == BreakoutAction.HOLD:
            await self._action_hold(direction)

        elif action == BreakoutAction.PAUSE:
            await self._action_pause(direction)

        elif action == BreakoutAction.STOP_LOSS:
            await self._action_stop_loss(direction, current_price)

        elif action == BreakoutAction.RESET_GRID:
            await self._action_reset_grid(current_price)

        elif action == BreakoutAction.EXPAND_GRID:
            await self._action_expand_grid(direction, current_price)

        elif action == BreakoutAction.AUTO_REBUILD:
            await self._action_auto_rebuild(direction, current_price)

    # =========================================================================
    # Action Implementations
    # =========================================================================

    async def _action_hold(self, direction: BreakoutDirection) -> None:
        """
        HOLD action - do nothing, wait for price return.

        Args:
            direction: Breakout direction
        """
        logger.info(f"HOLD action: waiting for price return from {direction.value}")
        # Set breakout state to track direction
        if direction == BreakoutDirection.UPPER:
            self._state = RiskState.BREAKOUT_UPPER
        else:
            self._state = RiskState.BREAKOUT_LOWER

    async def _action_pause(self, direction: BreakoutDirection) -> None:
        """
        PAUSE action - cancel all orders and stop trading.

        Args:
            direction: Breakout direction
        """
        logger.info(f"PAUSE action: cancelling orders due to {direction.value} breakout")

        # Cancel all pending orders
        cancelled = await self._order_manager.cancel_all_orders()

        # Update state
        self._state = RiskState.PAUSED

        logger.info(f"Trading paused: {cancelled} orders cancelled")

    async def _action_stop_loss(
        self,
        direction: BreakoutDirection,
        current_price: Decimal,
    ) -> None:
        """
        STOP_LOSS action - market sell all positions.

        Args:
            direction: Breakout direction
            current_price: Current market price
        """
        logger.warning(f"STOP_LOSS action triggered at {current_price}")

        # Cancel all pending orders first
        await self._order_manager.cancel_all_orders()

        # For downward breakout, we have positions to sell
        if direction == BreakoutDirection.LOWER:
            # Calculate total position from filled buy orders
            total_quantity = self._calculate_total_position()

            if total_quantity > 0:
                try:
                    # Market sell all positions
                    order = await self._order_manager._exchange.market_sell(
                        self._order_manager.symbol,
                        total_quantity,
                        self._order_manager._market_type,
                    )

                    # Calculate loss
                    loss = self._calculate_unrealized_pnl(current_price)

                    # Notify
                    await self._notify_stop_loss(current_price, total_quantity, loss)

                    logger.warning(
                        f"Stop loss executed: sold {total_quantity} at {current_price}, "
                        f"loss={loss}"
                    )

                except Exception as e:
                    logger.error(f"Failed to execute stop loss: {e}")

        # For upward breakout, all sells executed - we're in cash (already "stopped" profit)
        else:
            logger.info("Upper breakout stop loss: already in cash, no action needed")

        # Update state
        self._state = RiskState.STOPPED

    async def _action_reset_grid(self, current_price: Decimal) -> None:
        """
        RESET_GRID action - recalculate grid around current price.

        Args:
            current_price: Current market price (new center)
        """
        # Check cooldown
        if not self._can_reset():
            logger.info("Reset cooldown active, skipping reset")
            return

        logger.info(f"RESET_GRID action: resetting around {current_price}")

        # Cancel all orders
        await self._order_manager.cancel_all_orders()

        # Record reset time
        self._last_reset_time = datetime.now(timezone.utc)

        # Reset will be handled by the bot controller
        # This method signals the need for reset
        await self._notify_grid_reset(current_price)

        # Return to normal state
        self._state = RiskState.NORMAL

    async def _action_expand_grid(
        self,
        direction: BreakoutDirection,
        current_price: Decimal,
    ) -> None:
        """
        EXPAND_GRID action - extend grid range in breakout direction.

        Args:
            direction: Breakout direction
            current_price: Current market price
        """
        setup = self._order_manager.setup
        if setup is None:
            return

        logger.info(f"EXPAND_GRID action: expanding {direction.value}")

        atr_value = setup.atr_data.value
        multiplier = self._config.expand_atr_multiplier

        if direction == BreakoutDirection.UPPER:
            # Expand upper bound
            new_upper = current_price + atr_value * multiplier
            # Keep lower bound
            new_lower = setup.lower_price
        else:
            # Expand lower bound
            new_upper = setup.upper_price
            new_lower = current_price - atr_value * multiplier

        await self._notify_grid_expand(direction, new_upper, new_lower)

        # Return to normal state
        self._state = RiskState.NORMAL

        logger.info(f"Grid expanded: lower={new_lower}, upper={new_upper}")

    async def _action_auto_rebuild(
        self,
        direction: BreakoutDirection,
        current_price: Decimal,
    ) -> None:
        """
        AUTO_REBUILD action - automatically rebuild grid using dynamic adjuster.

        Uses the dynamic adjustment mechanism to rebuild the grid around
        the current price, respecting cooldown limits.

        Args:
            direction: Breakout direction
            current_price: Current market price (new center)
        """
        if not self._config.auto_rebuild_enabled:
            logger.info("Auto rebuild disabled, falling back to PAUSE")
            await self._action_pause(direction)
            return

        logger.info(
            f"AUTO_REBUILD action: rebuilding grid around {current_price} "
            f"(trigger: {direction.value})"
        )

        # Use dynamic adjustment mechanism
        trigger_direction = "upper" if direction == BreakoutDirection.UPPER else "lower"
        success = await self.execute_dynamic_adjust(current_price, trigger_direction)

        if success:
            # Return to normal state after successful rebuild
            self._state = RiskState.NORMAL
            logger.info(f"Auto rebuild successful, grid rebuilt around {current_price}")
        else:
            # If rebuild blocked by cooldown, fall back to PAUSE
            logger.warning("Auto rebuild blocked by cooldown, falling back to PAUSE")
            await self._action_pause(direction)

    # =========================================================================
    # Stop Loss Calculation
    # =========================================================================

    def check_stop_loss(self, current_price: Decimal) -> bool:
        """
        Check if stop loss threshold is breached.

        Calculates unrealized P&L and compares to threshold.

        Args:
            current_price: Current market price

        Returns:
            True if stop loss should be triggered
        """
        setup = self._order_manager.setup
        if setup is None:
            return False

        # Calculate unrealized P&L
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)

        # Calculate P&L percentage
        total_investment = setup.config.total_investment
        if total_investment == 0:
            return False

        pnl_percent = (unrealized_pnl / total_investment) * Decimal("100")

        # Check threshold
        threshold = -self._config.stop_loss_percent
        if pnl_percent <= threshold:
            logger.warning(
                f"Stop loss triggered: P&L {pnl_percent:.2f}% <= {threshold}%"
            )
            return True

        return False

    def check_price_stop_loss(self, current_price: Decimal) -> bool:
        """
        Check if price-based stop loss is triggered.

        Two modes:
        1. Absolute price: Triggers if current_price < price_stop_loss
        2. Range-based: Triggers if current_price < lower_bound * (1 - range_stop_loss_percent/100)

        Args:
            current_price: Current market price

        Returns:
            True if price stop loss should be triggered
        """
        if not self._config.price_stop_loss_enabled:
            return False

        # Mode 1: Absolute price stop loss
        if self._config.price_stop_loss is not None:
            if current_price < self._config.price_stop_loss:
                logger.warning(
                    f"Price stop loss triggered: {current_price} < {self._config.price_stop_loss}"
                )
                return True

        # Mode 2: Range-based stop loss (% below grid lower bound)
        if self._config.range_stop_loss_percent is not None:
            setup = self._order_manager.setup
            if setup is not None:
                lower_bound = setup.lower_price
                stop_price = lower_bound * (
                    Decimal("1") - self._config.range_stop_loss_percent / Decimal("100")
                )
                if current_price < stop_price:
                    logger.warning(
                        f"Range stop loss triggered: {current_price} < {stop_price} "
                        f"(lower_bound {lower_bound} - {self._config.range_stop_loss_percent}%)"
                    )
                    return True

        return False

    def get_price_stop_loss_level(self) -> Optional[Decimal]:
        """
        Get the current price stop loss level.

        Returns:
            Stop loss price level, or None if not configured
        """
        if not self._config.price_stop_loss_enabled:
            return None

        # Absolute price takes priority
        if self._config.price_stop_loss is not None:
            return self._config.price_stop_loss

        # Calculate from range percentage
        if self._config.range_stop_loss_percent is not None:
            setup = self._order_manager.setup
            if setup is not None:
                lower_bound = setup.lower_price
                return lower_bound * (
                    Decimal("1") - self._config.range_stop_loss_percent / Decimal("100")
                )

        return None

    async def execute_price_stop_loss(self, current_price: Decimal) -> None:
        """
        Execute price-based stop loss.

        1. Cancel all pending orders
        2. Market sell any positions
        3. Send notification
        4. Stop the bot

        Args:
            current_price: Current market price
        """
        logger.warning(f"Executing price stop loss at {current_price}")

        # Get stop loss level for notification
        stop_level = self.get_price_stop_loss_level()

        # Cancel all pending orders first
        cancelled = await self._order_manager.cancel_all_orders()
        logger.info(f"Price stop loss: cancelled {cancelled} orders")

        # Calculate position
        total_quantity = self._calculate_total_position()

        # Market sell if we have positions
        sell_succeeded = False
        if total_quantity > 0:
            try:
                order = await self._order_manager._exchange.market_sell(
                    self._order_manager.symbol,
                    total_quantity,
                    self._order_manager._market_type,
                )

                loss = self._calculate_unrealized_pnl(current_price)
                sell_succeeded = True

                logger.warning(
                    f"Price stop loss executed: sold {total_quantity} at ~{current_price}, "
                    f"unrealized loss was {loss}"
                )

                # Send notification
                await self._notify_price_stop_loss(
                    current_price=current_price,
                    stop_level=stop_level,
                    quantity_sold=total_quantity,
                    loss=loss,
                    had_positions=True,
                )

            except Exception as e:
                logger.error(f"Failed to execute price stop loss market sell: {e}")
                await self._notify_price_stop_loss(
                    current_price=current_price,
                    stop_level=stop_level,
                    quantity_sold=Decimal("0"),
                    loss=Decimal("0"),
                    had_positions=True,
                    error=str(e),
                )
                # Keep state as PAUSED so manual intervention can happen
                self._state = RiskState.PAUSED
                logger.warning(
                    "Position sell failed - state set to PAUSED for manual intervention. "
                    f"Position: {total_quantity}"
                )
                return  # Don't set to STOPPED
        else:
            # No positions, just cancelled orders
            sell_succeeded = True
            logger.info("Price stop loss: no positions to sell, orders cancelled")
            await self._notify_price_stop_loss(
                current_price=current_price,
                stop_level=stop_level,
                quantity_sold=Decimal("0"),
                loss=Decimal("0"),
                had_positions=False,
            )

        # Update state only if successful
        if sell_succeeded:
            self._state = RiskState.STOPPED

    async def _notify_price_stop_loss(
        self,
        current_price: Decimal,
        stop_level: Optional[Decimal],
        quantity_sold: Decimal,
        loss: Decimal,
        had_positions: bool,
        error: Optional[str] = None,
    ) -> None:
        """Send price stop loss notification."""
        try:
            if error:
                title = "🚨 價格止損執行失敗"
                message = (
                    f"價格跌破止損線，但賣出失敗\n\n"
                    f"當前價格: {current_price:,.2f}\n"
                    f"止損價格: {stop_level:,.2f}\n"
                    f"錯誤: {error}\n\n"
                    f"請手動處理！"
                )
                level = "error"
            elif had_positions:
                title = "🛑 價格止損已執行"
                message = (
                    f"價格跌破止損線，已市價賣出\n\n"
                    f"當前價格: {current_price:,.2f}\n"
                    f"止損價格: {stop_level:,.2f}\n"
                    f"賣出數量: {quantity_sold}\n"
                    f"預估虧損: {loss:,.2f} USDT\n\n"
                    f"機器人已停止運行"
                )
                level = "warning"
            else:
                title = "🛑 價格止損觸發"
                message = (
                    f"價格跌破止損線\n\n"
                    f"當前價格: {current_price:,.2f}\n"
                    f"止損價格: {stop_level:,.2f}\n"
                    f"持倉數量: 0（無需賣出）\n"
                    f"已取消所有掛單\n\n"
                    f"機器人已停止運行"
                )
                level = "warning"

            await self._notifier.send(
                title=title,
                message=message,
                level=level,
            )
        except Exception as e:
            logger.warning(f"Failed to send price stop loss notification: {e}")

    async def execute_pnl_stop_loss(self, current_price: Decimal) -> None:
        """
        Execute P&L-based stop loss.

        Unlike _action_stop_loss, this method is direction-agnostic and always
        checks for positions to sell based on unrealized P&L.

        Args:
            current_price: Current market price
        """
        logger.warning(f"P&L stop loss triggered at {current_price}")

        # Cancel all pending orders first
        cancelled = await self._order_manager.cancel_all_orders()
        logger.info(f"P&L stop loss: cancelled {cancelled} orders")

        # Calculate position and unrealized P&L
        total_quantity = self._calculate_total_position()
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)

        # Market sell if we have positions
        if total_quantity > 0:
            try:
                order = await self._order_manager._exchange.market_sell(
                    self._order_manager.symbol,
                    total_quantity,
                    self._order_manager._market_type,
                )

                # Notify
                await self._notify_stop_loss(current_price, total_quantity, unrealized_pnl)

                logger.warning(
                    f"P&L stop loss executed: sold {total_quantity} at {current_price}, "
                    f"loss={unrealized_pnl}"
                )

            except Exception as e:
                logger.error(f"Failed to execute P&L stop loss: {e}")
        else:
            logger.info("P&L stop loss: no positions to sell, orders cancelled")

        # Update state
        self._state = RiskState.STOPPED

    def _calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """
        Calculate unrealized P&L from filled buy positions.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L (negative if losing)
        """
        setup = self._order_manager.setup
        if setup is None:
            return Decimal("0")

        total_pnl = Decimal("0")

        # Get fee rate from order manager for estimating sell fee
        fee_rate = getattr(
            self._order_manager, 'DEFAULT_FEE_RATE', Decimal("0.001")
        )

        # Calculate from filled history
        for record in self._order_manager._filled_history:
            if record.side == OrderSide.BUY and record.paired_record is None:
                # Unpaired buy = open position
                # Include buy fee paid and estimate sell fee
                estimated_sell_fee = current_price * record.quantity * fee_rate
                pnl = (
                    (current_price - record.price) * record.quantity
                    - record.fee  # Buy fee (already paid)
                    - estimated_sell_fee  # Sell fee (estimated)
                )
                total_pnl += pnl

        return total_pnl

    def _calculate_total_position(self) -> Decimal:
        """
        Calculate total position quantity from filled buys.

        Returns:
            Total quantity held
        """
        total = Decimal("0")

        for record in self._order_manager._filled_history:
            if record.side == OrderSide.BUY and record.paired_record is None:
                total += record.quantity

        return total

    # =========================================================================
    # Price Return Handling
    # =========================================================================

    async def on_price_return(self, current_price: Decimal) -> bool:
        """
        Handle price returning to grid range after breakout.

        Uses state lock to prevent TOCTOU vulnerabilities during state transition.

        Args:
            current_price: Current market price

        Returns:
            True if trading resumed successfully
        """
        async with self._state_lock:
            # Check state under lock
            if self._state not in (RiskState.PAUSED, RiskState.BREAKOUT_UPPER, RiskState.BREAKOUT_LOWER):
                return False

            if not self.check_price_return(current_price):
                return False

            logger.info(f"Price returned to grid range at {current_price}")

            # Re-place orders at empty levels
            try:
                placed = await self._order_manager.place_initial_orders()
                if placed == 0:
                    logger.warning("Price returned but no orders were placed - check grid state")
                    return False
            except Exception as e:
                logger.error(f"Failed to place orders on price return: {e}")
                return False

            # State is protected by lock, so no need for separate verification
            # Resume trading after successful order placement
            self._state = RiskState.NORMAL

        # Notify outside of lock to avoid holding lock during I/O
        await self._notify_price_return(current_price, placed)

        logger.info(f"Trading resumed: {placed} orders placed")

        return True

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _can_reset(self) -> bool:
        """Check if reset cooldown has passed."""
        if self._last_reset_time is None:
            return True

        elapsed = datetime.now(timezone.utc) - self._last_reset_time
        cooldown_seconds = self._config.cooldown_days * 24 * 60 * 60

        return elapsed.total_seconds() >= cooldown_seconds

    def update_config(self, config: RiskConfig) -> None:
        """
        Update risk configuration.

        Args:
            config: New RiskConfig
        """
        self._config = config
        logger.info("Risk configuration updated")

    def reset_state(self) -> None:
        """Reset risk manager state to normal."""
        self._state = RiskState.NORMAL
        self._last_breakout = None
        logger.info("Risk manager state reset to normal")

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _notify_breakout(
        self,
        direction: BreakoutDirection,
        price: Decimal,
        action: BreakoutAction,
    ) -> None:
        """Send breakout alert notification."""
        try:
            setup = self._order_manager.setup
            symbol = self._order_manager.symbol

            direction_str = "Upper" if direction == BreakoutDirection.UPPER else "Lower"
            bound = setup.upper_price if direction == BreakoutDirection.UPPER else setup.lower_price

            message = (
                f"{symbol} {direction_str} Breakout!\n"
                f"Price: {price}\n"
                f"Bound: {bound}\n"
                f"Action: {action.value}"
            )

            await self._notifier.send_warning(
                title="Grid Breakout Alert",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send breakout notification: {e}")

    async def _notify_stop_loss(
        self,
        price: Decimal,
        quantity: Decimal,
        loss: Decimal,
    ) -> None:
        """Send stop loss notification."""
        try:
            symbol = self._order_manager.symbol

            message = (
                f"{symbol} Stop Loss Executed\n"
                f"Sold: {quantity} @ {price}\n"
                f"Loss: {loss:.4f}"
            )

            await self._notifier.send_error(
                title="Stop Loss Triggered",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send stop loss notification: {e}")

    async def _notify_grid_reset(self, new_center: Decimal) -> None:
        """Send grid reset notification."""
        try:
            symbol = self._order_manager.symbol

            message = (
                f"{symbol} Grid Reset\n"
                f"New center: {new_center}\n"
                f"Please reconfigure grid"
            )

            await self._notifier.send_info(
                title="Grid Reset Required",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send reset notification: {e}")

    async def _notify_grid_expand(
        self,
        direction: BreakoutDirection,
        new_upper: Decimal,
        new_lower: Decimal,
    ) -> None:
        """Send grid expand notification."""
        try:
            symbol = self._order_manager.symbol
            direction_str = "upward" if direction == BreakoutDirection.UPPER else "downward"

            message = (
                f"{symbol} Grid Expanded ({direction_str})\n"
                f"New range: {new_lower} - {new_upper}"
            )

            await self._notifier.send_info(
                title="Grid Expanded",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send expand notification: {e}")

    async def _notify_price_return(
        self,
        price: Decimal,
        orders_placed: int,
    ) -> None:
        """Send price return notification."""
        try:
            symbol = self._order_manager.symbol

            message = (
                f"{symbol} Price Returned to Grid\n"
                f"Price: {price}\n"
                f"Orders placed: {orders_placed}\n"
                f"Trading resumed"
            )

            await self._notifier.send_success(
                title="Trading Resumed",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send return notification: {e}")

    # =========================================================================
    # Monitoring Loop (Prompt 22)
    # =========================================================================

    async def start_monitoring(self) -> bool:
        """
        Start the monitoring loop.

        Continuously monitors:
        - Price breakout
        - Stop loss
        - Daily loss limit
        - Health checks

        Returns:
            True if started successfully
        """
        if self._running or self._monitoring_task:
            logger.warning("Monitoring already running")
            return False

        try:
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Update bot state
            await self.change_state(BotState.RUNNING, "Monitoring started")

            logger.info("Monitoring started")
            return True
        except Exception as e:
            # Cleanup on failure
            self._running = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
                self._monitoring_task = None
            logger.error(f"Failed to start monitoring: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        # Cancel and cleanup any pending background tasks
        if self._pending_tasks:
            for task in list(self._pending_tasks):
                if not task.done():
                    task.cancel()
            # Wait for all tasks to complete
            if self._pending_tasks:
                await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        logger.info("Monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Monitoring loop started")

        while self._running:
            try:
                # Skip if not in running state
                if self._bot_state != BotState.RUNNING:
                    await asyncio.sleep(1)
                    continue

                # Get current price
                current_price = await self._order_manager._data_manager.get_price(
                    self._order_manager.symbol,
                    self._order_manager._market_type,
                )

                if current_price:
                    # Check breakout (only if not already in breakout/paused/stopped state)
                    # This prevents duplicate handling of the same breakout event
                    if self._state == RiskState.NORMAL:
                        direction = self.check_breakout(current_price)
                        if direction != BreakoutDirection.NONE:
                            await self.handle_breakout(direction, current_price)
                            continue

                    # Check stop loss (P&L based) - direction-agnostic
                    if self.check_stop_loss(current_price):
                        await self.execute_pnl_stop_loss(current_price)
                        continue

                    # Check price stop loss (works even without filled positions)
                    if self.check_price_stop_loss(current_price):
                        await self.execute_price_stop_loss(current_price)
                        continue

                    # Check daily loss
                    if self.check_daily_loss():
                        await self.pause("Daily loss limit exceeded")
                        continue

                    # Check price return if paused
                    if self._state == RiskState.PAUSED:
                        await self.on_price_return(current_price)

                # Run health check if interval passed
                await self._maybe_run_health_check()

                # Check for daily reset
                await self._maybe_reset_daily_stats()

                await asyncio.sleep(self._config.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)

        logger.info("Monitoring loop ended")

    async def _maybe_run_health_check(self) -> None:
        """Run health check if interval has passed."""
        now = datetime.now(timezone.utc)

        # Run health check on first call or when interval has passed
        if self._last_health_check is None:
            await self.run_health_check()
            self._last_health_check = now
            return

        elapsed = (now - self._last_health_check).total_seconds()
        if elapsed >= self._config.health_check_interval:
            await self.run_health_check()
            self._last_health_check = now

    async def _maybe_reset_daily_stats(self) -> None:
        """Reset daily stats if new day (UTC)."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Use date comparison to handle midnight boundary correctly
        # This ensures reset happens even if initialized exactly at midnight
        if today_start.date() > self._daily_start_time.date():
            await self.reset_daily_stats()

    # =========================================================================
    # Health Check (Prompt 22)
    # =========================================================================

    async def run_health_check(self) -> dict[str, bool]:
        """
        Run health check on all components.

        Returns:
            Dict with health status of each component
        """
        status = {
            "exchange_connected": False,
            "data_manager_connected": False,
            "order_sync": False,
            "overall": False,
        }

        try:
            # Re-sync time with server before health check
            try:
                await self._order_manager._exchange.spot.sync_time()
            except Exception as e:
                logger.warning(f"Time sync failed: {e}")

            # Check exchange connection
            if self._order_manager._exchange.is_connected:
                status["exchange_connected"] = True

            # Check data manager
            if self._order_manager._data_manager.is_connected:
                status["data_manager_connected"] = True

            # Check order sync by comparing local vs exchange
            sync_stats = await self._order_manager.sync_orders()
            status["order_sync"] = sync_stats.get("external", 0) == 0

            # Overall health
            status["overall"] = all([
                status["exchange_connected"],
                status["data_manager_connected"],
            ])

            if status["overall"]:
                self._consecutive_health_failures = 0
            else:
                self._consecutive_health_failures += 1

                # Pause if consecutive failures
                if self._consecutive_health_failures >= 3:
                    # Notify first, then pause to ensure notification reflects pre-pause state
                    await self._notify_health_failure(status)
                    await self.pause("Health check failed 3 times")

        except Exception as e:
            logger.error(f"Health check error: {e}")
            self._consecutive_health_failures += 1

        return status

    # =========================================================================
    # Additional Risk Checks (Prompt 22)
    # =========================================================================

    def check_daily_loss(self) -> bool:
        """
        Check if daily loss limit is exceeded.

        Returns:
            True if daily loss exceeds limit
        """
        setup = self._order_manager.setup
        if setup is None:
            return False

        total_investment = setup.config.total_investment
        if total_investment == 0:
            return False

        daily_loss_percent = (self._daily_pnl / total_investment) * Decimal("100")

        # Check if loss exceeds limit (negative value comparison)
        if daily_loss_percent <= -self._config.daily_loss_limit:
            logger.warning(
                f"Daily loss limit triggered: {daily_loss_percent:.2f}% <= -{self._config.daily_loss_limit}%"
            )
            return True

        return False

    def check_consecutive_losses(self) -> bool:
        """
        Check if consecutive losses exceed threshold.

        Returns:
            True if consecutive losses exceed limit
        """
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            logger.warning(
                f"Consecutive loss limit: {self._consecutive_losses} >= {self._config.max_consecutive_losses}"
            )
            return True
        return False

    def check_volatility(self, kline: Kline) -> bool:
        """
        Check if kline shows abnormal volatility.

        Args:
            kline: Kline to check

        Returns:
            True if volatility exceeds threshold
        """
        if kline.open == 0:
            return False

        # Calculate kline range as percentage
        range_percent = (kline.range / kline.open) * Decimal("100")

        if range_percent >= self._config.volatility_threshold:
            logger.warning(
                f"Abnormal volatility detected: {range_percent:.2f}% >= {self._config.volatility_threshold}%"
            )
            return True

        return False

    def on_order_failure(self) -> bool:
        """
        Record an order failure.

        Returns:
            True if failure threshold exceeded
        """
        self._order_failures += 1
        logger.warning(f"Order failure #{self._order_failures}")

        if self._order_failures >= self._config.order_failure_threshold:
            logger.warning(f"Order failure threshold exceeded: {self._order_failures}")
            return True

        return False

    def on_trade_completed(self, profit: Decimal) -> None:
        """
        Record a completed trade (buy->sell round trip).

        Args:
            profit: Trade profit (positive or negative)
        """
        # Update daily P&L
        self._daily_pnl += profit

        # Track consecutive losses
        if profit < 0:
            if not self._last_trade_profitable:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 1
            self._last_trade_profitable = False
        else:
            self._consecutive_losses = 0
            self._last_trade_profitable = True

        logger.debug(
            f"Trade completed: profit={profit}, daily_pnl={self._daily_pnl}, "
            f"consecutive_losses={self._consecutive_losses}"
        )

        # Check consecutive loss trigger
        if self.check_consecutive_losses():
            # Create and track the pause task to ensure it completes
            pause_task = asyncio.create_task(self.pause("Consecutive loss limit exceeded"))
            self._pending_tasks.add(pause_task)

            def _on_pause_done(task: asyncio.Task) -> None:
                self._pending_tasks.discard(task)
                if task.exception():
                    logger.error(f"Pause task failed: {task.exception()}")

            pause_task.add_done_callback(_on_pause_done)

    async def reset_daily_stats(self) -> None:
        """Reset daily statistics (called at 00:00 UTC)."""
        # Save yesterday's stats (could save to database here)
        yesterday_pnl = self._daily_pnl

        # Reset counters
        self._daily_pnl = Decimal("0")
        self._consecutive_losses = 0
        self._order_failures = 0
        self._daily_start_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        logger.info(f"Daily stats reset. Yesterday P&L: {yesterday_pnl}")

        # Optionally send daily report
        await self._notify_daily_reset(yesterday_pnl)

    # =========================================================================
    # Bot State Management (Prompt 22)
    # =========================================================================

    async def change_state(
        self,
        new_state: BotState,
        reason: str = "",
    ) -> bool:
        """
        Change bot state with validation.

        Args:
            new_state: Target state
            reason: Reason for state change

        Returns:
            True if state changed successfully
        """
        old_state = self._bot_state

        # Validate transition
        valid_transitions = VALID_STATE_TRANSITIONS.get(old_state, set())
        if new_state not in valid_transitions and new_state != old_state:
            logger.warning(
                f"Invalid state transition: {old_state.value} -> {new_state.value}"
            )
            return False

        # Execute exit action for old state
        await self._on_state_exit(old_state)

        # Update state
        self._bot_state = new_state

        # Execute entry action for new state
        await self._on_state_enter(new_state, reason)

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                result = callback(old_state, new_state, reason)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        logger.info(f"Bot state changed: {old_state.value} -> {new_state.value} ({reason})")

        return True

    async def _on_state_exit(self, state: BotState) -> None:
        """Execute actions when exiting a state."""
        if state == BotState.RUNNING:
            # Could save state snapshot here
            pass

    async def _on_state_enter(self, state: BotState, reason: str) -> None:
        """Execute actions when entering a state."""
        if state == BotState.PAUSED:
            self._pause_reason = reason
        elif state == BotState.RUNNING:
            self._pause_reason = ""

        # Send notification
        await self._notify_state_change(state, reason)

    def add_state_change_callback(
        self,
        callback: Callable[[BotState, BotState, str], Any],
    ) -> None:
        """Add callback for state changes."""
        self._state_change_callbacks.append(callback)

    # =========================================================================
    # Manual Control Interface (Prompt 22)
    # =========================================================================

    async def pause(self, reason: str = "Manual pause") -> bool:
        """
        Pause the bot.

        Cancels all pending orders but keeps positions.
        Uses state lock to ensure consistent state updates.

        Args:
            reason: Pause reason

        Returns:
            True if paused successfully
        """
        async with self._state_lock:
            if self._bot_state not in (BotState.RUNNING, BotState.ERROR):
                logger.warning(f"Cannot pause from state: {self._bot_state.value}")
                return False

            # Update RiskState FIRST to ensure consistency when callbacks are triggered
            self._state = RiskState.PAUSED

            # Update BotState (this triggers callbacks/notifications)
            if not await self.change_state(BotState.PAUSED, reason):
                # Rollback RiskState on failure
                self._state = RiskState.NORMAL
                logger.error("Failed to change BotState to PAUSED")
                return False

        # Cancel all orders OUTSIDE of lock to avoid holding lock during I/O
        # If cancel fails, we're already in PAUSED state which is correct
        cancelled = await self._order_manager.cancel_all_orders()

        logger.info(f"Bot paused: {reason}. Cancelled {cancelled} orders.")

        return True

    async def resume(self) -> bool:
        """
        Resume the bot from paused state.

        Uses state lock to ensure consistent state updates.

        Returns:
            True if resumed successfully
        """
        # Check price outside of lock to avoid holding lock during I/O
        setup = self._order_manager.setup
        if setup:
            current_price = await self._order_manager._data_manager.get_price(
                self._order_manager.symbol,
                self._order_manager._market_type,
            )

            if current_price:
                # Check if we're still in breakout
                direction = self.check_breakout(current_price)
                if direction != BreakoutDirection.NONE:
                    logger.warning(f"Cannot resume: still in {direction.value} breakout")
                    return False

        async with self._state_lock:
            if self._bot_state != BotState.PAUSED:
                logger.warning(f"Cannot resume from state: {self._bot_state.value}")
                return False

            # Update RiskState FIRST to ensure consistency when callbacks are triggered
            self._state = RiskState.NORMAL

            # Update BotState (this triggers callbacks/notifications)
            if not await self.change_state(BotState.RUNNING, "Resumed"):
                # Rollback RiskState on failure
                self._state = RiskState.PAUSED
                logger.error("Failed to change BotState to RUNNING")
                return False

        # Re-place orders outside of lock
        placed = await self._order_manager.place_initial_orders()

        logger.info(f"Bot resumed. Placed {placed} orders.")

        return True

    async def stop(self, clear_position: bool = False) -> bool:
        """
        Stop the bot gracefully.

        Args:
            clear_position: If True, market sell all positions

        Returns:
            True if stopped successfully
        """
        if self._bot_state in (BotState.STOPPED, BotState.STOPPING):
            logger.warning(f"Already stopping/stopped: {self._bot_state.value}")
            return False

        # Set stopping state
        await self.change_state(BotState.STOPPING, "Stop requested")

        # Stop monitoring
        await self.stop_monitoring()

        # Cancel all orders
        await self._order_manager.cancel_all_orders()

        # Clear positions if requested
        if clear_position:
            total_qty = self._calculate_total_position()
            if total_qty > 0:
                try:
                    await self._order_manager._exchange.market_sell(
                        self._order_manager.symbol,
                        total_qty,
                        self._order_manager._market_type,
                    )
                    logger.info(f"Cleared position: sold {total_qty}")
                except Exception as e:
                    logger.error(f"Failed to clear position: {e}")

        # Get final stats
        stats = self._order_manager.get_statistics()

        # Set stopped state
        self._state = RiskState.STOPPED
        await self.change_state(BotState.STOPPED, "Stopped")

        # Send final notification
        await self._notify_bot_stopped(stats, clear_position)

        logger.info("Bot stopped")

        return True

    async def force_stop(self) -> bool:
        """
        Force stop the bot immediately with market close.

        Returns:
            True if stopped
        """
        logger.warning("Force stop requested")

        # Force state
        self._running = False
        self._bot_state = BotState.STOPPING

        # Cancel all orders
        try:
            await self._order_manager.cancel_all_orders()
        except Exception as e:
            logger.error(f"Cancel orders failed: {e}")

        # Market sell all positions
        total_qty = self._calculate_total_position()
        if total_qty > 0:
            try:
                await self._order_manager._exchange.market_sell(
                    self._order_manager.symbol,
                    total_qty,
                    self._order_manager._market_type,
                )
            except Exception as e:
                logger.error(f"Market sell failed: {e}")

        self._state = RiskState.STOPPED
        self._bot_state = BotState.STOPPED

        await self._notify_bot_stopped({}, True)

        logger.warning("Bot force stopped")

        return True

    # =========================================================================
    # Additional Notifications (Prompt 22)
    # =========================================================================

    async def _notify_health_failure(self, status: dict[str, bool]) -> None:
        """Send health check failure notification."""
        try:
            symbol = self._order_manager.symbol
            failed = [k for k, v in status.items() if not v and k != "overall"]

            message = (
                f"{symbol} Health Check Failed\n"
                f"Failed: {', '.join(failed)}\n"
                f"Trading paused"
            )

            await self._notifier.send_warning(
                title="Health Check Alert",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send health notification: {e}")

    async def _notify_daily_reset(self, yesterday_pnl: Decimal) -> None:
        """Send daily reset notification."""
        try:
            symbol = self._order_manager.symbol
            stats = self._order_manager.get_statistics()

            message = (
                f"{symbol} Daily Reset\n"
                f"Yesterday P&L: {yesterday_pnl:.4f}\n"
                f"Total Trades: {stats.get('trade_count', 0)}\n"
                f"Total Profit: {stats.get('total_profit', 0):.4f}"
            )

            await self._notifier.send_info(
                title="Daily Summary",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send daily reset notification: {e}")

    async def _notify_state_change(self, new_state: BotState, reason: str) -> None:
        """Send state change notification."""
        try:
            symbol = self._order_manager.symbol

            message = (
                f"{symbol} State: {new_state.value}\n"
                f"Reason: {reason}"
            )

            # Choose notification level based on state
            if new_state in (BotState.ERROR, BotState.STOPPED):
                await self._notifier.send_warning(
                    title="Bot State Change",
                    message=message,
                )
            elif new_state == BotState.PAUSED:
                await self._notifier.send_warning(
                    title="Bot Paused",
                    message=message,
                )
            else:
                await self._notifier.send_info(
                    title="Bot State Change",
                    message=message,
                )
        except Exception as e:
            logger.warning(f"Failed to send state change notification: {e}")

    async def _notify_bot_stopped(
        self,
        stats: dict,
        cleared_position: bool,
    ) -> None:
        """Send bot stopped notification with final stats."""
        try:
            symbol = self._order_manager.symbol

            message = (
                f"{symbol} Bot Stopped\n"
                f"Total Profit: {stats.get('total_profit', 0):.4f}\n"
                f"Trade Count: {stats.get('trade_count', 0)}\n"
                f"Position Cleared: {'Yes' if cleared_position else 'No'}"
            )

            await self._notifier.send_info(
                title="Bot Stopped",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send stopped notification: {e}")

    # =========================================================================
    # Dynamic Grid Adjustment
    # =========================================================================

    def set_dynamic_adjust_config(self, config: DynamicAdjustConfig) -> None:
        """
        Set dynamic adjustment configuration.

        Args:
            config: DynamicAdjustConfig instance
        """
        self._dynamic_adjust_config = config
        logger.info(
            f"Dynamic adjust configured: threshold={config.breakout_threshold}, "
            f"cooldown={config.cooldown_days}d, max_rebuilds={config.max_rebuilds}"
        )

    def set_rebuild_callback(
        self,
        callback: Callable[[Decimal], Any],
    ) -> None:
        """
        Set callback for grid rebuild.

        The callback receives the new center price and should
        trigger grid recalculation in the bot.

        Args:
            callback: Async function(new_center_price) -> None
        """
        self._on_rebuild_callback = callback

    @property
    def rebuild_history(self) -> list[RebuildRecord]:
        """Get rebuild history."""
        return self._rebuild_history.copy()

    @property
    def rebuilds_in_cooldown_period(self) -> int:
        """Get number of rebuilds within cooldown period."""
        return len(self._get_recent_rebuilds())

    @property
    def can_rebuild(self) -> bool:
        """Check if rebuild is allowed (not in cooldown)."""
        # Check auto_rebuild_enabled from RiskConfig first
        if not self._config.auto_rebuild_enabled:
            return False

        # Use DynamicAdjustConfig if set, otherwise use RiskConfig
        if self._dynamic_adjust_config:
            if not self._dynamic_adjust_config.enabled:
                return False
            max_rebuilds = self._dynamic_adjust_config.max_rebuilds
        else:
            max_rebuilds = self._config.max_rebuilds_in_period

        return self.rebuilds_in_cooldown_period < max_rebuilds

    @property
    def next_rebuild_available(self) -> Optional[datetime]:
        """
        Get when the next rebuild will be available.

        Returns:
            datetime when rebuild becomes available, or None if already available
        """
        if self.can_rebuild:
            return None

        recent = self._get_recent_rebuilds()
        if not recent:
            return None

        # Get cooldown days from DynamicAdjustConfig or RiskConfig
        if self._dynamic_adjust_config:
            cooldown_days = self._dynamic_adjust_config.cooldown_days
        else:
            cooldown_days = self._config.cooldown_days

        # Find the oldest rebuild that will expire first
        oldest = min(recent, key=lambda r: r.timestamp)
        cooldown_delta = timedelta(days=cooldown_days)
        return oldest.timestamp + cooldown_delta

    def _get_recent_rebuilds(self) -> list[RebuildRecord]:
        """
        Get rebuilds within the cooldown period.

        Returns:
            List of RebuildRecord within cooldown window
        """
        # Get cooldown days from DynamicAdjustConfig or RiskConfig
        if self._dynamic_adjust_config:
            cooldown_days = self._dynamic_adjust_config.cooldown_days
        else:
            cooldown_days = self._config.cooldown_days

        now = datetime.now(timezone.utc)
        cooldown_delta = timedelta(days=cooldown_days)
        cutoff = now - cooldown_delta

        return [r for r in self._rebuild_history if r.timestamp > cutoff]

    def _clean_expired_rebuilds(self) -> int:
        """
        Remove rebuild records older than cooldown period.

        Returns:
            Number of records removed
        """
        # Get cooldown days from DynamicAdjustConfig or RiskConfig
        if self._dynamic_adjust_config:
            cooldown_days = self._dynamic_adjust_config.cooldown_days
        else:
            cooldown_days = self._config.cooldown_days

        now = datetime.now(timezone.utc)
        cooldown_delta = timedelta(days=cooldown_days)
        cutoff = now - cooldown_delta

        before_count = len(self._rebuild_history)
        self._rebuild_history = [r for r in self._rebuild_history if r.timestamp > cutoff]
        removed = before_count - len(self._rebuild_history)

        if removed > 0:
            logger.debug(f"Cleaned {removed} expired rebuild records")

        return removed

    def get_remaining_rebuilds(self) -> int:
        """
        Get the number of remaining rebuilds allowed.

        Returns:
            Number of rebuilds still available within cooldown period
        """
        if self._dynamic_adjust_config:
            max_rebuilds = self._dynamic_adjust_config.max_rebuilds
        else:
            max_rebuilds = self._config.max_rebuilds_in_period

        used = self.rebuilds_in_cooldown_period
        return max(0, max_rebuilds - used)

    def get_cooldown_status(self) -> dict:
        """
        Get comprehensive cooldown status.

        Returns:
            Dict with cooldown state information:
            - is_cooling_down: Whether currently in cooldown
            - rebuilds_used: Number of rebuilds used in period
            - rebuilds_remaining: Number of rebuilds still allowed
            - cooldown_period_days: Cooldown period in days
            - oldest_record_expires: When oldest rebuild record expires
            - next_available_rebuild: When next rebuild will be available (if in cooldown)
            - recent_records: List of rebuild records within cooldown period
        """
        # Get config values
        if self._dynamic_adjust_config:
            cooldown_days = self._dynamic_adjust_config.cooldown_days
            max_rebuilds = self._dynamic_adjust_config.max_rebuilds
        else:
            cooldown_days = self._config.cooldown_days
            max_rebuilds = self._config.max_rebuilds_in_period

        # Get recent rebuilds
        recent = self._get_recent_rebuilds()
        rebuilds_used = len(recent)
        rebuilds_remaining = max(0, max_rebuilds - rebuilds_used)
        is_cooling_down = rebuilds_remaining == 0

        # Calculate oldest record expiry
        oldest_record_expires = None
        if recent:
            oldest = min(recent, key=lambda r: r.timestamp)
            oldest_record_expires = oldest.timestamp + timedelta(days=cooldown_days)

        # Next available rebuild time
        next_available = None
        if is_cooling_down and oldest_record_expires:
            next_available = oldest_record_expires

        return {
            "is_cooling_down": is_cooling_down,
            "rebuilds_used": rebuilds_used,
            "rebuilds_remaining": rebuilds_remaining,
            "cooldown_period_days": cooldown_days,
            "oldest_record_expires": oldest_record_expires,
            "next_available_rebuild": next_available,
            "recent_records": recent,
        }

    def check_dynamic_adjust_trigger(
        self,
        current_price: Decimal,
    ) -> Optional[str]:
        """
        Check if dynamic adjustment should be triggered.

        Trigger conditions:
        - Upper: current_price > upper_price × (1 + threshold)
        - Lower: current_price < lower_price × (1 - threshold)

        Args:
            current_price: Current market price

        Returns:
            "upper" or "lower" if triggered, None otherwise
        """
        # Check if auto rebuild is enabled
        if not self._config.auto_rebuild_enabled:
            return None

        # If DynamicAdjustConfig is set, check its enabled flag too
        if self._dynamic_adjust_config and not self._dynamic_adjust_config.enabled:
            return None

        setup = self._order_manager.setup
        if setup is None:
            return None

        # Get threshold from DynamicAdjustConfig or RiskConfig
        if self._dynamic_adjust_config:
            threshold = self._dynamic_adjust_config.breakout_threshold
        else:
            # Use RiskConfig's breakout_threshold (convert from % to decimal)
            threshold = self._config.breakout_threshold / Decimal("100")

        # Calculate trigger thresholds (e.g., 4% beyond bounds)
        upper_trigger = setup.upper_price * (Decimal("1") + threshold)
        lower_trigger = setup.lower_price * (Decimal("1") - threshold)

        if current_price > upper_trigger:
            logger.info(
                f"Dynamic adjust trigger: upper breakout "
                f"({current_price} > {upper_trigger})"
            )
            return "upper"
        elif current_price < lower_trigger:
            logger.info(
                f"Dynamic adjust trigger: lower breakout "
                f"({current_price} < {lower_trigger})"
            )
            return "lower"

        return None

    async def execute_dynamic_adjust(
        self,
        current_price: Decimal,
        trigger_direction: str,
    ) -> bool:
        """
        Execute dynamic grid adjustment.

        Flow:
        1. Clean expired rebuild records
        2. Check cooldown limit
        3. Cancel all orders
        4. Record rebuild event
        5. Trigger callback for grid recalculation

        Args:
            current_price: Current market price (new center)
            trigger_direction: "upper" or "lower"

        Returns:
            True if rebuild executed successfully
        """
        setup = self._order_manager.setup
        if setup is None:
            return False

        if not self._config.auto_rebuild_enabled:
            return False

        # Step 1: Clean expired records
        self._clean_expired_rebuilds()

        # Step 2: Check cooldown limit
        if not self.can_rebuild:
            recent_count = self.rebuilds_in_cooldown_period
            next_available = self.next_rebuild_available

            # Get max_rebuilds and cooldown_days from config
            if self._dynamic_adjust_config:
                max_rebuilds = self._dynamic_adjust_config.max_rebuilds
                cooldown_days = self._dynamic_adjust_config.cooldown_days
            else:
                max_rebuilds = self._config.max_rebuilds_in_period
                cooldown_days = self._config.cooldown_days

            logger.warning(
                f"Rebuild blocked by cooldown: {recent_count}/{max_rebuilds} "
                f"in last {cooldown_days} days. "
                f"Next available: {next_available}"
            )
            await self._notify_rebuild_blocked(recent_count, next_available)
            return False

        logger.info(
            f"Executing dynamic grid adjustment at {current_price} "
            f"(trigger: {trigger_direction})"
        )

        # Step 3: Cancel all orders
        cancelled = await self._order_manager.cancel_all_orders()
        logger.info(f"Cancelled {cancelled} orders for rebuild")

        # Record old bounds for history
        old_upper = setup.upper_price
        old_lower = setup.lower_price

        # Step 4: Trigger callback for grid recalculation
        # The callback will recalculate grid with current_price as center
        if self._on_rebuild_callback:
            try:
                result = self._on_rebuild_callback(current_price)
                if asyncio.iscoroutine(result):
                    await result

                # Get new setup after rebuild
                new_setup = self._order_manager.setup
                new_upper = new_setup.upper_price if new_setup else current_price
                new_lower = new_setup.lower_price if new_setup else current_price

            except Exception as e:
                logger.error(f"Rebuild callback failed: {e}")
                return False
        else:
            # No callback - estimate new bounds
            new_upper = current_price
            new_lower = current_price
            logger.warning("No rebuild callback set - grid not actually rebuilt")

        # Step 5: Record rebuild event
        record = RebuildRecord(
            timestamp=datetime.now(timezone.utc),
            reason=f"{trigger_direction}_breakout",
            old_upper=old_upper,
            old_lower=old_lower,
            new_upper=new_upper,
            new_lower=new_lower,
            trigger_price=current_price,
        )
        self._rebuild_history.append(record)

        # Send notification
        await self._notify_grid_rebuilt(record)

        logger.info(
            f"Grid rebuilt: old=[{old_upper}, {old_lower}] -> "
            f"new=[{new_upper}, {new_lower}], "
            f"rebuilds in period: {self.rebuilds_in_cooldown_period}"
        )

        return True

    async def check_and_execute_dynamic_adjust(
        self,
        current_price: Decimal,
    ) -> bool:
        """
        Combined check and execute for dynamic adjustment.

        Called on each K-line close or price update.

        Args:
            current_price: Current market price

        Returns:
            True if rebuild was triggered and executed
        """
        trigger = self.check_dynamic_adjust_trigger(current_price)
        if trigger is None:
            return False

        return await self.execute_dynamic_adjust(current_price, trigger)

    async def _notify_grid_rebuilt(self, record: RebuildRecord) -> None:
        """Send grid rebuild notification."""
        try:
            symbol = self._order_manager.symbol

            # Get max_rebuilds and cooldown_days from config
            if self._dynamic_adjust_config:
                max_rebuilds = self._dynamic_adjust_config.max_rebuilds
                cooldown_days = self._dynamic_adjust_config.cooldown_days
            else:
                max_rebuilds = self._config.max_rebuilds_in_period
                cooldown_days = self._config.cooldown_days

            remaining = max_rebuilds - self.rebuilds_in_cooldown_period

            message = (
                f"📊 {symbol} Grid Rebuilt\n\n"
                f"Trigger: {record.reason}\n"
                f"Price: {record.trigger_price}\n\n"
                f"Old Range: {record.old_lower} - {record.old_upper}\n"
                f"New Range: {record.new_lower} - {record.new_upper}\n\n"
                f"Rebuilds remaining: {remaining}/{max_rebuilds} "
                f"(in {cooldown_days} days)"
            )

            await self._notifier.send_info(
                title="Grid Dynamically Adjusted",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send rebuild notification: {e}")

    async def _notify_rebuild_blocked(
        self,
        current_count: int,
        next_available: Optional[datetime],
    ) -> None:
        """Send rebuild blocked notification."""
        try:
            symbol = self._order_manager.symbol
            next_str = next_available.strftime("%Y-%m-%d %H:%M UTC") if next_available else "N/A"

            # Get max_rebuilds from config
            if self._dynamic_adjust_config:
                max_rebuilds = self._dynamic_adjust_config.max_rebuilds
            else:
                max_rebuilds = self._config.max_rebuilds_in_period

            message = (
                f"⚠️ {symbol} Rebuild Blocked\n\n"
                f"Cooldown limit reached: {current_count}/{max_rebuilds}\n"
                f"Next rebuild available: {next_str}\n\n"
                f"The grid will not be adjusted until cooldown expires."
            )

            await self._notifier.send_warning(
                title="Grid Rebuild Blocked",
                message=message,
            )
        except Exception as e:
            logger.warning(f"Failed to send blocked notification: {e}")
