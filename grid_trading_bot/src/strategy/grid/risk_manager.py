"""
Grid Risk Manager.

Handles price breakout detection, risk actions, and grid recovery.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from core import get_logger
from core.models import MarketType, OrderSide
from notification import NotificationManager

from .order_manager import GridOrderManager

logger = get_logger(__name__)


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
    RESET_GRID = "reset"    # Reset grid around current price
    EXPAND_GRID = "expand"  # Expand grid range in breakout direction


class RiskState(str, Enum):
    """Current risk management state."""

    NORMAL = "normal"           # Normal operation
    BREAKOUT_UPPER = "breakout_upper"  # Upper breakout detected
    BREAKOUT_LOWER = "breakout_lower"  # Lower breakout detected
    PAUSED = "paused"           # Trading paused
    STOPPED = "stopped"         # Bot stopped (stop loss triggered)


@dataclass
class RiskConfig:
    """
    Risk management configuration.

    Example:
        >>> config = RiskConfig(
        ...     upper_breakout_action=BreakoutAction.PAUSE,
        ...     lower_breakout_action=BreakoutAction.STOP_LOSS,
        ...     stop_loss_percent=Decimal("15"),
        ... )
    """

    # Breakout actions
    upper_breakout_action: BreakoutAction = BreakoutAction.PAUSE
    lower_breakout_action: BreakoutAction = BreakoutAction.PAUSE

    # Stop loss threshold (percentage)
    stop_loss_percent: Decimal = field(default_factory=lambda: Decimal("20"))

    # Breakout buffer (percentage beyond grid bounds)
    breakout_buffer: Decimal = field(default_factory=lambda: Decimal("0.5"))

    # Auto reset settings
    auto_reset_enabled: bool = False
    reset_cooldown_minutes: int = 60

    # Expand grid settings
    expand_atr_multiplier: Decimal = field(default_factory=lambda: Decimal("1.5"))

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.stop_loss_percent, Decimal):
            self.stop_loss_percent = Decimal(str(self.stop_loss_percent))
        if not isinstance(self.breakout_buffer, Decimal):
            self.breakout_buffer = Decimal(str(self.breakout_buffer))
        if not isinstance(self.expand_atr_multiplier, Decimal):
            self.expand_atr_multiplier = Decimal(str(self.expand_atr_multiplier))


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
            self._state = RiskState.BREAKOUT_UPPER
        else:
            action = self._config.lower_breakout_action
            self._state = RiskState.BREAKOUT_LOWER

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
        # No action needed, just wait

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

        # Calculate from filled history
        for record in self._order_manager._filled_history:
            if record.side == OrderSide.BUY and record.paired_record is None:
                # Unpaired buy = open position
                pnl = (current_price - record.price) * record.quantity
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

        Args:
            current_price: Current market price

        Returns:
            True if trading resumed
        """
        if self._state not in (RiskState.PAUSED, RiskState.BREAKOUT_UPPER, RiskState.BREAKOUT_LOWER):
            return False

        if not self.check_price_return(current_price):
            return False

        logger.info(f"Price returned to grid range at {current_price}")

        # Resume trading
        self._state = RiskState.NORMAL

        # Re-place orders at empty levels
        placed = await self._order_manager.place_initial_orders()

        # Notify
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
        cooldown_seconds = self._config.reset_cooldown_minutes * 60

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
