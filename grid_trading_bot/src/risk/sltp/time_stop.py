"""
Time-Based Stop Loss.

Provides time-based exit rules for positions:
- Exit after maximum holding time
- Exit if no profit within time threshold
- Exit at specific times (session end, etc.)
- Gradual position reduction over time

Part of the SLTP management system.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from src.core import get_logger

logger = get_logger(__name__)


class TimeStopType(Enum):
    """Type of time-based stop."""

    MAX_HOLDING = "max_holding"  # Exit after max holding time
    NO_PROFIT = "no_profit"  # Exit if no profit within time
    SESSION_END = "session_end"  # Exit at session end
    SCHEDULED = "scheduled"  # Exit at specific time
    GRADUAL = "gradual"  # Gradually reduce position over time


class TimeStopCondition(Enum):
    """Condition for triggering time stop."""

    ALWAYS = "always"  # Always trigger at time
    IF_LOSS = "if_loss"  # Only if position is at loss
    IF_NO_PROFIT = "if_no_profit"  # Only if no profit (including break-even)
    IF_BELOW_TARGET = "if_below_target"  # Only if below profit target


class TimeStopAction(Enum):
    """Action to take when time stop triggers."""

    CLOSE_FULL = "close_full"  # Close entire position
    CLOSE_PARTIAL = "close_partial"  # Close percentage of position
    REDUCE_TO_SIZE = "reduce_to_size"  # Reduce to specific size
    MOVE_TO_BREAKEVEN = "move_to_breakeven"  # Move stop to break-even
    TIGHTEN_STOP = "tighten_stop"  # Tighten stop loss


@dataclass
class TimeStopConfig:
    """Configuration for time-based stop."""

    stop_type: TimeStopType = TimeStopType.MAX_HOLDING
    condition: TimeStopCondition = TimeStopCondition.ALWAYS
    action: TimeStopAction = TimeStopAction.CLOSE_FULL

    # Time settings
    max_holding_minutes: int = 60  # Max holding time in minutes
    no_profit_minutes: int = 30  # Time without profit before exit
    session_end_time: Optional[time] = None  # Session end time (UTC)
    scheduled_exit_time: Optional[time] = None  # Scheduled exit time (UTC)
    exit_before_close_minutes: int = 5  # Minutes before session close

    # Gradual exit settings
    gradual_start_minutes: int = 30  # Start gradual exit after N minutes
    gradual_interval_minutes: int = 10  # Interval between reductions
    gradual_reduction_pct: Decimal = Decimal("0.25")  # Reduce by 25% each time

    # Partial close settings
    partial_close_pct: Decimal = Decimal("0.50")  # Close 50% on partial

    # Tighten stop settings
    tighten_stop_pct: Decimal = Decimal("0.50")  # Tighten by 50%

    # Profit target for IF_BELOW_TARGET condition
    profit_target_pct: Decimal = Decimal("0.02")  # 2% target

    # Warning settings
    warning_before_minutes: int = 5  # Warn N minutes before trigger

    enabled: bool = True


@dataclass
class TimeStopState:
    """State tracking for time-based stop on a position."""

    symbol: str
    entry_time: datetime
    entry_price: Decimal
    is_long: bool
    quantity: Decimal

    # Tracking
    highest_pnl_pct: Decimal = Decimal("0")
    last_profit_time: Optional[datetime] = None
    gradual_exits_count: int = 0
    warning_sent: bool = False
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    trigger_reason: str = ""

    # Computed
    remaining_quantity: Decimal = field(init=False)

    def __post_init__(self):
        self.remaining_quantity = self.quantity
        self.last_profit_time = self.entry_time

    @property
    def holding_duration(self) -> timedelta:
        """Get current holding duration."""
        return datetime.now(timezone.utc) - self.entry_time

    @property
    def holding_minutes(self) -> float:
        """Get holding duration in minutes."""
        return self.holding_duration.total_seconds() / 60

    @property
    def time_since_profit(self) -> timedelta:
        """Get time since last profit."""
        if self.last_profit_time is None:
            return self.holding_duration
        return datetime.now(timezone.utc) - self.last_profit_time

    @property
    def minutes_since_profit(self) -> float:
        """Get minutes since last profit."""
        return self.time_since_profit.total_seconds() / 60

    def update_pnl(self, current_pnl_pct: Decimal) -> None:
        """Update P&L tracking."""
        if current_pnl_pct > self.highest_pnl_pct:
            self.highest_pnl_pct = current_pnl_pct

        if current_pnl_pct > Decimal("0"):
            self.last_profit_time = datetime.now(timezone.utc)

    def record_gradual_exit(self, quantity_closed: Decimal) -> None:
        """Record a gradual exit."""
        self.gradual_exits_count += 1
        self.remaining_quantity -= quantity_closed

    def mark_triggered(self, reason: str) -> None:
        """Mark time stop as triggered."""
        self.triggered = True
        self.triggered_at = datetime.now(timezone.utc)
        self.trigger_reason = reason


@dataclass
class TimeStopResult:
    """Result of time stop check."""

    should_exit: bool
    action: TimeStopAction
    reason: str
    quantity_to_close: Decimal = Decimal("0")
    new_stop_price: Optional[Decimal] = None
    is_warning: bool = False  # True if this is a warning, not actual trigger
    minutes_until_trigger: Optional[float] = None


class TimeBasedStopLoss:
    """
    Time-based stop loss manager.

    Manages time-based exit rules for positions including:
    - Maximum holding time exits
    - No-profit timeout exits
    - Session end exits
    - Gradual position reduction

    Example:
        >>> config = TimeStopConfig(
        ...     stop_type=TimeStopType.MAX_HOLDING,
        ...     max_holding_minutes=60,
        ...     condition=TimeStopCondition.IF_NO_PROFIT,
        ... )
        >>> manager = TimeBasedStopLoss(config)
        >>> state = manager.create_state("BTCUSDT", entry_price, is_long, quantity)
        >>> result = manager.check(state, current_price)
        >>> if result.should_exit:
        ...     # Execute exit
    """

    def __init__(
        self,
        config: TimeStopConfig,
        on_warning: Optional[Callable[[str, str, float], None]] = None,
        on_trigger: Optional[Callable[[str, TimeStopResult], None]] = None,
        get_session_end: Optional[Callable[[str], Optional[time]]] = None,
    ):
        """
        Initialize TimeBasedStopLoss.

        Args:
            config: Time stop configuration
            on_warning: Callback (symbol, message, minutes_left) for warnings
            on_trigger: Callback (symbol, result) when time stop triggers
            get_session_end: Callback to get session end time for symbol
        """
        self._config = config
        self._on_warning = on_warning
        self._on_trigger = on_trigger
        self._get_session_end = get_session_end

        # State tracking
        self._states: Dict[str, TimeStopState] = {}

        # Statistics
        self._total_checks: int = 0
        self._total_triggers: int = 0
        self._triggers_by_type: Dict[TimeStopType, int] = {t: 0 for t in TimeStopType}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> TimeStopConfig:
        """Get configuration."""
        return self._config

    @property
    def active_positions(self) -> List[str]:
        """Get symbols with active time stop tracking."""
        return [sym for sym, state in self._states.items() if not state.triggered]

    # =========================================================================
    # State Management
    # =========================================================================

    def create_state(
        self,
        symbol: str,
        entry_price: Decimal,
        is_long: bool,
        quantity: Decimal,
        entry_time: Optional[datetime] = None,
    ) -> TimeStopState:
        """
        Create time stop state for a new position.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            is_long: True if long position
            quantity: Position quantity
            entry_time: Entry timestamp (defaults to now)

        Returns:
            TimeStopState for tracking
        """
        state = TimeStopState(
            symbol=symbol,
            entry_time=entry_time or datetime.now(timezone.utc),
            entry_price=entry_price,
            is_long=is_long,
            quantity=quantity,
        )

        self._states[symbol] = state
        logger.info(f"Created time stop state for {symbol}")

        return state

    def get_state(self, symbol: str) -> Optional[TimeStopState]:
        """Get time stop state for symbol."""
        return self._states.get(symbol)

    def remove_state(self, symbol: str) -> bool:
        """Remove time stop state for symbol."""
        if symbol in self._states:
            del self._states[symbol]
            logger.info(f"Removed time stop state for {symbol}")
            return True
        return False

    # =========================================================================
    # Main Check Method
    # =========================================================================

    def check(
        self,
        symbol: str,
        current_price: Decimal,
        current_time: Optional[datetime] = None,
    ) -> TimeStopResult:
        """
        Check if time stop should trigger for a position.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_time: Current time (defaults to now UTC)

        Returns:
            TimeStopResult with action to take
        """
        state = self._states.get(symbol)
        if state is None or state.triggered:
            return TimeStopResult(
                should_exit=False,
                action=TimeStopAction.CLOSE_FULL,
                reason="No active state",
            )

        if not self._config.enabled:
            return TimeStopResult(
                should_exit=False,
                action=TimeStopAction.CLOSE_FULL,
                reason="Time stop disabled",
            )

        self._total_checks += 1
        now = current_time or datetime.now(timezone.utc)

        # Update P&L
        current_pnl_pct = self._calculate_pnl_pct(state, current_price)
        state.update_pnl(current_pnl_pct)

        # Check based on stop type
        if self._config.stop_type == TimeStopType.MAX_HOLDING:
            result = self._check_max_holding(state, current_pnl_pct, now)
        elif self._config.stop_type == TimeStopType.NO_PROFIT:
            result = self._check_no_profit(state, current_pnl_pct, now)
        elif self._config.stop_type == TimeStopType.SESSION_END:
            result = self._check_session_end(state, current_pnl_pct, now)
        elif self._config.stop_type == TimeStopType.SCHEDULED:
            result = self._check_scheduled(state, current_pnl_pct, now)
        elif self._config.stop_type == TimeStopType.GRADUAL:
            result = self._check_gradual(state, current_pnl_pct, now)
        else:
            result = TimeStopResult(
                should_exit=False,
                action=TimeStopAction.CLOSE_FULL,
                reason="Unknown stop type",
            )

        # Handle result
        if result.should_exit and not result.is_warning:
            # For gradual exits, only trigger callback but don't mark as fully triggered
            # unless we're closing the full remaining position
            if result.action == TimeStopAction.CLOSE_PARTIAL:
                self._handle_gradual_trigger(symbol, state, result)
            else:
                self._handle_trigger(symbol, state, result)
        elif result.is_warning and not state.warning_sent:
            self._handle_warning(symbol, state, result)

        return result

    # =========================================================================
    # Check Methods by Type
    # =========================================================================

    def _check_max_holding(
        self,
        state: TimeStopState,
        current_pnl_pct: Decimal,
        now: datetime,
    ) -> TimeStopResult:
        """Check max holding time stop."""
        holding_minutes = state.holding_minutes
        max_minutes = self._config.max_holding_minutes
        warning_minutes = max_minutes - self._config.warning_before_minutes

        # Check if condition is met
        if not self._check_condition(current_pnl_pct):
            return TimeStopResult(
                should_exit=False,
                action=self._config.action,
                reason="Condition not met",
            )

        # Trigger first (takes priority over warning)
        if holding_minutes >= max_minutes:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"Max holding time reached ({holding_minutes:.1f}m >= {max_minutes}m)",
                quantity_to_close=self._get_close_quantity(state),
            )

        # Warning
        if holding_minutes >= warning_minutes and not state.warning_sent:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"Approaching max holding time ({max_minutes - holding_minutes:.1f}m left)",
                quantity_to_close=self._get_close_quantity(state),
                is_warning=True,
                minutes_until_trigger=max_minutes - holding_minutes,
            )

        return TimeStopResult(
            should_exit=False,
            action=self._config.action,
            reason=f"Holding time OK ({holding_minutes:.1f}m / {max_minutes}m)",
            minutes_until_trigger=max_minutes - holding_minutes,
        )

    def _check_no_profit(
        self,
        state: TimeStopState,
        current_pnl_pct: Decimal,
        now: datetime,
    ) -> TimeStopResult:
        """Check no-profit timeout stop."""
        minutes_since_profit = state.minutes_since_profit
        threshold_minutes = self._config.no_profit_minutes
        warning_minutes = threshold_minutes - self._config.warning_before_minutes

        # If currently in profit, reset timer
        if current_pnl_pct > Decimal("0"):
            return TimeStopResult(
                should_exit=False,
                action=self._config.action,
                reason="Currently in profit",
            )

        # Trigger first (takes priority over warning)
        if minutes_since_profit >= threshold_minutes:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"No profit timeout ({minutes_since_profit:.1f}m >= {threshold_minutes}m)",
                quantity_to_close=self._get_close_quantity(state),
            )

        # Warning
        if minutes_since_profit >= warning_minutes and not state.warning_sent:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"No profit for {minutes_since_profit:.1f}m, "
                f"exit in {threshold_minutes - minutes_since_profit:.1f}m",
                quantity_to_close=self._get_close_quantity(state),
                is_warning=True,
                minutes_until_trigger=threshold_minutes - minutes_since_profit,
            )

        return TimeStopResult(
            should_exit=False,
            action=self._config.action,
            reason=f"Time since profit: {minutes_since_profit:.1f}m / {threshold_minutes}m",
            minutes_until_trigger=threshold_minutes - minutes_since_profit,
        )

    def _check_session_end(
        self,
        state: TimeStopState,
        current_pnl_pct: Decimal,
        now: datetime,
    ) -> TimeStopResult:
        """Check session end stop."""
        session_end = self._config.session_end_time

        # Try to get symbol-specific session end
        if self._get_session_end:
            symbol_end = self._get_session_end(state.symbol)
            if symbol_end:
                session_end = symbol_end

        if session_end is None:
            return TimeStopResult(
                should_exit=False,
                action=self._config.action,
                reason="No session end time configured",
            )

        # Convert to datetime for comparison
        today = now.date()
        session_end_dt = datetime.combine(today, session_end, tzinfo=timezone.utc)

        # If past session end, use tomorrow
        if now > session_end_dt:
            session_end_dt += timedelta(days=1)

        minutes_until_end = (session_end_dt - now).total_seconds() / 60
        exit_before = self._config.exit_before_close_minutes
        warning_minutes = exit_before + self._config.warning_before_minutes

        # Check if condition is met
        if not self._check_condition(current_pnl_pct):
            return TimeStopResult(
                should_exit=False,
                action=self._config.action,
                reason="Condition not met",
                minutes_until_trigger=minutes_until_end - exit_before,
            )

        # Trigger first (takes priority over warning)
        if minutes_until_end <= exit_before:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"Session end exit ({minutes_until_end:.1f}m until close)",
                quantity_to_close=self._get_close_quantity(state),
            )

        # Warning
        if minutes_until_end <= warning_minutes and not state.warning_sent:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"Session ending in {minutes_until_end:.1f}m",
                quantity_to_close=self._get_close_quantity(state),
                is_warning=True,
                minutes_until_trigger=minutes_until_end - exit_before,
            )

        return TimeStopResult(
            should_exit=False,
            action=self._config.action,
            reason=f"Session end in {minutes_until_end:.1f}m",
            minutes_until_trigger=minutes_until_end - exit_before,
        )

    def _check_scheduled(
        self,
        state: TimeStopState,
        current_pnl_pct: Decimal,
        now: datetime,
    ) -> TimeStopResult:
        """Check scheduled exit time stop."""
        exit_time = self._config.scheduled_exit_time

        if exit_time is None:
            return TimeStopResult(
                should_exit=False,
                action=self._config.action,
                reason="No scheduled exit time configured",
            )

        # Convert to datetime for comparison
        today = now.date()
        exit_dt = datetime.combine(today, exit_time, tzinfo=timezone.utc)

        # If past exit time, check if we should have exited
        if now > exit_dt:
            # Check if position was opened after scheduled time
            if state.entry_time > exit_dt:
                exit_dt += timedelta(days=1)
            else:
                return TimeStopResult(
                    should_exit=True,
                    action=self._config.action,
                    reason=f"Past scheduled exit time ({exit_time})",
                    quantity_to_close=self._get_close_quantity(state),
                )

        minutes_until_exit = (exit_dt - now).total_seconds() / 60

        # Check if condition is met
        if not self._check_condition(current_pnl_pct):
            return TimeStopResult(
                should_exit=False,
                action=self._config.action,
                reason="Condition not met",
                minutes_until_trigger=minutes_until_exit,
            )

        # Trigger first (takes priority over warning)
        if minutes_until_exit <= 0:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"Scheduled exit time reached ({exit_time})",
                quantity_to_close=self._get_close_quantity(state),
            )

        # Warning
        if minutes_until_exit <= self._config.warning_before_minutes and not state.warning_sent:
            return TimeStopResult(
                should_exit=True,
                action=self._config.action,
                reason=f"Scheduled exit in {minutes_until_exit:.1f}m",
                quantity_to_close=self._get_close_quantity(state),
                is_warning=True,
                minutes_until_trigger=minutes_until_exit,
            )

        return TimeStopResult(
            should_exit=False,
            action=self._config.action,
            reason=f"Scheduled exit in {minutes_until_exit:.1f}m",
            minutes_until_trigger=minutes_until_exit,
        )

    def _check_gradual(
        self,
        state: TimeStopState,
        current_pnl_pct: Decimal,
        now: datetime,
    ) -> TimeStopResult:
        """Check gradual exit stop."""
        holding_minutes = state.holding_minutes
        start_after = self._config.gradual_start_minutes
        interval = self._config.gradual_interval_minutes
        reduction_pct = self._config.gradual_reduction_pct

        # Not yet time to start gradual exit
        if holding_minutes < start_after:
            return TimeStopResult(
                should_exit=False,
                action=TimeStopAction.CLOSE_PARTIAL,
                reason=f"Gradual exit starts after {start_after}m",
                minutes_until_trigger=start_after - holding_minutes,
            )

        # Check if condition is met
        if not self._check_condition(current_pnl_pct):
            return TimeStopResult(
                should_exit=False,
                action=TimeStopAction.CLOSE_PARTIAL,
                reason="Condition not met",
            )

        # Calculate expected exits
        time_in_gradual = holding_minutes - start_after
        expected_exits = int(time_in_gradual / interval) + 1

        # Check if we need another exit
        if state.gradual_exits_count < expected_exits:
            # Calculate quantity to close
            quantity_to_close = state.remaining_quantity * reduction_pct

            # Make sure we don't close more than remaining
            if quantity_to_close >= state.remaining_quantity:
                quantity_to_close = state.remaining_quantity

            if state.remaining_quantity > Decimal("0"):
                return TimeStopResult(
                    should_exit=True,
                    action=TimeStopAction.CLOSE_PARTIAL,
                    reason=f"Gradual exit #{expected_exits} "
                    f"(after {holding_minutes:.1f}m)",
                    quantity_to_close=quantity_to_close,
                )

        # Calculate time until next exit
        next_exit_time = start_after + (state.gradual_exits_count * interval)
        minutes_until_next = next_exit_time - holding_minutes

        return TimeStopResult(
            should_exit=False,
            action=TimeStopAction.CLOSE_PARTIAL,
            reason=f"Gradual exit: {state.gradual_exits_count} exits done, "
            f"remaining qty: {state.remaining_quantity}",
            minutes_until_trigger=max(0, minutes_until_next),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _check_condition(self, current_pnl_pct: Decimal) -> bool:
        """Check if the configured condition is met."""
        condition = self._config.condition

        if condition == TimeStopCondition.ALWAYS:
            return True
        elif condition == TimeStopCondition.IF_LOSS:
            return current_pnl_pct < Decimal("0")
        elif condition == TimeStopCondition.IF_NO_PROFIT:
            return current_pnl_pct <= Decimal("0")
        elif condition == TimeStopCondition.IF_BELOW_TARGET:
            return current_pnl_pct < self._config.profit_target_pct

        return True

    def _calculate_pnl_pct(self, state: TimeStopState, current_price: Decimal) -> Decimal:
        """Calculate current P&L percentage."""
        if state.entry_price <= 0:
            return Decimal("0")

        if state.is_long:
            pnl_pct = (current_price - state.entry_price) / state.entry_price
        else:
            pnl_pct = (state.entry_price - current_price) / state.entry_price

        return pnl_pct * Decimal("100")

    def _get_close_quantity(self, state: TimeStopState) -> Decimal:
        """Get quantity to close based on action."""
        action = self._config.action

        if action == TimeStopAction.CLOSE_FULL:
            return state.remaining_quantity
        elif action == TimeStopAction.CLOSE_PARTIAL:
            return state.remaining_quantity * self._config.partial_close_pct
        elif action == TimeStopAction.REDUCE_TO_SIZE:
            # This would need additional config for target size
            return state.remaining_quantity * Decimal("0.5")

        return state.remaining_quantity

    def _handle_trigger(
        self, symbol: str, state: TimeStopState, result: TimeStopResult
    ) -> None:
        """Handle time stop trigger."""
        state.mark_triggered(result.reason)
        self._total_triggers += 1
        self._triggers_by_type[self._config.stop_type] += 1

        logger.warning(f"Time stop triggered for {symbol}: {result.reason}")

        if self._on_trigger:
            self._on_trigger(symbol, result)

    def _handle_gradual_trigger(
        self, symbol: str, state: TimeStopState, result: TimeStopResult
    ) -> None:
        """Handle gradual exit trigger (partial close, don't mark as fully triggered)."""
        logger.info(f"Gradual exit for {symbol}: {result.reason}")

        if self._on_trigger:
            self._on_trigger(symbol, result)

    def _handle_warning(
        self, symbol: str, state: TimeStopState, result: TimeStopResult
    ) -> None:
        """Handle time stop warning."""
        state.warning_sent = True

        logger.info(f"Time stop warning for {symbol}: {result.reason}")

        if self._on_warning and result.minutes_until_trigger is not None:
            self._on_warning(symbol, result.reason, result.minutes_until_trigger)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def check_all(
        self,
        prices: Dict[str, Decimal],
        current_time: Optional[datetime] = None,
    ) -> Dict[str, TimeStopResult]:
        """
        Check all active positions.

        Args:
            prices: Dict of symbol -> current price
            current_time: Current time (defaults to now UTC)

        Returns:
            Dict of symbol -> TimeStopResult
        """
        results = {}

        for symbol in list(self._states.keys()):
            if symbol in prices:
                results[symbol] = self.check(symbol, prices[symbol], current_time)

        return results

    def get_positions_near_trigger(
        self, minutes_threshold: float = 5.0
    ) -> List[tuple]:
        """
        Get positions near time stop trigger.

        Args:
            minutes_threshold: Minutes threshold for "near"

        Returns:
            List of (symbol, minutes_until_trigger)
        """
        near_positions = []

        for symbol, state in self._states.items():
            if state.triggered:
                continue

            # Estimate minutes until trigger based on type
            if self._config.stop_type == TimeStopType.MAX_HOLDING:
                minutes_left = self._config.max_holding_minutes - state.holding_minutes
            elif self._config.stop_type == TimeStopType.NO_PROFIT:
                minutes_left = self._config.no_profit_minutes - state.minutes_since_profit
            else:
                minutes_left = float("inf")

            if minutes_left <= minutes_threshold:
                near_positions.append((symbol, minutes_left))

        return sorted(near_positions, key=lambda x: x[1])

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get time stop statistics."""
        return {
            "total_checks": self._total_checks,
            "total_triggers": self._total_triggers,
            "triggers_by_type": {t.value: c for t, c in self._triggers_by_type.items()},
            "active_positions": len(self.active_positions),
            "enabled": self._config.enabled,
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._total_checks = 0
        self._total_triggers = 0
        self._triggers_by_type = {t: 0 for t in TimeStopType}

    # =========================================================================
    # Configuration
    # =========================================================================

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated time stop config: {key} = {value}")

    def enable(self) -> None:
        """Enable time stop."""
        self._config.enabled = True
        logger.info("Time stop enabled")

    def disable(self) -> None:
        """Disable time stop."""
        self._config.enabled = False
        logger.warning("Time stop disabled")
