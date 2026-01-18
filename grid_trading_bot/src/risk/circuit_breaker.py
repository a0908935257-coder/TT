"""
Circuit Breaker.

Provides automatic protection when risk thresholds are breached.
Triggers emergency stop, pauses bots, and manages cooldown period.
"""

from datetime import date, datetime, timedelta
from enum import Enum
from typing import Callable, List, Optional, Protocol

from src.core import get_logger

from .models import (
    CircuitBreakerState,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation (circuit not triggered)
    OPEN = "open"  # Circuit triggered (protection active)
    HALF_OPEN = "half_open"  # Testing recovery (optional state)


class CooldownNotFinishedError(Exception):
    """Raised when trying to reset before cooldown is finished."""

    pass


class BotCommanderProtocol(Protocol):
    """Protocol for bot commander interface."""

    async def pause_all(self) -> List[str]:
        """Pause all running bots."""
        ...

    async def resume_all(self) -> List[str]:
        """Resume all paused bots."""
        ...

    async def send_command(self, bot_id: str, command: str, **kwargs) -> bool:
        """Send command to a specific bot."""
        ...

    def get_running_bots(self) -> List[str]:
        """Get list of running bot IDs."""
        ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager interface."""

    async def send(
        self,
        title: str,
        message: str,
        level: str = "info",
        **kwargs,
    ) -> bool:
        """Send a notification."""
        ...


class CircuitBreaker:
    """
    Circuit breaker for risk management.

    Automatically triggers protection when risk thresholds are breached,
    pausing bots and optionally closing positions.

    Example:
        >>> breaker = CircuitBreaker(config, commander, notifier)
        >>> alerts = capital_monitor.check_alerts()
        >>> if await breaker.check_and_trigger(alerts):
        ...     print("Circuit breaker triggered!")
    """

    def __init__(
        self,
        config: RiskConfig,
        commander: Optional[BotCommanderProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        on_trigger: Optional[Callable[[str], None]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize CircuitBreaker.

        Args:
            config: Risk configuration with thresholds
            commander: Bot commander for pausing bots
            notifier: Notification manager for alerts
            on_trigger: Optional callback when triggered
            on_reset: Optional callback when reset
        """
        self._config = config
        self._commander = commander
        self._notifier = notifier
        self._on_trigger = on_trigger
        self._on_reset = on_reset

        # State tracking
        self._state = CircuitState.CLOSED
        self._triggered_at: Optional[datetime] = None
        self._trigger_reason: Optional[str] = None
        self._cooldown_until: Optional[datetime] = None

        # Daily tracking
        self._trigger_count_today: int = 0
        self._last_reset_date: Optional[date] = None

        # Protection tracking
        self._paused_bots: List[str] = []
        self._cancelled_orders: int = 0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> RiskConfig:
        """Get risk configuration."""
        return self._config

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_triggered(self) -> bool:
        """Check if circuit breaker is triggered."""
        return self._state == CircuitState.OPEN

    @property
    def triggered_at(self) -> Optional[datetime]:
        """Get time when triggered."""
        return self._triggered_at

    @property
    def trigger_reason(self) -> Optional[str]:
        """Get trigger reason."""
        return self._trigger_reason

    @property
    def trigger_count_today(self) -> int:
        """Get number of triggers today."""
        self._reset_daily_count_if_needed()
        return self._trigger_count_today

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def check_and_trigger(self, alerts: List[RiskAlert]) -> bool:
        """
        Check alerts and trigger circuit breaker if needed.

        Args:
            alerts: List of risk alerts to check

        Returns:
            True if circuit breaker was triggered
        """
        # Already triggered - skip
        if self._state == CircuitState.OPEN:
            return False

        # Check for DANGER level alerts
        danger_alerts = [a for a in alerts if a.level == RiskLevel.DANGER]

        if not danger_alerts:
            return False

        # Find most severe reason
        reason = danger_alerts[0].message

        # Trigger circuit breaker
        await self.trigger(reason)
        return True

    async def trigger(self, reason: str) -> None:
        """
        Trigger the circuit breaker.

        Args:
            reason: Reason for triggering
        """
        if self._state == CircuitState.OPEN:
            logger.warning("Circuit breaker already triggered")
            return

        now = datetime.now()

        # Update state
        self._state = CircuitState.OPEN
        self._triggered_at = now
        self._trigger_reason = reason
        self._cooldown_until = now + timedelta(
            seconds=self._config.circuit_breaker_cooldown
        )

        # Update daily count
        self._reset_daily_count_if_needed()
        self._trigger_count_today += 1

        logger.critical(f"Circuit breaker triggered: {reason}")

        # Execute protection
        await self._execute_protection()

        # Send notification
        await self._send_trigger_notification(reason)

        # Call callback if set
        if self._on_trigger:
            try:
                self._on_trigger(reason)
            except Exception as e:
                logger.error(f"Error in on_trigger callback: {e}")

    async def reset(self, force: bool = False) -> None:
        """
        Reset the circuit breaker.

        Args:
            force: Force reset even if cooldown not finished

        Raises:
            CooldownNotFinishedError: If cooldown not finished and force=False
        """
        if self._state != CircuitState.OPEN:
            logger.info("Circuit breaker not triggered, nothing to reset")
            return

        now = datetime.now()

        # Check cooldown unless forced
        if not force and self._cooldown_until and now < self._cooldown_until:
            remaining = self._cooldown_until - now
            raise CooldownNotFinishedError(
                f"Cooldown not finished, {remaining.total_seconds():.0f}s remaining"
            )

        # Reset state
        self._state = CircuitState.CLOSED
        self._triggered_at = None
        self._trigger_reason = None

        logger.info("Circuit breaker reset")

        # Send notification
        await self._send_reset_notification()

        # Auto-resume bots if enabled
        if self._config.auto_resume_enabled and self._commander:
            try:
                resumed = await self._commander.resume_all()
                logger.info(f"Auto-resumed {len(resumed)} bots")
            except Exception as e:
                logger.error(f"Failed to auto-resume bots: {e}")

        # Call callback if set
        if self._on_reset:
            try:
                self._on_reset()
            except Exception as e:
                logger.error(f"Error in on_reset callback: {e}")

    def get_state(self) -> CircuitBreakerState:
        """
        Get comprehensive circuit breaker state.

        Returns:
            CircuitBreakerState with all state information
        """
        self._reset_daily_count_if_needed()

        return CircuitBreakerState(
            is_triggered=(self._state == CircuitState.OPEN),
            triggered_at=self._triggered_at,
            trigger_reason=self._trigger_reason or "",
            cooldown_until=self._cooldown_until,
            trigger_count_today=self._trigger_count_today,
        )

    def get_cooldown_remaining(self) -> timedelta:
        """
        Get remaining cooldown time.

        Returns:
            Remaining cooldown duration (zero if not in cooldown)
        """
        if self._state != CircuitState.OPEN or not self._cooldown_until:
            return timedelta(0)

        now = datetime.now()
        if now >= self._cooldown_until:
            return timedelta(0)

        return self._cooldown_until - now

    def is_cooldown_finished(self) -> bool:
        """
        Check if cooldown period is finished.

        Returns:
            True if cooldown is finished or not in triggered state
        """
        if self._state != CircuitState.OPEN:
            return True

        if not self._cooldown_until:
            return True

        return datetime.now() >= self._cooldown_until

    # =========================================================================
    # Protection Methods
    # =========================================================================

    async def _execute_protection(self) -> None:
        """Execute protection actions when triggered."""
        self._paused_bots = []
        self._cancelled_orders = 0

        if not self._commander:
            logger.warning("No commander configured, skipping protection actions")
            return

        try:
            # 1. Pause all bots
            self._paused_bots = await self._commander.pause_all()
            logger.info(f"Paused {len(self._paused_bots)} bots")

            # 2. Cancel all orders for each bot
            for bot_id in self._commander.get_running_bots():
                try:
                    await self._commander.send_command(bot_id, "cancel_all_orders")
                    self._cancelled_orders += 1
                except Exception as e:
                    logger.error(f"Failed to cancel orders for bot {bot_id}: {e}")

        except Exception as e:
            logger.error(f"Error executing protection: {e}")

    # =========================================================================
    # Notification Methods
    # =========================================================================

    async def _send_trigger_notification(self, reason: str) -> None:
        """Send notification when triggered."""
        if not self._notifier:
            logger.debug("No notifier configured, skipping notification")
            return

        cooldown_str = self._format_duration(
            timedelta(seconds=self._config.circuit_breaker_cooldown)
        )
        recovery_time = self._cooldown_until.strftime("%Y-%m-%d %H:%M UTC") if self._cooldown_until else "Unknown"

        message = (
            f"Reason: {reason}\n\n"
            f"Actions taken:\n"
            f"- Paused {len(self._paused_bots)} bot(s)\n"
            f"- Cancelled orders for {self._cancelled_orders} bot(s)\n\n"
            f"Cooldown: {cooldown_str}\n"
            f"Expected recovery: {recovery_time}\n\n"
            f"Please check market conditions before manually resetting."
        )

        try:
            await self._notifier.send(
                title="EMERGENCY: Circuit Breaker Triggered",
                message=message,
                level="critical",
            )
        except Exception as e:
            logger.error(f"Failed to send trigger notification: {e}")

    async def _send_reset_notification(self) -> None:
        """Send notification when reset."""
        if not self._notifier:
            return

        message = (
            "Circuit breaker has been reset.\n"
            "System is back to normal operation.\n"
        )

        if self._config.auto_resume_enabled:
            message += "Bots will be automatically resumed."
        else:
            message += "You may manually restart bots when ready."

        try:
            await self._notifier.send(
                title="Circuit Breaker Reset",
                message=message,
                level="info",
            )
        except Exception as e:
            logger.error(f"Failed to send reset notification: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _reset_daily_count_if_needed(self) -> None:
        """Reset daily trigger count if it's a new day."""
        today = date.today()

        if self._last_reset_date != today:
            self._trigger_count_today = 0
            self._last_reset_date = today

    @staticmethod
    def _format_duration(duration: timedelta) -> str:
        """Format duration as human-readable string."""
        total_seconds = int(duration.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes} minute(s)"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            if minutes:
                return f"{hours} hour(s) {minutes} minute(s)"
            return f"{hours} hour(s)"

    # =========================================================================
    # Manual Control
    # =========================================================================

    def force_close(self) -> None:
        """
        Force close the circuit (back to normal) without cooldown check.

        Use with caution - bypasses safety cooldown.
        """
        logger.warning("Force closing circuit breaker - bypassing cooldown")

        self._state = CircuitState.CLOSED
        self._triggered_at = None
        self._trigger_reason = None
        self._cooldown_until = None

    def extend_cooldown(self, additional_seconds: int) -> None:
        """
        Extend the cooldown period.

        Args:
            additional_seconds: Additional seconds to add to cooldown
        """
        if self._state != CircuitState.OPEN or not self._cooldown_until:
            logger.warning("Cannot extend cooldown - circuit not triggered")
            return

        self._cooldown_until += timedelta(seconds=additional_seconds)
        logger.info(
            f"Cooldown extended by {additional_seconds}s, "
            f"new end time: {self._cooldown_until}"
        )
