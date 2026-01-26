"""
Risk Engine.

Integrates all risk management modules into a unified engine.
Provides centralized risk monitoring, alert generation, and action execution.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, List, Optional, Protocol

from src.core import get_logger

from .capital_monitor import CapitalMonitor
from .circuit_breaker import CircuitBreaker
from .drawdown_calculator import DrawdownCalculator
from .emergency_stop import EmergencyConfig, EmergencyStop
from .models import (
    CapitalSnapshot,
    CircuitBreakerState,
    DailyPnL,
    DrawdownInfo,
    GlobalRiskStatus,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)

logger = get_logger(__name__)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_account(self, market=None):
        """Get account information."""
        ...

    async def get_positions(self, symbol=None):
        """Get positions."""
        ...


class BotCommanderProtocol(Protocol):
    """Protocol for bot commander interface."""

    async def pause_all(self) -> List[str]:
        """Pause all bots."""
        ...

    async def stop_all(self) -> List[str]:
        """Stop all bots."""
        ...

    async def broadcast(self, command: str, **kwargs) -> dict:
        """Broadcast command to all bots."""
        ...

    def get_running_bots(self) -> List[str]:
        """Get running bot IDs."""
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


class RiskEngine:
    """
    Unified risk management engine.

    Integrates all risk management components:
    - CapitalMonitor: tracks capital changes
    - DrawdownCalculator: tracks drawdown
    - CircuitBreaker: handles automatic protection
    - EmergencyStop: handles extreme situations

    Example:
        >>> engine = RiskEngine(config, exchange, commander, notifier)
        >>> await engine.start()
        >>> status = await engine.check()
        >>> print(f"Risk level: {status.level}")
    """

    def __init__(
        self,
        config: RiskConfig,
        exchange: Optional[ExchangeProtocol] = None,
        commander: Optional[BotCommanderProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        check_interval: int = 10,
        on_level_change: Optional[Callable[[RiskLevel, RiskLevel], None]] = None,
    ):
        """
        Initialize RiskEngine.

        Args:
            config: Risk configuration
            exchange: Exchange client for account data
            commander: Bot commander for control
            notifier: Notification manager for alerts
            check_interval: Interval between checks in seconds
            on_level_change: Callback when risk level changes (old, new)
        """
        self._config = config
        self._exchange = exchange
        self._commander = commander
        self._notifier = notifier
        self._check_interval = check_interval
        self._on_level_change = on_level_change

        # Initialize sub-modules
        self._capital_monitor = CapitalMonitor(config, exchange)
        self._drawdown_calc = DrawdownCalculator(config)
        self._circuit_breaker = CircuitBreaker(config, commander, notifier)

        # Emergency stop config
        emergency_config = EmergencyConfig(
            auto_trigger_loss_pct=config.danger_loss_pct + Decimal("0.10"),  # 10% more than danger
            auto_close_positions=True,
            max_circuit_triggers=3,
            initial_capital=config.total_capital,
        )
        self._emergency_stop = EmergencyStop(
            emergency_config,
            commander=commander,
            exchange=exchange,
            notifier=notifier,
        )

        # State
        self._running = False
        self._last_status: Optional[GlobalRiskStatus] = None
        self._last_check_time: Optional[datetime] = None

        # Tasks
        self._check_task: Optional[asyncio.Task] = None
        self._daily_reset_task: Optional[asyncio.Task] = None

        # Tracking
        self._consecutive_losses: int = 0
        self._last_daily_reset: Optional[date] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> RiskConfig:
        """Get risk configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    @property
    def last_status(self) -> Optional[GlobalRiskStatus]:
        """Get last risk status."""
        return self._last_status

    @property
    def capital_monitor(self) -> CapitalMonitor:
        """Get capital monitor."""
        return self._capital_monitor

    @property
    def drawdown_calculator(self) -> DrawdownCalculator:
        """Get drawdown calculator."""
        return self._drawdown_calc

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker."""
        return self._circuit_breaker

    @property
    def emergency_stop(self) -> EmergencyStop:
        """Get emergency stop."""
        return self._emergency_stop

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the risk engine."""
        if self._running:
            logger.warning("Risk engine already running")
            return

        self._running = True
        logger.info("Risk engine started")

        # Start check loop
        self._check_task = asyncio.create_task(self._check_loop())

        # Start daily reset loop
        self._daily_reset_task = asyncio.create_task(self._daily_reset_loop())

    async def stop(self) -> None:
        """Stop the risk engine."""
        if not self._running:
            return

        self._running = False
        logger.info("Risk engine stopping")

        # Cancel tasks
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

        if self._daily_reset_task:
            self._daily_reset_task.cancel()
            try:
                await self._daily_reset_task
            except asyncio.CancelledError:
                pass
            self._daily_reset_task = None

        logger.info("Risk engine stopped")

    # =========================================================================
    # Check Loop
    # =========================================================================

    async def _check_loop(self) -> None:
        """Risk check loop."""
        while self._running:
            try:
                await self.check()
            except Exception as e:
                logger.error(f"Risk check error: {e}")

            await asyncio.sleep(self._check_interval)

    async def _daily_reset_loop(self) -> None:
        """Daily reset loop - resets at midnight UTC."""
        while self._running:
            try:
                # Check if we need to reset - use UTC date for consistency
                today = datetime.now(timezone.utc).date()
                if self._last_daily_reset != today:
                    await self._perform_daily_reset()
                    self._last_daily_reset = today

            except Exception as e:
                logger.error(f"Daily reset error: {e}")

            # Sleep until next check (every minute)
            await asyncio.sleep(60)

    async def _perform_daily_reset(self) -> None:
        """Perform daily reset of statistics."""
        logger.info("Performing daily reset")

        # Reset capital monitor daily stats
        self._capital_monitor.reset_daily()

        # Reset circuit breaker daily count
        self._circuit_breaker._trigger_count_today = 0

        # Reset consecutive losses
        self._consecutive_losses = 0

    # =========================================================================
    # Core Check Method
    # =========================================================================

    async def check(self) -> GlobalRiskStatus:
        """
        Execute risk check.

        Returns:
            GlobalRiskStatus with current risk state
        """
        self._last_check_time = datetime.now(timezone.utc)

        # 1. Update capital snapshot
        if self._exchange:
            try:
                capital = await self._capital_monitor.update()
            except Exception as e:
                logger.error(f"Failed to update capital: {e}")
                capital = self._capital_monitor.get_current()
        else:
            capital = self._capital_monitor.get_current()

        # Create default capital if none
        if capital is None:
            capital = CapitalSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_capital=self._config.total_capital,
                available_balance=self._config.total_capital,
                position_value=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
            )

        # 2. Update drawdown calculation
        drawdown = self._drawdown_calc.update(capital.total_capital)

        # 3. Collect all alerts
        alerts: List[RiskAlert] = []
        alerts.extend(self._capital_monitor.check_alerts())
        alerts.extend(self._drawdown_calc.check_alerts())

        # Add consecutive loss alert if needed
        if self._consecutive_losses >= self._config.consecutive_loss_danger:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.DANGER,
                    metric="consecutive_losses",
                    current_value=Decimal(str(self._consecutive_losses)),
                    threshold=Decimal(str(self._config.consecutive_loss_danger)),
                    message=f"Consecutive losses: {self._consecutive_losses}",
                    action_taken=RiskAction.PAUSE_ALL_BOTS,
                )
            )
        elif self._consecutive_losses >= self._config.consecutive_loss_warning:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.WARNING,
                    metric="consecutive_losses",
                    current_value=Decimal(str(self._consecutive_losses)),
                    threshold=Decimal(str(self._config.consecutive_loss_warning)),
                    message=f"Consecutive losses: {self._consecutive_losses}",
                    action_taken=RiskAction.NOTIFY,
                )
            )

        # 4. Calculate risk level
        level = self._calculate_risk_level(alerts)

        # 5. Check circuit breaker
        if level.value >= RiskLevel.DANGER.value:
            triggered = await self._circuit_breaker.check_and_trigger(alerts)
            if triggered:
                level = RiskLevel.CIRCUIT_BREAK

        # 6. Build status
        status = GlobalRiskStatus(
            level=level,
            capital=capital,
            drawdown=drawdown,
            daily_pnl=self._capital_monitor.get_daily_pnl(),
            circuit_breaker=self._circuit_breaker.get_state(),
            active_alerts=alerts,
            last_updated=datetime.now(timezone.utc),
        )

        # 7. Check emergency stop
        emergency_reason = self._emergency_stop.check_auto_trigger(status)
        if emergency_reason:
            await self._emergency_stop.activate(emergency_reason)
            level = RiskLevel.CIRCUIT_BREAK
            status.level = level

        # 8. Execute risk actions
        await self._execute_risk_action(level, alerts)

        # 9. Check for level change and save status
        # Important: Save status BEFORE triggering callback so callback sees current state
        old_level = self._last_status.level if self._last_status else None
        level_changed = old_level is not None and old_level != level

        # Save status first
        self._last_status = status

        # Then trigger callback (callback will see updated state)
        if level_changed:
            logger.info(f"Risk level changed: {old_level} -> {level}")
            if self._on_level_change:
                try:
                    self._on_level_change(old_level, level)
                except Exception as e:
                    logger.error(f"Error in level change callback: {e}")

        return status

    def _calculate_risk_level(self, alerts: List[RiskAlert]) -> RiskLevel:
        """
        Calculate overall risk level from alerts.

        Args:
            alerts: List of risk alerts

        Returns:
            Highest risk level from alerts
        """
        if not alerts:
            return RiskLevel.NORMAL

        # Get maximum level
        max_level = max(alert.level.value for alert in alerts)

        for level in RiskLevel:
            if level.value == max_level:
                return level

        return RiskLevel.NORMAL

    async def _execute_risk_action(
        self, level: RiskLevel, alerts: List[RiskAlert]
    ) -> None:
        """
        Execute actions based on risk level.

        Args:
            level: Current risk level
            alerts: Active alerts
        """
        if level == RiskLevel.NORMAL:
            return

        elif level == RiskLevel.WARNING:
            await self._send_warning_notification(alerts)

        elif level == RiskLevel.RISK:
            # Pause new orders
            if self._commander:
                try:
                    await self._commander.broadcast("pause_new_orders")
                except Exception as e:
                    logger.error(f"Failed to pause new orders: {e}")
            await self._send_risk_notification(alerts)

        elif level == RiskLevel.DANGER:
            # Circuit breaker handles this
            pass

        elif level == RiskLevel.CIRCUIT_BREAK:
            # Emergency stop handles this
            pass

    # =========================================================================
    # Notification Methods
    # =========================================================================

    async def _send_warning_notification(self, alerts: List[RiskAlert]) -> None:
        """Send warning notification."""
        if not self._notifier:
            return

        messages = [f"- {a.message}" for a in alerts if a.level == RiskLevel.WARNING]
        if not messages:
            return

        try:
            await self._notifier.send(
                title="Risk Warning",
                message="\n".join(messages),
                level="warning",
            )
        except Exception as e:
            logger.error(f"Failed to send warning notification: {e}")

    async def _send_risk_notification(self, alerts: List[RiskAlert]) -> None:
        """Send risk notification."""
        if not self._notifier:
            return

        messages = [f"- {a.message}" for a in alerts if a.level.value >= RiskLevel.RISK.value]
        if not messages:
            return

        try:
            await self._notifier.send(
                title="Risk Alert - New Orders Paused",
                message="\n".join(messages),
                level="error",
            )
        except Exception as e:
            logger.error(f"Failed to send risk notification: {e}")

    # =========================================================================
    # Public Methods
    # =========================================================================

    def get_status(self) -> Optional[GlobalRiskStatus]:
        """
        Get current risk status.

        Returns:
            Last GlobalRiskStatus or None if no check yet
        """
        return self._last_status

    async def trigger_emergency(self, reason: str) -> None:
        """
        Manually trigger emergency stop.

        Args:
            reason: Reason for emergency stop
        """
        logger.critical(f"Manual emergency stop: {reason}")
        await self._emergency_stop.activate(reason)

    async def reset_circuit_breaker(self, force: bool = False) -> bool:
        """
        Reset circuit breaker.

        Args:
            force: Force reset even if cooldown not finished

        Returns:
            True if reset successful
        """
        try:
            await self._circuit_breaker.reset(force=force)
            logger.info("Circuit breaker reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset circuit breaker: {e}")
            return False

    def record_trade_result(self, is_win: bool) -> None:
        """
        Record trade result for consecutive loss tracking.

        Args:
            is_win: Whether the trade was profitable
        """
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Also record in capital monitor
        self._capital_monitor.record_trade(is_win)

    def update_capital(
        self,
        total_capital: Decimal,
        available_balance: Decimal,
        position_value: Decimal = Decimal("0"),
        unrealized_pnl: Decimal = Decimal("0"),
    ) -> CapitalSnapshot:
        """
        Manually update capital (for when not using exchange).

        Args:
            total_capital: Total capital
            available_balance: Available balance
            position_value: Position value
            unrealized_pnl: Unrealized P&L

        Returns:
            New CapitalSnapshot
        """
        snapshot = self._capital_monitor.update_from_values(
            total_capital=total_capital,
            available_balance=available_balance,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
        )

        # Update drawdown
        self._drawdown_calc.update(total_capital)

        return snapshot

    def get_statistics(self) -> dict:
        """
        Get comprehensive risk statistics.

        Returns:
            Dictionary with risk statistics
        """
        drawdown_stats = self._drawdown_calc.get_statistics()
        daily_pnl = self._capital_monitor.get_daily_pnl()
        change, change_pct = self._capital_monitor.get_capital_change()

        return {
            "current_level": self._last_status.level.name if self._last_status else "UNKNOWN",
            "total_capital_change": change,
            "total_capital_change_pct": change_pct,
            "daily_pnl": daily_pnl.pnl,
            "daily_pnl_pct": daily_pnl.pnl_pct,
            "current_drawdown_pct": drawdown_stats["current_drawdown_pct"],
            "max_drawdown_pct": drawdown_stats["max_drawdown_pct"],
            "consecutive_losses": self._consecutive_losses,
            "circuit_breaker_triggered": self._circuit_breaker.is_triggered,
            "emergency_stop_activated": self._emergency_stop.is_activated,
            "last_check_time": self._last_check_time,
        }
