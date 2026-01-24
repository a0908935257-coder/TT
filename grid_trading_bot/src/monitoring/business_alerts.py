"""
Business alert rules module.

Provides pre-configured alert rules for trading business metrics
including order success rate, P&L limits, and risk thresholds.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from src.core import get_logger
from src.monitoring.alerts import (
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from src.monitoring.system_metrics import SystemThresholds

logger = get_logger(__name__)


@dataclass
class OrderStats:
    """Order statistics for a time window."""

    total_orders: int = 0
    successful_orders: int = 0
    rejected_orders: int = 0
    cancelled_orders: int = 0
    failed_orders: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate order success rate."""
        if self.total_orders == 0:
            return 1.0
        return self.successful_orders / self.total_orders

    @property
    def rejection_rate(self) -> float:
        """Calculate order rejection rate."""
        if self.total_orders == 0:
            return 0.0
        return self.rejected_orders / self.total_orders

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average order latency."""
        if self.successful_orders == 0:
            return 0.0
        return self.total_latency_ms / self.successful_orders


@dataclass
class PnLStats:
    """P&L statistics."""

    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    peak_value: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    daily_limit: Decimal = Decimal("0")

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def drawdown(self) -> Decimal:
        """Current drawdown from peak."""
        if self.peak_value == 0:
            return Decimal("0")
        return self.peak_value - self.current_value

    @property
    def drawdown_percent(self) -> float:
        """Drawdown as percentage."""
        if self.peak_value == 0:
            return 0.0
        return float(self.drawdown / self.peak_value)

    @property
    def daily_loss_percent(self) -> float:
        """Daily loss as percentage of limit."""
        if self.daily_limit == 0:
            return 0.0
        if self.daily_pnl >= 0:
            return 0.0
        return float(abs(self.daily_pnl) / self.daily_limit)


class BusinessMetricsTracker:
    """
    Tracks business metrics for alert evaluation.

    Maintains rolling windows of order and P&L statistics.
    """

    def __init__(
        self,
        order_window_seconds: int = 60,
        max_history: int = 1440,  # 24 hours at 1 minute intervals
    ):
        """
        Initialize the tracker.

        Args:
            order_window_seconds: Window for order statistics
            max_history: Maximum history entries to keep
        """
        self._order_window = order_window_seconds
        self._order_history: Deque[Tuple[datetime, OrderStats]] = deque(
            maxlen=max_history
        )
        self._pnl_history: Deque[Tuple[datetime, PnLStats]] = deque(maxlen=max_history)

        # Current window stats
        self._current_orders = OrderStats()
        self._current_pnl = PnLStats()
        self._window_start = datetime.now(timezone.utc)

        # Risk control tracking
        self._risk_blocks: Deque[datetime] = deque(maxlen=100)

    def record_order(
        self,
        success: bool,
        rejected: bool = False,
        cancelled: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        """Record an order outcome."""
        self._maybe_rotate_window()

        self._current_orders.total_orders += 1
        if success:
            self._current_orders.successful_orders += 1
            self._current_orders.total_latency_ms += latency_ms
        elif rejected:
            self._current_orders.rejected_orders += 1
        elif cancelled:
            self._current_orders.cancelled_orders += 1
        else:
            self._current_orders.failed_orders += 1

    def update_pnl(
        self,
        realized: Decimal,
        unrealized: Decimal,
        current_value: Decimal,
        daily_pnl: Decimal,
        daily_limit: Decimal,
    ) -> None:
        """Update P&L statistics."""
        self._current_pnl.realized_pnl = realized
        self._current_pnl.unrealized_pnl = unrealized
        self._current_pnl.current_value = current_value
        self._current_pnl.daily_pnl = daily_pnl
        self._current_pnl.daily_limit = daily_limit

        # Update peak
        if current_value > self._current_pnl.peak_value:
            self._current_pnl.peak_value = current_value

    def record_risk_block(self) -> None:
        """Record a risk control block event."""
        self._risk_blocks.append(datetime.now(timezone.utc))

    def get_risk_blocks_count(self, seconds: int = 300) -> int:
        """Get number of risk blocks in time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        return sum(1 for t in self._risk_blocks if t >= cutoff)

    def _maybe_rotate_window(self) -> None:
        """Rotate to new window if needed."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self._window_start).total_seconds()

        if elapsed >= self._order_window:
            # Save current window
            self._order_history.append((self._window_start, self._current_orders))
            self._pnl_history.append((self._window_start, self._current_pnl))

            # Start new window
            self._current_orders = OrderStats()
            self._window_start = now

    def get_order_stats(self) -> OrderStats:
        """Get current order statistics."""
        return self._current_orders

    def get_pnl_stats(self) -> PnLStats:
        """Get current P&L statistics."""
        return self._current_pnl

    def get_hourly_order_stats(self) -> OrderStats:
        """Get aggregated order stats for last hour."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        aggregated = OrderStats()

        for timestamp, stats in self._order_history:
            if timestamp >= cutoff:
                aggregated.total_orders += stats.total_orders
                aggregated.successful_orders += stats.successful_orders
                aggregated.rejected_orders += stats.rejected_orders
                aggregated.cancelled_orders += stats.cancelled_orders
                aggregated.failed_orders += stats.failed_orders
                aggregated.total_latency_ms += stats.total_latency_ms

        # Include current window
        aggregated.total_orders += self._current_orders.total_orders
        aggregated.successful_orders += self._current_orders.successful_orders
        aggregated.rejected_orders += self._current_orders.rejected_orders
        aggregated.cancelled_orders += self._current_orders.cancelled_orders
        aggregated.failed_orders += self._current_orders.failed_orders
        aggregated.total_latency_ms += self._current_orders.total_latency_ms

        return aggregated


class BusinessAlertRules:
    """
    Pre-configured business alert rules.

    Creates and registers standard business monitoring rules.
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        metrics_tracker: BusinessMetricsTracker,
        thresholds: Optional[SystemThresholds] = None,
    ):
        """
        Initialize business alert rules.

        Args:
            alert_manager: Alert manager to register rules with
            metrics_tracker: Business metrics tracker
            thresholds: System thresholds
        """
        self._manager = alert_manager
        self._tracker = metrics_tracker
        self._thresholds = thresholds or SystemThresholds()
        self._rules_registered = False

    def register_all_rules(self) -> None:
        """Register all business alert rules."""
        if self._rules_registered:
            return

        self._register_order_rules()
        self._register_pnl_rules()
        self._register_risk_rules()
        self._rules_registered = True

        logger.info("Business alert rules registered")

    def _register_order_rules(self) -> None:
        """Register order-related alert rules."""
        # Order success rate warning
        self._manager.register_rule(
            AlertRule(
                name="order_success_rate_low",
                condition=lambda: (
                    self._tracker.get_order_stats().success_rate
                    < self._thresholds.order_success_rate_warning
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD],
                cooldown_seconds=300,
                description="Order success rate below threshold",
                labels={"category": "orders", "metric": "success_rate"},
            )
        )

        # Order rejection rate warning
        self._manager.register_rule(
            AlertRule(
                name="order_rejection_rate_high",
                condition=lambda: (
                    self._tracker.get_order_stats().rejection_rate
                    > self._thresholds.order_rejection_rate_warning
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD],
                cooldown_seconds=300,
                description="Order rejection rate above threshold",
                labels={"category": "orders", "metric": "rejection_rate"},
            )
        )

        # Order latency warning
        self._manager.register_rule(
            AlertRule(
                name="order_latency_warning",
                condition=lambda: (
                    self._tracker.get_order_stats().avg_latency_ms
                    > self._thresholds.order_latency_warning_ms
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_seconds=120,
                description="Order latency above warning threshold",
                labels={"category": "latency", "metric": "order_latency"},
            )
        )

        # Order latency critical
        self._manager.register_rule(
            AlertRule(
                name="order_latency_critical",
                condition=lambda: (
                    self._tracker.get_order_stats().avg_latency_ms
                    > self._thresholds.order_latency_critical_ms
                ),
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD, AlertChannel.EMAIL],
                cooldown_seconds=300,
                escalation_after_minutes=15,
                description="Order latency critical",
                labels={"category": "latency", "metric": "order_latency"},
            )
        )

    def _register_pnl_rules(self) -> None:
        """Register P&L-related alert rules."""
        # Daily loss warning
        self._manager.register_rule(
            AlertRule(
                name="daily_loss_warning",
                condition=lambda: (
                    self._tracker.get_pnl_stats().daily_loss_percent
                    >= self._thresholds.daily_loss_warning_percent
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD, AlertChannel.EMAIL],
                cooldown_seconds=1800,  # 30 minutes
                description="Daily loss approaching limit",
                labels={"category": "pnl", "metric": "daily_loss"},
            )
        )

        # Daily loss critical (100% of limit)
        self._manager.register_rule(
            AlertRule(
                name="daily_loss_critical",
                condition=lambda: (
                    self._tracker.get_pnl_stats().daily_loss_percent >= 1.0
                ),
                severity=AlertSeverity.CRITICAL,
                channels=[
                    AlertChannel.LOG,
                    AlertChannel.DISCORD,
                    AlertChannel.EMAIL,
                    AlertChannel.SMS,
                    AlertChannel.PAGERDUTY,
                ],
                cooldown_seconds=3600,
                escalation_after_minutes=5,
                description="Daily loss limit reached",
                labels={"category": "pnl", "metric": "daily_loss"},
            )
        )

        # Drawdown warning
        self._manager.register_rule(
            AlertRule(
                name="drawdown_warning",
                condition=lambda: (
                    self._tracker.get_pnl_stats().drawdown_percent
                    >= self._thresholds.max_drawdown_warning_percent
                ),
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD],
                cooldown_seconds=600,
                description="Drawdown above warning threshold",
                labels={"category": "pnl", "metric": "drawdown"},
            )
        )

        # Drawdown critical
        self._manager.register_rule(
            AlertRule(
                name="drawdown_critical",
                condition=lambda: (
                    self._tracker.get_pnl_stats().drawdown_percent
                    >= self._thresholds.max_drawdown_critical_percent
                ),
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD, AlertChannel.EMAIL],
                cooldown_seconds=1800,
                escalation_after_minutes=30,
                description="Drawdown critical",
                labels={"category": "pnl", "metric": "drawdown"},
            )
        )

    def _register_risk_rules(self) -> None:
        """Register risk control alert rules."""
        # Multiple risk blocks
        self._manager.register_rule(
            AlertRule(
                name="risk_blocks_frequent",
                condition=lambda: self._tracker.get_risk_blocks_count(300) >= 3,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD],
                cooldown_seconds=600,
                description="Multiple risk control blocks in short time",
                labels={"category": "risk", "metric": "blocks"},
            )
        )

        # Many risk blocks - critical
        self._manager.register_rule(
            AlertRule(
                name="risk_blocks_critical",
                condition=lambda: self._tracker.get_risk_blocks_count(300) >= 10,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.DISCORD, AlertChannel.EMAIL],
                cooldown_seconds=1800,
                description="Excessive risk control blocks",
                labels={"category": "risk", "metric": "blocks"},
            )
        )

    async def evaluate_all(self) -> List[str]:
        """
        Evaluate all rules and fire alerts as needed.

        Returns:
            List of fired alert IDs
        """
        fired_alerts = []

        # Get current stats
        order_stats = self._tracker.get_order_stats()
        pnl_stats = self._tracker.get_pnl_stats()

        # Order success rate
        if order_stats.total_orders > 0:
            if order_stats.success_rate < self._thresholds.order_success_rate_warning:
                alert = await self._manager.fire(
                    "order_success_rate_low",
                    f"Order Success Rate Low: {order_stats.success_rate:.1%}",
                    f"Success rate {order_stats.success_rate:.1%} is below "
                    f"threshold {self._thresholds.order_success_rate_warning:.0%}. "
                    f"Total orders: {order_stats.total_orders}, "
                    f"Successful: {order_stats.successful_orders}",
                    source="business_monitor",
                )
                if alert:
                    fired_alerts.append(alert.alert_id)

            if order_stats.rejection_rate > self._thresholds.order_rejection_rate_warning:
                alert = await self._manager.fire(
                    "order_rejection_rate_high",
                    f"Order Rejection Rate High: {order_stats.rejection_rate:.1%}",
                    f"Rejection rate {order_stats.rejection_rate:.1%} exceeds "
                    f"threshold {self._thresholds.order_rejection_rate_warning:.0%}. "
                    f"Rejected: {order_stats.rejected_orders}",
                    source="business_monitor",
                )
                if alert:
                    fired_alerts.append(alert.alert_id)

        # P&L alerts
        if pnl_stats.daily_loss_percent >= self._thresholds.daily_loss_warning_percent:
            severity = "warning"
            rule_name = "daily_loss_warning"
            if pnl_stats.daily_loss_percent >= 1.0:
                severity = "critical"
                rule_name = "daily_loss_critical"

            alert = await self._manager.fire(
                rule_name,
                f"Daily Loss {severity.title()}: {pnl_stats.daily_loss_percent:.0%} of limit",
                f"Daily P&L: {pnl_stats.daily_pnl}, "
                f"Limit: {pnl_stats.daily_limit}, "
                f"Usage: {pnl_stats.daily_loss_percent:.1%}",
                source="business_monitor",
            )
            if alert:
                fired_alerts.append(alert.alert_id)

        # Drawdown
        if pnl_stats.drawdown_percent >= self._thresholds.max_drawdown_warning_percent:
            rule_name = "drawdown_warning"
            if pnl_stats.drawdown_percent >= self._thresholds.max_drawdown_critical_percent:
                rule_name = "drawdown_critical"

            alert = await self._manager.fire(
                rule_name,
                f"Drawdown Alert: {pnl_stats.drawdown_percent:.1%}",
                f"Current drawdown: {pnl_stats.drawdown_percent:.2%}, "
                f"Peak: {pnl_stats.peak_value}, "
                f"Current: {pnl_stats.current_value}",
                source="business_monitor",
            )
            if alert:
                fired_alerts.append(alert.alert_id)

        # Risk blocks
        risk_blocks = self._tracker.get_risk_blocks_count(300)
        if risk_blocks >= 3:
            rule_name = "risk_blocks_frequent"
            if risk_blocks >= 10:
                rule_name = "risk_blocks_critical"

            alert = await self._manager.fire(
                rule_name,
                f"Risk Control Blocks: {risk_blocks} in 5 minutes",
                f"{risk_blocks} risk control blocks in the last 5 minutes. "
                "Review trading parameters and market conditions.",
                source="business_monitor",
            )
            if alert:
                fired_alerts.append(alert.alert_id)

        return fired_alerts


class BusinessAlertMonitor:
    """
    Continuous business metrics monitoring.

    Periodically evaluates business alert rules.
    """

    def __init__(
        self,
        rules: BusinessAlertRules,
        interval_seconds: float = 60.0,
    ):
        """
        Initialize the monitor.

        Args:
            rules: Business alert rules
            interval_seconds: Evaluation interval
        """
        self._rules = rules
        self._interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return

        self._rules.register_all_rules()
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Business alert monitor started (interval: {self._interval}s)")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Business alert monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                fired = await self._rules.evaluate_all()
                if fired:
                    logger.info(f"Fired {len(fired)} business alerts")

                await asyncio.sleep(self._interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in business monitor: {e}")
                await asyncio.sleep(10.0)
