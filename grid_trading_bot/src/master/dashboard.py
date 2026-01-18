"""
Dashboard module for unified monitoring.

Provides aggregated views of all bot metrics, alerts, and system status.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from src.core import get_logger
from src.master.aggregator import BotMetrics, MetricsAggregator
from src.master.models import BotState

if TYPE_CHECKING:
    from src.master.health import HealthCheckResult, HealthChecker
    from src.master.registry import BotRegistry

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification model."""

    bot_id: str
    level: AlertLevel
    message: str
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "bot_id": self.bot_id,
            "level": self.level.value,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }

    def acknowledge(self, by: Optional[str] = None) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = by


@dataclass
class DashboardSummary:
    """Summary statistics for all bots."""

    total_bots: int = 0
    running_bots: int = 0
    paused_bots: int = 0
    stopped_bots: int = 0
    error_bots: int = 0

    total_investment: Decimal = Decimal("0")
    total_value: Decimal = Decimal("0")
    total_profit: Decimal = Decimal("0")
    total_profit_rate: Decimal = Decimal("0")

    today_profit: Decimal = Decimal("0")
    today_trades: int = 0
    total_trades: int = 0

    total_pending_orders: int = 0
    healthy_bots: int = 0
    unhealthy_bots: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_bots": self.total_bots,
            "running_bots": self.running_bots,
            "paused_bots": self.paused_bots,
            "stopped_bots": self.stopped_bots,
            "error_bots": self.error_bots,
            "total_investment": str(self.total_investment),
            "total_value": str(self.total_value),
            "total_profit": str(self.total_profit),
            "total_profit_rate": str(self.total_profit_rate),
            "today_profit": str(self.today_profit),
            "today_trades": self.today_trades,
            "total_trades": self.total_trades,
            "total_pending_orders": self.total_pending_orders,
            "healthy_bots": self.healthy_bots,
            "unhealthy_bots": self.unhealthy_bots,
        }


@dataclass
class DashboardData:
    """Complete dashboard data model."""

    summary: DashboardSummary
    bots: List[BotMetrics]
    alerts: List[Alert]
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary.to_dict(),
            "bots": [b.to_dict() for b in self.bots],
            "alerts": [a.to_dict() for a in self.alerts],
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class BotDetail:
    """Detailed information for a single bot."""

    bot_info: Dict[str, Any]
    metrics: Optional[BotMetrics]
    health: Optional["HealthCheckResult"]
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    grid_status: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_info": self.bot_info,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "health": self.health.to_dict() if self.health else None,
            "recent_trades": self.recent_trades,
            "recent_events": self.recent_events,
            "grid_status": self.grid_status,
        }


class Dashboard:
    """
    Unified monitoring dashboard.

    Provides aggregated views of all bot metrics, alerts, and system status.
    """

    def __init__(
        self,
        registry: "BotRegistry",
        aggregator: MetricsAggregator,
        health_checker: Optional["HealthChecker"] = None,
    ):
        """
        Initialize the dashboard.

        Args:
            registry: Bot registry instance
            aggregator: Metrics aggregator instance
            health_checker: Optional health checker instance
        """
        self._registry = registry
        self._aggregator = aggregator
        self._health_checker = health_checker
        self._alerts: List[Alert] = []
        self._max_alerts: int = 100
        self._snapshot_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._db = None  # Optional database for snapshots

    @property
    def alert_count(self) -> int:
        """Get total alert count."""
        return len(self._alerts)

    @property
    def unacknowledged_alert_count(self) -> int:
        """Get unacknowledged alert count."""
        return len([a for a in self._alerts if not a.acknowledged])

    def get_data(self) -> DashboardData:
        """
        Get complete dashboard data.

        Returns:
            DashboardData with summary, bots, and alerts
        """
        # Refresh metrics
        self._aggregator.refresh()

        # Collect all bot metrics
        bots = self._aggregator.collect_all()

        # Calculate summary
        summary = self._calculate_summary(bots)

        # Get unacknowledged alerts
        alerts = [a for a in self._alerts if not a.acknowledged]

        return DashboardData(
            summary=summary,
            bots=bots,
            alerts=alerts,
            updated_at=datetime.now(timezone.utc),
        )

    def get_summary(self) -> DashboardSummary:
        """
        Get dashboard summary only.

        Returns:
            DashboardSummary with aggregated stats
        """
        bots = self._aggregator.collect_all()
        return self._calculate_summary(bots)

    def _calculate_summary(self, bots: List[BotMetrics]) -> DashboardSummary:
        """
        Calculate summary from bot metrics.

        Args:
            bots: List of bot metrics

        Returns:
            DashboardSummary
        """
        if not bots:
            return DashboardSummary()

        total_investment = sum(b.total_investment for b in bots)
        total_value = sum(b.current_value for b in bots)
        total_profit = sum(b.total_profit for b in bots)

        # Calculate profit rate
        total_profit_rate = Decimal("0")
        if total_investment > 0:
            total_profit_rate = (total_profit / total_investment) * Decimal("100")

        # Count by state
        running_bots = len([b for b in bots if b.state == BotState.RUNNING])
        paused_bots = len([b for b in bots if b.state == BotState.PAUSED])
        stopped_bots = len([b for b in bots if b.state == BotState.STOPPED])
        error_bots = len([b for b in bots if b.state == BotState.ERROR])

        # Get health counts if checker available
        healthy_bots = running_bots  # Default assumption
        unhealthy_bots = error_bots

        return DashboardSummary(
            total_bots=len(bots),
            running_bots=running_bots,
            paused_bots=paused_bots,
            stopped_bots=stopped_bots,
            error_bots=error_bots,
            total_investment=total_investment,
            total_value=total_value,
            total_profit=total_profit,
            total_profit_rate=total_profit_rate,
            today_profit=sum(b.today_profit for b in bots),
            today_trades=sum(b.today_trades for b in bots),
            total_trades=sum(b.total_trades for b in bots),
            total_pending_orders=sum(b.pending_orders for b in bots),
            healthy_bots=healthy_bots,
            unhealthy_bots=unhealthy_bots,
        )

    async def get_bot_detail(self, bot_id: str) -> Optional[BotDetail]:
        """
        Get detailed information for a single bot.

        Args:
            bot_id: The bot ID

        Returns:
            BotDetail or None if not found
        """
        bot_info = self._registry.get(bot_id)
        if not bot_info:
            return None

        # Get metrics
        metrics = self._aggregator.collect(bot_id)

        # Get health if checker available
        health = None
        if self._health_checker:
            health = await self._health_checker.check(bot_id)

        # Get recent events from registry
        events = self._registry.get_events(bot_id=bot_id, limit=10)
        recent_events = [e.to_dict() for e in events]

        # Get grid-specific status if available
        grid_status = None
        instance = self._registry.get_bot_instance(bot_id)
        if instance and hasattr(instance, "get_status"):
            try:
                status = instance.get_status()
                grid_status = {
                    "upper_price": str(status.get("upper_price", "")),
                    "lower_price": str(status.get("lower_price", "")),
                    "grid_version": status.get("grid_version", 1),
                    "grid_count": status.get("grid_count", 0),
                }
            except Exception as e:
                logger.warning(f"Error getting grid status: {e}")

        return BotDetail(
            bot_info=bot_info.to_dict(),
            metrics=metrics,
            health=health,
            recent_trades=[],  # Would need trade history from bot
            recent_events=recent_events,
            grid_status=grid_status,
        )

    def add_alert(self, alert: Alert) -> None:
        """
        Add a new alert.

        Args:
            alert: The alert to add
        """
        self._alerts.insert(0, alert)

        # Trim old alerts
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[: self._max_alerts]

        logger.info(f"Alert added: [{alert.level.value}] {alert.bot_id}: {alert.message}")

    def create_alert(
        self,
        bot_id: str,
        level: AlertLevel,
        message: str,
    ) -> Alert:
        """
        Create and add a new alert.

        Args:
            bot_id: The bot ID
            level: Alert level
            message: Alert message

        Returns:
            The created Alert
        """
        alert = Alert(bot_id=bot_id, level=level, message=message)
        self.add_alert(alert)
        return alert

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        bot_id: Optional[str] = None,
        unacknowledged_only: bool = False,
    ) -> List[Alert]:
        """
        Get alerts with optional filtering.

        Args:
            level: Filter by level
            bot_id: Filter by bot ID
            unacknowledged_only: Only return unacknowledged alerts

        Returns:
            Filtered list of alerts
        """
        alerts = self._alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        if bot_id:
            alerts = [a for a in alerts if a.bot_id == bot_id]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str, by: Optional[str] = None) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: The alert ID
            by: Who acknowledged (optional)

        Returns:
            True if acknowledged, False if not found
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge(by)
                logger.info(f"Alert {alert_id} acknowledged by {by or 'system'}")
                return True
        return False

    def acknowledge_all(self, by: Optional[str] = None) -> int:
        """
        Acknowledge all unacknowledged alerts.

        Args:
            by: Who acknowledged (optional)

        Returns:
            Number of alerts acknowledged
        """
        count = 0
        for alert in self._alerts:
            if not alert.acknowledged:
                alert.acknowledge(by)
                count += 1

        if count > 0:
            logger.info(f"Acknowledged {count} alerts by {by or 'system'}")

        return count

    def clear_alerts(self, acknowledged_only: bool = True) -> int:
        """
        Clear alerts.

        Args:
            acknowledged_only: Only clear acknowledged alerts

        Returns:
            Number of alerts cleared
        """
        if acknowledged_only:
            original_count = len(self._alerts)
            self._alerts = [a for a in self._alerts if not a.acknowledged]
            return original_count - len(self._alerts)
        else:
            count = len(self._alerts)
            self._alerts = []
            return count

    def get_rankings(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get bot rankings by various metrics.

        Args:
            limit: Maximum number of entries per ranking

        Returns:
            Dictionary with rankings by different metrics
        """
        bots = list(self._aggregator._cache.values())

        if not bots:
            return {
                "by_profit": [],
                "by_profit_rate": [],
                "by_trades": [],
            }

        # Rank by profit
        by_profit = sorted(bots, key=lambda b: b.total_profit, reverse=True)[:limit]
        profit_ranking = [
            {"bot_id": b.bot_id, "profit": str(b.total_profit), "rank": i + 1}
            for i, b in enumerate(by_profit)
        ]

        # Rank by profit rate
        by_rate = sorted(bots, key=lambda b: b.profit_rate, reverse=True)[:limit]
        rate_ranking = [
            {"bot_id": b.bot_id, "rate": str(b.profit_rate), "rank": i + 1}
            for i, b in enumerate(by_rate)
        ]

        # Rank by trades
        by_trades = sorted(bots, key=lambda b: b.total_trades, reverse=True)[:limit]
        trades_ranking = [
            {"bot_id": b.bot_id, "trades": b.total_trades, "rank": i + 1}
            for i, b in enumerate(by_trades)
        ]

        return {
            "by_profit": profit_ranking,
            "by_profit_rate": rate_ranking,
            "by_trades": trades_ranking,
        }

    async def start_snapshot_loop(self, interval: int = 3600) -> None:
        """
        Start periodic snapshot saving.

        Args:
            interval: Snapshot interval in seconds (default 1 hour)
        """
        if self._running:
            logger.warning("Snapshot loop already running")
            return

        self._running = True
        self._snapshot_task = asyncio.create_task(self._snapshot_loop(interval))
        logger.info(f"Started snapshot loop with {interval}s interval")

    async def stop_snapshot_loop(self) -> None:
        """Stop the snapshot loop."""
        self._running = False
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
            self._snapshot_task = None
        logger.info("Stopped snapshot loop")

    async def _snapshot_loop(self, interval: int) -> None:
        """Internal snapshot loop."""
        while self._running:
            try:
                await self._save_snapshot()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
                await asyncio.sleep(60)  # Wait a bit before retry

    async def _save_snapshot(self) -> None:
        """Save current state snapshot."""
        if not self._db:
            return

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": self.get_summary().to_dict(),
            "bots": [b.to_dict() for b in self._aggregator.collect_all()],
        }

        try:
            await self._db.save_snapshot(snapshot)
            logger.debug("Saved dashboard snapshot")
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    async def get_history(
        self,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> List[Dict[str, Any]]:
        """
        Get historical dashboard data.

        Args:
            start: Start time
            end: End time
            interval: Data interval ("1h", "1d", etc.)

        Returns:
            List of historical snapshots
        """
        if not self._db:
            return []

        try:
            return await self._db.get_snapshots(start, end, interval)
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
