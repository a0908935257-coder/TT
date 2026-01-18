"""
Metrics Aggregator for collecting bot metrics.

Collects and caches metrics from all registered bots for dashboard display.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core import get_logger
from src.master.models import BotState, BotType

if TYPE_CHECKING:
    from src.master.registry import BotRegistry

logger = get_logger(__name__)


@dataclass
class BotMetrics:
    """Metrics data for a single bot."""

    bot_id: str
    bot_type: BotType
    symbol: str
    state: BotState

    # Time metrics
    uptime: timedelta = field(default_factory=lambda: timedelta())
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Investment metrics
    total_investment: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    total_profit: Decimal = Decimal("0")
    profit_rate: Decimal = Decimal("0")  # Percentage

    # Trade metrics
    total_trades: int = 0
    today_profit: Decimal = Decimal("0")
    today_trades: int = 0

    # Order metrics
    pending_orders: int = 0
    position_value: Decimal = Decimal("0")

    # Last activity
    last_trade_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_id": self.bot_id,
            "bot_type": self.bot_type.value,
            "symbol": self.symbol,
            "state": self.state.value,
            "uptime_seconds": self.uptime.total_seconds(),
            "updated_at": self.updated_at.isoformat(),
            "total_investment": str(self.total_investment),
            "current_value": str(self.current_value),
            "total_profit": str(self.total_profit),
            "profit_rate": str(self.profit_rate),
            "total_trades": self.total_trades,
            "today_profit": str(self.today_profit),
            "today_trades": self.today_trades,
            "pending_orders": self.pending_orders,
            "position_value": str(self.position_value),
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
        }


class MetricsAggregator:
    """
    Aggregates metrics from all registered bots.

    Provides cached access to bot metrics for dashboard display.
    """

    def __init__(self, registry: "BotRegistry"):
        """
        Initialize the metrics aggregator.

        Args:
            registry: Bot registry instance
        """
        self._registry = registry
        self._cache: Dict[str, BotMetrics] = {}
        self._last_update: Optional[datetime] = None
        self._cache_ttl: int = 5  # Cache TTL in seconds

    @property
    def last_update(self) -> Optional[datetime]:
        """Get the last update time."""
        return self._last_update

    @property
    def cache_size(self) -> int:
        """Get the number of cached metrics."""
        return len(self._cache)

    def collect(self, bot_id: str) -> Optional[BotMetrics]:
        """
        Collect metrics for a single bot.

        Args:
            bot_id: The bot ID to collect metrics for

        Returns:
            BotMetrics if bot exists, None otherwise
        """
        bot_info = self._registry.get(bot_id)
        if not bot_info:
            logger.warning(f"Bot {bot_id} not found in registry")
            return None

        # Get bot instance if available
        instance = self._registry.get_bot_instance(bot_id)

        # Calculate uptime
        uptime = timedelta()
        if bot_info.started_at:
            if bot_info.stopped_at:
                uptime = bot_info.stopped_at - bot_info.started_at
            else:
                uptime = datetime.now(timezone.utc) - bot_info.started_at

        # Get metrics from instance if available
        total_investment = Decimal("0")
        current_value = Decimal("0")
        total_profit = Decimal("0")
        total_trades = 0
        today_profit = Decimal("0")
        today_trades = 0
        pending_orders = 0
        position_value = Decimal("0")
        last_trade_at = None

        if instance:
            try:
                # Try to get statistics from bot instance
                if hasattr(instance, "get_statistics"):
                    stats = instance.get_statistics()
                    total_profit = Decimal(str(stats.get("total_profit", 0)))
                    total_trades = stats.get("trade_count", 0)

                # Try to get config for investment
                if hasattr(instance, "config"):
                    config = instance.config
                    if hasattr(config, "total_investment"):
                        total_investment = config.total_investment

                # Try to get order count
                if hasattr(instance, "order_manager"):
                    pending_orders = instance.order_manager.active_order_count

                # Try to get today's stats
                if hasattr(instance, "get_today_stats"):
                    today_stats = instance.get_today_stats()
                    today_profit = Decimal(str(today_stats.get("profit", 0)))
                    today_trades = today_stats.get("trades", 0)

            except Exception as e:
                logger.warning(f"Error collecting metrics for {bot_id}: {e}")

        # Calculate profit rate
        profit_rate = Decimal("0")
        if total_investment > 0:
            profit_rate = (total_profit / total_investment) * Decimal("100")

        # Calculate current value
        current_value = total_investment + total_profit

        metrics = BotMetrics(
            bot_id=bot_id,
            bot_type=bot_info.bot_type,
            symbol=bot_info.symbol,
            state=bot_info.state,
            uptime=uptime,
            updated_at=datetime.now(timezone.utc),
            total_investment=total_investment,
            current_value=current_value,
            total_profit=total_profit,
            profit_rate=profit_rate,
            total_trades=total_trades,
            today_profit=today_profit,
            today_trades=today_trades,
            pending_orders=pending_orders,
            position_value=position_value,
            last_trade_at=last_trade_at,
        )

        # Update cache
        self._cache[bot_id] = metrics

        return metrics

    def collect_all(self) -> List[BotMetrics]:
        """
        Collect metrics for all registered bots.

        Returns:
            List of BotMetrics for all bots
        """
        metrics_list = []

        for bot_info in self._registry.get_all():
            metrics = self.collect(bot_info.bot_id)
            if metrics:
                metrics_list.append(metrics)

        self._last_update = datetime.now(timezone.utc)

        return metrics_list

    def get_cached(self, bot_id: str) -> Optional[BotMetrics]:
        """
        Get cached metrics for a bot.

        Args:
            bot_id: The bot ID

        Returns:
            Cached BotMetrics if available, None otherwise
        """
        metrics = self._cache.get(bot_id)

        # Check if cache is stale
        if metrics:
            age = (datetime.now(timezone.utc) - metrics.updated_at).total_seconds()
            if age > self._cache_ttl:
                # Cache is stale, refresh
                return self.collect(bot_id)

        return metrics

    def refresh(self) -> None:
        """Refresh all cached metrics."""
        self.collect_all()
        logger.debug(f"Refreshed metrics for {len(self._cache)} bots")

    def clear_cache(self) -> None:
        """Clear all cached metrics."""
        self._cache.clear()
        self._last_update = None

    def remove_from_cache(self, bot_id: str) -> None:
        """
        Remove a bot from the cache.

        Args:
            bot_id: The bot ID to remove
        """
        if bot_id in self._cache:
            del self._cache[bot_id]

    def get_by_state(self, state: BotState) -> List[BotMetrics]:
        """
        Get cached metrics filtered by state.

        Args:
            state: The state to filter by

        Returns:
            List of BotMetrics matching the state
        """
        return [m for m in self._cache.values() if m.state == state]

    def get_by_type(self, bot_type: BotType) -> List[BotMetrics]:
        """
        Get cached metrics filtered by bot type.

        Args:
            bot_type: The bot type to filter by

        Returns:
            List of BotMetrics matching the type
        """
        return [m for m in self._cache.values() if m.bot_type == bot_type]

    def get_totals(self) -> Dict[str, Any]:
        """
        Get aggregated totals from cached metrics.

        Returns:
            Dictionary with total values
        """
        if not self._cache:
            return {
                "total_investment": Decimal("0"),
                "total_value": Decimal("0"),
                "total_profit": Decimal("0"),
                "total_trades": 0,
                "today_profit": Decimal("0"),
                "today_trades": 0,
                "pending_orders": 0,
            }

        return {
            "total_investment": sum(m.total_investment for m in self._cache.values()),
            "total_value": sum(m.current_value for m in self._cache.values()),
            "total_profit": sum(m.total_profit for m in self._cache.values()),
            "total_trades": sum(m.total_trades for m in self._cache.values()),
            "today_profit": sum(m.today_profit for m in self._cache.values()),
            "today_trades": sum(m.today_trades for m in self._cache.values()),
            "pending_orders": sum(m.pending_orders for m in self._cache.values()),
        }
