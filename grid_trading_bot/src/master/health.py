"""
Health Checker.

Performs comprehensive health checks on trading bots.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore

from src.core import get_logger

from .models import BotState
from .registry import BotRegistry

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"        # All checks passed
    DEGRADED = "degraded"      # Some checks have warnings
    UNHEALTHY = "unhealthy"    # Critical checks failed
    UNKNOWN = "unknown"        # Unable to determine status


@dataclass
class CheckItem:
    """
    Individual health check result.

    Attributes:
        name: Check name
        status: Health status
        message: Descriptive message
        value: Optional metric value
    """

    name: str
    status: HealthStatus
    message: str = ""
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
        }


@dataclass
class HealthCheckResult:
    """
    Complete health check result for a bot.

    Attributes:
        bot_id: Bot identifier
        status: Overall health status
        checks: Individual check results
        checked_at: Check timestamp
    """

    bot_id: str
    status: HealthStatus
    checks: dict[str, CheckItem] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_id": self.bot_id,
            "status": self.status.value,
            "checks": {name: item.to_dict() for name, item in self.checks.items()},
            "checked_at": self.checked_at.isoformat(),
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return self.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)


class BotInstanceProtocol(Protocol):
    """Protocol for bot instances."""

    @property
    def bot_id(self) -> str: ...

    @property
    def state(self) -> BotState: ...

    @property
    def start_time(self) -> Optional[datetime]: ...


class HealthChecker:
    """
    Health Checker.

    Performs comprehensive health checks on trading bots.

    Example:
        >>> checker = HealthChecker(registry)
        >>> result = await checker.check("bot_001")
        >>> print(result.status)
    """

    # Thresholds
    HEARTBEAT_HEALTHY_SECONDS = 30
    HEARTBEAT_DEGRADED_SECONDS = 60
    ORDER_DIFF_DEGRADED = 2
    ORDER_DIFF_UNHEALTHY = 5
    MEMORY_DEGRADED_PERCENT = 70
    MEMORY_UNHEALTHY_PERCENT = 85

    def __init__(
        self,
        registry: BotRegistry,
        heartbeat_monitor: Optional[Any] = None,
    ):
        """
        Initialize HealthChecker.

        Args:
            registry: BotRegistry instance
            heartbeat_monitor: Optional HeartbeatMonitor for heartbeat checks
        """
        self._registry = registry
        self._heartbeat_monitor = heartbeat_monitor

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def check(self, bot_id: str) -> HealthCheckResult:
        """
        Perform health check on a single bot.

        Args:
            bot_id: Bot identifier

        Returns:
            HealthCheckResult with all check results
        """
        # Get bot info
        bot_info = self._registry.get(bot_id)
        if bot_info is None:
            return HealthCheckResult(
                bot_id=bot_id,
                status=HealthStatus.UNKNOWN,
                checks={
                    "registration": CheckItem(
                        name="registration",
                        status=HealthStatus.UNHEALTHY,
                        message="Bot not registered",
                    )
                },
            )

        # Get bot instance
        instance = self._registry.get_bot_instance(bot_id)

        # Perform individual checks
        checks: dict[str, CheckItem] = {}

        # 1. Heartbeat check
        checks["heartbeat"] = await self._check_heartbeat(bot_id, bot_info)

        # 2. State check
        checks["state"] = self._check_state(bot_info)

        # 3. Exchange connection check (if instance available)
        if instance:
            checks["exchange"] = await self._check_exchange(instance)

        # 4. Order sync check (if instance available)
        if instance:
            checks["orders"] = await self._check_orders(instance)

        # 5. Memory check
        checks["memory"] = self._check_memory()

        # Calculate overall status
        overall_status = self._calculate_overall_status(checks)

        return HealthCheckResult(
            bot_id=bot_id,
            status=overall_status,
            checks=checks,
        )

    async def check_all(self) -> list[HealthCheckResult]:
        """
        Perform health check on all registered bots.

        Returns:
            List of HealthCheckResult for all bots
        """
        results = []
        all_bots = self._registry.get_all()

        # Run checks concurrently
        tasks = [self.check(bot.bot_id) for bot in all_bots]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                bot_id = all_bots[i].bot_id
                logger.error(f"Health check failed for {bot_id}: {result}")
                valid_results.append(
                    HealthCheckResult(
                        bot_id=bot_id,
                        status=HealthStatus.UNKNOWN,
                        checks={
                            "error": CheckItem(
                                name="error",
                                status=HealthStatus.UNHEALTHY,
                                message=str(result),
                            )
                        },
                    )
                )
            else:
                valid_results.append(result)

        return valid_results

    def get_summary(self) -> dict[str, Any]:
        """
        Get health summary for all bots.

        This is a synchronous summary based on cached data.

        Returns:
            Dict with health summary statistics
        """
        all_bots = self._registry.get_all()

        summary = {
            "total": len(all_bots),
            "by_state": {},
            "healthy_count": 0,
            "degraded_count": 0,
            "unhealthy_count": 0,
        }

        for bot in all_bots:
            state = bot.state.value
            summary["by_state"][state] = summary["by_state"].get(state, 0) + 1

        return summary

    # =========================================================================
    # Individual Checks
    # =========================================================================

    async def _check_heartbeat(self, bot_id: str, bot_info: Any) -> CheckItem:
        """
        Check heartbeat status.

        Args:
            bot_id: Bot identifier
            bot_info: BotInfo instance

        Returns:
            CheckItem with heartbeat status
        """
        # Use heartbeat monitor if available
        if self._heartbeat_monitor:
            status = self._heartbeat_monitor.get_status(bot_id)
            if status.get("last_heartbeat") is None:
                return CheckItem(
                    name="heartbeat",
                    status=HealthStatus.UNKNOWN,
                    message="No heartbeat received",
                )

            elapsed = status.get("elapsed_seconds", float("inf"))
        else:
            # Fall back to registry last_heartbeat
            last_heartbeat = bot_info.last_heartbeat
            if last_heartbeat is None:
                return CheckItem(
                    name="heartbeat",
                    status=HealthStatus.UNKNOWN,
                    message="No heartbeat recorded",
                )

            elapsed = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()

        # Evaluate status based on elapsed time
        if elapsed < self.HEARTBEAT_HEALTHY_SECONDS:
            return CheckItem(
                name="heartbeat",
                status=HealthStatus.HEALTHY,
                message="Heartbeat normal",
                value=elapsed,
            )
        elif elapsed < self.HEARTBEAT_DEGRADED_SECONDS:
            return CheckItem(
                name="heartbeat",
                status=HealthStatus.DEGRADED,
                message=f"Heartbeat delayed ({elapsed:.1f}s)",
                value=elapsed,
            )
        else:
            return CheckItem(
                name="heartbeat",
                status=HealthStatus.UNHEALTHY,
                message=f"Heartbeat timeout ({elapsed:.1f}s)",
                value=elapsed,
            )

    def _check_state(self, bot_info: Any) -> CheckItem:
        """
        Check bot state.

        Args:
            bot_info: BotInfo instance

        Returns:
            CheckItem with state status
        """
        state = bot_info.state

        if state == BotState.RUNNING:
            return CheckItem(
                name="state",
                status=HealthStatus.HEALTHY,
                message="Running normally",
                value=state.value,
            )
        elif state == BotState.PAUSED:
            return CheckItem(
                name="state",
                status=HealthStatus.DEGRADED,
                message="Bot is paused",
                value=state.value,
            )
        elif state == BotState.ERROR:
            return CheckItem(
                name="state",
                status=HealthStatus.UNHEALTHY,
                message=f"Bot in error state: {bot_info.error_message or 'Unknown'}",
                value=state.value,
            )
        elif state in (BotState.STOPPED, BotState.STOPPING):
            return CheckItem(
                name="state",
                status=HealthStatus.DEGRADED,
                message=f"Bot is {state.value}",
                value=state.value,
            )
        else:
            return CheckItem(
                name="state",
                status=HealthStatus.UNKNOWN,
                message=f"State: {state.value}",
                value=state.value,
            )

    async def _check_exchange(self, instance: Any) -> CheckItem:
        """
        Check exchange connection.

        Args:
            instance: Bot instance

        Returns:
            CheckItem with exchange connection status
        """
        try:
            # Try to get exchange client
            exchange = getattr(instance, "_exchange", None)
            if exchange is None:
                exchange = getattr(instance, "exchange", None)

            if exchange is None:
                return CheckItem(
                    name="exchange",
                    status=HealthStatus.UNKNOWN,
                    message="No exchange client found",
                )

            # Try to ping or get account info
            if hasattr(exchange, "ping"):
                await exchange.ping()
            elif hasattr(exchange, "get_account"):
                await exchange.get_account()
            elif hasattr(exchange, "get_time"):
                await exchange.get_time()
            else:
                return CheckItem(
                    name="exchange",
                    status=HealthStatus.UNKNOWN,
                    message="Cannot verify exchange connection",
                )

            return CheckItem(
                name="exchange",
                status=HealthStatus.HEALTHY,
                message="Exchange connection OK",
            )

        except Exception as e:
            return CheckItem(
                name="exchange",
                status=HealthStatus.UNHEALTHY,
                message=f"Exchange error: {str(e)}",
            )

    async def _check_orders(self, instance: Any) -> CheckItem:
        """
        Check order synchronization.

        Args:
            instance: Bot instance

        Returns:
            CheckItem with order sync status
        """
        try:
            # Get order manager
            order_manager = getattr(instance, "_order_manager", None)
            if order_manager is None:
                order_manager = getattr(instance, "order_manager", None)

            if order_manager is None:
                return CheckItem(
                    name="orders",
                    status=HealthStatus.UNKNOWN,
                    message="No order manager found",
                )

            # Get local order count
            local_count = 0
            if hasattr(order_manager, "active_order_count"):
                local_count = order_manager.active_order_count
            elif hasattr(order_manager, "get_all_orders"):
                local_orders = order_manager.get_all_orders()
                local_count = len(local_orders) if local_orders else 0

            # Try to get exchange orders
            exchange = getattr(instance, "_exchange", None) or getattr(instance, "exchange", None)
            symbol = getattr(instance, "_config", None)
            if symbol and hasattr(symbol, "symbol"):
                symbol = symbol.symbol
            else:
                symbol = getattr(instance, "symbol", None)

            if exchange and symbol and hasattr(exchange, "get_open_orders"):
                exchange_orders = await exchange.get_open_orders(symbol)
                exchange_count = len(exchange_orders) if exchange_orders else 0

                diff = abs(local_count - exchange_count)

                if diff == 0:
                    return CheckItem(
                        name="orders",
                        status=HealthStatus.HEALTHY,
                        message=f"Orders synced ({local_count} orders)",
                        value={"local": local_count, "exchange": exchange_count, "diff": 0},
                    )
                elif diff <= self.ORDER_DIFF_DEGRADED:
                    return CheckItem(
                        name="orders",
                        status=HealthStatus.DEGRADED,
                        message=f"Order mismatch: local={local_count}, exchange={exchange_count}",
                        value={"local": local_count, "exchange": exchange_count, "diff": diff},
                    )
                else:
                    return CheckItem(
                        name="orders",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Order sync issue: diff={diff}",
                        value={"local": local_count, "exchange": exchange_count, "diff": diff},
                    )

            return CheckItem(
                name="orders",
                status=HealthStatus.UNKNOWN,
                message=f"Cannot verify orders (local count: {local_count})",
                value={"local": local_count},
            )

        except Exception as e:
            return CheckItem(
                name="orders",
                status=HealthStatus.DEGRADED,
                message=f"Order check error: {str(e)}",
            )

    def _check_memory(self) -> CheckItem:
        """
        Check system memory usage.

        Returns:
            CheckItem with memory status
        """
        if not HAS_PSUTIL:
            return CheckItem(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not available",
            )

        try:
            memory = psutil.virtual_memory()
            percent = memory.percent

            if percent < self.MEMORY_DEGRADED_PERCENT:
                return CheckItem(
                    name="memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Memory usage: {percent:.1f}%",
                    value=percent,
                )
            elif percent < self.MEMORY_UNHEALTHY_PERCENT:
                return CheckItem(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"Memory usage high: {percent:.1f}%",
                    value=percent,
                )
            else:
                return CheckItem(
                    name="memory",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory usage critical: {percent:.1f}%",
                    value=percent,
                )

        except Exception as e:
            return CheckItem(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Cannot check memory: {str(e)}",
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_overall_status(self, checks: dict[str, CheckItem]) -> HealthStatus:
        """
        Calculate overall health status from individual checks.

        Priority:
        1. Any UNHEALTHY -> UNHEALTHY
        2. Any DEGRADED -> DEGRADED
        3. All HEALTHY -> HEALTHY
        4. Otherwise -> UNKNOWN

        Args:
            checks: Dict of CheckItem results

        Returns:
            Overall HealthStatus
        """
        if not checks:
            return HealthStatus.UNKNOWN

        has_unhealthy = False
        has_degraded = False
        has_healthy = False

        for check in checks.values():
            if check.status == HealthStatus.UNHEALTHY:
                has_unhealthy = True
            elif check.status == HealthStatus.DEGRADED:
                has_degraded = True
            elif check.status == HealthStatus.HEALTHY:
                has_healthy = True

        if has_unhealthy:
            return HealthStatus.UNHEALTHY
        elif has_degraded:
            return HealthStatus.DEGRADED
        elif has_healthy:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
