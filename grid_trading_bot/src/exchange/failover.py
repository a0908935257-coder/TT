"""
Exchange Failover Manager.

Provides automatic failover between multiple exchange sources
for high availability and fault tolerance.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.core import get_logger
from src.core.retry import (
    RetryConfig,
    RetryStrategy,
    retry_async,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from src.core.models import Kline, MarketType

from .base import (
    BaseExchangeAPI,
    ExchangeConfig,
    ExchangeStatus,
    ExchangeType,
    OrderResult,
    TickerData,
    BalanceData,
    PositionData,
)

logger = get_logger(__name__)

T = TypeVar("T")


class FailoverStrategy(str, Enum):
    """Failover strategy types."""
    PRIORITY = "priority"          # Use exchanges by priority order
    ROUND_ROBIN = "round_robin"    # Rotate between exchanges
    FASTEST = "fastest"            # Use fastest responding exchange
    RANDOM = "random"              # Random selection


@dataclass
class ExchangeHealth:
    """Health status for an exchange."""
    exchange_type: ExchangeType
    is_healthy: bool = True
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.last_success = datetime.now(timezone.utc)
        self.consecutive_failures = 0
        self.total_requests += 1

        # Update average latency (exponential moving average)
        alpha = 0.2
        if self.average_latency_ms == 0:
            self.average_latency_ms = latency_ms
        else:
            self.average_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self.average_latency_ms
            )

        # Update error rate
        self.error_rate = self.failed_requests / self.total_requests

    def record_failure(self) -> None:
        """Record a failed request."""
        self.last_failure = datetime.now(timezone.utc)
        self.consecutive_failures += 1
        self.total_requests += 1
        self.failed_requests += 1
        self.error_rate = self.failed_requests / self.total_requests

        # Mark as unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False

    def check_recovery(self, recovery_time: timedelta = timedelta(minutes=5)) -> None:
        """Check if exchange should be marked healthy again."""
        if not self.is_healthy and self.last_failure:
            time_since_failure = datetime.now(timezone.utc) - self.last_failure
            if time_since_failure >= recovery_time:
                self.is_healthy = True
                logger.info(f"Exchange {self.exchange_type.value} marked healthy after recovery period")


@dataclass
class FailoverConfig:
    """
    Configuration for failover manager.

    Attributes:
        strategy: Failover strategy
        max_retries_per_exchange: Max retries before switching
        health_check_interval: Seconds between health checks
        recovery_time: Seconds before retrying failed exchange
        enable_circuit_breaker: Use circuit breaker pattern
        circuit_breaker_config: Circuit breaker configuration
    """
    strategy: FailoverStrategy = FailoverStrategy.PRIORITY
    max_retries_per_exchange: int = 2
    health_check_interval: float = 60.0
    recovery_time: float = 300.0  # 5 minutes
    enable_circuit_breaker: bool = True
    circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=lambda: CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=60.0,
        )
    )


class FailoverManager:
    """
    Manages failover between multiple exchange sources.

    Provides automatic switching between exchanges when one fails,
    with health monitoring and recovery mechanisms.

    Example:
        >>> manager = FailoverManager(config)
        >>> manager.add_exchange(binance_api, priority=1)
        >>> manager.add_exchange(okx_api, priority=2)
        >>>
        >>> # Will automatically failover to OKX if Binance fails
        >>> ticker = await manager.get_ticker("BTCUSDT")
    """

    def __init__(self, config: Optional[FailoverConfig] = None):
        """
        Initialize failover manager.

        Args:
            config: Failover configuration
        """
        self.config = config or FailoverConfig()

        self._exchanges: Dict[ExchangeType, BaseExchangeAPI] = {}
        self._priorities: Dict[ExchangeType, int] = {}
        self._health: Dict[ExchangeType, ExchangeHealth] = {}
        self._circuit_breakers: Dict[ExchangeType, CircuitBreaker] = {}

        self._current_exchange: Optional[ExchangeType] = None
        self._round_robin_index = 0
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # Retry configuration
        self._retry_config = RetryConfig(
            max_attempts=self.config.max_retries_per_exchange,
            base_delay=1.0,
            max_delay=10.0,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
        )

    def add_exchange(
        self,
        api: BaseExchangeAPI,
        priority: int = 0,
    ) -> None:
        """
        Add an exchange to the failover pool.

        Args:
            api: Exchange API instance
            priority: Priority (lower = higher priority)
        """
        exchange_type = api.exchange_type

        self._exchanges[exchange_type] = api
        self._priorities[exchange_type] = priority
        self._health[exchange_type] = ExchangeHealth(exchange_type=exchange_type)

        if self.config.enable_circuit_breaker:
            self._circuit_breakers[exchange_type] = CircuitBreaker(
                name=f"exchange_{exchange_type.value}",
                config=self.config.circuit_breaker_config,
            )

        # Set current exchange if first one
        if self._current_exchange is None:
            self._current_exchange = exchange_type

        logger.info(f"Added exchange {exchange_type.value} with priority {priority}")

    def remove_exchange(self, exchange_type: ExchangeType) -> None:
        """Remove an exchange from the pool."""
        self._exchanges.pop(exchange_type, None)
        self._priorities.pop(exchange_type, None)
        self._health.pop(exchange_type, None)
        self._circuit_breakers.pop(exchange_type, None)

        if self._current_exchange == exchange_type:
            self._current_exchange = self._get_next_exchange()

    async def start(self) -> None:
        """Start the failover manager with health checks."""
        if self._running:
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Failover manager started")

    async def stop(self) -> None:
        """Stop the failover manager."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Failover manager stopped")

    def _get_sorted_exchanges(self) -> List[ExchangeType]:
        """Get exchanges sorted by priority."""
        return sorted(
            self._exchanges.keys(),
            key=lambda x: self._priorities.get(x, 999)
        )

    def _get_healthy_exchanges(self) -> List[ExchangeType]:
        """Get list of healthy exchanges."""
        healthy = []
        for exchange_type in self._get_sorted_exchanges():
            health = self._health.get(exchange_type)
            if health and health.is_healthy:
                healthy.append(exchange_type)
        return healthy

    def _get_next_exchange(self) -> Optional[ExchangeType]:
        """Get next exchange based on strategy."""
        healthy = self._get_healthy_exchanges()
        if not healthy:
            # No healthy exchanges, try all
            healthy = self._get_sorted_exchanges()

        if not healthy:
            return None

        if self.config.strategy == FailoverStrategy.PRIORITY:
            return healthy[0]

        elif self.config.strategy == FailoverStrategy.ROUND_ROBIN:
            self._round_robin_index = (self._round_robin_index + 1) % len(healthy)
            return healthy[self._round_robin_index]

        elif self.config.strategy == FailoverStrategy.FASTEST:
            # Sort by latency
            sorted_by_latency = sorted(
                healthy,
                key=lambda x: self._health.get(x, ExchangeHealth(x)).average_latency_ms
            )
            return sorted_by_latency[0]

        elif self.config.strategy == FailoverStrategy.RANDOM:
            import random
            return random.choice(healthy)

        return healthy[0]

    async def _execute_with_failover(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a method with automatic failover.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from the method

        Raises:
            Exception: If all exchanges fail
        """
        tried_exchanges: List[ExchangeType] = []
        last_exception: Optional[Exception] = None

        for exchange_type in self._get_sorted_exchanges():
            if exchange_type in tried_exchanges:
                continue

            tried_exchanges.append(exchange_type)
            api = self._exchanges.get(exchange_type)
            health = self._health.get(exchange_type)
            circuit_breaker = self._circuit_breakers.get(exchange_type)

            if not api or not health:
                continue

            # Check health
            health.check_recovery(timedelta(seconds=self.config.recovery_time))
            if not health.is_healthy and len(tried_exchanges) < len(self._exchanges):
                logger.debug(f"Skipping unhealthy exchange: {exchange_type.value}")
                continue

            try:
                # Get the method
                method = getattr(api, method_name, None)
                if not method:
                    logger.error(f"Method {method_name} not found on {exchange_type.value}")
                    continue

                # Execute with circuit breaker
                start_time = datetime.now(timezone.utc)

                if circuit_breaker:
                    result = await circuit_breaker.call(method, *args, **kwargs)
                else:
                    result = await method(*args, **kwargs)

                # Record success
                latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                health.record_success(latency_ms)
                self._current_exchange = exchange_type

                return result

            except Exception as e:
                last_exception = e
                health.record_failure()
                logger.warning(
                    f"Exchange {exchange_type.value} failed for {method_name}: "
                    f"{type(e).__name__}: {e}"
                )

                # Continue to next exchange
                continue

        # All exchanges failed
        raise last_exception or RuntimeError("All exchanges failed")

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for exchange_type, api in self._exchanges.items():
                    health = self._health.get(exchange_type)
                    if not health:
                        continue

                    try:
                        start_time = datetime.now(timezone.utc)
                        await api.ping()
                        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                        health.record_success(latency_ms)
                        logger.debug(
                            f"Health check OK for {exchange_type.value}: {latency_ms:.1f}ms"
                        )

                    except Exception as e:
                        health.record_failure()
                        logger.warning(
                            f"Health check failed for {exchange_type.value}: {e}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    # =========================================================================
    # Proxied Methods with Failover
    # =========================================================================

    async def get_ticker(self, symbol: str) -> TickerData:
        """Get ticker with failover."""
        return await self._execute_with_failover("get_ticker", symbol)

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """Get order book with failover."""
        return await self._execute_with_failover("get_orderbook", symbol, limit)

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Kline]:
        """Get K-lines with failover."""
        return await self._execute_with_failover(
            "get_klines", symbol, interval, limit, start_time, end_time
        )

    async def get_balance(
        self,
        asset: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> Dict[str, BalanceData]:
        """Get balance with failover."""
        return await self._execute_with_failover("get_balance", asset, market_type)

    async def get_positions(
        self,
        symbol: Optional[str] = None,
    ) -> List[PositionData]:
        """Get positions with failover."""
        return await self._execute_with_failover("get_positions", symbol)

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        market_type: MarketType = MarketType.SPOT,
        **kwargs: Any,
    ) -> OrderResult:
        """
        Place order - NO failover for trading operations.

        Trading operations are NOT failed over to prevent duplicate orders.
        """
        # Use current exchange only for trading
        exchange_type = self._current_exchange
        if not exchange_type:
            raise RuntimeError("No exchange available")

        api = self._exchanges.get(exchange_type)
        if not api:
            raise RuntimeError(f"Exchange {exchange_type.value} not found")

        return await api.place_order(
            symbol, side, order_type, quantity, price, market_type, **kwargs
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Cancel order - NO failover for trading operations.
        """
        exchange_type = self._current_exchange
        if not exchange_type:
            raise RuntimeError("No exchange available")

        api = self._exchanges.get(exchange_type)
        if not api:
            raise RuntimeError(f"Exchange {exchange_type.value} not found")

        return await api.cancel_order(symbol, order_id, market_type)

    # =========================================================================
    # Status and Management
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get failover manager status."""
        return {
            "strategy": self.config.strategy.value,
            "current_exchange": self._current_exchange.value if self._current_exchange else None,
            "running": self._running,
            "exchanges": {
                et.value: {
                    "priority": self._priorities.get(et, 999),
                    "health": {
                        "is_healthy": h.is_healthy,
                        "consecutive_failures": h.consecutive_failures,
                        "average_latency_ms": round(h.average_latency_ms, 2),
                        "error_rate": round(h.error_rate, 4),
                        "total_requests": h.total_requests,
                    } if (h := self._health.get(et)) else None,
                    "circuit_breaker": (
                        cb.state.value if (cb := self._circuit_breakers.get(et)) else None
                    ),
                }
                for et in self._exchanges.keys()
            },
        }

    def get_health_report(self) -> Dict[str, ExchangeHealth]:
        """Get health report for all exchanges."""
        return dict(self._health)

    def force_exchange(self, exchange_type: ExchangeType) -> None:
        """Force use of a specific exchange."""
        if exchange_type not in self._exchanges:
            raise ValueError(f"Exchange {exchange_type.value} not in pool")
        self._current_exchange = exchange_type
        logger.info(f"Forced exchange to {exchange_type.value}")

    def reset_health(self, exchange_type: Optional[ExchangeType] = None) -> None:
        """Reset health status for exchanges."""
        if exchange_type:
            if exchange_type in self._health:
                self._health[exchange_type] = ExchangeHealth(exchange_type=exchange_type)
                if exchange_type in self._circuit_breakers:
                    self._circuit_breakers[exchange_type].reset()
        else:
            for et in self._exchanges.keys():
                self._health[et] = ExchangeHealth(exchange_type=et)
                if et in self._circuit_breakers:
                    self._circuit_breakers[et].reset()
