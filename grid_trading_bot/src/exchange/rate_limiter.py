"""
Rate Limiter for Exchange API Requests.

Provides rate limiting functionality to prevent exceeding exchange limits:
- Token bucket algorithm for smooth rate limiting
- Sliding window for accurate burst control
- Per-endpoint rate limiting
- Response header parsing (X-MBX-USED-WEIGHT, Retry-After)
- Adaptive backoff based on server responses
- Request queuing and prioritization

Binance Rate Limits Reference:
- Spot: 1200 weight/minute (orders have higher weight)
- Futures: 2400 weight/minute
- Order rate: 10 orders/second, 100,000 orders/day
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class RateLimitType(str, Enum):
    """Type of rate limit."""
    REQUEST_WEIGHT = "request_weight"  # General request weight
    ORDER_COUNT = "order_count"         # Order count per second
    RAW_REQUESTS = "raw_requests"       # Raw request count


class RequestPriority(int, Enum):
    """Request priority levels."""
    CRITICAL = 0   # Cancel orders, emergencies
    HIGH = 1       # Place orders
    MEDIUM = 2     # Query orders, positions
    LOW = 3        # Market data, info queries


@dataclass
class RateLimitConfig:
    """
    Rate limit configuration.

    Attributes:
        max_weight_per_minute: Maximum request weight per minute
        max_orders_per_second: Maximum orders per second
        max_orders_per_day: Maximum orders per day
        burst_tolerance_pct: Allow burst up to this % above limit
        warning_threshold_pct: Warn when reaching this % of limit
        backoff_multiplier: Multiplier for exponential backoff
        max_backoff_seconds: Maximum backoff time
        enable_adaptive: Enable adaptive rate limiting
    """
    max_weight_per_minute: int = 1200
    max_orders_per_second: int = 10
    max_orders_per_day: int = 100000
    burst_tolerance_pct: Decimal = field(default_factory=lambda: Decimal("0.10"))
    warning_threshold_pct: Decimal = field(default_factory=lambda: Decimal("0.80"))
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0
    enable_adaptive: bool = True


@dataclass
class EndpointWeight:
    """Weight configuration for specific endpoint."""
    endpoint: str
    weight: int = 1
    is_order: bool = False


@dataclass
class RateLimitStatus:
    """Current rate limit status."""
    current_weight: int
    max_weight: int
    current_order_count: int
    max_order_count: int
    weight_usage_pct: Decimal
    order_usage_pct: Decimal
    is_limited: bool
    retry_after_seconds: Optional[float] = None
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_warning(self) -> bool:
        """Check if approaching limit."""
        return self.weight_usage_pct >= Decimal("80") or self.order_usage_pct >= Decimal("80")


@dataclass
class RequestRecord:
    """Record of a single request."""
    timestamp: datetime
    weight: int
    endpoint: str
    is_order: bool = False


# =============================================================================
# Token Bucket
# =============================================================================


class TokenBucket:
    """
    Token bucket rate limiter.

    Provides smooth rate limiting with burst handling.
    Tokens are replenished at a constant rate.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
        initial_tokens: Optional[int] = None,
    ):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum bucket capacity
            refill_rate: Rate of token replenishment (tokens/second)
            initial_tokens: Starting tokens (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(initial_tokens if initial_tokens is not None else capacity)
        self._last_refill = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    @property
    def tokens(self) -> float:
        """Get current token count (after refill)."""
        self._refill()
        return self._tokens

    @property
    def is_empty(self) -> bool:
        """Check if bucket is empty."""
        return self.tokens <= 0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_refill).total_seconds()

        if elapsed > 0:
            added = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + added)
            self._last_refill = now

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum wait time (None for no wait)

        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if timeout is None or timeout <= 0:
                return False

            # Calculate wait time
            needed = tokens - self._tokens
            wait_time = needed / self.refill_rate

            if wait_time > timeout:
                return False

            # Wait and acquire
            await asyncio.sleep(wait_time)
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired immediately
        """
        return await self.acquire(tokens, timeout=0)

    def peek(self, tokens: int = 1) -> float:
        """
        Check wait time for acquiring tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated wait time in seconds (0 if available now)
        """
        self._refill()

        if self._tokens >= tokens:
            return 0.0

        needed = tokens - self._tokens
        return needed / self.refill_rate


# =============================================================================
# Sliding Window
# =============================================================================


class SlidingWindow:
    """
    Sliding window rate limiter.

    Provides accurate rate limiting over a time window.
    More precise than token bucket for bursty traffic.
    """

    def __init__(
        self,
        window_seconds: float,
        max_requests: int,
    ):
        """
        Initialize sliding window.

        Args:
            window_seconds: Window size in seconds
            max_requests: Maximum requests in window
        """
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self._requests: Deque[datetime] = deque()
        self._weights: Deque[Tuple[datetime, int]] = deque()
        self._lock = asyncio.Lock()

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        while self._weights and self._weights[0][0] < cutoff:
            self._weights.popleft()

    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        self._cleanup()
        return len(self._requests)

    @property
    def current_weight(self) -> int:
        """Get current weight sum in window."""
        self._cleanup()
        return sum(w for _, w in self._weights)

    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return max(0, self.max_requests - self.current_count)

    async def record(self, weight: int = 1) -> None:
        """
        Record a request.

        Args:
            weight: Request weight
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._requests.append(now)
            self._weights.append((now, weight))
            self._cleanup()

    async def can_proceed(self, weight: int = 1) -> bool:
        """
        Check if request can proceed.

        Args:
            weight: Request weight

        Returns:
            True if within limits
        """
        async with self._lock:
            self._cleanup()
            return self.current_weight + weight <= self.max_requests

    async def wait_if_needed(self, weight: int = 1) -> float:
        """
        Wait if rate limited, return wait time.

        Args:
            weight: Request weight

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            self._cleanup()

            if self.current_weight + weight <= self.max_requests:
                return 0.0

            # Calculate wait time until oldest request expires
            if self._weights:
                oldest = self._weights[0][0]
                wait_time = (
                    oldest + timedelta(seconds=self.window_seconds) -
                    datetime.now(timezone.utc)
                ).total_seconds()

                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return wait_time

            return 0.0


# =============================================================================
# Request Queue
# =============================================================================


@dataclass
class QueuedRequest:
    """Request waiting in queue."""
    id: str
    priority: RequestPriority
    weight: int
    callback: Callable[[], Any]
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    future: asyncio.Future = field(default_factory=asyncio.Future)


class RequestQueue:
    """
    Priority-based request queue.

    Queues requests when rate limited and processes them
    in priority order when capacity becomes available.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize request queue.

        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queues: Dict[RequestPriority, Deque[QueuedRequest]] = {
            priority: deque() for priority in RequestPriority
        }
        self._lock = asyncio.Lock()
        self._processing = False
        self._request_counter = 0

    @property
    def size(self) -> int:
        """Get total queue size."""
        return sum(len(q) for q in self._queues.values())

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.size >= self.max_size

    async def enqueue(
        self,
        callback: Callable[[], Any],
        priority: RequestPriority = RequestPriority.MEDIUM,
        weight: int = 1,
    ) -> asyncio.Future:
        """
        Add request to queue.

        Args:
            callback: Async callable to execute
            priority: Request priority
            weight: Request weight

        Returns:
            Future that resolves when request completes
        """
        async with self._lock:
            if self.is_full:
                raise RuntimeError("Request queue is full")

            self._request_counter += 1
            request = QueuedRequest(
                id=f"req_{self._request_counter}",
                priority=priority,
                weight=weight,
                callback=callback,
            )

            self._queues[priority].append(request)
            logger.debug(f"Queued request {request.id} with priority {priority.name}")

            return request.future

    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Get next request by priority.

        Returns:
            Next request or None if empty
        """
        async with self._lock:
            for priority in RequestPriority:
                if self._queues[priority]:
                    return self._queues[priority].popleft()
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "total": self.size,
            "critical": len(self._queues[RequestPriority.CRITICAL]),
            "high": len(self._queues[RequestPriority.HIGH]),
            "medium": len(self._queues[RequestPriority.MEDIUM]),
            "low": len(self._queues[RequestPriority.LOW]),
        }


# =============================================================================
# Main Rate Limiter
# =============================================================================


class RateLimiter:
    """
    Comprehensive rate limiter for exchange API requests.

    Combines token bucket, sliding window, and request queuing
    for robust rate limiting with prioritization.

    Features:
    - Token bucket for smooth rate limiting
    - Sliding window for accurate weight tracking
    - Priority-based request queuing
    - Response header parsing
    - Adaptive rate adjustment
    - Per-endpoint weight configuration

    Example:
        >>> config = RateLimitConfig(max_weight_per_minute=1200)
        >>> limiter = RateLimiter(config)
        >>>
        >>> # Check before request
        >>> if await limiter.acquire(weight=5):
        ...     response = await make_request()
        ...     limiter.update_from_headers(response.headers)
    """

    # Default endpoint weights for Binance
    DEFAULT_ENDPOINT_WEIGHTS = {
        # Market data
        "/api/v3/ticker/price": 1,
        "/api/v3/ticker/24hr": 1,
        "/api/v3/depth": 5,  # 5-50 depending on limit
        "/api/v3/klines": 1,
        "/api/v3/trades": 5,
        "/api/v3/exchangeInfo": 10,
        # Account
        "/api/v3/account": 10,
        "/api/v3/openOrders": 3,
        "/api/v3/allOrders": 10,
        # Orders
        "/api/v3/order": 1,  # Place/query single order
        "/api/v3/order/test": 1,
        # Futures
        "/fapi/v1/ticker/price": 1,
        "/fapi/v1/depth": 5,
        "/fapi/v1/klines": 1,
        "/fapi/v2/account": 5,
        "/fapi/v2/positionRisk": 5,
        "/fapi/v1/order": 1,
        "/fapi/v1/openOrders": 1,
    }

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        endpoint_weights: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
            endpoint_weights: Custom endpoint weights
        """
        self.config = config or RateLimitConfig()
        self._endpoint_weights = {
            **self.DEFAULT_ENDPOINT_WEIGHTS,
            **(endpoint_weights or {}),
        }

        # Rate limiting components
        tokens_per_second = self.config.max_weight_per_minute / 60.0
        self._weight_bucket = TokenBucket(
            capacity=self.config.max_weight_per_minute,
            refill_rate=tokens_per_second,
        )
        self._weight_window = SlidingWindow(
            window_seconds=60.0,
            max_requests=self.config.max_weight_per_minute,
        )
        self._order_window = SlidingWindow(
            window_seconds=1.0,
            max_requests=self.config.max_orders_per_second,
        )

        # Request queue
        self._queue = RequestQueue()

        # State tracking
        self._is_limited = False
        self._retry_after: Optional[float] = None
        self._server_weight: Optional[int] = None
        self._consecutive_limits = 0
        self._backoff_seconds = 1.0

        # History for analysis
        self._request_history: Deque[RequestRecord] = deque(maxlen=1000)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Background processing
        self._queue_processor_task: Optional[asyncio.Task] = None

    @property
    def is_limited(self) -> bool:
        """Check if currently rate limited."""
        return self._is_limited

    @property
    def current_weight(self) -> int:
        """Get current weight usage."""
        if self._server_weight is not None:
            return self._server_weight
        return self._weight_window.current_weight

    @property
    def available_weight(self) -> int:
        """Get available weight."""
        return max(0, self.config.max_weight_per_minute - self.current_weight)

    def get_endpoint_weight(self, endpoint: str) -> int:
        """
        Get weight for an endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Configured weight for endpoint
        """
        # Check exact match
        if endpoint in self._endpoint_weights:
            return self._endpoint_weights[endpoint]

        # Check prefix match
        for path, weight in self._endpoint_weights.items():
            if endpoint.startswith(path):
                return weight

        return 1  # Default weight

    async def acquire(
        self,
        weight: int = 1,
        is_order: bool = False,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire rate limit capacity.

        Args:
            weight: Request weight
            is_order: Whether this is an order request
            endpoint: API endpoint (for logging)
            timeout: Maximum wait time

        Returns:
            True if acquired, False if timeout/denied
        """
        async with self._lock:
            # Check if currently limited
            if self._is_limited:
                if self._retry_after:
                    if timeout and self._retry_after > timeout:
                        return False
                    logger.warning(
                        f"Rate limited, waiting {self._retry_after:.1f}s"
                    )
                    await asyncio.sleep(self._retry_after)
                    self._is_limited = False
                    self._retry_after = None

            # Check weight limit
            if not await self._weight_bucket.acquire(weight, timeout):
                logger.warning(f"Weight limit exceeded, weight={weight}")
                return False

            # Check order limit
            if is_order:
                if not await self._order_window.can_proceed():
                    wait_time = await self._order_window.wait_if_needed()
                    logger.debug(f"Order limit, waited {wait_time:.2f}s")

            # Record request
            await self._weight_window.record(weight)
            if is_order:
                await self._order_window.record()

            self._request_history.append(RequestRecord(
                timestamp=datetime.now(timezone.utc),
                weight=weight,
                endpoint=endpoint or "unknown",
                is_order=is_order,
            ))

            return True

    async def release(self, weight: int = 1) -> None:
        """
        Release weight (for cancellation before request sent).

        Args:
            weight: Weight to release
        """
        # Token bucket doesn't support release, but we can note it
        logger.debug(f"Released weight: {weight}")

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """
        Update rate limit state from response headers.

        Parses Binance-style headers:
        - X-MBX-USED-WEIGHT-1M: Used weight in 1 minute
        - X-MBX-ORDER-COUNT-1S: Order count in 1 second
        - X-MBX-ORDER-COUNT-1D: Order count in 1 day
        - Retry-After: Seconds to wait when rate limited

        Args:
            headers: Response headers
        """
        # Parse used weight
        weight_key = "X-MBX-USED-WEIGHT-1M"
        if weight_key in headers:
            try:
                self._server_weight = int(headers[weight_key])

                # Check warning threshold
                usage_pct = Decimal(str(self._server_weight)) / Decimal(
                    str(self.config.max_weight_per_minute)
                ) * Decimal("100")

                if usage_pct >= Decimal("80"):
                    logger.warning(
                        f"Rate limit warning: {usage_pct:.1f}% used "
                        f"({self._server_weight}/{self.config.max_weight_per_minute})"
                    )
            except ValueError:
                pass

        # Parse order counts
        order_1s_key = "X-MBX-ORDER-COUNT-1S"
        if order_1s_key in headers:
            try:
                order_count = int(headers[order_1s_key])
                if order_count >= self.config.max_orders_per_second:
                    logger.warning(f"Order rate limit: {order_count}/s")
            except ValueError:
                pass

        # Parse retry-after
        if "Retry-After" in headers:
            try:
                self._retry_after = float(headers["Retry-After"])
                self._is_limited = True
                self._consecutive_limits += 1
                logger.warning(f"Rate limited, retry after {self._retry_after}s")
            except ValueError:
                pass

    def handle_rate_limit_error(self, retry_after: Optional[float] = None) -> None:
        """
        Handle rate limit error response.

        Args:
            retry_after: Seconds to wait (if provided by exchange)
        """
        self._is_limited = True
        self._consecutive_limits += 1

        if retry_after:
            self._retry_after = retry_after
        else:
            # Exponential backoff
            self._backoff_seconds = min(
                self._backoff_seconds * self.config.backoff_multiplier,
                self.config.max_backoff_seconds,
            )
            self._retry_after = self._backoff_seconds

        logger.warning(
            f"Rate limit error, consecutive={self._consecutive_limits}, "
            f"backoff={self._retry_after:.1f}s"
        )

    def handle_success(self) -> None:
        """Handle successful request (reset backoff)."""
        if self._consecutive_limits > 0:
            self._consecutive_limits = 0
            self._backoff_seconds = 1.0
            logger.debug("Rate limit backoff reset")

    def get_status(self) -> RateLimitStatus:
        """
        Get current rate limit status.

        Returns:
            RateLimitStatus with current state
        """
        current_weight = self.current_weight
        max_weight = self.config.max_weight_per_minute
        current_orders = self._order_window.current_count
        max_orders = self.config.max_orders_per_second

        return RateLimitStatus(
            current_weight=current_weight,
            max_weight=max_weight,
            current_order_count=current_orders,
            max_order_count=max_orders,
            weight_usage_pct=Decimal(str(current_weight / max_weight * 100)),
            order_usage_pct=Decimal(str(current_orders / max_orders * 100)) if max_orders > 0 else Decimal("0"),
            is_limited=self._is_limited,
            retry_after_seconds=self._retry_after,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(minutes=5)

        recent_requests = [
            r for r in self._request_history
            if r.timestamp > recent_cutoff
        ]

        return {
            "current_weight": self.current_weight,
            "available_weight": self.available_weight,
            "max_weight": self.config.max_weight_per_minute,
            "is_limited": self._is_limited,
            "consecutive_limits": self._consecutive_limits,
            "backoff_seconds": self._backoff_seconds,
            "queue_size": self._queue.size,
            "requests_last_5min": len(recent_requests),
            "orders_last_5min": sum(1 for r in recent_requests if r.is_order),
            "total_weight_last_5min": sum(r.weight for r in recent_requests),
        }

    async def start_queue_processor(self) -> None:
        """Start background queue processor."""
        if self._queue_processor_task is None:
            self._queue_processor_task = asyncio.create_task(
                self._process_queue()
            )
            logger.info("Rate limiter queue processor started")

    async def stop_queue_processor(self) -> None:
        """Stop background queue processor."""
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
            self._queue_processor_task = None
            logger.info("Rate limiter queue processor stopped")

    async def _process_queue(self) -> None:
        """Background queue processor."""
        while True:
            try:
                request = await self._queue.dequeue()

                if request is None:
                    await asyncio.sleep(0.1)
                    continue

                # Wait for rate limit
                if not await self.acquire(weight=request.weight):
                    # Re-queue with same priority
                    await self._queue.enqueue(
                        request.callback,
                        request.priority,
                        request.weight,
                    )
                    await asyncio.sleep(0.1)
                    continue

                # Execute request
                try:
                    result = await request.callback()
                    request.future.set_result(result)
                except Exception as e:
                    request.future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1.0)


# =============================================================================
# Rate Limiter Manager
# =============================================================================


class RateLimiterManager:
    """
    Manager for multiple rate limiters.

    Manages separate rate limiters for different exchanges
    and market types (spot/futures).
    """

    def __init__(self):
        """Initialize rate limiter manager."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    def get_limiter(
        self,
        exchange: str,
        market_type: str = "spot",
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimiter:
        """
        Get or create rate limiter for exchange/market.

        Args:
            exchange: Exchange name
            market_type: Market type (spot/futures)
            config: Optional custom config

        Returns:
            RateLimiter instance
        """
        key = f"{exchange}_{market_type}"

        if key not in self._limiters:
            # Default configs by exchange
            if config is None:
                if exchange == "binance":
                    if market_type == "futures":
                        config = RateLimitConfig(max_weight_per_minute=2400)
                    else:
                        config = RateLimitConfig(max_weight_per_minute=1200)
                else:
                    config = RateLimitConfig()

            self._limiters[key] = RateLimiter(config)
            logger.info(f"Created rate limiter for {key}")

        return self._limiters[key]

    def get_all_status(self) -> Dict[str, RateLimitStatus]:
        """Get status of all limiters."""
        return {
            key: limiter.get_status()
            for key, limiter in self._limiters.items()
        }

    async def start_all(self) -> None:
        """Start all queue processors."""
        for limiter in self._limiters.values():
            await limiter.start_queue_processor()

    async def stop_all(self) -> None:
        """Stop all queue processors."""
        for limiter in self._limiters.values():
            await limiter.stop_queue_processor()


# Global rate limiter manager
_rate_limiter_manager: Optional[RateLimiterManager] = None


def get_rate_limiter_manager() -> RateLimiterManager:
    """Get global rate limiter manager."""
    global _rate_limiter_manager
    if _rate_limiter_manager is None:
        _rate_limiter_manager = RateLimiterManager()
    return _rate_limiter_manager
