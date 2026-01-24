"""
Circuit breaker module.

Implements the circuit breaker pattern for resilient service calls
with automatic failure detection and recovery.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Deque, Dict, Optional, TypeVar, Union

from src.core import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max concurrent calls in half-open
    window_seconds: float = 60.0  # Sliding window for failure counting
    exclude_exceptions: tuple = ()  # Exceptions that don't count as failures


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    state: CircuitState
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    times_opened: int = 0
    rejection_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": (
                self.last_state_change.isoformat() if self.last_state_change else None
            ),
            "times_opened": self.times_opened,
            "rejection_count": self.rejection_count,
            "failure_rate": (
                self.total_failures / self.total_calls
                if self.total_calls > 0
                else 0.0
            ),
        }


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class FailureRecord:
    """Record of a failure."""

    timestamp: datetime
    exception_type: str
    message: str


class CircuitBreaker:
    """
    Circuit breaker for resilient service calls.

    Prevents cascading failures by failing fast when a service
    is unhealthy, and automatically tests for recovery.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Name identifier for this circuit
            config: Circuit configuration
        """
        self._name = name
        self._config = config or CircuitConfig()

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats(state=self._state)
        self._failures: Deque[FailureRecord] = deque()
        self._lock = asyncio.Lock()

        self._half_open_calls = 0
        self._opened_at: Optional[float] = None

        # Callbacks
        self._on_open: Optional[Callable[[], None]] = None
        self._on_close: Optional[Callable[[], None]] = None
        self._on_half_open: Optional[Callable[[], None]] = None

    @property
    def name(self) -> str:
        """Get circuit name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    def on_state_change(
        self,
        on_open: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        on_half_open: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set state change callbacks."""
        self._on_open = on_open
        self._on_close = on_close
        self._on_half_open = on_half_open

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            # Check if we should allow the call
            if not await self._should_allow_call():
                self._stats.rejection_count += 1
                retry_after = self._get_retry_after()
                raise CircuitBreakerError(
                    f"Circuit '{self._name}' is open",
                    retry_after=retry_after,
                )

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            # Execute the call
            self._stats.total_calls += 1
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            await self._record_success()
            return result

        except Exception as e:
            # Check if this exception should be excluded
            if isinstance(e, self._config.exclude_exceptions):
                await self._record_success()
                raise

            # Record failure
            await self._record_failure(e)
            raise

        finally:
            if self._state == CircuitState.HALF_OPEN:
                async with self._lock:
                    self._half_open_calls -= 1

    async def _should_allow_call(self) -> bool:
        """Check if a call should be allowed."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._opened_at is not None:
                elapsed = time.time() - self._opened_at
                if elapsed >= self._config.timeout_seconds:
                    await self._transition_to_half_open()
                    return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            return self._half_open_calls < self._config.half_open_max_calls

        return False

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                if (
                    self._stats.consecutive_successes
                    >= self._config.success_threshold
                ):
                    await self._transition_to_closed()

    async def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.now(timezone.utc)

            # Record failure details
            self._failures.append(
                FailureRecord(
                    timestamp=datetime.now(timezone.utc),
                    exception_type=type(exception).__name__,
                    message=str(exception)[:200],
                )
            )

            # Clean old failures
            self._clean_old_failures()

            # Check if we should open
            if self._state == CircuitState.CLOSED:
                if self._count_recent_failures() >= self._config.failure_threshold:
                    await self._transition_to_open()

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                await self._transition_to_open()

    def _clean_old_failures(self) -> None:
        """Remove failures outside the window."""
        cutoff = time.time() - self._config.window_seconds
        while (
            self._failures
            and self._failures[0].timestamp.timestamp() < cutoff
        ):
            self._failures.popleft()

    def _count_recent_failures(self) -> int:
        """Count failures in the current window."""
        cutoff = time.time() - self._config.window_seconds
        return sum(
            1 for f in self._failures if f.timestamp.timestamp() >= cutoff
        )

    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        if self._state == CircuitState.OPEN:
            return

        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        self._stats.state = CircuitState.OPEN
        self._stats.times_opened += 1
        self._stats.last_state_change = datetime.now(timezone.utc)

        logger.warning(f"Circuit '{self._name}' opened")

        if self._on_open:
            try:
                self._on_open()
            except Exception as e:
                logger.error(f"Error in on_open callback: {e}")

    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._stats.state = CircuitState.HALF_OPEN
        self._stats.consecutive_successes = 0
        self._stats.last_state_change = datetime.now(timezone.utc)

        logger.info(f"Circuit '{self._name}' half-opened")

        if self._on_half_open:
            try:
                self._on_half_open()
            except Exception as e:
                logger.error(f"Error in on_half_open callback: {e}")

    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._stats.state = CircuitState.CLOSED
        self._stats.last_state_change = datetime.now(timezone.utc)
        self._failures.clear()

        logger.info(f"Circuit '{self._name}' closed")

        if self._on_close:
            try:
                self._on_close()
            except Exception as e:
                logger.error(f"Error in on_close callback: {e}")

    def _get_retry_after(self) -> Optional[float]:
        """Get time until retry is possible."""
        if self._opened_at is None:
            return None
        elapsed = time.time() - self._opened_at
        remaining = self._config.timeout_seconds - elapsed
        return max(0, remaining)

    async def reset(self) -> None:
        """Manually reset the circuit to closed."""
        async with self._lock:
            await self._transition_to_closed()
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0

    async def force_open(self) -> None:
        """Manually open the circuit."""
        async with self._lock:
            await self._transition_to_open()

    def get_stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    def get_recent_failures(self, limit: int = 10) -> list[FailureRecord]:
        """Get recent failure records."""
        return list(self._failures)[-limit:]


class CircuitBreakerRegistry:
    """Registry of circuit breakers."""

    def __init__(self):
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(name, config)
        return self._circuits[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._circuits.get(name)

    def get_all(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        return dict(self._circuits)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuits."""
        return {
            name: circuit.get_stats().to_dict()
            for name, circuit in self._circuits.items()
        }

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for circuit in self._circuits.values():
            await circuit.reset()


# Global registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from global registry."""
    return _global_registry.get_or_create(name, config)


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _global_registry


def circuit_protected(
    circuit_name: Optional[str] = None,
    config: Optional[CircuitConfig] = None,
) -> Callable[[F], F]:
    """
    Decorator to protect a function with a circuit breaker.

    Args:
        circuit_name: Circuit name (defaults to function name)
        config: Circuit configuration

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        name = circuit_name or func.__name__
        circuit = get_circuit_breaker(name, config)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await circuit.call(func, *args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # For sync functions, we need to run in event loop
                import asyncio

                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    circuit.call(func, *args, **kwargs)
                )

            return sync_wrapper  # type: ignore

    return decorator


class RateLimitingCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker with integrated rate limiting.

    Combines circuit breaking with rate limiting for
    comprehensive protection.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
        max_calls_per_second: float = 100.0,
    ):
        """
        Initialize with rate limiting.

        Args:
            name: Circuit name
            config: Circuit config
            max_calls_per_second: Maximum calls per second
        """
        super().__init__(name, config)
        self._max_calls_per_second = max_calls_per_second
        self._tokens = max_calls_per_second
        self._last_refill = time.time()
        self._rate_lock = asyncio.Lock()

    async def _refill_tokens(self) -> None:
        """Refill rate limit tokens."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_calls_per_second,
            self._tokens + elapsed * self._max_calls_per_second,
        )
        self._last_refill = now

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute with rate limiting and circuit breaking."""
        async with self._rate_lock:
            await self._refill_tokens()
            if self._tokens < 1:
                raise CircuitBreakerError(
                    f"Rate limit exceeded for '{self._name}'",
                    retry_after=1.0 / self._max_calls_per_second,
                )
            self._tokens -= 1

        return await super().call(func, *args, **kwargs)
