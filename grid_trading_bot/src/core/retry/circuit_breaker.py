"""
Circuit Breaker Pattern.

Prevents cascading failures by failing fast when a service is unhealthy.
"""

import asyncio
import functools
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from src.core import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open to close
        timeout: Seconds before attempting recovery (open -> half-open)
        half_open_max_calls: Max calls allowed in half-open state
        excluded_exceptions: Exceptions that don't count as failures
        on_state_change: Callback when state changes
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3
    excluded_exceptions: tuple = ()
    on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(timezone.utc)

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now(timezone.utc)

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self.total_calls += 1
        self.rejected_calls += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is open. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit Breaker implementation.

    Monitors failures and prevents cascading failures by rejecting
    requests when a threshold is reached.

    States:
    - CLOSED: Normal operation
    - OPEN: Requests rejected, waiting for timeout
    - HALF_OPEN: Testing with limited requests

    Example:
        >>> breaker = CircuitBreaker("api", config)
        >>> try:
        ...     result = await breaker.call(fetch_data, "BTCUSDT")
        ... except CircuitOpenError as e:
        ...     print(f"Service unavailable, retry in {e.retry_after}s")
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name for this circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._opened_at: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        self.stats = CircuitStats()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit state."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self.stats.state_changes += 1

            logger.info(
                f"Circuit breaker '{self.name}' state change: "
                f"{old_state.value} -> {new_state.value}"
            )

            if self.config.on_state_change:
                self.config.on_state_change(old_state, new_state)

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset (move to half-open)."""
        if self._state != CircuitState.OPEN:
            return False

        if self._opened_at is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
        return elapsed >= self.config.timeout

    def _get_retry_after(self) -> float:
        """Get seconds until retry is allowed."""
        if self._opened_at is None:
            return 0.0

        elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
        remaining = self.config.timeout - elapsed
        return max(0.0, remaining)

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from the function
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._change_state(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                else:
                    self.stats.record_rejection()
                    raise CircuitOpenError(self.name, self._get_retry_after())

            # Check half-open call limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self.stats.record_rejection()
                    raise CircuitOpenError(self.name, self._get_retry_after())
                self._half_open_calls += 1

        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            # Check if this exception should be excluded
            if isinstance(e, self.config.excluded_exceptions):
                await self._on_success()
                raise

            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.stats.record_success()

            if self._state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.config.success_threshold:
                    self._change_state(CircuitState.CLOSED)

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.stats.record_failure()

            if self._state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.config.failure_threshold:
                    self._change_state(CircuitState.OPEN)
                    self._opened_at = datetime.now(timezone.utc)

            elif self._state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
                self._opened_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._half_open_calls = 0
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")

    def force_open(self) -> None:
        """Manually force the circuit to open state."""
        self._change_state(CircuitState.OPEN)
        self._opened_at = datetime.now(timezone.utc)
        logger.warning(f"Circuit breaker '{self.name}' manually forced open")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "retry_after": self._get_retry_after() if self._state == CircuitState.OPEN else 0,
            "half_open_calls": self._half_open_calls,
            "stats": self.stats.to_dict(),
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management and monitoring of all circuit breakers.
    """

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """
        Get existing or create new circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration for new breaker

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def remove(self, name: str) -> None:
        """Remove circuit breaker."""
        self._breakers.pop(name, None)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuits."""
        return [name for name, breaker in self._breakers.items() if breaker.is_open]


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable:
    """
    Decorator for adding circuit breaker to functions.

    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration

    Returns:
        Decorated function

    Example:
        >>> @with_circuit_breaker("binance_api")
        ... async def fetch_ticker(symbol: str):
        ...     return await api.get_ticker(symbol)
    """
    breaker = circuit_registry.get_or_create(name, config)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await breaker.call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # For sync functions, we need to run in event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(breaker.call(func, *args, **kwargs))
            return sync_wrapper

    return decorator
