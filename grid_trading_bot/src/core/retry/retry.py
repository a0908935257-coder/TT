"""
Retry Mechanism.

Provides retry functionality with exponential backoff for fault-tolerant operations.
"""

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from src.core import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    FIXED = "fixed"                    # Fixed delay between retries
    LINEAR = "linear"                  # Linearly increasing delay
    EXPONENTIAL = "exponential"        # Exponentially increasing delay
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential with random jitter


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        strategy: Backoff strategy to use
        jitter_factor: Random jitter factor (0.0 - 1.0)
        retryable_exceptions: Tuple of exceptions that trigger retry
        non_retryable_exceptions: Tuple of exceptions that should not retry
        on_retry: Callback function called on each retry
        on_failure: Callback function called on final failure
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    jitter_factor: float = 0.25
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_failure: Optional[Callable[[int, Exception], None]] = None

    # Multiplier for exponential backoff
    exponential_base: float = 2.0

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt number.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay

        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt

        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        elif self.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            base_delay = self.base_delay * (self.exponential_base ** (attempt - 1))
            jitter = base_delay * self.jitter_factor * random.random()
            delay = base_delay + jitter

        else:
            delay = self.base_delay

        return min(delay, self.max_delay)

    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if the exception should trigger a retry.

        Args:
            exception: The exception that occurred

        Returns:
            True if should retry
        """
        # Check non-retryable first
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # Check retryable
        return isinstance(exception, self.retryable_exceptions)


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_delay: float = 0.0
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_attempt(
        self,
        attempt: int,
        success: bool,
        delay: float = 0.0,
        exception: Optional[Exception] = None,
    ) -> None:
        """Record an attempt."""
        self.attempt_history.append({
            "attempt": attempt,
            "success": success,
            "delay": delay,
            "exception": str(exception) if exception else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> RetryResult:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        RetryResult with outcome

    Example:
        >>> result = await retry_async(
        ...     fetch_data,
        ...     "BTCUSDT",
        ...     config=RetryConfig(max_attempts=5)
        ... )
        >>> if result.success:
        ...     print(result.result)
    """
    config = config or RetryConfig()
    retry_result = RetryResult(success=False)

    for attempt in range(1, config.max_attempts + 1):
        retry_result.attempts = attempt

        try:
            result = await func(*args, **kwargs)
            retry_result.success = True
            retry_result.result = result
            retry_result.add_attempt(attempt, success=True)
            return retry_result

        except Exception as e:
            retry_result.exception = e

            # Check if we should retry
            if not config.should_retry(e):
                logger.warning(
                    f"Non-retryable exception on attempt {attempt}: {type(e).__name__}: {e}"
                )
                retry_result.add_attempt(attempt, success=False, exception=e)
                break

            # Check if we have attempts left
            if attempt >= config.max_attempts:
                logger.error(
                    f"Max attempts ({config.max_attempts}) reached. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                retry_result.add_attempt(attempt, success=False, exception=e)
                if config.on_failure:
                    config.on_failure(attempt, e)
                break

            # Calculate delay
            delay = config.calculate_delay(attempt)
            retry_result.total_delay += delay

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            retry_result.add_attempt(attempt, success=False, delay=delay, exception=e)

            # Call on_retry callback
            if config.on_retry:
                config.on_retry(attempt, e, delay)

            # Wait before retry
            await asyncio.sleep(delay)

    return retry_result


def retry_sync(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> RetryResult:
    """
    Execute a sync function with retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        RetryResult with outcome
    """
    config = config or RetryConfig()
    retry_result = RetryResult(success=False)

    for attempt in range(1, config.max_attempts + 1):
        retry_result.attempts = attempt

        try:
            result = func(*args, **kwargs)
            retry_result.success = True
            retry_result.result = result
            retry_result.add_attempt(attempt, success=True)
            return retry_result

        except Exception as e:
            retry_result.exception = e

            if not config.should_retry(e):
                logger.warning(
                    f"Non-retryable exception on attempt {attempt}: {type(e).__name__}: {e}"
                )
                retry_result.add_attempt(attempt, success=False, exception=e)
                break

            if attempt >= config.max_attempts:
                logger.error(
                    f"Max attempts ({config.max_attempts}) reached. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                retry_result.add_attempt(attempt, success=False, exception=e)
                if config.on_failure:
                    config.on_failure(attempt, e)
                break

            delay = config.calculate_delay(attempt)
            retry_result.total_delay += delay

            logger.warning(
                f"Attempt {attempt}/{config.max_attempts} failed: {type(e).__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            retry_result.add_attempt(attempt, success=False, delay=delay, exception=e)

            if config.on_retry:
                config.on_retry(attempt, e, delay)

            time.sleep(delay)

    return retry_result


def with_retry(
    config: Optional[RetryConfig] = None,
    **config_kwargs: Any,
) -> Callable:
    """
    Decorator for adding retry logic to functions.

    Can be used with both sync and async functions.

    Args:
        config: RetryConfig instance
        **config_kwargs: Arguments to create RetryConfig

    Returns:
        Decorated function

    Example:
        >>> @with_retry(max_attempts=5, base_delay=2.0)
        ... async def fetch_data(symbol: str):
        ...     return await api.get_ticker(symbol)
    """
    if config is None:
        config = RetryConfig(**config_kwargs)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await retry_async(func, *args, config=config, **kwargs)
                if result.success:
                    return result.result
                raise result.exception or RuntimeError("Retry failed")
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = retry_sync(func, *args, config=config, **kwargs)
                if result.success:
                    return result.result
                raise result.exception or RuntimeError("Retry failed")
            return sync_wrapper

    return decorator


# Common retry configurations
RETRY_API_DEFAULT = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
)

RETRY_API_AGGRESSIVE = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
)

RETRY_NETWORK = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=120.0,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
)

RETRY_CRITICAL = RetryConfig(
    max_attempts=10,
    base_delay=1.0,
    max_delay=300.0,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
)
