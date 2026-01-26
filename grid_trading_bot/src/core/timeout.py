"""
Unified Timeout Mechanism.

Provides centralized timeout handling for all async operations.
Prevents system hangs from unresponsive operations.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """Custom timeout error with operation context."""

    def __init__(self, operation: str, timeout: float, message: Optional[str] = None):
        self.operation = operation
        self.timeout = timeout
        self.message = message or f"Operation '{operation}' timed out after {timeout}s"
        super().__init__(self.message)


@dataclass
class TimeoutConfig:
    """
    Centralized timeout configuration.

    Attributes:
        default: Default timeout for general operations (seconds)
        command_execution: Timeout for bot commands (start/stop/pause/resume)
        fund_allocation: Timeout for fund allocation operations
        bot_notification: Timeout for notifying individual bots
        health_check: Timeout for health check operations
        database: Timeout for database operations
    """

    default: float = 30.0
    command_execution: float = 60.0
    fund_allocation: float = 15.0
    bot_notification: float = 10.0
    health_check: float = 10.0
    database: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeoutConfig":
        """Create from dictionary."""
        return cls(
            default=float(data.get("default", 30.0)),
            command_execution=float(data.get("command_execution", 60.0)),
            fund_allocation=float(data.get("fund_allocation", 15.0)),
            bot_notification=float(data.get("bot_notification", 10.0)),
            health_check=float(data.get("health_check", 10.0)),
            database=float(data.get("database", 30.0)),
        )


# Global timeout config instance
_timeout_config: Optional[TimeoutConfig] = None


def get_timeout_config() -> TimeoutConfig:
    """Get global timeout configuration."""
    global _timeout_config
    if _timeout_config is None:
        _timeout_config = TimeoutConfig()
    return _timeout_config


def set_timeout_config(config: TimeoutConfig) -> None:
    """Set global timeout configuration."""
    global _timeout_config
    _timeout_config = config


async def with_timeout(
    coro: Any,
    timeout: float,
    operation_name: str,
    on_timeout: Optional[Callable[[], Any]] = None,
    default_result: Optional[T] = None,
    raise_on_timeout: bool = True,
) -> T:
    """
    Execute a coroutine with timeout protection.

    Provides unified timeout handling with:
    - Configurable timeout duration
    - Optional callback on timeout
    - Optional default result instead of raising
    - Detailed logging

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        operation_name: Name for logging/error messages
        on_timeout: Optional callback to execute on timeout
        default_result: Optional default result if timeout (requires raise_on_timeout=False)
        raise_on_timeout: Whether to raise TimeoutError or return default_result

    Returns:
        Result of the coroutine or default_result on timeout

    Raises:
        TimeoutError: If operation times out and raise_on_timeout is True

    Example:
        >>> result = await with_timeout(
        ...     bot.start(),
        ...     timeout=60.0,
        ...     operation_name="bot_start",
        ... )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)

    except asyncio.TimeoutError:
        logger.error(
            f"Timeout: {operation_name} exceeded {timeout}s"
        )

        # Execute timeout callback if provided
        if on_timeout:
            try:
                result = on_timeout()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Timeout callback failed for {operation_name}: {e}")

        if raise_on_timeout:
            raise TimeoutError(operation_name, timeout)

        return default_result  # type: ignore


async def with_timeout_and_cancel(
    coro: Any,
    timeout: float,
    operation_name: str,
    cancel_callback: Optional[Callable[[], Any]] = None,
) -> T:
    """
    Execute a coroutine with timeout, ensuring proper cancellation.

    Similar to with_timeout but ensures the underlying task is properly
    cancelled when timeout occurs.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        operation_name: Name for logging/error messages
        cancel_callback: Callback to execute after cancellation

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If operation times out
    """
    task = asyncio.create_task(coro)

    try:
        return await asyncio.wait_for(task, timeout=timeout)

    except asyncio.TimeoutError:
        logger.error(f"Timeout: {operation_name} exceeded {timeout}s, cancelling task")

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Execute cancel callback if provided
        if cancel_callback:
            try:
                result = cancel_callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Cancel callback failed for {operation_name}: {e}")

        raise TimeoutError(operation_name, timeout)


async def with_retry_timeout(
    coro_factory: Callable[[], Any],
    timeout: float,
    operation_name: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> T:
    """
    Execute a coroutine with timeout and automatic retry on timeout.

    Args:
        coro_factory: Factory function that returns a new coroutine each call
        timeout: Timeout in seconds per attempt
        operation_name: Name for logging/error messages
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If all retry attempts time out

    Example:
        >>> result = await with_retry_timeout(
        ...     lambda: exchange.get_account(),
        ...     timeout=10.0,
        ...     operation_name="get_account",
        ...     max_retries=3,
        ... )
    """
    last_error: Optional[TimeoutError] = None

    for attempt in range(max_retries):
        try:
            return await with_timeout(
                coro_factory(),
                timeout=timeout,
                operation_name=f"{operation_name}_attempt_{attempt + 1}",
            )
        except TimeoutError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"{operation_name} timeout on attempt {attempt + 1}/{max_retries}, "
                    f"retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

    # All retries exhausted
    raise TimeoutError(
        operation_name,
        timeout,
        f"Operation '{operation_name}' failed after {max_retries} attempts",
    )


class TimeoutContext:
    """
    Context manager for operations with timeout.

    Provides a context manager interface for timeout operations
    with automatic cleanup.

    Example:
        >>> async with TimeoutContext(30.0, "database_query") as ctx:
        ...     result = await db.query(...)
    """

    def __init__(
        self,
        timeout: float,
        operation_name: str,
        on_timeout: Optional[Callable[[], Any]] = None,
    ):
        self.timeout = timeout
        self.operation_name = operation_name
        self.on_timeout = on_timeout
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "TimeoutContext":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is asyncio.TimeoutError:
            logger.error(
                f"Timeout: {self.operation_name} exceeded {self.timeout}s"
            )
            if self.on_timeout:
                try:
                    result = self.on_timeout()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(
                        f"Timeout callback failed for {self.operation_name}: {e}"
                    )
            raise TimeoutError(self.operation_name, self.timeout)
        return False


# Convenience functions for common timeout operations
async def command_timeout(coro: Any, operation_name: str) -> T:
    """Execute with command_execution timeout."""
    config = get_timeout_config()
    return await with_timeout(coro, config.command_execution, operation_name)


async def fund_timeout(coro: Any, operation_name: str) -> T:
    """Execute with fund_allocation timeout."""
    config = get_timeout_config()
    return await with_timeout(coro, config.fund_allocation, operation_name)


async def notification_timeout(
    coro: Any,
    operation_name: str,
    default_result: Optional[T] = None,
    raise_on_timeout: bool = True,
) -> T:
    """
    Execute with bot_notification timeout, optionally returning default on timeout.

    Args:
        coro: Coroutine to execute
        operation_name: Name for logging/error messages
        default_result: Optional default result if timeout
        raise_on_timeout: Whether to raise TimeoutError on timeout.
                         If False and default_result is provided, returns default_result.
                         If True (default), raises TimeoutError on timeout.

    Returns:
        Result of the coroutine or default_result on timeout (if not raising)
    """
    config = get_timeout_config()
    # Only suppress exception if both raise_on_timeout=False AND default_result is provided
    should_raise = raise_on_timeout and default_result is None
    return await with_timeout(
        coro,
        config.bot_notification,
        operation_name,
        default_result=default_result,
        raise_on_timeout=should_raise,
    )
