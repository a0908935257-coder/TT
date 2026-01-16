"""
Utility functions for Grid Trading Bot.

Includes time conversion, decimal operations, string utilities, and retry decorator.
"""

import functools
import random
import string
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Callable, TypeVar

T = TypeVar("T")


# =============================================================================
# Time-related functions
# =============================================================================


def timestamp_to_datetime(ts: int, unit: str = "ms") -> datetime:
    """
    Convert timestamp to datetime (UTC).

    Args:
        ts: Timestamp value
        unit: "ms" for milliseconds, "s" for seconds

    Returns:
        UTC datetime object

    Example:
        >>> timestamp_to_datetime(1704067200000)
        datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    if unit == "ms":
        ts = ts / 1000
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime, unit: str = "ms") -> int:
    """
    Convert datetime to timestamp.

    Args:
        dt: Datetime object (assumes UTC if no timezone)
        unit: "ms" for milliseconds, "s" for seconds

    Returns:
        Integer timestamp

    Example:
        >>> dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        >>> datetime_to_timestamp(dt)
        1704067200000
    """
    # If naive datetime, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    ts = dt.timestamp()
    if unit == "ms":
        return int(ts * 1000)
    return int(ts)


def now_timestamp(unit: str = "ms") -> int:
    """
    Get current timestamp.

    Args:
        unit: "ms" for milliseconds, "s" for seconds

    Returns:
        Current timestamp as integer

    Example:
        >>> ts = now_timestamp()  # e.g., 1704067200000
    """
    ts = time.time()
    if unit == "ms":
        return int(ts * 1000)
    return int(ts)


# =============================================================================
# Numeric functions
# =============================================================================


def round_decimal(
    value: Decimal,
    precision: int,
    rounding: str = ROUND_DOWN,
) -> Decimal:
    """
    Round a Decimal to specified precision.

    Args:
        value: Decimal value to round
        precision: Number of decimal places
        rounding: Rounding mode (default ROUND_DOWN)

    Returns:
        Rounded Decimal

    Example:
        >>> round_decimal(Decimal("123.456"), 2)
        Decimal('123.45')
    """
    if precision < 0:
        precision = 0

    quantize_str = "1." + "0" * precision if precision > 0 else "1"
    return value.quantize(Decimal(quantize_str), rounding=rounding)


def round_to_tick(value: Decimal, tick_size: Decimal) -> Decimal:
    """
    Round value to the nearest tick size.

    Args:
        value: Value to round
        tick_size: Minimum price/quantity increment (e.g., 0.01)

    Returns:
        Value rounded down to tick size

    Example:
        >>> round_to_tick(Decimal("123.456"), Decimal("0.01"))
        Decimal('123.45')
        >>> round_to_tick(Decimal("123.456"), Decimal("0.05"))
        Decimal('123.45')
    """
    if tick_size <= 0:
        return value

    # Calculate how many ticks fit into the value
    ticks = value / tick_size
    # Round down to whole number of ticks
    whole_ticks = ticks.to_integral_value(rounding=ROUND_DOWN)
    # Multiply back
    return whole_ticks * tick_size


def format_quantity(quantity: Decimal, precision: int) -> str:
    """
    Format quantity with specified precision, removing trailing zeros.

    Args:
        quantity: Quantity to format
        precision: Maximum decimal places

    Returns:
        Formatted string without trailing zeros

    Example:
        >>> format_quantity(Decimal("1.50000"), 5)
        '1.5'
        >>> format_quantity(Decimal("1.00000"), 5)
        '1'
    """
    rounded = round_decimal(quantity, precision)
    # Normalize removes trailing zeros
    normalized = rounded.normalize()

    # Handle scientific notation for very small numbers
    result = str(normalized)
    if "E" in result:
        result = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")

    return result


def format_price(price: Decimal, precision: int) -> str:
    """
    Format price with specified precision, removing trailing zeros.

    Args:
        price: Price to format
        precision: Maximum decimal places

    Returns:
        Formatted price string

    Example:
        >>> format_price(Decimal("50000.00"), 2)
        '50000'
    """
    return format_quantity(price, precision)


def calculate_percentage(part: Decimal, whole: Decimal) -> Decimal:
    """
    Calculate percentage (part/whole * 100).

    Args:
        part: Numerator
        whole: Denominator

    Returns:
        Percentage as Decimal

    Example:
        >>> calculate_percentage(Decimal("25"), Decimal("100"))
        Decimal('25')
    """
    if whole == 0:
        return Decimal("0")
    return (part / whole) * Decimal("100")


# =============================================================================
# String functions
# =============================================================================


def generate_client_order_id(prefix: str = "BOT") -> str:
    """
    Generate a unique client order ID.

    Format: PREFIX_TIMESTAMP_RANDOM

    Args:
        prefix: Prefix for the order ID

    Returns:
        Unique order ID string

    Example:
        >>> generate_client_order_id("GRID")
        'GRID_1704067200000_A3B2C1'
    """
    timestamp = now_timestamp()
    random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{random_suffix}"


def mask_secret(secret: str, show_chars: int = 4) -> str:
    """
    Mask a sensitive string, showing only first and last N characters.

    Args:
        secret: String to mask
        show_chars: Number of characters to show at start and end

    Returns:
        Masked string

    Example:
        >>> mask_secret("abcdefghijklmnop", 4)
        'abcd********mnop'
        >>> mask_secret("short", 4)
        '***'
    """
    if len(secret) <= show_chars * 2:
        return "*" * min(len(secret), 3)

    start = secret[:show_chars]
    end = secret[-show_chars:]
    middle_len = len(secret) - show_chars * 2
    return f"{start}{'*' * middle_len}{end}"


# =============================================================================
# Retry decorator
# =============================================================================


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exceptions: Tuple of exception types to catch
        backoff: Multiplier for delay after each retry

    Returns:
        Decorated function

    Example:
        >>> @retry(max_retries=3, delay=1.0)
        ... def fetch_data():
        ...     # May fail occasionally
        ...     pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff

            # All retries exhausted
            raise last_exception  # type: ignore

        return wrapper
    return decorator


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
    backoff: float = 2.0,
) -> Callable:
    """
    Async retry decorator with exponential backoff.

    Same as retry() but for async functions.
    """
    import asyncio

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception  # type: ignore

        return wrapper
    return decorator
