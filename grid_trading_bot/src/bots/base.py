"""
Base Bot Abstract Class.

Provides unified interface for all trading bots, enabling Master to manage
different bot types consistently.

Design Pattern: Template Method
- Public methods (start, stop, pause, resume) define the algorithm skeleton
- Abstract methods (_do_start, _do_stop, etc.) let subclasses implement specifics
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.models import MarketType

from src.core import get_logger
from src.core.models import Kline
from src.master.models import BotState

logger = get_logger(__name__)


# Data protection constants
DEFAULT_STALE_THRESHOLD_SECONDS = 120  # 2 minutes for data freshness
DEFAULT_MAX_PRICE_CHANGE_PCT = Decimal("0.10")  # 10% max single-candle change
DEFAULT_MAX_GAP_MULTIPLIER = 3  # Allow up to 3x expected interval gap

# Indicator validation constants
MIN_INDICATOR_DATA_POINTS = 5  # Minimum data points before trusting indicators
MAX_REASONABLE_PRICE = Decimal("10000000")  # $10M max price (sanity check)
MIN_REASONABLE_PRICE = Decimal("0.00000001")  # Minimum price (8 decimals)

# Kline deduplication
MAX_PROCESSED_KLINES_HISTORY = 10  # Keep track of last N processed kline close_times


class InvalidStateError(Exception):
    """Raised when an operation is invalid for the current bot state."""

    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class BotStats:
    """
    Bot performance statistics.

    Tracks trading performance metrics for reporting and monitoring.
    """

    total_trades: int = 0
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    today_trades: int = 0
    today_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    start_time: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None

    def reset_daily(self) -> None:
        """Reset daily statistics."""
        self.today_trades = 0
        self.today_profit = Decimal("0")

    def record_trade(self, profit: Decimal, fee: Decimal) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        self.total_profit += profit
        self.total_fees += fee
        self.today_trades += 1
        self.today_profit += profit
        self.last_trade_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "total_profit": str(self.total_profit),
            "total_fees": str(self.total_fees),
            "today_trades": self.today_trades,
            "today_profit": str(self.today_profit),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
        }


class BaseBot(ABC):
    """
    Abstract base class for all trading bots.

    All bot implementations (Grid, DCA, Arbitrage, etc.) must inherit from
    this class and implement the required abstract methods.

    Features:
    - Unified lifecycle management (start, stop, pause, resume)
    - Template method pattern for consistent behavior
    - Heartbeat mechanism for Master monitoring
    - Health check framework
    - Standard statistics tracking

    Example:
        class GridBot(BaseBot):
            @property
            def bot_type(self) -> str:
                return "grid"

            @property
            def symbol(self) -> str:
                return self._config.symbol

            async def _do_start(self) -> None:
                # Initialize grid, place orders
                pass

            async def _do_stop(self, clear_position: bool = False) -> None:
                # Cancel orders, save state
                pass
    """

    def __init__(
        self,
        bot_id: str,
        config: Any,
        exchange: Any,
        data_manager: Any,
        notifier: Any,
        heartbeat_callback: Optional[Callable] = None,
    ):
        """
        Initialize BaseBot.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration (type depends on bot implementation)
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance
            heartbeat_callback: Optional callback for sending heartbeats to Master
        """
        self._bot_id = bot_id
        self._config = config
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier
        self._heartbeat_callback = heartbeat_callback

        self._state: BotState = BotState.REGISTERED
        self._stats: BotStats = BotStats()
        self._running: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._error_message: Optional[str] = None

        # Track fire-and-forget notification tasks to prevent resource leaks
        # NOTE: asyncio single-thread safe, no lock needed for set add/discard
        self._notification_tasks: set[asyncio.Task] = set()

        # Position reconciliation (initialized here to avoid hasattr checks)
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._last_known_position: Optional[Dict[str, Any]] = None
        self._position_mismatch_count: int = 0

        # Expected position side for Hedge Mode filtering (set by subclass)
        # When set to "LONG" or "SHORT", position sync/reconciliation will only
        # consider positions matching this side, ignoring others.
        # This prevents multi-bot conflicts when trading same symbol in Hedge Mode.
        self._expected_position_side: Optional[str] = None

    # =========================================================================
    # Read-only Properties
    # =========================================================================

    @property
    def bot_id(self) -> str:
        """Get bot ID."""
        return self._bot_id

    @property
    def state(self) -> BotState:
        """Get current bot state."""
        return self._state

    @property
    def config(self) -> Any:
        """Get bot configuration."""
        return self._config

    @property
    def stats(self) -> BotStats:
        """Get bot statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running and self._state == BotState.RUNNING

    @property
    def error_message(self) -> Optional[str]:
        """Get error message if in error state."""
        return self._error_message

    # =========================================================================
    # Abstract Properties (Subclass Must Implement)
    # =========================================================================

    @property
    @abstractmethod
    def bot_type(self) -> str:
        """
        Return bot type identifier.

        Examples: "grid", "dca", "arbitrage"
        """
        pass

    @property
    @abstractmethod
    def symbol(self) -> str:
        """
        Return trading symbol.

        Examples: "BTCUSDT", "ETHUSDT"
        """
        pass

    # =========================================================================
    # Notification Task Management
    # =========================================================================

    def _create_notification_task(self, coro) -> asyncio.Task:
        """
        Create a tracked notification task.

        Fire-and-forget tasks are tracked to:
        1. Prevent garbage collection before completion
        2. Log exceptions that would otherwise be silent
        3. Allow cleanup during bot stop

        Args:
            coro: Coroutine to execute

        Returns:
            The created task
        """
        task = asyncio.create_task(coro)
        self._notification_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._notification_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.warning(f"[{self._bot_id}] Notification task failed: {exc}")

        task.add_done_callback(_on_done)
        return task

    async def _cleanup_notification_tasks(self) -> None:
        """Cancel and cleanup all pending notification tasks."""
        if not self._notification_tasks:
            return

        for task in list(self._notification_tasks):
            if not task.done():
                task.cancel()

        if self._notification_tasks:
            await asyncio.gather(*self._notification_tasks, return_exceptions=True)
        self._notification_tasks.clear()

    def _create_background_task(self, coro, name: str) -> asyncio.Task:
        """
        Create a tracked background task with proper exception handling.

        Unlike _create_notification_task (fire-and-forget notifications),
        this is for general-purpose background coroutines that need
        done_callback exception logging to prevent silent failures.

        Args:
            coro: Coroutine to execute
            name: Descriptive task name for logging

        Returns:
            The created task
        """
        task = asyncio.create_task(coro, name=f"{self._bot_id}:{name}")
        self._notification_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._notification_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.error(
                    f"[{self._bot_id}] Background task '{name}' failed: {exc}",
                    exc_info=exc,
                )

        task.add_done_callback(_on_done)
        return task

    # =========================================================================
    # Kline Validation (Prevents Backtest vs Live Data Inconsistency)
    # =========================================================================

    def _validate_kline_closed(
        self,
        kline: Kline,
        require_closed: bool = True,
    ) -> bool:
        """
        Validate that a kline is closed before processing.

        This prevents the common issue where:
        - Backtest uses completed (closed) klines
        - Live trading receives incomplete klines from WebSocket
        - Strategy makes decisions on incomplete data

        Args:
            kline: The kline to validate
            require_closed: If True, reject unclosed klines

        Returns:
            True if kline is valid for processing
        """
        if require_closed and not kline.is_closed:
            logger.debug(
                f"[{self._bot_id}] Skipping unclosed kline: "
                f"{kline.symbol} {kline.open_time}"
            )
            return False
        return True

    def _should_process_kline(
        self,
        kline: Kline,
        require_closed: bool = True,
        check_symbol: bool = True,
    ) -> bool:
        """
        Comprehensive kline validation before processing.

        Checks:
        1. Kline is closed (if required)
        2. Symbol matches bot's trading symbol
        3. Bot is in running state

        Args:
            kline: The kline to validate
            require_closed: If True, reject unclosed klines
            check_symbol: If True, verify symbol matches

        Returns:
            True if kline should be processed
        """
        # Check bot state
        if not self.is_running:
            return False

        # Check kline closed status
        if not self._validate_kline_closed(kline, require_closed):
            return False

        # Check symbol match
        if check_symbol and kline.symbol != self.symbol:
            logger.warning(
                f"[{self._bot_id}] Kline symbol mismatch: "
                f"expected {self.symbol}, got {kline.symbol}"
            )
            return False

        # F-5: Kline deduplication - check if already processed
        if hasattr(self, '_processed_kline_times'):
            if kline.close_time in self._processed_kline_times:
                logger.debug(
                    f"[{self._bot_id}] Skipping duplicate kline: "
                    f"close_time={kline.close_time} already processed"
                )
                return False

        return True

    # =========================================================================
    # Data Protection (Freshness, Integrity, Anomaly Detection)
    # =========================================================================

    def _validate_kline_freshness(
        self,
        kline: Kline,
        max_stale_seconds: int = DEFAULT_STALE_THRESHOLD_SECONDS,
    ) -> bool:
        """
        Validate that kline data is not stale.

        Protects against:
        - Network delays causing outdated data
        - WebSocket reconnection gaps
        - Exchange API delays

        Args:
            kline: The kline to validate
            max_stale_seconds: Maximum allowed data age in seconds

        Returns:
            True if data is fresh enough to use
        """
        now = datetime.now(timezone.utc)
        kline_age = (now - kline.close_time).total_seconds()

        if kline_age > max_stale_seconds:
            logger.warning(
                f"[{self._bot_id}] Stale kline detected: "
                f"age={kline_age:.1f}s > threshold={max_stale_seconds}s, "
                f"close_time={kline.close_time}"
            )
            return False

        return True

    def _validate_kline_integrity(
        self,
        kline: Kline,
        prev_kline: Optional[Kline] = None,
        max_price_change_pct: Decimal = DEFAULT_MAX_PRICE_CHANGE_PCT,
    ) -> bool:
        """
        Validate kline data integrity.

        Checks:
        1. OHLC relationships (high >= low, high >= open/close, low <= open/close)
        2. Price sanity (not zero or negative)
        3. Volume sanity (not negative)
        4. Price change limits (detect erroneous spikes)

        Args:
            kline: The kline to validate
            prev_kline: Previous kline for change comparison
            max_price_change_pct: Maximum allowed price change (default 10%)

        Returns:
            True if data integrity is valid
        """
        # Check OHLC relationships
        if kline.high < kline.low:
            logger.error(
                f"[{self._bot_id}] Invalid OHLC: high ({kline.high}) < low ({kline.low})"
            )
            return False

        if kline.high < kline.open or kline.high < kline.close:
            logger.error(
                f"[{self._bot_id}] Invalid OHLC: high ({kline.high}) < open/close"
            )
            return False

        if kline.low > kline.open or kline.low > kline.close:
            logger.error(
                f"[{self._bot_id}] Invalid OHLC: low ({kline.low}) > open/close"
            )
            return False

        # Check price sanity
        if kline.close <= 0 or kline.open <= 0:
            logger.error(
                f"[{self._bot_id}] Invalid price: close={kline.close}, open={kline.open}"
            )
            return False

        # Check volume sanity
        if kline.volume < 0:
            logger.error(f"[{self._bot_id}] Invalid volume: {kline.volume}")
            return False

        # Check price change limits (if previous kline available)
        if prev_kline and prev_kline.close > 0:
            price_change = abs(kline.close - prev_kline.close) / prev_kline.close
            if price_change > max_price_change_pct:
                logger.warning(
                    f"[{self._bot_id}] Extreme price change detected: "
                    f"{float(price_change)*100:.2f}% > {float(max_price_change_pct)*100:.0f}%, "
                    f"prev={prev_kline.close}, curr={kline.close}"
                )
                # Note: We warn but still return True - trading decision is up to the bot
                # This is informational; flash crash detection is separate

        return True

    def _check_data_gap(
        self,
        kline: Kline,
        prev_kline: Optional[Kline],
        expected_interval_seconds: int,
        max_gap_multiplier: int = DEFAULT_MAX_GAP_MULTIPLIER,
    ) -> bool:
        """
        Check for data gaps between klines.

        Protects against:
        - Missing data during WebSocket disconnection
        - Data gaps causing indicator miscalculation

        Args:
            kline: Current kline
            prev_kline: Previous kline
            expected_interval_seconds: Expected time between klines (e.g., 900 for 15m)
            max_gap_multiplier: Maximum allowed gap (multiplier of expected interval)

        Returns:
            True if no significant gap, False if gap detected
        """
        if prev_kline is None:
            return True

        actual_interval = (kline.close_time - prev_kline.close_time).total_seconds()
        max_allowed_gap = expected_interval_seconds * max_gap_multiplier

        if actual_interval > max_allowed_gap:
            missing_candles = int(actual_interval / expected_interval_seconds) - 1
            logger.warning(
                f"[{self._bot_id}] Data gap detected: {missing_candles} missing candles, "
                f"gap={actual_interval:.0f}s, expected={expected_interval_seconds}s"
            )
            return False

        return True

    # =========================================================================
    # Indicator Validation (Boundary Conditions)
    # =========================================================================

    def _validate_indicator_value(
        self,
        value: Decimal,
        name: str,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None,
        allow_zero: bool = True,
        allow_negative: bool = False,
    ) -> bool:
        """
        Validate an indicator value for boundary conditions.

        Protects against:
        - NaN/Infinity values
        - Division by zero results
        - Out of bounds calculations
        - Negative values where inappropriate

        Args:
            value: The indicator value to validate
            name: Name of the indicator (for logging)
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            allow_zero: Whether zero is acceptable
            allow_negative: Whether negative values are acceptable

        Returns:
            True if value is valid
        """
        import math

        # Check for NaN/Infinity (convert to float for math functions)
        try:
            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                logger.error(
                    f"[{self._bot_id}] Invalid {name}: NaN or Infinity detected"
                )
                return False
        except (ValueError, TypeError, OverflowError) as e:
            logger.error(f"[{self._bot_id}] Invalid {name}: Cannot convert to float - {e}")
            return False

        # Check zero
        if not allow_zero and value == Decimal("0"):
            logger.warning(f"[{self._bot_id}] Invalid {name}: Zero not allowed")
            return False

        # Check negative
        if not allow_negative and value < Decimal("0"):
            logger.warning(f"[{self._bot_id}] Invalid {name}: Negative value {value}")
            return False

        # Check min bound
        if min_value is not None and value < min_value:
            logger.warning(
                f"[{self._bot_id}] {name} below minimum: {value} < {min_value}"
            )
            return False

        # Check max bound
        if max_value is not None and value > max_value:
            logger.warning(
                f"[{self._bot_id}] {name} above maximum: {value} > {max_value}"
            )
            return False

        return True

    def _validate_price(self, price: Decimal, context: str = "price") -> bool:
        """
        Validate a price value.

        Args:
            price: The price to validate
            context: Description of the price (for logging)

        Returns:
            True if price is valid
        """
        return self._validate_indicator_value(
            value=price,
            name=context,
            min_value=MIN_REASONABLE_PRICE,
            max_value=MAX_REASONABLE_PRICE,
            allow_zero=False,
            allow_negative=False,
        )

    def _validate_quantity(self, quantity: Decimal, context: str = "quantity") -> bool:
        """
        Validate an order quantity.

        Args:
            quantity: The quantity to validate
            context: Description of the quantity (for logging)

        Returns:
            True if quantity is valid
        """
        return self._validate_indicator_value(
            value=quantity,
            name=context,
            min_value=Decimal("0"),
            max_value=Decimal("1000000"),  # 1M units max
            allow_zero=False,
            allow_negative=False,
        )

    def _safe_divide(
        self,
        numerator: Decimal,
        denominator: Decimal,
        default: Decimal = Decimal("0"),
        context: str = "division",
    ) -> Decimal:
        """
        Safely divide two Decimals with zero protection.

        Args:
            numerator: The numerator
            denominator: The denominator
            default: Default value if division fails
            context: Description of the calculation (for logging)

        Returns:
            Result of division, or default if denominator is zero
        """
        if denominator == Decimal("0"):
            logger.debug(
                f"[{self._bot_id}] Division by zero in {context}, returning {default}"
            )
            return default

        try:
            result = numerator / denominator
            # Validate result
            if not self._validate_indicator_value(result, context, allow_negative=True):
                return default
            return result
        except Exception as e:
            logger.warning(f"[{self._bot_id}] Division error in {context}: {e}")
            return default

    def _validate_sufficient_data(
        self,
        data_length: int,
        required: int,
        context: str = "indicator",
    ) -> bool:
        """
        Validate that sufficient data exists for indicator calculation.

        Args:
            data_length: Current data length
            required: Required minimum data points
            context: Name of the calculation (for logging)

        Returns:
            True if sufficient data exists
        """
        if data_length < required:
            logger.debug(
                f"[{self._bot_id}] Insufficient data for {context}: "
                f"{data_length} < {required} required"
            )
            return False
        return True

    def _parse_interval_to_seconds(self, interval: str) -> int:
        """
        Parse timeframe string to seconds.

        Args:
            interval: Timeframe string (e.g., "1m", "5m", "15m", "1h", "4h")

        Returns:
            Interval in seconds
        """
        multipliers = {
            "m": 60,
            "h": 3600,
            "d": 86400,
        }
        unit = interval[-1].lower()
        value = int(interval[:-1])
        return value * multipliers.get(unit, 60)

    # =========================================================================
    # Data Connection Health
    # =========================================================================

    def _init_data_health_tracking(self) -> None:
        """Initialize data health tracking attributes."""
        self._last_kline_time: Optional[datetime] = None
        self._consecutive_stale_count: int = 0
        self._data_healthy: bool = True
        self._data_gap_detected: bool = False
        # F-5: Kline deduplication
        self._processed_kline_times: set[datetime] = set()

    def _update_data_health(
        self,
        kline: Kline,
        stale_threshold: int = DEFAULT_STALE_THRESHOLD_SECONDS,
    ) -> None:
        """
        Update data health status based on received kline.

        Args:
            kline: The received kline
            stale_threshold: Threshold for stale data in seconds
        """
        now = datetime.now(timezone.utc)
        self._last_kline_time = now

        # Check if data was stale
        kline_age = (now - kline.close_time).total_seconds()
        if kline_age > stale_threshold:
            self._consecutive_stale_count += 1
            if self._consecutive_stale_count >= 3:
                self._data_healthy = False
                logger.warning(
                    f"[{self._bot_id}] Data health degraded: "
                    f"{self._consecutive_stale_count} consecutive stale klines"
                )
        else:
            self._consecutive_stale_count = 0
            self._data_healthy = True

    def _mark_kline_processed(self, kline: Kline) -> None:
        """
        Mark a kline as processed for deduplication.

        Maintains a bounded set of recently processed kline close_times.
        """
        if not hasattr(self, '_processed_kline_times'):
            self._processed_kline_times = set()

        self._processed_kline_times.add(kline.close_time)

        # Limit memory usage - keep only last N entries
        max_history = MAX_PROCESSED_KLINES_HISTORY
        if len(self._processed_kline_times) > max_history * 2:
            sorted_times = sorted(self._processed_kline_times, reverse=True)
            self._processed_kline_times = set(sorted_times[:max_history])
            logger.debug(f"[{self._bot_id}] Trimmed processed kline history to {max_history} entries")

    def _is_data_connection_healthy(
        self,
        max_silence_seconds: int = 180,
    ) -> bool:
        """
        Check if data connection is healthy.

        Args:
            max_silence_seconds: Maximum time without data before unhealthy

        Returns:
            True if connection is healthy
        """
        if not hasattr(self, "_last_kline_time") or self._last_kline_time is None:
            return True  # No data yet, assume healthy

        silence = (datetime.now(timezone.utc) - self._last_kline_time).total_seconds()
        if silence > max_silence_seconds:
            logger.warning(
                f"[{self._bot_id}] No data for {silence:.0f}s - connection may be unhealthy"
            )
            return False

        return getattr(self, "_data_healthy", True)

    async def _handle_data_anomaly(
        self,
        anomaly_type: str,
        severity: str,
        message: str,
    ) -> bool:
        """
        Handle detected data anomaly.

        Args:
            anomaly_type: Type of anomaly (e.g., "stale", "gap", "spike")
            severity: Severity level ("low", "medium", "high", "critical")
            message: Description of the anomaly

        Returns:
            True if trading should continue, False if should pause
        """
        logger.warning(f"[{self._bot_id}] Data anomaly: {anomaly_type} ({severity}) - {message}")

        # Critical anomalies should block trading
        if severity == "critical":
            if self._notifier:
                await self._notifier.send_warning(
                    title=f"{self.bot_type}: Data Anomaly",
                    message=f"Type: {anomaly_type}\n{message}\nTrading paused.",
                )
            return False

        # High severity anomalies should warn but may continue
        if severity == "high":
            if self._notifier:
                await self._notifier.send_warning(
                    title=f"{self.bot_type}: Data Warning",
                    message=f"Type: {anomaly_type}\n{message}",
                )

        return True

    # =========================================================================
    # Time Synchronization Health Check
    # =========================================================================

    async def _check_time_sync_health(
        self,
        max_offset_ms: int = 1000,
        critical_offset_ms: int = 5000,
    ) -> bool:
        """
        Check if exchange time synchronization is healthy.

        Protects against:
        - Order rejection due to timestamp issues
        - Cross-market timing inconsistencies

        Args:
            max_offset_ms: Warning threshold in milliseconds
            critical_offset_ms: Critical threshold (block trading)

        Returns:
            True if time sync is healthy, False if critical drift
        """
        try:
            # Get time offsets from exchange client
            if hasattr(self._exchange, 'get_time_offsets'):
                offsets = self._exchange.get_time_offsets()
                futures_offset = offsets.get('futures', 0)

                if abs(futures_offset) > critical_offset_ms:
                    logger.error(
                        f"[{self._bot_id}] Critical time drift: {futures_offset}ms - "
                        f"orders may be rejected"
                    )
                    # Attempt to resync
                    if hasattr(self._exchange, 'force_time_sync'):
                        await self._exchange.force_time_sync()
                        logger.info(f"[{self._bot_id}] Forced time resync")
                    return False

                if abs(futures_offset) > max_offset_ms:
                    logger.warning(
                        f"[{self._bot_id}] Time drift warning: {futures_offset}ms"
                    )

            return True

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Failed to check time sync: {e}")
            return True  # Don't block trading on check failure

    async def _ensure_time_sync(
        self,
        max_offset_ms: int = 3000,
        retry_count: int = 3,
    ) -> None:
        """
        Ensure time is synchronized with exchange before starting.

        This is called at bot startup to:
        1. Force an immediate time sync with the exchange
        2. Verify the time offset is within acceptable limits
        3. Log warnings if drift is detected

        Args:
            max_offset_ms: Maximum acceptable time offset in milliseconds
            retry_count: Number of sync retry attempts

        Note:
            This method does not block bot startup on failure, but logs
            warnings to alert operators of potential API signature issues.
        """
        try:
            logger.info(f"[{self._bot_id}] Ensuring time synchronization...")

            for attempt in range(retry_count):
                # Force time sync
                if hasattr(self._exchange, 'force_time_sync'):
                    offsets = await self._exchange.force_time_sync()

                    spot_offset = offsets.get('spot', 0)
                    futures_offset = offsets.get('futures', 0)

                    logger.info(
                        f"[{self._bot_id}] Time sync result: "
                        f"spot={spot_offset}ms, futures={futures_offset}ms"
                    )

                    # Check if offset is acceptable
                    max_offset = max(abs(spot_offset), abs(futures_offset))
                    if max_offset <= max_offset_ms:
                        logger.info(
                            f"[{self._bot_id}] Time synchronization OK "
                            f"(offset={max_offset}ms <= {max_offset_ms}ms)"
                        )
                        return

                    # Offset too large - retry
                    if attempt < retry_count - 1:
                        logger.warning(
                            f"[{self._bot_id}] Time offset {max_offset}ms exceeds "
                            f"threshold {max_offset_ms}ms, retry {attempt + 1}/{retry_count}"
                        )
                        await asyncio.sleep(1)  # Wait before retry
                    else:
                        logger.warning(
                            f"[{self._bot_id}] Time offset {max_offset}ms exceeds "
                            f"threshold after {retry_count} attempts. "
                            f"API requests may be rejected due to timestamp issues."
                        )

                        # Send alert notification
                        if self._notifier:
                            try:
                                await self._notifier.send_alert(
                                    title="Time Sync Warning",
                                    message=(
                                        f"Bot {self._bot_id} detected large time offset:\n"
                                        f"• Spot: {spot_offset}ms\n"
                                        f"• Futures: {futures_offset}ms\n"
                                        f"• Threshold: {max_offset_ms}ms\n\n"
                                        f"Orders may be rejected. Please check system time."
                                    ),
                                    level="warning",
                                )
                            except Exception:
                                pass  # Don't fail startup on notification error
                else:
                    logger.debug(
                        f"[{self._bot_id}] Exchange client does not support force_time_sync"
                    )
                    return

        except Exception as e:
            logger.warning(
                f"[{self._bot_id}] Failed to ensure time sync: {e}. "
                f"Continuing with startup..."
            )

    async def _validate_config_on_start(self) -> None:
        """
        Validate bot configuration before starting.

        This validates:
        1. Required parameters are present
        2. Parameter values are within valid ranges
        3. Unit formats are correct (e.g., percentages as decimals)
        4. Configuration checksum for version tracking

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            logger.info(f"[{self._bot_id}] Validating configuration...")

            # Get bot type and config
            bot_type = self.bot_type
            config = self._config

            # Convert dataclass to dict if needed
            if hasattr(config, '__dataclass_fields__'):
                config_dict = {}
                for field_name in config.__dataclass_fields__:
                    config_dict[field_name] = getattr(config, field_name)
            elif hasattr(config, '__dict__'):
                config_dict = config.__dict__.copy()
            else:
                config_dict = dict(config) if config else {}

            # Log configuration checksum for tracking
            try:
                from src.config.validator import compute_config_checksum
                checksum = compute_config_checksum(
                    {k: str(v) for k, v in config_dict.items()}
                )
                logger.info(
                    f"[{self._bot_id}] Config checksum: {checksum} "
                    f"(use for sync verification)"
                )
            except ImportError:
                pass

            # Validate percentage fields are in correct format
            pct_fields = [
                'stop_loss_pct', 'position_size_pct', 'max_position_pct',
                'grid_range_pct', 'daily_loss_limit_pct', 'trailing_stop_pct',
                'rebuild_threshold_pct', 'fallback_range_pct'
            ]

            warnings = []
            for field in pct_fields:
                if field in config_dict:
                    value = config_dict[field]
                    try:
                        from decimal import Decimal
                        decimal_val = Decimal(str(value))
                        if decimal_val > Decimal("1"):
                            warnings.append(
                                f"{field}={value} looks like whole percentage, "
                                f"expected decimal (e.g., 0.05 for 5%)"
                            )
                    except Exception as e:
                        logger.debug(f"Config value conversion error: {e}")

            if warnings:
                for warning in warnings:
                    logger.warning(f"[{self._bot_id}] Config warning: {warning}")

                # Send alert if notifier available
                if self._notifier:
                    try:
                        await self._notifier.send_alert(
                            title="Configuration Warning",
                            message=(
                                f"Bot {self._bot_id} configuration warnings:\n"
                                + "\n".join(f"• {w}" for w in warnings)
                                + "\n\nBot will start, but please verify configuration."
                            ),
                            level="warning",
                        )
                    except Exception as e:
                        logger.debug(f"Failed to send config warning notification: {e}")

            logger.info(f"[{self._bot_id}] Configuration validation passed")

        except Exception as e:
            logger.warning(
                f"[{self._bot_id}] Config validation error (non-fatal): {e}"
            )

    async def _verify_leverage_on_start(self) -> None:
        """
        Ensure exchange leverage matches config on startup.

        Calls set_leverage to enforce the configured value. This is idempotent
        on Binance — if already correct, it's a no-op.
        """
        config_leverage = getattr(self._config, 'leverage', None)
        if config_leverage is None:
            return

        symbol = getattr(self._config, 'symbol', None)
        if not symbol:
            return

        try:
            result = await self._exchange.futures.set_leverage(
                symbol=symbol,
                leverage=int(config_leverage),
            )
            actual = result.get('leverage', config_leverage)
            logger.info(f"[{self._bot_id}] Leverage verified/set: {actual}x")
        except Exception as e:
            logger.warning(
                f"[{self._bot_id}] Leverage verification failed (non-fatal): {e}"
            )

    # =========================================================================
    # Orphan Order Cleanup on Startup
    # =========================================================================

    async def _cleanup_orphan_orders_on_start(self) -> None:
        """
        FIX: Cancel stale orders from previous session on startup.

        After a crash, limit orders from the previous session may remain
        on the exchange. This method finds and cancels them.
        """
        try:
            symbol = getattr(self, '_symbol', None) or getattr(
                getattr(self, '_config', None), 'symbol', None
            )
            if not symbol or not self._exchange:
                return

            open_orders = await self._exchange.get_open_orders(symbol)
            if not open_orders:
                return

            bot_orders = [
                o for o in open_orders
                if o.client_order_id and o.client_order_id.startswith(self._bot_id)
            ]

            if bot_orders:
                logger.warning(
                    f"[{self._bot_id}] Found {len(bot_orders)} orphan orders "
                    f"from previous session, cancelling..."
                )
                for order in bot_orders:
                    try:
                        await self._exchange.futures_cancel_order(
                            symbol=symbol,
                            order_id=order.order_id,
                            bot_id=self._bot_id,
                        )
                        logger.info(f"[{self._bot_id}] Cancelled orphan order {order.order_id}")
                    except Exception as cancel_err:
                        logger.warning(
                            f"[{self._bot_id}] Failed to cancel orphan order "
                            f"{order.order_id}: {cancel_err}"
                        )
        except Exception as e:
            logger.warning(
                f"[{self._bot_id}] Orphan order cleanup failed (non-fatal): {e}"
            )

    async def _validate_sl_against_liquidation(
        self,
        stop_price: "Decimal",
        position_side: "PositionSide",
        entry_price: "Decimal",
        symbol: str,
    ) -> "Decimal":
        """
        FIX F-1: Validate stop loss price against exchange liquidation price.

        Ensures SL triggers before liquidation to prevent forced liquidation
        at worse price. Fetches liquidation price from exchange position data.

        Args:
            stop_price: Calculated stop loss price
            position_side: LONG or SHORT
            entry_price: Position entry price
            symbol: Trading symbol

        Returns:
            Adjusted stop_price (unchanged if valid, adjusted if too close to liquidation)
        """
        try:
            positions = await self._exchange.futures.get_positions(symbol)
            if not positions:
                return stop_price

            pos = positions[0]
            liq_price = getattr(pos, 'liquidation_price', None)
            if not liq_price or liq_price <= 0:
                return stop_price

            tick_size = getattr(self, '_tick_size', None)
            if not tick_size or tick_size <= 0:
                tick_size = Decimal("0.01")

            from decimal import ROUND_DOWN, ROUND_UP

            if position_side == PositionSide.LONG:
                # Long: SL must be ABOVE liquidation price
                if stop_price <= liq_price:
                    old_sl = stop_price
                    stop_price = liq_price + (entry_price - liq_price) * Decimal("0.1")
                    stop_price = (stop_price / tick_size).quantize(Decimal("1"), rounding=ROUND_UP) * tick_size
                    logger.warning(
                        f"[{self._bot_id}] SL {old_sl} <= liquidation {liq_price}. "
                        f"Adjusted to {stop_price} (10% above liquidation)"
                    )
            else:
                # Short: SL must be BELOW liquidation price
                if stop_price >= liq_price:
                    old_sl = stop_price
                    stop_price = liq_price - (liq_price - entry_price) * Decimal("0.1")
                    stop_price = (stop_price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
                    logger.warning(
                        f"[{self._bot_id}] SL {old_sl} >= liquidation {liq_price}. "
                        f"Adjusted to {stop_price} (10% below liquidation)"
                    )

            return stop_price

        except Exception as e:
            logger.debug(f"[{self._bot_id}] Could not validate SL vs liquidation: {e}")
            return stop_price

    # =========================================================================
    # Liquidity and Order Book Validation
    # =========================================================================

    async def _check_liquidity(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        max_slippage_pct: Decimal = Decimal("0.005"),
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if sufficient liquidity exists for the order.

        Protects against:
        - Large order slippage
        - Partial fills at poor prices
        - Incomplete depth data

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            max_slippage_pct: Maximum acceptable slippage (default 0.5%)

        Returns:
            Tuple of (has_liquidity, estimated_avg_price)
        """
        try:
            # Get order book depth
            orderbook = await self._exchange.futures.get_orderbook(
                symbol=symbol,
                limit=20,  # Get top 20 levels
            )

            if not orderbook:
                logger.warning(f"[{self._bot_id}] Failed to get orderbook for {symbol}")
                return True, None  # Don't block on failure

            # Check the relevant side
            if side.upper() == "BUY":
                levels = orderbook.asks  # Buying consumes asks
            else:
                levels = orderbook.bids  # Selling consumes bids

            if not levels:
                logger.warning(f"[{self._bot_id}] Empty {side} orderbook for {symbol}")
                return False, None

            # Validate depth data completeness
            if len(levels) < 5:
                logger.warning(
                    f"[{self._bot_id}] Insufficient depth levels: {len(levels)} < 5"
                )
                # Don't block but log the warning

            # Calculate average fill price
            remaining_qty = quantity
            total_cost = Decimal("0")
            best_price = Decimal(str(levels[0][0]))

            for price, qty in levels:
                price = Decimal(str(price))
                qty = Decimal(str(qty))

                if remaining_qty <= 0:
                    break

                fill_qty = min(remaining_qty, qty)
                total_cost += fill_qty * price
                remaining_qty -= fill_qty

            if remaining_qty > 0:
                logger.warning(
                    f"[{self._bot_id}] Insufficient liquidity: "
                    f"need {quantity}, only {quantity - remaining_qty} available in top 20 levels"
                )
                return False, None

            avg_price = total_cost / quantity

            # Calculate slippage from best price
            slippage = abs(avg_price - best_price) / best_price
            if slippage > max_slippage_pct:
                logger.warning(
                    f"[{self._bot_id}] High slippage detected: "
                    f"{float(slippage)*100:.2f}% > {float(max_slippage_pct)*100:.1f}%"
                )
                return False, avg_price

            return True, avg_price

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Liquidity check failed: {e}")
            return True, None  # Don't block on failure

    def _calculate_order_splits(
        self,
        quantity: Decimal,
        max_order_pct: Decimal = Decimal("0.10"),
        min_orders: int = 1,
        max_orders: int = 5,
    ) -> list[Decimal]:
        """
        Split large orders to reduce market impact.

        Args:
            quantity: Total order quantity
            max_order_pct: Max percentage of quantity per order
            min_orders: Minimum number of splits
            max_orders: Maximum number of splits

        Returns:
            List of order quantities
        """
        # For now, return single order (grid bots use small sizes)
        # This is a placeholder for future large order handling
        return [quantity]

    # =========================================================================
    # Pre-Trade Validation (combines all checks)
    # =========================================================================

    async def _validate_pre_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        check_time_sync: bool = True,
        check_liquidity: bool = False,
        check_position_sync: bool = True,
    ) -> bool:
        """
        Comprehensive pre-trade validation.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            check_time_sync: Whether to check time synchronization
            check_liquidity: Whether to check liquidity (for large orders)
            check_position_sync: Whether to verify position sync with exchange

        Returns:
            True if all checks pass
        """
        # Check data connection health
        if not self._is_data_connection_healthy():
            logger.warning(
                f"[{self._bot_id}] Data connection unhealthy - skipping trade"
            )
            return False

        # Check time synchronization
        if check_time_sync:
            if not await self._check_time_sync_health():
                logger.warning(
                    f"[{self._bot_id}] Time sync unhealthy - skipping trade"
                )
                return False

        # Check position sync with exchange (prevent duplicate orders)
        if check_position_sync:
            position_side = "LONG" if side.upper() == "BUY" else "SHORT"
            if not await self._check_position_before_open(position_side, quantity):
                logger.warning(
                    f"[{self._bot_id}] Position sync check failed - skipping trade"
                )
                return False

        # Check liquidity for large orders
        if check_liquidity:
            has_liquidity, _ = await self._check_liquidity(symbol, side, quantity)
            if not has_liquidity:
                logger.warning(
                    f"[{self._bot_id}] Insufficient liquidity - skipping trade"
                )
                return False

        return True

    # =========================================================================
    # Position State Synchronization
    # =========================================================================

    async def _verify_position_sync(
        self,
        expected_quantity: Optional[Decimal] = None,
        expected_side: Optional[str] = None,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify local position matches exchange state.

        Protects against:
        - Local state desync after order execution
        - Position closed externally (stop loss, liquidation)
        - Duplicate order prevention

        Args:
            expected_quantity: Expected position quantity (None to just fetch)
            expected_side: Expected position side ("LONG" or "SHORT")

        Returns:
            Tuple of (is_synced, exchange_position_data)
        """
        try:
            # Fetch current position from exchange
            positions = await self._exchange.get_positions(self.symbol)

            exchange_position = None
            for pos in positions:
                if pos.symbol == self.symbol and pos.quantity != Decimal("0"):
                    exchange_position = {
                        "symbol": pos.symbol,
                        "quantity": abs(pos.quantity),
                        "side": "LONG" if pos.quantity > 0 else "SHORT",
                        "entry_price": pos.entry_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    break

            # If no expectation, just return exchange state
            if expected_quantity is None:
                return True, exchange_position

            # Verify against expectation
            if exchange_position is None:
                if expected_quantity > 0:
                    logger.warning(
                        f"[{self._bot_id}] Position desync: "
                        f"expected {expected_side} {expected_quantity}, "
                        f"but no position on exchange"
                    )
                    return False, None
                return True, None

            # Check quantity match (allow 0.1% tolerance for rounding)
            qty_diff = abs(exchange_position["quantity"] - expected_quantity)
            tolerance = expected_quantity * Decimal("0.001")
            if qty_diff > tolerance:
                logger.warning(
                    f"[{self._bot_id}] Position quantity mismatch: "
                    f"local={expected_quantity}, exchange={exchange_position['quantity']}"
                )
                return False, exchange_position

            # Check side match
            if expected_side and exchange_position["side"] != expected_side:
                logger.warning(
                    f"[{self._bot_id}] Position side mismatch: "
                    f"local={expected_side}, exchange={exchange_position['side']}"
                )
                return False, exchange_position

            return True, exchange_position

        except Exception as e:
            logger.error(f"[{self._bot_id}] Failed to verify position sync: {e}")
            return False, None

    async def _sync_position_from_exchange(self) -> Optional[Dict[str, Any]]:
        """
        Force sync local position from exchange.

        Should be called after order execution to ensure consistency.

        In Hedge Mode, if _expected_position_side is set, only returns
        positions matching that side (LONG or SHORT). This prevents
        multi-bot conflicts when trading the same symbol.

        Returns:
            Exchange position data or None if no position
        """
        try:
            positions = await self._exchange.get_positions(self.symbol)

            for pos in positions:
                if pos.symbol == self.symbol and pos.quantity != Decimal("0"):
                    pos_side = "LONG" if pos.quantity > 0 else "SHORT"

                    # In Hedge Mode, filter by expected position side
                    if self._expected_position_side:
                        if pos_side != self._expected_position_side:
                            logger.debug(
                                f"[{self._bot_id}] Skipping {pos_side} position - "
                                f"expected {self._expected_position_side}"
                            )
                            continue

                    position_data = {
                        "symbol": pos.symbol,
                        "quantity": abs(pos.quantity),
                        "side": pos_side,
                        "entry_price": pos.entry_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    logger.debug(
                        f"[{self._bot_id}] Synced position from exchange: "
                        f"{position_data['side']} {position_data['quantity']} @ {position_data['entry_price']}"
                    )
                    return position_data

            logger.debug(f"[{self._bot_id}] No position found on exchange")
            return None

        except Exception as e:
            logger.error(f"[{self._bot_id}] Failed to sync position: {e}")
            return None

    async def _check_position_before_open(
        self,
        side: str,
        quantity: Decimal,
    ) -> bool:
        """
        Check if safe to open new position.

        Verifies:
        1. No unexpected existing position
        2. Position limit not exceeded
        3. Exchange state matches local expectation

        Args:
            side: "LONG" or "SHORT"
            quantity: Quantity to open

        Returns:
            True if safe to proceed
        """
        try:
            # Get current exchange position
            is_synced, exchange_pos = await self._verify_position_sync()

            if not is_synced:
                logger.warning(
                    f"[{self._bot_id}] Cannot open position - sync check failed"
                )
                return False

            # If we have local position tracking, verify it matches
            if hasattr(self, '_position') and self._position is not None:
                local_qty = getattr(self._position, 'quantity', Decimal("0"))
                local_side = getattr(self._position, 'side', None)
                if local_side:
                    local_side = local_side.value if hasattr(local_side, 'value') else str(local_side)

                if exchange_pos is None and local_qty > 0:
                    # Local thinks we have position, exchange says no
                    logger.warning(
                        f"[{self._bot_id}] Position closed externally - "
                        f"clearing local state"
                    )
                    self._position = None
                    return False

                if exchange_pos and local_qty == 0:
                    # Exchange has position, local doesn't know
                    logger.warning(
                        f"[{self._bot_id}] Unexpected exchange position: "
                        f"{exchange_pos['side']} {exchange_pos['quantity']} - "
                        f"syncing local state"
                    )
                    # Don't open new position until reconciled
                    return False

            return True

        except Exception as e:
            logger.error(f"[{self._bot_id}] Position check failed: {e}")
            return False

    # =========================================================================
    # State Persistence and Recovery
    # =========================================================================

    # State schema version for compatibility checking
    # Increment MAJOR for breaking changes, MINOR for new fields, PATCH for fixes
    STATE_SCHEMA_VERSION = 2
    STATE_SCHEMA_VERSION_STR = "2.0.0"

    # Maximum age for state to be considered valid (24 hours)
    MAX_STATE_AGE_HOURS = 24

    # Maximum number of state snapshots to keep for rollback
    MAX_STATE_SNAPSHOTS = 5

    # State migration functions (override in subclass if needed)
    # Format: {(from_version, to_version): migration_function}
    STATE_MIGRATIONS: Dict[tuple, Callable] = {}

    def _create_state_snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot of current bot state.

        Includes metadata for validation on restore:
        - Schema version for compatibility
        - Timestamp for staleness detection
        - Checksum for integrity

        Returns:
            State snapshot dictionary
        """
        import hashlib
        import json

        now = datetime.now(timezone.utc)

        # Collect state data (subclasses should override _get_state_data)
        state_data = self._get_state_data()

        # Create snapshot with metadata
        snapshot = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "bot_id": self._bot_id,
            "bot_type": self.bot_type,
            "symbol": self.symbol,
            "timestamp": now.isoformat(),
            "state": state_data,
        }

        # Add checksum for integrity verification
        state_json = json.dumps(snapshot, sort_keys=True, default=str)
        snapshot["checksum"] = hashlib.md5(state_json.encode()).hexdigest()

        return snapshot

    def _get_state_data(self) -> Dict[str, Any]:
        """
        Get bot-specific state data for persistence.

        Subclasses should override to include their specific state.

        Returns:
            Dictionary of state data
        """
        return {
            "bot_state": self._state.value if self._state else "unknown",
        }

    def _validate_state_snapshot(
        self,
        snapshot: Dict[str, Any],
        max_age_hours: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Validate a state snapshot before restoration.

        Checks:
        1. Schema version compatibility
        2. Timestamp staleness
        3. Bot ID match
        4. Checksum integrity

        Args:
            snapshot: The state snapshot to validate
            max_age_hours: Maximum age in hours (default: MAX_STATE_AGE_HOURS)

        Returns:
            Tuple of (is_valid, error_message)
        """
        import hashlib
        import json

        if max_age_hours is None:
            max_age_hours = self.MAX_STATE_AGE_HOURS

        # Check schema version
        schema_version = snapshot.get("schema_version", 0)
        if schema_version != self.STATE_SCHEMA_VERSION:
            return False, (
                f"Schema version mismatch: saved={schema_version}, "
                f"current={self.STATE_SCHEMA_VERSION}"
            )

        # Check bot ID match
        saved_bot_id = snapshot.get("bot_id")
        if saved_bot_id != self._bot_id:
            return False, f"Bot ID mismatch: saved={saved_bot_id}, current={self._bot_id}"

        # Check timestamp staleness
        timestamp_str = snapshot.get("timestamp")
        if timestamp_str:
            try:
                saved_time = datetime.fromisoformat(timestamp_str)
                age = datetime.now(timezone.utc) - saved_time
                age_hours = age.total_seconds() / 3600

                if age_hours > max_age_hours:
                    return False, (
                        f"State too old: {age_hours:.1f} hours > {max_age_hours} hours max"
                    )
            except Exception as e:
                return False, f"Invalid timestamp: {e}"
        else:
            return False, "Missing timestamp in snapshot"

        # Verify checksum
        saved_checksum = snapshot.get("checksum")
        if not saved_checksum:
            return False, "Missing checksum in snapshot"
        if saved_checksum:
            # Recalculate checksum without the checksum field
            snapshot_copy = {k: v for k, v in snapshot.items() if k != "checksum"}
            state_json = json.dumps(snapshot_copy, sort_keys=True, default=str)
            calculated_checksum = hashlib.md5(state_json.encode()).hexdigest()

            if calculated_checksum != saved_checksum:
                return False, "Checksum mismatch - state may be corrupted"

        return True, "State snapshot is valid"

    def _migrate_state_data(
        self,
        state_data: Dict[str, Any],
        from_version: int,
        to_version: int,
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Migrate state data between schema versions.

        Supports both upgrade (older -> newer) and downgrade (newer -> older).

        Args:
            state_data: State data to migrate
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            Tuple of (migrated_data, success, message)
        """
        if from_version == to_version:
            return state_data, True, "No migration needed"

        # Check if migration is possible
        direction = "upgrade" if from_version < to_version else "downgrade"

        logger.info(
            f"[{self._bot_id}] Attempting {direction} migration: "
            f"v{from_version} -> v{to_version}"
        )

        # Look for migration function
        migration_key = (from_version, to_version)
        if migration_key in self.STATE_MIGRATIONS:
            try:
                migrated = self.STATE_MIGRATIONS[migration_key](state_data.copy())
                logger.info(
                    f"[{self._bot_id}] Migration successful: v{from_version} -> v{to_version}"
                )
                return migrated, True, f"Migrated from v{from_version} to v{to_version}"
            except Exception as e:
                logger.error(f"[{self._bot_id}] Migration failed: {e}")
                return state_data, False, f"Migration failed: {e}"

        # Try step-by-step migration
        current_version = from_version
        current_data = state_data.copy()
        step = 1 if from_version < to_version else -1

        while current_version != to_version:
            next_version = current_version + step
            step_key = (current_version, next_version)

            if step_key in self.STATE_MIGRATIONS:
                try:
                    current_data = self.STATE_MIGRATIONS[step_key](current_data)
                    current_version = next_version
                except Exception as e:
                    logger.error(
                        f"[{self._bot_id}] Step migration failed at "
                        f"v{current_version} -> v{next_version}: {e}"
                    )
                    return state_data, False, f"Step migration failed: {e}"
            else:
                # No migration path found
                logger.warning(
                    f"[{self._bot_id}] No migration path from "
                    f"v{current_version} to v{next_version}"
                )
                return state_data, False, f"No migration path from v{from_version} to v{to_version}"

        return current_data, True, f"Migrated from v{from_version} to v{to_version}"

    def _validate_and_migrate_snapshot(
        self,
        snapshot: Dict[str, Any],
        max_age_hours: Optional[int] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate and potentially migrate a state snapshot.

        Args:
            snapshot: The state snapshot
            max_age_hours: Maximum age in hours

        Returns:
            Tuple of (is_valid, message, migrated_state_data)
        """
        # Basic validation first
        is_valid, message = self._validate_state_snapshot(snapshot, max_age_hours)

        if is_valid:
            return True, message, snapshot.get("state", {})

        # Check if it's a version mismatch that we can migrate
        if "Schema version mismatch" in message:
            saved_version = snapshot.get("schema_version", 0)

            # Try migration
            state_data = snapshot.get("state", {})
            migrated_data, success, migrate_msg = self._migrate_state_data(
                state_data, saved_version, self.STATE_SCHEMA_VERSION
            )

            if success:
                logger.info(f"[{self._bot_id}] {migrate_msg}")
                return True, migrate_msg, migrated_data
            else:
                return False, f"Cannot restore: {message}. Migration failed: {migrate_msg}", {}

        return False, message, {}

    async def create_rollback_snapshot(
        self,
        description: str = "Manual snapshot",
        mark_stable: bool = False,
    ) -> Optional[str]:
        """
        Create a rollback snapshot of current state.

        Args:
            description: Description of this snapshot
            mark_stable: Whether to mark as known stable state

        Returns:
            Snapshot ID if successful, None otherwise
        """
        try:
            from src.infrastructure.version_manager import Version, VersionManager

            # Get or create version manager
            if not hasattr(self, '_version_manager'):
                self._version_manager = VersionManager(
                    current_version=Version.parse(self.STATE_SCHEMA_VERSION_STR),
                    storage_path=Path(f"data/snapshots/{self._bot_id}"),
                    max_snapshots=self.MAX_STATE_SNAPSHOTS,
                )

            # Create snapshot
            state_data = self._get_state_data()
            snapshot = self._version_manager.create_snapshot(
                state_data=state_data,
                description=description,
                bot_id=self._bot_id,
                mark_stable=mark_stable,
                tags={"bot_type": self.bot_type, "symbol": self.symbol},
            )

            logger.info(
                f"[{self._bot_id}] Created rollback snapshot: {snapshot.snapshot_id} "
                f"(stable={mark_stable})"
            )

            return snapshot.snapshot_id

        except ImportError:
            logger.debug(f"[{self._bot_id}] Version manager not available")
            return None
        except Exception as e:
            logger.error(f"[{self._bot_id}] Failed to create rollback snapshot: {e}")
            return None

    async def rollback_to_snapshot(
        self,
        snapshot_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Rollback to a previous state snapshot.

        Args:
            snapshot_id: ID of snapshot to rollback to (None = last stable)

        Returns:
            Tuple of (success, message)
        """
        try:
            from src.infrastructure.version_manager import VersionManager

            if not hasattr(self, '_version_manager'):
                return False, "Version manager not initialized - no snapshots available"

            # Get snapshot
            if snapshot_id:
                success, state_data, message = self._version_manager.rollback_to_snapshot(
                    snapshot_id
                )
            else:
                success, state_data, message = self._version_manager.rollback_to_stable()

            if not success:
                return False, message

            # Apply state (subclass should implement _apply_restored_state)
            if hasattr(self, '_apply_restored_state'):
                await self._apply_restored_state(state_data)

            logger.info(f"[{self._bot_id}] Rollback successful: {message}")
            return True, message

        except ImportError:
            return False, "Version manager not available"
        except Exception as e:
            logger.error(f"[{self._bot_id}] Rollback failed: {e}")
            return False, f"Rollback failed: {e}"

    async def mark_current_state_stable(self) -> bool:
        """
        Mark current state as stable (known good state).

        Call this after confirming the bot is working correctly.

        Returns:
            True if successful
        """
        snapshot_id = await self.create_rollback_snapshot(
            description="Marked as stable",
            mark_stable=True,
        )
        return snapshot_id is not None

    async def _save_state_atomic(
        self,
        state_data: Dict[str, Any],
        save_func: Callable,
    ) -> bool:
        """
        Save state atomically with error handling.

        Ensures all-or-nothing state persistence.

        Args:
            state_data: State data to save
            save_func: Async function to perform the save

        Returns:
            True if save succeeded
        """
        try:
            # Create snapshot with metadata
            import hashlib
            import json
            snapshot = self._create_state_snapshot()
            snapshot["state"].update(state_data)

            # Recompute checksum after mutation
            snapshot_no_checksum = {k: v for k, v in snapshot.items() if k != "checksum"}
            state_json = json.dumps(snapshot_no_checksum, sort_keys=True, default=str)
            snapshot["checksum"] = hashlib.md5(state_json.encode()).hexdigest()

            # Perform save
            await save_func(snapshot)

            logger.debug(f"[{self._bot_id}] State saved successfully")
            return True

        except Exception as e:
            logger.error(f"[{self._bot_id}] State save failed: {e}")
            # Notify if possible
            if self._notifier:
                await self._notifier.send_warning(
                    title=f"{self.bot_type}: State Save Failed",
                    message=f"Error: {e}\nState may not be recoverable on restart.",
                )
            return False

    async def _restore_state_with_validation(
        self,
        load_func: Callable,
        apply_func: Callable,
    ) -> bool:
        """
        Restore state with validation and error handling.

        Args:
            load_func: Async function to load state snapshot
            apply_func: Async function to apply state to bot

        Returns:
            True if restore succeeded
        """
        try:
            # Load state snapshot
            snapshot = await load_func()

            if snapshot is None:
                logger.info(f"[{self._bot_id}] No saved state found - starting fresh")
                return False

            # Validate snapshot
            is_valid, message = self._validate_state_snapshot(snapshot)
            if not is_valid:
                logger.warning(
                    f"[{self._bot_id}] State validation failed: {message} - starting fresh"
                )
                return False

            # Apply state
            await apply_func(snapshot["state"])

            logger.info(
                f"[{self._bot_id}] State restored successfully from "
                f"{snapshot.get('timestamp', 'unknown time')}"
            )
            return True

        except Exception as e:
            logger.error(f"[{self._bot_id}] State restore failed: {e}")
            return False

    # =========================================================================
    # Concurrency Protection
    # =========================================================================

    async def _acquire_lock_with_timeout(
        self,
        lock: asyncio.Lock,
        timeout_seconds: float = 30.0,
        context: str = "operation",
    ) -> bool:
        """
        Acquire a lock with timeout to prevent deadlocks.

        Args:
            lock: The asyncio.Lock to acquire
            timeout_seconds: Maximum wait time
            context: Description of the operation (for logging)

        Returns:
            True if lock acquired, False if timeout
        """
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[{self._bot_id}] Lock timeout after {timeout_seconds}s for {context}"
            )
            return False

    def _init_state_lock(self) -> None:
        """Initialize state modification lock."""
        if not hasattr(self, "_state_lock"):
            self._state_lock = asyncio.Lock()

    async def _modify_state_safely(
        self,
        modify_func: Callable,
        context: str = "state modification",
    ) -> bool:
        """
        Modify bot state with lock protection.

        Prevents concurrent state modifications that could cause corruption.

        Args:
            modify_func: Async function that modifies state
            context: Description of the modification

        Returns:
            True if modification succeeded
        """
        self._init_state_lock()

        if not await self._acquire_lock_with_timeout(self._state_lock, context=context):
            logger.error(f"[{self._bot_id}] Failed to acquire lock for {context}")
            return False

        try:
            await modify_func()
            return True
        except Exception as e:
            logger.error(f"[{self._bot_id}] Error during {context}: {e}")
            return False
        finally:
            self._state_lock.release()

    # =========================================================================
    # Position Reconciliation (Detect Manual Operations)
    # =========================================================================

    # Position reconciliation interval (seconds)
    POSITION_RECONCILIATION_INTERVAL = 30

    def _start_position_reconciliation(self) -> None:
        """Start background position reconciliation task."""
        # Attributes are initialized in __init__, no need for hasattr checks
        if self._reconciliation_task is not None:
            return

        async def reconciliation_loop():
            while self._state == BotState.RUNNING:
                await asyncio.sleep(self.POSITION_RECONCILIATION_INTERVAL)
                if self._state == BotState.RUNNING:
                    try:
                        await self._reconcile_position()
                    except Exception as e:
                        logger.warning(f"[{self._bot_id}] Position reconciliation error: {e}")

        self._reconciliation_task = asyncio.create_task(reconciliation_loop())
        logger.debug(f"[{self._bot_id}] Position reconciliation started")

    def _stop_position_reconciliation(self) -> None:
        """Stop background position reconciliation task."""
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            self._reconciliation_task = None
            logger.debug(f"[{self._bot_id}] Position reconciliation stopped")

    async def _reconcile_position(self) -> bool:
        """
        Reconcile local position state with exchange.

        Detects:
        - Manual position closure
        - External stop loss/liquidation
        - Position size changes from partial fills/manual trades

        Returns:
            True if positions are in sync, False if mismatch detected
        """
        try:
            # Get current exchange position
            exchange_pos = await self._sync_position_from_exchange()

            # Get local position state
            local_position = getattr(self, "_position", None)
            local_qty = Decimal("0")
            local_side = None

            if local_position:
                local_qty = getattr(local_position, "quantity", Decimal("0"))
                local_side = getattr(local_position, "side", None)
                if local_side and hasattr(local_side, "value"):
                    local_side = local_side.value

            # Check for mismatches
            mismatch_detected = False
            mismatch_reason = ""

            if exchange_pos is None and local_qty > 0:
                # Position closed externally
                mismatch_detected = True
                mismatch_reason = "Position closed externally (stop loss/liquidation/manual)"
                self._position_mismatch_count += 1

                # Clear local state
                self._position = None

                # Notify
                logger.warning(
                    f"[{self._bot_id}] POSITION MISMATCH: {mismatch_reason} - "
                    f"local had {local_side} {local_qty}"
                )

                if self._notifier:
                    await self._notifier.send_warning(
                        title=f"⚠️ Position Closed Externally",
                        message=(
                            f"Bot: {self._bot_id}\n"
                            f"Symbol: {self.symbol}\n"
                            f"Lost position: {local_side} {local_qty}\n"
                            f"Reason: External closure detected"
                        ),
                    )

            elif exchange_pos and local_qty == 0:
                # Exchange has position we don't know about
                mismatch_detected = True
                mismatch_reason = "Unexpected position on exchange (manual open)"
                self._position_mismatch_count += 1

                logger.warning(
                    f"[{self._bot_id}] POSITION MISMATCH: {mismatch_reason} - "
                    f"exchange has {exchange_pos['side']} {exchange_pos['quantity']}"
                )

                if self._notifier:
                    await self._notifier.send_warning(
                        title=f"⚠️ Unexpected Position Detected",
                        message=(
                            f"Bot: {self._bot_id}\n"
                            f"Symbol: {self.symbol}\n"
                            f"Found position: {exchange_pos['side']} {exchange_pos['quantity']}\n"
                            f"Action: Bot pausing to avoid conflicts"
                        ),
                    )

            elif exchange_pos and local_qty > 0:
                # Both have positions - check for size mismatch
                exchange_qty = exchange_pos.get("quantity", Decimal("0"))
                exchange_side = exchange_pos.get("side", "")

                # Allow 0.1% tolerance for quantity
                qty_diff = abs(exchange_qty - local_qty) / local_qty if local_qty > 0 else Decimal("0")

                if qty_diff > Decimal("0.001"):
                    mismatch_detected = True
                    mismatch_reason = f"Position size mismatch: local={local_qty}, exchange={exchange_qty}"
                    self._position_mismatch_count += 1

                    logger.warning(
                        f"[{self._bot_id}] POSITION MISMATCH: {mismatch_reason}"
                    )

                elif exchange_side.upper() != (local_side.upper() if local_side else ""):
                    mismatch_detected = True
                    mismatch_reason = f"Position side mismatch: local={local_side}, exchange={exchange_side}"
                    self._position_mismatch_count += 1

                    logger.warning(
                        f"[{self._bot_id}] POSITION MISMATCH: {mismatch_reason}"
                    )

            # Reset mismatch count if in sync
            if not mismatch_detected:
                self._position_mismatch_count = 0

            # Store last known position for comparison
            self._last_known_position = exchange_pos

            return not mismatch_detected

        except Exception as e:
            logger.error(f"[{self._bot_id}] Position reconciliation error: {e}")
            return False

    # =========================================================================
    # Order Rejection Handling (Balance Check, Error Classification)
    # =========================================================================

    # Order error codes
    ORDER_ERROR_INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    ORDER_ERROR_INVALID_QUANTITY = "INVALID_QUANTITY"
    ORDER_ERROR_INVALID_PRICE = "INVALID_PRICE"
    ORDER_ERROR_POSITION_LIMIT = "POSITION_LIMIT"
    ORDER_ERROR_RATE_LIMIT = "RATE_LIMIT"
    ORDER_ERROR_TIMEOUT = "TIMEOUT"
    ORDER_ERROR_UNKNOWN = "UNKNOWN"

    # =========================================================================
    # Multi-Strategy Resource Management
    # =========================================================================

    def _init_resource_tracking(self) -> None:
        """Initialize multi-strategy resource tracking."""
        if not hasattr(self, "_pending_order_value"):
            self._pending_order_value: Decimal = Decimal("0")
        if not hasattr(self, "_pending_orders_lock"):
            self._pending_orders_lock = asyncio.Lock()
        if not hasattr(self, "_capital_lock"):
            self._capital_lock = asyncio.Lock()
        if not hasattr(self, "_allocated_capital"):
            self._allocated_capital: Optional[Decimal] = None
        if not hasattr(self, "_capital_usage"):
            self._capital_usage: Decimal = Decimal("0")

    def set_allocated_capital(self, amount: Decimal) -> None:
        """
        Set the allocated capital for this bot.

        This should be called by FundManager when allocating funds.
        The bot will not exceed this allocation even if more balance is available.

        Args:
            amount: Maximum capital this bot can use
        """
        self._init_resource_tracking()
        self._allocated_capital = amount
        logger.info(f"[{self._bot_id}] Capital allocation set: {amount} USDT")

    def get_allocated_capital(self) -> Optional[Decimal]:
        """Get the allocated capital for this bot."""
        self._init_resource_tracking()
        return self._allocated_capital

    def get_capital_usage(self) -> Decimal:
        """Get current capital usage (open positions + pending orders)."""
        self._init_resource_tracking()
        return self._capital_usage + self._pending_order_value

    def get_available_capital(self) -> Decimal:
        """
        Get available capital considering allocation and pending orders.

        Returns:
            Available capital (allocation - usage - pending)
        """
        self._init_resource_tracking()
        if self._allocated_capital is None:
            return Decimal("999999999")  # No limit

        usage = self.get_capital_usage()
        available = self._allocated_capital - usage
        return max(available, Decimal("0"))

    async def _reserve_capital_for_order(
        self,
        required_margin: Decimal,
    ) -> tuple[bool, str]:
        """
        Reserve capital for a pending order.

        This prevents concurrent orders from exceeding allocated capital.

        Args:
            required_margin: Margin required for the order

        Returns:
            Tuple of (success, message)
        """
        self._init_resource_tracking()

        async with self._pending_orders_lock:
            available = self.get_available_capital()

            if required_margin > available:
                return False, (
                    f"Exceeds allocation: need {required_margin:.2f}, "
                    f"available {available:.2f} "
                    f"(allocated: {self._allocated_capital}, "
                    f"usage: {self._capital_usage}, "
                    f"pending: {self._pending_order_value})"
                )

            self._pending_order_value += required_margin
            logger.debug(
                f"[{self._bot_id}] Reserved {required_margin:.2f} USDT "
                f"(total pending: {self._pending_order_value:.2f})"
            )
            return True, "Capital reserved"

    async def _release_capital_reservation(
        self,
        amount: Decimal,
        add_to_usage: bool = False,
    ) -> None:
        """
        Release capital reservation after order completes.

        Args:
            amount: Amount to release from pending
            add_to_usage: If True, add to capital_usage (order filled)
        """
        self._init_resource_tracking()

        async with self._pending_orders_lock:
            self._pending_order_value = max(
                Decimal("0"),
                self._pending_order_value - amount
            )

            if add_to_usage:
                self._capital_usage += amount

            logger.debug(
                f"[{self._bot_id}] Released {amount:.2f} USDT reservation "
                f"(add_to_usage={add_to_usage})"
            )

    def _update_capital_usage(self, delta: Decimal) -> None:
        """
        Update capital usage after position change.

        Args:
            delta: Change in capital usage (positive = increased, negative = reduced)
        """
        self._init_resource_tracking()
        self._capital_usage = max(Decimal("0"), self._capital_usage + delta)

    async def _check_balance_for_order(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
    ) -> tuple[bool, str]:
        """
        Check if account has sufficient balance for order.

        This method checks:
        1. Exchange available balance
        2. Allocated capital limit (if set)
        3. Pending order reservations

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (or estimated price for market orders)
            leverage: Position leverage

        Returns:
            Tuple of (has_sufficient_balance, message)
        """
        self._init_resource_tracking()

        try:
            # Calculate required margin
            if leverage <= 0:
                raise ValueError(f"Invalid leverage: {leverage}, must be > 0")
            notional_value = quantity * price
            required_margin = notional_value / Decimal(leverage)

            # Add buffer for fees (0.1%)
            required_with_fees = required_margin * Decimal("1.001")

            # Check 1: Allocated capital (if set)
            if self._allocated_capital is not None:
                available_allocation = self.get_available_capital()
                if required_with_fees > available_allocation:
                    return False, (
                        f"Exceeds allocation: need {required_with_fees:.2f} USDT, "
                        f"available {available_allocation:.2f} USDT "
                        f"(allocated: {self._allocated_capital})"
                    )

            # Check 2: Exchange balance
            balance = await self._exchange.futures.get_balance("USDT")
            available = balance.free if balance else Decimal("0")

            if available < required_with_fees:
                return False, (
                    f"Insufficient balance: need {required_with_fees:.2f} USDT "
                    f"(margin + fees), have {available:.2f} USDT"
                )

            return True, f"Balance OK: {available:.2f} USDT available"

        except Exception as e:
            logger.error(f"[{self._bot_id}] Balance check failed: {e}")
            return False, f"Balance check error: {e}"

    async def _check_balance_and_reserve(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
    ) -> tuple[bool, str, Decimal]:
        """
        Check balance and reserve capital atomically.

        This combines balance check and reservation to prevent race conditions.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            leverage: Position leverage

        Returns:
            Tuple of (success, message, reserved_amount)
        """
        # Calculate required margin
        notional_value = quantity * price
        required_margin = notional_value / Decimal(leverage)
        required_with_fees = required_margin * Decimal("1.001")

        # Check and reserve under lock to prevent TOCTOU race
        async with self._capital_lock:
            ok, msg = await self._check_balance_for_order(
                symbol, quantity, price, leverage
            )
            if not ok:
                return False, msg, Decimal("0")

            # Reserve capital
            reserve_ok, reserve_msg = await self._reserve_capital_for_order(required_with_fees)
            if not reserve_ok:
                return False, reserve_msg, Decimal("0")

        return True, msg, required_with_fees

    # =========================================================================
    # API Rate Limit Awareness
    # =========================================================================

    async def _check_rate_limit_ok(self) -> tuple[bool, str]:
        """
        Check if API rate limit allows placing an order.

        This method should be called before placing orders to avoid
        rate limit errors.

        Returns:
            Tuple of (is_ok, message)
        """
        try:
            # Check if exchange has rate limit info
            if hasattr(self._exchange, "get_rate_limit_status"):
                status = await self._exchange.get_rate_limit_status()
                if status.get("is_limited", False):
                    retry_after = status.get("retry_after_seconds", 60)
                    return False, f"Rate limited, retry after {retry_after}s"

            # Check order queue status
            if hasattr(self._exchange, "get_order_queue_status"):
                queue_status = self._exchange.get_order_queue_status()
                if queue_status.get("queue_size", 0) > 50:
                    return False, (
                        f"Order queue congested: {queue_status['queue_size']} pending"
                    )

            return True, "Rate limit OK"

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Rate limit check failed: {e}")
            return True, "Rate limit check unavailable"

    async def _wait_for_rate_limit(
        self,
        max_wait_seconds: float = 30.0,
        check_interval: float = 1.0,
    ) -> bool:
        """
        Wait for rate limit to clear.

        Args:
            max_wait_seconds: Maximum time to wait
            check_interval: Check interval

        Returns:
            True if rate limit cleared within timeout
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                logger.warning(
                    f"[{self._bot_id}] Rate limit wait timeout after {max_wait_seconds}s"
                )
                return False

            is_ok, msg = await self._check_rate_limit_ok()
            if is_ok:
                return True

            logger.debug(f"[{self._bot_id}] Waiting for rate limit: {msg}")
            await asyncio.sleep(check_interval)

    # =========================================================================
    # Position Coordination
    # =========================================================================

    async def _check_position_limit(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        leverage: int = 1,
    ) -> tuple[bool, str]:
        """
        Check if position limit allows the order.

        This method checks:
        1. Bot's own position limit (from config)
        2. Shared position manager limit (if available)
        3. Account-wide position limit

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            leverage: Position leverage

        Returns:
            Tuple of (is_allowed, message)
        """
        try:
            # Check config-based position limit
            max_position_pct = getattr(self._config, "max_position_pct", Decimal("0.5"))
            capital = self._allocated_capital or Decimal("10000")

            # Get current position
            current_position = await self._get_current_position(symbol)
            current_qty = current_position.get("quantity", Decimal("0")) if current_position else Decimal("0")

            # Calculate new total position
            if side.upper() == "BUY":
                new_qty = current_qty + quantity
            else:
                new_qty = current_qty - quantity

            # Get price for notional calculation
            ticker = await self._exchange.futures.get_ticker(symbol)
            price = Decimal(str(ticker.last_price)) if ticker else Decimal("1")

            # Check position value vs capital
            position_value = abs(new_qty) * price / Decimal(leverage)
            max_position_value = capital * max_position_pct

            if position_value > max_position_value:
                return False, (
                    f"Position limit exceeded: {position_value:.2f} > "
                    f"{max_position_value:.2f} ({max_position_pct*100}% of {capital})"
                )

            return True, "Position within limits"

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Position limit check failed: {e}")
            return True, "Position limit check unavailable"

    async def _get_current_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position dict or None if no position
        """
        try:
            positions = await self._exchange.futures.get_positions()
            for pos in positions:
                if getattr(pos, "symbol", "") == symbol:
                    qty = getattr(pos, "position_amount", Decimal("0"))
                    if qty != 0:
                        return {
                            "symbol": symbol,
                            "quantity": qty,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "entry_price": getattr(pos, "entry_price", Decimal("0")),
                        }
            return None
        except Exception as e:
            logger.warning(f"[{self._bot_id}] Get position failed: {e}")
            return None

    # =========================================================================
    # Comprehensive Pre-Order Check
    # =========================================================================

    async def _comprehensive_pre_order_check(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
        reserve_capital: bool = True,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive pre-order validation including all resource checks.

        This method performs:
        1. Balance check (exchange + allocation)
        2. Rate limit check
        3. Position limit check
        4. Capital reservation (optional)

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            leverage: Position leverage
            reserve_capital: Whether to reserve capital on success

        Returns:
            Tuple of (is_allowed, message, details)
        """
        details = {
            "balance_ok": False,
            "rate_limit_ok": False,
            "position_ok": False,
            "reserved_amount": Decimal("0"),
        }

        # 1. Check balance and optionally reserve
        if reserve_capital:
            balance_ok, balance_msg, reserved = await self._check_balance_and_reserve(
                symbol, quantity, price, leverage
            )
            details["reserved_amount"] = reserved
        else:
            balance_ok, balance_msg = await self._check_balance_for_order(
                symbol, quantity, price, leverage
            )

        details["balance_ok"] = balance_ok
        if not balance_ok:
            return False, balance_msg, details

        # 2. Check rate limit
        rate_ok, rate_msg = await self._check_rate_limit_ok()
        details["rate_limit_ok"] = rate_ok
        if not rate_ok:
            # Release reservation if we reserved capital
            if reserve_capital and details["reserved_amount"] > 0:
                await self._release_capital_reservation(details["reserved_amount"])
            return False, rate_msg, details

        # 3. Check position limit
        position_ok, position_msg = await self._check_position_limit(
            symbol, side, quantity, leverage
        )
        details["position_ok"] = position_ok
        if not position_ok:
            # Release reservation
            if reserve_capital and details["reserved_amount"] > 0:
                await self._release_capital_reservation(details["reserved_amount"])
            return False, position_msg, details

        return True, "All pre-order checks passed", details

    # =========================================================================
    # Computational Resource Management
    # =========================================================================

    # Resource thresholds
    CPU_WARNING_THRESHOLD = 70.0  # Warn at 70% CPU
    CPU_CRITICAL_THRESHOLD = 90.0  # Block new orders at 90% CPU
    MEMORY_WARNING_THRESHOLD = 75.0  # Warn at 75% memory
    MEMORY_CRITICAL_THRESHOLD = 90.0  # Block at 90% memory

    # Throttling configuration
    RESOURCE_CHECK_INTERVAL = 5.0  # Check every 5 seconds
    THROTTLE_DELAY_SECONDS = 1.0  # Delay when throttling
    MAX_THROTTLE_WAIT = 30.0  # Max wait for resources

    def _init_resource_monitoring(self) -> None:
        """Initialize resource monitoring state."""
        if not hasattr(self, "_last_resource_check"):
            self._last_resource_check: float = 0
        if not hasattr(self, "_resource_status"):
            self._resource_status: Dict[str, Any] = {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "is_throttled": False,
                "throttle_reason": None,
            }
        if not hasattr(self, "_throttle_count"):
            self._throttle_count: int = 0

    async def _check_system_resources(self) -> Dict[str, Any]:
        """
        Check current system resource usage.

        Returns:
            Dict with cpu_percent, memory_percent, is_healthy
        """
        self._init_resource_monitoring()

        try:
            import psutil

            # Get CPU usage (non-blocking average over 0.1 second)
            cpu_percent = psutil.cpu_percent(interval=None)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Determine health status
            is_healthy = (
                cpu_percent < self.CPU_CRITICAL_THRESHOLD and
                memory_percent < self.MEMORY_CRITICAL_THRESHOLD
            )

            # Update cached status
            self._resource_status = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "is_healthy": is_healthy,
                "is_warning": (
                    cpu_percent >= self.CPU_WARNING_THRESHOLD or
                    memory_percent >= self.MEMORY_WARNING_THRESHOLD
                ),
                "timestamp": time.time(),
            }
            self._last_resource_check = time.time()

            return self._resource_status

        except ImportError:
            # psutil not available
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "is_healthy": True,
                "is_warning": False,
                "error": "psutil not available",
            }
        except Exception as e:
            logger.warning(f"[{self._bot_id}] Resource check failed: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "is_healthy": True,
                "error": str(e),
            }

    async def _should_throttle_for_resources(self) -> tuple[bool, str]:
        """
        Check if operations should be throttled due to resource constraints.

        Returns:
            Tuple of (should_throttle, reason)
        """
        self._init_resource_monitoring()

        # Use cached status if recent
        if time.time() - self._last_resource_check < self.RESOURCE_CHECK_INTERVAL:
            status = self._resource_status
        else:
            status = await self._check_system_resources()

        # Check CPU
        if status.get("cpu_percent", 0) >= self.CPU_CRITICAL_THRESHOLD:
            return True, f"CPU critical: {status['cpu_percent']:.1f}%"

        # Check memory
        if status.get("memory_percent", 0) >= self.MEMORY_CRITICAL_THRESHOLD:
            return True, f"Memory critical: {status['memory_percent']:.1f}%"

        return False, ""

    async def _wait_for_resources(
        self,
        max_wait_seconds: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Wait for system resources to become available.

        Args:
            max_wait_seconds: Maximum wait time (default: MAX_THROTTLE_WAIT)

        Returns:
            Tuple of (resources_ok, message)
        """
        if max_wait_seconds is None:
            max_wait_seconds = self.MAX_THROTTLE_WAIT

        start_time = time.time()
        wait_count = 0

        while True:
            should_throttle, reason = await self._should_throttle_for_resources()

            if not should_throttle:
                if wait_count > 0:
                    logger.info(
                        f"[{self._bot_id}] Resources available after "
                        f"{wait_count} throttle cycles"
                    )
                return True, "Resources OK"

            elapsed = time.time() - start_time
            if elapsed >= max_wait_seconds:
                logger.warning(
                    f"[{self._bot_id}] Resource wait timeout after {elapsed:.1f}s: {reason}"
                )
                self._throttle_count += 1
                return False, f"Resource timeout: {reason}"

            wait_count += 1
            logger.debug(
                f"[{self._bot_id}] Throttling for resources: {reason} "
                f"(wait #{wait_count})"
            )
            await asyncio.sleep(self.THROTTLE_DELAY_SECONDS)

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource monitoring statistics."""
        self._init_resource_monitoring()
        return {
            **self._resource_status,
            "throttle_count": self._throttle_count,
            "last_check": self._last_resource_check,
        }

    # =========================================================================
    # Signal Hedging Detection
    # =========================================================================

    async def _check_hedging_conflict(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> tuple[bool, Optional[Dict]]:
        """
        Check if this order would create a hedging conflict with other bots.

        A hedging conflict occurs when:
        - Bot A is LONG on symbol
        - Bot B wants to go SHORT on same symbol
        - This results in paying double fees for zero net exposure

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Order price

        Returns:
            Tuple of (has_conflict, conflict_details)
        """
        try:
            # Check if SignalCoordinator is available
            if not hasattr(self, "_signal_coordinator") or self._signal_coordinator is None:
                return False, None

            # Get active signals for this symbol from other bots
            active_signals = await self._get_active_signals_for_symbol(symbol)

            if not active_signals:
                return False, None

            # Determine our direction
            our_direction = "LONG" if side.upper() == "BUY" else "SHORT"

            # Check for opposite direction signals
            for signal in active_signals:
                if signal.get("bot_id") == self._bot_id:
                    continue  # Skip our own signals

                signal_direction = signal.get("direction", "")

                # Check for opposite directions
                if (our_direction == "LONG" and signal_direction == "SHORT") or \
                   (our_direction == "SHORT" and signal_direction == "LONG"):

                    # Calculate potential double fee waste
                    fee_rate = Decimal("0.0004")  # 0.04% taker fee
                    our_fee = quantity * price * fee_rate
                    their_qty = Decimal(str(signal.get("quantity", 0)))
                    their_price = Decimal(str(signal.get("price", price)))
                    their_fee = their_qty * their_price * fee_rate

                    # Net exposure calculation
                    net_exposure = abs(quantity - their_qty) * price

                    conflict_details = {
                        "conflict_type": "HEDGING",
                        "our_direction": our_direction,
                        "our_quantity": str(quantity),
                        "conflicting_bot": signal.get("bot_id"),
                        "their_direction": signal_direction,
                        "their_quantity": str(their_qty),
                        "net_exposure": str(net_exposure),
                        "wasted_fees": str(our_fee + their_fee),
                        "symbol": symbol,
                    }

                    logger.warning(
                        f"[{self._bot_id}] Hedging conflict detected on {symbol}: "
                        f"We want {our_direction} {quantity}, "
                        f"{signal.get('bot_id')} has {signal_direction} {their_qty}. "
                        f"Net exposure: {net_exposure}, Wasted fees: {our_fee + their_fee}"
                    )

                    return True, conflict_details

            return False, None

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Hedging check failed: {e}")
            return False, None

    async def _get_active_signals_for_symbol(self, symbol: str) -> list[Dict]:
        """
        Get active signals from all bots for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of active signal details
        """
        try:
            if hasattr(self, "_signal_coordinator") and self._signal_coordinator:
                # Use SignalCoordinator if available
                return await self._signal_coordinator.get_active_signals(symbol)

            return []

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Get active signals failed: {e}")
            return []

    # =========================================================================
    # Priority Management
    # =========================================================================

    # Bot priority levels
    PRIORITY_CRITICAL = 100  # Stop loss, emergency
    PRIORITY_HIGH = 75  # Trend following signals
    PRIORITY_NORMAL = 50  # Regular grid orders
    PRIORITY_LOW = 25  # Background rebalancing

    def _init_priority_management(self) -> None:
        """Initialize priority management state."""
        if not hasattr(self, "_bot_priority"):
            self._bot_priority: int = self.PRIORITY_NORMAL
        if not hasattr(self, "_pending_requests"):
            self._pending_requests: Dict[str, Dict] = {}
        if not hasattr(self, "_priority_escalation_enabled"):
            self._priority_escalation_enabled: bool = True

    def set_bot_priority(self, priority: int) -> None:
        """
        Set this bot's priority level.

        Higher priority bots get preferential order execution.

        Args:
            priority: Priority level (0-100, higher = more important)
        """
        self._init_priority_management()
        self._bot_priority = max(0, min(100, priority))
        logger.info(f"[{self._bot_id}] Priority set to {self._bot_priority}")

    def get_bot_priority(self) -> int:
        """Get this bot's current priority level."""
        self._init_priority_management()
        return self._bot_priority

    def _calculate_order_priority(
        self,
        order_type: str,
        is_reduce_only: bool = False,
        is_stop_loss: bool = False,
        signal_urgency: int = 0,
    ) -> int:
        """
        Calculate priority for a specific order.

        Args:
            order_type: Order type (MARKET, LIMIT, etc.)
            is_reduce_only: Whether order reduces position
            is_stop_loss: Whether order is a stop loss
            signal_urgency: Additional urgency factor (0-50)

        Returns:
            Order priority (higher = execute first)
        """
        self._init_priority_management()

        base_priority = self._bot_priority

        # Stop loss orders are always critical
        if is_stop_loss:
            return self.PRIORITY_CRITICAL

        # Reduce-only orders get boost (closing positions is important)
        if is_reduce_only:
            base_priority += 20

        # Market orders slightly higher priority than limit
        if order_type.upper() == "MARKET":
            base_priority += 10

        # Add signal urgency
        base_priority += min(50, max(0, signal_urgency))

        return min(100, base_priority)

    async def _track_pending_request(
        self,
        request_id: str,
        request_type: str,
        priority: int,
    ) -> None:
        """
        Track a pending request for priority escalation.

        Args:
            request_id: Unique request identifier
            request_type: Type of request (order, signal, etc.)
            priority: Initial priority
        """
        self._init_priority_management()

        self._pending_requests[request_id] = {
            "type": request_type,
            "priority": priority,
            "original_priority": priority,
            "submitted_at": time.time(),
            "escalation_count": 0,
        }

    def _escalate_pending_requests(self) -> int:
        """
        Escalate priority of long-waiting requests.

        Requests waiting more than 30 seconds get priority boost.

        Returns:
            Number of requests escalated
        """
        self._init_priority_management()

        if not self._priority_escalation_enabled:
            return 0

        escalated = 0
        current_time = time.time()
        escalation_threshold = 30.0  # 30 seconds

        for request_id, request in list(self._pending_requests.items()):
            wait_time = current_time - request["submitted_at"]

            # Escalate if waiting too long
            if wait_time > escalation_threshold:
                # Increase priority by 10 per escalation, max 3 times
                if request["escalation_count"] < 3:
                    request["priority"] = min(
                        self.PRIORITY_CRITICAL,
                        request["priority"] + 10
                    )
                    request["escalation_count"] += 1
                    escalated += 1

                    logger.info(
                        f"[{self._bot_id}] Escalated {request_id} priority: "
                        f"{request['original_priority']} -> {request['priority']} "
                        f"(waited {wait_time:.1f}s)"
                    )

        return escalated

    def _complete_pending_request(self, request_id: str) -> None:
        """Mark a pending request as completed."""
        self._init_priority_management()
        self._pending_requests.pop(request_id, None)

    def get_priority_stats(self) -> Dict[str, Any]:
        """Get priority management statistics."""
        self._init_priority_management()

        pending_count = len(self._pending_requests)
        avg_wait = 0.0

        if pending_count > 0:
            current_time = time.time()
            total_wait = sum(
                current_time - r["submitted_at"]
                for r in self._pending_requests.values()
            )
            avg_wait = total_wait / pending_count

        return {
            "bot_priority": self._bot_priority,
            "pending_requests": pending_count,
            "average_wait_seconds": avg_wait,
            "escalation_enabled": self._priority_escalation_enabled,
        }

    # =========================================================================
    # Enhanced Pre-Order Validation
    # =========================================================================

    async def _full_pre_order_validation(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
        order_type: str = "MARKET",
        is_reduce_only: bool = False,
        is_stop_loss: bool = False,
        check_hedging: bool = True,
        check_resources: bool = True,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Full pre-order validation including resource and hedging checks.

        This method performs ALL checks:
        1. System resource availability
        2. Balance and capital allocation
        3. Rate limit status
        4. Position limits
        5. Hedging conflict detection

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            leverage: Position leverage
            order_type: Order type
            is_reduce_only: Whether reducing position
            is_stop_loss: Whether stop loss order
            check_hedging: Whether to check for hedging conflicts
            check_resources: Whether to check system resources

        Returns:
            Tuple of (is_allowed, message, details)
        """
        details = {
            "resources_ok": True,
            "balance_ok": False,
            "rate_limit_ok": False,
            "position_ok": False,
            "hedging_ok": True,
            "order_priority": 0,
            "reserved_amount": Decimal("0"),
        }

        # Calculate order priority
        details["order_priority"] = self._calculate_order_priority(
            order_type=order_type,
            is_reduce_only=is_reduce_only,
            is_stop_loss=is_stop_loss,
        )

        # 1. Check system resources (skip for stop loss)
        if check_resources and not is_stop_loss:
            should_throttle, throttle_reason = await self._should_throttle_for_resources()
            if should_throttle:
                # Wait for resources if not critical
                if details["order_priority"] < self.PRIORITY_CRITICAL:
                    resources_ok, resource_msg = await self._wait_for_resources()
                    if not resources_ok:
                        details["resources_ok"] = False
                        return False, resource_msg, details

        # 2. Run comprehensive pre-order check (balance, rate limit, position)
        pre_check_ok, pre_check_msg, pre_check_details = await self._comprehensive_pre_order_check(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            leverage=leverage,
            reserve_capital=True,
        )

        details.update(pre_check_details)

        if not pre_check_ok:
            return False, pre_check_msg, details

        # 3. Check hedging conflict (skip for reduce-only)
        if check_hedging and not is_reduce_only:
            has_hedging, hedging_details = await self._check_hedging_conflict(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
            )

            if has_hedging:
                details["hedging_ok"] = False
                details["hedging_conflict"] = hedging_details

                # Release reserved capital
                if details["reserved_amount"] > 0:
                    await self._release_capital_reservation(details["reserved_amount"])

                return False, (
                    f"Hedging conflict with {hedging_details.get('conflicting_bot')}: "
                    f"Would waste {hedging_details.get('wasted_fees')} in fees"
                ), details

        return True, "All validations passed", details

    def _classify_order_error(self, error: Exception) -> tuple[str, str]:
        """
        Classify order error for proper handling.

        Uses exception type hierarchy first (precise), then falls back
        to string matching (legacy compatibility).

        Args:
            error: The exception from order placement

        Returns:
            Tuple of (error_code, human_readable_message)
        """
        # Type-based classification (preferred — uses core exception hierarchy)
        from src.core.exceptions import (
            AuthenticationError,
            ConnectionError as ExchangeConnectionError,
            InsufficientBalanceError,
            OrderError,
            RateLimitError,
        )

        if isinstance(error, RateLimitError):
            return self.ORDER_ERROR_RATE_LIMIT, f"Rate limit exceeded: {error}"

        if isinstance(error, InsufficientBalanceError):
            return self.ORDER_ERROR_INSUFFICIENT_BALANCE, f"Insufficient balance: {error}"

        if isinstance(error, AuthenticationError):
            return self.ORDER_ERROR_UNKNOWN, f"Authentication error (fatal): {error}"

        if isinstance(error, ExchangeConnectionError):
            return self.ORDER_ERROR_TIMEOUT, f"Connection error (retryable): {error}"

        if isinstance(error, OrderError):
            # Sub-classify OrderError by message
            error_str = str(error).lower()
            if "quantity" in error_str or "lot" in error_str or "min" in error_str:
                return self.ORDER_ERROR_INVALID_QUANTITY, f"Invalid quantity: {error}"
            if "price" in error_str or "tick" in error_str:
                return self.ORDER_ERROR_INVALID_PRICE, f"Invalid price: {error}"
            if "position" in error_str and "limit" in error_str:
                return self.ORDER_ERROR_POSITION_LIMIT, f"Position limit: {error}"
            return self.ORDER_ERROR_UNKNOWN, f"Order error: {error}"

        if isinstance(error, (TimeoutError, asyncio.TimeoutError, OSError)):
            return self.ORDER_ERROR_TIMEOUT, f"Timeout/network error (retryable): {error}"

        # Fallback: string-based classification (legacy compatibility)
        error_str = str(error).lower()

        if "insufficient" in error_str or "balance" in error_str or "margin" in error_str:
            return self.ORDER_ERROR_INSUFFICIENT_BALANCE, "Insufficient balance or margin"

        if "quantity" in error_str or "lot" in error_str or "min" in error_str:
            return self.ORDER_ERROR_INVALID_QUANTITY, "Invalid order quantity"

        if "price" in error_str or "tick" in error_str:
            return self.ORDER_ERROR_INVALID_PRICE, "Invalid price"

        if "position" in error_str and "limit" in error_str:
            return self.ORDER_ERROR_POSITION_LIMIT, "Position limit exceeded"

        if "rate" in error_str or "limit" in error_str or "429" in error_str:
            return self.ORDER_ERROR_RATE_LIMIT, "Rate limit exceeded"

        if "timeout" in error_str or "timed out" in error_str:
            return self.ORDER_ERROR_TIMEOUT, "Order timed out"

        return self.ORDER_ERROR_UNKNOWN, f"Unknown error: {error}"

    async def _handle_order_rejection(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        error: Exception,
    ) -> bool:
        """
        Handle order rejection with proper logging and notification.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            error: The exception from order placement

        Returns:
            True if error was handled and retry is possible
        """
        error_code, message = self._classify_order_error(error)

        logger.warning(
            f"[{self._bot_id}] Order rejected: {error_code} - {message}\n"
            f"  Symbol: {symbol}, Side: {side}, Qty: {quantity}"
        )

        # Track rejection for statistics
        if not hasattr(self, "_order_rejection_count"):
            self._order_rejection_count = 0
        self._order_rejection_count += 1

        # Notify for important rejections
        if error_code in [
            self.ORDER_ERROR_INSUFFICIENT_BALANCE,
            self.ORDER_ERROR_POSITION_LIMIT,
        ]:
            if self._notifier:
                await self._notifier.send_warning(
                    title=f"🚫 Order Rejected: {error_code}",
                    message=(
                        f"Bot: {self._bot_id}\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {side}\n"
                        f"Quantity: {quantity}\n"
                        f"Reason: {message}"
                    ),
                )

        # Return True if retry is possible
        return error_code == self.ORDER_ERROR_RATE_LIMIT

    async def _on_close_position_failure(self, symbol: str, error: Exception) -> None:
        """
        Handle close position failure after all retries exhausted.

        Pauses the bot and sends critical alert requiring manual intervention.
        """
        logger.critical(
            f"[{self._bot_id}] CLOSE POSITION FAILED for {symbol}: {error}\n"
            f"Bot is being PAUSED - manual intervention required!"
        )

        # Pause the bot
        self._init_strategy_risk_tracking()
        self._strategy_risk["is_paused_by_risk"] = True

        if self._notifier:
            await self._notifier.send_alert(
                title=f"🚨 CRITICAL: Close Position Failed - {self._bot_id}",
                message=(
                    f"Symbol: {symbol}\n"
                    f"Error: {error}\n"
                    f"Action: Bot PAUSED - manual intervention required!\n"
                    f"Check exchange for open positions and resolve manually."
                ),
            )

    # =========================================================================
    # Order Precision Handling (Price/Quantity Normalization)
    # =========================================================================

    def _init_symbol_info_cache(self) -> None:
        """Initialize symbol info cache."""
        if not hasattr(self, "_symbol_info_cache"):
            self._symbol_info_cache: Dict[str, Any] = {}

    async def _get_symbol_info(self, symbol: str) -> Optional[Any]:
        """
        Get and cache symbol info from exchange.

        Args:
            symbol: Trading symbol

        Returns:
            SymbolInfo object or None
        """
        self._init_symbol_info_cache()

        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        try:
            symbol_info = await self._exchange.get_symbol_info(
                symbol=symbol,
                market=MarketType.FUTURES,
            )
            if symbol_info:
                self._symbol_info_cache[symbol] = symbol_info
                logger.debug(
                    f"[{self._bot_id}] Cached symbol info for {symbol}: "
                    f"tick_size={symbol_info.tick_size}, "
                    f"step_size={symbol_info.step_size}, "
                    f"min_qty={symbol_info.min_quantity}, "
                    f"min_notional={symbol_info.min_notional}"
                )
            return symbol_info
        except Exception as e:
            logger.warning(f"[{self._bot_id}] Failed to get symbol info: {e}")
            return None

    def _normalize_price(
        self,
        price: Decimal,
        symbol_info: Optional[Any] = None,
        tick_size: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Normalize price to exchange tick size.

        Args:
            price: Raw price
            symbol_info: SymbolInfo object (optional)
            tick_size: Override tick size (optional)

        Returns:
            Price rounded to valid tick size
        """
        if tick_size is None and symbol_info:
            tick_size = getattr(symbol_info, "tick_size", None)

        if tick_size is None or tick_size <= 0:
            # Fallback: round to 2 decimal places
            return price.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        # Round down to nearest tick size
        # Example: price=100.123, tick_size=0.01 -> 100.12
        normalized = (price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
        return normalized

    def _normalize_quantity(
        self,
        quantity: Decimal,
        symbol_info: Optional[Any] = None,
        step_size: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Normalize quantity to exchange step size.

        Args:
            quantity: Raw quantity
            symbol_info: SymbolInfo object (optional)
            step_size: Override step size (optional)

        Returns:
            Quantity rounded to valid step size
        """
        if step_size is None and symbol_info:
            step_size = getattr(symbol_info, "step_size", None)

        if step_size is None or step_size <= 0:
            # Fallback: round to 3 decimal places
            return quantity.quantize(Decimal("0.001"), rounding=ROUND_DOWN)

        # Round down to nearest step size
        # Example: quantity=0.12345, step_size=0.001 -> 0.123
        normalized = (quantity / step_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * step_size
        return normalized

    async def _validate_order_precision(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
    ) -> tuple[bool, Decimal, Decimal, str]:
        """
        Validate and normalize price/quantity for exchange requirements.

        Args:
            symbol: Trading symbol
            price: Order price
            quantity: Order quantity

        Returns:
            Tuple of (is_valid, normalized_price, normalized_quantity, message)
        """
        symbol_info = await self._get_symbol_info(symbol)

        # Normalize price and quantity
        norm_price = self._normalize_price(price, symbol_info)
        norm_quantity = self._normalize_quantity(quantity, symbol_info)

        if symbol_info:
            # Check minimum quantity
            min_qty = getattr(symbol_info, "min_quantity", Decimal("0"))
            if norm_quantity < min_qty:
                return False, norm_price, norm_quantity, (
                    f"Quantity {norm_quantity} below minimum {min_qty}"
                )

            # Check minimum notional value
            min_notional = getattr(symbol_info, "min_notional", Decimal("0"))
            notional_value = norm_price * norm_quantity
            if min_notional > 0 and notional_value < min_notional:
                return False, norm_price, norm_quantity, (
                    f"Notional value {notional_value:.2f} below minimum {min_notional}"
                )

        # Basic validation
        if norm_price <= 0:
            return False, norm_price, norm_quantity, "Invalid price after normalization"

        if norm_quantity <= 0:
            return False, norm_price, norm_quantity, "Invalid quantity after normalization"

        return True, norm_price, norm_quantity, "OK"

    # =========================================================================
    # Order Deduplication (Prevent Duplicate Orders on Retry)
    # =========================================================================

    # Pending order tracking window (seconds)
    PENDING_ORDER_WINDOW_SECONDS = 60

    def _init_pending_order_tracking(self) -> None:
        """Initialize pending order tracking."""
        if not hasattr(self, "_pending_orders"):
            # Dict: order_key -> (timestamp, order_details)
            self._pending_orders: Dict[str, tuple[float, Dict[str, Any]]] = {}
        if not hasattr(self, "_last_order_cleanup"):
            self._last_order_cleanup: float = 0

    def _generate_order_key(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
    ) -> str:
        """
        Generate unique key for order deduplication.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Order price (optional for market orders)

        Returns:
            Unique order key
        """
        # Normalize Decimals to remove trailing zeros for consistent keys
        qty_normalized = quantity.normalize()
        # For market orders, use a simpler key
        if price is None:
            return f"{symbol}_{side}_{qty_normalized}"
        price_normalized = price.normalize()
        return f"{symbol}_{side}_{qty_normalized}_{price_normalized}"

    def _cleanup_old_pending_orders(self) -> None:
        """Remove expired pending orders."""
        now = time.time()

        # Only cleanup every 10 seconds
        if now - self._last_order_cleanup < 10:
            return

        self._last_order_cleanup = now
        expired_keys = []

        for key, (timestamp, _) in self._pending_orders.items():
            if now - timestamp > self.PENDING_ORDER_WINDOW_SECONDS:
                expired_keys.append(key)

        for key in expired_keys:
            del self._pending_orders[key]

    def _is_duplicate_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
    ) -> bool:
        """
        Check if order is a potential duplicate.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price

        Returns:
            True if this appears to be a duplicate order
        """
        self._init_pending_order_tracking()
        self._cleanup_old_pending_orders()

        order_key = self._generate_order_key(symbol, side, quantity, price)
        now = time.time()

        if order_key in self._pending_orders:
            timestamp, details = self._pending_orders[order_key]
            age = now - timestamp
            logger.warning(
                f"[{self._bot_id}] Duplicate order detected: {order_key} "
                f"(pending for {age:.1f}s)"
            )
            return True

        return False

    def _mark_order_pending(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
    ) -> str:
        """
        Mark order as pending to prevent duplicates.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price

        Returns:
            Order key for later clearing
        """
        self._init_pending_order_tracking()

        order_key = self._generate_order_key(symbol, side, quantity, price)
        self._pending_orders[order_key] = (
            time.time(),
            {"symbol": symbol, "side": side, "quantity": str(quantity), "price": str(price)},
        )
        return order_key

    def _clear_pending_order(self, order_key: str) -> None:
        """
        Clear pending order after completion/failure.

        Args:
            order_key: Order key from _mark_order_pending
        """
        self._init_pending_order_tracking()
        if order_key in self._pending_orders:
            del self._pending_orders[order_key]

    async def _check_existing_position_before_retry(
        self,
        symbol: str,
        expected_side: str,
    ) -> bool:
        """
        Check if position already exists before retry (to detect if original order went through).

        Args:
            symbol: Trading symbol
            expected_side: Expected position side

        Returns:
            True if position exists (don't retry)
        """
        try:
            exchange_pos = await self._sync_position_from_exchange()

            if exchange_pos:
                exchange_side = exchange_pos.get("side", "").upper()
                exchange_qty = exchange_pos.get("quantity", Decimal("0"))

                if exchange_qty > 0 and exchange_side == expected_side.upper():
                    logger.info(
                        f"[{self._bot_id}] Position already exists from previous attempt: "
                        f"{exchange_side} {exchange_qty} - skipping retry"
                    )
                    return True

            return False

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Error checking position before retry: {e}")
            return False

    # =========================================================================
    # Order Timeout Handling (with Deduplication)
    # =========================================================================

    # Default order timeout (seconds)
    DEFAULT_ORDER_TIMEOUT_SECONDS = 45

    # Maximum retries for timed-out orders
    MAX_ORDER_RETRIES = 2

    async def _place_order_with_timeout(
        self,
        order_func: Callable,
        timeout_seconds: Optional[float] = None,
        retry_count: int = 0,
        order_side: Optional[str] = None,
        order_quantity: Optional[Decimal] = None,
    ) -> Optional[Any]:
        """
        Place order with timeout, retry logic, and deduplication check.

        Args:
            order_func: Async function that places the order
            timeout_seconds: Timeout in seconds (default: DEFAULT_ORDER_TIMEOUT_SECONDS)
            retry_count: Current retry attempt
            order_side: Order side for position check on retry (BUY/SELL)
            order_quantity: Order quantity for deduplication

        Returns:
            Order result or None if failed
        """
        if timeout_seconds is None:
            timeout_seconds = self.DEFAULT_ORDER_TIMEOUT_SECONDS

        try:
            # Execute order with timeout
            result = await asyncio.wait_for(order_func(), timeout=timeout_seconds)
            return result

        except asyncio.TimeoutError:
            logger.warning(
                f"[{self._bot_id}] Order timed out after {timeout_seconds}s "
                f"(attempt {retry_count + 1}/{self.MAX_ORDER_RETRIES + 1})"
            )

            # Before retry, check if position already exists (original order may have succeeded)
            if retry_count < self.MAX_ORDER_RETRIES:
                # Check if the original order actually went through
                if order_side:
                    # Map BUY/SELL to LONG/SHORT
                    expected_side = "LONG" if order_side.upper() == "BUY" else "SHORT"
                    position_exists = await self._check_existing_position_before_retry(
                        symbol=self.symbol,
                        expected_side=expected_side,
                    )
                    if position_exists:
                        logger.info(
                            f"[{self._bot_id}] Original order succeeded - "
                            f"position found on exchange, skipping retry"
                        )
                        # Return None but don't consider this a failure
                        # The bot should detect the position via reconciliation
                        return None

                logger.info(f"[{self._bot_id}] Retrying order...")
                await asyncio.sleep(1)  # Brief delay before retry
                return await self._place_order_with_timeout(
                    order_func, timeout_seconds, retry_count + 1,
                    order_side=order_side, order_quantity=order_quantity
                )

            # Max retries reached
            logger.error(
                f"[{self._bot_id}] Order failed after {self.MAX_ORDER_RETRIES + 1} attempts"
            )

            if self._notifier:
                await self._notifier.send_warning(
                    title="⏱️ Order Timeout",
                    message=(
                        f"Bot: {self._bot_id}\n"
                        f"Order timed out after {self.MAX_ORDER_RETRIES + 1} attempts\n"
                        f"Check exchange manually"
                    ),
                )

            return None

        except Exception as e:
            # Handle other errors
            await self._handle_order_rejection(
                self.symbol, "UNKNOWN", Decimal("0"), e
            )
            return None

    async def _verify_order_fill(
        self,
        order_id: str,
        symbol: str,
        expected_quantity: Decimal,
        timeout_seconds: float = 10.0,
        poll_interval: float = 1.0,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify order was filled by polling order status.

        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            expected_quantity: Expected fill quantity
            timeout_seconds: Maximum time to wait for fill
            poll_interval: Time between status checks

        Returns:
            Tuple of (is_filled, order_details)
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"[{self._bot_id}] Order fill verification timed out: {order_id}"
                )
                return False, None

            try:
                # Query order status
                order = await self._exchange.futures.get_order(
                    symbol=symbol,
                    order_id=order_id,
                )

                if order:
                    status = getattr(order, "status", "").upper()
                    filled_qty = getattr(order, "filled_qty", Decimal("0"))

                    if status == "FILLED":
                        logger.debug(
                            f"[{self._bot_id}] Order {order_id} filled: {filled_qty}"
                        )
                        return True, {
                            "order_id": order_id,
                            "status": status,
                            "filled_qty": filled_qty,
                            "avg_price": getattr(order, "avg_price", None),
                        }

                    if status in ["CANCELED", "CANCELLED", "REJECTED", "EXPIRED"]:
                        logger.warning(
                            f"[{self._bot_id}] Order {order_id} ended with status: {status}"
                        )
                        return False, {"order_id": order_id, "status": status}

                    # Still pending - continue polling
                    logger.debug(
                        f"[{self._bot_id}] Order {order_id} status: {status}, "
                        f"filled: {filled_qty}/{expected_quantity}"
                    )

            except Exception as e:
                logger.warning(f"[{self._bot_id}] Error checking order status: {e}")

            await asyncio.sleep(poll_interval)

    # =========================================================================
    # Order Cancellation with Verification
    # =========================================================================

    # Cancel retry configuration
    CANCEL_MAX_RETRIES = 3
    CANCEL_BASE_DELAY_SECONDS = 0.5  # Exponential backoff base
    CANCEL_TIMEOUT_SECONDS = 10.0

    # Cancel error categories
    CANCEL_ERROR_ALREADY_FILLED = "ALREADY_FILLED"
    CANCEL_ERROR_ALREADY_CANCELLED = "ALREADY_CANCELLED"
    CANCEL_ERROR_NOT_FOUND = "NOT_FOUND"
    CANCEL_ERROR_INVALID_STATE = "INVALID_STATE"
    CANCEL_ERROR_TIMEOUT = "TIMEOUT"
    CANCEL_ERROR_NETWORK = "NETWORK"
    CANCEL_ERROR_UNKNOWN = "UNKNOWN"

    def _classify_cancel_error(self, error: Exception) -> tuple[str, bool]:
        """
        Classify cancellation error for handling decision.

        Args:
            error: The exception raised

        Returns:
            Tuple of (error_category, should_retry)
        """
        error_str = str(error).lower()
        error_code = ""

        # Extract error code if available (Binance format: "APIError(code=-2011)")
        if "code=" in error_str:
            try:
                import re
                match = re.search(r"code=(-?\d+)", error_str)
                if match:
                    error_code = match.group(1)
            except Exception as e:
                logger.debug(f"Failed to parse error code from string: {e}")

        # Order not found / unknown order
        if "unknown order" in error_str or "not found" in error_str or error_code in ["-2011", "-2013"]:
            # Need to check if it was filled or cancelled
            return self.CANCEL_ERROR_NOT_FOUND, False

        # Order already cancelled
        if "already canceled" in error_str or "already cancelled" in error_str:
            return self.CANCEL_ERROR_ALREADY_CANCELLED, False

        # Invalid order state
        if "invalid" in error_str and "state" in error_str:
            return self.CANCEL_ERROR_INVALID_STATE, False

        # Timeout
        if "timeout" in error_str:
            return self.CANCEL_ERROR_TIMEOUT, True

        # Network errors
        if "connection" in error_str or "network" in error_str or "refused" in error_str:
            return self.CANCEL_ERROR_NETWORK, True

        # Rate limit
        if "rate" in error_str or "too many" in error_str:
            return self.CANCEL_ERROR_NETWORK, True

        return self.CANCEL_ERROR_UNKNOWN, True

    async def _get_order_final_state(
        self,
        order_id: str,
        symbol: str,
    ) -> Optional[Dict]:
        """
        Get final state of an order after cancellation attempt.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Order state dict or None if unable to fetch
        """
        try:
            order = await self._exchange.futures.get_order(
                symbol=symbol,
                order_id=order_id,
            )

            if order:
                return {
                    "order_id": order_id,
                    "status": getattr(order, "status", "").upper(),
                    "filled_qty": getattr(order, "filled_qty", Decimal("0")),
                    "quantity": getattr(order, "quantity", Decimal("0")),
                    "avg_price": getattr(order, "avg_price", None),
                    "side": getattr(order, "side", ""),
                }
        except Exception as e:
            logger.warning(f"[{self._bot_id}] Error getting order state: {e}")

        return None

    async def _cancel_order_with_timeout(
        self,
        order_id: str,
        symbol: str,
        timeout_seconds: float = 10.0,
    ) -> bool:
        """
        Cancel order with timeout protection (simple version for backward compatibility).

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            timeout_seconds: Timeout in seconds

        Returns:
            True if cancelled successfully or order already gone
        """
        result = await self._cancel_order_with_verification(
            order_id=order_id,
            symbol=symbol,
            timeout_seconds=timeout_seconds,
            max_retries=1,  # Single attempt for simple version
            verify_state=False,  # Skip verification for simple version
        )
        return result["is_cancelled"] or result["is_filled"]

    async def _cancel_order_with_verification(
        self,
        order_id: str,
        symbol: str,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        verify_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Cancel order with retry, verification, and detailed result.

        This method provides robust cancellation with:
        1. Retry logic with exponential backoff
        2. Post-cancel state verification
        3. Partial fill detection
        4. Detailed error classification

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            timeout_seconds: Timeout per attempt
            max_retries: Maximum retry attempts
            verify_state: Whether to verify final state after cancel

        Returns:
            Dict with:
            - is_cancelled: True if order was successfully cancelled
            - is_filled: True if order was already fully filled
            - partial_fill: True if order had partial fill
            - filled_qty: Quantity that was filled
            - unfilled_qty: Quantity that was not filled
            - final_status: Final order status
            - error_category: Error category if failed
            - attempts: Number of attempts made
        """
        if timeout_seconds is None:
            timeout_seconds = self.CANCEL_TIMEOUT_SECONDS
        if max_retries is None:
            max_retries = self.CANCEL_MAX_RETRIES

        result = {
            "order_id": order_id,
            "is_cancelled": False,
            "is_filled": False,
            "partial_fill": False,
            "filled_qty": Decimal("0"),
            "unfilled_qty": Decimal("0"),
            "final_status": "",
            "error_category": None,
            "error_message": None,
            "attempts": 0,
        }

        for attempt in range(max_retries):
            result["attempts"] = attempt + 1

            try:
                # Attempt cancellation
                await asyncio.wait_for(
                    self._exchange.futures.cancel_order(
                        symbol=symbol,
                        order_id=order_id,
                    ),
                    timeout=timeout_seconds,
                )

                logger.info(f"[{self._bot_id}] Order {order_id} cancel request sent")
                result["is_cancelled"] = True
                break

            except asyncio.TimeoutError:
                error_category = self.CANCEL_ERROR_TIMEOUT
                should_retry = True
                result["error_category"] = error_category
                result["error_message"] = f"Cancel timed out after {timeout_seconds}s"

                logger.warning(
                    f"[{self._bot_id}] Cancel order timeout for {order_id} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

            except Exception as e:
                error_category, should_retry = self._classify_cancel_error(e)
                result["error_category"] = error_category
                result["error_message"] = str(e)

                # Handle specific error categories
                if error_category == self.CANCEL_ERROR_NOT_FOUND:
                    # Order not found - need to verify if filled or already cancelled
                    logger.debug(
                        f"[{self._bot_id}] Order {order_id} not found - checking state"
                    )
                    verify_state = True  # Force verification
                    break

                elif error_category == self.CANCEL_ERROR_ALREADY_CANCELLED:
                    result["is_cancelled"] = True
                    logger.info(
                        f"[{self._bot_id}] Order {order_id} was already cancelled"
                    )
                    break

                elif error_category == self.CANCEL_ERROR_INVALID_STATE:
                    # Could be filled during cancel - need to verify
                    logger.warning(
                        f"[{self._bot_id}] Order {order_id} invalid state for cancel"
                    )
                    verify_state = True
                    break

                else:
                    logger.warning(
                        f"[{self._bot_id}] Cancel error for {order_id}: {e} "
                        f"(category={error_category}, attempt {attempt + 1}/{max_retries})"
                    )

            # Retry with exponential backoff if applicable
            if should_retry and attempt < max_retries - 1:
                delay = self.CANCEL_BASE_DELAY_SECONDS * (2 ** attempt)
                await asyncio.sleep(delay)
            elif not should_retry:
                break

        # Verify final state if requested or needed
        if verify_state:
            final_state = await self._get_order_final_state(order_id, symbol)

            if final_state:
                result["final_status"] = final_state["status"]
                result["filled_qty"] = final_state["filled_qty"]

                total_qty = final_state["quantity"]
                if total_qty:
                    result["unfilled_qty"] = total_qty - final_state["filled_qty"]

                # Determine actual outcome
                if final_state["status"] == "FILLED":
                    result["is_filled"] = True
                    result["is_cancelled"] = False
                    logger.info(
                        f"[{self._bot_id}] Order {order_id} was filled "
                        f"(qty={final_state['filled_qty']})"
                    )

                elif final_state["status"] in ["CANCELED", "CANCELLED"]:
                    result["is_cancelled"] = True
                    if final_state["filled_qty"] > 0:
                        result["partial_fill"] = True
                        logger.info(
                            f"[{self._bot_id}] Order {order_id} cancelled with partial fill "
                            f"(filled={final_state['filled_qty']})"
                        )
                    else:
                        logger.info(
                            f"[{self._bot_id}] Order {order_id} cancelled (no fill)"
                        )

                elif final_state["status"] == "EXPIRED":
                    result["is_cancelled"] = True  # Treat as cancelled
                    if final_state["filled_qty"] > 0:
                        result["partial_fill"] = True
                    logger.info(f"[{self._bot_id}] Order {order_id} expired")

                elif final_state["status"] in ["NEW", "PARTIALLY_FILLED"]:
                    # Cancel didn't take effect yet - may need another attempt
                    if result["attempts"] >= max_retries:
                        logger.error(
                            f"[{self._bot_id}] Order {order_id} still active after "
                            f"{max_retries} cancel attempts"
                        )
                    else:
                        result["error_category"] = self.CANCEL_ERROR_INVALID_STATE

        return result

    async def _cancel_all_open_orders(
        self,
        symbol: str,
        verify_each: bool = True,
    ) -> Dict[str, Any]:
        """
        Cancel all open orders for a symbol with verification.

        Args:
            symbol: Trading symbol
            verify_each: Whether to verify each cancellation

        Returns:
            Dict with cancellation results
        """
        results = {
            "total_orders": 0,
            "cancelled": 0,
            "already_filled": 0,
            "failed": 0,
            "partial_fills": [],
            "errors": [],
        }

        try:
            # Get all open orders
            open_orders = await self._exchange.futures.get_open_orders(symbol=symbol)
            results["total_orders"] = len(open_orders)

            if not open_orders:
                logger.debug(f"[{self._bot_id}] No open orders to cancel for {symbol}")
                return results

            # Cancel each order
            for order in open_orders:
                order_id = str(getattr(order, "order_id", ""))
                if not order_id:
                    continue

                cancel_result = await self._cancel_order_with_verification(
                    order_id=order_id,
                    symbol=symbol,
                    verify_state=verify_each,
                )

                if cancel_result["is_cancelled"]:
                    results["cancelled"] += 1
                    if cancel_result["partial_fill"]:
                        results["partial_fills"].append({
                            "order_id": order_id,
                            "filled_qty": str(cancel_result["filled_qty"]),
                        })
                elif cancel_result["is_filled"]:
                    results["already_filled"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "order_id": order_id,
                        "error": cancel_result["error_message"],
                    })

            logger.info(
                f"[{self._bot_id}] Cancelled {results['cancelled']}/{results['total_orders']} "
                f"orders for {symbol}"
            )

        except Exception as e:
            logger.error(f"[{self._bot_id}] Error cancelling all orders: {e}")
            results["errors"].append({"error": str(e)})

        return results

    async def _safe_cancel_with_position_update(
        self,
        order_id: str,
        symbol: str,
        expected_side: str,
    ) -> Dict[str, Any]:
        """
        Safely cancel order and update position based on any partial fills.

        This is the recommended method for cancelling orders when you need
        to properly account for any fills that occurred.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            expected_side: Expected order side (BUY/SELL) for position update

        Returns:
            Dict with cancel result and position update info
        """
        result = {
            "cancel_success": False,
            "position_updated": False,
            "filled_qty": Decimal("0"),
            "avg_price": None,
            "needs_position_sync": False,
        }

        # Cancel with full verification
        cancel_result = await self._cancel_order_with_verification(
            order_id=order_id,
            symbol=symbol,
            verify_state=True,
        )

        result["cancel_success"] = cancel_result["is_cancelled"] or cancel_result["is_filled"]
        result["filled_qty"] = cancel_result["filled_qty"]

        # If there was any fill, we need to update position
        if cancel_result["filled_qty"] > 0:
            # Get full order details for avg_price
            order_state = await self._get_order_final_state(order_id, symbol)
            if order_state:
                result["avg_price"] = order_state.get("avg_price")

            # Flag that position needs sync
            result["needs_position_sync"] = True

            logger.info(
                f"[{self._bot_id}] Order {order_id} had fill during cancel: "
                f"{cancel_result['filled_qty']} @ {result['avg_price']}"
            )

            # Notify about the unexpected fill
            if self._notifier and cancel_result["partial_fill"]:
                await self._notifier.send_warning(
                    title="⚠️ Partial Fill During Cancel",
                    message=(
                        f"Bot: {self._bot_id}\n"
                        f"Order: {order_id}\n"
                        f"Filled: {cancel_result['filled_qty']}\n"
                        f"Side: {expected_side}\n"
                        f"Please verify position"
                    ),
                )

        return result

    async def _cancel_algo_order_with_verification(
        self,
        algo_id: str,
        symbol: str,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Cancel algo order (stop loss/take profit) with retry and verification.

        Args:
            algo_id: Algo order ID to cancel
            symbol: Trading symbol
            max_retries: Maximum retry attempts

        Returns:
            Dict with:
            - is_cancelled: True if cancelled successfully
            - was_triggered: True if order was triggered (executed)
            - error_category: Error category if failed
            - attempts: Number of attempts made
        """
        if max_retries is None:
            max_retries = self.CANCEL_MAX_RETRIES

        result = {
            "algo_id": algo_id,
            "is_cancelled": False,
            "was_triggered": False,
            "error_category": None,
            "error_message": None,
            "attempts": 0,
        }

        for attempt in range(max_retries):
            result["attempts"] = attempt + 1

            try:
                await asyncio.wait_for(
                    self._exchange.futures.cancel_algo_order(
                        symbol=symbol,
                        algo_id=algo_id,
                    ),
                    timeout=self.CANCEL_TIMEOUT_SECONDS,
                )

                logger.info(f"[{self._bot_id}] Algo order {algo_id} cancelled")
                result["is_cancelled"] = True
                return result

            except asyncio.TimeoutError:
                result["error_category"] = self.CANCEL_ERROR_TIMEOUT
                result["error_message"] = "Cancel timed out"
                logger.warning(
                    f"[{self._bot_id}] Algo cancel timeout for {algo_id} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

            except Exception as e:
                error_str = str(e).lower()
                result["error_message"] = str(e)

                # Check if already triggered (executed)
                if "triggered" in error_str or "executed" in error_str:
                    result["was_triggered"] = True
                    result["error_category"] = self.CANCEL_ERROR_ALREADY_FILLED
                    logger.info(
                        f"[{self._bot_id}] Algo order {algo_id} was already triggered"
                    )
                    return result

                # Check if not found (already cancelled or expired)
                elif "not found" in error_str or "unknown" in error_str:
                    result["is_cancelled"] = True  # Treat as success
                    result["error_category"] = self.CANCEL_ERROR_NOT_FOUND
                    logger.debug(
                        f"[{self._bot_id}] Algo order {algo_id} already gone"
                    )
                    return result

                else:
                    error_category, should_retry = self._classify_cancel_error(e)
                    result["error_category"] = error_category
                    logger.warning(
                        f"[{self._bot_id}] Algo cancel error for {algo_id}: {e}"
                    )

                    if not should_retry:
                        return result

            # Retry with exponential backoff
            if attempt < max_retries - 1:
                delay = self.CANCEL_BASE_DELAY_SECONDS * (2 ** attempt)
                await asyncio.sleep(delay)

        return result

    # =========================================================================
    # Order Type Validation and Fallback
    # =========================================================================

    # Supported order types for different exchanges
    SUPPORTED_ORDER_TYPES = ["MARKET", "LIMIT", "STOP_MARKET", "STOP_LIMIT"]

    async def _validate_order_type(
        self,
        order_type: str,
        symbol: str,
    ) -> tuple[bool, str]:
        """
        Validate order type is supported by exchange.

        Args:
            order_type: Order type to validate
            symbol: Trading symbol

        Returns:
            Tuple of (is_valid, message)
        """
        order_type_upper = order_type.upper()

        if order_type_upper not in self.SUPPORTED_ORDER_TYPES:
            return False, f"Unsupported order type: {order_type}"

        # Check exchange-specific support via symbol info
        symbol_info = await self._get_symbol_info(symbol)
        if symbol_info:
            supported_types = getattr(symbol_info, "order_types", None)
            if supported_types and order_type_upper not in supported_types:
                return False, f"Order type {order_type} not supported for {symbol}"

        return True, "OK"

    def _get_fallback_order_type(self, original_type: str) -> Optional[str]:
        """
        Get fallback order type if original is not supported.

        Args:
            original_type: Original order type

        Returns:
            Fallback order type or None
        """
        fallback_map = {
            "LIMIT": "MARKET",  # If LIMIT fails, use MARKET
            "STOP_LIMIT": "STOP_MARKET",  # If STOP_LIMIT fails, use STOP_MARKET
        }
        return fallback_map.get(original_type.upper())

    # =========================================================================
    # Slippage Protection
    # =========================================================================

    # Default slippage limits
    DEFAULT_MAX_SLIPPAGE_PCT = Decimal("0.5")  # 0.5% max slippage
    WARNING_SLIPPAGE_PCT = Decimal("0.1")  # 0.1% warning threshold

    def _init_slippage_tracking(self) -> None:
        """Initialize slippage tracking."""
        if not hasattr(self, "_slippage_records"):
            self._slippage_records: list = []
        if not hasattr(self, "_max_slippage_pct"):
            self._max_slippage_pct = self.DEFAULT_MAX_SLIPPAGE_PCT

    def _record_slippage(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
        side: str,
        quantity: Decimal,
    ) -> Decimal:
        """
        Record and calculate slippage for an order.

        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            side: Order side (BUY/SELL)
            quantity: Order quantity

        Returns:
            Slippage percentage (positive = unfavorable)
        """
        self._init_slippage_tracking()

        # Calculate slippage (positive = unfavorable for trader)
        if side.upper() == "BUY":
            # For BUY: slippage = (actual - expected) / expected
            slippage = actual_price - expected_price
        else:
            # For SELL: slippage = (expected - actual) / expected
            slippage = expected_price - actual_price

        slippage_pct = (slippage / expected_price * Decimal("100")) if expected_price > 0 else Decimal("0")

        # Record
        record = {
            "timestamp": time.time(),
            "expected_price": str(expected_price),
            "actual_price": str(actual_price),
            "side": side,
            "quantity": str(quantity),
            "slippage": str(slippage),
            "slippage_pct": str(slippage_pct),
        }
        self._slippage_records.append(record)

        # Keep last 100 records
        if len(self._slippage_records) > 100:
            self._slippage_records = self._slippage_records[-100:]

        # Log significant slippage
        if abs(slippage_pct) > self.WARNING_SLIPPAGE_PCT:
            logger.warning(
                f"[{self._bot_id}] Significant slippage: {slippage_pct:.3f}% "
                f"(expected={expected_price}, actual={actual_price}, side={side})"
            )

        return slippage_pct

    def _check_slippage_acceptable(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
        side: str,
        max_slippage_pct: Optional[Decimal] = None,
    ) -> tuple[bool, Decimal]:
        """
        Check if slippage is within acceptable limits.

        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            side: Order side (BUY/SELL)
            max_slippage_pct: Maximum acceptable slippage percentage

        Returns:
            Tuple of (is_acceptable, slippage_pct)
        """
        if max_slippage_pct is None:
            max_slippage_pct = self._max_slippage_pct if hasattr(self, "_max_slippage_pct") else self.DEFAULT_MAX_SLIPPAGE_PCT

        # Calculate slippage
        if side.upper() == "BUY":
            slippage = actual_price - expected_price
        else:
            slippage = expected_price - actual_price

        slippage_pct = (slippage / expected_price * Decimal("100")) if expected_price > 0 else Decimal("0")

        # Positive slippage means unfavorable
        is_acceptable = slippage_pct <= max_slippage_pct

        if not is_acceptable:
            logger.warning(
                f"[{self._bot_id}] Slippage exceeds limit: {slippage_pct:.3f}% > {max_slippage_pct}%"
            )

        return is_acceptable, slippage_pct

    def get_slippage_stats(self) -> Dict[str, Any]:
        """
        Get slippage statistics.

        Returns:
            Dictionary with slippage statistics
        """
        self._init_slippage_tracking()

        if not self._slippage_records:
            return {
                "count": 0,
                "avg_slippage_pct": "0",
                "max_slippage_pct": "0",
                "min_slippage_pct": "0",
            }

        slippages = [Decimal(r["slippage_pct"]) for r in self._slippage_records]
        return {
            "count": len(slippages),
            "avg_slippage_pct": str(sum(slippages) / len(slippages)),
            "max_slippage_pct": str(max(slippages)),
            "min_slippage_pct": str(min(slippages)),
            "recent": self._slippage_records[-5:],
        }

    async def _estimate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """
        Estimate slippage before placing order using orderbook.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity

        Returns:
            Tuple of (estimated_fill_price, estimated_slippage_pct)
        """
        try:
            # Get orderbook
            orderbook = await self._exchange.futures.get_orderbook(symbol, limit=20)

            if not orderbook:
                return Decimal("0"), Decimal("0")

            # Use asks for BUY, bids for SELL
            levels = orderbook.asks if side.upper() == "BUY" else orderbook.bids

            if not levels:
                return Decimal("0"), Decimal("0")

            # Best price
            best_price = levels[0][0]

            # Calculate average fill price
            remaining_qty = quantity
            total_cost = Decimal("0")

            for price, qty in levels:
                fill_qty = min(remaining_qty, qty)
                total_cost += price * fill_qty
                remaining_qty -= fill_qty
                if remaining_qty <= 0:
                    break

            if remaining_qty > 0:
                # Not enough liquidity
                logger.warning(
                    f"[{self._bot_id}] Insufficient liquidity for {quantity} {symbol}"
                )
                return Decimal("0"), Decimal("100")  # 100% slippage = no liquidity

            avg_fill_price = total_cost / quantity
            slippage_pct = abs(avg_fill_price - best_price) / best_price * Decimal("100")

            return avg_fill_price, slippage_pct

        except Exception as e:
            logger.warning(f"[{self._bot_id}] Failed to estimate slippage: {e}")
            return Decimal("0"), Decimal("0")

    # =========================================================================
    # Partial Fill Handling
    # =========================================================================

    # Minimum fill percentage to consider order successful
    MIN_FILL_PERCENTAGE = Decimal("80")  # 80% minimum fill

    async def _handle_partial_fill(
        self,
        order_id: str,
        symbol: str,
        expected_quantity: Decimal,
        filled_quantity: Decimal,
        avg_price: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Handle partial fill situation.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            expected_quantity: Expected order quantity
            filled_quantity: Actually filled quantity
            avg_price: Average fill price

        Returns:
            Dictionary with fill details and recommended action
        """
        fill_pct = (filled_quantity / expected_quantity * Decimal("100")) if expected_quantity > 0 else Decimal("0")
        unfilled_qty = expected_quantity - filled_quantity

        result = {
            "order_id": order_id,
            "expected_quantity": str(expected_quantity),
            "filled_quantity": str(filled_quantity),
            "unfilled_quantity": str(unfilled_qty),
            "fill_percentage": str(fill_pct),
            "avg_price": str(avg_price) if avg_price else None,
            "is_acceptable": fill_pct >= self.MIN_FILL_PERCENTAGE,
            "action": "none",
        }

        if filled_quantity == Decimal("0"):
            result["action"] = "retry_or_cancel"
            logger.warning(
                f"[{self._bot_id}] Order {order_id} not filled at all - "
                f"consider retry or cancel"
            )

        elif fill_pct < self.MIN_FILL_PERCENTAGE:
            result["action"] = "cancel_remaining"
            logger.warning(
                f"[{self._bot_id}] Order {order_id} partial fill below threshold: "
                f"{fill_pct:.1f}% < {self.MIN_FILL_PERCENTAGE}%"
            )

        else:
            result["action"] = "accept"
            if unfilled_qty > 0:
                logger.info(
                    f"[{self._bot_id}] Order {order_id} acceptable partial fill: "
                    f"{fill_pct:.1f}% ({filled_quantity}/{expected_quantity})"
                )

        return result

    async def _wait_for_fill_with_partial_handling(
        self,
        order_id: str,
        symbol: str,
        expected_quantity: Decimal,
        timeout_seconds: float = 30.0,
        poll_interval: float = 1.0,
        cancel_on_timeout: bool = True,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Wait for order fill with proper partial fill handling.

        Args:
            order_id: Order ID to monitor
            symbol: Trading symbol
            expected_quantity: Expected fill quantity
            timeout_seconds: Maximum wait time
            poll_interval: Polling interval
            cancel_on_timeout: Whether to cancel remaining on timeout

        Returns:
            Tuple of (is_successful, fill_details)
        """
        start_time = time.time()
        last_filled_qty = Decimal("0")

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                # Timeout - check final status
                try:
                    order = await self._exchange.futures.get_order(
                        symbol=symbol,
                        order_id=order_id,
                    )

                    if order:
                        filled_qty = getattr(order, "filled_qty", Decimal("0"))
                        avg_price = getattr(order, "avg_price", None)
                        status = getattr(order, "status", "").upper()

                        # Handle partial fill on timeout
                        if filled_qty > 0 and filled_qty < expected_quantity:
                            fill_result = await self._handle_partial_fill(
                                order_id, symbol, expected_quantity, filled_qty, avg_price
                            )

                            # Cancel remaining if requested
                            if cancel_on_timeout and status not in ["FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"]:
                                logger.info(
                                    f"[{self._bot_id}] Cancelling remaining order {order_id} "
                                    f"after timeout"
                                )
                                await self._cancel_order_with_timeout(order_id, symbol)

                            return fill_result["is_acceptable"], fill_result

                        elif filled_qty == expected_quantity:
                            return True, {
                                "order_id": order_id,
                                "filled_quantity": str(filled_qty),
                                "avg_price": str(avg_price) if avg_price else None,
                                "action": "filled",
                            }

                except Exception as e:
                    logger.error(f"[{self._bot_id}] Error getting final order status: {e}")

                return False, {
                    "order_id": order_id,
                    "action": "timeout",
                    "message": f"Order timed out after {timeout_seconds}s",
                }

            try:
                # Poll order status
                order = await self._exchange.futures.get_order(
                    symbol=symbol,
                    order_id=order_id,
                )

                if order:
                    status = getattr(order, "status", "").upper()
                    filled_qty = getattr(order, "filled_qty", Decimal("0"))
                    avg_price = getattr(order, "avg_price", None)

                    if status == "FILLED":
                        return True, {
                            "order_id": order_id,
                            "filled_quantity": str(filled_qty),
                            "avg_price": str(avg_price) if avg_price else None,
                            "action": "filled",
                        }

                    if status in ["CANCELED", "CANCELLED", "REJECTED", "EXPIRED"]:
                        # Order ended - handle any partial fill
                        if filled_qty > 0:
                            fill_result = await self._handle_partial_fill(
                                order_id, symbol, expected_quantity, filled_qty, avg_price
                            )
                            return fill_result["is_acceptable"], fill_result

                        return False, {
                            "order_id": order_id,
                            "status": status,
                            "action": "failed",
                        }

                    # Still active - log progress
                    if filled_qty > last_filled_qty:
                        logger.debug(
                            f"[{self._bot_id}] Order {order_id} partial fill progress: "
                            f"{filled_qty}/{expected_quantity}"
                        )
                        last_filled_qty = filled_qty

            except Exception as e:
                logger.warning(f"[{self._bot_id}] Error polling order status: {e}")

            await asyncio.sleep(poll_interval)

    async def _place_order_with_slippage_check(
        self,
        order_func: Callable,
        symbol: str,
        side: str,
        quantity: Decimal,
        expected_price: Decimal,
        max_slippage_pct: Optional[Decimal] = None,
    ) -> tuple[Optional[Any], Dict[str, Any]]:
        """
        Place order with pre and post slippage checks.

        Args:
            order_func: Async function that places the order
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            expected_price: Expected execution price
            max_slippage_pct: Maximum acceptable slippage

        Returns:
            Tuple of (order_result, slippage_info)
        """
        slippage_info = {
            "pre_check_passed": True,
            "estimated_slippage_pct": "0",
            "actual_slippage_pct": "0",
            "slippage_acceptable": True,
        }

        # Pre-order slippage estimate
        est_price, est_slippage = await self._estimate_slippage(symbol, side, quantity)
        slippage_info["estimated_slippage_pct"] = str(est_slippage)

        if max_slippage_pct is None:
            max_slippage_pct = self.DEFAULT_MAX_SLIPPAGE_PCT

        if est_slippage > max_slippage_pct:
            logger.warning(
                f"[{self._bot_id}] Pre-order slippage check failed: "
                f"estimated {est_slippage:.2f}% > max {max_slippage_pct}%"
            )
            slippage_info["pre_check_passed"] = False

            if self._notifier:
                await self._notifier.send_warning(
                    title="⚠️ High Slippage Warning",
                    message=(
                        f"Bot: {self._bot_id}\n"
                        f"Symbol: {symbol}\n"
                        f"Estimated slippage: {est_slippage:.2f}%\n"
                        f"Max allowed: {max_slippage_pct}%\n"
                        f"Order blocked"
                    ),
                )

            return None, slippage_info

        # Place order
        order = await order_func()

        if order:
            actual_price = getattr(order, "avg_price", None) or expected_price
            is_acceptable, actual_slippage = self._check_slippage_acceptable(
                expected_price, actual_price, side, max_slippage_pct
            )
            slippage_info["actual_slippage_pct"] = str(actual_slippage)
            slippage_info["slippage_acceptable"] = is_acceptable

            # Record slippage
            self._record_slippage(expected_price, actual_price, side, quantity)

            if not is_acceptable:
                logger.warning(
                    f"[{self._bot_id}] Post-order slippage exceeded: "
                    f"{actual_slippage:.2f}% > {max_slippage_pct}%"
                )

        return order, slippage_info

    # =========================================================================
    # Pre-Order Validation (Combined Checks)
    # =========================================================================

    async def _validate_before_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
    ) -> tuple[bool, str]:
        """
        Comprehensive pre-order validation.

        Checks:
        1. Position reconciliation
        2. Balance sufficiency
        3. Time synchronization
        4. Data health

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            leverage: Position leverage

        Returns:
            Tuple of (is_valid, message)
        """
        # 1. Reconcile position first
        position_ok = await self._reconcile_position()
        if not position_ok:
            return False, "Position mismatch detected - cannot place order"

        # 2. Check balance
        balance_ok, balance_msg = await self._check_balance_for_order(
            symbol, quantity, price, leverage
        )
        if not balance_ok:
            return False, balance_msg

        # 3. Check time sync
        time_ok = await self._check_time_sync_health()
        if not time_ok:
            return False, "Time sync critical - please wait for resync"

        # 4. Check data connection
        if not self._is_data_connection_healthy():
            return False, "Data connection unhealthy - may have stale data"

        return True, "All pre-order checks passed"

    # =========================================================================
    # Fill Confirmation with Polling Fallback
    # =========================================================================

    # Constants for fill confirmation
    FILL_CONFIRMATION_WAIT_SECONDS = 5  # Wait for WebSocket notification
    FILL_POLLING_INTERVAL_SECONDS = 2  # Poll interval if no WebSocket
    FILL_CONFIRMATION_TIMEOUT_SECONDS = 60  # Total timeout

    def _init_fill_tracking(self) -> None:
        """Initialize fill tracking for WebSocket + polling confirmation."""
        if not hasattr(self, "_pending_fill_callbacks"):
            self._pending_fill_callbacks: Dict[str, Callable] = {}
        if not hasattr(self, "_confirmed_fills"):
            self._confirmed_fills: Dict[str, Dict] = {}
        if not hasattr(self, "_fill_confirmation_events"):
            self._fill_confirmation_events: Dict[str, asyncio.Event] = {}
        if not hasattr(self, "_pending_fills"):
            # Early-arrival buffer: stores WS fill data for orders whose
            # REST response hasn't returned yet (race condition mitigation)
            self._pending_fills: Dict[str, Dict] = {}

    def _register_fill_callback(self, order_id: str, callback: Callable) -> None:
        """Register a callback to be triggered when fill is confirmed."""
        self._init_fill_tracking()
        self._pending_fill_callbacks[order_id] = callback
        self._fill_confirmation_events[order_id] = asyncio.Event()

    def _unregister_fill_callback(self, order_id: str) -> None:
        """Unregister fill callback."""
        self._init_fill_tracking()
        self._pending_fill_callbacks.pop(order_id, None)
        self._fill_confirmation_events.pop(order_id, None)

    def _drain_pending_fill(self, order_id: str) -> Optional[Dict]:
        """
        Check and consume any early-arrival fill for this order.

        Call this after REST order placement returns to handle the race
        where the WS fill arrived before the REST response.

        Args:
            order_id: The order ID to check

        Returns:
            Fill data if an early-arrival was buffered, None otherwise
        """
        self._init_fill_tracking()
        fill_data = self._pending_fills.pop(order_id, None)
        if fill_data:
            logger.info(
                f"[{self._bot_id}] Drained early-arrival fill for order {order_id}"
            )
        return fill_data

    def _on_fill_notification(self, order_id: str, fill_data: Dict) -> None:
        """
        Called when WebSocket fill notification is received.

        This should be called from the user data stream handler when
        an executionReport event is received with status=FILLED.

        Race condition handling: If the WS fill arrives before the REST
        order placement returns (i.e., no callback registered yet), the
        fill is buffered in _pending_fills for the caller to consume
        after the REST response.
        """
        self._init_fill_tracking()

        # Record the fill (with cleanup to prevent memory leak)
        self._confirmed_fills[order_id] = {
            "timestamp": time.time(),
            "data": fill_data,
            "source": "websocket",
        }
        # Keep only recent 1000 entries
        if len(self._confirmed_fills) > 1000:
            oldest_keys = sorted(
                self._confirmed_fills.keys(),
                key=lambda k: self._confirmed_fills[k]["timestamp"],
            )[:len(self._confirmed_fills) - 1000]
            for k in oldest_keys:
                del self._confirmed_fills[k]

        # Clean up stale pending fills (> 60 seconds old)
        stale_keys = [
            k for k, v in self._pending_fills.items()
            if time.time() - v.get("buffered_at", 0) > 60
        ]
        for k in stale_keys:
            del self._pending_fills[k]

        # Trigger callback if registered
        if order_id in self._pending_fill_callbacks:
            try:
                self._pending_fill_callbacks[order_id](fill_data)
            except Exception as e:
                logger.error(f"[{self._bot_id}] Error in fill callback: {e}")
        else:
            # No callback registered yet — buffer as early arrival
            self._pending_fills[order_id] = {**fill_data, "buffered_at": time.time()}
            logger.info(
                f"[{self._bot_id}] Early fill arrival buffered for order {order_id}"
            )

        # Signal event for anyone waiting
        if order_id in self._fill_confirmation_events:
            self._fill_confirmation_events[order_id].set()

        logger.debug(f"[{self._bot_id}] Fill notification received for {order_id}")

    async def _confirm_fill_with_polling(
        self,
        order_id: str,
        symbol: str,
        expected_quantity: Optional[Decimal] = None,
        initial_wait_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
        max_wait_seconds: Optional[float] = None,
    ) -> tuple[bool, Optional[Dict]]:
        """
        Confirm order fill using WebSocket + polling fallback.

        Strategy:
        1. Wait for WebSocket notification (5s default)
        2. If no notification, poll every 2s
        3. Timeout after 60s total

        Args:
            order_id: Order ID to confirm
            symbol: Trading symbol
            expected_quantity: Expected fill quantity (for partial fill detection)
            initial_wait_seconds: Time to wait for WebSocket notification
            poll_interval_seconds: Polling interval
            max_wait_seconds: Total timeout

        Returns:
            Tuple of (is_confirmed, order_data)
        """
        self._init_fill_tracking()

        if initial_wait_seconds is None:
            initial_wait_seconds = self.FILL_CONFIRMATION_WAIT_SECONDS
        if poll_interval_seconds is None:
            poll_interval_seconds = self.FILL_POLLING_INTERVAL_SECONDS
        if max_wait_seconds is None:
            max_wait_seconds = self.FILL_CONFIRMATION_TIMEOUT_SECONDS

        start_time = time.time()

        # Create event for this order
        self._fill_confirmation_events[order_id] = asyncio.Event()

        try:
            # Step 1: Wait for WebSocket notification
            try:
                await asyncio.wait_for(
                    self._fill_confirmation_events[order_id].wait(),
                    timeout=initial_wait_seconds,
                )

                # WebSocket notification received!
                if order_id in self._confirmed_fills:
                    fill_data = self._confirmed_fills[order_id]["data"]
                    logger.info(
                        f"[{self._bot_id}] Order {order_id} fill confirmed via WebSocket"
                    )
                    return True, fill_data

            except asyncio.TimeoutError:
                # No WebSocket notification - fall back to polling
                logger.warning(
                    f"[{self._bot_id}] No WebSocket fill notification for {order_id} "
                    f"after {initial_wait_seconds}s - starting polling"
                )

            # Step 2: Poll order status
            while True:
                elapsed = time.time() - start_time

                if elapsed > max_wait_seconds:
                    logger.error(
                        f"[{self._bot_id}] Order {order_id} fill confirmation timeout "
                        f"after {max_wait_seconds}s"
                    )
                    return False, None

                try:
                    order = await self._exchange.futures.get_order(
                        symbol=symbol,
                        order_id=order_id,
                    )

                    if order:
                        status = getattr(order, "status", "").upper()
                        filled_qty = getattr(order, "filled_qty", Decimal("0"))

                        if status == "FILLED":
                            order_data = {
                                "order_id": order_id,
                                "status": status,
                                "filled_qty": str(filled_qty),
                                "avg_price": str(getattr(order, "avg_price", None)),
                                "source": "polling",
                                "elapsed_seconds": elapsed,
                            }

                            # Record for future reference
                            self._confirmed_fills[order_id] = {
                                "timestamp": time.time(),
                                "data": order_data,
                                "source": "polling",
                            }

                            logger.info(
                                f"[{self._bot_id}] Order {order_id} fill confirmed via polling "
                                f"(elapsed: {elapsed:.1f}s)"
                            )
                            return True, order_data

                        elif status in ["CANCELED", "CANCELLED", "REJECTED", "EXPIRED"]:
                            logger.warning(
                                f"[{self._bot_id}] Order {order_id} ended with status: {status}"
                            )

                            # Check for partial fill
                            if filled_qty > 0:
                                order_data = {
                                    "order_id": order_id,
                                    "status": status,
                                    "filled_qty": str(filled_qty),
                                    "avg_price": str(getattr(order, "avg_price", None)),
                                    "partial_fill": True,
                                    "source": "polling",
                                }
                                return True, order_data

                            return False, {
                                "order_id": order_id,
                                "status": status,
                                "message": f"Order {status}",
                            }

                        # Still pending - continue polling
                        logger.debug(
                            f"[{self._bot_id}] Order {order_id} still pending "
                            f"(status={status}, filled={filled_qty})"
                        )

                except Exception as e:
                    logger.warning(f"[{self._bot_id}] Error polling order {order_id}: {e}")

                await asyncio.sleep(poll_interval_seconds)

        finally:
            # Cleanup
            self._unregister_fill_callback(order_id)

    # =========================================================================
    # Order Expiration Monitoring
    # =========================================================================

    # Constants for order expiration
    DEFAULT_ORDER_MAX_AGE_SECONDS = 300  # 5 minutes for limit orders
    ORDER_EXPIRATION_CHECK_INTERVAL = 10  # Check every 10 seconds

    def _init_order_monitoring(self) -> None:
        """Initialize order monitoring structures."""
        if not hasattr(self, "_monitored_orders"):
            self._monitored_orders: Dict[str, Dict] = {}
        if not hasattr(self, "_order_monitor_task"):
            self._order_monitor_task: Optional[asyncio.Task] = None

    async def _monitor_order_expiration(
        self,
        order_id: str,
        symbol: str,
        max_age_seconds: Optional[int] = None,
        cancel_if_expired: bool = True,
        on_expired_callback: Optional[Callable] = None,
    ) -> None:
        """
        Monitor an order for expiration and auto-cancel if stuck.

        Args:
            order_id: Order ID to monitor
            symbol: Trading symbol
            max_age_seconds: Maximum order age before considering it expired
            cancel_if_expired: Whether to cancel expired orders
            on_expired_callback: Callback when order expires
        """
        self._init_order_monitoring()

        if max_age_seconds is None:
            max_age_seconds = self.DEFAULT_ORDER_MAX_AGE_SECONDS

        created_at = time.time()

        self._monitored_orders[order_id] = {
            "symbol": symbol,
            "created_at": created_at,
            "max_age_seconds": max_age_seconds,
            "cancel_if_expired": cancel_if_expired,
            "on_expired_callback": on_expired_callback,
        }

        logger.debug(
            f"[{self._bot_id}] Started monitoring order {order_id} "
            f"(max age: {max_age_seconds}s)"
        )

        while True:
            try:
                # Check if still monitoring this order
                if order_id not in self._monitored_orders:
                    break

                order = await self._exchange.futures.get_order(
                    symbol=symbol,
                    order_id=order_id,
                )

                if order:
                    status = getattr(order, "status", "").upper()

                    # Order reached terminal state
                    if status in ["FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"]:
                        logger.info(
                            f"[{self._bot_id}] Order {order_id} reached terminal state: {status}"
                        )
                        self._monitored_orders.pop(order_id, None)
                        break

                    # Check age
                    age = time.time() - created_at

                    if age > max_age_seconds:
                        filled_qty = getattr(order, "filled_qty", Decimal("0"))

                        logger.warning(
                            f"[{self._bot_id}] Order {order_id} exceeded max age "
                            f"({max_age_seconds}s), filled={filled_qty}"
                        )

                        if cancel_if_expired:
                            try:
                                await self._cancel_order_with_timeout(order_id, symbol)
                                logger.info(
                                    f"[{self._bot_id}] Expired order {order_id} cancelled"
                                )
                            except Exception as e:
                                logger.error(
                                    f"[{self._bot_id}] Failed to cancel expired order "
                                    f"{order_id}: {e}"
                                )

                        if on_expired_callback:
                            try:
                                on_expired_callback(order_id, filled_qty)
                            except Exception as e:
                                logger.error(
                                    f"[{self._bot_id}] Error in expiration callback: {e}"
                                )

                        self._monitored_orders.pop(order_id, None)
                        break

            except Exception as e:
                logger.warning(
                    f"[{self._bot_id}] Error monitoring order {order_id}: {e}"
                )

            await asyncio.sleep(self.ORDER_EXPIRATION_CHECK_INTERVAL)

    def _stop_monitoring_order(self, order_id: str) -> None:
        """Stop monitoring an order."""
        self._init_order_monitoring()
        self._monitored_orders.pop(order_id, None)

    async def _get_stale_orders(
        self,
        symbol: str,
        max_age_seconds: Optional[int] = None,
    ) -> list[Dict]:
        """
        Get list of orders that have been open too long.

        Args:
            symbol: Trading symbol
            max_age_seconds: Maximum acceptable order age

        Returns:
            List of stale order details
        """
        if max_age_seconds is None:
            max_age_seconds = self.DEFAULT_ORDER_MAX_AGE_SECONDS

        stale_orders = []
        current_time = time.time()

        try:
            open_orders = await self._exchange.futures.get_open_orders(symbol=symbol)

            for order in open_orders:
                # Get order timestamp
                order_time = getattr(order, "timestamp", None)
                if order_time:
                    # Convert to seconds if milliseconds
                    if order_time > 1e12:
                        order_time = order_time / 1000

                    age = current_time - order_time

                    if age > max_age_seconds:
                        stale_orders.append({
                            "order_id": getattr(order, "order_id", ""),
                            "symbol": symbol,
                            "side": getattr(order, "side", ""),
                            "price": str(getattr(order, "price", "")),
                            "quantity": str(getattr(order, "quantity", "")),
                            "filled_qty": str(getattr(order, "filled_qty", "0")),
                            "age_seconds": age,
                            "status": getattr(order, "status", ""),
                        })

        except Exception as e:
            logger.error(f"[{self._bot_id}] Error getting stale orders: {e}")

        return stale_orders

    # =========================================================================
    # Order Execution with Retry and Fallback
    # =========================================================================

    # Constants for order retry
    MAX_ORDER_RETRIES = 3
    ORDER_RETRY_DELAY_SECONDS = 1
    LIMIT_ORDER_TIMEOUT_SECONDS = 10  # Wait for limit order fill

    async def _execute_order_with_retry(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        limit_price: Optional[Decimal] = None,
        order_type: str = "LIMIT",
        max_retries: Optional[int] = None,
        fallback_to_market: bool = True,
        reduce_only: bool = False,
        position_side: str = "BOTH",
    ) -> tuple[bool, Optional[Any], Dict]:
        """
        Execute order with retry and fallback strategies.

        Strategy:
        1. Try preferred order type (LIMIT or MARKET)
        2. If LIMIT fails/times out, try more aggressive price
        3. If still fails, fall back to MARKET order

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            limit_price: Limit price (required for LIMIT orders)
            order_type: Order type (LIMIT/MARKET)
            max_retries: Maximum retry attempts
            fallback_to_market: Fall back to MARKET if LIMIT fails
            reduce_only: Whether this is a reduce-only order
            position_side: Position side (BOTH for one-way, LONG/SHORT for hedge mode)

        Returns:
            Tuple of (success, order, execution_details)
        """
        if max_retries is None:
            max_retries = self.MAX_ORDER_RETRIES

        execution_details = {
            "attempts": [],
            "final_order_type": order_type,
            "total_attempts": 0,
            "fell_back_to_market": False,
        }

        # Try LIMIT order with progressive price adjustments
        if order_type.upper() == "LIMIT" and limit_price:
            for attempt in range(max_retries):
                try:
                    # Adjust price for each attempt (more aggressive)
                    adjustment_pct = Decimal("0.1") * Decimal(str(attempt + 1)) / Decimal("100")

                    if side.upper() == "BUY":
                        # For BUY: increase price (more aggressive)
                        adjusted_price = limit_price * (Decimal("1") + adjustment_pct)
                    else:
                        # For SELL: decrease price (more aggressive)
                        adjusted_price = limit_price * (Decimal("1") - adjustment_pct)

                    # Normalize price
                    symbol_info = await self._get_symbol_info(symbol)
                    adjusted_price = self._normalize_price(adjusted_price, symbol_info)

                    logger.info(
                        f"[{self._bot_id}] LIMIT order attempt {attempt + 1}: "
                        f"{side} {quantity} @ {adjusted_price}"
                    )

                    order = await self._exchange.futures_create_order(
                        symbol=symbol,
                        side=side,
                        order_type="LIMIT",
                        quantity=quantity,
                        price=adjusted_price,
                        time_in_force="GTC",
                        reduce_only=reduce_only,
                        position_side=position_side,
                        bot_id=self._bot_id,
                    )

                    if order:
                        order_id = getattr(order, "order_id", "")

                        # Wait for fill with timeout
                        is_filled, fill_data = await self._confirm_fill_with_polling(
                            order_id=str(order_id),
                            symbol=symbol,
                            expected_quantity=quantity,
                            max_wait_seconds=self.LIMIT_ORDER_TIMEOUT_SECONDS,
                        )

                        execution_details["attempts"].append({
                            "attempt": attempt + 1,
                            "order_type": "LIMIT",
                            "price": str(adjusted_price),
                            "success": is_filled,
                            "order_id": str(order_id),
                        })
                        execution_details["total_attempts"] = attempt + 1

                        if is_filled:
                            logger.info(
                                f"[{self._bot_id}] LIMIT order filled on attempt {attempt + 1}"
                            )
                            return True, order, execution_details
                        else:
                            # Cancel unfilled order
                            logger.warning(
                                f"[{self._bot_id}] LIMIT order {order_id} not filled - cancelling"
                            )
                            await self._cancel_order_with_timeout(str(order_id), symbol)

                except Exception as e:
                    error_code, error_cat = self._classify_order_error(e)
                    logger.warning(
                        f"[{self._bot_id}] LIMIT order attempt {attempt + 1} failed: "
                        f"{error_code} ({error_cat})"
                    )

                    execution_details["attempts"].append({
                        "attempt": attempt + 1,
                        "order_type": "LIMIT",
                        "price": str(limit_price),
                        "success": False,
                        "error": str(error_code),
                    })

                    # Some errors shouldn't be retried
                    if error_code in ["INSUFFICIENT_BALANCE", "POSITION_LIMIT", "SYMBOL_NOT_FOUND"]:
                        logger.error(
                            f"[{self._bot_id}] Non-retryable error: {error_cat}"
                        )
                        return False, None, execution_details

                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.ORDER_RETRY_DELAY_SECONDS)

        # Fall back to MARKET order
        if fallback_to_market or order_type.upper() == "MARKET":
            logger.warning(
                f"[{self._bot_id}] Falling back to MARKET order"
            )
            execution_details["fell_back_to_market"] = True

            try:
                order = await self._exchange.futures_create_order(
                    symbol=symbol,
                    side=side,
                    order_type="MARKET",
                    quantity=quantity,
                    reduce_only=reduce_only,
                    position_side=position_side,
                    bot_id=self._bot_id,
                )

                if order:
                    execution_details["attempts"].append({
                        "attempt": execution_details["total_attempts"] + 1,
                        "order_type": "MARKET",
                        "success": True,
                        "order_id": str(getattr(order, "order_id", "")),
                    })
                    execution_details["total_attempts"] += 1
                    execution_details["final_order_type"] = "MARKET"

                    logger.info(
                        f"[{self._bot_id}] MARKET order executed successfully"
                    )
                    return True, order, execution_details

            except Exception as e:
                error_code, error_cat = self._classify_order_error(e)
                logger.error(
                    f"[{self._bot_id}] MARKET order also failed: {error_code}"
                )

                execution_details["attempts"].append({
                    "attempt": execution_details["total_attempts"] + 1,
                    "order_type": "MARKET",
                    "success": False,
                    "error": str(error_code),
                })

        return False, None, execution_details

    # =========================================================================
    # Liquidity Monitoring
    # =========================================================================

    async def _check_orderbook_liquidity(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        depth_levels: int = 5,
        min_liquidity_ratio: Decimal = Decimal("2.0"),
    ) -> tuple[bool, Dict]:
        """
        Check if orderbook has sufficient liquidity for an order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            depth_levels: Number of orderbook levels to check
            min_liquidity_ratio: Minimum ratio of available liquidity to order size

        Returns:
            Tuple of (has_liquidity, liquidity_details)
        """
        try:
            orderbook = await self._exchange.futures.get_orderbook(symbol, limit=20)

            if not orderbook:
                return False, {"error": "Failed to get orderbook"}

            # Get relevant side
            levels = orderbook.asks if side.upper() == "BUY" else orderbook.bids

            if not levels or len(levels) < depth_levels:
                return False, {
                    "error": "Insufficient orderbook depth",
                    "available_levels": len(levels) if levels else 0,
                }

            # Calculate available liquidity in top N levels
            available_qty = Decimal("0")
            total_value = Decimal("0")

            for i, (price, qty) in enumerate(levels[:depth_levels]):
                available_qty += qty
                total_value += price * qty

            liquidity_ratio = available_qty / quantity if quantity > 0 else Decimal("0")
            avg_price = total_value / available_qty if available_qty > 0 else Decimal("0")

            has_liquidity = liquidity_ratio >= min_liquidity_ratio

            details = {
                "available_qty": str(available_qty),
                "required_qty": str(quantity),
                "liquidity_ratio": str(liquidity_ratio),
                "avg_price_in_depth": str(avg_price),
                "best_price": str(levels[0][0]) if levels else "0",
                "depth_levels_checked": depth_levels,
                "has_sufficient_liquidity": has_liquidity,
            }

            if not has_liquidity:
                logger.warning(
                    f"[{self._bot_id}] Insufficient liquidity for {side} {quantity} {symbol}: "
                    f"ratio={liquidity_ratio:.2f} < {min_liquidity_ratio}"
                )

            return has_liquidity, details

        except Exception as e:
            logger.error(f"[{self._bot_id}] Error checking liquidity: {e}")
            return False, {"error": str(e)}

    async def _wait_for_liquidity(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        timeout_seconds: float = 30.0,
        check_interval: float = 2.0,
    ) -> tuple[bool, Dict]:
        """
        Wait for sufficient liquidity in orderbook.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            timeout_seconds: Maximum wait time
            check_interval: Check interval

        Returns:
            Tuple of (has_liquidity, final_liquidity_details)
        """
        start_time = time.time()
        last_details = {}

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                logger.warning(
                    f"[{self._bot_id}] Liquidity wait timeout after {timeout_seconds}s"
                )
                return False, {
                    "timeout": True,
                    "elapsed_seconds": elapsed,
                    **last_details,
                }

            has_liquidity, details = await self._check_orderbook_liquidity(
                symbol, side, quantity
            )
            last_details = details

            if has_liquidity:
                logger.info(
                    f"[{self._bot_id}] Sufficient liquidity found after {elapsed:.1f}s"
                )
                return True, {
                    "elapsed_seconds": elapsed,
                    **details,
                }

            await asyncio.sleep(check_interval)

    # =========================================================================
    # Abstract Lifecycle Methods (Subclass Must Implement)
    # =========================================================================

    @abstractmethod
    async def _do_start(self) -> None:
        """
        Actual start logic. Subclass implements.

        Called after state transition to INITIALIZING.
        Should initialize resources, place initial orders, etc.

        Raises:
            Exception: If start fails
        """
        pass

    @abstractmethod
    async def _do_stop(self, clear_position: bool = False) -> None:
        """
        Actual stop logic. Subclass implements.

        Called after state transition to STOPPING.
        Should cancel orders, save state, optionally clear positions.

        Args:
            clear_position: If True, close all positions before stopping
        """
        pass

    @abstractmethod
    async def _do_pause(self) -> None:
        """
        Actual pause logic. Subclass implements.

        Called when transitioning to PAUSED state.
        Typically cancels pending orders but keeps positions.
        """
        pass

    @abstractmethod
    async def _do_resume(self) -> None:
        """
        Actual resume logic. Subclass implements.

        Called when transitioning from PAUSED to RUNNING.
        Typically re-places orders based on current state.
        """
        pass

    @abstractmethod
    def _get_extra_status(self) -> Dict[str, Any]:
        """
        Return extra status fields specific to this bot type.

        Subclass should return bot-specific metrics like:
        - Grid bot: upper_price, lower_price, grid_count, pending_orders
        - DCA bot: next_buy_time, average_cost, total_bought

        Returns:
            Dictionary with extra status fields
        """
        pass

    @abstractmethod
    async def _extra_health_checks(self) -> Dict[str, bool]:
        """
        Perform extra health checks specific to this bot type.

        Subclass should check bot-specific health indicators like:
        - Grid bot: orders_synced, within_price_range
        - DCA bot: schedule_active, funds_available

        Returns:
            Dictionary mapping check name to pass/fail boolean
        """
        pass

    # =========================================================================
    # Public Lifecycle Methods (Template Method Pattern)
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the bot.

        State Transition: REGISTERED/STOPPED -> INITIALIZING -> RUNNING

        Returns:
            True if started successfully

        Raises:
            InvalidStateError: If current state doesn't allow starting
        """
        if self._state not in (BotState.REGISTERED, BotState.STOPPED):
            raise InvalidStateError(f"Cannot start from state {self._state.value}")

        try:
            logger.info(f"Starting bot: {self._bot_id}")
            self._state = BotState.INITIALIZING
            self._stats.start_time = datetime.now(timezone.utc)
            self._error_message = None

            # Force time synchronization before starting
            await self._ensure_time_sync()

            # Validate configuration before starting
            await self._validate_config_on_start()

            # Verify exchange leverage matches config
            await self._verify_leverage_on_start()

            # FIX: Cancel orphan orders from previous session on startup
            await self._cleanup_orphan_orders_on_start()

            # Call subclass implementation
            await self._do_start()

            # Update state - set _running first to avoid race condition
            # where heartbeat checks see RUNNING state but _running is False
            self._running = True
            self._state = BotState.RUNNING

            # Start heartbeat
            self._start_heartbeat()

            # Start position reconciliation
            self._start_position_reconciliation()

            logger.info(f"Bot {self._bot_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start bot {self._bot_id}: {e}", exc_info=True)
            # Cleanup any resources that may have been started
            await self._stop_heartbeat()
            self._stop_position_reconciliation()
            self._running = False
            self._state = BotState.ERROR
            self._error_message = str(e)
            return False

    async def stop(self, reason: str = "Manual stop", clear_position: bool = False) -> bool:
        """
        Stop the bot.

        State Transition: RUNNING/PAUSED/ERROR -> STOPPING -> STOPPED

        Args:
            reason: Stop reason for logging
            clear_position: If True, close all positions

        Returns:
            True if stopped successfully

        Raises:
            InvalidStateError: If current state doesn't allow stopping
        """
        if self._state not in (BotState.RUNNING, BotState.PAUSED, BotState.ERROR):
            raise InvalidStateError(f"Cannot stop from state {self._state.value}")

        try:
            logger.info(f"Stopping bot {self._bot_id}: {reason}")
            # Set state to STOPPING first, but keep _running = True
            # to allow ongoing operations to complete gracefully
            self._state = BotState.STOPPING

            # Stop heartbeat first to prevent stale heartbeats
            await self._stop_heartbeat()

            # Stop position reconciliation
            self._stop_position_reconciliation()

            # Cleanup any pending notification tasks
            await self._cleanup_notification_tasks()

            # Call subclass implementation with timeout to prevent hang
            try:
                async with asyncio.timeout(30):
                    await self._do_stop(clear_position)
            except asyncio.TimeoutError:
                logger.error(
                    f"Bot {self._bot_id} stop timed out after 30s, forcing shutdown"
                )

            # Only set _running = False after cleanup is complete
            # This ensures order manager and other components can finish their work
            self._running = False

            # Update state to STOPPED
            self._state = BotState.STOPPED

            logger.info(f"Bot {self._bot_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping bot {self._bot_id}: {e}")
            self._running = False
            self._state = BotState.ERROR
            self._error_message = str(e)
            return False

    async def pause(self, reason: str = "Manual pause") -> bool:
        """
        Pause the bot.

        State Transition: RUNNING -> PAUSED

        Args:
            reason: Pause reason for logging

        Returns:
            True if paused successfully

        Raises:
            InvalidStateError: If current state doesn't allow pausing
        """
        if self._state != BotState.RUNNING:
            raise InvalidStateError(f"Cannot pause from state {self._state.value}")

        try:
            logger.info(f"Pausing bot {self._bot_id}: {reason}")

            # Call subclass implementation
            await self._do_pause()

            # Stop heartbeat before updating state
            await self._stop_heartbeat()

            # Update state
            self._running = False
            self._state = BotState.PAUSED

            logger.info(f"Bot {self._bot_id} paused")
            return True

        except Exception as e:
            logger.error(f"Error pausing bot {self._bot_id}: {e}")
            self._running = False
            self._state = BotState.ERROR
            self._error_message = str(e)
            return False

    async def resume(self) -> bool:
        """
        Resume the bot from paused state.

        State Transition: PAUSED -> RUNNING

        Returns:
            True if resumed successfully

        Raises:
            InvalidStateError: If current state doesn't allow resuming
        """
        if self._state != BotState.PAUSED:
            raise InvalidStateError(f"Cannot resume from state {self._state.value}")

        try:
            logger.info(f"Resuming bot {self._bot_id}")

            # Call subclass implementation
            await self._do_resume()

            # Update state
            self._running = True
            self._state = BotState.RUNNING

            # Restart heartbeat
            self._start_heartbeat()

            logger.info(f"Bot {self._bot_id} resumed")
            return True

        except Exception as e:
            logger.error(f"Error resuming bot {self._bot_id}: {e}")
            self._state = BotState.ERROR
            self._error_message = str(e)
            self._running = False
            return False

    # =========================================================================
    # Status Query
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive bot status.

        Returns:
            Dictionary with bot status including:
            - Basic info: bot_id, bot_type, symbol, state
            - Statistics: total_trades, total_profit, today_profit
            - Timing: uptime, error_message
            - Extra fields from subclass
        """
        status = {
            "bot_id": self._bot_id,
            "bot_type": self.bot_type,
            "symbol": self.symbol,
            "state": self._state.value,
            "total_trades": self._stats.total_trades,
            "total_profit": str(self._stats.total_profit),
            "total_fees": str(self._stats.total_fees),
            "today_trades": self._stats.today_trades,
            "today_profit": str(self._stats.today_profit),
            "uptime": self._get_uptime(),
            "error_message": self._error_message,
        }

        # Add subclass-specific fields
        status.update(self._get_extra_status())

        return status

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def _start_heartbeat(self) -> None:
        """Start heartbeat task."""
        if self._heartbeat_task is not None or self._heartbeat_callback is None:
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.debug(f"Heartbeat started for {self._bot_id}")

    async def _stop_heartbeat(self) -> None:
        """Stop heartbeat task and await cancellation."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.debug(f"Heartbeat stopped for {self._bot_id}")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop - sends heartbeat every 10 seconds."""
        while self._running:
            try:
                await asyncio.sleep(10)
                # Check all state conditions to avoid sending stale heartbeat
                # during state transitions (e.g., when bot is stopping)
                if self._running and self._state == BotState.RUNNING and self._heartbeat_callback:
                    await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to Master via callback."""
        if not self._heartbeat_callback:
            return

        try:
            # Import here to avoid circular imports
            from src.master.heartbeat import HeartbeatData

            heartbeat = HeartbeatData(
                bot_id=self._bot_id,
                state=self._state,
                metrics=self._get_heartbeat_metrics(),
            )
            # Callback may be async (HeartbeatMonitor.receive)
            result = self._heartbeat_callback(heartbeat)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")

    def _get_heartbeat_metrics(self) -> Dict[str, Any]:
        """Get metrics to include in heartbeat."""
        return {
            "uptime_seconds": self._get_uptime_seconds(),
            "total_trades": self._stats.total_trades,
            "total_profit": float(self._stats.total_profit),
        }

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dictionary with:
            - healthy: True if all checks pass
            - checks: Dictionary of individual check results
        """
        checks = {
            "state": self._state == BotState.RUNNING,
            "exchange": await self._check_exchange(),
        }

        # Add subclass-specific checks
        extra_checks = await self._extra_health_checks()
        checks.update(extra_checks)

        all_healthy = all(checks.values())

        return {
            "healthy": all_healthy,
            "checks": checks,
        }

    async def _check_exchange(self) -> bool:
        """Check exchange connection."""
        try:
            await self._exchange.get_account()
            return True
        except Exception as e:
            logger.warning("Exchange health check failed: %s", e)
            return False

    # =========================================================================
    # Capital Management
    # =========================================================================

    async def update_capital(self, new_allocated_capital: Decimal) -> bool:
        """
        Update capital allocation for this bot from Fund Manager.

        Called by FundManager when capital allocation changes.
        Updates allocated_capital (preferred) or falls back to max_capital.
        Subclasses can override to implement specific behavior.

        Args:
            new_allocated_capital: New capital amount from Fund Manager

        Returns:
            True if update was successful
        """
        try:
            # Ensure lock is initialized
            if not hasattr(self, "_capital_lock"):
                self._capital_lock = asyncio.Lock()

            # Update config under lock to prevent race conditions
            async with self._capital_lock:
                # Prefer allocated_capital (Fund Manager allocation)
                if hasattr(self._config, "allocated_capital"):
                    previous = getattr(self._config, "allocated_capital", None)
                    self._config.allocated_capital = new_allocated_capital
                    logger.info(
                        f"Bot {self._bot_id} allocated_capital updated: "
                        f"{previous} -> {new_allocated_capital}"
                    )
                elif hasattr(self._config, "max_capital"):
                    # Fallback for configs without allocated_capital
                    previous = getattr(self._config, "max_capital", None)
                    self._config.max_capital = new_allocated_capital
                    logger.info(
                        f"Bot {self._bot_id} max_capital updated (fallback): "
                        f"{previous} -> {new_allocated_capital}"
                    )

            # Notify subclass of capital change (they can override _on_capital_updated)
            await self._on_capital_updated(new_allocated_capital)

            return True

        except Exception as e:
            logger.error(f"Failed to update capital for {self._bot_id}: {e}")
            return False

    async def _on_capital_updated(self, new_max_capital: Decimal) -> None:
        """
        Hook for subclasses to handle capital updates.

        Override this method to implement custom behavior when capital changes.
        Default implementation does nothing.

        Args:
            new_max_capital: New maximum capital amount
        """
        pass

    def get_max_capital(self) -> Optional[Decimal]:
        """
        Get current maximum capital allocation.

        Returns:
            Maximum capital or None if not configured
        """
        if hasattr(self._config, "max_capital"):
            return self._config.max_capital
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_uptime(self) -> str:
        """Get formatted uptime string."""
        if not self._stats.start_time:
            return "0s"

        delta = datetime.now(timezone.utc) - self._stats.start_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _get_uptime_seconds(self) -> int:
        """Get uptime in seconds."""
        if not self._stats.start_time:
            return 0
        return int((datetime.now(timezone.utc) - self._stats.start_time).total_seconds())

    # =========================================================================
    # MFE/MAE Calculation Helper
    # =========================================================================

    @staticmethod
    def calculate_mfe_mae(
        side: str,
        entry_price: Decimal,
        highest_price: Decimal,
        lowest_price: Decimal,
        leverage: int = 1,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate Max Favorable Excursion and Max Adverse Excursion.

        Args:
            side: "LONG" or "SHORT"
            entry_price: Position entry price
            highest_price: Highest price during position
            lowest_price: Lowest price during position
            leverage: Position leverage

        Returns:
            (mfe_pct, mae_pct) as percentages
        """
        if entry_price <= 0 or highest_price <= 0 or lowest_price <= 0:
            return Decimal("0"), Decimal("0")

        lev = Decimal(leverage)
        if side.upper() == "LONG":
            mfe = (highest_price - entry_price) / entry_price * lev * Decimal("100")
            mae = (entry_price - lowest_price) / entry_price * lev * Decimal("100")
        else:  # SHORT
            mfe = (entry_price - lowest_price) / entry_price * lev * Decimal("100")
            mae = (highest_price - entry_price) / entry_price * lev * Decimal("100")

        return max(mfe, Decimal("0")), max(mae, Decimal("0"))

    # =========================================================================
    # Per-Strategy Risk Isolation (風控相互影響)
    # Prevents one strategy's losses from affecting healthy strategies
    # =========================================================================

    # Strategy-level risk thresholds
    STRATEGY_LOSS_WARNING_PCT = Decimal("0.03")  # 3% strategy loss warning
    STRATEGY_LOSS_PAUSE_PCT = Decimal("0.05")    # 5% strategy loss -> pause self
    STRATEGY_LOSS_STOP_PCT = Decimal("0.10")     # 10% strategy loss -> stop self
    STRATEGY_DRAWDOWN_WARNING_PCT = Decimal("0.05")  # 5% drawdown warning
    STRATEGY_DRAWDOWN_PAUSE_PCT = Decimal("0.08")    # 8% drawdown -> pause
    STRATEGY_CONSECUTIVE_LOSS_WARNING = 3  # 3 consecutive losses
    STRATEGY_CONSECUTIVE_LOSS_PAUSE = 5    # 5 consecutive losses -> pause
    STRATEGY_MAX_DAILY_TRADES = 100        # Max trades per bot per day (0 = unlimited)

    def _init_strategy_risk_tracking(self) -> None:
        """Initialize per-strategy risk tracking."""
        if not hasattr(self, "_strategy_stop_requested"):
            self._strategy_stop_requested = False
        if not hasattr(self, "_strategy_pause_requested"):
            self._strategy_pause_requested = False
        if not hasattr(self, "_strategy_risk"):
            self._strategy_risk = {
                "initial_capital": Decimal("0"),
                "current_capital": Decimal("0"),
                "peak_capital": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "daily_loss": Decimal("0"),
                "daily_trades": 0,
                "daily_reset_date": None,
                "is_paused_by_risk": False,
                "is_stopped_by_risk": False,
                "last_check_time": None,
                "risk_events": [],  # Recent risk events
            }

    def set_strategy_initial_capital(self, capital: Decimal) -> None:
        """
        Set initial capital for per-strategy risk tracking.

        Args:
            capital: Initial capital allocated to this strategy
        """
        self._init_strategy_risk_tracking()
        self._strategy_risk["initial_capital"] = capital
        self._strategy_risk["current_capital"] = capital
        self._strategy_risk["peak_capital"] = capital
        logger.info(f"[{self._bot_id}] Strategy initial capital set: {capital}")

    def update_strategy_capital(
        self,
        realized_pnl: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None,
    ) -> None:
        """
        Update strategy capital tracking.

        Args:
            realized_pnl: Realized P&L to add
            unrealized_pnl: Current unrealized P&L
        """
        self._init_strategy_risk_tracking()
        risk = self._strategy_risk

        if realized_pnl is not None:
            risk["realized_pnl"] += realized_pnl
            # Track win/loss streaks
            if realized_pnl > 0:
                risk["consecutive_wins"] += 1
                risk["consecutive_losses"] = 0
            elif realized_pnl < 0:
                risk["consecutive_losses"] += 1
                risk["consecutive_wins"] = 0
                risk["daily_loss"] += abs(realized_pnl)

            risk["daily_trades"] += 1

        if unrealized_pnl is not None:
            risk["unrealized_pnl"] = unrealized_pnl

        # Update current capital
        risk["current_capital"] = (
            risk["initial_capital"] + risk["realized_pnl"] + risk["unrealized_pnl"]
        )

        # Update peak
        if risk["current_capital"] > risk["peak_capital"]:
            risk["peak_capital"] = risk["current_capital"]

    async def check_strategy_risk(self) -> Dict[str, Any]:
        """
        Check per-strategy risk levels.

        Returns:
            Dictionary with risk check results and actions taken
        """
        self._init_strategy_risk_tracking()
        risk = self._strategy_risk
        risk["last_check_time"] = datetime.now(timezone.utc)

        # Check for daily reset
        today = datetime.now(timezone.utc).date()
        if risk["daily_reset_date"] != today:
            risk["daily_loss"] = Decimal("0")
            risk["daily_trades"] = 0
            risk["daily_reset_date"] = today

        result = {
            "risk_level": "NORMAL",
            "action": None,
            "alerts": [],
            "metrics": {},
        }

        initial = risk["initial_capital"]
        if initial <= 0:
            return result

        current = risk["current_capital"]

        # Calculate metrics
        total_pnl = risk["realized_pnl"] + risk["unrealized_pnl"]
        loss_pct = abs(total_pnl / initial) if total_pnl < 0 else Decimal("0")
        drawdown_pct = (
            (risk["peak_capital"] - current) / risk["peak_capital"]
            if risk["peak_capital"] > 0 else Decimal("0")
        )

        result["metrics"] = {
            "total_pnl": total_pnl,
            "loss_pct": loss_pct,
            "drawdown_pct": drawdown_pct,
            "consecutive_losses": risk["consecutive_losses"],
            "daily_loss": risk["daily_loss"],
            "current_capital": current,
            "peak_capital": risk["peak_capital"],
        }

        # Check thresholds - from most severe to least
        if loss_pct >= self.STRATEGY_LOSS_STOP_PCT:
            result["risk_level"] = "CRITICAL"
            result["action"] = "STOP_SELF"
            result["alerts"].append(
                f"Strategy loss {loss_pct:.2%} >= {self.STRATEGY_LOSS_STOP_PCT:.2%} - STOPPING"
            )
            risk["is_stopped_by_risk"] = True
            await self._on_strategy_risk_stop(result)

        elif loss_pct >= self.STRATEGY_LOSS_PAUSE_PCT or drawdown_pct >= self.STRATEGY_DRAWDOWN_PAUSE_PCT:
            result["risk_level"] = "DANGER"
            result["action"] = "PAUSE_SELF"
            if loss_pct >= self.STRATEGY_LOSS_PAUSE_PCT:
                result["alerts"].append(
                    f"Strategy loss {loss_pct:.2%} >= {self.STRATEGY_LOSS_PAUSE_PCT:.2%}"
                )
            if drawdown_pct >= self.STRATEGY_DRAWDOWN_PAUSE_PCT:
                result["alerts"].append(
                    f"Strategy drawdown {drawdown_pct:.2%} >= {self.STRATEGY_DRAWDOWN_PAUSE_PCT:.2%}"
                )
            risk["is_paused_by_risk"] = True
            await self._on_strategy_risk_pause(result)

        elif risk["consecutive_losses"] >= self.STRATEGY_CONSECUTIVE_LOSS_PAUSE:
            result["risk_level"] = "DANGER"
            result["action"] = "PAUSE_SELF"
            result["alerts"].append(
                f"Consecutive losses {risk['consecutive_losses']} >= {self.STRATEGY_CONSECUTIVE_LOSS_PAUSE}"
            )
            risk["is_paused_by_risk"] = True
            await self._on_strategy_risk_pause(result)

        elif loss_pct >= self.STRATEGY_LOSS_WARNING_PCT or drawdown_pct >= self.STRATEGY_DRAWDOWN_WARNING_PCT:
            result["risk_level"] = "WARNING"
            result["action"] = "NOTIFY"
            if loss_pct >= self.STRATEGY_LOSS_WARNING_PCT:
                result["alerts"].append(
                    f"Strategy loss {loss_pct:.2%} >= {self.STRATEGY_LOSS_WARNING_PCT:.2%}"
                )
            if drawdown_pct >= self.STRATEGY_DRAWDOWN_WARNING_PCT:
                result["alerts"].append(
                    f"Strategy drawdown {drawdown_pct:.2%} >= {self.STRATEGY_DRAWDOWN_WARNING_PCT:.2%}"
                )
            await self._on_strategy_risk_warning(result)

        elif risk["consecutive_losses"] >= self.STRATEGY_CONSECUTIVE_LOSS_WARNING:
            result["risk_level"] = "WARNING"
            result["action"] = "NOTIFY"
            result["alerts"].append(
                f"Consecutive losses {risk['consecutive_losses']} >= {self.STRATEGY_CONSECUTIVE_LOSS_WARNING}"
            )
            await self._on_strategy_risk_warning(result)

        # Check daily trade count limit
        if self.STRATEGY_MAX_DAILY_TRADES > 0 and risk["daily_trades"] >= self.STRATEGY_MAX_DAILY_TRADES:
            if result["risk_level"] in ("NORMAL", "WARNING"):
                result["risk_level"] = "DANGER"
                result["action"] = "PAUSE_SELF"
                risk["is_paused_by_risk"] = True
                await self._on_strategy_risk_pause(result)
            result["alerts"].append(
                f"Daily trade limit reached ({risk['daily_trades']} >= {self.STRATEGY_MAX_DAILY_TRADES})"
            )

        # Record event if there's an alert
        if result["alerts"]:
            risk["risk_events"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": result["risk_level"],
                "action": result["action"],
                "alerts": result["alerts"],
            })
            # Keep only last 50 events
            risk["risk_events"] = risk["risk_events"][-50:]

        return result

    async def _on_strategy_risk_warning(self, result: Dict[str, Any]) -> None:
        """
        Handle strategy risk warning - notify but don't stop.
        Does NOT affect other strategies.
        """
        logger.warning(
            f"[{self._bot_id}] Strategy risk WARNING: {result['alerts']}"
        )
        if self._notifier:
            await self._notifier.send_warning(
                title=f"⚠️ Strategy Risk Warning: {self._bot_id}",
                message=(
                    f"Alerts: {', '.join(result['alerts'])}\n"
                    f"Loss: {result['metrics'].get('loss_pct', 0):.2%}\n"
                    f"Drawdown: {result['metrics'].get('drawdown_pct', 0):.2%}\n"
                    f"Consecutive Losses: {result['metrics'].get('consecutive_losses', 0)}"
                ),
            )

    async def _on_strategy_risk_pause(self, result: Dict[str, Any]) -> None:
        """
        Handle strategy risk pause - pause ONLY this bot, not others.
        Other healthy strategies continue operating.
        """
        logger.error(
            f"[{self._bot_id}] Strategy risk PAUSE: {result['alerts']}"
        )
        if self._notifier:
            await self._notifier.send_alert(
                title=f"🛑 Strategy Paused (Risk): {self._bot_id}",
                message=(
                    f"This strategy is paused due to risk limits.\n"
                    f"Other strategies are NOT affected.\n\n"
                    f"Alerts: {', '.join(result['alerts'])}\n"
                    f"Loss: {result['metrics'].get('loss_pct', 0):.2%}\n"
                    f"Drawdown: {result['metrics'].get('drawdown_pct', 0):.2%}"
                ),
            )
        # Signal pause request instead of calling pause() directly
        self._strategy_pause_requested = True
        logger.error(f"[{self._bot_id}] Strategy pause requested - will execute at next safe point")

    async def _on_strategy_risk_stop(self, result: Dict[str, Any]) -> None:
        """
        Handle strategy risk stop - stop ONLY this bot with position clearing.
        Other healthy strategies continue operating.
        """
        logger.critical(
            f"[{self._bot_id}] Strategy risk STOP: {result['alerts']}"
        )
        if self._notifier:
            await self._notifier.send_alert(
                title=f"🚨 Strategy Stopped (Risk): {self._bot_id}",
                message=(
                    f"This strategy is STOPPED due to severe risk breach.\n"
                    f"Other strategies are NOT affected.\n\n"
                    f"Alerts: {', '.join(result['alerts'])}\n"
                    f"Loss: {result['metrics'].get('loss_pct', 0):.2%}\n"
                    f"Action: Closing positions and stopping"
                ),
            )
        # Signal stop request instead of calling stop() directly
        # (calling stop() in check_strategy_risk interrupts the execution flow)
        self._strategy_stop_requested = True
        logger.critical(f"[{self._bot_id}] Strategy stop requested - will execute at next safe point")

    def reset_strategy_risk_pause(self) -> bool:
        """
        Manually reset strategy risk pause state.
        Call this after reviewing and acknowledging the risk event.

        Returns:
            True if reset successful
        """
        self._init_strategy_risk_tracking()
        if not self._strategy_risk["is_paused_by_risk"]:
            return False

        self._strategy_risk["is_paused_by_risk"] = False
        self._strategy_risk["consecutive_losses"] = 0
        logger.info(f"[{self._bot_id}] Strategy risk pause reset")
        return True

    def get_strategy_risk_status(self) -> Dict[str, Any]:
        """
        Get current strategy risk status.

        Returns:
            Dictionary with all risk metrics
        """
        self._init_strategy_risk_tracking()
        risk = self._strategy_risk
        initial = risk["initial_capital"]

        total_pnl = risk["realized_pnl"] + risk["unrealized_pnl"]
        loss_pct = abs(total_pnl / initial) if initial > 0 and total_pnl < 0 else Decimal("0")
        drawdown_pct = (
            (risk["peak_capital"] - risk["current_capital"]) / risk["peak_capital"]
            if risk["peak_capital"] > 0 else Decimal("0")
        )

        return {
            "bot_id": self._bot_id,
            "initial_capital": str(risk["initial_capital"]),
            "current_capital": str(risk["current_capital"]),
            "peak_capital": str(risk["peak_capital"]),
            "realized_pnl": str(risk["realized_pnl"]),
            "unrealized_pnl": str(risk["unrealized_pnl"]),
            "total_pnl": str(total_pnl),
            "loss_pct": str(loss_pct),
            "drawdown_pct": str(drawdown_pct),
            "consecutive_losses": risk["consecutive_losses"],
            "consecutive_wins": risk["consecutive_wins"],
            "daily_loss": str(risk["daily_loss"]),
            "daily_trades": risk["daily_trades"],
            "max_daily_trades": self.STRATEGY_MAX_DAILY_TRADES,
            "is_paused_by_risk": risk["is_paused_by_risk"],
            "is_stopped_by_risk": risk["is_stopped_by_risk"],
            "last_check_time": risk["last_check_time"].isoformat() if risk["last_check_time"] else None,
            "recent_events": risk["risk_events"][-5:],
        }

    # =========================================================================
    # Virtual Position Ledger (淨倉位管理)
    # Track per-bot virtual positions separately from net exchange position
    # =========================================================================

    def _init_virtual_position_ledger(self) -> None:
        """Initialize virtual position tracking."""
        if not hasattr(self, "_virtual_positions"):
            self._virtual_positions: Dict[str, Dict[str, Any]] = {}
            # symbol -> {
            #   "quantity": Decimal,
            #   "side": str,  # "LONG" or "SHORT"
            #   "avg_entry_price": Decimal,
            #   "fills": List[Dict],  # Individual fills for attribution
            #   "unrealized_pnl": Decimal,
            #   "realized_pnl": Decimal,
            #   "last_sync_time": datetime,
            #   "drift_detected": bool,
            # }

    def record_virtual_fill(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: Decimal,
        price: Decimal,
        order_id: str,
        fee: Decimal = Decimal("0"),
        is_reduce_only: bool = False,
        leverage: int = 1,
        skip_pnl_update: bool = False,
    ) -> Dict[str, Any]:
        """
        Record a fill in the virtual position ledger.

        This tracks what THIS bot thinks it owns, separately from exchange net position.

        Args:
            symbol: Trading symbol
            side: Fill side (BUY or SELL)
            quantity: Fill quantity
            price: Fill price
            order_id: Order ID for attribution
            fee: Trading fee
            is_reduce_only: Whether this is a reduce-only order

        Returns:
            Updated virtual position state
        """
        self._init_virtual_position_ledger()

        fill_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "side": side,
            "quantity": str(quantity),
            "price": str(price),
            "fee": str(fee),
            "is_reduce_only": is_reduce_only,
        }

        if symbol not in self._virtual_positions:
            # New position
            self._virtual_positions[symbol] = {
                "quantity": Decimal("0"),
                "side": None,
                "avg_entry_price": Decimal("0"),
                "fills": [],
                "unrealized_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "last_sync_time": None,
                "drift_detected": False,
            }

        pos = self._virtual_positions[symbol]
        pos["fills"].append(fill_record)

        # Keep only last 100 fills per symbol
        pos["fills"] = pos["fills"][-100:]

        # Calculate position change
        current_qty = pos["quantity"]
        current_side = pos["side"]
        realized = None  # Set when a closing trade calculates realized PnL

        if side == "BUY":
            if current_side is None or current_side == "LONG" or current_qty == 0:
                # Adding to long position
                total_value = (current_qty * pos["avg_entry_price"]) + (quantity * price)
                pos["quantity"] = current_qty + quantity
                if pos["quantity"] > 0:
                    pos["avg_entry_price"] = total_value / pos["quantity"]
                pos["side"] = "LONG"
            else:
                # Reducing short position (BUY to close SHORT)
                close_qty = min(quantity, current_qty)
                close_fee = fee * close_qty / quantity if quantity > 0 else fee
                realized = (pos["avg_entry_price"] - price) * close_qty
                pos["realized_pnl"] += realized - close_fee
                pos["quantity"] = current_qty - quantity
                if pos["quantity"] <= 0:
                    if pos["quantity"] < 0:
                        # Flipped to long
                        pos["quantity"] = abs(pos["quantity"])
                        pos["side"] = "LONG"
                        pos["avg_entry_price"] = price
                    else:
                        pos["side"] = None
                        pos["avg_entry_price"] = Decimal("0")

        elif side == "SELL":
            if current_side is None or current_side == "SHORT" or current_qty == 0:
                # Adding to short position
                total_value = (current_qty * pos["avg_entry_price"]) + (quantity * price)
                pos["quantity"] = current_qty + quantity
                if pos["quantity"] > 0:
                    pos["avg_entry_price"] = total_value / pos["quantity"]
                pos["side"] = "SHORT"
            else:
                # Reducing long position (SELL to close LONG)
                close_qty = min(quantity, current_qty)
                close_fee = fee * close_qty / quantity if quantity > 0 else fee
                realized = (price - pos["avg_entry_price"]) * close_qty
                pos["realized_pnl"] += realized - close_fee
                pos["quantity"] = current_qty - quantity
                if pos["quantity"] <= 0:
                    if pos["quantity"] < 0:
                        # Flipped to short
                        pos["quantity"] = abs(pos["quantity"])
                        pos["side"] = "SHORT"
                        pos["avg_entry_price"] = price
                    else:
                        pos["side"] = None
                        pos["avg_entry_price"] = Decimal("0")

        # Update strategy risk tracking (skip if caller already updated, e.g. FIFO close)
        if not skip_pnl_update:
            if fill_record["is_reduce_only"] or (side == "SELL" and current_side == "LONG") or (side == "BUY" and current_side == "SHORT"):
                # This was a closing trade - realized is defined in the closing branches above
                if realized is not None:
                    # Use proportional close_fee (same as virtual ledger) to avoid double-counting
                    close_qty = min(quantity, current_qty) if current_qty > 0 else quantity
                    proportional_fee = fee * close_qty / quantity if quantity > 0 else fee
                    self.update_strategy_capital(realized_pnl=realized - proportional_fee)

        logger.debug(
            f"[{self._bot_id}] Virtual fill recorded: {side} {quantity} {symbol} @ {price}, "
            f"Position now: {pos['side']} {pos['quantity']}"
        )

        return self._get_virtual_position_summary(symbol)

    def update_virtual_unrealized_pnl(
        self,
        symbol: str,
        current_price: Decimal,
        leverage: int = 1,
    ) -> Decimal:
        """
        Update unrealized P&L for a virtual position.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Unrealized P&L amount
        """
        self._init_virtual_position_ledger()

        if symbol not in self._virtual_positions:
            return Decimal("0")

        pos = self._virtual_positions[symbol]
        if pos["quantity"] <= 0:
            pos["unrealized_pnl"] = Decimal("0")
            return Decimal("0")

        if pos["side"] == "LONG":
            pos["unrealized_pnl"] = (current_price - pos["avg_entry_price"]) * pos["quantity"]
        elif pos["side"] == "SHORT":
            pos["unrealized_pnl"] = (pos["avg_entry_price"] - current_price) * pos["quantity"]

        # Update strategy risk tracking
        self.update_strategy_capital(unrealized_pnl=pos["unrealized_pnl"])

        return pos["unrealized_pnl"]

    def get_virtual_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get virtual position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Virtual position dict or None
        """
        self._init_virtual_position_ledger()
        return self._get_virtual_position_summary(symbol)

    def _get_virtual_position_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get formatted virtual position summary."""
        if symbol not in self._virtual_positions:
            return None

        pos = self._virtual_positions[symbol]
        return {
            "symbol": symbol,
            "bot_id": self._bot_id,
            "quantity": str(pos["quantity"]),
            "side": pos["side"],
            "avg_entry_price": str(pos["avg_entry_price"]),
            "unrealized_pnl": str(pos["unrealized_pnl"]),
            "realized_pnl": str(pos["realized_pnl"]),
            "fill_count": len(pos["fills"]),
            "last_sync_time": pos["last_sync_time"].isoformat() if pos["last_sync_time"] else None,
            "drift_detected": pos["drift_detected"],
        }

    def get_all_virtual_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all virtual positions for this bot."""
        self._init_virtual_position_ledger()
        return {
            symbol: self._get_virtual_position_summary(symbol)
            for symbol in self._virtual_positions
        }

    async def reconcile_virtual_position(
        self,
        symbol: str,
        exchange_quantity: Decimal,
        exchange_side: Optional[str],
        tolerance_pct: Decimal = Decimal("0.01"),  # 1% tolerance
    ) -> Dict[str, Any]:
        """
        Reconcile virtual position with exchange position.

        Detects drift between what this bot thinks it owns vs exchange state.

        Args:
            symbol: Trading symbol
            exchange_quantity: Exchange reported quantity
            exchange_side: Exchange reported side
            tolerance_pct: Acceptable drift tolerance

        Returns:
            Reconciliation result with drift details
        """
        self._init_virtual_position_ledger()

        result = {
            "symbol": symbol,
            "bot_id": self._bot_id,
            "virtual_quantity": Decimal("0"),
            "virtual_side": None,
            "exchange_quantity": exchange_quantity,
            "exchange_side": exchange_side,
            "drift_quantity": Decimal("0"),
            "drift_pct": Decimal("0"),
            "drift_detected": False,
            "reconciled": True,
            "action_needed": None,
        }

        if symbol not in self._virtual_positions:
            # No virtual position but exchange has position
            if exchange_quantity > 0:
                result["drift_detected"] = True
                result["drift_quantity"] = exchange_quantity
                result["action_needed"] = "INVESTIGATE_ORPHAN_POSITION"
                result["reconciled"] = False
            return result

        pos = self._virtual_positions[symbol]
        virtual_qty = pos["quantity"]
        virtual_side = pos["side"]

        result["virtual_quantity"] = virtual_qty
        result["virtual_side"] = virtual_side

        # Side mismatch
        if virtual_side and exchange_side and virtual_side != exchange_side.upper():
            result["drift_detected"] = True
            result["action_needed"] = "SIDE_MISMATCH"
            result["reconciled"] = False
            pos["drift_detected"] = True
            logger.error(
                f"[{self._bot_id}] Position side mismatch for {symbol}: "
                f"Virtual={virtual_side}, Exchange={exchange_side}"
            )
            return result

        # Quantity drift
        drift = abs(virtual_qty - exchange_quantity)
        result["drift_quantity"] = drift

        if virtual_qty > 0:
            result["drift_pct"] = drift / virtual_qty
        elif exchange_quantity > 0:
            result["drift_pct"] = Decimal("1")  # 100% drift if we think we have 0

        if result["drift_pct"] > tolerance_pct:
            result["drift_detected"] = True
            pos["drift_detected"] = True

            if virtual_qty > exchange_quantity:
                result["action_needed"] = "VIRTUAL_OVERCOUNTED"
            else:
                result["action_needed"] = "VIRTUAL_UNDERCOUNTED"

            result["reconciled"] = False

            logger.warning(
                f"[{self._bot_id}] Position drift detected for {symbol}: "
                f"Virtual={virtual_qty}, Exchange={exchange_quantity}, Drift={result['drift_pct']:.2%}"
            )
        else:
            pos["drift_detected"] = False

        pos["last_sync_time"] = datetime.now(timezone.utc)

        return result

    def correct_virtual_position(
        self,
        symbol: str,
        correct_quantity: Decimal,
        correct_side: Optional[str],
        reason: str = "Manual correction",
    ) -> bool:
        """
        Manually correct virtual position to match exchange.

        Use this after investigating drift.

        Args:
            symbol: Trading symbol
            correct_quantity: Correct quantity
            correct_side: Correct side
            reason: Reason for correction

        Returns:
            True if correction applied
        """
        self._init_virtual_position_ledger()

        if symbol not in self._virtual_positions:
            self._virtual_positions[symbol] = {
                "quantity": Decimal("0"),
                "side": None,
                "avg_entry_price": Decimal("0"),
                "fills": [],
                "unrealized_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "last_sync_time": None,
                "drift_detected": False,
            }

        pos = self._virtual_positions[symbol]
        old_qty = pos["quantity"]
        old_side = pos["side"]

        pos["quantity"] = correct_quantity
        pos["side"] = correct_side.upper() if correct_side else None
        pos["drift_detected"] = False
        pos["last_sync_time"] = datetime.now(timezone.utc)

        # Record correction as a fill
        pos["fills"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": "MANUAL_CORRECTION",
            "side": "CORRECTION",
            "quantity": str(correct_quantity - old_qty),
            "price": "0",
            "fee": "0",
            "reason": reason,
            "old_quantity": str(old_qty),
            "old_side": old_side,
        })

        logger.info(
            f"[{self._bot_id}] Virtual position corrected for {symbol}: "
            f"{old_side} {old_qty} -> {pos['side']} {pos['quantity']}, Reason: {reason}"
        )

        return True

    # =========================================================================
    # Cost Basis and P&L Attribution (持倉歸屬不清)
    # FIFO-based cost basis tracking for accurate profit attribution
    # =========================================================================

    def _init_cost_basis_tracking(self) -> None:
        """Initialize cost basis tracking."""
        if not hasattr(self, "_cost_basis_ledger"):
            self._cost_basis_ledger: Dict[str, List[Dict[str, Any]]] = {}
            # symbol -> List of open lots (FIFO order)
            # Each lot: {
            #   "lot_id": str,
            #   "timestamp": datetime,
            #   "quantity": Decimal,
            #   "remaining_quantity": Decimal,
            #   "entry_price": Decimal,
            #   "order_id": str,
            #   "fee": Decimal,
            # }

    def record_cost_basis_entry(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        order_id: str,
        fee: Decimal = Decimal("0"),
    ) -> str:
        """
        Record a new cost basis entry (opening position).

        Uses FIFO (First-In-First-Out) for P&L calculation.

        Args:
            symbol: Trading symbol
            quantity: Entry quantity
            price: Entry price
            order_id: Order ID for tracking
            fee: Entry fee

        Returns:
            Lot ID for this entry
        """
        self._init_cost_basis_tracking()

        if symbol not in self._cost_basis_ledger:
            self._cost_basis_ledger[symbol] = []

        lot_id = f"{self._bot_id}_{symbol}_{order_id}_{int(time.time()*1000)}"

        lot = {
            "lot_id": lot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quantity": quantity,
            "remaining_quantity": quantity,
            "entry_price": price,
            "order_id": order_id,
            "fee": fee,
        }

        self._cost_basis_ledger[symbol].append(lot)

        logger.debug(
            f"[{self._bot_id}] Cost basis entry: {lot_id}, "
            f"{quantity} {symbol} @ {price}"
        )

        return lot_id

    def close_cost_basis_fifo(
        self,
        symbol: str,
        close_quantity: Decimal,
        close_price: Decimal,
        close_order_id: str,
        close_fee: Decimal = Decimal("0"),
        leverage: int = 1,
        side: str = "SELL",
    ) -> Dict[str, Any]:
        """
        Close position using FIFO cost basis matching.

        Matches against oldest open lots first.

        Args:
            symbol: Trading symbol
            close_quantity: Quantity to close
            close_price: Closing price
            close_order_id: Close order ID
            close_fee: Closing fee

        Returns:
            P&L attribution results
        """
        self._init_cost_basis_tracking()

        result = {
            "symbol": symbol,
            "bot_id": self._bot_id,
            "close_order_id": close_order_id,
            "close_quantity": str(close_quantity),
            "close_price": str(close_price),
            "matched_lots": [],
            "total_cost_basis": Decimal("0"),
            "total_realized_pnl": Decimal("0"),
            "total_fees": Decimal("0"),
            "unmatched_quantity": Decimal("0"),
        }

        if symbol not in self._cost_basis_ledger:
            result["unmatched_quantity"] = close_quantity
            logger.warning(
                f"[{self._bot_id}] No cost basis found for {symbol}, "
                f"unmatched close: {close_quantity}"
            )
            return result

        remaining_to_close = close_quantity
        lots = self._cost_basis_ledger[symbol]

        for lot in lots:
            if remaining_to_close <= 0:
                break

            if lot["remaining_quantity"] <= 0:
                continue

            # Match against this lot (FIFO)
            match_qty = min(remaining_to_close, lot["remaining_quantity"])

            # Calculate P&L for this lot
            cost_basis = match_qty * lot["entry_price"]
            proceeds = match_qty * close_price
            # Proportional entry fee (guard against division by zero)
            entry_fee_portion = lot["fee"] * (match_qty / lot["quantity"]) if lot["quantity"] > 0 else Decimal("0")
            # Proportional exit fee
            exit_fee_portion = close_fee * (match_qty / close_quantity) if close_quantity > 0 else Decimal("0")

            lot_pnl = (proceeds - cost_basis) - entry_fee_portion - exit_fee_portion

            # Record match
            result["matched_lots"].append({
                "lot_id": lot["lot_id"],
                "matched_quantity": str(match_qty),
                "entry_price": str(lot["entry_price"]),
                "entry_timestamp": lot["timestamp"],
                "cost_basis": str(cost_basis),
                "proceeds": str(proceeds),
                "pnl": str(lot_pnl),
                "entry_fee_portion": str(entry_fee_portion),
                "exit_fee_portion": str(exit_fee_portion),
            })

            result["total_cost_basis"] += cost_basis
            result["total_realized_pnl"] += lot_pnl
            result["total_fees"] += entry_fee_portion + exit_fee_portion

            # Update lot
            lot["remaining_quantity"] -= match_qty
            remaining_to_close -= match_qty

        result["unmatched_quantity"] = remaining_to_close

        # Clean up empty lots
        self._cost_basis_ledger[symbol] = [
            lot for lot in lots if lot["remaining_quantity"] > 0
        ]

        # Update strategy risk with realized P&L
        self.update_strategy_capital(realized_pnl=result["total_realized_pnl"])

        # Also record in virtual ledger (skip PnL update to avoid double counting -
        # FIFO result already updated strategy capital above)
        self.record_virtual_fill(
            symbol=symbol,
            side=side,  # Use caller-specified side
            quantity=close_quantity,
            price=close_price,
            order_id=close_order_id,
            fee=close_fee,
            is_reduce_only=True,
            leverage=leverage,
            skip_pnl_update=True,
        )

        logger.info(
            f"[{self._bot_id}] FIFO close: {close_quantity} {symbol} @ {close_price}, "
            f"P&L: {result['total_realized_pnl']}, "
            f"Matched {len(result['matched_lots'])} lots"
        )

        return result

    def get_open_lots(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all open lots for a symbol (for position attribution).

        Args:
            symbol: Trading symbol

        Returns:
            List of open lots with remaining quantity
        """
        self._init_cost_basis_tracking()

        if symbol not in self._cost_basis_ledger:
            return []

        return [
            {
                "lot_id": lot["lot_id"],
                "timestamp": lot["timestamp"],
                "entry_price": str(lot["entry_price"]),
                "original_quantity": str(lot["quantity"]),
                "remaining_quantity": str(lot["remaining_quantity"]),
                "order_id": lot["order_id"],
                "fee": str(lot["fee"]),
            }
            for lot in self._cost_basis_ledger[symbol]
            if lot["remaining_quantity"] > 0
        ]

    def get_weighted_avg_cost_basis(self, symbol: str) -> Optional[Decimal]:
        """
        Get weighted average cost basis for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Weighted average entry price or None
        """
        self._init_cost_basis_tracking()

        if symbol not in self._cost_basis_ledger:
            return None

        total_value = Decimal("0")
        total_qty = Decimal("0")

        for lot in self._cost_basis_ledger[symbol]:
            if lot["remaining_quantity"] > 0:
                total_value += lot["remaining_quantity"] * lot["entry_price"]
                total_qty += lot["remaining_quantity"]

        if total_qty <= 0:
            return None

        return total_value / total_qty

    def get_total_position_from_lots(self, symbol: str) -> Decimal:
        """
        Get total position size from open lots.

        Args:
            symbol: Trading symbol

        Returns:
            Total quantity from all open lots
        """
        self._init_cost_basis_tracking()

        if symbol not in self._cost_basis_ledger:
            return Decimal("0")

        return sum(
            lot["remaining_quantity"]
            for lot in self._cost_basis_ledger[symbol]
            if lot["remaining_quantity"] > 0
        )

    def get_pnl_attribution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive P&L attribution summary for this bot.

        Returns:
            Summary of all positions, realized P&L, and attribution
        """
        self._init_virtual_position_ledger()
        self._init_cost_basis_tracking()

        summary = {
            "bot_id": self._bot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": {},
            "total_realized_pnl": Decimal("0"),
            "total_unrealized_pnl": Decimal("0"),
            "total_open_lots": 0,
        }

        # From virtual positions
        for symbol, pos in self._virtual_positions.items():
            summary["total_realized_pnl"] += pos["realized_pnl"]
            summary["total_unrealized_pnl"] += pos["unrealized_pnl"]

            summary["positions"][symbol] = {
                "virtual_quantity": str(pos["quantity"]),
                "virtual_side": pos["side"],
                "avg_entry_price": str(pos["avg_entry_price"]),
                "realized_pnl": str(pos["realized_pnl"]),
                "unrealized_pnl": str(pos["unrealized_pnl"]),
                "open_lots": [],
            }

        # Add lot details
        for symbol, lots in self._cost_basis_ledger.items():
            open_lots = [
                {
                    "lot_id": lot["lot_id"],
                    "entry_price": str(lot["entry_price"]),
                    "remaining_quantity": str(lot["remaining_quantity"]),
                    "timestamp": lot["timestamp"],
                }
                for lot in lots
                if lot["remaining_quantity"] > 0
            ]

            summary["total_open_lots"] += len(open_lots)

            if symbol in summary["positions"]:
                summary["positions"][symbol]["open_lots"] = open_lots
            elif open_lots:
                summary["positions"][symbol] = {
                    "virtual_quantity": "0",
                    "virtual_side": None,
                    "open_lots": open_lots,
                }

        summary["total_realized_pnl"] = str(summary["total_realized_pnl"])
        summary["total_unrealized_pnl"] = str(summary["total_unrealized_pnl"])

        return summary

    # =========================================================================
    # Proportional Partial Close Allocation (部分平倉分配)
    # Fairly distribute partial closes across strategies sharing a position
    # =========================================================================

    def _init_partial_close_tracker(self) -> None:
        """Initialize partial close tracking."""
        if not hasattr(self, "_partial_close_log"):
            self._partial_close_log: List[Dict[str, Any]] = []
            # Log of all partial closes for audit

    def allocate_partial_close(
        self,
        symbol: str,
        total_close_quantity: Decimal,
        close_price: Decimal,
        close_order_id: str,
        contributing_bots: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """
        Allocate a partial close proportionally across contributing strategies.

        When multiple strategies share a position (via SharedPositionManager),
        this method distributes the close quantity fairly based on each
        strategy's contribution.

        Args:
            symbol: Trading symbol
            total_close_quantity: Total quantity being closed
            close_price: Close price
            close_order_id: Order ID for tracking
            contributing_bots: Dict of bot_id -> contributed_quantity
                              If None, uses virtual position for this bot only

        Returns:
            Allocation result with per-bot P&L attribution
        """
        self._init_partial_close_tracker()
        self._init_cost_basis_tracking()

        result = {
            "symbol": symbol,
            "total_close_quantity": str(total_close_quantity),
            "close_price": str(close_price),
            "close_order_id": close_order_id,
            "allocations": {},
            "total_pnl": Decimal("0"),
            "unallocated_quantity": Decimal("0"),
        }

        # If no contributing bots provided, this bot owns entire position
        if contributing_bots is None:
            contributing_bots = {self._bot_id: total_close_quantity}

        # Calculate total contribution
        total_contribution = sum(contributing_bots.values())

        if total_contribution <= 0:
            result["unallocated_quantity"] = total_close_quantity
            logger.warning(
                f"[{self._bot_id}] No contributions found for partial close: {symbol}"
            )
            return result

        remaining_to_allocate = total_close_quantity

        # Allocate proportionally to each bot
        for bot_id, contribution in contributing_bots.items():
            if contribution <= 0:
                continue

            # Calculate proportional share
            share_pct = contribution / total_contribution
            allocated_qty = (total_close_quantity * share_pct).quantize(
                Decimal("0.00000001")
            )

            # Ensure we don't over-allocate
            allocated_qty = min(allocated_qty, remaining_to_allocate, contribution)

            if allocated_qty <= 0:
                continue

            # For this bot, use FIFO cost basis
            if bot_id == self._bot_id:
                close_result = self.close_cost_basis_fifo(
                    symbol=symbol,
                    close_quantity=allocated_qty,
                    close_price=close_price,
                    close_order_id=close_order_id,
                )
                bot_pnl = close_result.get("total_realized_pnl", Decimal("0"))
            else:
                # For other bots, calculate estimated P&L
                # (They should call their own close_cost_basis_fifo)
                bot_pnl = Decimal("0")  # Unknown for other bots

            result["allocations"][bot_id] = {
                "original_contribution": str(contribution),
                "allocated_quantity": str(allocated_qty),
                "share_pct": str(share_pct),
                "realized_pnl": str(bot_pnl),
            }

            result["total_pnl"] += bot_pnl
            remaining_to_allocate -= allocated_qty

        result["unallocated_quantity"] = str(remaining_to_allocate)
        result["total_pnl"] = str(result["total_pnl"])

        # Log partial close for audit
        self._partial_close_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "close_order_id": close_order_id,
            "total_quantity": str(total_close_quantity),
            "allocations": result["allocations"],
        })

        # Keep only last 100 entries
        self._partial_close_log = self._partial_close_log[-100:]

        logger.info(
            f"[{self._bot_id}] Partial close allocated: {symbol}, "
            f"Total={total_close_quantity}, "
            f"Bots={len(result['allocations'])}, "
            f"P&L={result['total_pnl']}"
        )

        return result

    def get_partial_close_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get partial close history for audit."""
        self._init_partial_close_tracker()

        history = self._partial_close_log
        if symbol:
            history = [h for h in history if h["symbol"] == symbol]

        return history[-limit:]

    # =========================================================================
    # Risk Bypass Prevention (風控繞過防護)
    # Ensure risk controls cannot be silently bypassed
    # =========================================================================

    # Data freshness thresholds
    RISK_DATA_MAX_AGE_SECONDS = 60  # Max age of risk data before considered stale
    RISK_CHECK_RETRY_COUNT = 3  # Retry count for failed risk checks
    RISK_CHECK_RETRY_DELAY = 1.0  # Seconds between retries

    def _init_risk_bypass_prevention(self) -> None:
        """Initialize risk bypass prevention tracking."""
        if not hasattr(self, "_risk_bypass_tracking"):
            self._risk_bypass_tracking = {
                "last_capital_update": None,
                "last_position_sync": None,
                "failed_risk_checks": 0,
                "bypass_alerts": [],
                "is_risk_data_stale": False,
                "forced_pause_due_to_stale_data": False,
            }

    async def verify_risk_data_freshness(self) -> tuple[bool, str]:
        """
        Verify that risk-related data is fresh enough for decision making.

        Returns:
            Tuple of (is_fresh, message)
        """
        self._init_risk_bypass_prevention()
        tracking = self._risk_bypass_tracking
        now = datetime.now(timezone.utc)

        issues = []

        # Check capital data freshness
        if tracking["last_capital_update"]:
            age = (now - tracking["last_capital_update"]).total_seconds()
            if age > self.RISK_DATA_MAX_AGE_SECONDS:
                issues.append(f"Capital data stale ({age:.0f}s old)")
        else:
            issues.append("Capital data never updated")

        # Check position sync freshness
        if tracking["last_position_sync"]:
            age = (now - tracking["last_position_sync"]).total_seconds()
            if age > self.RISK_DATA_MAX_AGE_SECONDS * 2:  # More lenient for position
                issues.append(f"Position data stale ({age:.0f}s old)")

        if issues:
            tracking["is_risk_data_stale"] = True
            msg = "; ".join(issues)
            logger.warning(f"[{self._bot_id}] Risk data freshness issue: {msg}")
            return False, msg

        tracking["is_risk_data_stale"] = False
        return True, "Data is fresh"

    def mark_capital_updated(self) -> None:
        """Mark capital data as freshly updated."""
        self._init_risk_bypass_prevention()
        self._risk_bypass_tracking["last_capital_update"] = datetime.now(timezone.utc)

    def mark_position_synced(self) -> None:
        """Mark position data as freshly synced."""
        self._init_risk_bypass_prevention()
        self._risk_bypass_tracking["last_position_sync"] = datetime.now(timezone.utc)

    async def execute_risk_action_with_verification(
        self,
        action: str,
        action_func: Any,  # Callable or coroutine
        verification_func: Optional[Any] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute a risk action with verification and retry logic.

        Prevents silent failures in critical risk operations.

        Args:
            action: Description of the action
            action_func: Function/coroutine to execute
            verification_func: Optional function to verify success
            max_retries: Maximum retry attempts

        Returns:
            Result dict with success status and details
        """
        self._init_risk_bypass_prevention()

        result = {
            "action": action,
            "success": False,
            "attempts": 0,
            "verified": False,
            "error": None,
            "details": {},
        }

        for attempt in range(max_retries):
            result["attempts"] = attempt + 1

            try:
                # Execute action
                if asyncio.iscoroutinefunction(action_func):
                    action_result = await action_func()
                elif asyncio.iscoroutine(action_func):
                    action_result = await action_func
                else:
                    action_result = action_func()

                result["details"]["action_result"] = str(action_result)[:200]

                # Verify if verification function provided
                if verification_func:
                    await asyncio.sleep(0.5)  # Brief delay for state to settle

                    if asyncio.iscoroutinefunction(verification_func):
                        verified = await verification_func()
                    else:
                        verified = verification_func()

                    result["verified"] = bool(verified)

                    if not verified:
                        logger.warning(
                            f"[{self._bot_id}] Risk action '{action}' executed but "
                            f"verification failed (attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(self.RISK_CHECK_RETRY_DELAY)
                            continue
                        else:
                            result["error"] = "Verification failed after all retries"
                            return result
                else:
                    result["verified"] = True

                result["success"] = True
                logger.info(
                    f"[{self._bot_id}] Risk action '{action}' executed successfully "
                    f"(attempt {attempt + 1})"
                )
                return result

            except Exception as e:
                result["error"] = str(e)
                logger.error(
                    f"[{self._bot_id}] Risk action '{action}' failed "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(self.RISK_CHECK_RETRY_DELAY)

        # All retries failed - record bypass alert
        self._risk_bypass_tracking["failed_risk_checks"] += 1
        self._risk_bypass_tracking["bypass_alerts"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "error": result["error"],
            "attempts": result["attempts"],
        })

        # Keep only last 50 alerts
        self._risk_bypass_tracking["bypass_alerts"] = (
            self._risk_bypass_tracking["bypass_alerts"][-50:]
        )

        logger.critical(
            f"[{self._bot_id}] RISK BYPASS ALERT: Action '{action}' failed after "
            f"{max_retries} attempts. Error: {result['error']}"
        )

        return result

    async def safe_pre_trade_risk_check(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Perform pre-trade risk check with bypass prevention.

        This method ensures risk checks cannot be silently skipped.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price

        Returns:
            Tuple of (is_allowed, message, details)
        """
        self._init_risk_bypass_prevention()

        # Check circuit breaker first (H1 fix)
        cb_active, cb_reason = self.is_circuit_breaker_active()
        if cb_active:
            return False, f"Circuit breaker active: {cb_reason}", {"circuit_breaker": True}

        tracking = self._risk_bypass_tracking

        details = {
            "data_fresh": False,
            "risk_check_passed": False,
            "strategy_risk_ok": False,
            "bypass_prevented": False,
        }

        # 1. Verify data freshness
        is_fresh, freshness_msg = await self.verify_risk_data_freshness()
        details["data_fresh"] = is_fresh

        if not is_fresh:
            # Force update before proceeding
            try:
                await self._update_capital()
                self.mark_capital_updated()
                details["data_fresh"] = True
            except Exception as e:
                # Data stale and update failed - block trade
                details["bypass_prevented"] = True
                logger.warning(
                    f"[{self._bot_id}] Trade blocked due to stale data: {freshness_msg}"
                )
                return False, f"Risk data stale and refresh failed: {e}", details

        # 2. Check strategy-level risk
        try:
            risk_result = await self.check_strategy_risk()
            details["strategy_risk_level"] = risk_result.get("risk_level")

            # Process deferred stop/pause requests from strategy risk check
            if getattr(self, "_strategy_stop_requested", False):
                self._strategy_stop_requested = False
                await self.stop(clear_position=True)
                return False, "Strategy stopped by risk check", details
            if getattr(self, "_strategy_pause_requested", False):
                self._strategy_pause_requested = False
                await self.pause()
                return False, "Strategy paused by risk check", details

            if risk_result["risk_level"] in ["DANGER", "CRITICAL"]:
                details["strategy_risk_ok"] = False
                return False, f"Strategy risk too high: {risk_result['risk_level']}", details

            details["strategy_risk_ok"] = True

        except Exception as e:
            # Risk check failed - err on side of caution
            details["bypass_prevented"] = True
            logger.error(f"[{self._bot_id}] Strategy risk check failed: {e}")
            return False, f"Risk check failed (blocking trade): {e}", details

        # 3. Run full pre-order validation if available
        try:
            if hasattr(self, "_full_pre_order_validation"):
                is_allowed, msg, validation_details = await self._full_pre_order_validation(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                )
                details["risk_check_passed"] = is_allowed
                details.update(validation_details)

                if not is_allowed:
                    return False, msg, details
            else:
                details["risk_check_passed"] = True

        except Exception as e:
            details["bypass_prevented"] = True
            logger.error(f"[{self._bot_id}] Pre-order validation failed: {e}")
            return False, f"Pre-order validation failed: {e}", details

        return True, "All risk checks passed", details

    def get_risk_bypass_status(self) -> Dict[str, Any]:
        """Get current risk bypass prevention status."""
        self._init_risk_bypass_prevention()
        tracking = self._risk_bypass_tracking

        return {
            "bot_id": self._bot_id,
            "last_capital_update": (
                tracking["last_capital_update"].isoformat()
                if tracking["last_capital_update"] else None
            ),
            "last_position_sync": (
                tracking["last_position_sync"].isoformat()
                if tracking["last_position_sync"] else None
            ),
            "failed_risk_checks": tracking["failed_risk_checks"],
            "is_risk_data_stale": tracking["is_risk_data_stale"],
            "forced_pause_due_to_stale_data": tracking["forced_pause_due_to_stale_data"],
            "recent_bypass_alerts": tracking["bypass_alerts"][-5:],
        }

    # =========================================================================
    # Risk False Positive Prevention (風控誤判防護)
    # Prevent legitimate trades from being incorrectly blocked
    # =========================================================================

    # Alert persistence thresholds
    ALERT_PERSISTENCE_WINDOW_SECONDS = 30  # Time window for alert persistence check
    ALERT_PERSISTENCE_COUNT = 2  # Number of consecutive alerts before action
    CONSECUTIVE_LOSS_DECAY_HOURS = 24  # Hours before consecutive losses decay

    def _init_false_positive_prevention(self) -> None:
        """Initialize false positive prevention tracking."""
        if not hasattr(self, "_fp_prevention"):
            self._fp_prevention = {
                "alert_history": [],  # Recent alerts for persistence check
                "blocked_orders": [],  # Orders blocked for review
                "false_positive_overrides": [],  # Manual overrides
                "consecutive_loss_start_time": None,
                "price_deviation_overrides": {},  # symbol -> max_deviation
            }

    def record_risk_alert(
        self,
        alert_type: str,
        level: str,
        metric_value: Decimal,
        threshold: Decimal,
        message: str,
    ) -> bool:
        """
        Record a risk alert and check if it persists.

        Returns True if alert persists (should take action).

        Args:
            alert_type: Type of alert (e.g., "consecutive_loss", "drawdown")
            level: Alert level
            metric_value: Current metric value
            threshold: Threshold that was exceeded
            message: Alert message

        Returns:
            True if alert persists and action should be taken
        """
        self._init_false_positive_prevention()
        now = datetime.now(timezone.utc)

        # Record alert
        alert = {
            "timestamp": now.isoformat(),
            "type": alert_type,
            "level": level,
            "metric_value": str(metric_value),
            "threshold": str(threshold),
            "message": message,
        }
        self._fp_prevention["alert_history"].append(alert)

        # Keep only recent alerts
        cutoff = now - timedelta(seconds=self.ALERT_PERSISTENCE_WINDOW_SECONDS * 2)
        self._fp_prevention["alert_history"] = [
            a for a in self._fp_prevention["alert_history"]
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]

        # Check persistence
        recent_alerts = [
            a for a in self._fp_prevention["alert_history"]
            if a["type"] == alert_type
            and datetime.fromisoformat(a["timestamp"]) > (
                now - timedelta(seconds=self.ALERT_PERSISTENCE_WINDOW_SECONDS)
            )
        ]

        if len(recent_alerts) >= self.ALERT_PERSISTENCE_COUNT:
            logger.info(
                f"[{self._bot_id}] Alert '{alert_type}' persists "
                f"({len(recent_alerts)} occurrences in {self.ALERT_PERSISTENCE_WINDOW_SECONDS}s)"
            )
            return True

        logger.debug(
            f"[{self._bot_id}] Alert '{alert_type}' recorded but not persistent yet "
            f"({len(recent_alerts)}/{self.ALERT_PERSISTENCE_COUNT})"
        )
        return False

    def apply_consecutive_loss_decay(self) -> int:
        """
        Apply time decay to consecutive loss counter.

        Consecutive losses should gradually reset over time to prevent
        permanent strategy lockout from a single unlucky streak.

        Returns:
            Adjusted consecutive loss count
        """
        self._init_strategy_risk_tracking()
        self._init_false_positive_prevention()

        risk = self._strategy_risk
        fp = self._fp_prevention
        now = datetime.now(timezone.utc)

        current_losses = risk["consecutive_losses"]

        if current_losses == 0:
            fp["consecutive_loss_start_time"] = None
            return 0

        if fp["consecutive_loss_start_time"] is None:
            fp["consecutive_loss_start_time"] = now
            return current_losses

        # Calculate decay
        hours_elapsed = (now - fp["consecutive_loss_start_time"]).total_seconds() / 3600

        if hours_elapsed >= self.CONSECUTIVE_LOSS_DECAY_HOURS:
            # Full decay - reset to 0
            decay_amount = current_losses
            risk["consecutive_losses"] = 0
            fp["consecutive_loss_start_time"] = None
            logger.info(
                f"[{self._bot_id}] Consecutive losses fully decayed "
                f"(was {current_losses}, now 0 after {hours_elapsed:.1f}h)"
            )
            return 0
        elif hours_elapsed >= self.CONSECUTIVE_LOSS_DECAY_HOURS / 2:
            # Half decay - reduce by half (minimum 1)
            decay_amount = max(1, current_losses // 2)
            risk["consecutive_losses"] = current_losses - decay_amount
            logger.info(
                f"[{self._bot_id}] Consecutive losses partially decayed "
                f"(was {current_losses}, now {risk['consecutive_losses']} "
                f"after {hours_elapsed:.1f}h)"
            )
            return risk["consecutive_losses"]

        return current_losses

    def set_price_deviation_override(
        self,
        symbol: str,
        max_deviation_pct: Decimal,
        reason: str = "",
    ) -> None:
        """
        Set custom price deviation threshold for a specific symbol.

        Use this to prevent false positives for volatile assets.

        Args:
            symbol: Trading symbol
            max_deviation_pct: Maximum allowed deviation (e.g., 0.05 for 5%)
            reason: Reason for override
        """
        self._init_false_positive_prevention()

        self._fp_prevention["price_deviation_overrides"][symbol] = {
            "max_deviation_pct": max_deviation_pct,
            "reason": reason,
            "set_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"[{self._bot_id}] Price deviation override set for {symbol}: "
            f"{max_deviation_pct:.2%}, Reason: {reason}"
        )

    def get_price_deviation_threshold(self, symbol: str) -> Decimal:
        """
        Get price deviation threshold for a symbol.

        Returns symbol-specific override if set, otherwise default.

        Args:
            symbol: Trading symbol

        Returns:
            Maximum allowed price deviation percentage
        """
        self._init_false_positive_prevention()

        override = self._fp_prevention["price_deviation_overrides"].get(symbol)
        if override:
            return override["max_deviation_pct"]

        # Default threshold (can be overridden per-symbol)
        return Decimal("0.05")  # 5% default (more lenient than pre_trade_checker)

    def record_blocked_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        block_reason: str,
    ) -> str:
        """
        Record an order that was blocked for later review.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            block_reason: Reason for blocking

        Returns:
            Block ID for reference
        """
        self._init_false_positive_prevention()

        block_id = f"{self._bot_id}_{int(time.time()*1000)}"

        record = {
            "block_id": block_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": str(quantity),
            "price": str(price),
            "block_reason": block_reason,
            "reviewed": False,
            "override_applied": False,
        }

        self._fp_prevention["blocked_orders"].append(record)

        # Keep only last 100 blocked orders
        self._fp_prevention["blocked_orders"] = (
            self._fp_prevention["blocked_orders"][-100:]
        )

        logger.warning(
            f"[{self._bot_id}] Order blocked: {side} {quantity} {symbol} @ {price}, "
            f"Reason: {block_reason}, Block ID: {block_id}"
        )

        return block_id

    def get_blocked_orders(
        self,
        symbol: Optional[str] = None,
        unreviewed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get blocked orders for review."""
        self._init_false_positive_prevention()

        orders = self._fp_prevention["blocked_orders"]

        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]

        if unreviewed_only:
            orders = [o for o in orders if not o["reviewed"]]

        return orders

    def override_blocked_order(
        self,
        block_id: str,
        reason: str,
    ) -> bool:
        """
        Override a blocked order (mark as reviewed and allow similar in future).

        Args:
            block_id: Block ID from record_blocked_order
            reason: Reason for override

        Returns:
            True if override applied
        """
        self._init_false_positive_prevention()

        for order in self._fp_prevention["blocked_orders"]:
            if order["block_id"] == block_id:
                order["reviewed"] = True
                order["override_applied"] = True
                order["override_reason"] = reason
                order["override_at"] = datetime.now(timezone.utc).isoformat()

                # Record override for future reference
                self._fp_prevention["false_positive_overrides"].append({
                    "block_id": block_id,
                    "symbol": order["symbol"],
                    "block_reason": order["block_reason"],
                    "override_reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                logger.info(
                    f"[{self._bot_id}] Override applied for block {block_id}: {reason}"
                )
                return True

        return False

    def get_false_positive_status(self) -> Dict[str, Any]:
        """Get false positive prevention status."""
        self._init_false_positive_prevention()
        fp = self._fp_prevention

        return {
            "bot_id": self._bot_id,
            "recent_alerts": len(fp["alert_history"]),
            "blocked_orders_total": len(fp["blocked_orders"]),
            "blocked_orders_unreviewed": len([
                o for o in fp["blocked_orders"] if not o["reviewed"]
            ]),
            "overrides_applied": len(fp["false_positive_overrides"]),
            "price_deviation_overrides": {
                symbol: {
                    "max_pct": str(v["max_deviation_pct"]),
                    "reason": v["reason"],
                }
                for symbol, v in fp["price_deviation_overrides"].items()
            },
        }

    # =========================================================================
    # Risk Control Latency Prevention (風控延遲防護)
    # Synchronous risk gate to prevent orders slipping through
    # =========================================================================

    # Risk check cache TTL (milliseconds)
    RISK_CACHE_TTL_MS = 500  # Risk data valid for 500ms
    RISK_GATE_TIMEOUT_SECONDS = 5  # Max wait for risk gate

    def _init_risk_latency_prevention(self) -> None:
        """Initialize risk latency prevention."""
        if not hasattr(self, "_risk_latency"):
            self._risk_latency = {
                "order_lock": asyncio.Lock(),
                "last_risk_check_time": None,
                "last_risk_check_result": None,
                "orders_blocked_by_latency": 0,
                "orders_passed": 0,
                "avg_check_latency_ms": 0,
                "check_latency_samples": [],
            }

    async def acquire_risk_gate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        timeout_seconds: Optional[float] = None,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Acquire risk gate with atomic check-and-reserve.

        This ensures no orders can slip through between risk check and execution.
        Uses a lock to make the entire check-reserve-execute flow atomic.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price
            timeout_seconds: Max wait time for gate

        Returns:
            Tuple of (acquired, message, details)
        """
        self._init_risk_latency_prevention()
        latency = self._risk_latency
        timeout = timeout_seconds or self.RISK_GATE_TIMEOUT_SECONDS

        details = {
            "gate_acquired": False,
            "wait_time_ms": 0,
            "risk_check_fresh": False,
            "cached_result_used": False,
        }

        start_time = time.time()

        try:
            # Try to acquire the order lock with timeout
            acquired = await asyncio.wait_for(
                latency["order_lock"].acquire(),
                timeout=timeout,
            )

            if not acquired:
                details["wait_time_ms"] = (time.time() - start_time) * 1000
                latency["orders_blocked_by_latency"] += 1
                return False, "Failed to acquire risk gate (timeout)", details

            details["gate_acquired"] = True
            details["wait_time_ms"] = (time.time() - start_time) * 1000

            # Check if we have a recent valid risk check result
            now = datetime.now(timezone.utc)
            if latency["last_risk_check_time"]:
                age_ms = (now - latency["last_risk_check_time"]).total_seconds() * 1000
                if age_ms < self.RISK_CACHE_TTL_MS and latency["last_risk_check_result"]:
                    # Use cached result
                    cached = latency["last_risk_check_result"]
                    details["cached_result_used"] = True
                    details["cache_age_ms"] = age_ms

                    if not cached.get("is_allowed", False):
                        latency["order_lock"].release()
                        return False, f"Cached risk check failed: {cached.get('message')}", details

                    details["risk_check_fresh"] = True
                    latency["orders_passed"] += 1
                    # Don't release lock - caller will release after order execution
                    return True, "Risk gate acquired (cached)", details

            # Perform fresh risk check
            check_start = time.time()
            is_allowed, message, check_details = await self.safe_pre_trade_risk_check(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
            )
            check_latency_ms = (time.time() - check_start) * 1000

            # Update latency tracking
            latency["check_latency_samples"].append(check_latency_ms)
            if len(latency["check_latency_samples"]) > 100:
                latency["check_latency_samples"] = latency["check_latency_samples"][-100:]
            latency["avg_check_latency_ms"] = (
                sum(latency["check_latency_samples"]) / len(latency["check_latency_samples"])
            )

            # Cache the result
            latency["last_risk_check_time"] = now
            latency["last_risk_check_result"] = {
                "is_allowed": is_allowed,
                "message": message,
                "details": check_details,
            }

            details["risk_check_fresh"] = True
            details["check_latency_ms"] = check_latency_ms
            details.update(check_details)

            if not is_allowed:
                latency["order_lock"].release()
                latency["orders_blocked_by_latency"] += 1
                return False, message, details

            latency["orders_passed"] += 1
            # Don't release lock - caller will release after order execution
            return True, "Risk gate acquired", details

        except asyncio.TimeoutError:
            details["wait_time_ms"] = timeout * 1000
            latency["orders_blocked_by_latency"] += 1
            return False, f"Risk gate timeout after {timeout}s", details

        except Exception as e:
            if latency["order_lock"].locked():
                latency["order_lock"].release()
            logger.error(f"[{self._bot_id}] Risk gate error: {e}")
            return False, f"Risk gate error: {e}", details

    def release_risk_gate(self) -> None:
        """Release the risk gate after order execution."""
        self._init_risk_latency_prevention()
        if self._risk_latency["order_lock"].locked():
            self._risk_latency["order_lock"].release()

    def invalidate_risk_cache(self) -> None:
        """Invalidate the risk check cache (call after significant state changes)."""
        self._init_risk_latency_prevention()
        self._risk_latency["last_risk_check_time"] = None
        self._risk_latency["last_risk_check_result"] = None

    def get_risk_latency_stats(self) -> Dict[str, Any]:
        """Get risk latency statistics."""
        self._init_risk_latency_prevention()
        latency = self._risk_latency

        return {
            "bot_id": self._bot_id,
            "orders_passed": latency["orders_passed"],
            "orders_blocked_by_latency": latency["orders_blocked_by_latency"],
            "avg_check_latency_ms": round(latency["avg_check_latency_ms"], 2),
            "cache_ttl_ms": self.RISK_CACHE_TTL_MS,
            "gate_timeout_seconds": self.RISK_GATE_TIMEOUT_SECONDS,
        }

    # =========================================================================
    # Multi-Strategy Risk Coordination (多策略風控協調)
    # Global risk aggregation across all bots
    # =========================================================================

    # Global risk thresholds (aggregate across all strategies)
    GLOBAL_MAX_TOTAL_EXPOSURE_PCT = Decimal("0.80")  # 80% max total exposure
    GLOBAL_MAX_SINGLE_SYMBOL_PCT = Decimal("0.50")   # 50% max per symbol
    GLOBAL_MAX_DRAWDOWN_PCT = Decimal("0.15")        # 15% max drawdown
    GLOBAL_MAX_DAILY_LOSS_PCT = Decimal("0.05")      # 5% max daily loss

    # Class-level tracking for cross-bot coordination
    _global_risk_tracker: Optional[Dict[str, Any]] = None
    _global_risk_lock: Optional[asyncio.Lock] = None

    @classmethod
    def _init_global_risk_tracker(cls) -> None:
        """Initialize global risk tracker (class-level, shared across all bots)."""
        if cls._global_risk_tracker is None:
            cls._global_risk_tracker = {
                "total_capital": Decimal("0"),
                "bot_exposures": {},  # bot_id -> exposure amount
                "symbol_exposures": {},  # symbol -> total exposure
                "daily_pnl": Decimal("0"),
                "daily_start_time": datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                "peak_capital": Decimal("0"),
                "current_drawdown_pct": Decimal("0"),
                "last_update_time": None,
                "registered_bots": set(),
                "blocked_orders_global": 0,
            }
        if cls._global_risk_lock is None:
            cls._global_risk_lock = asyncio.Lock()

    @classmethod
    async def register_bot_for_global_risk(
        cls,
        bot_id: str,
        allocated_capital: Decimal,
    ) -> None:
        """
        Register a bot for global risk tracking.

        Args:
            bot_id: Bot identifier
            allocated_capital: Capital allocated to this bot
        """
        cls._init_global_risk_tracker()

        async with cls._global_risk_lock:
            cls._global_risk_tracker["registered_bots"].add(bot_id)
            cls._global_risk_tracker["bot_exposures"][bot_id] = Decimal("0")

            # Update total capital
            cls._global_risk_tracker["total_capital"] += allocated_capital
            if cls._global_risk_tracker["total_capital"] > cls._global_risk_tracker["peak_capital"]:
                cls._global_risk_tracker["peak_capital"] = cls._global_risk_tracker["total_capital"]

            logger.info(
                f"Bot {bot_id} registered for global risk tracking, "
                f"Total capital: {cls._global_risk_tracker['total_capital']}"
            )

    @classmethod
    async def update_bot_exposure(
        cls,
        bot_id: str,
        symbol: str,
        exposure_amount: Decimal,
    ) -> None:
        """
        Update a bot's exposure for global risk tracking.

        Args:
            bot_id: Bot identifier
            symbol: Trading symbol
            exposure_amount: Current exposure amount (notional value)
        """
        cls._init_global_risk_tracker()

        async with cls._global_risk_lock:
            # Update bot exposure
            cls._global_risk_tracker["bot_exposures"][bot_id] = exposure_amount

            # Track per-bot symbol exposure for proper aggregation
            if "bot_symbol_exposures" not in cls._global_risk_tracker:
                cls._global_risk_tracker["bot_symbol_exposures"] = {}
            cls._global_risk_tracker["bot_symbol_exposures"][(bot_id, symbol)] = exposure_amount

            # Recalculate symbol total from all bots
            symbol_total = sum(
                v for (bid, sym), v in cls._global_risk_tracker["bot_symbol_exposures"].items()
                if sym == symbol
            )
            cls._global_risk_tracker["symbol_exposures"][symbol] = symbol_total
            cls._global_risk_tracker["last_update_time"] = datetime.now(timezone.utc)

    @classmethod
    async def update_global_pnl(cls, pnl_change: Decimal) -> None:
        """Update global daily P&L."""
        cls._init_global_risk_tracker()

        async with cls._global_risk_lock:
            tracker = cls._global_risk_tracker

            # Reset daily P&L if new day
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if today_start > tracker["daily_start_time"]:
                tracker["daily_pnl"] = Decimal("0")
                tracker["daily_start_time"] = today_start

            tracker["daily_pnl"] += pnl_change

            # Update drawdown
            tracker["total_capital"] += pnl_change
            if tracker["total_capital"] > tracker["peak_capital"]:
                tracker["peak_capital"] = tracker["total_capital"]

            if tracker["peak_capital"] > 0:
                tracker["current_drawdown_pct"] = (
                    (tracker["peak_capital"] - tracker["total_capital"]) /
                    tracker["peak_capital"]
                )

    @classmethod
    async def check_global_risk_limits(
        cls,
        bot_id: str,
        symbol: str,
        additional_exposure: Decimal,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Check if an order would violate global risk limits.

        This is the SYNCHRONOUS global check that happens at trade time.

        Args:
            bot_id: Bot requesting the check
            symbol: Symbol to trade
            additional_exposure: Additional exposure from this order

        Returns:
            Tuple of (is_allowed, message, details)
        """
        cls._init_global_risk_tracker()

        details = {
            "global_check": True,
            "violations": [],
        }

        async with cls._global_risk_lock:
            tracker = cls._global_risk_tracker
            total_capital = tracker["total_capital"]

            if total_capital <= 0:
                return True, "No capital tracked (global check skipped)", details

            # 1. Check total exposure limit
            current_total_exposure = sum(tracker["bot_exposures"].values())
            new_total_exposure = current_total_exposure + additional_exposure
            total_exposure_pct = new_total_exposure / total_capital

            if total_exposure_pct > cls.GLOBAL_MAX_TOTAL_EXPOSURE_PCT:
                details["violations"].append({
                    "type": "TOTAL_EXPOSURE",
                    "current_pct": str(total_exposure_pct),
                    "limit_pct": str(cls.GLOBAL_MAX_TOTAL_EXPOSURE_PCT),
                })

            # 2. Check single symbol exposure limit
            current_symbol_exposure = tracker["symbol_exposures"].get(symbol, Decimal("0"))
            new_symbol_exposure = current_symbol_exposure + additional_exposure
            symbol_exposure_pct = new_symbol_exposure / total_capital

            if symbol_exposure_pct > cls.GLOBAL_MAX_SINGLE_SYMBOL_PCT:
                details["violations"].append({
                    "type": "SYMBOL_EXPOSURE",
                    "symbol": symbol,
                    "current_pct": str(symbol_exposure_pct),
                    "limit_pct": str(cls.GLOBAL_MAX_SINGLE_SYMBOL_PCT),
                })

            # 3. Check drawdown limit
            if tracker["current_drawdown_pct"] > cls.GLOBAL_MAX_DRAWDOWN_PCT:
                details["violations"].append({
                    "type": "DRAWDOWN",
                    "current_pct": str(tracker["current_drawdown_pct"]),
                    "limit_pct": str(cls.GLOBAL_MAX_DRAWDOWN_PCT),
                })

            # 4. Check daily loss limit
            if total_capital > 0:
                daily_loss_pct = abs(tracker["daily_pnl"] / total_capital) if tracker["daily_pnl"] < 0 else Decimal("0")
                if daily_loss_pct > cls.GLOBAL_MAX_DAILY_LOSS_PCT:
                    details["violations"].append({
                        "type": "DAILY_LOSS",
                        "current_pct": str(daily_loss_pct),
                        "limit_pct": str(cls.GLOBAL_MAX_DAILY_LOSS_PCT),
                    })

            # Return result
            if details["violations"]:
                tracker["blocked_orders_global"] += 1
                violation_types = [v["type"] for v in details["violations"]]
                return False, f"Global risk limits exceeded: {', '.join(violation_types)}", details

            details["total_exposure_pct"] = str(total_exposure_pct)
            details["symbol_exposure_pct"] = str(symbol_exposure_pct)
            details["drawdown_pct"] = str(tracker["current_drawdown_pct"])

            return True, "Global risk check passed", details

    @classmethod
    def get_global_risk_status(cls) -> Dict[str, Any]:
        """Get global risk status across all bots."""
        cls._init_global_risk_tracker()
        tracker = cls._global_risk_tracker

        total_exposure = sum(tracker["bot_exposures"].values())
        total_capital = tracker["total_capital"]

        return {
            "total_capital": str(total_capital),
            "peak_capital": str(tracker["peak_capital"]),
            "total_exposure": str(total_exposure),
            "total_exposure_pct": str(total_exposure / total_capital) if total_capital > 0 else "0",
            "current_drawdown_pct": str(tracker["current_drawdown_pct"]),
            "daily_pnl": str(tracker["daily_pnl"]),
            "registered_bots": list(tracker["registered_bots"]),
            "bot_exposures": {k: str(v) for k, v in tracker["bot_exposures"].items()},
            "symbol_exposures": {k: str(v) for k, v in tracker["symbol_exposures"].items()},
            "blocked_orders_global": tracker["blocked_orders_global"],
            "last_update_time": tracker["last_update_time"].isoformat() if tracker["last_update_time"] else None,
        }

    # =========================================================================
    # Risk Parameter Validation (風控參數驗證)
    # Validate risk parameters to prevent misconfiguration
    # =========================================================================

    @staticmethod
    def validate_risk_parameters(params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate risk parameters for correctness.

        Checks for:
        - Range validity (min > 0, max > min)
        - Percentage constraints (0 < pct <= 1 for ratios)
        - Logical consistency (warning < danger thresholds)
        - Unit correctness

        Args:
            params: Dictionary of risk parameters to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Define validation rules
        percentage_params = [
            "position_size_pct", "max_position_pct", "stop_loss_pct",
            "take_profit_pct", "daily_loss_limit_pct", "max_drawdown_pct",
            "warning_loss_pct", "danger_loss_pct", "fee_rate",
            "slippage_tolerance_pct", "price_deviation_pct",
        ]

        positive_params = [
            "leverage", "grid_count", "atr_period", "rsi_period",
            "bb_period", "cooldown_seconds", "timeout_seconds",
            "max_retries", "order_timeout_ms",
        ]

        min_max_pairs = [
            ("min_order_value", "max_order_value"),
            ("min_quantity", "max_quantity"),
            ("lower_price", "upper_price"),
        ]

        threshold_pairs = [
            ("warning_loss_pct", "danger_loss_pct"),  # warning < danger
            ("warning_drawdown_pct", "danger_drawdown_pct"),
        ]

        # Validate percentage parameters (0 < value <= 1 or 0 < value <= 100)
        for param in percentage_params:
            if param in params:
                value = params[param]
                try:
                    decimal_value = Decimal(str(value))

                    # Check if it's in 0-1 range or 0-100 range
                    if decimal_value <= 0:
                        errors.append(f"{param} must be > 0 (got {value})")
                    elif decimal_value > 100:
                        errors.append(f"{param} appears invalid: {value} (expected percentage)")
                    elif decimal_value > 1 and "pct" in param.lower():
                        # Likely using percentage as whole number (e.g., 50 instead of 0.50)
                        logger.warning(
                            f"Parameter {param}={value} may be using wrong unit "
                            f"(expected 0-1, got {value})"
                        )
                except (ValueError, TypeError, decimal.InvalidOperation):
                    errors.append(f"{param} is not a valid number: {value}")

        # Validate positive parameters
        for param in positive_params:
            if param in params:
                value = params[param]
                try:
                    num_value = float(value) if not isinstance(value, (int, float)) else value
                    if num_value <= 0:
                        errors.append(f"{param} must be > 0 (got {value})")
                except (ValueError, TypeError):
                    errors.append(f"{param} is not a valid number: {value}")

        # Validate min/max pairs
        for min_param, max_param in min_max_pairs:
            if min_param in params and max_param in params:
                try:
                    min_val = Decimal(str(params[min_param]))
                    max_val = Decimal(str(params[max_param]))
                    if min_val >= max_val:
                        errors.append(
                            f"{min_param} ({min_val}) must be < {max_param} ({max_val})"
                        )
                except (ValueError, TypeError, decimal.InvalidOperation):
                    pass  # Already caught by other validators

        # Validate threshold pairs (warning < danger)
        for warning_param, danger_param in threshold_pairs:
            if warning_param in params and danger_param in params:
                try:
                    warning_val = Decimal(str(params[warning_param]))
                    danger_val = Decimal(str(params[danger_param]))
                    if warning_val >= danger_val:
                        errors.append(
                            f"{warning_param} ({warning_val}) must be < {danger_param} ({danger_val})"
                        )
                except (ValueError, TypeError, decimal.InvalidOperation):
                    pass

        # Validate leverage
        if "leverage" in params:
            try:
                leverage = int(params["leverage"])
                if leverage < 1 or leverage > 125:
                    errors.append(f"leverage must be 1-125 (got {leverage})")
            except (ValueError, TypeError):
                errors.append(f"leverage is not a valid integer: {params['leverage']}")

        # Validate grid count
        if "grid_count" in params:
            try:
                grid_count = int(params["grid_count"])
                if grid_count < 2 or grid_count > 200:
                    errors.append(f"grid_count must be 2-200 (got {grid_count})")
            except (ValueError, TypeError):
                errors.append(f"grid_count is not a valid integer: {params['grid_count']}")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def normalize_percentage_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize percentage parameters to decimal form (0-1).

        If a parameter looks like it's in percentage form (e.g., 50 instead of 0.50),
        convert it to decimal form.

        Args:
            params: Dictionary of parameters

        Returns:
            Normalized parameters
        """
        percentage_params = [
            "position_size_pct", "max_position_pct", "stop_loss_pct",
            "take_profit_pct", "daily_loss_limit_pct", "max_drawdown_pct",
            "fee_rate",
        ]

        normalized = params.copy()

        for param in percentage_params:
            if param in normalized:
                try:
                    value = Decimal(str(normalized[param]))
                    # If value > 1, assume it's in percentage form (e.g., 50 = 50%)
                    if value > 1 and value <= 100:
                        normalized[param] = value / 100
                        logger.info(
                            f"Normalized {param}: {value}% -> {normalized[param]}"
                        )
                except (ValueError, TypeError, decimal.InvalidOperation):
                    pass

        return normalized

    def validate_bot_config(self) -> tuple[bool, List[str]]:
        """
        Validate this bot's configuration.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not hasattr(self, "_config"):
            return False, ["No config found"]

        # Convert config to dict for validation
        config_dict = {}
        for attr in dir(self._config):
            if not attr.startswith("_"):
                try:
                    value = getattr(self._config, attr)
                    if not callable(value):
                        config_dict[attr] = value
                except Exception as e:
                    logger.debug(f"Failed to get config attribute {attr}: {e}")

        return self.validate_risk_parameters(config_dict)

    async def pre_trade_with_global_check(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Complete pre-trade validation including global risk check.

        This is the RECOMMENDED method to call before placing any order.
        It combines:
        1. Risk gate acquisition (atomic check-and-reserve)
        2. Global multi-strategy risk check
        3. Strategy-level risk check

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Order price

        Returns:
            Tuple of (is_allowed, message, details)
        """
        details = {
            "gate_check": False,
            "global_check": False,
            "strategy_check": False,
        }

        # 1. Acquire risk gate (atomic lock + strategy check)
        gate_acquired, gate_msg, gate_details = await self.acquire_risk_gate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )

        details["gate_check"] = gate_acquired
        details["gate_details"] = gate_details

        if not gate_acquired:
            return False, f"Risk gate: {gate_msg}", details

        try:
            # 2. Check global risk limits
            exposure = quantity * price
            global_ok, global_msg, global_details = await self.check_global_risk_limits(
                bot_id=self._bot_id,
                symbol=symbol,
                additional_exposure=exposure,
            )

            details["global_check"] = global_ok
            details["global_details"] = global_details

            if not global_ok:
                self.release_risk_gate()
                return False, f"Global risk: {global_msg}", details

            # All checks passed
            details["strategy_check"] = True
            return True, "All pre-trade checks passed", details

        except Exception as e:
            self.release_risk_gate()
            logger.error(f"[{self._bot_id}] Pre-trade global check error: {e}")
            return False, f"Pre-trade error: {e}", details

    # =========================================================================
    # Stop Loss Protection System (止損保護系統)
    # =========================================================================
    #
    # Three-layer protection:
    # 1. Exchange Stop Loss - Primary (交易所止損單)
    # 2. Software Backup Stop Loss - Secondary (軟體備份止損)
    # 3. Emergency Hard Stop - Final (緊急硬止損)
    #
    # Addresses:
    # - 止損未觸發: Monitor stop loss order status, detect failures
    # - 止損滑點過大: Use limit orders with buffer, track slippage
    # - 跳空越過止損: Time-based emergency stop, hard loss cap
    # =========================================================================

    # Stop Loss Configuration Constants
    STOP_LOSS_MONITOR_INTERVAL_SECONDS = 5  # Check stop loss status every 5s
    STOP_LOSS_MAX_FAILURES = 3  # Max consecutive failures before emergency action
    STOP_LOSS_LIMIT_BUFFER_PCT = Decimal("0.005")  # 0.5% buffer for limit stop
    STOP_LOSS_MAX_SLIPPAGE_PCT = Decimal("0.02")  # 2% max acceptable slippage
    EMERGENCY_LOSS_PCT = Decimal("0.10")  # 10% hard stop (emergency)
    HARD_LOSS_CAP_PCT = Decimal("0.15")  # 15% absolute max loss cap
    GAP_DETECTION_THRESHOLD_PCT = Decimal("0.03")  # 3% gap detection threshold
    TIERED_EXIT_LEVELS = 3  # Number of exit levels for tiered stop loss

    def _init_stop_loss_protection(self) -> None:
        """Initialize stop loss protection state."""
        if not hasattr(self, "_sl_protection_state"):
            self._sl_protection_state: Dict[str, Any] = {
                "last_check_time": None,
                "consecutive_failures": 0,
                "last_known_sl_status": None,
                "sl_order_placed_time": None,
                "emergency_mode": False,
                "slippage_history": [],
                "gap_events": [],
                "tiered_exits_remaining": self.TIERED_EXIT_LEVELS,
                "backup_sl_active": False,
                "last_price": None,
                "last_price_time": None,
            }
        if not hasattr(self, "_sl_monitor_lock"):
            self._sl_monitor_lock = asyncio.Lock()

    # =========================================================================
    # Layer 1: Stop Loss Order Monitoring (止損單監控)
    # =========================================================================

    async def monitor_stop_loss_order(
        self,
        symbol: str,
        stop_loss_order_id: str,
        expected_quantity: Decimal,
    ) -> Dict[str, Any]:
        """
        Monitor stop loss order status and detect failures.

        This should be called periodically from background monitor.

        Args:
            symbol: Trading symbol
            stop_loss_order_id: Stop loss order ID to monitor
            expected_quantity: Expected quantity of stop loss order

        Returns:
            Dict with status and any issues detected
        """
        self._init_stop_loss_protection()

        result = {
            "is_active": False,
            "is_triggered": False,
            "is_failed": False,
            "failure_reason": None,
            "needs_replacement": False,
            "needs_emergency_action": False,
        }

        async with self._sl_monitor_lock:
            try:
                # Query order status from exchange
                order_status = await self._query_stop_loss_status(
                    symbol=symbol,
                    order_id=stop_loss_order_id,
                )

                if order_status is None:
                    # Order not found - may have been cancelled or filled
                    result["is_failed"] = True
                    result["failure_reason"] = "ORDER_NOT_FOUND"
                    self._sl_protection_state["consecutive_failures"] += 1

                elif order_status.get("status") == "FILLED":
                    result["is_triggered"] = True
                    self._sl_protection_state["consecutive_failures"] = 0
                    self._sl_protection_state["last_known_sl_status"] = "FILLED"

                elif order_status.get("status") in ["NEW", "PENDING"]:
                    result["is_active"] = True
                    self._sl_protection_state["consecutive_failures"] = 0
                    self._sl_protection_state["last_known_sl_status"] = "ACTIVE"

                elif order_status.get("status") == "PARTIALLY_FILLED":
                    result["is_active"] = True
                    filled_qty = Decimal(str(order_status.get("filled_qty", "0")))
                    logger.warning(
                        f"[{self._bot_id}] Stop loss order {stop_loss_order_id} "
                        f"partially filled: {filled_qty}/{expected_quantity} — "
                        f"remaining position may be unprotected"
                    )
                    self._sl_protection_state["consecutive_failures"] = 0
                    self._sl_protection_state["last_known_sl_status"] = "PARTIALLY_FILLED"

                elif order_status.get("status") in ["CANCELED", "CANCELLED", "REJECTED", "EXPIRED"]:
                    result["is_failed"] = True
                    result["failure_reason"] = order_status.get("status")
                    result["needs_replacement"] = True
                    self._sl_protection_state["consecutive_failures"] += 1

                    logger.warning(
                        f"[{self._bot_id}] Stop loss order {stop_loss_order_id} "
                        f"failed: {order_status.get('status')}"
                    )

                # Check for excessive failures
                if self._sl_protection_state["consecutive_failures"] >= self.STOP_LOSS_MAX_FAILURES:
                    result["needs_emergency_action"] = True
                    self._sl_protection_state["emergency_mode"] = True
                    logger.error(
                        f"[{self._bot_id}] Stop loss protection: "
                        f"{self._sl_protection_state['consecutive_failures']} consecutive failures - "
                        f"EMERGENCY MODE ACTIVATED"
                    )

                self._sl_protection_state["last_check_time"] = datetime.now(timezone.utc)
                return result

            except Exception as e:
                logger.error(f"[{self._bot_id}] Stop loss monitor error: {e}")
                self._sl_protection_state["consecutive_failures"] += 1
                result["is_failed"] = True
                result["failure_reason"] = f"MONITOR_ERROR: {e}"
                return result

    async def _query_stop_loss_status(
        self,
        symbol: str,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query stop loss order status from exchange.

        Args:
            symbol: Trading symbol
            order_id: Order ID to query

        Returns:
            Order status dict or None if not found
        """
        try:
            # Try to get order from exchange
            order = await self._exchange.futures.get_order(
                symbol=symbol,
                order_id=order_id,
            )

            if order:
                return {
                    "status": getattr(order, "status", "UNKNOWN"),
                    "filled_qty": getattr(order, "filled_qty", Decimal("0")),
                    "avg_price": getattr(order, "avg_price", None),
                }
            return None

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "unknown" in error_str:
                return None
            raise

    async def replace_failed_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss_pct: Decimal,
    ) -> Dict[str, Any]:
        """
        Replace a failed stop loss order with a new one.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            quantity: Position quantity
            entry_price: Position entry price
            stop_loss_pct: Stop loss percentage

        Returns:
            Dict with new order details or failure info
        """
        result = {
            "success": False,
            "new_order_id": None,
            "error": None,
        }

        try:
            # Calculate stop price
            if side.upper() == "LONG":
                stop_price = entry_price * (Decimal("1") - stop_loss_pct)
                close_side = "SELL"
            else:
                stop_price = entry_price * (Decimal("1") + stop_loss_pct)
                close_side = "BUY"

            # Round to appropriate precision using symbol tick_size
            symbol_info = await self._get_symbol_info(symbol)
            stop_price = self._normalize_price(stop_price, symbol_info)

            # Place new stop loss order
            new_order = await self._exchange.futures_create_order(
                symbol=symbol,
                side=close_side,
                order_type="STOP_MARKET",
                quantity=quantity,
                stop_price=stop_price,
                reduce_only=True,
                position_side=side.upper(),
                bot_id=self._bot_id,
            )

            if new_order:
                result["success"] = True
                result["new_order_id"] = str(getattr(new_order, "order_id", ""))
                result["stop_price"] = stop_price

                self._sl_protection_state["consecutive_failures"] = 0
                self._sl_protection_state["sl_order_placed_time"] = datetime.now(timezone.utc)

                logger.info(
                    f"[{self._bot_id}] Replaced failed stop loss: "
                    f"new order {result['new_order_id']} @ {stop_price}"
                )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{self._bot_id}] Failed to replace stop loss: {e}")

        return result

    # =========================================================================
    # Layer 2: Software Backup Stop Loss (軟體備份止損)
    # =========================================================================

    async def check_backup_stop_loss(
        self,
        symbol: str,
        current_price: Decimal,
        entry_price: Decimal,
        position_side: str,
        stop_loss_pct: Decimal,
        quantity: Decimal,
    ) -> Dict[str, Any]:
        """
        Software-based backup stop loss check.

        This runs independently of exchange stop loss orders and provides
        a backup in case the exchange order fails.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Position entry price
            position_side: Position side (LONG/SHORT)
            stop_loss_pct: Stop loss percentage
            quantity: Position quantity

        Returns:
            Dict with action needed and details
        """
        self._init_stop_loss_protection()

        result = {
            "should_close": False,
            "reason": None,
            "urgency": "normal",  # normal, high, critical
            "close_quantity": Decimal("0"),
            "slippage_warning": False,
        }

        # Calculate P&L percentage
        if position_side.upper() == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Check if loss exceeds stop loss threshold
        if pnl_pct < -stop_loss_pct:
            result["should_close"] = True
            result["reason"] = "BACKUP_STOP_LOSS"
            result["close_quantity"] = quantity
            result["pnl_pct"] = pnl_pct

            # Determine urgency based on loss magnitude
            if pnl_pct < -self.EMERGENCY_LOSS_PCT:
                result["urgency"] = "critical"
                logger.error(
                    f"[{self._bot_id}] CRITICAL: Backup stop loss triggered at "
                    f"{pnl_pct*100:.2f}% loss (critical threshold)"
                )
            elif pnl_pct < -(stop_loss_pct * Decimal("1.5")):
                result["urgency"] = "high"
                logger.warning(
                    f"[{self._bot_id}] HIGH: Backup stop loss triggered at "
                    f"{pnl_pct*100:.2f}% loss (1.5x stop loss)"
                )
            else:
                logger.warning(
                    f"[{self._bot_id}] Backup stop loss triggered at "
                    f"{pnl_pct*100:.2f}% loss"
                )

            # Mark backup stop loss as active
            self._sl_protection_state["backup_sl_active"] = True

        return result

    # =========================================================================
    # Layer 3: Emergency Hard Stop (緊急硬止損)
    # =========================================================================

    async def check_emergency_stop(
        self,
        symbol: str,
        current_price: Decimal,
        entry_price: Decimal,
        position_side: str,
        quantity: Decimal,
        leverage: int = 1,
    ) -> Dict[str, Any]:
        """
        Emergency hard stop check - absolute loss cap regardless of other stops.

        This is the final safety net that prevents catastrophic losses.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Position entry price
            position_side: Position side (LONG/SHORT)
            quantity: Position quantity
            leverage: Position leverage

        Returns:
            Dict with emergency action needed
        """
        self._init_stop_loss_protection()

        result = {
            "emergency_close": False,
            "reason": None,
            "loss_pct": Decimal("0"),
            "estimated_loss": Decimal("0"),
        }

        # Calculate P&L percentage (with leverage effect)
        if position_side.upper() == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        result["loss_pct"] = pnl_pct
        result["estimated_loss"] = abs(pnl_pct) * quantity * entry_price

        # Check hard loss cap
        if pnl_pct < -self.HARD_LOSS_CAP_PCT:
            result["emergency_close"] = True
            result["reason"] = "HARD_LOSS_CAP"

            logger.critical(
                f"[{self._bot_id}] EMERGENCY HARD STOP: "
                f"Loss {pnl_pct*100:.2f}% exceeds hard cap {self.HARD_LOSS_CAP_PCT*100:.1f}% - "
                f"FORCING IMMEDIATE CLOSE"
            )

            # Send critical notification if available
            if hasattr(self, "_notifier") and self._notifier:
                try:
                    await self._notifier.send_alert(
                        level="CRITICAL",
                        title=f"EMERGENCY STOP - {symbol}",
                        message=(
                            f"Hard loss cap triggered!\n"
                            f"Loss: {pnl_pct*100:.2f}%\n"
                            f"Position closing immediately"
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to send hard loss cap notification: {e}")

        # Check if in emergency mode due to stop loss failures
        elif self._sl_protection_state.get("emergency_mode"):
            result["emergency_close"] = True
            result["reason"] = "STOP_LOSS_FAILURES"

            logger.critical(
                f"[{self._bot_id}] EMERGENCY CLOSE due to stop loss failures - "
                f"Current P&L: {pnl_pct*100:.2f}%"
            )

        return result

    async def execute_emergency_close(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Execute emergency position close with maximum priority.

        Args:
            symbol: Trading symbol
            side: Position side to close
            quantity: Quantity to close
            reason: Reason for emergency close

        Returns:
            Dict with execution result
        """
        result = {
            "success": False,
            "filled_quantity": Decimal("0"),
            "avg_price": None,
            "error": None,
        }

        try:
            # Determine close side
            close_side = "SELL" if side.upper() == "LONG" else "BUY"

            logger.critical(
                f"[{self._bot_id}] Executing EMERGENCY CLOSE: "
                f"{close_side} {quantity} {symbol}, reason: {reason}"
            )

            # Place market order with reduce_only
            order = await self._exchange.futures_create_order(
                symbol=symbol,
                side=close_side,
                order_type="MARKET",
                quantity=quantity,
                reduce_only=True,
                position_side=side.upper(),
                bot_id=self._bot_id,
            )

            if order:
                result["success"] = True
                result["filled_quantity"] = getattr(order, "filled_qty", quantity)
                result["avg_price"] = getattr(order, "avg_price", None)

                logger.critical(
                    f"[{self._bot_id}] Emergency close executed: "
                    f"{result['filled_quantity']} @ {result['avg_price']}"
                )

                # Reset emergency state
                self._sl_protection_state["emergency_mode"] = False
                self._sl_protection_state["consecutive_failures"] = 0

        except Exception as e:
            result["error"] = str(e)
            logger.critical(
                f"[{self._bot_id}] EMERGENCY CLOSE FAILED: {e} - "
                f"Manual intervention required!"
            )

            # Send critical alert
            if hasattr(self, "_notifier") and self._notifier:
                try:
                    await self._notifier.send_alert(
                        level="CRITICAL",
                        title=f"EMERGENCY CLOSE FAILED - {symbol}",
                        message=(
                            f"Emergency close order FAILED!\n"
                            f"Error: {e}\n"
                            f"MANUAL INTERVENTION REQUIRED!"
                        ),
                    )
                except Exception:
                    pass

            # Trigger circuit breaker - emergency close is last line of defense
            try:
                await self.trigger_circuit_breaker_safe(
                    reason=f"EMERGENCY_CLOSE_FAILED: {e}",
                )
            except Exception:
                # If circuit breaker also fails, force stop
                self._running = False
                self._state = BotState.ERROR
                self._error_message = f"Emergency close AND circuit breaker failed: {e}"

        return result

    # =========================================================================
    # Gap Detection and Protection (跳空保護)
    # =========================================================================

    def detect_price_gap(
        self,
        current_price: Decimal,
        previous_price: Decimal,
    ) -> Dict[str, Any]:
        """
        Detect price gaps that could bypass stop loss.

        Args:
            current_price: Current market price
            previous_price: Previous price point

        Returns:
            Dict with gap detection results
        """
        self._init_stop_loss_protection()

        result = {
            "gap_detected": False,
            "gap_pct": Decimal("0"),
            "gap_direction": None,  # "up" or "down"
            "is_significant": False,
        }

        if previous_price <= 0:
            return result

        # Calculate gap percentage
        gap_pct = (current_price - previous_price) / previous_price
        result["gap_pct"] = gap_pct

        if gap_pct > self.GAP_DETECTION_THRESHOLD_PCT:
            result["gap_detected"] = True
            result["gap_direction"] = "up"
            result["is_significant"] = True
        elif gap_pct < -self.GAP_DETECTION_THRESHOLD_PCT:
            result["gap_detected"] = True
            result["gap_direction"] = "down"
            result["is_significant"] = True

        if result["gap_detected"]:
            # Record gap event
            self._sl_protection_state["gap_events"].append({
                "time": datetime.now(timezone.utc),
                "previous_price": previous_price,
                "current_price": current_price,
                "gap_pct": gap_pct,
            })

            # Keep only last 10 gap events
            self._sl_protection_state["gap_events"] = \
                self._sl_protection_state["gap_events"][-10:]

            logger.warning(
                f"[{self._bot_id}] Price gap detected: {gap_pct*100:.2f}% "
                f"({previous_price} -> {current_price})"
            )

        # Update last price
        self._sl_protection_state["last_price"] = current_price
        self._sl_protection_state["last_price_time"] = datetime.now(timezone.utc)

        return result

    async def handle_gap_through_stop_loss(
        self,
        symbol: str,
        current_price: Decimal,
        stop_price: Decimal,
        entry_price: Decimal,
        position_side: str,
        quantity: Decimal,
    ) -> Dict[str, Any]:
        """
        Handle case where price gapped through stop loss level.

        Args:
            symbol: Trading symbol
            current_price: Current price (after gap)
            stop_price: Original stop loss price
            entry_price: Position entry price
            position_side: Position side
            quantity: Position quantity

        Returns:
            Dict with recommended action
        """
        result = {
            "gapped_through": False,
            "estimated_slippage_pct": Decimal("0"),
            "recommended_action": None,
            "tiered_exit": False,
        }

        # Check if price gapped through stop
        if position_side.upper() == "LONG":
            if current_price < stop_price:
                result["gapped_through"] = True
                result["estimated_slippage_pct"] = (stop_price - current_price) / stop_price
        else:  # SHORT
            if current_price > stop_price:
                result["gapped_through"] = True
                result["estimated_slippage_pct"] = (current_price - stop_price) / stop_price

        if result["gapped_through"]:
            logger.warning(
                f"[{self._bot_id}] Price gapped through stop loss: "
                f"stop={stop_price}, current={current_price}, "
                f"slippage={result['estimated_slippage_pct']*100:.2f}%"
            )

            # Recommend immediate close
            result["recommended_action"] = "IMMEDIATE_CLOSE"

            # Record slippage
            self._sl_protection_state["slippage_history"].append({
                "time": datetime.now(timezone.utc),
                "expected_price": stop_price,
                "actual_price": current_price,
                "slippage_pct": result["estimated_slippage_pct"],
            })

            # Keep only last 20 slippage records
            self._sl_protection_state["slippage_history"] = \
                self._sl_protection_state["slippage_history"][-20:]

        return result

    # =========================================================================
    # Slippage Protection (滑點保護)
    # =========================================================================

    async def place_stop_loss_with_limit(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
        limit_buffer_pct: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Place stop loss with limit order instead of market order.

        This reduces slippage by setting a maximum execution price.

        Args:
            symbol: Trading symbol
            side: Position side (determines close side)
            quantity: Order quantity
            stop_price: Stop trigger price
            limit_buffer_pct: Buffer percentage for limit price

        Returns:
            Dict with order result
        """
        if limit_buffer_pct is None:
            limit_buffer_pct = self.STOP_LOSS_LIMIT_BUFFER_PCT

        result = {
            "success": False,
            "order_id": None,
            "stop_price": stop_price,
            "limit_price": None,
            "error": None,
        }

        try:
            # Calculate limit price with buffer
            if side.upper() == "LONG":
                close_side = "SELL"
                # For long position stop, limit should be below stop price
                limit_price = stop_price * (Decimal("1") - limit_buffer_pct)
            else:
                close_side = "BUY"
                # For short position stop, limit should be above stop price
                limit_price = stop_price * (Decimal("1") + limit_buffer_pct)

            symbol_info = await self._get_symbol_info(symbol)
            limit_price = self._normalize_price(limit_price, symbol_info)
            result["limit_price"] = limit_price

            # Place STOP_LIMIT order
            order = await self._exchange.futures_create_order(
                symbol=symbol,
                side=close_side,
                order_type="STOP_LIMIT",
                quantity=quantity,
                price=limit_price,
                stop_price=stop_price,
                reduce_only=True,
                position_side=side.upper(),
                bot_id=self._bot_id,
            )

            if order:
                result["success"] = True
                result["order_id"] = str(getattr(order, "order_id", ""))

                logger.info(
                    f"[{self._bot_id}] Stop-limit order placed: "
                    f"stop={stop_price}, limit={limit_price}"
                )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{self._bot_id}] Failed to place stop-limit: {e}")

            # Fallback to market stop if limit fails
            logger.warning(
                f"[{self._bot_id}] Falling back to STOP_MARKET due to limit failure"
            )

        return result

    # =========================================================================
    # Tiered Exit Strategy (分層出場策略)
    # =========================================================================

    async def execute_tiered_stop_loss(
        self,
        symbol: str,
        side: str,
        total_quantity: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
        base_stop_pct: Decimal,
    ) -> Dict[str, Any]:
        """
        Execute tiered stop loss - close position in multiple tranches.

        This reduces market impact and can achieve better average exit price.

        Args:
            symbol: Trading symbol
            side: Position side
            total_quantity: Total quantity to close
            entry_price: Original entry price
            current_price: Current market price
            base_stop_pct: Base stop loss percentage

        Returns:
            Dict with tiered exit results
        """
        self._init_stop_loss_protection()

        result = {
            "total_closed": Decimal("0"),
            "tranches": [],
            "avg_exit_price": Decimal("0"),
            "remaining_quantity": total_quantity,
        }

        # Calculate tranche sizes (33%, 33%, 34%)
        num_tranches = min(
            self._sl_protection_state["tiered_exits_remaining"],
            self.TIERED_EXIT_LEVELS
        )

        if num_tranches <= 0:
            logger.warning(f"[{self._bot_id}] No tiered exits remaining")
            return result

        tranche_size = total_quantity / Decimal(num_tranches)
        close_side = "SELL" if side.upper() == "LONG" else "BUY"

        total_value = Decimal("0")

        for i in range(num_tranches):
            # Last tranche gets remaining quantity to avoid rounding issues
            if i == num_tranches - 1:
                qty = result["remaining_quantity"]
            else:
                qty = tranche_size.quantize(Decimal("0.001"))

            if qty <= 0:
                break

            try:
                order = await self._exchange.futures_create_order(
                    symbol=symbol,
                    side=close_side,
                    order_type="MARKET",
                    quantity=qty,
                    reduce_only=True,
                    position_side=side.upper(),
                    bot_id=self._bot_id,
                )

                if order:
                    filled_qty = getattr(order, "filled_qty", qty)
                    fill_price = getattr(order, "avg_price", current_price)

                    result["tranches"].append({
                        "tranche": i + 1,
                        "quantity": filled_qty,
                        "price": fill_price,
                    })

                    result["total_closed"] += filled_qty
                    result["remaining_quantity"] -= filled_qty
                    total_value += filled_qty * fill_price

                    self._sl_protection_state["tiered_exits_remaining"] -= 1

                    logger.info(
                        f"[{self._bot_id}] Tiered exit {i+1}/{num_tranches}: "
                        f"{filled_qty} @ {fill_price}"
                    )

                    # Small delay between tranches
                    if i < num_tranches - 1:
                        await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"[{self._bot_id}] Tiered exit tranche {i+1} failed: {e}")

        # Calculate average exit price
        if result["total_closed"] > 0:
            result["avg_exit_price"] = total_value / result["total_closed"]

        return result

    # =========================================================================
    # Comprehensive Stop Loss Check (綜合止損檢查)
    # =========================================================================

    async def comprehensive_stop_loss_check(
        self,
        symbol: str,
        current_price: Decimal,
        entry_price: Decimal,
        position_side: str,
        quantity: Decimal,
        stop_loss_pct: Decimal,
        stop_loss_order_id: Optional[str] = None,
        leverage: int = 1,
    ) -> Dict[str, Any]:
        """
        Comprehensive stop loss check combining all protection layers.

        This is the RECOMMENDED method to call from background monitors.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            entry_price: Position entry price
            position_side: Position side (LONG/SHORT)
            quantity: Position quantity
            stop_loss_pct: Stop loss percentage
            stop_loss_order_id: Exchange stop loss order ID (if any)
            leverage: Position leverage

        Returns:
            Dict with comprehensive check results and recommended actions
        """
        self._init_stop_loss_protection()

        result = {
            "action_needed": False,
            "action_type": None,  # None, REPLACE_SL, BACKUP_CLOSE, EMERGENCY_CLOSE
            "urgency": "normal",
            "details": {},
        }

        # Layer 1: Check exchange stop loss order status
        if stop_loss_order_id:
            sl_status = await self.monitor_stop_loss_order(
                symbol=symbol,
                stop_loss_order_id=stop_loss_order_id,
                expected_quantity=quantity,
            )

            result["details"]["exchange_sl"] = sl_status

            if sl_status["needs_emergency_action"]:
                result["action_needed"] = True
                result["action_type"] = "EMERGENCY_CLOSE"
                result["urgency"] = "critical"
                return result

            if sl_status["needs_replacement"]:
                result["action_needed"] = True
                result["action_type"] = "REPLACE_SL"
                result["urgency"] = "high"
                return result

        # Layer 2: Check backup software stop loss
        backup_check = await self.check_backup_stop_loss(
            symbol=symbol,
            current_price=current_price,
            entry_price=entry_price,
            position_side=position_side,
            stop_loss_pct=stop_loss_pct,
            quantity=quantity,
        )

        result["details"]["backup_sl"] = backup_check

        if backup_check["should_close"]:
            result["action_needed"] = True
            result["action_type"] = "BACKUP_CLOSE"
            result["urgency"] = backup_check["urgency"]

            # If exchange stop loss should have triggered but didn't, escalate
            if stop_loss_order_id and backup_check["urgency"] == "critical":
                result["action_type"] = "EMERGENCY_CLOSE"

            return result

        # Layer 3: Check emergency hard stop
        emergency_check = await self.check_emergency_stop(
            symbol=symbol,
            current_price=current_price,
            entry_price=entry_price,
            position_side=position_side,
            quantity=quantity,
            leverage=leverage,
        )

        result["details"]["emergency"] = emergency_check

        if emergency_check["emergency_close"]:
            result["action_needed"] = True
            result["action_type"] = "EMERGENCY_CLOSE"
            result["urgency"] = "critical"
            return result

        # Gap detection (for monitoring/alerting)
        last_price = self._sl_protection_state.get("last_price")
        if last_price:
            gap_check = self.detect_price_gap(current_price, last_price)
            result["details"]["gap_check"] = gap_check

        # Update last price
        self._sl_protection_state["last_price"] = current_price
        self._sl_protection_state["last_price_time"] = datetime.now(timezone.utc)

        return result

    def get_stop_loss_protection_stats(self) -> Dict[str, Any]:
        """Get stop loss protection statistics."""
        self._init_stop_loss_protection()

        state = self._sl_protection_state

        # Calculate average slippage
        avg_slippage = Decimal("0")
        if state["slippage_history"]:
            total_slippage = sum(
                s["slippage_pct"] for s in state["slippage_history"]
            )
            avg_slippage = total_slippage / len(state["slippage_history"])

        return {
            "consecutive_failures": state["consecutive_failures"],
            "emergency_mode": state["emergency_mode"],
            "backup_sl_active": state["backup_sl_active"],
            "tiered_exits_remaining": state["tiered_exits_remaining"],
            "total_gap_events": len(state["gap_events"]),
            "total_slippage_events": len(state["slippage_history"]),
            "avg_slippage_pct": str(avg_slippage),
            "last_check_time": state["last_check_time"].isoformat() if state["last_check_time"] else None,
        }

    def reset_stop_loss_protection(self) -> None:
        """Reset stop loss protection state (call when position is closed)."""
        self._init_stop_loss_protection()

        self._sl_protection_state["consecutive_failures"] = 0
        self._sl_protection_state["emergency_mode"] = False
        self._sl_protection_state["backup_sl_active"] = False
        self._sl_protection_state["tiered_exits_remaining"] = self.TIERED_EXIT_LEVELS
        self._sl_protection_state["last_known_sl_status"] = None

        logger.debug(f"[{self._bot_id}] Stop loss protection state reset")

    # =========================================================================
    # Stop Loss Oscillation Prevention (止損震盪防護)
    # =========================================================================
    #
    # Prevents repeated stop loss triggers when price oscillates around
    # the stop loss level, which can lead to accumulated losses.
    # =========================================================================

    # Oscillation Prevention Configuration
    STOP_LOSS_COOLDOWN_SECONDS = 300  # 5 minutes cooldown after stop loss
    STOP_LOSS_BUFFER_ZONE_PCT = Decimal("0.005")  # 0.5% buffer zone
    MAX_STOP_LOSS_TRIGGERS_PER_HOUR = 3  # Max triggers allowed per hour
    STOP_LOSS_TRIGGER_RESET_HOURS = 1  # Reset trigger count after this many hours

    def _init_oscillation_prevention(self) -> None:
        """Initialize oscillation prevention state."""
        if not hasattr(self, "_oscillation_state"):
            self._oscillation_state: Dict[str, Any] = {
                "last_stop_loss_time": None,
                "stop_loss_triggers": [],  # List of trigger timestamps
                "cooldown_active": False,
                "cooldown_until": None,
                "buffer_zone_active": False,
                "last_stop_price": None,
                "progressive_reduction_level": 0,  # 0=full, 1=66%, 2=33%, 3=blocked
            }

    def check_stop_loss_cooldown(self) -> tuple[bool, str]:
        """
        Check if stop loss is in cooldown period.

        Returns:
            Tuple of (is_allowed, message)
        """
        self._init_oscillation_prevention()
        state = self._oscillation_state

        now = datetime.now(timezone.utc)

        # Check if in cooldown period
        if state["cooldown_until"] and now < state["cooldown_until"]:
            remaining = (state["cooldown_until"] - now).total_seconds()
            return False, f"Stop loss cooldown active ({remaining:.0f}s remaining)"

        # Check trigger count per hour
        recent_triggers = [
            t for t in state["stop_loss_triggers"]
            if (now - t).total_seconds() < self.STOP_LOSS_TRIGGER_RESET_HOURS * 3600
        ]
        state["stop_loss_triggers"] = recent_triggers  # Clean up old triggers

        if len(recent_triggers) >= self.MAX_STOP_LOSS_TRIGGERS_PER_HOUR:
            # Too many triggers - extend cooldown
            extended_cooldown = self.STOP_LOSS_COOLDOWN_SECONDS * 2
            state["cooldown_until"] = now + timedelta(seconds=extended_cooldown)
            state["cooldown_active"] = True

            logger.warning(
                f"[{self._bot_id}] Stop loss trigger limit reached "
                f"({len(recent_triggers)}/{self.MAX_STOP_LOSS_TRIGGERS_PER_HOUR}) - "
                f"extended cooldown {extended_cooldown}s"
            )
            return False, f"Too many stop losses ({len(recent_triggers)} in past hour)"

        state["cooldown_active"] = False
        return True, "Stop loss allowed"

    def record_stop_loss_trigger(self) -> None:
        """Record a stop loss trigger event."""
        self._init_oscillation_prevention()
        state = self._oscillation_state

        now = datetime.now(timezone.utc)
        state["last_stop_loss_time"] = now
        state["stop_loss_triggers"].append(now)

        # Set cooldown
        state["cooldown_until"] = now + timedelta(seconds=self.STOP_LOSS_COOLDOWN_SECONDS)
        state["cooldown_active"] = True

        # Update progressive reduction level
        recent_count = len([
            t for t in state["stop_loss_triggers"]
            if (now - t).total_seconds() < self.STOP_LOSS_TRIGGER_RESET_HOURS * 3600
        ])

        if recent_count >= 3:
            state["progressive_reduction_level"] = 3  # Block entries
        elif recent_count >= 2:
            state["progressive_reduction_level"] = 2  # 33% size
        elif recent_count >= 1:
            state["progressive_reduction_level"] = 1  # 66% size

        logger.info(
            f"[{self._bot_id}] Stop loss triggered - cooldown {self.STOP_LOSS_COOLDOWN_SECONDS}s, "
            f"reduction level {state['progressive_reduction_level']}"
        )

    def get_position_size_reduction(self) -> Decimal:
        """
        Get position size reduction factor based on recent stop losses.

        Returns:
            Decimal multiplier (1.0 = full size, 0.66 = 66%, 0.33 = 33%, 0 = blocked)
        """
        self._init_oscillation_prevention()
        level = self._oscillation_state["progressive_reduction_level"]

        if level >= 3:
            return Decimal("0")  # Blocked
        elif level == 2:
            return Decimal("0.33")
        elif level == 1:
            return Decimal("0.66")
        else:
            return Decimal("1.0")

    def check_buffer_zone(
        self,
        current_price: Decimal,
        stop_price: Decimal,
        position_side: str,
    ) -> bool:
        """
        Check if price is in buffer zone around stop loss.

        When price is in buffer zone, avoid new entries that could
        immediately hit stop loss.

        Args:
            current_price: Current market price
            stop_price: Stop loss price
            position_side: Position side (LONG/SHORT)

        Returns:
            True if price is in buffer zone (should avoid entry)
        """
        self._init_oscillation_prevention()

        buffer = stop_price * self.STOP_LOSS_BUFFER_ZONE_PCT

        if position_side.upper() == "LONG":
            # For long, buffer zone is above stop price
            buffer_upper = stop_price + buffer
            in_buffer = current_price <= buffer_upper
        else:
            # For short, buffer zone is below stop price
            buffer_lower = stop_price - buffer
            in_buffer = current_price >= buffer_lower

        if in_buffer:
            self._oscillation_state["buffer_zone_active"] = True
            logger.debug(
                f"[{self._bot_id}] Price {current_price} in buffer zone "
                f"around stop {stop_price}"
            )

        return in_buffer

    def reset_oscillation_state(self) -> None:
        """Reset oscillation prevention state (call after extended no-trade period)."""
        self._init_oscillation_prevention()

        # Only reset if enough time has passed since last trigger
        if self._oscillation_state["last_stop_loss_time"]:
            elapsed = (
                datetime.now(timezone.utc) -
                self._oscillation_state["last_stop_loss_time"]
            ).total_seconds()

            if elapsed > self.STOP_LOSS_TRIGGER_RESET_HOURS * 3600:
                self._oscillation_state["progressive_reduction_level"] = 0
                self._oscillation_state["stop_loss_triggers"] = []
                logger.info(f"[{self._bot_id}] Oscillation state reset after {elapsed/3600:.1f}h")

    def get_oscillation_stats(self) -> Dict[str, Any]:
        """Get oscillation prevention statistics."""
        self._init_oscillation_prevention()
        state = self._oscillation_state

        now = datetime.now(timezone.utc)
        recent_triggers = len([
            t for t in state["stop_loss_triggers"]
            if (now - t).total_seconds() < 3600
        ])

        return {
            "cooldown_active": state["cooldown_active"],
            "cooldown_remaining_seconds": (
                (state["cooldown_until"] - now).total_seconds()
                if state["cooldown_until"] and now < state["cooldown_until"]
                else 0
            ),
            "triggers_last_hour": recent_triggers,
            "max_triggers_per_hour": self.MAX_STOP_LOSS_TRIGGERS_PER_HOUR,
            "progressive_reduction_level": state["progressive_reduction_level"],
            "position_size_multiplier": str(self.get_position_size_reduction()),
        }

    # =========================================================================
    # Stop Loss Sync Management (止損同步管理)
    # =========================================================================
    #
    # Ensures stop loss orders stay in sync with strategy parameters
    # and position changes.
    # =========================================================================

    def _init_stop_loss_sync(self) -> None:
        """Initialize stop loss sync state."""
        if not hasattr(self, "_sl_sync_state"):
            self._sl_sync_state: Dict[str, Any] = {
                "current_sl_order_id": None,
                "current_sl_price": None,
                "current_sl_quantity": None,
                "sl_created_at": None,
                "position_entry_price_at_sl_creation": None,
                "position_quantity_at_sl_creation": None,
                "config_stop_loss_pct_at_creation": None,
                "last_sync_check": None,
                "sync_failures": 0,
                "version": 0,
            }

    def register_stop_loss_order(
        self,
        order_id: str,
        stop_price: Decimal,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss_pct: Decimal,
    ) -> None:
        """
        Register a stop loss order for sync tracking.

        Args:
            order_id: Stop loss order ID
            stop_price: Stop loss trigger price
            quantity: Order quantity
            entry_price: Position entry price at time of creation
            stop_loss_pct: Config stop loss percentage at time of creation
        """
        self._init_stop_loss_sync()

        self._sl_sync_state["current_sl_order_id"] = order_id
        self._sl_sync_state["current_sl_price"] = stop_price
        self._sl_sync_state["current_sl_quantity"] = quantity
        self._sl_sync_state["sl_created_at"] = datetime.now(timezone.utc)
        self._sl_sync_state["position_entry_price_at_sl_creation"] = entry_price
        self._sl_sync_state["position_quantity_at_sl_creation"] = quantity
        self._sl_sync_state["config_stop_loss_pct_at_creation"] = stop_loss_pct
        self._sl_sync_state["version"] += 1

        logger.debug(
            f"[{self._bot_id}] Registered stop loss: {order_id} @ {stop_price}, "
            f"v{self._sl_sync_state['version']}"
        )

    def check_stop_loss_sync(
        self,
        current_entry_price: Decimal,
        current_quantity: Decimal,
        current_stop_loss_pct: Decimal,
        position_side: str,
    ) -> Dict[str, Any]:
        """
        Check if stop loss is in sync with current position and config.

        Args:
            current_entry_price: Current position entry price
            current_quantity: Current position quantity
            current_stop_loss_pct: Current config stop loss percentage
            position_side: Position side (LONG/SHORT)

        Returns:
            Dict with sync status and recommended actions
        """
        self._init_stop_loss_sync()
        state = self._sl_sync_state

        result = {
            "is_synced": True,
            "needs_update": False,
            "reasons": [],
            "recommended_new_price": None,
            "recommended_new_quantity": None,
        }

        if not state["current_sl_order_id"]:
            result["is_synced"] = False
            result["needs_update"] = True
            result["reasons"].append("NO_STOP_LOSS_ORDER")
            return result

        # Check if entry price changed significantly (DCA)
        if state["position_entry_price_at_sl_creation"]:
            entry_change_pct = abs(
                (current_entry_price - state["position_entry_price_at_sl_creation"]) /
                state["position_entry_price_at_sl_creation"]
            )

            if entry_change_pct > Decimal("0.001"):  # 0.1% threshold
                result["is_synced"] = False
                result["needs_update"] = True
                result["reasons"].append(f"ENTRY_PRICE_CHANGED ({entry_change_pct*100:.2f}%)")

                # Calculate new stop price
                if position_side.upper() == "LONG":
                    result["recommended_new_price"] = current_entry_price * (
                        Decimal("1") - current_stop_loss_pct
                    )
                else:
                    result["recommended_new_price"] = current_entry_price * (
                        Decimal("1") + current_stop_loss_pct
                    )

        # Check if quantity changed
        if state["position_quantity_at_sl_creation"]:
            if current_quantity != state["position_quantity_at_sl_creation"]:
                result["is_synced"] = False
                result["needs_update"] = True
                result["reasons"].append("QUANTITY_CHANGED")
                result["recommended_new_quantity"] = current_quantity

        # Check if stop loss percentage config changed
        if state["config_stop_loss_pct_at_creation"]:
            if current_stop_loss_pct != state["config_stop_loss_pct_at_creation"]:
                result["is_synced"] = False
                result["needs_update"] = True
                result["reasons"].append("CONFIG_CHANGED")

                # Recalculate stop price with new config
                if position_side.upper() == "LONG":
                    result["recommended_new_price"] = current_entry_price * (
                        Decimal("1") - current_stop_loss_pct
                    )
                else:
                    result["recommended_new_price"] = current_entry_price * (
                        Decimal("1") + current_stop_loss_pct
                    )

        # Round recommended price
        if result["recommended_new_price"]:
            result["recommended_new_price"] = result["recommended_new_price"].quantize(
                Decimal("0.1")
            )

        state["last_sync_check"] = datetime.now(timezone.utc)

        if not result["is_synced"]:
            logger.warning(
                f"[{self._bot_id}] Stop loss out of sync: {result['reasons']}"
            )

        return result

    async def sync_stop_loss_order(
        self,
        symbol: str,
        position_side: str,
        current_entry_price: Decimal,
        current_quantity: Decimal,
        current_stop_loss_pct: Decimal,
    ) -> Dict[str, Any]:
        """
        Sync stop loss order with current position and config.

        This method:
        1. Checks if sync is needed
        2. Cancels old stop loss if exists
        3. Places new stop loss with correct parameters

        Args:
            symbol: Trading symbol
            position_side: Position side
            current_entry_price: Current entry price
            current_quantity: Current quantity
            current_stop_loss_pct: Current stop loss config

        Returns:
            Dict with sync result
        """
        self._init_stop_loss_sync()

        result = {
            "success": False,
            "action_taken": None,
            "new_order_id": None,
            "new_stop_price": None,
            "error": None,
        }

        # Check if sync is needed
        sync_check = self.check_stop_loss_sync(
            current_entry_price=current_entry_price,
            current_quantity=current_quantity,
            current_stop_loss_pct=current_stop_loss_pct,
            position_side=position_side,
        )

        if sync_check["is_synced"]:
            result["success"] = True
            result["action_taken"] = "ALREADY_SYNCED"
            return result

        try:
            # Cancel existing stop loss if any
            if self._sl_sync_state["current_sl_order_id"]:
                cancel_result = await self._cancel_algo_order_with_verification(
                    algo_id=self._sl_sync_state["current_sl_order_id"],
                    symbol=symbol,
                )

                if not cancel_result["is_cancelled"] and not cancel_result["was_triggered"]:
                    result["error"] = f"Failed to cancel old stop loss: {cancel_result['error_message']}"
                    self._sl_sync_state["sync_failures"] += 1
                    return result

            # Calculate new stop price
            if position_side.upper() == "LONG":
                new_stop_price = current_entry_price * (Decimal("1") - current_stop_loss_pct)
                close_side = "SELL"
            else:
                new_stop_price = current_entry_price * (Decimal("1") + current_stop_loss_pct)
                close_side = "BUY"

            # Use symbol tick size for price precision
            symbol_info = await self._get_symbol_info(symbol)
            new_stop_price = self._normalize_price(new_stop_price, symbol_info)

            # Place new stop loss
            new_order = await self._exchange.futures_create_order(
                symbol=symbol,
                side=close_side,
                order_type="STOP_MARKET",
                quantity=current_quantity,
                stop_price=new_stop_price,
                reduce_only=True,
                position_side=position_side.upper(),
                bot_id=self._bot_id,
            )

            if new_order:
                new_order_id = str(getattr(new_order, "order_id", ""))

                # Register the new stop loss
                self.register_stop_loss_order(
                    order_id=new_order_id,
                    stop_price=new_stop_price,
                    quantity=current_quantity,
                    entry_price=current_entry_price,
                    stop_loss_pct=current_stop_loss_pct,
                )

                result["success"] = True
                result["action_taken"] = "SYNCED"
                result["new_order_id"] = new_order_id
                result["new_stop_price"] = new_stop_price
                self._sl_sync_state["sync_failures"] = 0

                logger.info(
                    f"[{self._bot_id}] Stop loss synced: {new_order_id} @ {new_stop_price}, "
                    f"reasons: {sync_check['reasons']}"
                )

        except Exception as e:
            result["error"] = str(e)
            self._sl_sync_state["sync_failures"] += 1
            logger.error(f"[{self._bot_id}] Stop loss sync failed: {e}")

        return result

    def clear_stop_loss_sync(self) -> None:
        """Clear stop loss sync state (call when position is closed)."""
        self._init_stop_loss_sync()

        self._sl_sync_state["current_sl_order_id"] = None
        self._sl_sync_state["current_sl_price"] = None
        self._sl_sync_state["current_sl_quantity"] = None
        self._sl_sync_state["position_entry_price_at_sl_creation"] = None
        self._sl_sync_state["position_quantity_at_sl_creation"] = None
        self._sl_sync_state["config_stop_loss_pct_at_creation"] = None

        logger.debug(f"[{self._bot_id}] Stop loss sync state cleared")

    # =========================================================================
    # Circuit Breaker Position Handling (熔斷後持倉處理)
    # =========================================================================
    #
    # Ensures all positions are properly closed when circuit breaker
    # triggers, preventing continued risk exposure.
    # =========================================================================

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FORCE_CLOSE = True  # Force close positions on circuit breaker
    CIRCUIT_BREAKER_CANCEL_ORDERS = True  # Cancel all pending orders
    CIRCUIT_BREAKER_LOCKOUT_HOURS = 24  # Lockout period after circuit breaker

    def _init_circuit_breaker(self) -> None:
        """Initialize circuit breaker state."""
        if not hasattr(self, "_circuit_breaker_state"):
            self._circuit_breaker_state: Dict[str, Any] = {
                "is_triggered": False,
                "trigger_time": None,
                "trigger_reason": None,
                "lockout_until": None,
                "positions_closed": [],
                "orders_cancelled": [],
                "trigger_count_today": 0,
                "last_trigger_date": None,
            }

    def is_circuit_breaker_active(self) -> tuple[bool, Optional[str]]:
        """
        Check if circuit breaker is currently active.

        Returns:
            Tuple of (is_active, reason)
        """
        self._init_circuit_breaker()
        state = self._circuit_breaker_state

        if not state["is_triggered"]:
            return False, None

        now = datetime.now(timezone.utc)

        # Check if lockout period has expired
        if state["lockout_until"] and now >= state["lockout_until"]:
            # Auto-reset after lockout
            self._reset_circuit_breaker()
            return False, None

        remaining = (state["lockout_until"] - now).total_seconds() / 3600 if state["lockout_until"] else 0
        return True, f"Circuit breaker active: {state['trigger_reason']} ({remaining:.1f}h remaining)"

    async def trigger_circuit_breaker(
        self,
        reason: str,
        positions: Optional[List[Dict]] = None,
        force_close: bool = True,
    ) -> Dict[str, Any]:
        """
        Trigger circuit breaker and handle all positions.

        This method:
        1. Marks circuit breaker as triggered
        2. Closes all open positions (if force_close=True)
        3. Cancels all pending orders
        4. Sets lockout period

        Args:
            reason: Reason for triggering (e.g., "DAILY_LOSS_LIMIT", "MAX_DRAWDOWN")
            positions: List of positions to close (optional, will query if not provided)
            force_close: Whether to force close positions

        Returns:
            Dict with circuit breaker result
        """
        self._init_circuit_breaker()
        state = self._circuit_breaker_state

        now = datetime.now(timezone.utc)

        result = {
            "triggered": True,
            "reason": reason,
            "positions_closed": [],
            "orders_cancelled": [],
            "errors": [],
            "lockout_until": None,
        }

        logger.critical(
            f"[{self._bot_id}] CIRCUIT BREAKER TRIGGERED: {reason}"
        )

        # Update state
        state["is_triggered"] = True
        state["trigger_time"] = now
        state["trigger_reason"] = reason
        state["lockout_until"] = now + timedelta(hours=self.CIRCUIT_BREAKER_LOCKOUT_HOURS)
        result["lockout_until"] = state["lockout_until"]

        # Update daily trigger count
        today = now.date()
        if state["last_trigger_date"] != today:
            state["trigger_count_today"] = 0
            state["last_trigger_date"] = today
        state["trigger_count_today"] += 1

        # Force close all positions
        if force_close and self.CIRCUIT_BREAKER_FORCE_CLOSE:
            close_result = await self._circuit_breaker_close_all_positions()
            result["positions_closed"] = close_result.get("closed", [])
            if close_result.get("errors"):
                result["errors"].extend(close_result["errors"])

        # Cancel all pending orders
        if self.CIRCUIT_BREAKER_CANCEL_ORDERS:
            cancel_result = await self._circuit_breaker_cancel_all_orders()
            result["orders_cancelled"] = cancel_result.get("cancelled", [])
            if cancel_result.get("errors"):
                result["errors"].extend(cancel_result["errors"])

        state["positions_closed"] = result["positions_closed"]
        state["orders_cancelled"] = result["orders_cancelled"]

        # Send critical notification
        if hasattr(self, "_notifier") and self._notifier:
            try:
                await self._notifier.send_alert(
                    level="CRITICAL",
                    title=f"CIRCUIT BREAKER - {self._bot_id}",
                    message=(
                        f"Circuit breaker triggered!\n"
                        f"Reason: {reason}\n"
                        f"Positions closed: {len(result['positions_closed'])}\n"
                        f"Orders cancelled: {len(result['orders_cancelled'])}\n"
                        f"Lockout until: {state['lockout_until'].isoformat()}"
                    ),
                )
            except Exception as e:
                logger.debug(f"Failed to send circuit breaker notification: {e}")

        return result

    async def _circuit_breaker_close_all_positions(self) -> Dict[str, Any]:
        """
        Close all positions as part of circuit breaker.

        Returns:
            Dict with closed positions and errors
        """
        result = {
            "closed": [],
            "errors": [],
        }

        try:
            # Get symbol from config if available
            symbol = getattr(self._config, "symbol", None) if hasattr(self, "_config") else None

            if symbol:
                positions = await self._exchange.futures.get_positions(symbol)
            else:
                # Try to get all positions
                positions = await self._exchange.futures.get_positions()

            for pos in positions:
                if pos.quantity == 0 or abs(pos.quantity) == Decimal("0"):
                    continue

                try:
                    # Determine close side (handle negative qty for shorts)
                    close_side = "SELL" if pos.side.value.upper() == "LONG" else "BUY"
                    pos_symbol = getattr(pos, "symbol", symbol)

                    # Place market close order
                    order = await self._exchange.futures_create_order(
                        symbol=pos_symbol,
                        side=close_side,
                        order_type="MARKET",
                        quantity=pos.quantity,
                        reduce_only=True,
                        position_side=pos.side.value.upper(),
                        bot_id=self._bot_id,
                    )

                    if order:
                        result["closed"].append({
                            "symbol": pos_symbol,
                            "side": pos.side.value,
                            "quantity": str(pos.quantity),
                            "order_id": str(getattr(order, "order_id", "")),
                        })

                        logger.critical(
                            f"[{self._bot_id}] Circuit breaker closed position: "
                            f"{pos.side.value} {pos.quantity} {pos_symbol}"
                        )

                except Exception as e:
                    error_msg = f"Failed to close {pos.symbol}: {e}"
                    result["errors"].append(error_msg)
                    logger.error(f"[{self._bot_id}] {error_msg}")

        except Exception as e:
            result["errors"].append(f"Failed to get positions: {e}")
            logger.error(f"[{self._bot_id}] Circuit breaker get positions failed: {e}")

        return result

    async def _circuit_breaker_cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all pending orders as part of circuit breaker.

        Returns:
            Dict with cancelled orders and errors
        """
        result = {
            "cancelled": [],
            "errors": [],
        }

        try:
            symbol = getattr(self._config, "symbol", None) if hasattr(self, "_config") else None

            if symbol:
                orders = await self._exchange.futures.get_open_orders(symbol)
            else:
                orders = await self._exchange.futures.get_open_orders()

            for order in orders:
                try:
                    order_symbol = getattr(order, "symbol", symbol)
                    order_id = str(getattr(order, "order_id", ""))

                    await self._exchange.futures.cancel_order(
                        symbol=order_symbol,
                        order_id=order_id,
                    )

                    result["cancelled"].append({
                        "symbol": order_symbol,
                        "order_id": order_id,
                        "type": getattr(order, "order_type", "UNKNOWN"),
                    })

                    logger.info(
                        f"[{self._bot_id}] Circuit breaker cancelled order: {order_id}"
                    )

                except Exception as e:
                    error_msg = f"Failed to cancel order {getattr(order, 'order_id', 'unknown')}: {e}"
                    result["errors"].append(error_msg)
                    logger.error(f"[{self._bot_id}] {error_msg}")

        except Exception as e:
            result["errors"].append(f"Failed to get orders: {e}")
            logger.error(f"[{self._bot_id}] Circuit breaker get orders failed: {e}")

        return result

    def check_entry_allowed(self) -> tuple[bool, str]:
        """
        Check if new entries are allowed (considering circuit breaker, cooldown, etc).

        This is a comprehensive check that should be called before any entry.

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check circuit breaker
        cb_active, cb_reason = self.is_circuit_breaker_active()
        if cb_active:
            return False, cb_reason

        # Check stop loss cooldown
        sl_allowed, sl_reason = self.check_stop_loss_cooldown()
        if not sl_allowed:
            return False, sl_reason

        # Check position size reduction
        size_mult = self.get_position_size_reduction()
        if size_mult <= 0:
            return False, "Position entries blocked due to repeated stop losses"

        # Check daily loss limit (if config supports it)
        daily_loss_limit = getattr(self._config, 'daily_loss_limit_pct', None)
        if daily_loss_limit is not None:
            daily_pnl = getattr(self, '_daily_pnl', Decimal("0"))
            capital = getattr(self._config, 'max_capital', None) or getattr(self, '_capital', None) or Decimal("1000")
            if daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / capital
                if daily_loss_pct >= daily_loss_limit:
                    return False, f"Daily loss limit reached: {daily_loss_pct:.1%} >= {daily_loss_limit:.1%}"

        # Check consecutive losses limit (if config supports it)
        max_consecutive = getattr(self._config, 'max_consecutive_losses', None)
        if max_consecutive is not None:
            consecutive = getattr(self, '_consecutive_losses', 0)
            if consecutive >= max_consecutive:
                return False, f"Max consecutive losses reached: {consecutive} >= {max_consecutive}"

        return True, "Entry allowed"

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state after lockout period."""
        self._init_circuit_breaker()

        self._circuit_breaker_state["is_triggered"] = False
        self._circuit_breaker_state["trigger_reason"] = None
        # Keep historical data for reference
        # Reset oscillation state as well
        self.reset_oscillation_state()

        logger.info(f"[{self._bot_id}] Circuit breaker reset after lockout period")

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        self._init_circuit_breaker()
        state = self._circuit_breaker_state

        now = datetime.now(timezone.utc)
        remaining_hours = 0
        if state["lockout_until"] and now < state["lockout_until"]:
            remaining_hours = (state["lockout_until"] - now).total_seconds() / 3600

        return {
            "is_triggered": state["is_triggered"],
            "trigger_reason": state["trigger_reason"],
            "trigger_time": state["trigger_time"].isoformat() if state["trigger_time"] else None,
            "lockout_remaining_hours": remaining_hours,
            "positions_closed": len(state["positions_closed"]),
            "orders_cancelled": len(state["orders_cancelled"]),
            "trigger_count_today": state["trigger_count_today"],
        }

    async def ensure_no_positions_after_circuit_breaker(self) -> Dict[str, Any]:
        """
        Verify and ensure no positions remain after circuit breaker.

        This is a safety check that should be called periodically after
        circuit breaker triggers to ensure no positions slipped through.

        Returns:
            Dict with verification result
        """
        self._init_circuit_breaker()

        if not self._circuit_breaker_state["is_triggered"]:
            return {"checked": False, "reason": "Circuit breaker not active"}

        result = {
            "checked": True,
            "positions_found": 0,
            "positions_closed": [],
            "errors": [],
        }

        try:
            symbol = getattr(self._config, "symbol", None) if hasattr(self, "_config") else None

            if symbol:
                positions = await self._exchange.futures.get_positions(symbol)
            else:
                positions = await self._exchange.futures.get_positions()

            active_positions = [p for p in positions if p.quantity != 0 and p.quantity != Decimal("0")]
            result["positions_found"] = len(active_positions)

            if active_positions:
                logger.warning(
                    f"[{self._bot_id}] Found {len(active_positions)} positions after "
                    f"circuit breaker - force closing"
                )

                close_result = await self._circuit_breaker_close_all_positions()
                result["positions_closed"] = close_result.get("closed", [])
                result["errors"] = close_result.get("errors", [])

        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"[{self._bot_id}] Position verification failed: {e}")

        return result

    # =========================================================================
    # Comprehensive Entry Validation (綜合進場驗證)
    # =========================================================================

    async def validate_entry_comprehensive(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive entry validation including all protection checks.

        This is the RECOMMENDED method to call before any new position entry.

        Args:
            symbol: Trading symbol
            side: Entry side (BUY/SELL)
            quantity: Entry quantity
            price: Entry price
            stop_loss_price: Optional stop loss price for buffer zone check

        Returns:
            Tuple of (is_allowed, message, details)
        """
        details = {
            "circuit_breaker": {"passed": False},
            "cooldown": {"passed": False},
            "position_size": {"passed": False, "multiplier": "1.0"},
            "buffer_zone": {"passed": True},  # Default to passed if no stop price
            "global_risk": {"passed": False},
        }

        # 1. Check circuit breaker
        cb_active, cb_reason = self.is_circuit_breaker_active()
        if cb_active:
            details["circuit_breaker"]["reason"] = cb_reason
            return False, cb_reason, details
        details["circuit_breaker"]["passed"] = True

        # 2. Check stop loss cooldown
        sl_allowed, sl_reason = self.check_stop_loss_cooldown()
        if not sl_allowed:
            details["cooldown"]["reason"] = sl_reason
            return False, sl_reason, details
        details["cooldown"]["passed"] = True

        # 3. Check position size reduction
        size_mult = self.get_position_size_reduction()
        details["position_size"]["multiplier"] = str(size_mult)
        if size_mult <= 0:
            details["position_size"]["reason"] = "Blocked due to repeated stop losses"
            return False, "Entry blocked due to repeated stop losses", details
        details["position_size"]["passed"] = True

        # 4. Check buffer zone if stop loss price provided
        if stop_loss_price:
            position_side = "LONG" if side.upper() == "BUY" else "SHORT"
            in_buffer = self.check_buffer_zone(price, stop_loss_price, position_side)
            if in_buffer:
                details["buffer_zone"]["passed"] = False
                details["buffer_zone"]["reason"] = "Price in stop loss buffer zone"
                return False, "Price too close to stop loss level", details

        # 5. Check global risk limits
        exposure = quantity * price
        global_ok, global_msg, global_details = await self.check_global_risk_limits(
            bot_id=self._bot_id,
            symbol=symbol,
            additional_exposure=exposure,
        )
        details["global_risk"] = {"passed": global_ok, "details": global_details}
        if not global_ok:
            return False, f"Global risk: {global_msg}", details

        # Apply position size reduction to quantity
        adjusted_quantity = quantity * size_mult
        details["adjusted_quantity"] = str(adjusted_quantity)

        return True, "Entry validation passed", details

    # =========================================================================
    # Circuit Breaker False Positive Prevention (熔斷誤觸發防護)
    # =========================================================================
    #
    # Prevents false circuit breaker triggers caused by:
    # - Data anomalies (bad ticks, API errors)
    # - Temporary price spikes
    # - Calculation errors
    # =========================================================================

    # False Positive Prevention Configuration
    CB_CONFIRMATION_CHECKS = 3  # Number of consecutive checks required
    CB_CONFIRMATION_INTERVAL_SECONDS = 5  # Interval between confirmation checks
    CB_MAX_PRICE_CHANGE_PCT = Decimal("0.10")  # 10% max realistic price change
    CB_MIN_DATA_FRESHNESS_SECONDS = 30  # Data must be within 30 seconds
    CB_REQUIRE_MULTIPLE_SOURCES = False  # Require confirmation from multiple sources

    def _init_cb_validation(self) -> None:
        """Initialize circuit breaker validation state."""
        if not hasattr(self, "_cb_validation_state"):
            self._cb_validation_state: Dict[str, Any] = {
                "pending_trigger": False,
                "pending_reason": None,
                "confirmation_count": 0,
                "first_trigger_time": None,
                "last_prices": [],  # Recent prices for anomaly detection
                "last_capital_values": [],  # Recent capital for anomaly detection
                "false_positive_count": 0,
                "last_false_positive_time": None,
            }

    def validate_circuit_breaker_trigger(
        self,
        reason: str,
        current_price: Optional[Decimal] = None,
        current_capital: Optional[Decimal] = None,
        loss_pct: Optional[Decimal] = None,
    ) -> tuple[bool, str]:
        """
        Validate if circuit breaker should actually trigger.

        This prevents false positives from data anomalies.

        Args:
            reason: Reason for potential trigger
            current_price: Current market price
            current_capital: Current account capital
            loss_pct: Calculated loss percentage

        Returns:
            Tuple of (should_trigger, validation_message)
        """
        self._init_cb_validation()
        state = self._cb_validation_state

        now = datetime.now(timezone.utc)

        # Check 1: Price sanity check - detect impossible price movements
        if current_price and state["last_prices"]:
            last_price = state["last_prices"][-1]["price"]
            price_change_pct = abs((current_price - last_price) / last_price)

            if price_change_pct > self.CB_MAX_PRICE_CHANGE_PCT:
                logger.warning(
                    f"[{self._bot_id}] Circuit breaker validation: "
                    f"Price change {price_change_pct*100:.2f}% exceeds max "
                    f"{self.CB_MAX_PRICE_CHANGE_PCT*100:.1f}% - possible data anomaly"
                )
                state["false_positive_count"] += 1
                state["last_false_positive_time"] = now
                return False, f"Price anomaly detected: {price_change_pct*100:.2f}% change"

        # Check 2: Capital sanity check - detect impossible capital changes
        if current_capital and state["last_capital_values"]:
            last_capital = state["last_capital_values"][-1]["value"]
            if last_capital > 0:
                capital_change_pct = abs((current_capital - last_capital) / last_capital)

                # Capital shouldn't change more than 50% in a short period
                if capital_change_pct > Decimal("0.50"):
                    logger.warning(
                        f"[{self._bot_id}] Circuit breaker validation: "
                        f"Capital change {capital_change_pct*100:.2f}% too large - possible data error"
                    )
                    state["false_positive_count"] += 1
                    state["last_false_positive_time"] = now
                    return False, f"Capital anomaly detected: {capital_change_pct*100:.2f}% change"

        # Check 3: Data freshness - ensure we have recent data
        if hasattr(self, "_last_capital_update_time") and self._last_capital_update_time:
            data_age = (now - self._last_capital_update_time).total_seconds()
            if data_age > self.CB_MIN_DATA_FRESHNESS_SECONDS:
                logger.warning(
                    f"[{self._bot_id}] Circuit breaker validation: "
                    f"Data is {data_age:.1f}s old - may be stale"
                )
                # Don't trigger on stale data, but don't count as false positive
                return False, f"Data too stale: {data_age:.1f}s old"

        # Check 4: Require confirmation (multiple consecutive checks)
        if not state["pending_trigger"]:
            # First trigger - start confirmation process
            state["pending_trigger"] = True
            state["pending_reason"] = reason
            state["confirmation_count"] = 1
            state["first_trigger_time"] = now

            logger.info(
                f"[{self._bot_id}] Circuit breaker pending: {reason} "
                f"(confirmation 1/{self.CB_CONFIRMATION_CHECKS})"
            )
            return False, f"Confirmation required (1/{self.CB_CONFIRMATION_CHECKS})"

        else:
            # Subsequent trigger - check if same reason and increment
            if state["pending_reason"] == reason:
                state["confirmation_count"] += 1

                if state["confirmation_count"] >= self.CB_CONFIRMATION_CHECKS:
                    # Confirmed - allow trigger
                    logger.info(
                        f"[{self._bot_id}] Circuit breaker confirmed after "
                        f"{state['confirmation_count']} checks"
                    )
                    self._reset_cb_validation()
                    return True, "Circuit breaker confirmed"
                else:
                    logger.info(
                        f"[{self._bot_id}] Circuit breaker pending: {reason} "
                        f"(confirmation {state['confirmation_count']}/{self.CB_CONFIRMATION_CHECKS})"
                    )
                    return False, f"Confirmation required ({state['confirmation_count']}/{self.CB_CONFIRMATION_CHECKS})"
            else:
                # Different reason - reset confirmation
                state["pending_reason"] = reason
                state["confirmation_count"] = 1
                state["first_trigger_time"] = now
                return False, f"New reason, confirmation reset (1/{self.CB_CONFIRMATION_CHECKS})"

    def record_price_for_validation(self, price: Decimal) -> None:
        """Record price for circuit breaker validation."""
        self._init_cb_validation()

        self._cb_validation_state["last_prices"].append({
            "price": price,
            "time": datetime.now(timezone.utc),
        })

        # Keep only last 10 prices
        self._cb_validation_state["last_prices"] = \
            self._cb_validation_state["last_prices"][-10:]

    def record_capital_for_validation(self, capital: Decimal) -> None:
        """Record capital for circuit breaker validation."""
        self._init_cb_validation()

        self._cb_validation_state["last_capital_values"].append({
            "value": capital,
            "time": datetime.now(timezone.utc),
        })

        # Keep only last 10 values
        self._cb_validation_state["last_capital_values"] = \
            self._cb_validation_state["last_capital_values"][-10:]

    def _reset_cb_validation(self) -> None:
        """Reset circuit breaker validation state."""
        self._init_cb_validation()
        self._cb_validation_state["pending_trigger"] = False
        self._cb_validation_state["pending_reason"] = None
        self._cb_validation_state["confirmation_count"] = 0
        self._cb_validation_state["first_trigger_time"] = None

    def cancel_pending_circuit_breaker(self) -> None:
        """
        Cancel a pending circuit breaker trigger.

        Call this when conditions improve before confirmation completes.
        """
        self._init_cb_validation()

        if self._cb_validation_state["pending_trigger"]:
            logger.info(
                f"[{self._bot_id}] Pending circuit breaker cancelled - "
                f"conditions improved"
            )
            self._reset_cb_validation()

    # =========================================================================
    # Smart Circuit Breaker Recovery (智慧熔斷恢復)
    # =========================================================================
    #
    # Determines optimal timing and conditions for safe trading resumption.
    # =========================================================================

    # Recovery Configuration
    CB_MIN_RECOVERY_HOURS = 4  # Minimum hours before recovery check
    CB_MAX_RECOVERY_HOURS = 48  # Maximum hours before forced review
    CB_RECOVERY_VOLATILITY_THRESHOLD = Decimal("0.02")  # 2% max volatility for recovery
    CB_RECOVERY_POSITION_SIZE_PCT = Decimal("0.50")  # Start with 50% position size
    CB_GRADUATED_RECOVERY_LEVELS = 3  # Number of graduated recovery levels

    def _init_cb_recovery(self) -> None:
        """Initialize circuit breaker recovery state."""
        if not hasattr(self, "_cb_recovery_state"):
            self._cb_recovery_state: Dict[str, Any] = {
                "recovery_level": 0,  # 0=locked, 1=limited, 2=reduced, 3=full
                "recovery_start_time": None,
                "last_recovery_check": None,
                "recovery_conditions_met": False,
                "manual_override": False,
                "recovery_position_multiplier": Decimal("0"),
                "consecutive_profitable_trades": 0,
                "volatility_readings": [],
            }

    def check_recovery_conditions(
        self,
        current_volatility: Optional[Decimal] = None,
        market_trend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if conditions are suitable for circuit breaker recovery.

        Args:
            current_volatility: Current market volatility (e.g., ATR/price)
            market_trend: Current market trend ("bullish", "bearish", "neutral")

        Returns:
            Dict with recovery assessment
        """
        self._init_cb_recovery()
        self._init_circuit_breaker()

        cb_state = self._circuit_breaker_state
        recovery_state = self._cb_recovery_state

        now = datetime.now(timezone.utc)

        result = {
            "can_recover": False,
            "recovery_level": recovery_state["recovery_level"],
            "position_multiplier": recovery_state["recovery_position_multiplier"],
            "reasons": [],
            "recommendations": [],
        }

        # Check if circuit breaker is active
        if not cb_state["is_triggered"]:
            result["can_recover"] = True
            result["recovery_level"] = 3
            result["position_multiplier"] = Decimal("1.0")
            result["reasons"].append("Circuit breaker not active")
            return result

        # Check minimum recovery time
        if cb_state["trigger_time"]:
            hours_since_trigger = (now - cb_state["trigger_time"]).total_seconds() / 3600

            if hours_since_trigger < self.CB_MIN_RECOVERY_HOURS:
                result["reasons"].append(
                    f"Minimum recovery time not met ({hours_since_trigger:.1f}h / "
                    f"{self.CB_MIN_RECOVERY_HOURS}h)"
                )
                result["recommendations"].append(
                    f"Wait {self.CB_MIN_RECOVERY_HOURS - hours_since_trigger:.1f} more hours"
                )

            elif hours_since_trigger > self.CB_MAX_RECOVERY_HOURS:
                result["recommendations"].append(
                    f"Extended lockout ({hours_since_trigger:.1f}h) - manual review recommended"
                )

        # Check volatility conditions
        if current_volatility is not None:
            recovery_state["volatility_readings"].append({
                "value": current_volatility,
                "time": now,
            })
            # Keep only last 10 readings
            recovery_state["volatility_readings"] = \
                recovery_state["volatility_readings"][-10:]

            if current_volatility > self.CB_RECOVERY_VOLATILITY_THRESHOLD:
                result["reasons"].append(
                    f"Volatility too high ({current_volatility*100:.2f}% > "
                    f"{self.CB_RECOVERY_VOLATILITY_THRESHOLD*100:.1f}%)"
                )
                result["recommendations"].append("Wait for volatility to decrease")
            else:
                result["reasons"].append("Volatility acceptable")

        # Check for manual override
        if recovery_state["manual_override"]:
            result["can_recover"] = True
            result["reasons"].append("Manual override active")

        # Determine recovery level based on conditions
        conditions_count = len([r for r in result["reasons"] if "acceptable" in r.lower() or "met" in r.lower()])

        if recovery_state["manual_override"]:
            recovery_level = min(recovery_state["recovery_level"] + 1, 3)
        elif conditions_count >= 2 and hours_since_trigger >= self.CB_MIN_RECOVERY_HOURS:
            recovery_level = 1  # Limited recovery
        else:
            recovery_level = 0  # Still locked

        recovery_state["recovery_level"] = recovery_level
        result["recovery_level"] = recovery_level

        # Set position multiplier based on recovery level
        if recovery_level == 0:
            result["position_multiplier"] = Decimal("0")
        elif recovery_level == 1:
            result["position_multiplier"] = Decimal("0.33")  # 33%
        elif recovery_level == 2:
            result["position_multiplier"] = Decimal("0.66")  # 66%
        else:
            result["position_multiplier"] = Decimal("1.0")  # Full

        recovery_state["recovery_position_multiplier"] = result["position_multiplier"]
        recovery_state["last_recovery_check"] = now

        result["can_recover"] = recovery_level > 0

        return result

    def request_manual_recovery(self, override_level: int = 1) -> Dict[str, Any]:
        """
        Request manual recovery from circuit breaker.

        Args:
            override_level: Recovery level to set (1=limited, 2=reduced, 3=full)

        Returns:
            Dict with recovery result
        """
        self._init_cb_recovery()
        self._init_circuit_breaker()

        result = {
            "success": False,
            "previous_level": self._cb_recovery_state["recovery_level"],
            "new_level": override_level,
            "message": None,
        }

        if not self._circuit_breaker_state["is_triggered"]:
            result["success"] = True
            result["message"] = "Circuit breaker not active - no recovery needed"
            return result

        # Validate override level
        override_level = max(1, min(3, override_level))

        self._cb_recovery_state["manual_override"] = True
        self._cb_recovery_state["recovery_level"] = override_level
        self._cb_recovery_state["recovery_start_time"] = datetime.now(timezone.utc)

        # Set position multiplier
        if override_level == 1:
            self._cb_recovery_state["recovery_position_multiplier"] = Decimal("0.33")
        elif override_level == 2:
            self._cb_recovery_state["recovery_position_multiplier"] = Decimal("0.66")
        else:
            self._cb_recovery_state["recovery_position_multiplier"] = Decimal("1.0")

        result["success"] = True
        result["message"] = f"Manual recovery to level {override_level}"

        logger.info(
            f"[{self._bot_id}] Manual circuit breaker recovery: level {override_level}, "
            f"position multiplier {self._cb_recovery_state['recovery_position_multiplier']}"
        )

        return result

    def record_trade_result_for_recovery(self, is_profitable: bool) -> None:
        """
        Record trade result for graduated recovery assessment.

        Args:
            is_profitable: Whether the trade was profitable
        """
        self._init_cb_recovery()

        if is_profitable:
            self._cb_recovery_state["consecutive_profitable_trades"] += 1

            # Automatically upgrade recovery level after successful trades
            if self._cb_recovery_state["consecutive_profitable_trades"] >= 3:
                current_level = self._cb_recovery_state["recovery_level"]
                if current_level < 3:
                    self._cb_recovery_state["recovery_level"] = current_level + 1
                    self._cb_recovery_state["consecutive_profitable_trades"] = 0

                    # Update position multiplier
                    if self._cb_recovery_state["recovery_level"] == 2:
                        self._cb_recovery_state["recovery_position_multiplier"] = Decimal("0.66")
                    elif self._cb_recovery_state["recovery_level"] == 3:
                        self._cb_recovery_state["recovery_position_multiplier"] = Decimal("1.0")

                    logger.info(
                        f"[{self._bot_id}] Recovery level upgraded to "
                        f"{self._cb_recovery_state['recovery_level']} after profitable trades"
                    )
        else:
            # Reset consecutive count on loss
            self._cb_recovery_state["consecutive_profitable_trades"] = 0

            # Downgrade recovery level on loss during recovery
            if self._cb_recovery_state["recovery_level"] > 1:
                self._cb_recovery_state["recovery_level"] -= 1

                if self._cb_recovery_state["recovery_level"] == 1:
                    self._cb_recovery_state["recovery_position_multiplier"] = Decimal("0.33")
                elif self._cb_recovery_state["recovery_level"] == 2:
                    self._cb_recovery_state["recovery_position_multiplier"] = Decimal("0.66")

                logger.warning(
                    f"[{self._bot_id}] Recovery level downgraded to "
                    f"{self._cb_recovery_state['recovery_level']} after loss"
                )

    def get_recovery_position_multiplier(self) -> Decimal:
        """Get current position size multiplier based on recovery state."""
        self._init_cb_recovery()

        # If circuit breaker not active, full size
        if not hasattr(self, "_circuit_breaker_state") or \
           not self._circuit_breaker_state.get("is_triggered"):
            return Decimal("1.0")

        return self._cb_recovery_state["recovery_position_multiplier"]

    def full_recovery(self) -> None:
        """Perform full recovery from circuit breaker."""
        self._init_cb_recovery()
        self._init_circuit_breaker()

        self._circuit_breaker_state["is_triggered"] = False
        self._circuit_breaker_state["trigger_reason"] = None
        self._circuit_breaker_state["lockout_until"] = None

        self._cb_recovery_state["recovery_level"] = 3
        self._cb_recovery_state["recovery_position_multiplier"] = Decimal("1.0")
        self._cb_recovery_state["manual_override"] = False
        self._cb_recovery_state["consecutive_profitable_trades"] = 0

        logger.info(f"[{self._bot_id}] Full circuit breaker recovery completed")

    # =========================================================================
    # Partial Circuit Breaker (部分熔斷邏輯)
    # =========================================================================
    #
    # Allows selective circuit breaker affecting only specific strategies
    # while considering inter-strategy dependencies.
    # =========================================================================

    # Partial Circuit Breaker Configuration
    CB_ALLOW_PARTIAL = True  # Allow partial (strategy-level) circuit breaker
    CB_DEPENDENCY_CHECK = True  # Check strategy dependencies before partial CB

    # Class-level tracking for multi-strategy coordination
    _global_cb_registry: Optional[Dict[str, Any]] = None
    _global_cb_lock: Optional[asyncio.Lock] = None

    @classmethod
    def _init_global_cb_registry(cls) -> None:
        """Initialize global circuit breaker registry."""
        if cls._global_cb_registry is None:
            cls._global_cb_registry = {
                "strategies": {},  # bot_id -> CB state
                "dependencies": {},  # bot_id -> list of dependent bot_ids
                "global_cb_active": False,
                "global_cb_reason": None,
            }
        if cls._global_cb_lock is None:
            cls._global_cb_lock = asyncio.Lock()

    @classmethod
    async def register_strategy_for_cb(
        cls,
        bot_id: str,
        strategy_type: str,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Register a strategy for partial circuit breaker coordination.

        Args:
            bot_id: Bot identifier
            strategy_type: Type of strategy (e.g., "grid", "trend", "scalp")
            dependencies: List of bot_ids this strategy depends on
        """
        cls._init_global_cb_registry()

        async with cls._global_cb_lock:
            cls._global_cb_registry["strategies"][bot_id] = {
                "strategy_type": strategy_type,
                "cb_active": False,
                "cb_reason": None,
                "cb_time": None,
                "can_trade": True,
            }

            if dependencies:
                cls._global_cb_registry["dependencies"][bot_id] = dependencies

            logger.debug(
                f"Registered strategy {bot_id} ({strategy_type}) for CB coordination"
            )

    @classmethod
    async def trigger_partial_circuit_breaker(
        cls,
        bot_id: str,
        reason: str,
        affect_dependents: bool = True,
    ) -> Dict[str, Any]:
        """
        Trigger circuit breaker for a specific strategy only.

        Args:
            bot_id: Bot to trigger CB for
            reason: Reason for CB
            affect_dependents: Whether to also affect dependent strategies

        Returns:
            Dict with affected strategies
        """
        cls._init_global_cb_registry()

        result = {
            "triggered_for": [bot_id],
            "affected_dependents": [],
            "reason": reason,
        }

        async with cls._global_cb_lock:
            # Mark the primary strategy
            if bot_id in cls._global_cb_registry["strategies"]:
                cls._global_cb_registry["strategies"][bot_id]["cb_active"] = True
                cls._global_cb_registry["strategies"][bot_id]["cb_reason"] = reason
                cls._global_cb_registry["strategies"][bot_id]["cb_time"] = datetime.now(timezone.utc)
                cls._global_cb_registry["strategies"][bot_id]["can_trade"] = False

            # Find and affect dependents if requested
            if affect_dependents and cls.CB_DEPENDENCY_CHECK:
                for dep_bot_id, deps in cls._global_cb_registry["dependencies"].items():
                    if bot_id in deps:
                        # This strategy depends on the CB'd strategy
                        if dep_bot_id in cls._global_cb_registry["strategies"]:
                            cls._global_cb_registry["strategies"][dep_bot_id]["can_trade"] = False
                            result["affected_dependents"].append(dep_bot_id)

                            logger.warning(
                                f"Strategy {dep_bot_id} affected by partial CB "
                                f"(depends on {bot_id})"
                            )

        logger.info(
            f"Partial circuit breaker triggered for {bot_id}: {reason}, "
            f"affected dependents: {result['affected_dependents']}"
        )

        return result

    @classmethod
    async def trigger_global_circuit_breaker(
        cls,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Trigger global circuit breaker affecting all strategies.

        Args:
            reason: Reason for global CB

        Returns:
            Dict with all affected strategies
        """
        cls._init_global_cb_registry()

        result = {
            "global": True,
            "reason": reason,
            "affected_strategies": [],
        }

        async with cls._global_cb_lock:
            cls._global_cb_registry["global_cb_active"] = True
            cls._global_cb_registry["global_cb_reason"] = reason

            for bot_id, state in cls._global_cb_registry["strategies"].items():
                state["cb_active"] = True
                state["cb_reason"] = f"GLOBAL: {reason}"
                state["cb_time"] = datetime.now(timezone.utc)
                state["can_trade"] = False
                result["affected_strategies"].append(bot_id)

        logger.critical(
            f"GLOBAL circuit breaker triggered: {reason}, "
            f"affected: {result['affected_strategies']}"
        )

        return result

    @classmethod
    async def check_strategy_can_trade(cls, bot_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if a specific strategy can trade.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        cls._init_global_cb_registry()

        async with cls._global_cb_lock:
            # Check global CB first
            if cls._global_cb_registry["global_cb_active"]:
                return False, f"Global CB: {cls._global_cb_registry['global_cb_reason']}"

            # Check strategy-specific CB
            if bot_id in cls._global_cb_registry["strategies"]:
                state = cls._global_cb_registry["strategies"][bot_id]
                if not state["can_trade"]:
                    return False, state.get("cb_reason", "Strategy CB active")

            # Check if any dependency is in CB
            if bot_id in cls._global_cb_registry["dependencies"]:
                for dep_id in cls._global_cb_registry["dependencies"][bot_id]:
                    if dep_id in cls._global_cb_registry["strategies"]:
                        dep_state = cls._global_cb_registry["strategies"][dep_id]
                        if dep_state["cb_active"]:
                            return False, f"Dependency {dep_id} in CB"

        return True, None

    @classmethod
    async def recover_partial_circuit_breaker(
        cls,
        bot_id: str,
        recover_dependents: bool = True,
    ) -> Dict[str, Any]:
        """
        Recover a specific strategy from circuit breaker.

        Args:
            bot_id: Bot to recover
            recover_dependents: Whether to also recover dependent strategies

        Returns:
            Dict with recovered strategies
        """
        cls._init_global_cb_registry()

        result = {
            "recovered": [bot_id],
            "recovered_dependents": [],
        }

        async with cls._global_cb_lock:
            if bot_id in cls._global_cb_registry["strategies"]:
                cls._global_cb_registry["strategies"][bot_id]["cb_active"] = False
                cls._global_cb_registry["strategies"][bot_id]["cb_reason"] = None
                cls._global_cb_registry["strategies"][bot_id]["can_trade"] = True

            # Recover dependents if requested
            if recover_dependents:
                for dep_bot_id, deps in cls._global_cb_registry["dependencies"].items():
                    if bot_id in deps:
                        # Check if all dependencies are now clear
                        all_deps_clear = True
                        for d in deps:
                            if d in cls._global_cb_registry["strategies"]:
                                if cls._global_cb_registry["strategies"][d]["cb_active"]:
                                    all_deps_clear = False
                                    break

                        if all_deps_clear and dep_bot_id in cls._global_cb_registry["strategies"]:
                            cls._global_cb_registry["strategies"][dep_bot_id]["can_trade"] = True
                            result["recovered_dependents"].append(dep_bot_id)

        logger.info(
            f"Partial CB recovery for {bot_id}, "
            f"recovered dependents: {result['recovered_dependents']}"
        )

        return result

    @classmethod
    async def recover_global_circuit_breaker(cls) -> Dict[str, Any]:
        """
        Recover from global circuit breaker.

        Returns:
            Dict with all recovered strategies
        """
        cls._init_global_cb_registry()

        result = {
            "recovered_strategies": [],
        }

        async with cls._global_cb_lock:
            cls._global_cb_registry["global_cb_active"] = False
            cls._global_cb_registry["global_cb_reason"] = None

            for bot_id, state in cls._global_cb_registry["strategies"].items():
                state["cb_active"] = False
                state["cb_reason"] = None
                state["can_trade"] = True
                result["recovered_strategies"].append(bot_id)

        logger.info(
            f"Global CB recovery, recovered: {result['recovered_strategies']}"
        )

        return result

    @classmethod
    def get_cb_coordination_status(cls) -> Dict[str, Any]:
        """Get circuit breaker coordination status across all strategies."""
        cls._init_global_cb_registry()

        return {
            "global_cb_active": cls._global_cb_registry["global_cb_active"],
            "global_cb_reason": cls._global_cb_registry["global_cb_reason"],
            "strategies": {
                bot_id: {
                    "type": state["strategy_type"],
                    "cb_active": state["cb_active"],
                    "can_trade": state["can_trade"],
                    "reason": state.get("cb_reason"),
                }
                for bot_id, state in cls._global_cb_registry["strategies"].items()
            },
            "dependencies": cls._global_cb_registry["dependencies"],
        }

    # =========================================================================
    # Enhanced Circuit Breaker Trigger with Validation
    # =========================================================================

    async def trigger_circuit_breaker_safe(
        self,
        reason: str,
        current_price: Optional[Decimal] = None,
        current_capital: Optional[Decimal] = None,
        loss_pct: Optional[Decimal] = None,
        partial: bool = False,
    ) -> Dict[str, Any]:
        """
        Safely trigger circuit breaker with validation.

        This is the RECOMMENDED method to use instead of trigger_circuit_breaker.

        Args:
            reason: Reason for triggering
            current_price: Current market price
            current_capital: Current account capital
            loss_pct: Calculated loss percentage
            partial: Whether to trigger partial CB (strategy-only) vs full

        Returns:
            Dict with trigger result
        """
        result = {
            "triggered": False,
            "validated": False,
            "validation_message": None,
            "partial": partial,
        }

        # Validate the trigger
        should_trigger, validation_msg = self.validate_circuit_breaker_trigger(
            reason=reason,
            current_price=current_price,
            current_capital=current_capital,
            loss_pct=loss_pct,
        )

        result["validated"] = should_trigger
        result["validation_message"] = validation_msg

        if not should_trigger:
            logger.info(
                f"[{self._bot_id}] Circuit breaker NOT triggered: {validation_msg}"
            )
            return result

        # Trigger validated - proceed
        if partial and self.CB_ALLOW_PARTIAL:
            # Partial (strategy-only) circuit breaker
            cb_result = await self.trigger_partial_circuit_breaker(
                bot_id=self._bot_id,
                reason=reason,
                affect_dependents=self.CB_DEPENDENCY_CHECK,
            )
            result["triggered"] = True
            result["affected"] = cb_result

            # Also trigger local circuit breaker
            await self.trigger_circuit_breaker(reason=reason, force_close=False)

        else:
            # Full circuit breaker for this strategy
            cb_result = await self.trigger_circuit_breaker(
                reason=reason,
                force_close=True,
            )
            result["triggered"] = True
            result["details"] = cb_result

        return result

    async def check_can_trade_with_cb(self) -> tuple[bool, str]:
        """
        Check if trading is allowed considering all CB states.

        This combines local and global CB checks.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check local circuit breaker
        local_active, local_reason = self.is_circuit_breaker_active()
        if local_active:
            return False, local_reason

        # Check global/partial CB coordination
        global_can_trade, global_reason = await self.check_strategy_can_trade(self._bot_id)
        if not global_can_trade:
            return False, global_reason

        # Check recovery state
        recovery_mult = self.get_recovery_position_multiplier()
        if recovery_mult <= 0:
            return False, "Recovery position multiplier is 0"

        return True, "Trading allowed"

    def get_comprehensive_cb_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        self._init_cb_validation()
        self._init_cb_recovery()
        self._init_circuit_breaker()

        return {
            "local": {
                "is_triggered": self._circuit_breaker_state["is_triggered"],
                "trigger_reason": self._circuit_breaker_state["trigger_reason"],
                "lockout_remaining": (
                    (self._circuit_breaker_state["lockout_until"] - datetime.now(timezone.utc)).total_seconds() / 3600
                    if self._circuit_breaker_state["lockout_until"] and
                       datetime.now(timezone.utc) < self._circuit_breaker_state["lockout_until"]
                    else 0
                ),
            },
            "validation": {
                "pending_trigger": self._cb_validation_state["pending_trigger"],
                "confirmation_count": self._cb_validation_state["confirmation_count"],
                "false_positive_count": self._cb_validation_state["false_positive_count"],
            },
            "recovery": {
                "recovery_level": self._cb_recovery_state["recovery_level"],
                "position_multiplier": str(self._cb_recovery_state["recovery_position_multiplier"]),
                "manual_override": self._cb_recovery_state["manual_override"],
                "consecutive_profitable": self._cb_recovery_state["consecutive_profitable_trades"],
            },
            "global": self.get_cb_coordination_status() if hasattr(self, "_global_cb_registry") else None,
        }

    # =========================================================================
    # Network Resilience Management (網路彈性管理)
    # =========================================================================
    #
    # Handles:
    # 1. 網路斷線與交易所失去連接 - Network disconnection detection and recovery
    # 2. 網路抖動連接時斷時續 - Network jitter detection and smoothing
    # 3. DNS 解析失敗無法解析交易所域名 - DNS resolution failure with fallback
    #
    # =========================================================================

    # Network Health Constants
    NETWORK_HEALTH_CHECK_INTERVAL = 30  # Check every 30 seconds
    NETWORK_MAX_LATENCY_MS = 5000  # Max acceptable latency (5s)
    NETWORK_JITTER_THRESHOLD_MS = 500  # Jitter warning threshold
    NETWORK_JITTER_CRITICAL_MS = 2000  # Jitter critical threshold
    NETWORK_DISCONNECT_THRESHOLD = 3  # Consecutive failures before disconnect
    NETWORK_RECONNECT_MAX_ATTEMPTS = 100  # Max reconnection attempts (increased for WSL stability)
    NETWORK_RECONNECT_BASE_DELAY = 5  # Base delay for exponential backoff (seconds)
    NETWORK_RECONNECT_MAX_DELAY = 300  # Max delay (5 minutes)

    # DNS Constants
    DNS_CACHE_TTL_SECONDS = 300  # Cache DNS for 5 minutes
    DNS_RESOLVE_TIMEOUT = 10  # DNS resolution timeout (seconds)
    DNS_MAX_RETRIES = 3  # Max DNS resolution retries

    # Known exchange endpoints for DNS fallback
    # IPs must be verified real Binance IPs before use - empty = DNS-only (safe default)
    EXCHANGE_DNS_FALLBACKS: Dict[str, list] = {
        "api.binance.com": [],
        "fapi.binance.com": [],
        "stream.binance.com": [],
        "fstream.binance.com": [],
        "testnet.binance.vision": [],
        "testnet.binancefuture.com": [],
    }

    def _init_network_health(self) -> None:
        """Initialize network health tracking state."""
        if hasattr(self, "_network_health_initialized") and self._network_health_initialized:
            return

        self._network_health_state: Dict[str, Any] = {
            # Connection state
            "is_connected": True,
            "last_successful_request": datetime.now(timezone.utc),
            "last_failed_request": None,
            "consecutive_failures": 0,
            "consecutive_successes": 0,

            # Latency tracking
            "latency_samples": [],  # Last N latency measurements
            "avg_latency_ms": 0,
            "max_latency_ms": 0,
            "min_latency_ms": 0,

            # Jitter tracking (latency variance)
            "jitter_samples": [],
            "current_jitter_ms": 0,
            "jitter_state": "stable",  # stable, unstable, critical

            # Disconnection tracking
            "disconnect_count": 0,
            "last_disconnect_time": None,
            "total_downtime_seconds": 0,
            "current_downtime_start": None,

            # Reconnection state
            "reconnect_attempts": 0,
            "last_reconnect_attempt": None,
            "reconnect_delay": self.NETWORK_RECONNECT_BASE_DELAY,

            # Health score (0-100)
            "health_score": 100,
        }

        # DNS cache
        self._dns_cache: Dict[str, Dict[str, Any]] = {}

        # Network event history
        self._network_events: list = []
        self._max_network_events = 100

        self._network_health_initialized = True
        logger.debug(f"[{self._bot_id}] Network health tracking initialized")

    def record_network_request(
        self,
        success: bool,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record the result of a network request for health tracking.

        Args:
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
            error: Error message if failed
        """
        self._init_network_health()
        now = datetime.now(timezone.utc)
        state = self._network_health_state

        if success:
            state["last_successful_request"] = now
            state["consecutive_failures"] = 0
            state["consecutive_successes"] += 1

            # Track latency
            if latency_ms is not None:
                self._update_latency_tracking(latency_ms)

            # Check if recovering from disconnect
            if state["current_downtime_start"]:
                downtime = (now - state["current_downtime_start"]).total_seconds()
                state["total_downtime_seconds"] += downtime
                state["current_downtime_start"] = None

                logger.info(
                    f"[{self._bot_id}] Network connection restored after {downtime:.1f}s"
                )
                self._record_network_event("connection_restored", {
                    "downtime_seconds": downtime,
                })

                # Notify (tracked to prevent resource leak)
                if self._notifier:
                    self._create_notification_task(
                        self._notifier.notify_connection_restored(
                            service_name=f"{self.bot_type}:{self._bot_id}",
                            downtime=downtime,
                        )
                    )

            # Reset reconnection state on sustained success
            if state["consecutive_successes"] >= 5:
                state["reconnect_attempts"] = 0
                state["reconnect_delay"] = self.NETWORK_RECONNECT_BASE_DELAY

        else:
            state["last_failed_request"] = now
            state["consecutive_failures"] += 1
            state["consecutive_successes"] = 0

            self._record_network_event("request_failed", {
                "error": error,
                "consecutive_failures": state["consecutive_failures"],
            })

            # Check for disconnect threshold
            if state["consecutive_failures"] >= self.NETWORK_DISCONNECT_THRESHOLD:
                if not state["current_downtime_start"]:
                    state["is_connected"] = False
                    state["disconnect_count"] += 1
                    state["last_disconnect_time"] = now
                    state["current_downtime_start"] = now

                    logger.error(
                        f"[{self._bot_id}] Network disconnected "
                        f"(failures: {state['consecutive_failures']})"
                    )
                    self._record_network_event("disconnected", {
                        "error": error,
                        "disconnect_count": state["disconnect_count"],
                    })

                    # Notify (tracked to prevent resource leak)
                    if self._notifier:
                        self._create_notification_task(
                            self._notifier.notify_connection_lost(
                                service_name=f"{self.bot_type}:{self._bot_id}",
                                error=error,
                            )
                        )

        # Update health score
        self._update_network_health_score()

    def _update_latency_tracking(self, latency_ms: float) -> None:
        """Update latency and jitter tracking."""
        state = self._network_health_state
        samples = state["latency_samples"]

        # Keep last 20 samples
        samples.append(latency_ms)
        if len(samples) > 20:
            samples.pop(0)

        # Update statistics
        if samples:
            state["avg_latency_ms"] = sum(samples) / len(samples)
            state["max_latency_ms"] = max(samples)
            state["min_latency_ms"] = min(samples)

            # Calculate jitter (variance in latency)
            if len(samples) >= 2:
                jitter = abs(samples[-1] - samples[-2])
                jitter_samples = state["jitter_samples"]
                jitter_samples.append(jitter)
                if len(jitter_samples) > 20:
                    jitter_samples.pop(0)

                state["current_jitter_ms"] = sum(jitter_samples) / len(jitter_samples)

                # Classify jitter state
                if state["current_jitter_ms"] > self.NETWORK_JITTER_CRITICAL_MS:
                    if state["jitter_state"] != "critical":
                        state["jitter_state"] = "critical"
                        logger.warning(
                            f"[{self._bot_id}] Network jitter CRITICAL: "
                            f"{state['current_jitter_ms']:.0f}ms"
                        )
                        self._record_network_event("jitter_critical", {
                            "jitter_ms": state["current_jitter_ms"],
                        })
                elif state["current_jitter_ms"] > self.NETWORK_JITTER_THRESHOLD_MS:
                    if state["jitter_state"] != "unstable":
                        state["jitter_state"] = "unstable"
                        logger.info(
                            f"[{self._bot_id}] Network jitter unstable: "
                            f"{state['current_jitter_ms']:.0f}ms"
                        )
                else:
                    state["jitter_state"] = "stable"

    def _update_network_health_score(self) -> None:
        """Calculate overall network health score (0-100)."""
        state = self._network_health_state
        score = 100

        # Deduct for disconnection
        if not state["is_connected"]:
            score -= 50

        # Deduct for consecutive failures
        score -= min(state["consecutive_failures"] * 10, 30)

        # Deduct for high latency
        if state["avg_latency_ms"] > self.NETWORK_MAX_LATENCY_MS:
            score -= 20
        elif state["avg_latency_ms"] > self.NETWORK_MAX_LATENCY_MS / 2:
            score -= 10

        # Deduct for jitter
        if state["jitter_state"] == "critical":
            score -= 20
        elif state["jitter_state"] == "unstable":
            score -= 10

        # Deduct for recent disconnects
        if state["disconnect_count"] > 0:
            # Reduce impact over time (last 24h)
            recent_disconnects = min(state["disconnect_count"], 10)
            score -= recent_disconnects * 2

        state["health_score"] = max(0, min(100, score))

    def _record_network_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record a network event for history."""
        self._init_network_health()

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "details": details,
        }

        self._network_events.append(event)
        if len(self._network_events) > self._max_network_events:
            self._network_events.pop(0)

    def is_network_healthy(self) -> tuple[bool, str]:
        """
        Check if network connection is healthy for trading.

        Returns:
            Tuple of (is_healthy, reason)
        """
        self._init_network_health()
        state = self._network_health_state

        # Check connection state
        if not state["is_connected"]:
            return False, f"Network disconnected (failures: {state['consecutive_failures']})"

        # Check latency
        if state["avg_latency_ms"] > self.NETWORK_MAX_LATENCY_MS:
            return False, f"Network latency too high: {state['avg_latency_ms']:.0f}ms"

        # Check jitter (unstable connection)
        if state["jitter_state"] == "critical":
            return False, f"Network jitter critical: {state['current_jitter_ms']:.0f}ms - connection unstable"

        # Check recent activity
        if state["last_successful_request"]:
            silence = (datetime.now(timezone.utc) - state["last_successful_request"]).total_seconds()
            if silence > 300:  # 5 minutes
                return False, f"No successful requests for {silence:.0f}s"

        # Check health score
        if state["health_score"] < 30:
            return False, f"Network health score too low: {state['health_score']}"

        return True, "Network healthy"

    async def attempt_network_reconnect(self) -> bool:
        """
        Attempt to reconnect to the exchange.

        Implements exponential backoff with jitter.

        Returns:
            True if reconnection successful
        """
        self._init_network_health()
        state = self._network_health_state

        # Check max attempts
        if state["reconnect_attempts"] >= self.NETWORK_RECONNECT_MAX_ATTEMPTS:
            logger.error(
                f"[{self._bot_id}] Max reconnection attempts "
                f"({self.NETWORK_RECONNECT_MAX_ATTEMPTS}) reached"
            )
            return False

        state["reconnect_attempts"] += 1
        state["last_reconnect_attempt"] = datetime.now(timezone.utc)

        # Calculate delay with jitter
        import random
        jitter = random.uniform(0.5, 1.5)
        delay = min(
            state["reconnect_delay"] * jitter,
            self.NETWORK_RECONNECT_MAX_DELAY
        )

        logger.info(
            f"[{self._bot_id}] Reconnection attempt {state['reconnect_attempts']}/"
            f"{self.NETWORK_RECONNECT_MAX_ATTEMPTS} in {delay:.1f}s"
        )

        await asyncio.sleep(delay)

        # Attempt reconnection via exchange client
        try:
            # Try DNS resolution first
            dns_ok, dns_error = await self._verify_dns_resolution()
            if not dns_ok:
                logger.warning(f"[{self._bot_id}] DNS verification failed: {dns_error}")
                # Try fallback IPs
                await self._try_dns_fallback()

            # Attempt connection test
            if hasattr(self._exchange, 'get_price'):
                start_time = time.time()
                try:
                    # Simple ping: get price
                    symbol = getattr(self._config, 'symbol', 'BTCUSDT')
                    await asyncio.wait_for(
                        self._exchange.get_price(symbol),
                        timeout=30.0
                    )
                    latency_ms = (time.time() - start_time) * 1000

                    # Success!
                    self.record_network_request(True, latency_ms)
                    state["is_connected"] = True

                    logger.info(
                        f"[{self._bot_id}] Reconnection successful "
                        f"(latency: {latency_ms:.0f}ms)"
                    )
                    self._record_network_event("reconnected", {
                        "attempt": state["reconnect_attempts"],
                        "latency_ms": latency_ms,
                    })

                    # Reset delay on success
                    state["reconnect_delay"] = self.NETWORK_RECONNECT_BASE_DELAY
                    return True

                except asyncio.TimeoutError:
                    logger.warning(f"[{self._bot_id}] Reconnection timed out")
                    self.record_network_request(False, error="timeout")

                except Exception as e:
                    logger.warning(f"[{self._bot_id}] Reconnection failed: {e}")
                    self.record_network_request(False, error=str(e))

            # Exponential backoff
            state["reconnect_delay"] = min(
                state["reconnect_delay"] * 2,
                self.NETWORK_RECONNECT_MAX_DELAY
            )
            return False

        except Exception as e:
            logger.error(f"[{self._bot_id}] Reconnection error: {e}")
            state["reconnect_delay"] = min(
                state["reconnect_delay"] * 2,
                self.NETWORK_RECONNECT_MAX_DELAY
            )
            return False

    # =========================================================================
    # DNS Resolution Management
    # =========================================================================

    async def _verify_dns_resolution(self) -> tuple[bool, Optional[str]]:
        """
        Verify DNS resolution for exchange endpoints.

        Returns:
            Tuple of (success, error_message)
        """
        import socket

        endpoints_to_check = ["api.binance.com", "fapi.binance.com"]

        for endpoint in endpoints_to_check:
            try:
                # Check cache first
                cached = self._get_cached_dns(endpoint)
                if cached:
                    continue

                # Resolve DNS
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, socket.gethostbyname, endpoint),
                    timeout=self.DNS_RESOLVE_TIMEOUT
                )

                # Cache the result
                self._cache_dns(endpoint, result)
                logger.debug(f"[{self._bot_id}] DNS resolved {endpoint} -> {result}")

            except socket.gaierror as e:
                error_msg = f"DNS resolution failed for {endpoint}: {e}"
                logger.error(f"[{self._bot_id}] {error_msg}")
                self._record_network_event("dns_failure", {
                    "endpoint": endpoint,
                    "error": str(e),
                })
                return False, error_msg

            except asyncio.TimeoutError:
                error_msg = f"DNS resolution timeout for {endpoint}"
                logger.error(f"[{self._bot_id}] {error_msg}")
                self._record_network_event("dns_timeout", {
                    "endpoint": endpoint,
                })
                return False, error_msg

            except Exception as e:
                error_msg = f"DNS error for {endpoint}: {e}"
                logger.error(f"[{self._bot_id}] {error_msg}")
                return False, error_msg

        return True, None

    def _get_cached_dns(self, hostname: str) -> Optional[str]:
        """Get DNS result from cache if valid."""
        self._init_network_health()

        if hostname in self._dns_cache:
            cache_entry = self._dns_cache[hostname]
            cache_time = datetime.fromisoformat(cache_entry["cached_at"])
            age = (datetime.now(timezone.utc) - cache_time).total_seconds()

            if age < self.DNS_CACHE_TTL_SECONDS:
                return cache_entry["ip"]

            # Cache expired
            del self._dns_cache[hostname]

        return None

    def _cache_dns(self, hostname: str, ip: str) -> None:
        """Cache a DNS resolution result."""
        self._init_network_health()

        self._dns_cache[hostname] = {
            "ip": ip,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _try_dns_fallback(self) -> bool:
        """
        Try to use fallback IP addresses when DNS fails.

        Returns:
            True if fallback connection successful
        """
        import socket

        for hostname, fallback_ips in self.EXCHANGE_DNS_FALLBACKS.items():
            if not fallback_ips:
                continue

            for ip in fallback_ips:
                try:
                    # Test connection to fallback IP
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((ip, 443))
                    sock.close()

                    if result == 0:
                        logger.info(
                            f"[{self._bot_id}] DNS fallback successful: "
                            f"{hostname} -> {ip}"
                        )
                        self._cache_dns(hostname, ip)
                        self._record_network_event("dns_fallback_success", {
                            "hostname": hostname,
                            "ip": ip,
                        })
                        return True

                except Exception as e:
                    logger.debug(f"[{self._bot_id}] Fallback IP {ip} failed: {e}")
                    continue

        logger.error(f"[{self._bot_id}] All DNS fallback IPs failed")
        return False

    async def resolve_dns_with_retry(self, hostname: str) -> Optional[str]:
        """
        Resolve DNS with retry and fallback.

        Args:
            hostname: Hostname to resolve

        Returns:
            IP address or None if failed
        """
        import socket

        # Check cache first
        cached = self._get_cached_dns(hostname)
        if cached:
            return cached

        # Try resolution with retries
        for attempt in range(self.DNS_MAX_RETRIES):
            try:
                loop = asyncio.get_event_loop()
                ip = await asyncio.wait_for(
                    loop.run_in_executor(None, socket.gethostbyname, hostname),
                    timeout=self.DNS_RESOLVE_TIMEOUT
                )

                self._cache_dns(hostname, ip)
                return ip

            except (socket.gaierror, asyncio.TimeoutError) as e:
                logger.warning(
                    f"[{self._bot_id}] DNS resolution attempt {attempt + 1}/"
                    f"{self.DNS_MAX_RETRIES} failed for {hostname}: {e}"
                )

                if attempt < self.DNS_MAX_RETRIES - 1:
                    await asyncio.sleep(1)  # Brief delay between retries

        # Try fallback IPs
        if hostname in self.EXCHANGE_DNS_FALLBACKS:
            for ip in self.EXCHANGE_DNS_FALLBACKS[hostname]:
                try:
                    # Quick connectivity test
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((ip, 443))
                    sock.close()

                    if result == 0:
                        logger.info(
                            f"[{self._bot_id}] Using DNS fallback: {hostname} -> {ip}"
                        )
                        self._cache_dns(hostname, ip)
                        return ip

                except Exception:
                    continue

        logger.error(f"[{self._bot_id}] DNS resolution failed for {hostname}")
        return None

    # =========================================================================
    # Network Jitter Smoothing
    # =========================================================================

    def should_delay_for_jitter(self) -> tuple[bool, float]:
        """
        Check if operation should be delayed due to network jitter.

        Returns:
            Tuple of (should_delay, recommended_delay_seconds)
        """
        self._init_network_health()
        state = self._network_health_state

        if state["jitter_state"] == "critical":
            # Critical jitter: wait for stabilization
            return True, 5.0

        if state["jitter_state"] == "unstable":
            # Unstable: add small delay
            return True, 1.0

        return False, 0.0

    async def wait_for_network_stability(
        self,
        timeout_seconds: float = 60.0,
        check_interval: float = 2.0,
    ) -> bool:
        """
        Wait for network to stabilize before proceeding.

        Args:
            timeout_seconds: Maximum wait time
            check_interval: How often to check

        Returns:
            True if network stabilized, False if timeout
        """
        self._init_network_health()
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            is_healthy, reason = self.is_network_healthy()

            if is_healthy:
                state = self._network_health_state
                if state["jitter_state"] == "stable":
                    return True

            await asyncio.sleep(check_interval)

        logger.warning(
            f"[{self._bot_id}] Network stability timeout after {timeout_seconds}s"
        )
        return False

    # =========================================================================
    # Comprehensive Network Status
    # =========================================================================

    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        self._init_network_health()
        state = self._network_health_state

        is_healthy, health_reason = self.is_network_healthy()

        return {
            "is_healthy": is_healthy,
            "health_reason": health_reason,
            "health_score": state["health_score"],
            "connection": {
                "is_connected": state["is_connected"],
                "consecutive_failures": state["consecutive_failures"],
                "consecutive_successes": state["consecutive_successes"],
                "disconnect_count": state["disconnect_count"],
                "total_downtime_seconds": state["total_downtime_seconds"],
            },
            "latency": {
                "avg_ms": round(state["avg_latency_ms"], 1),
                "max_ms": round(state["max_latency_ms"], 1),
                "min_ms": round(state["min_latency_ms"], 1),
            },
            "jitter": {
                "current_ms": round(state["current_jitter_ms"], 1),
                "state": state["jitter_state"],
            },
            "reconnection": {
                "attempts": state["reconnect_attempts"],
                "max_attempts": self.NETWORK_RECONNECT_MAX_ATTEMPTS,
                "current_delay": state["reconnect_delay"],
            },
            "dns_cache_size": len(self._dns_cache),
            "recent_events": self._network_events[-5:] if self._network_events else [],
        }

    async def check_network_before_trade(self) -> tuple[bool, str]:
        """
        Comprehensive network check before executing a trade.

        Combines all network health checks.

        Returns:
            Tuple of (can_trade, reason)
        """
        self._init_network_health()

        # 1. Basic network health
        is_healthy, reason = self.is_network_healthy()
        if not is_healthy:
            return False, f"Network unhealthy: {reason}"

        # 2. Check for jitter delay
        should_delay, delay_seconds = self.should_delay_for_jitter()
        if should_delay and delay_seconds > 3.0:
            return False, f"Network jitter too high, recommended delay: {delay_seconds}s"

        # 3. DNS verification (periodic)
        state = self._network_health_state
        if state["consecutive_successes"] == 0:
            # No recent success, verify DNS
            dns_ok, dns_error = await self._verify_dns_resolution()
            if not dns_ok:
                return False, f"DNS resolution failed: {dns_error}"

        # 4. Data connection health (from existing method)
        if not self._is_data_connection_healthy():
            return False, "Data connection unhealthy"

        return True, "Network OK"

    async def handle_network_error(
        self,
        error: Exception,
        operation: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Handle a network error with appropriate recovery actions.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed

        Returns:
            Dict with handling result and recommended action
        """
        self._init_network_health()
        error_str = str(error).lower()

        result = {
            "handled": False,
            "action": "none",
            "should_retry": False,
            "retry_delay": 0,
            "error_type": "unknown",
        }

        # Classify error type
        if "timeout" in error_str:
            result["error_type"] = "timeout"
            result["should_retry"] = True
            result["retry_delay"] = 2
        elif "connection" in error_str or "refused" in error_str:
            result["error_type"] = "connection"
            result["should_retry"] = True
            result["retry_delay"] = 5
        elif "dns" in error_str or "resolve" in error_str or "getaddrinfo" in error_str:
            result["error_type"] = "dns"
            result["should_retry"] = True
            result["retry_delay"] = 10
            result["action"] = "verify_dns"
        elif "ssl" in error_str or "certificate" in error_str:
            result["error_type"] = "ssl"
            result["should_retry"] = False
            result["action"] = "check_ssl"
        else:
            result["error_type"] = "other"
            result["should_retry"] = True
            result["retry_delay"] = 1

        # Record the failure
        self.record_network_request(False, error=f"{operation}: {error}")

        # Take action based on error type
        if result["error_type"] == "dns":
            # Try DNS fallback
            await self._try_dns_fallback()

        # Check if we need to attempt reconnection
        state = self._network_health_state
        if not state["is_connected"]:
            result["action"] = "reconnect"
            result["should_retry"] = False  # Don't retry, need full reconnect

        result["handled"] = True

        logger.info(
            f"[{self._bot_id}] Network error handled: type={result['error_type']}, "
            f"action={result['action']}, retry={result['should_retry']}"
        )

        return result

    # =========================================================================
    # SSL Certificate Resilience Management (SSL 證書彈性管理)
    # =========================================================================
    #
    # Handles:
    # - SSL 證書問題證書過期或不匹配
    # - Certificate validation and expiry monitoring
    # - SSL error handling and recovery
    #
    # =========================================================================

    # SSL Constants
    SSL_CHECK_INTERVAL_HOURS = 24  # Check certificate every 24 hours
    SSL_EXPIRY_WARNING_DAYS = 30  # Warn 30 days before expiry
    SSL_EXPIRY_CRITICAL_DAYS = 7  # Critical alert 7 days before expiry
    SSL_ERROR_RETRY_DELAY = 30  # Seconds to wait before retry on SSL error
    SSL_MAX_CONSECUTIVE_ERRORS = 5  # Max errors before alerting

    # Known exchange certificate info (for validation)
    EXCHANGE_CERT_FINGERPRINTS: Dict[str, list] = {
        # These should be updated with actual fingerprints
        "api.binance.com": [],
        "fapi.binance.com": [],
        "stream.binance.com": [],
        "fstream.binance.com": [],
    }

    def _init_ssl_health(self) -> None:
        """Initialize SSL certificate health tracking state."""
        if hasattr(self, "_ssl_health_initialized") and self._ssl_health_initialized:
            return

        self._ssl_health_state: Dict[str, Any] = {
            # Certificate state
            "last_check_time": None,
            "certificate_valid": True,
            "certificate_expiry": None,
            "days_until_expiry": None,

            # Error tracking
            "consecutive_ssl_errors": 0,
            "last_ssl_error": None,
            "last_ssl_error_time": None,
            "total_ssl_errors": 0,

            # Certificate info cache
            "cert_info": {},  # hostname -> cert info

            # Hostname verification
            "hostname_mismatch_detected": False,
            "last_verified_hostname": None,
        }

        # SSL event history
        self._ssl_events: list = []
        self._max_ssl_events = 50

        self._ssl_health_initialized = True
        logger.debug(f"[{self._bot_id}] SSL health tracking initialized")

    def _record_ssl_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record an SSL-related event for history."""
        self._init_ssl_health()

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "details": details,
        }

        self._ssl_events.append(event)
        if len(self._ssl_events) > self._max_ssl_events:
            self._ssl_events.pop(0)

    async def check_ssl_certificate(
        self,
        hostname: str = "api.binance.com",
        port: int = 443,
    ) -> Dict[str, Any]:
        """
        Check SSL certificate validity for a hostname.

        Args:
            hostname: Hostname to check
            port: Port number (default 443)

        Returns:
            Dict with certificate status and details
        """
        import ssl
        import socket

        self._init_ssl_health()
        state = self._ssl_health_state

        result = {
            "valid": False,
            "hostname": hostname,
            "error": None,
            "expiry_date": None,
            "days_until_expiry": None,
            "issuer": None,
            "subject": None,
            "hostname_match": True,
        }

        try:
            # Create SSL context
            context = ssl.create_default_context()

            # Connect and get certificate (run in executor to avoid blocking event loop)
            loop = asyncio.get_event_loop()
            cert = await loop.run_in_executor(None, self._fetch_ssl_cert_sync, hostname, port, context)

            if cert:
                # Parse certificate info
                result["valid"] = True

                # Get expiry date
                not_after = cert.get("notAfter")
                if not_after:
                    # Parse the date string
                    from datetime import datetime as dt
                    try:
                        # Format: 'Mar 15 12:00:00 2024 GMT'
                        expiry = dt.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                        expiry = expiry.replace(tzinfo=timezone.utc)
                        result["expiry_date"] = expiry.isoformat()

                        days_left = (expiry - datetime.now(timezone.utc)).days
                        result["days_until_expiry"] = days_left

                        state["certificate_expiry"] = expiry
                        state["days_until_expiry"] = days_left

                        # Check expiry warnings
                        if days_left <= 0:
                            result["valid"] = False
                            result["error"] = "Certificate expired"
                            self._record_ssl_event("certificate_expired", {
                                "hostname": hostname,
                                "expiry": expiry.isoformat(),
                            })
                        elif days_left <= self.SSL_EXPIRY_CRITICAL_DAYS:
                            logger.error(
                                f"[{self._bot_id}] CRITICAL: SSL certificate for {hostname} "
                                f"expires in {days_left} days!"
                            )
                            self._record_ssl_event("certificate_expiry_critical", {
                                "hostname": hostname,
                                "days_left": days_left,
                            })
                        elif days_left <= self.SSL_EXPIRY_WARNING_DAYS:
                            logger.warning(
                                f"[{self._bot_id}] SSL certificate for {hostname} "
                                f"expires in {days_left} days"
                            )
                            self._record_ssl_event("certificate_expiry_warning", {
                                "hostname": hostname,
                                "days_left": days_left,
                            })

                    except ValueError as e:
                        logger.warning(f"Could not parse cert date: {not_after}")

                # Get issuer
                issuer = cert.get("issuer")
                if issuer:
                    issuer_dict = {k: v for ((k, v),) in issuer}
                    result["issuer"] = issuer_dict.get("organizationName", "Unknown")

                # Get subject
                subject = cert.get("subject")
                if subject:
                    subject_dict = {k: v for ((k, v),) in subject}
                    result["subject"] = subject_dict.get("commonName", "Unknown")

                # Verify hostname match
                san = cert.get("subjectAltName", [])
                hostnames = [name for (typ, name) in san if typ == "DNS"]
                cn = subject_dict.get("commonName", "") if subject else ""

                hostname_match = (
                    hostname in hostnames or
                    hostname == cn or
                    any(self._match_wildcard(hostname, h) for h in hostnames)
                )

                if not hostname_match:
                    result["hostname_match"] = False
                    result["error"] = f"Hostname mismatch: {hostname} not in {hostnames}"
                    state["hostname_mismatch_detected"] = True
                    self._record_ssl_event("hostname_mismatch", {
                        "hostname": hostname,
                        "cert_names": hostnames,
                    })
                else:
                    state["hostname_mismatch_detected"] = False

                # Cache cert info
                state["cert_info"][hostname] = {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "expiry": result["expiry_date"],
                    "issuer": result["issuer"],
                    "valid": result["valid"],
                }

                # Reset error count on success
                state["consecutive_ssl_errors"] = 0
                state["certificate_valid"] = result["valid"]
                state["last_check_time"] = datetime.now(timezone.utc)
                state["last_verified_hostname"] = hostname

        except ssl.SSLCertVerificationError as e:
            result["error"] = f"Certificate verification failed: {e}"
            state["consecutive_ssl_errors"] += 1
            state["last_ssl_error"] = str(e)
            state["last_ssl_error_time"] = datetime.now(timezone.utc)
            state["total_ssl_errors"] += 1
            state["certificate_valid"] = False

            self._record_ssl_event("ssl_verification_error", {
                "hostname": hostname,
                "error": str(e),
            })

            logger.error(f"[{self._bot_id}] SSL verification error for {hostname}: {e}")

        except ssl.SSLError as e:
            result["error"] = f"SSL error: {e}"
            state["consecutive_ssl_errors"] += 1
            state["last_ssl_error"] = str(e)
            state["last_ssl_error_time"] = datetime.now(timezone.utc)
            state["total_ssl_errors"] += 1

            self._record_ssl_event("ssl_error", {
                "hostname": hostname,
                "error": str(e),
            })

            logger.error(f"[{self._bot_id}] SSL error for {hostname}: {e}")

        except socket.timeout:
            result["error"] = "Connection timeout"
            self._record_ssl_event("ssl_check_timeout", {"hostname": hostname})

        except Exception as e:
            result["error"] = f"Check failed: {e}"
            state["consecutive_ssl_errors"] += 1
            state["last_ssl_error"] = str(e)
            state["last_ssl_error_time"] = datetime.now(timezone.utc)
            state["total_ssl_errors"] += 1

            self._record_ssl_event("ssl_check_failed", {
                "hostname": hostname,
                "error": str(e),
            })

        return result

    @staticmethod
    def _fetch_ssl_cert_sync(hostname: str, port: int, context) -> Optional[dict]:
        """Fetch SSL certificate synchronously (meant to be called via run_in_executor)."""
        import ssl
        import socket
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return ssock.getpeercert()

    def _match_wildcard(self, hostname: str, pattern: str) -> bool:
        """Match hostname against wildcard pattern (e.g., *.binance.com)."""
        if not pattern.startswith("*."):
            return False

        pattern_suffix = pattern[2:]  # Remove "*."
        hostname_parts = hostname.split(".")
        pattern_parts = pattern_suffix.split(".")

        if len(hostname_parts) != len(pattern_parts) + 1:
            return False

        return ".".join(hostname_parts[1:]) == pattern_suffix

    async def verify_exchange_ssl(self) -> Dict[str, Any]:
        """
        Verify SSL certificates for all exchange endpoints.

        Returns:
            Dict with verification results for each endpoint
        """
        endpoints = [
            "api.binance.com",
            "fapi.binance.com",
        ]

        results = {
            "all_valid": True,
            "endpoints": {},
            "errors": [],
        }

        for endpoint in endpoints:
            check_result = await self.check_ssl_certificate(endpoint)
            results["endpoints"][endpoint] = check_result

            if not check_result["valid"]:
                results["all_valid"] = False
                results["errors"].append({
                    "endpoint": endpoint,
                    "error": check_result["error"],
                })

        return results

    def is_ssl_healthy(self) -> tuple[bool, str]:
        """
        Check if SSL status is healthy for trading.

        Returns:
            Tuple of (is_healthy, reason)
        """
        self._init_ssl_health()
        state = self._ssl_health_state

        # Check for consecutive errors
        if state["consecutive_ssl_errors"] >= self.SSL_MAX_CONSECUTIVE_ERRORS:
            return False, f"Too many SSL errors: {state['consecutive_ssl_errors']}"

        # Check certificate validity
        if not state["certificate_valid"]:
            return False, "SSL certificate invalid or verification failed"

        # Check hostname mismatch
        if state["hostname_mismatch_detected"]:
            return False, "SSL hostname mismatch detected"

        # Check expiry
        if state["days_until_expiry"] is not None:
            if state["days_until_expiry"] <= 0:
                return False, "SSL certificate expired"
            if state["days_until_expiry"] <= self.SSL_EXPIRY_CRITICAL_DAYS:
                return False, f"SSL certificate expires in {state['days_until_expiry']} days (critical)"

        return True, "SSL healthy"

    async def handle_ssl_error(
        self,
        error: Exception,
        operation: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Handle an SSL error with appropriate recovery actions.

        Args:
            error: The SSL exception that occurred
            operation: Description of the operation that failed

        Returns:
            Dict with handling result and recommended action
        """
        import ssl

        self._init_ssl_health()
        state = self._ssl_health_state
        error_str = str(error).lower()

        result = {
            "handled": False,
            "error_type": "unknown",
            "should_retry": False,
            "retry_delay": self.SSL_ERROR_RETRY_DELAY,
            "action": "none",
            "needs_manual_intervention": False,
        }

        # Classify SSL error type
        if "certificate verify failed" in error_str or "certificate_verify_failed" in error_str:
            result["error_type"] = "cert_verification"
            result["should_retry"] = False
            result["action"] = "verify_certificate"
            result["needs_manual_intervention"] = True

        elif "certificate has expired" in error_str or "certificate_expired" in error_str:
            result["error_type"] = "cert_expired"
            result["should_retry"] = False
            result["action"] = "check_system_time"
            result["needs_manual_intervention"] = True

        elif "hostname" in error_str and ("mismatch" in error_str or "doesn't match" in error_str):
            result["error_type"] = "hostname_mismatch"
            result["should_retry"] = False
            result["action"] = "verify_hostname"
            result["needs_manual_intervention"] = True

        elif "handshake" in error_str or "handshake_failure" in error_str:
            result["error_type"] = "handshake"
            result["should_retry"] = True
            result["retry_delay"] = 10
            result["action"] = "retry_connection"

        elif "protocol" in error_str or "version" in error_str:
            result["error_type"] = "protocol"
            result["should_retry"] = False
            result["action"] = "check_tls_version"

        elif "connection reset" in error_str or "connection refused" in error_str:
            result["error_type"] = "connection"
            result["should_retry"] = True
            result["retry_delay"] = 5
            result["action"] = "retry_connection"

        else:
            result["error_type"] = "other"
            result["should_retry"] = True
            result["retry_delay"] = 15

        # Update state
        state["consecutive_ssl_errors"] += 1
        state["last_ssl_error"] = str(error)
        state["last_ssl_error_time"] = datetime.now(timezone.utc)
        state["total_ssl_errors"] += 1

        # Record event
        self._record_ssl_event("ssl_error_handled", {
            "operation": operation,
            "error_type": result["error_type"],
            "error": str(error)[:200],
            "action": result["action"],
        })

        # Alert if too many errors
        if state["consecutive_ssl_errors"] >= self.SSL_MAX_CONSECUTIVE_ERRORS:
            logger.error(
                f"[{self._bot_id}] SSL error threshold reached "
                f"({state['consecutive_ssl_errors']} errors)"
            )
            if self._notifier:
                self._create_notification_task(
                    self._notifier.send_error(
                        title=f"{self.bot_type}: SSL Error Alert",
                        message=(
                            f"Multiple SSL errors detected.\n"
                            f"Error type: {result['error_type']}\n"
                            f"Action needed: {result['action']}"
                        ),
                    )
                )

        result["handled"] = True

        logger.warning(
            f"[{self._bot_id}] SSL error handled: type={result['error_type']}, "
            f"action={result['action']}, retry={result['should_retry']}"
        )

        return result

    async def check_ssl_before_trade(self) -> tuple[bool, str]:
        """
        Check SSL status before executing a trade.

        Returns:
            Tuple of (can_trade, reason)
        """
        self._init_ssl_health()
        state = self._ssl_health_state

        # Check basic SSL health
        is_healthy, reason = self.is_ssl_healthy()
        if not is_healthy:
            return False, f"SSL unhealthy: {reason}"

        # If haven't checked recently, do a quick check
        if state["last_check_time"] is None:
            # First check
            result = await self.check_ssl_certificate()
            if not result["valid"]:
                return False, f"SSL certificate check failed: {result['error']}"
        else:
            # Check if need to re-verify
            hours_since_check = (
                datetime.now(timezone.utc) - state["last_check_time"]
            ).total_seconds() / 3600

            if hours_since_check > self.SSL_CHECK_INTERVAL_HOURS:
                result = await self.check_ssl_certificate()
                if not result["valid"]:
                    return False, f"SSL certificate check failed: {result['error']}"

        return True, "SSL OK"

    def get_ssl_status(self) -> Dict[str, Any]:
        """Get comprehensive SSL status."""
        self._init_ssl_health()
        state = self._ssl_health_state

        is_healthy, health_reason = self.is_ssl_healthy()

        return {
            "is_healthy": is_healthy,
            "health_reason": health_reason,
            "certificate": {
                "valid": state["certificate_valid"],
                "expiry": state["certificate_expiry"].isoformat() if state["certificate_expiry"] else None,
                "days_until_expiry": state["days_until_expiry"],
                "hostname_mismatch": state["hostname_mismatch_detected"],
            },
            "errors": {
                "consecutive": state["consecutive_ssl_errors"],
                "total": state["total_ssl_errors"],
                "last_error": state["last_ssl_error"],
                "last_error_time": state["last_ssl_error_time"].isoformat() if state["last_ssl_error_time"] else None,
            },
            "last_check": state["last_check_time"].isoformat() if state["last_check_time"] else None,
            "cert_cache": state["cert_info"],
            "recent_events": self._ssl_events[-5:] if self._ssl_events else [],
        }

    async def monitor_ssl_expiry(self) -> Optional[Dict[str, Any]]:
        """
        Monitor SSL certificate expiry and send alerts.

        Should be called periodically (e.g., daily).

        Returns:
            Dict with expiry status or None if no issues
        """
        result = await self.verify_exchange_ssl()

        alerts = []
        for endpoint, check in result["endpoints"].items():
            days_left = check.get("days_until_expiry")

            if days_left is not None:
                if days_left <= 0:
                    alerts.append({
                        "endpoint": endpoint,
                        "level": "EXPIRED",
                        "days_left": days_left,
                    })
                elif days_left <= self.SSL_EXPIRY_CRITICAL_DAYS:
                    alerts.append({
                        "endpoint": endpoint,
                        "level": "CRITICAL",
                        "days_left": days_left,
                    })
                elif days_left <= self.SSL_EXPIRY_WARNING_DAYS:
                    alerts.append({
                        "endpoint": endpoint,
                        "level": "WARNING",
                        "days_left": days_left,
                    })

        if alerts and self._notifier:
            alert_text = "\n".join([
                f"{a['endpoint']}: {a['level']} - {a['days_left']} days"
                for a in alerts
            ])
            await self._notifier.send_warning(
                title="SSL Certificate Expiry Alert",
                message=f"Certificate expiry detected:\n{alert_text}",
            )

        return {"alerts": alerts} if alerts else None
