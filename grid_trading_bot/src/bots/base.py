"""
Base Bot Abstract Class.

Provides unified interface for all trading bots, enabling Master to manage
different bot types consistently.

Design Pattern: Template Method
- Public methods (start, stop, pause, resume) define the algorithm skeleton
- Abstract methods (_do_start, _do_stop, etc.) let subclasses implement specifics
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

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

            if not levels or len(levels) == 0:
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

        Returns:
            Exchange position data or None if no position
        """
        try:
            positions = await self._exchange.get_positions(self.symbol)

            for pos in positions:
                if pos.symbol == self.symbol and pos.quantity != Decimal("0"):
                    position_data = {
                        "symbol": pos.symbol,
                        "quantity": abs(pos.quantity),
                        "side": "LONG" if pos.quantity > 0 else "SHORT",
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

            # Call subclass implementation
            await self._do_start()

            # Update state
            self._state = BotState.RUNNING
            self._running = True

            # Start heartbeat
            self._start_heartbeat()

            logger.info(f"Bot {self._bot_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start bot {self._bot_id}: {e}")
            self._state = BotState.ERROR
            self._error_message = str(e)
            raise

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
            self._state = BotState.STOPPING
            self._running = False

            # Stop heartbeat
            self._stop_heartbeat()

            # Call subclass implementation
            await self._do_stop(clear_position)

            # Update state
            self._state = BotState.STOPPED

            logger.info(f"Bot {self._bot_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping bot {self._bot_id}: {e}")
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

            # Update state
            self._state = BotState.PAUSED

            logger.info(f"Bot {self._bot_id} paused")
            return True

        except Exception as e:
            logger.error(f"Error pausing bot {self._bot_id}: {e}")
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
            self._state = BotState.RUNNING

            logger.info(f"Bot {self._bot_id} resumed")
            return True

        except Exception as e:
            logger.error(f"Error resuming bot {self._bot_id}: {e}")
            self._state = BotState.ERROR
            self._error_message = str(e)
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

    def _stop_heartbeat(self) -> None:
        """Stop heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
            logger.debug(f"Heartbeat stopped for {self._bot_id}")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop - sends heartbeat every 10 seconds."""
        while self._running:
            try:
                await asyncio.sleep(10)
                if self._running and self._heartbeat_callback:
                    self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    def _send_heartbeat(self) -> None:
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
            self._heartbeat_callback(heartbeat)
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
        except Exception:
            return False

    # =========================================================================
    # Capital Management
    # =========================================================================

    async def update_capital(self, new_max_capital: Decimal) -> bool:
        """
        Update maximum capital allocation for this bot.

        Called by FundManager when capital allocation changes.
        Subclasses can override to implement specific behavior.

        Args:
            new_max_capital: New maximum capital amount

        Returns:
            True if update was successful
        """
        try:
            # Store previous value for logging
            previous = getattr(self._config, "max_capital", None)

            # Update config if it has max_capital attribute
            if hasattr(self._config, "max_capital"):
                self._config.max_capital = new_max_capital
                logger.info(
                    f"Bot {self._bot_id} capital updated: "
                    f"{previous} -> {new_max_capital}"
                )

            # Notify subclass of capital change (they can override _on_capital_updated)
            await self._on_capital_updated(new_max_capital)

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
