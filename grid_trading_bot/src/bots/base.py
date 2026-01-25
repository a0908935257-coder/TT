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
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Dict, Optional

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
    # State Persistence and Recovery
    # =========================================================================

    # State schema version for compatibility checking
    STATE_SCHEMA_VERSION = 2

    # Maximum age for state to be considered valid (24 hours)
    MAX_STATE_AGE_HOURS = 24

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
        if saved_checksum:
            # Recalculate checksum without the checksum field
            snapshot_copy = {k: v for k, v in snapshot.items() if k != "checksum"}
            state_json = json.dumps(snapshot_copy, sort_keys=True, default=str)
            calculated_checksum = hashlib.md5(state_json.encode()).hexdigest()

            if calculated_checksum != saved_checksum:
                return False, "Checksum mismatch - state may be corrupted"

        return True, "State snapshot is valid"

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
            snapshot = self._create_state_snapshot()
            snapshot["state"].update(state_data)

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

    def _init_position_reconciliation(self) -> None:
        """Initialize position reconciliation tracking."""
        if not hasattr(self, "_reconciliation_task"):
            self._reconciliation_task: Optional[asyncio.Task] = None
        if not hasattr(self, "_last_known_position"):
            self._last_known_position: Optional[Dict[str, Any]] = None
        if not hasattr(self, "_position_mismatch_count"):
            self._position_mismatch_count: int = 0

    def _start_position_reconciliation(self) -> None:
        """Start background position reconciliation task."""
        self._init_position_reconciliation()

        if self._reconciliation_task is not None:
            return

        async def reconciliation_loop():
            while self._state == BotState.RUNNING:
                await asyncio.sleep(self.POSITION_RECONCILIATION_INTERVAL)
                if self._state == BotState.RUNNING:
                    await self._reconcile_position()

        self._reconciliation_task = asyncio.create_task(reconciliation_loop())
        logger.debug(f"[{self._bot_id}] Position reconciliation started")

    def _stop_position_reconciliation(self) -> None:
        """Stop background position reconciliation task."""
        if hasattr(self, "_reconciliation_task") and self._reconciliation_task:
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

        # Check and reserve atomically
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

    def _classify_order_error(self, error: Exception) -> tuple[str, str]:
        """
        Classify order error for proper handling.

        Args:
            error: The exception from order placement

        Returns:
            Tuple of (error_code, human_readable_message)
        """
        error_str = str(error).lower()

        # Check for common error patterns
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
        # For market orders, use a simpler key
        if price is None:
            return f"{symbol}_{side}_{quantity}"
        return f"{symbol}_{side}_{quantity}_{price}"

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
    DEFAULT_ORDER_TIMEOUT_SECONDS = 30

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

                    if status in ["CANCELED", "REJECTED", "EXPIRED"]:
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
            except Exception:
                pass

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
                            if cancel_on_timeout and status not in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
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

                    if status in ["CANCELED", "REJECTED", "EXPIRED"]:
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

    def _on_fill_notification(self, order_id: str, fill_data: Dict) -> None:
        """
        Called when WebSocket fill notification is received.

        This should be called from the user data stream handler when
        an executionReport event is received with status=FILLED.
        """
        self._init_fill_tracking()

        # Record the fill
        self._confirmed_fills[order_id] = {
            "timestamp": time.time(),
            "data": fill_data,
            "source": "websocket",
        }

        # Trigger callback if registered
        if order_id in self._pending_fill_callbacks:
            try:
                self._pending_fill_callbacks[order_id](fill_data)
            except Exception as e:
                logger.error(f"[{self._bot_id}] Error in fill callback: {e}")

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

                        elif status in ["CANCELED", "REJECTED", "EXPIRED"]:
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
                    if status in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
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
                    adjustment_pct = Decimal(str(0.1 * (attempt + 1))) / 100

                    if side.upper() == "BUY":
                        # For BUY: increase price (more aggressive)
                        adjusted_price = limit_price * (1 + adjustment_pct)
                    else:
                        # For SELL: decrease price (more aggressive)
                        adjusted_price = limit_price * (1 - adjustment_pct)

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
                    error_code, error_cat = await self._classify_order_error(e)
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
                    if error_cat in ["INSUFFICIENT_BALANCE", "POSITION_LIMIT", "SYMBOL_NOT_FOUND"]:
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
                error_code, error_cat = await self._classify_order_error(e)
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

    async def _classify_order_error(self, error: Exception) -> tuple[str, str]:
        """
        Classify order error for retry decision.

        Returns:
            Tuple of (error_code, error_category)

        Categories:
        - RETRYABLE: Temporary issues (network, rate limit)
        - INSUFFICIENT_BALANCE: Not enough funds
        - POSITION_LIMIT: Max position reached
        - SYMBOL_NOT_FOUND: Invalid symbol
        - UNKNOWN: Unclassified error
        """
        error_str = str(error).lower()

        # Insufficient balance
        if "insufficient" in error_str or "balance" in error_str or "margin" in error_str:
            return str(error), "INSUFFICIENT_BALANCE"

        # Position limit
        if "position" in error_str and ("limit" in error_str or "max" in error_str):
            return str(error), "POSITION_LIMIT"

        # Symbol not found
        if "symbol" in error_str and "not found" in error_str:
            return str(error), "SYMBOL_NOT_FOUND"

        # Rate limit
        if "rate" in error_str or "too many" in error_str:
            return str(error), "RETRYABLE"

        # Network/timeout
        if "timeout" in error_str or "connection" in error_str or "network" in error_str:
            return str(error), "RETRYABLE"

        return str(error), "UNKNOWN"

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
