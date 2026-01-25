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

    async def _check_balance_for_order(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
    ) -> tuple[bool, str]:
        """
        Check if account has sufficient balance for order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (or estimated price for market orders)
            leverage: Position leverage

        Returns:
            Tuple of (has_sufficient_balance, message)
        """
        try:
            # Get account balance
            balance = await self._exchange.futures.get_balance("USDT")
            available = balance.free if balance else Decimal("0")

            # Calculate required margin
            notional_value = quantity * price
            required_margin = notional_value / Decimal(leverage)

            # Add buffer for fees (0.1%)
            required_with_fees = required_margin * Decimal("1.001")

            if available < required_with_fees:
                return False, (
                    f"Insufficient balance: need {required_with_fees:.2f} USDT "
                    f"(margin + fees), have {available:.2f} USDT"
                )

            return True, f"Balance OK: {available:.2f} USDT available"

        except Exception as e:
            logger.error(f"[{self._bot_id}] Balance check failed: {e}")
            return False, f"Balance check error: {e}"

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

    async def _cancel_order_with_timeout(
        self,
        order_id: str,
        symbol: str,
        timeout_seconds: float = 10.0,
    ) -> bool:
        """
        Cancel order with timeout protection.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            timeout_seconds: Timeout in seconds

        Returns:
            True if cancelled successfully
        """
        try:
            cancel_result = await asyncio.wait_for(
                self._exchange.futures.cancel_order(
                    symbol=symbol,
                    order_id=order_id,
                ),
                timeout=timeout_seconds,
            )
            logger.info(f"[{self._bot_id}] Order {order_id} cancelled")
            return True

        except asyncio.TimeoutError:
            logger.warning(
                f"[{self._bot_id}] Cancel order timed out for {order_id}"
            )
            return False

        except Exception as e:
            # Order might already be filled/cancelled
            error_str = str(e).lower()
            if "unknown order" in error_str or "not found" in error_str:
                logger.debug(f"[{self._bot_id}] Order {order_id} already gone")
                return True
            logger.error(f"[{self._bot_id}] Cancel order error: {e}")
            return False

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
