"""
RSI-Grid Hybrid Bot Indicators.

Provides RSI, ATR, and SMA calculation for the hybrid trading strategy.
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Union

from src.core import get_logger
from src.core.models import Kline

from .models import RSIZone

logger = get_logger(__name__)


def is_valid_number(value: Union[Decimal, float, None]) -> bool:
    """
    Check if a value is a valid number (not NaN, Inf, or None).

    Args:
        value: Value to check (Decimal, float, or None)

    Returns:
        True if value is a valid finite number, False otherwise
    """
    if value is None:
        return False

    try:
        if isinstance(value, Decimal):
            # Decimal NaN/Inf checks
            if value.is_nan() or value.is_infinite():
                return False
            return True
        elif isinstance(value, (int, float)):
            # Float NaN/Inf checks
            if math.isnan(value) or math.isinf(value):
                return False
            return True
        return False
    except (TypeError, InvalidOperation):
        return False


def safe_decimal(value: Union[Decimal, float, str, None], default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Decimal value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, Decimal):
            if value.is_nan() or value.is_infinite():
                logger.warning(f"Invalid Decimal value detected: {value}, using default {default}")
                return default
            return value

        result = Decimal(str(value))
        if result.is_nan() or result.is_infinite():
            logger.warning(f"Converted to invalid Decimal: {value}, using default {default}")
            return default
        return result
    except (InvalidOperation, TypeError, ValueError) as e:
        logger.warning(f"Failed to convert {value} to Decimal: {e}, using default {default}")
        return default


@dataclass
class RSIResult:
    """RSI calculation result."""
    timestamp: datetime
    rsi: Decimal
    avg_gain: Decimal
    avg_loss: Decimal
    zone: RSIZone


class RSICalculator:
    """
    RSI (Relative Strength Index) Calculator.

    Uses Wilder's smoothing method for accurate RSI calculation.
    Includes zone classification for RSI-Grid strategy.

    Example:
        >>> calc = RSICalculator(period=14, oversold=30, overbought=70)
        >>> calc.initialize(klines)
        >>> result = calc.update(new_kline)
        >>> if result.zone == RSIZone.OVERSOLD:
        ...     print("Oversold - only LONG allowed")
    """

    def __init__(
        self,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ):
        """
        Initialize RSI Calculator.

        Args:
            period: RSI period (default 14)
            oversold: Oversold threshold (default 30)
            overbought: Overbought threshold (default 70)
        """
        self._period = period
        self._oversold = oversold
        self._overbought = overbought

        # State
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None
        self._prev_close: Optional[Decimal] = None
        self._rsi: Optional[Decimal] = None
        self._initialized = False

        # Cache for preventing duplicate calculations
        self._cache_timestamp: Optional[datetime] = None
        self._cache_result: Optional[RSIResult] = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def initialize(self, klines: List[Kline]) -> Optional[RSIResult]:
        """
        Initialize RSI with historical klines.

        Args:
            klines: Historical klines (need at least period + 1)

        Returns:
            Initial RSI result or None if insufficient data
        """
        if len(klines) < self._period + 1:
            logger.warning(f"Insufficient klines for RSI init: {len(klines)} < {self._period + 1}")
            return None

        # Calculate initial average gain/loss from first 'period' changes
        gains = []
        losses = []

        for i in range(1, self._period + 1):
            change = klines[i].close - klines[i - 1].close
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        self._avg_gain = sum(gains) / self._period
        self._avg_loss = sum(losses) / self._period

        # Process remaining klines using Wilder's smoothing
        for i in range(self._period + 1, len(klines)):
            change = klines[i].close - klines[i - 1].close
            if change > 0:
                current_gain = change
                current_loss = Decimal("0")
            else:
                current_gain = Decimal("0")
                current_loss = abs(change)

            # Wilder's smoothing: new_avg = (prev_avg * (period-1) + current) / period
            self._avg_gain = (self._avg_gain * (self._period - 1) + current_gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + current_loss) / self._period

        # Calculate RSI
        self._rsi = self._calculate_rsi()
        self._prev_close = Decimal(str(klines[-1].close))
        self._initialized = True

        logger.info(f"RSI initialized: {self._rsi:.2f}, period={self._period}")

        return RSIResult(
            timestamp=klines[-1].close_time,
            rsi=self._rsi,
            avg_gain=self._avg_gain,
            avg_loss=self._avg_loss,
            zone=self._get_zone(self._rsi),
        )

    def update(self, kline: Kline) -> Optional[RSIResult]:
        """
        Update RSI with new kline (with caching).

        Args:
            kline: New kline data

        Returns:
            Updated RSI result or None if not initialized
        """
        if not self._initialized or self._prev_close is None:
            return None

        # Check cache - return cached result if same kline
        if self._cache_timestamp == kline.close_time and self._cache_result is not None:
            self._cache_hits += 1
            return self._cache_result

        self._cache_misses += 1

        current_close = Decimal(str(kline.close))
        change = current_close - self._prev_close

        if change > 0:
            current_gain = change
            current_loss = Decimal("0")
        else:
            current_gain = Decimal("0")
            current_loss = abs(change)

        # Wilder's smoothing
        self._avg_gain = (self._avg_gain * (self._period - 1) + current_gain) / self._period
        self._avg_loss = (self._avg_loss * (self._period - 1) + current_loss) / self._period

        self._rsi = self._calculate_rsi()
        self._prev_close = current_close

        result = RSIResult(
            timestamp=kline.close_time,
            rsi=self._rsi,
            avg_gain=self._avg_gain,
            avg_loss=self._avg_loss,
            zone=self._get_zone(self._rsi),
        )

        # Update cache
        self._cache_timestamp = kline.close_time
        self._cache_result = result

        return result

    def _calculate_rsi(self) -> Decimal:
        """Calculate RSI from average gain/loss with NaN/Inf protection."""
        # Validate inputs
        if not is_valid_number(self._avg_gain) or not is_valid_number(self._avg_loss):
            logger.warning("Invalid avg_gain or avg_loss, returning neutral RSI (50)")
            return Decimal("50")

        if self._avg_loss == 0:
            return Decimal("100")

        try:
            rs = self._avg_gain / self._avg_loss
            if not is_valid_number(rs):
                logger.warning(f"Invalid RS calculated: {rs}, returning neutral RSI (50)")
                return Decimal("50")

            rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

            # Final validation
            if not is_valid_number(rsi):
                logger.warning(f"Invalid RSI calculated: {rsi}, returning neutral RSI (50)")
                return Decimal("50")

            # Clamp RSI to valid range [0, 100]
            return max(Decimal("0"), min(Decimal("100"), rsi))

        except (InvalidOperation, ZeroDivisionError) as e:
            logger.warning(f"RSI calculation error: {e}, returning neutral RSI (50)")
            return Decimal("50")

    def _get_zone(self, rsi: Decimal) -> RSIZone:
        """Get RSI zone classification."""
        if rsi < self._oversold:
            return RSIZone.OVERSOLD
        elif rsi > self._overbought:
            return RSIZone.OVERBOUGHT
        return RSIZone.NEUTRAL

    @property
    def rsi(self) -> Optional[Decimal]:
        """Current RSI value."""
        return self._rsi

    @property
    def zone(self) -> RSIZone:
        """Current RSI zone."""
        if self._rsi is None:
            return RSIZone.NEUTRAL
        return self._get_zone(self._rsi)

    @property
    def is_oversold(self) -> bool:
        """Check if RSI is in oversold territory."""
        return self._rsi is not None and self._rsi < self._oversold

    @property
    def is_overbought(self) -> bool:
        """Check if RSI is in overbought territory."""
        return self._rsi is not None and self._rsi > self._overbought

    @property
    def is_neutral(self) -> bool:
        """Check if RSI is in neutral territory."""
        if self._rsi is None:
            return True
        return self._oversold <= self._rsi <= self._overbought

    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            "rsi": float(self._rsi) if self._rsi else None,
            "avg_gain": float(self._avg_gain) if self._avg_gain else None,
            "avg_loss": float(self._avg_loss) if self._avg_loss else None,
            "zone": self.zone.value,
            "is_oversold": self.is_oversold,
            "is_overbought": self.is_overbought,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate_pct": round(hit_rate, 2),
        }

    def reset(self) -> None:
        """Reset calculator state."""
        self._avg_gain = None
        self._avg_loss = None
        self._prev_close = None
        self._rsi = None
        self._initialized = False
        self._cache_timestamp = None
        self._cache_result = None
        self._cache_hits = 0
        self._cache_misses = 0


@dataclass
class ATRResult:
    """ATR calculation result."""
    timestamp: datetime
    atr: Decimal
    tr: Decimal  # Current True Range


class ATRCalculator:
    """
    ATR (Average True Range) Calculator.

    Uses Wilder's smoothing method for ATR calculation.

    Example:
        >>> calc = ATRCalculator(period=14)
        >>> calc.initialize(klines)
        >>> result = calc.update(new_kline)
        >>> grid_range = result.atr * 3.0  # Use ATR for grid range
    """

    def __init__(self, period: int = 14):
        """
        Initialize ATR Calculator.

        Args:
            period: ATR period (default 14)
        """
        self._period = period

        # State
        self._atr: Optional[Decimal] = None
        self._prev_close: Optional[Decimal] = None
        self._initialized = False

        # Cache for preventing duplicate calculations
        self._cache_timestamp: Optional[datetime] = None
        self._cache_result: Optional[ATRResult] = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def initialize(self, klines: List[Kline]) -> Optional[ATRResult]:
        """
        Initialize ATR with historical klines.

        Args:
            klines: Historical klines (need at least period + 1)

        Returns:
            Initial ATR result or None if insufficient data
        """
        if len(klines) < self._period + 1:
            logger.warning(f"Insufficient klines for ATR init: {len(klines)} < {self._period + 1}")
            return None

        # Calculate initial ATR as simple average of first 'period' TRs
        tr_values = []
        for i in range(1, self._period + 1):
            high = Decimal(str(klines[i].high))
            low = Decimal(str(klines[i].low))
            prev_close = Decimal(str(klines[i - 1].close))

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        self._atr = sum(tr_values) / Decimal(self._period)

        # Process remaining klines using Wilder's smoothing
        for i in range(self._period + 1, len(klines)):
            high = Decimal(str(klines[i].high))
            low = Decimal(str(klines[i].low))
            prev_close = Decimal(str(klines[i - 1].close))

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

            # Wilder's smoothing
            self._atr = (self._atr * (self._period - 1) + tr) / self._period

        self._prev_close = Decimal(str(klines[-1].close))
        self._initialized = True

        logger.info(f"ATR initialized: {self._atr:.2f}, period={self._period}")

        return ATRResult(
            timestamp=klines[-1].close_time,
            atr=self._atr,
            tr=tr_values[-1] if tr_values else self._atr,
        )

    def update(self, kline: Kline) -> Optional[ATRResult]:
        """
        Update ATR with new kline (with caching).

        Args:
            kline: New kline data

        Returns:
            Updated ATR result or None if not initialized
        """
        if not self._initialized or self._prev_close is None:
            return None

        # Check cache - return cached result if same kline
        if self._cache_timestamp == kline.close_time and self._cache_result is not None:
            self._cache_hits += 1
            return self._cache_result

        self._cache_misses += 1

        # Safe conversion with NaN/Inf protection
        high = safe_decimal(kline.high)
        low = safe_decimal(kline.low)
        close = safe_decimal(kline.close)

        # Validate inputs
        if not all(is_valid_number(v) for v in [high, low, close]):
            logger.warning("Invalid kline data for ATR calculation, skipping update")
            return None

        try:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close)
            )

            if not is_valid_number(tr):
                logger.warning(f"Invalid TR calculated: {tr}, skipping update")
                return None

            # Wilder's smoothing
            new_atr = (self._atr * (self._period - 1) + tr) / self._period

            if not is_valid_number(new_atr):
                logger.warning(f"Invalid ATR calculated: {new_atr}, keeping previous value")
                result = ATRResult(
                    timestamp=kline.close_time,
                    atr=self._atr,
                    tr=tr,
                )
            else:
                self._atr = new_atr
                self._prev_close = close
                result = ATRResult(
                    timestamp=kline.close_time,
                    atr=self._atr,
                    tr=tr,
                )

            # Update cache
            self._cache_timestamp = kline.close_time
            self._cache_result = result

            return result

        except (InvalidOperation, ZeroDivisionError) as e:
            logger.warning(f"ATR calculation error: {e}")
            return None

    @property
    def atr(self) -> Optional[Decimal]:
        """Current ATR value."""
        return self._atr

    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            "atr": float(self._atr) if self._atr else None,
            "prev_close": float(self._prev_close) if self._prev_close else None,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate_pct": round(hit_rate, 2),
        }

    def reset(self) -> None:
        """Reset calculator state."""
        self._atr = None
        self._prev_close = None
        self._initialized = False
        self._cache_timestamp = None
        self._cache_result = None
        self._cache_hits = 0
        self._cache_misses = 0


class SMACalculator:
    """
    SMA (Simple Moving Average) Calculator.

    Example:
        >>> calc = SMACalculator(period=20)
        >>> sma = calc.calculate(closes)
        >>> if current_price > sma:
        ...     print("Uptrend")
    """

    def __init__(self, period: int = 20):
        """
        Initialize SMA Calculator.

        Args:
            period: SMA period (default 20)
        """
        self._period = period
        self._closes: deque = deque(maxlen=period)
        self._sma: Optional[Decimal] = None

    def update(self, close: Decimal) -> Optional[Decimal]:
        """
        Update SMA with new close price.

        Args:
            close: New close price

        Returns:
            Current SMA or None if insufficient data or invalid input
        """
        # Safe conversion with NaN/Inf protection
        close = safe_decimal(close)

        if not is_valid_number(close):
            logger.warning(f"Invalid close price for SMA: {close}, skipping update")
            return self._sma  # Return previous SMA

        self._closes.append(close)

        if len(self._closes) >= self._period:
            try:
                new_sma = sum(self._closes) / Decimal(self._period)
                if is_valid_number(new_sma):
                    self._sma = new_sma
                else:
                    logger.warning(f"Invalid SMA calculated: {new_sma}, keeping previous value")
            except (InvalidOperation, ZeroDivisionError) as e:
                logger.warning(f"SMA calculation error: {e}")
            return self._sma

        return None

    def calculate(self, closes: List[Decimal]) -> Optional[Decimal]:
        """
        Calculate SMA from list of closes.

        Args:
            closes: List of close prices

        Returns:
            SMA value or None if insufficient data or calculation error
        """
        if len(closes) < self._period:
            return None

        try:
            # Filter out invalid values
            recent = [safe_decimal(c) for c in closes[-self._period:]]
            valid_closes = [c for c in recent if is_valid_number(c)]

            if len(valid_closes) < self._period:
                logger.warning("Insufficient valid closes for SMA calculation")
                return None

            result = sum(valid_closes) / Decimal(self._period)
            if not is_valid_number(result):
                logger.warning(f"Invalid SMA result: {result}")
                return None
            return result
        except (InvalidOperation, ZeroDivisionError) as e:
            logger.warning(f"SMA calculate error: {e}")
            return None

    @property
    def sma(self) -> Optional[Decimal]:
        """Current SMA value."""
        return self._sma

    def get_trend(self, current_price: Decimal) -> int:
        """
        Get trend direction based on SMA.

        Returns:
            1: Uptrend (price > SMA)
            -1: Downtrend (price < SMA)
            0: Neutral
        """
        if self._sma is None or not is_valid_number(self._sma):
            return 0

        current_price = safe_decimal(current_price)
        if not is_valid_number(current_price) or self._sma == 0:
            return 0

        try:
            diff_pct = (current_price - self._sma) / self._sma * Decimal("100")

            if not is_valid_number(diff_pct):
                return 0

            if diff_pct > Decimal("0.5"):
                return 1  # Uptrend
            elif diff_pct < Decimal("-0.5"):
                return -1  # Downtrend
            return 0
        except (InvalidOperation, ZeroDivisionError):
            return 0

    def reset(self) -> None:
        """Reset calculator state."""
        self._closes.clear()
        self._sma = None
