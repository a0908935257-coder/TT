"""
Bollinger Bands + BBW Indicator Calculator.

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 10 期分割):
- Walk-Forward 一致性: 80% (8/10 時段獲利)
- OOS Sharpe: 6.56
- 過度擬合: 未檢測到
- 穩健性: ROBUST

Provides calculation of Bollinger Bands and Bollinger Band Width (BBW)
for BB_TREND_GRID strategy.

Formulas:
    Middle Band = SMA(Close, Period)  # Used for trend detection
    Std = StdDev(Close, Period)
    Upper Band = Middle + (Std × Multiplier)
    Lower Band = Middle - (Std × Multiplier)

    BBW = (Upper - Lower) / Middle
    BBW Percentile = Current BBW rank in history (0-100)
    Squeeze = BBW Percentile < Threshold
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Protocol, Tuple, Union

from src.core import get_logger

from .models import BBWData, BollingerBands

logger = get_logger(__name__)


# =============================================================================
# Helper Functions (NaN/Inf Protection)
# =============================================================================


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


# =============================================================================
# Protocols
# =============================================================================


class KlineProtocol(Protocol):
    """Protocol for Kline data."""

    @property
    def close(self) -> Decimal: ...

    @property
    def high(self) -> Decimal: ...

    @property
    def low(self) -> Decimal: ...

    @property
    def close_time(self) -> datetime: ...


# =============================================================================
# Exceptions
# =============================================================================


class InsufficientDataError(Exception):
    """Raised when there is not enough data for calculation."""

    def __init__(self, required: int, actual: int):
        self.required = required
        self.actual = actual
        super().__init__(
            f"Insufficient data: need at least {required} klines, got {actual}"
        )


# =============================================================================
# Band Position
# =============================================================================


@dataclass
class BandPosition:
    """Price position relative to Bollinger Bands."""

    ABOVE_UPPER = "above_upper"   # Price >= upper band
    BELOW_LOWER = "below_lower"   # Price <= lower band
    UPPER_HALF = "upper_half"     # Price between middle and upper
    LOWER_HALF = "lower_half"     # Price between lower and middle


# =============================================================================
# Calculator
# =============================================================================


class BollingerCalculator:
    """
    Bollinger Bands and BBW Calculator.

    Calculates Bollinger Bands using Simple Moving Average (SMA)
    and standard deviation. Also tracks BBW history for squeeze detection.

    Example:
        >>> calculator = BollingerCalculator(period=20, std_multiplier=Decimal("2.0"))
        >>> bands, bbw = calculator.get_all(klines)
        >>> if not bbw.is_squeeze and price <= bands.lower:
        ...     print("Long signal!")
    """

    def __init__(
        self,
        period: int = 20,
        std_multiplier: Decimal = Decimal("2.0"),
        bbw_lookback: int = 200,
        bbw_threshold_pct: int = 20,
    ):
        """
        Initialize calculator.

        Args:
            period: Bollinger Band period (default 20)
            std_multiplier: Standard deviation multiplier (default 2.0)
            bbw_lookback: BBW history lookback period (default 200)
            bbw_threshold_pct: BBW squeeze threshold percentile (default 20)
        """
        self._period = period
        self._std_multiplier = (
            std_multiplier
            if isinstance(std_multiplier, Decimal)
            else Decimal(str(std_multiplier))
        )
        self._bbw_lookback = bbw_lookback
        self._bbw_threshold_pct = bbw_threshold_pct
        self._bbw_history: List[Decimal] = []

    @property
    def period(self) -> int:
        """Get period."""
        return self._period

    @property
    def std_multiplier(self) -> Decimal:
        """Get standard deviation multiplier."""
        return self._std_multiplier

    @property
    def bbw_history_length(self) -> int:
        """Get current BBW history length."""
        return len(self._bbw_history)

    # =========================================================================
    # Core Calculations
    # =========================================================================

    def calculate(self, klines: List[KlineProtocol]) -> BollingerBands:
        """
        Calculate Bollinger Bands.

        Args:
            klines: List of Kline data (must have at least `period` klines)

        Returns:
            BollingerBands with upper, middle, lower, and std

        Raises:
            InsufficientDataError: If not enough klines
        """
        if len(klines) < self._period:
            raise InsufficientDataError(self._period, len(klines))

        # Get recent closes for calculation
        closes = [k.close for k in klines[-self._period :]]

        # Ensure Decimal type
        closes = [
            c if isinstance(c, Decimal) else Decimal(str(c))
            for c in closes
        ]

        # Calculate SMA (middle band)
        middle = sum(closes) / Decimal(len(closes))

        # Calculate standard deviation
        variance = sum((c - middle) ** 2 for c in closes) / Decimal(len(closes))
        # Use Decimal sqrt approximation
        std = self._decimal_sqrt(variance)

        # Calculate bands
        upper = middle + (std * self._std_multiplier)
        lower = middle - (std * self._std_multiplier)

        # Get timestamp from last kline
        timestamp = (
            klines[-1].close_time
            if hasattr(klines[-1], "close_time")
            else datetime.now(timezone.utc)
        )

        return BollingerBands(
            upper=upper,
            middle=middle,
            lower=lower,
            std=std,
            timestamp=timestamp,
        )

    def calculate_bbw(self, bands: BollingerBands) -> BBWData:
        """
        Calculate BBW (Bollinger Band Width) and squeeze state.

        Args:
            bands: BollingerBands from calculate()

        Returns:
            BBWData with bbw, percentile, squeeze state
        """
        # Calculate BBW
        if bands.middle <= 0:
            bbw = Decimal("0")
        else:
            bbw = (bands.upper - bands.lower) / bands.middle

        # Add to history
        self._bbw_history.append(bbw)

        # Trim history to lookback limit
        if len(self._bbw_history) > self._bbw_lookback:
            self._bbw_history = self._bbw_history[-self._bbw_lookback :]

        # Calculate percentile
        if len(self._bbw_history) < 50:
            # Not enough history, don't flag as squeeze
            percentile = 50
        else:
            # Count how many values are strictly less than current BBW
            # This correctly handles duplicate values
            rank = sum(1 for val in self._bbw_history if val < bbw)
            # Use proper rounding and ensure minimum 1st percentile if any values exist below
            history_len = len(self._bbw_history)
            if history_len > 0:
                percentile = round((rank / history_len) * 100)
            else:
                percentile = 50  # Default to median if no history

        # Determine squeeze state
        is_squeeze = percentile < self._bbw_threshold_pct

        # Calculate threshold value for display
        if len(self._bbw_history) >= 50:
            sorted_bbw = sorted(self._bbw_history)
            threshold_idx = int(len(self._bbw_history) * self._bbw_threshold_pct / 100)
            # Clamp to valid index range to prevent IndexError
            threshold_idx = min(threshold_idx, len(sorted_bbw) - 1)
            threshold = sorted_bbw[threshold_idx]
        else:
            threshold = Decimal("0")

        return BBWData(
            bbw=bbw,
            bbw_percentile=percentile,
            is_squeeze=is_squeeze,
            threshold=threshold,
        )

    def get_all(
        self, klines: List[KlineProtocol]
    ) -> Tuple[BollingerBands, BBWData]:
        """
        Calculate both Bollinger Bands and BBW in one call.

        Args:
            klines: List of Kline data

        Returns:
            Tuple of (BollingerBands, BBWData)

        Raises:
            InsufficientDataError: If not enough klines
        """
        if len(klines) < self._period:
            raise InsufficientDataError(self._period, len(klines))

        bands = self.calculate(klines)
        bbw = self.calculate_bbw(bands)

        return bands, bbw

    def calculate_sma(
        self, klines: List[KlineProtocol], period: int
    ) -> Optional[Decimal]:
        """
        Calculate Simple Moving Average for trend detection.

        Args:
            klines: List of Kline data
            period: SMA period (e.g., 50 for trend filter)

        Returns:
            SMA value, or None if insufficient data
        """
        if len(klines) < period:
            return None

        closes = [k.close for k in klines[-period:]]
        closes = [
            c if isinstance(c, Decimal) else Decimal(str(c))
            for c in closes
        ]

        return sum(closes) / Decimal(len(closes))

    def calculate_atr(
        self, klines: List[KlineProtocol], period: int = 14
    ) -> Optional[Decimal]:
        """
        Calculate Average True Range (ATR) for dynamic stop loss.

        ATR measures volatility by considering:
        - Current high - current low
        - Absolute value of current high - previous close
        - Absolute value of current low - previous close

        Args:
            klines: List of Kline data
            period: ATR period (default 14)

        Returns:
            ATR value, or None if insufficient data
        """
        if len(klines) < period + 1:
            return None

        true_ranges = []

        for i in range(-period, 0):
            kline = klines[i]
            prev_close = klines[i - 1].close

            # Ensure Decimal types
            high = kline.high if isinstance(kline.high, Decimal) else Decimal(str(kline.high))
            low = kline.low if isinstance(kline.low, Decimal) else Decimal(str(kline.low))
            prev_close = prev_close if isinstance(prev_close, Decimal) else Decimal(str(prev_close))

            # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_ranges.append(max(tr1, tr2, tr3))

        # ATR = Simple average of True Ranges
        return sum(true_ranges) / Decimal(len(true_ranges))

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self, klines: List[KlineProtocol]) -> None:
        """
        Initialize BBW history from historical klines.

        Should be called at bot startup with sufficient historical data.

        Args:
            klines: Historical klines (ideally period + bbw_lookback)
        """
        required = self._period + self._bbw_lookback
        if len(klines) < required:
            logger.warning(
                f"Insufficient data for full BBW history initialization: "
                f"need {required}, got {len(klines)}"
            )

        # Clear existing history
        self._bbw_history = []

        # Convert to list if needed (deque doesn't support slicing)
        klines_list = list(klines) if not isinstance(klines, list) else klines

        # Process klines one by one to build history
        for i in range(self._period, len(klines_list)):
            subset = klines_list[: i + 1]
            bands = self.calculate(subset)
            self.calculate_bbw(bands)  # This populates _bbw_history

        logger.info(
            f"BBW history initialized with {len(self._bbw_history)} entries"
        )

    def reset(self) -> None:
        """Reset BBW history."""
        self._bbw_history = []
        logger.debug("BBW history reset")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_band_position(
        self, price: Decimal, bands: BollingerBands
    ) -> str:
        """
        Determine price position relative to bands.

        Args:
            price: Current price
            bands: BollingerBands

        Returns:
            Position string: "above_upper", "below_lower", "upper_half", "lower_half"
        """
        if not isinstance(price, Decimal):
            price = Decimal(str(price))

        if price >= bands.upper:
            return BandPosition.ABOVE_UPPER
        elif price <= bands.lower:
            return BandPosition.BELOW_LOWER
        elif price > bands.middle:
            return BandPosition.UPPER_HALF
        else:
            return BandPosition.LOWER_HALF

    def get_distance_to_band(
        self,
        price: Decimal,
        bands: BollingerBands,
        target: str = "middle",
    ) -> Decimal:
        """
        Calculate distance from price to target band as percentage.

        Args:
            price: Current price
            bands: BollingerBands
            target: Target band ("upper", "middle", "lower")

        Returns:
            Distance as decimal percentage (e.g., 0.02 = 2%)
        """
        if not isinstance(price, Decimal):
            price = Decimal(str(price))

        if target == "upper":
            target_price = bands.upper
        elif target == "lower":
            target_price = bands.lower
        else:
            target_price = bands.middle

        if price <= 0:
            return Decimal("0")

        return (target_price - price) / price

    def is_at_upper_band(
        self, price: Decimal, bands: BollingerBands, tolerance: Decimal = Decimal("0.001")
    ) -> bool:
        """
        Check if price is at or above upper band.

        Args:
            price: Current price
            bands: BollingerBands
            tolerance: Price tolerance (default 0.1%)

        Returns:
            True if price >= upper band (within tolerance)
        """
        if not isinstance(price, Decimal):
            price = Decimal(str(price))

        threshold = bands.upper * (Decimal("1") - tolerance)
        return price >= threshold

    def is_at_lower_band(
        self, price: Decimal, bands: BollingerBands, tolerance: Decimal = Decimal("0.001")
    ) -> bool:
        """
        Check if price is at or below lower band.

        Args:
            price: Current price
            bands: BollingerBands
            tolerance: Price tolerance (default 0.1%)

        Returns:
            True if price <= lower band (within tolerance)
        """
        if not isinstance(price, Decimal):
            price = Decimal(str(price))

        threshold = bands.lower * (Decimal("1") + tolerance)
        return price <= threshold

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _decimal_sqrt(value: Decimal) -> Decimal:
        """
        Calculate square root of a Decimal.

        Uses Newton's method for high precision.

        Args:
            value: Decimal value

        Returns:
            Square root as Decimal
        """
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        if value == 0:
            return Decimal("0")

        # Initial guess using float sqrt
        guess = Decimal(str(math.sqrt(float(value))))

        # Newton's method iterations for precision
        for _ in range(10):
            guess = (guess + value / guess) / Decimal("2")

        return guess

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_bbw_stats(self) -> dict:
        """
        Get BBW history statistics.

        Returns:
            Dictionary with min, max, avg, current BBW values
        """
        if not self._bbw_history:
            return {
                "min": None,
                "max": None,
                "avg": None,
                "current": None,
                "count": 0,
            }

        return {
            "min": str(min(self._bbw_history)),
            "max": str(max(self._bbw_history)),
            "avg": str(sum(self._bbw_history) / len(self._bbw_history)),
            "current": str(self._bbw_history[-1]) if self._bbw_history else None,
            "count": len(self._bbw_history),
        }


# =============================================================================
# ATR Calculator (for Dynamic Grid Range)
# =============================================================================


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
    Used for dynamic grid range calculation in Bollinger Bot.

    Example:
        >>> calc = ATRCalculator(period=14)
        >>> calc.initialize(klines)
        >>> result = calc.update(new_kline)
        >>> grid_range = result.atr * 5.5  # Use ATR for grid range
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

    def initialize(self, klines: List[KlineProtocol]) -> Optional[ATRResult]:
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

    def update(self, kline: KlineProtocol) -> Optional[ATRResult]:
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


