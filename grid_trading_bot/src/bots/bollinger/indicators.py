"""
Bollinger Bands + BBW Indicator Calculator.

Provides calculation of Bollinger Bands and Bollinger Band Width (BBW)
for mean reversion strategy.

Conforms to Prompt 65 specification.

Formulas:
    Middle Band = SMA(Close, Period)
    Std = StdDev(Close, Period)
    Upper Band = Middle + (Std × Multiplier)
    Lower Band = Middle - (Std × Multiplier)

    BBW = (Upper - Lower) / Middle
    BBW Percentile = Current BBW rank in history (0-100)
    Squeeze = BBW Percentile < Threshold
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple

from src.core import get_logger

from .models import BBWData, BollingerBands, SupertrendData

logger = get_logger(__name__)


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
            sorted_bbw = sorted(self._bbw_history)
            # Find rank of current BBW
            rank = 0
            for val in sorted_bbw:
                if val < bbw:
                    rank += 1
                else:
                    break
            percentile = int((rank / len(sorted_bbw)) * 100)

        # Determine squeeze state
        is_squeeze = percentile < self._bbw_threshold_pct

        # Calculate threshold value for display
        if len(self._bbw_history) >= 50:
            threshold_idx = int(len(self._bbw_history) * self._bbw_threshold_pct / 100)
            threshold = sorted(self._bbw_history)[threshold_idx]
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

        # Process klines one by one to build history
        for i in range(self._period, len(klines)):
            subset = klines[: i + 1]
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
# Supertrend Calculator
# =============================================================================


class SupertrendCalculator:
    """
    Supertrend indicator calculator for BOLLINGER_TREND mode.

    Supertrend is a trend-following indicator based on ATR:
    - When price is above Supertrend line: Bullish (trend = 1)
    - When price is below Supertrend line: Bearish (trend = -1)

    Formula:
        Upper Band = HL2 + (ATR × Multiplier)
        Lower Band = HL2 - (ATR × Multiplier)
        Supertrend = Lower Band (if bullish) or Upper Band (if bearish)

    Example:
        >>> calc = SupertrendCalculator(atr_period=20, atr_multiplier=Decimal("3.5"))
        >>> for kline in klines:
        ...     data = calc.update(kline)
        ...     if data and data.is_bullish:
        ...         print("Uptrend")
    """

    def __init__(
        self,
        atr_period: int = 20,
        atr_multiplier: Decimal = Decimal("3.5"),
    ):
        """
        Initialize SupertrendCalculator.

        Args:
            atr_period: ATR calculation period (default 20)
            atr_multiplier: ATR multiplier for bands (default 3.5)
        """
        self._atr_period = atr_period
        self._atr_multiplier = (
            atr_multiplier
            if isinstance(atr_multiplier, Decimal)
            else Decimal(str(atr_multiplier))
        )

        # Historical data for ATR calculation
        self._klines: deque = deque(maxlen=atr_period + 10)

        # Previous Supertrend values
        self._prev_upper_band: Optional[Decimal] = None
        self._prev_lower_band: Optional[Decimal] = None
        self._prev_supertrend: Optional[Decimal] = None
        self._prev_trend: int = 0
        self._prev_close: Optional[Decimal] = None

        # Current values
        self._current: Optional[SupertrendData] = None

    @property
    def trend(self) -> int:
        """Get current trend (1 = bullish, -1 = bearish, 0 = unknown)."""
        return self._prev_trend

    @property
    def is_bullish(self) -> bool:
        """Check if current trend is bullish."""
        return self._prev_trend == 1

    @property
    def is_bearish(self) -> bool:
        """Check if current trend is bearish."""
        return self._prev_trend == -1

    @property
    def current(self) -> Optional[SupertrendData]:
        """Get current Supertrend data."""
        return self._current

    def reset(self) -> None:
        """Reset calculator state."""
        self._klines.clear()
        self._prev_upper_band = None
        self._prev_lower_band = None
        self._prev_supertrend = None
        self._prev_trend = 0
        self._prev_close = None
        self._current = None

    def update(self, kline: KlineProtocol) -> Optional[SupertrendData]:
        """
        Update Supertrend with new kline data.

        Args:
            kline: New kline data

        Returns:
            SupertrendData if enough data, None otherwise
        """
        self._klines.append(kline)

        if len(self._klines) < self._atr_period + 1:
            return None

        # Calculate ATR
        atr = self._calculate_atr()
        if atr is None:
            return None

        # Calculate bands using HL2 (high + low) / 2
        high = kline.high if isinstance(kline.high, Decimal) else Decimal(str(kline.high))
        low = kline.low if isinstance(kline.low, Decimal) else Decimal(str(kline.low))
        close = kline.close if isinstance(kline.close, Decimal) else Decimal(str(kline.close))

        hl2 = (high + low) / Decimal("2")
        upper_band = hl2 + self._atr_multiplier * atr
        lower_band = hl2 - self._atr_multiplier * atr

        # Adjust bands based on previous values
        if self._prev_upper_band is not None and self._prev_close is not None:
            if self._prev_close > self._prev_upper_band:
                lower_band = max(lower_band, self._prev_lower_band)
            if self._prev_close < self._prev_lower_band:
                upper_band = min(upper_band, self._prev_upper_band)

        # Determine trend
        if self._prev_trend == 0:
            # Initial trend based on close vs upper band
            trend = 1 if close > upper_band else -1
        elif self._prev_trend == 1:
            # Was bullish
            trend = 1 if close > self._prev_lower_band else -1
        else:
            # Was bearish
            trend = -1 if close < self._prev_upper_band else 1

        # Supertrend value
        supertrend = lower_band if trend == 1 else upper_band

        # Store for next iteration
        self._prev_upper_band = upper_band
        self._prev_lower_band = lower_band
        self._prev_supertrend = supertrend
        self._prev_trend = trend
        self._prev_close = close

        timestamp = (
            kline.close_time
            if hasattr(kline, "close_time")
            else datetime.now(timezone.utc)
        )

        self._current = SupertrendData(
            upper_band=upper_band,
            lower_band=lower_band,
            supertrend=supertrend,
            trend=trend,
            atr=atr,
            timestamp=timestamp,
        )

        return self._current

    def _calculate_atr(self) -> Optional[Decimal]:
        """Calculate Average True Range."""
        if len(self._klines) < self._atr_period + 1:
            return None

        true_ranges = []
        klines_list = list(self._klines)

        for i in range(len(klines_list) - self._atr_period, len(klines_list)):
            kline = klines_list[i]
            prev_close = klines_list[i - 1].close

            # Ensure Decimal types
            high = kline.high if isinstance(kline.high, Decimal) else Decimal(str(kline.high))
            low = kline.low if isinstance(kline.low, Decimal) else Decimal(str(kline.low))
            prev_close = prev_close if isinstance(prev_close, Decimal) else Decimal(str(prev_close))

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_ranges.append(max(tr1, tr2, tr3))

        return sum(true_ranges) / Decimal(len(true_ranges))

    def initialize(self, klines: List[KlineProtocol]) -> Optional[SupertrendData]:
        """
        Initialize Supertrend from historical klines.

        Args:
            klines: List of historical klines (oldest first)

        Returns:
            Latest SupertrendData if successful
        """
        self.reset()

        for kline in klines:
            self.update(kline)

        if self._current:
            logger.info(
                f"Supertrend initialized: trend={'BULL' if self.is_bullish else 'BEAR'}, "
                f"value={self._current.supertrend:.2f}, ATR={self._current.atr:.2f}"
            )

        return self._current
