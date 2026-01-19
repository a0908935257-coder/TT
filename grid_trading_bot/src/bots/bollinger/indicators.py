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
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple

from src.core import get_logger

from .models import BBWData, BollingerBands

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
