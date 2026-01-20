"""
Supertrend Indicator Calculator.

Calculates Supertrend indicator based on ATR.
"""

from collections import deque
from decimal import Decimal
from typing import Optional, List

from src.core.models import Kline
from src.core import get_logger
from .models import SupertrendData

logger = get_logger(__name__)


class SupertrendIndicator:
    """
    Supertrend indicator calculator.

    Supertrend is a trend-following indicator based on ATR.
    - When price is above Supertrend line: Bullish (trend = 1)
    - When price is below Supertrend line: Bearish (trend = -1)
    """

    def __init__(self, atr_period: int = 10, atr_multiplier: Decimal = Decimal("3.0")):
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier

        # Historical data for ATR calculation
        self._klines: deque[Kline] = deque(maxlen=atr_period + 10)

        # Previous Supertrend values
        self._prev_upper_band: Optional[Decimal] = None
        self._prev_lower_band: Optional[Decimal] = None
        self._prev_supertrend: Optional[Decimal] = None
        self._prev_trend: int = 0
        self._prev_close: Optional[Decimal] = None

        # Current values
        self._current: Optional[SupertrendData] = None

    def reset(self) -> None:
        """Reset indicator state."""
        self._klines.clear()
        self._prev_upper_band = None
        self._prev_lower_band = None
        self._prev_supertrend = None
        self._prev_trend = 0
        self._prev_close = None
        self._current = None

    def update(self, kline: Kline) -> Optional[SupertrendData]:
        """
        Update indicator with new kline data.

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

        # Calculate bands
        hl2 = (kline.high + kline.low) / 2
        upper_band = hl2 + self._atr_multiplier * atr
        lower_band = hl2 - self._atr_multiplier * atr

        # Adjust bands based on previous values
        if self._prev_upper_band is not None and self._prev_close is not None:
            if self._prev_close > self._prev_upper_band:
                lower_band = max(lower_band, self._prev_lower_band)
            if self._prev_close < self._prev_lower_band:
                upper_band = min(upper_band, self._prev_upper_band)

        # Determine trend
        close = kline.close

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

        self._current = SupertrendData(
            timestamp=kline.close_time,
            upper_band=upper_band,
            lower_band=lower_band,
            supertrend=supertrend,
            trend=trend,
            atr=atr,
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

            tr1 = kline.high - kline.low
            tr2 = abs(kline.high - prev_close)
            tr3 = abs(kline.low - prev_close)

            true_ranges.append(max(tr1, tr2, tr3))

        return sum(true_ranges) / Decimal(len(true_ranges))

    @property
    def current(self) -> Optional[SupertrendData]:
        """Get current Supertrend data."""
        return self._current

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

    def initialize_from_klines(self, klines: List[Kline]) -> Optional[SupertrendData]:
        """
        Initialize indicator from historical klines.

        Args:
            klines: List of historical klines (oldest first)

        Returns:
            Latest SupertrendData if successful
        """
        self.reset()

        for kline in klines:
            result = self.update(kline)

        if self._current:
            logger.info(
                f"Supertrend initialized: trend={'BULL' if self.is_bullish else 'BEAR'}, "
                f"value={self._current.supertrend:.2f}, ATR={self._current.atr:.2f}"
            )

        return self._current
