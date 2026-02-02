"""
Backtest-local indicator calculators.

Decoupled copies of indicator classes used by backtest strategies.
These are independent from live bot indicators so that changes to
live trading code cannot silently alter backtest results.

Source classes:
- BollingerCalculator: from src/bots/bollinger/indicators.py
- SupertrendIndicator: from src/bots/supertrend/indicators.py
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple


# =============================================================================
# Bollinger Bands models (backtest-local)
# =============================================================================


@dataclass
class BollingerBands:
    """Bollinger Bands calculation result."""

    upper: Decimal
    middle: Decimal
    lower: Decimal
    std: Decimal
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        for attr in ("upper", "middle", "lower", "std"):
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))


@dataclass
class BBWData:
    """Bollinger Band Width filter data."""

    bbw: Decimal
    bbw_percentile: int
    is_squeeze: bool
    threshold: Decimal

    def __post_init__(self):
        if not isinstance(self.bbw, Decimal):
            self.bbw = Decimal(str(self.bbw))
        if not isinstance(self.threshold, Decimal):
            self.threshold = Decimal(str(self.threshold))


# =============================================================================
# Supertrend models (backtest-local)
# =============================================================================


@dataclass
class SupertrendData:
    """Supertrend indicator data."""

    timestamp: datetime
    upper_band: Decimal
    lower_band: Decimal
    supertrend: Decimal
    trend: int  # 1 = bullish, -1 = bearish
    atr: Decimal

    @property
    def is_bullish(self) -> bool:
        return self.trend == 1

    @property
    def is_bearish(self) -> bool:
        return self.trend == -1


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
# BollingerCalculator
# =============================================================================


class BollingerCalculator:
    """
    Bollinger Bands and BBW Calculator (backtest-local copy).

    Calculates Bollinger Bands using SMA and standard deviation.
    Also tracks BBW history for squeeze detection.
    """

    def __init__(
        self,
        period: int = 20,
        std_multiplier: Decimal = Decimal("2.0"),
        bbw_lookback: int = 200,
        bbw_threshold_pct: int = 20,
    ):
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
        return self._period

    @property
    def std_multiplier(self) -> Decimal:
        return self._std_multiplier

    def calculate(self, klines: List[KlineProtocol]) -> BollingerBands:
        """Calculate Bollinger Bands."""
        if len(klines) < self._period:
            raise InsufficientDataError(self._period, len(klines))

        closes = [k.close for k in klines[-self._period:]]
        closes = [
            c if isinstance(c, Decimal) else Decimal(str(c))
            for c in closes
        ]

        middle = sum(closes) / Decimal(len(closes))
        variance = sum((c - middle) ** 2 for c in closes) / Decimal(len(closes))
        std = self._decimal_sqrt(variance)

        upper = middle + (std * self._std_multiplier)
        lower = middle - (std * self._std_multiplier)

        timestamp = (
            klines[-1].close_time
            if hasattr(klines[-1], "close_time")
            else datetime.now(timezone.utc)
        )

        return BollingerBands(
            upper=upper, middle=middle, lower=lower, std=std, timestamp=timestamp,
        )

    def calculate_bbw(self, bands: BollingerBands) -> BBWData:
        """Calculate BBW and squeeze state."""
        if bands.middle <= 0:
            bbw = Decimal("0")
        else:
            bbw = (bands.upper - bands.lower) / bands.middle

        self._bbw_history.append(bbw)
        if len(self._bbw_history) > self._bbw_lookback:
            self._bbw_history = self._bbw_history[-self._bbw_lookback:]

        if len(self._bbw_history) < 50:
            percentile = 50
        else:
            rank = sum(1 for val in self._bbw_history if val < bbw)
            history_len = len(self._bbw_history)
            percentile = round((rank / history_len) * 100) if history_len > 0 else 50

        is_squeeze = percentile < self._bbw_threshold_pct

        if len(self._bbw_history) >= 50:
            sorted_bbw = sorted(self._bbw_history)
            threshold_idx = int(len(self._bbw_history) * self._bbw_threshold_pct / 100)
            threshold_idx = min(threshold_idx, len(sorted_bbw) - 1)
            threshold = sorted_bbw[threshold_idx]
        else:
            threshold = Decimal("0")

        return BBWData(bbw=bbw, bbw_percentile=percentile, is_squeeze=is_squeeze, threshold=threshold)

    def get_all(
        self, klines: List[KlineProtocol]
    ) -> Tuple[BollingerBands, BBWData]:
        """Calculate both Bollinger Bands and BBW in one call."""
        if len(klines) < self._period:
            raise InsufficientDataError(self._period, len(klines))
        bands = self.calculate(klines)
        bbw = self.calculate_bbw(bands)
        return bands, bbw

    def reset(self) -> None:
        """Reset BBW history."""
        self._bbw_history = []

    @staticmethod
    def _decimal_sqrt(value: Decimal) -> Decimal:
        """Calculate square root of a Decimal using Newton's method."""
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        if value == 0:
            return Decimal("0")
        guess = Decimal(str(math.sqrt(float(value))))
        for _ in range(10):
            guess = (guess + value / guess) / Decimal("2")
        return guess


# =============================================================================
# SupertrendIndicator
# =============================================================================


class SupertrendIndicator:
    """
    Supertrend indicator calculator (backtest-local copy).

    Supertrend is a trend-following indicator based on ATR.
    - Price above Supertrend line: Bullish (trend = 1)
    - Price below Supertrend line: Bearish (trend = -1)
    """

    def __init__(self, atr_period: int = 10, atr_multiplier: Decimal = Decimal("3.0")):
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier

        self._klines: deque = deque(maxlen=atr_period + 10)

        self._prev_upper_band: Optional[Decimal] = None
        self._prev_lower_band: Optional[Decimal] = None
        self._prev_supertrend: Optional[Decimal] = None
        self._prev_trend: int = 0
        self._prev_close: Optional[Decimal] = None

        self._prev_atr: Optional[Decimal] = None
        self._current: Optional[SupertrendData] = None

    def reset(self) -> None:
        """Reset indicator state."""
        self._klines.clear()
        self._prev_upper_band = None
        self._prev_lower_band = None
        self._prev_supertrend = None
        self._prev_trend = 0
        self._prev_close = None
        self._prev_atr = None
        self._current = None

    def update(self, kline) -> Optional[SupertrendData]:
        """Update indicator with new kline data."""
        self._klines.append(kline)

        if len(self._klines) < self._atr_period + 1:
            return None

        atr = self._calculate_atr()
        if atr is None:
            return None

        hl2 = (kline.high + kline.low) / 2
        upper_band = hl2 + self._atr_multiplier * atr
        lower_band = hl2 - self._atr_multiplier * atr

        if self._prev_lower_band is not None and self._prev_close is not None:
            if self._prev_close >= self._prev_lower_band:
                lower_band = max(lower_band, self._prev_lower_band)
            if self._prev_close <= self._prev_upper_band:
                upper_band = min(upper_band, self._prev_upper_band)

        close = kline.close

        if self._prev_trend == 0:
            trend = 1 if close > upper_band else -1
        elif self._prev_trend == 1:
            trend = 1 if close > lower_band else -1
        else:
            trend = -1 if close < upper_band else 1

        supertrend = lower_band if trend == 1 else upper_band

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
        """Calculate Average True Range using Wilder's Smoothing."""
        if len(self._klines) < self._atr_period + 1:
            return None

        klines_list = list(self._klines)
        kline = klines_list[-1]
        prev_close = klines_list[-2].close
        tr = max(
            kline.high - kline.low,
            abs(kline.high - prev_close),
            abs(kline.low - prev_close),
        )

        if self._prev_atr is None:
            # First calculation: simple average to seed
            true_ranges = []
            for i in range(len(klines_list) - self._atr_period, len(klines_list)):
                k = klines_list[i]
                pc = klines_list[i - 1].close
                true_ranges.append(max(
                    k.high - k.low,
                    abs(k.high - pc),
                    abs(k.low - pc),
                ))
            atr = sum(true_ranges) / Decimal(self._atr_period)
        else:
            # Wilder's smoothing
            atr = (self._prev_atr * (self._atr_period - 1) + tr) / self._atr_period

        self._prev_atr = atr
        return atr

    @property
    def current(self) -> Optional[SupertrendData]:
        return self._current

    @property
    def trend(self) -> int:
        return self._prev_trend

    @property
    def is_bullish(self) -> bool:
        return self._prev_trend == 1

    @property
    def is_bearish(self) -> bool:
        return self._prev_trend == -1
