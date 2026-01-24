"""
RSI-Grid Hybrid Bot Indicators.

Provides RSI, ATR, and SMA calculation for the hybrid trading strategy.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from src.core import get_logger
from src.core.models import Kline

from .models import RSIZone

logger = get_logger(__name__)


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
            change = float(klines[i].close) - float(klines[i - 1].close)
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        self._avg_gain = Decimal(str(sum(gains) / self._period))
        self._avg_loss = Decimal(str(sum(losses) / self._period))

        # Process remaining klines using Wilder's smoothing
        for i in range(self._period + 1, len(klines)):
            change = float(klines[i].close) - float(klines[i - 1].close)
            if change > 0:
                current_gain = Decimal(str(change))
                current_loss = Decimal("0")
            else:
                current_gain = Decimal("0")
                current_loss = Decimal(str(abs(change)))

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
        Update RSI with new kline.

        Args:
            kline: New kline data

        Returns:
            Updated RSI result or None if not initialized
        """
        if not self._initialized or self._prev_close is None:
            return None

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

        return RSIResult(
            timestamp=kline.close_time,
            rsi=self._rsi,
            avg_gain=self._avg_gain,
            avg_loss=self._avg_loss,
            zone=self._get_zone(self._rsi),
        )

    def _calculate_rsi(self) -> Decimal:
        """Calculate RSI from average gain/loss."""
        if self._avg_loss == 0:
            return Decimal("100")
        rs = self._avg_gain / self._avg_loss
        return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

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
        }

    def reset(self) -> None:
        """Reset calculator state."""
        self._avg_gain = None
        self._avg_loss = None
        self._prev_close = None
        self._rsi = None
        self._initialized = False


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
        Update ATR with new kline.

        Args:
            kline: New kline data

        Returns:
            Updated ATR result or None if not initialized
        """
        if not self._initialized or self._prev_close is None:
            return None

        high = Decimal(str(kline.high))
        low = Decimal(str(kline.low))

        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close)
        )

        # Wilder's smoothing
        self._atr = (self._atr * (self._period - 1) + tr) / self._period
        self._prev_close = Decimal(str(kline.close))

        return ATRResult(
            timestamp=kline.close_time,
            atr=self._atr,
            tr=tr,
        )

    @property
    def atr(self) -> Optional[Decimal]:
        """Current ATR value."""
        return self._atr

    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            "atr": float(self._atr) if self._atr else None,
            "prev_close": float(self._prev_close) if self._prev_close else None,
        }

    def reset(self) -> None:
        """Reset calculator state."""
        self._atr = None
        self._prev_close = None
        self._initialized = False


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
            Current SMA or None if insufficient data
        """
        if not isinstance(close, Decimal):
            close = Decimal(str(close))

        self._closes.append(close)

        if len(self._closes) >= self._period:
            self._sma = sum(self._closes) / Decimal(self._period)
            return self._sma

        return None

    def calculate(self, closes: List[Decimal]) -> Optional[Decimal]:
        """
        Calculate SMA from list of closes.

        Args:
            closes: List of close prices

        Returns:
            SMA value or None if insufficient data
        """
        if len(closes) < self._period:
            return None

        recent = closes[-self._period:]
        return sum(recent) / Decimal(self._period)

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
        if self._sma is None:
            return 0

        diff_pct = (current_price - self._sma) / self._sma * Decimal("100")

        if diff_pct > Decimal("0.5"):
            return 1  # Uptrend
        elif diff_pct < Decimal("-0.5"):
            return -1  # Downtrend
        return 0

    def reset(self) -> None:
        """Reset calculator state."""
        self._closes.clear()
        self._sma = None
