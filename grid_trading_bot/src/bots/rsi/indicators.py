"""
RSI Indicator Calculator.

Provides RSI (Relative Strength Index) calculation for mean reversion trading.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from src.core import get_logger
from src.core.models import Kline

logger = get_logger(__name__)


@dataclass
class RSIResult:
    """RSI calculation result."""
    timestamp: datetime
    rsi: Decimal
    avg_gain: Decimal
    avg_loss: Decimal


class RSICalculator:
    """
    RSI (Relative Strength Index) Calculator.

    Uses Wilder's smoothing method for accurate RSI calculation.

    Example:
        >>> calc = RSICalculator(period=14, oversold=20, overbought=80)
        >>> calc.initialize(klines)
        >>> result = calc.update(new_kline)
        >>> if result.rsi < 20:
        ...     print("Oversold - potential long entry")
    """

    def __init__(
        self,
        period: int = 14,
        oversold: int = 20,
        overbought: int = 80,
    ):
        """
        Initialize RSI Calculator.

        Args:
            period: RSI period (default 14)
            oversold: Oversold threshold (default 20)
            overbought: Overbought threshold (default 80)
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

        # History for initialization
        self._price_changes: deque = deque(maxlen=period)

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
        )

    def _calculate_rsi(self) -> Decimal:
        """Calculate RSI from average gain/loss."""
        if self._avg_loss == 0:
            return Decimal("100")
        rs = self._avg_gain / self._avg_loss
        return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

    @property
    def rsi(self) -> Optional[Decimal]:
        """Current RSI value."""
        return self._rsi

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

    @property
    def signal(self) -> str:
        """Get current signal based on RSI."""
        if self.is_oversold:
            return "long"
        elif self.is_overbought:
            return "short"
        return "none"

    def get_state(self) -> dict:
        """Get current state for persistence."""
        return {
            "rsi": float(self._rsi) if self._rsi else None,
            "avg_gain": float(self._avg_gain) if self._avg_gain else None,
            "avg_loss": float(self._avg_loss) if self._avg_loss else None,
            "is_oversold": self.is_oversold,
            "is_overbought": self.is_overbought,
            "signal": self.signal,
        }
