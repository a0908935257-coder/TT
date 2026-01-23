"""
DSL Indicator Factory.

Creates and manages indicator instances from DSL definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from ....core.models import Kline
from .models import IndicatorDefinition


class IndicatorError(Exception):
    """Exception raised when indicator calculation fails."""
    pass


@dataclass
class IndicatorResult:
    """
    Result of an indicator calculation.

    Attributes:
        values: Dictionary of field name to value
        is_ready: Whether indicator has enough data
    """
    values: dict[str, Any]
    is_ready: bool = True


class BaseIndicator(ABC):
    """
    Base class for DSL indicators.

    All indicators must implement calculate() which returns
    the current values for all output fields.
    """

    @abstractmethod
    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        """
        Calculate indicator values from klines.

        Args:
            klines: List of klines (historical data)

        Returns:
            IndicatorResult with field values
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state."""
        pass

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """Return required warmup period."""
        pass


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_multiplier: float = 2.0, **kwargs):
        self._period = int(period)
        self._std_multiplier = Decimal(str(std_multiplier))

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._period:
            return IndicatorResult(values={}, is_ready=False)

        # Get recent closes
        closes = [k.close for k in klines[-self._period:]]

        # Calculate SMA
        sma = sum(closes) / Decimal(len(closes))

        # Calculate standard deviation
        variance = sum((c - sma) ** 2 for c in closes) / Decimal(len(closes))
        std = variance.sqrt() if variance > 0 else Decimal("0")

        # Calculate bands
        upper = sma + (self._std_multiplier * std)
        lower = sma - (self._std_multiplier * std)

        # Calculate bandwidth and %B
        bandwidth = (upper - lower) / sma if sma > 0 else Decimal("0")
        current_close = klines[-1].close
        percent_b = (current_close - lower) / (upper - lower) if upper != lower else Decimal("0.5")

        return IndicatorResult(
            values={
                "upper": upper,
                "middle": sma,
                "lower": lower,
                "bandwidth": bandwidth,
                "percent_b": percent_b,
            }
        )

    def reset(self) -> None:
        pass

    @property
    def warmup_period(self) -> int:
        return self._period


class SupertrendIndicator(BaseIndicator):
    """Supertrend indicator."""

    def __init__(self, atr_period: int = 20, atr_multiplier: float = 3.5, **kwargs):
        self._atr_period = int(atr_period)
        self._atr_multiplier = Decimal(str(atr_multiplier))
        self._trend: int = 0
        self._supertrend_up: Optional[Decimal] = None
        self._supertrend_down: Optional[Decimal] = None
        self._prev_close: Optional[Decimal] = None

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._atr_period + 1:
            return IndicatorResult(values={}, is_ready=False)

        # Calculate ATR
        true_ranges = []
        for i in range(-self._atr_period, 0):
            kline = klines[i]
            prev_close = klines[i - 1].close
            tr = max(
                kline.high - kline.low,
                abs(kline.high - prev_close),
                abs(kline.low - prev_close),
            )
            true_ranges.append(tr)

        atr = sum(true_ranges) / Decimal(len(true_ranges))

        kline = klines[-1]
        hl2 = (kline.high + kline.low) / Decimal("2")
        basic_upper = hl2 + (self._atr_multiplier * atr)
        basic_lower = hl2 - (self._atr_multiplier * atr)

        # Initialize on first run
        if self._supertrend_up is None:
            self._supertrend_up = basic_lower
            self._supertrend_down = basic_upper
            self._trend = 1 if kline.close > hl2 else -1
            self._prev_close = kline.close
        else:
            # Update supertrend values
            if basic_lower > self._supertrend_up or self._prev_close < self._supertrend_up:
                self._supertrend_up = basic_lower
            else:
                self._supertrend_up = max(basic_lower, self._supertrend_up)

            if basic_upper < self._supertrend_down or self._prev_close > self._supertrend_down:
                self._supertrend_down = basic_upper
            else:
                self._supertrend_down = min(basic_upper, self._supertrend_down)

            # Determine trend
            if self._trend == 1:
                if kline.close < self._supertrend_up:
                    self._trend = -1
            else:
                if kline.close > self._supertrend_down:
                    self._trend = 1

            self._prev_close = kline.close

        value = self._supertrend_up if self._trend == 1 else self._supertrend_down

        return IndicatorResult(
            values={
                "trend": self._trend,
                "value": value,
                "upper_band": self._supertrend_down,
                "lower_band": self._supertrend_up,
            }
        )

    def reset(self) -> None:
        self._trend = 0
        self._supertrend_up = None
        self._supertrend_down = None
        self._prev_close = None

    @property
    def warmup_period(self) -> int:
        return self._atr_period + 50


class SMAIndicator(BaseIndicator):
    """Simple Moving Average indicator."""

    def __init__(self, period: int = 20, **kwargs):
        self._period = int(period)

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._period:
            return IndicatorResult(values={}, is_ready=False)

        closes = [k.close for k in klines[-self._period:]]
        sma = sum(closes) / Decimal(len(closes))

        return IndicatorResult(values={"value": sma})

    def reset(self) -> None:
        pass

    @property
    def warmup_period(self) -> int:
        return self._period


class EMAIndicator(BaseIndicator):
    """Exponential Moving Average indicator."""

    def __init__(self, period: int = 20, **kwargs):
        self._period = int(period)
        self._ema: Optional[Decimal] = None
        self._multiplier = Decimal("2") / Decimal(str(period + 1))

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._period:
            return IndicatorResult(values={}, is_ready=False)

        if self._ema is None:
            # Initialize with SMA
            closes = [k.close for k in klines[-self._period:]]
            self._ema = sum(closes) / Decimal(len(closes))
        else:
            # Update EMA
            current_close = klines[-1].close
            self._ema = (current_close - self._ema) * self._multiplier + self._ema

        return IndicatorResult(values={"value": self._ema})

    def reset(self) -> None:
        self._ema = None

    @property
    def warmup_period(self) -> int:
        return self._period


class RSIIndicator(BaseIndicator):
    """Relative Strength Index indicator."""

    def __init__(self, period: int = 14, **kwargs):
        self._period = int(period)
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._period + 1:
            return IndicatorResult(values={}, is_ready=False)

        if self._avg_gain is None:
            # Initialize with simple average
            gains = []
            losses = []
            for i in range(-self._period, 0):
                change = klines[i].close - klines[i - 1].close
                if change > 0:
                    gains.append(change)
                    losses.append(Decimal("0"))
                else:
                    gains.append(Decimal("0"))
                    losses.append(abs(change))

            self._avg_gain = sum(gains) / Decimal(self._period)
            self._avg_loss = sum(losses) / Decimal(self._period)
        else:
            # Update with smoothed average
            change = klines[-1].close - klines[-2].close
            gain = max(change, Decimal("0"))
            loss = abs(min(change, Decimal("0")))

            self._avg_gain = (self._avg_gain * Decimal(self._period - 1) + gain) / Decimal(self._period)
            self._avg_loss = (self._avg_loss * Decimal(self._period - 1) + loss) / Decimal(self._period)

        if self._avg_loss == 0:
            rsi = Decimal("100")
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return IndicatorResult(values={"value": rsi})

    def reset(self) -> None:
        self._avg_gain = None
        self._avg_loss = None

    @property
    def warmup_period(self) -> int:
        return self._period + 1


class ATRIndicator(BaseIndicator):
    """Average True Range indicator."""

    def __init__(self, period: int = 14, **kwargs):
        self._period = int(period)

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._period + 1:
            return IndicatorResult(values={}, is_ready=False)

        true_ranges = []
        for i in range(-self._period, 0):
            kline = klines[i]
            prev_close = klines[i - 1].close
            tr = max(
                kline.high - kline.low,
                abs(kline.high - prev_close),
                abs(kline.low - prev_close),
            )
            true_ranges.append(tr)

        atr = sum(true_ranges) / Decimal(len(true_ranges))

        return IndicatorResult(values={"value": atr})

    def reset(self) -> None:
        pass

    @property
    def warmup_period(self) -> int:
        return self._period + 1


class MACDIndicator(BaseIndicator):
    """MACD indicator."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs):
        self._fast_period = int(fast_period)
        self._slow_period = int(slow_period)
        self._signal_period = int(signal_period)
        self._fast_ema: Optional[Decimal] = None
        self._slow_ema: Optional[Decimal] = None
        self._signal_ema: Optional[Decimal] = None
        self._fast_mult = Decimal("2") / Decimal(str(fast_period + 1))
        self._slow_mult = Decimal("2") / Decimal(str(slow_period + 1))
        self._signal_mult = Decimal("2") / Decimal(str(signal_period + 1))

    def calculate(self, klines: list[Kline]) -> IndicatorResult:
        if len(klines) < self._slow_period:
            return IndicatorResult(values={}, is_ready=False)

        current_close = klines[-1].close

        if self._fast_ema is None:
            # Initialize EMAs
            fast_closes = [k.close for k in klines[-self._fast_period:]]
            slow_closes = [k.close for k in klines[-self._slow_period:]]
            self._fast_ema = sum(fast_closes) / Decimal(len(fast_closes))
            self._slow_ema = sum(slow_closes) / Decimal(len(slow_closes))
            self._signal_ema = self._fast_ema - self._slow_ema
        else:
            # Update EMAs
            self._fast_ema = (current_close - self._fast_ema) * self._fast_mult + self._fast_ema
            self._slow_ema = (current_close - self._slow_ema) * self._slow_mult + self._slow_ema

        macd_line = self._fast_ema - self._slow_ema

        if self._signal_ema is not None:
            self._signal_ema = (macd_line - self._signal_ema) * self._signal_mult + self._signal_ema
        else:
            self._signal_ema = macd_line

        histogram = macd_line - self._signal_ema

        return IndicatorResult(
            values={
                "macd": macd_line,
                "signal": self._signal_ema,
                "histogram": histogram,
            }
        )

    def reset(self) -> None:
        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None

    @property
    def warmup_period(self) -> int:
        return self._slow_period + self._signal_period


class IndicatorFactory:
    """
    Factory for creating indicator instances.

    Supports registration of custom indicator types.

    Example:
        factory = IndicatorFactory()
        indicator = factory.create(indicator_definition, resolved_params)
    """

    # Default indicator registry
    _REGISTRY: dict[str, type[BaseIndicator]] = {
        "BollingerBands": BollingerBandsIndicator,
        "Supertrend": SupertrendIndicator,
        "SMA": SMAIndicator,
        "EMA": EMAIndicator,
        "RSI": RSIIndicator,
        "ATR": ATRIndicator,
        "MACD": MACDIndicator,
    }

    def __init__(self):
        self._registry = self._REGISTRY.copy()

    def register(self, name: str, indicator_class: type[BaseIndicator]) -> None:
        """Register a custom indicator type."""
        self._registry[name] = indicator_class

    def create(
        self,
        definition: IndicatorDefinition,
        param_values: dict[str, Any],
    ) -> BaseIndicator:
        """
        Create an indicator instance from definition.

        Args:
            definition: Indicator definition from DSL
            param_values: Resolved parameter values

        Returns:
            Indicator instance

        Raises:
            IndicatorError: If indicator type is unknown
        """
        if definition.indicator_type not in self._registry:
            raise IndicatorError(f"Unknown indicator type: {definition.indicator_type}")

        indicator_class = self._registry[definition.indicator_type]
        resolved_params = definition.params.resolve(param_values)

        return indicator_class(**resolved_params)

    def get_supported_types(self) -> list[str]:
        """Get list of supported indicator types."""
        return list(self._registry.keys())

    def is_supported(self, indicator_type: str) -> bool:
        """Check if an indicator type is supported."""
        return indicator_type in self._registry
