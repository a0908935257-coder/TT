"""
Multi-Timeframe Trend Following Strategy.

Example strategy demonstrating multi-timeframe capabilities.

Strategy Logic:
- Use 4h timeframe to determine trend direction (SMA crossover)
- Use 1h timeframe for entry timing (pullback to EMA)
- Enter when 1h price pulls back to EMA in direction of 4h trend

This demonstrates:
1. How to define multiple timeframes
2. How to access different timeframe data
3. How to combine signals from multiple timeframes
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ...core.models import Kline
from ..multi_timeframe import MultiTimeframeContext, MultiTimeframeStrategy
from ..order import Signal


@dataclass
class MultiTFTrendConfig:
    """
    Configuration for Multi-Timeframe Trend Strategy.

    Attributes:
        trend_sma_period: SMA period for trend detection on higher TF
        entry_ema_period: EMA period for entry timing on base TF
        pullback_threshold: How close to EMA for valid pullback (%)
        stop_loss_atr_mult: ATR multiplier for stop loss
        take_profit_atr_mult: ATR multiplier for take profit
        atr_period: ATR calculation period
    """

    trend_sma_period: int = 20
    entry_ema_period: int = 21
    pullback_threshold: Decimal = Decimal("0.005")  # 0.5%
    stop_loss_atr_mult: Decimal = Decimal("2.0")
    take_profit_atr_mult: Decimal = Decimal("3.0")
    atr_period: int = 14


class MultiTFTrendStrategy(MultiTimeframeStrategy):
    """
    Multi-Timeframe Trend Following Strategy.

    Uses higher timeframe (4h) for trend direction and lower
    timeframe (1h) for entry timing.

    Example:
        config = MultiTFTrendConfig(
            trend_sma_period=20,
            entry_ema_period=21,
        )
        strategy = MultiTFTrendStrategy(config)
        engine = MultiTimeframeEngine(backtest_config)
        result = engine.run(klines_1h, strategy)
    """

    def __init__(self, config: Optional[MultiTFTrendConfig] = None):
        """Initialize strategy."""
        self._config = config or MultiTFTrendConfig()
        self._prev_trend: Optional[int] = None  # 1=up, -1=down, 0=neutral

    def timeframes(self) -> list[str]:
        """Return required timeframes: 1h (base) and 4h (trend)."""
        return ["1h", "4h"]

    def warmup_periods(self) -> dict[str, int]:
        """Return warmup periods for each timeframe."""
        return {
            "1h": max(self._config.entry_ema_period, self._config.atr_period) + 10,
            "4h": self._config.trend_sma_period + 5,
        }

    def warmup_period(self) -> int:
        """Return base timeframe warmup period."""
        return self.warmup_periods()["1h"]

    def reset(self) -> None:
        """Reset strategy state."""
        self._prev_trend = None

    def on_kline(
        self,
        kline: Kline,
        context: MultiTimeframeContext
    ) -> Optional[Signal]:
        """
        Process kline with multi-timeframe context.

        Args:
            kline: Current 1h kline
            context: Multi-timeframe context with 1h and 4h data

        Returns:
            Trading signal or None
        """
        # Already have position
        if context.has_position:
            return None

        # Get 4h data for trend
        tf_4h = context.get_timeframe("4h")
        if not tf_4h or tf_4h.bars_available < self._config.trend_sma_period:
            return None

        # Calculate 4h SMA for trend
        closes_4h = tf_4h.get_closes(self._config.trend_sma_period + 1)
        if len(closes_4h) < self._config.trend_sma_period + 1:
            return None

        sma_current = self._calculate_sma(closes_4h[-self._config.trend_sma_period:])
        sma_prev = self._calculate_sma(closes_4h[-self._config.trend_sma_period - 1:-1])

        if sma_current is None or sma_prev is None:
            return None

        # Determine trend: price above rising SMA = uptrend
        current_4h_close = tf_4h.current_price
        if current_4h_close is None:
            return None

        trend = 0
        if current_4h_close > sma_current and sma_current > sma_prev:
            trend = 1  # Uptrend
        elif current_4h_close < sma_current and sma_current < sma_prev:
            trend = -1  # Downtrend

        # Get 1h data for entry
        tf_1h = context.get_timeframe("1h")
        if not tf_1h or tf_1h.bars_available < self._config.entry_ema_period:
            return None

        closes_1h = tf_1h.get_closes(self._config.entry_ema_period + 1)
        if len(closes_1h) < self._config.entry_ema_period:
            return None

        # Calculate 1h EMA
        ema = self._calculate_ema(closes_1h, self._config.entry_ema_period)
        if ema is None:
            return None

        current_price = kline.close

        # Calculate ATR for stops
        atr = self._calculate_atr(tf_1h.get_klines_window(self._config.atr_period + 1))
        if atr is None or atr <= 0:
            return None

        # Entry conditions
        pullback_pct = abs(current_price - ema) / ema

        # Long entry: Uptrend + price pulls back to EMA
        if trend == 1 and pullback_pct <= self._config.pullback_threshold:
            if current_price >= ema:  # Must be at or above EMA
                stop_loss = current_price - (atr * self._config.stop_loss_atr_mult)
                take_profit = current_price + (atr * self._config.take_profit_atr_mult)

                return Signal.long_entry(
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="mtf_trend_long",
                )

        # Short entry: Downtrend + price pulls back to EMA
        if trend == -1 and pullback_pct <= self._config.pullback_threshold:
            if current_price <= ema:  # Must be at or below EMA
                stop_loss = current_price + (atr * self._config.stop_loss_atr_mult)
                take_profit = current_price - (atr * self._config.take_profit_atr_mult)

                return Signal.short_entry(
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="mtf_trend_short",
                )

        return None

    def _calculate_sma(self, prices: list[Decimal]) -> Optional[Decimal]:
        """Calculate Simple Moving Average."""
        if not prices:
            return None
        return sum(prices) / Decimal(len(prices))

    def _calculate_ema(
        self,
        prices: list[Decimal],
        period: int
    ) -> Optional[Decimal]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None

        multiplier = Decimal("2") / Decimal(period + 1)

        # Initialize with SMA
        ema = sum(prices[:period]) / Decimal(period)

        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_atr(self, klines: list[Kline]) -> Optional[Decimal]:
        """Calculate Average True Range."""
        if len(klines) < 2:
            return None

        tr_values = []
        for i in range(1, len(klines)):
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        if not tr_values:
            return None

        return sum(tr_values) / Decimal(len(tr_values))
