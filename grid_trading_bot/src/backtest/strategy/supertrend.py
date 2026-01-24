"""
Supertrend Strategy Adapter.

Adapts the Supertrend trend-following strategy for the unified backtest framework.

Strategy Logic:
- Entry: Supertrend flip (bullish = LONG, bearish = SHORT)
- Filters: RSI overbought/oversold, ADX trend strength
- Exit: Supertrend flip or stop loss

VALIDATION STATUS: NOT PASSED (2024-01-05 ~ 2026-01-24)
- Multiple configurations tested with RSI and ADX filters
- Walk-Forward consistency: 0-17% (failed overfitting test)
- Issue: Trend-following struggles in crypto markets due to:
  * Many false breakouts in ranging periods
  * High volatility causing frequent whipsaws
  * Low win rate (2-10%) even with filters
- Tested improvements: ADX filter (>25), RSI filter, various ATR settings
  * ADX filter reduced false signals but also reduced trade count
  * All configurations failed to achieve positive returns with >30 trades
- Recommendation: Use GridFutures (range) or RSI (momentum) instead
- Note: Pure trend-following strategies generally don't work well in
        crypto markets which spend ~70% of time ranging
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from ...bots.supertrend.indicators import SupertrendIndicator
from ...core.models import Kline
from ..order import Signal
from ..position import Position
from ..result import Trade
from .base import BacktestContext, BacktestStrategy


@dataclass
class SupertrendStrategyConfig:
    """
    Configuration for Supertrend backtest strategy.

    Attributes:
        atr_period: ATR calculation period
        atr_multiplier: ATR multiplier for bands
        stop_loss_pct: Stop loss percentage
        use_rsi_filter: Enable RSI filter
        rsi_period: RSI calculation period
        rsi_overbought: RSI overbought threshold (don't go long above this)
        rsi_oversold: RSI oversold threshold (don't go short below this)
    """

    atr_period: int = 25
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.0"))
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.03"))
    use_rsi_filter: bool = True
    rsi_period: int = 14
    rsi_overbought: int = 60
    rsi_oversold: int = 40
    use_adx_filter: bool = False  # ADX trend strength filter
    adx_period: int = 14
    adx_threshold: int = 25  # Only trade when ADX > threshold


class RSICalculator:
    """Simple RSI calculator for strategy filtering."""

    def __init__(self, period: int = 14):
        self._period = period

    def calculate(self, closes: list[Decimal]) -> Optional[Decimal]:
        """
        Calculate RSI from close prices.

        Args:
            closes: List of close prices (at least period + 1)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(closes) < self._period + 1:
            return None

        # Calculate price changes
        changes = []
        for i in range(1, len(closes)):
            changes.append(closes[i] - closes[i - 1])

        # Take last 'period' changes
        recent_changes = changes[-self._period:]

        gains = [c for c in recent_changes if c > 0]
        losses = [abs(c) for c in recent_changes if c < 0]

        avg_gain = sum(gains) / Decimal(self._period) if gains else Decimal("0")
        avg_loss = sum(losses) / Decimal(self._period) if losses else Decimal("0")

        if avg_loss == 0:
            return Decimal("100") if avg_gain > 0 else Decimal("50")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi


class ADXCalculator:
    """ADX (Average Directional Index) calculator for trend strength filtering."""

    def __init__(self, period: int = 14):
        self._period = period

    def calculate(self, highs: list[Decimal], lows: list[Decimal], closes: list[Decimal]) -> Optional[Decimal]:
        """
        Calculate ADX from OHLC data.

        ADX measures trend strength:
        - ADX > 25: Strong trend (good for trend following)
        - ADX < 20: Weak trend / ranging (avoid trading)

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices

        Returns:
            ADX value (0-100) or None if insufficient data
        """
        n = len(closes)
        if n < self._period * 2 + 1:
            return None

        # Calculate True Range and Directional Movement
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, n):
            high = highs[i]
            low = lows[i]
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            prev_close = closes[i - 1]

            # True Range
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_list.append(tr)

            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low

            plus_dm = up_move if up_move > down_move and up_move > 0 else Decimal("0")
            minus_dm = down_move if down_move > up_move and down_move > 0 else Decimal("0")

            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        # Calculate smoothed averages (Wilder's smoothing)
        period = self._period

        # Initial sums
        atr = sum(tr_list[:period]) / Decimal(period)
        plus_di_sum = sum(plus_dm_list[:period])
        minus_di_sum = sum(minus_dm_list[:period])

        # Calculate DI values over time
        dx_list = []

        for i in range(period, len(tr_list)):
            # Wilder's smoothing
            atr = (atr * Decimal(period - 1) + tr_list[i]) / Decimal(period)
            plus_di_sum = (plus_di_sum * Decimal(period - 1) + plus_dm_list[i]) / Decimal(period)
            minus_di_sum = (minus_di_sum * Decimal(period - 1) + minus_dm_list[i]) / Decimal(period)

            if atr == 0:
                continue

            plus_di = (plus_di_sum / atr) * Decimal("100")
            minus_di = (minus_di_sum / atr) * Decimal("100")

            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_list.append(Decimal("0"))
            else:
                dx = abs(plus_di - minus_di) / di_sum * Decimal("100")
                dx_list.append(dx)

        if len(dx_list) < period:
            return None

        # ADX is smoothed average of DX
        adx = sum(dx_list[-period:]) / Decimal(period)
        return adx


class SupertrendBacktestStrategy(BacktestStrategy):
    """
    Supertrend trend-following strategy adapter.

    Enters on Supertrend direction flip, with optional RSI filter.
    Exits on Supertrend flip or stop loss.

    Example:
        config = SupertrendStrategyConfig(atr_period=25)
        strategy = SupertrendBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[SupertrendStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self._config = config or SupertrendStrategyConfig()

        # Initialize indicator
        self._supertrend = SupertrendIndicator(
            atr_period=self._config.atr_period,
            atr_multiplier=self._config.atr_multiplier,
        )

        # RSI calculator
        self._rsi = RSICalculator(period=self._config.rsi_period)

        # ADX calculator
        self._adx = ADXCalculator(period=self._config.adx_period)

        # State tracking
        self._prev_trend: int = 0
        self._current_rsi: Optional[Decimal] = None
        self._current_adx: Optional[Decimal] = None

    @property
    def config(self) -> SupertrendStrategyConfig:
        """Get strategy configuration."""
        return self._config

    def warmup_period(self) -> int:
        """Return warmup period needed for indicators."""
        return max(
            self._config.atr_period + 20,
            self._config.rsi_period + 10,
            self._config.adx_period * 2 + 10  # ADX needs 2x period
        )

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        Entry Logic:
        - LONG: Supertrend flips to bullish (trend changes from -1 to 1)
        - SHORT: Supertrend flips to bearish (trend changes from 1 to -1)

        RSI Filter (if enabled):
        - Don't go LONG when RSI > overbought (60)
        - Don't go SHORT when RSI < oversold (40)

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Signal if entry conditions met, None otherwise
        """
        # Skip if already in position
        if context.has_position:
            return None

        # Update Supertrend
        st_data = self._supertrend.update(kline)
        if st_data is None:
            return None

        current_trend = st_data.trend

        # Calculate RSI if filter enabled
        if self._config.use_rsi_filter:
            closes = context.get_closes(self._config.rsi_period + 5)
            self._current_rsi = self._rsi.calculate(closes)

        # Calculate ADX if filter enabled
        if self._config.use_adx_filter:
            klines_window = context.get_klines_window(self._config.adx_period * 2 + 5)
            if len(klines_window) >= self._config.adx_period * 2 + 1:
                highs = [k.high for k in klines_window]
                lows = [k.low for k in klines_window]
                closes = [k.close for k in klines_window]
                self._current_adx = self._adx.calculate(highs, lows, closes)

            # Skip if ADX is below threshold (weak trend)
            if self._current_adx is not None:
                if self._current_adx < self._config.adx_threshold:
                    self._prev_trend = current_trend
                    return None

        # Check for trend flip
        if self._prev_trend != 0 and current_trend != self._prev_trend:
            # Trend flipped!
            current_price = kline.close

            if current_trend == 1:
                # Flipped to bullish - consider LONG
                # RSI filter: don't go long if overbought
                if self._config.use_rsi_filter and self._current_rsi is not None:
                    if self._current_rsi > self._config.rsi_overbought:
                        self._prev_trend = current_trend
                        return None

                stop_loss = current_price * (Decimal("1") - self._config.stop_loss_pct)

                self._prev_trend = current_trend
                return Signal.long_entry(
                    price=current_price,
                    stop_loss=stop_loss,
                    reason="supertrend_bullish_flip",
                )

            elif current_trend == -1:
                # Flipped to bearish - consider SHORT
                # RSI filter: don't go short if oversold
                if self._config.use_rsi_filter and self._current_rsi is not None:
                    if self._current_rsi < self._config.rsi_oversold:
                        self._prev_trend = current_trend
                        return None

                stop_loss = current_price * (Decimal("1") + self._config.stop_loss_pct)

                self._prev_trend = current_trend
                return Signal.short_entry(
                    price=current_price,
                    stop_loss=stop_loss,
                    reason="supertrend_bearish_flip",
                )

        self._prev_trend = current_trend
        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        Exit on Supertrend flip (trend reversal).

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal if Supertrend flipped, None otherwise
        """
        # Get current trend
        current_trend = self._supertrend.trend

        # Check for Supertrend flip
        if position.side == "LONG" and current_trend == -1:
            return Signal.close_all(reason="supertrend_flip_bearish")

        elif position.side == "SHORT" and current_trend == 1:
            return Signal.close_all(reason="supertrend_flip_bullish")

        return None

    def on_position_opened(self, position: Position) -> None:
        """Track state at entry."""
        pass

    def on_position_closed(self, trade: Trade) -> None:
        """Reset tracking after trade."""
        pass

    def reset(self) -> None:
        """Reset strategy state."""
        self._supertrend.reset()
        self._prev_trend = 0
        self._current_rsi = None
        self._current_adx = None
