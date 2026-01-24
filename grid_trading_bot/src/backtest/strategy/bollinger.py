"""
Bollinger Band Strategy Adapter.

Adapts the Bollinger Band strategy for the unified backtest framework.

Strategy Modes:
1. TREND_BOUNCE: Supertrend + BB combination
   - Entry: Supertrend bullish + price touches BB lower (LONG)
            Supertrend bearish + price touches BB upper (SHORT)
   - Exit: Supertrend flip or ATR stop loss

2. MEAN_REVERSION (default): Pure Bollinger Band mean reversion
   - Entry: Price touches/crosses BB lower (LONG), upper (SHORT)
   - Exit: Price reaches BB middle (SMA) or opposite band

VALIDATION STATUS: NOT PASSED (2024-01-05 ~ 2026-01-24)
- TREND_BOUNCE mode: 0% Walk-Forward consistency
- MEAN_REVERSION mode: Negative returns across all configurations
  * Exit at middle: -5% to -17% return, 36-44% win rate
  * Exit at opposite band: -1% to -6% return, 30-33% win rate
- Issue: Mean reversion conflicts with crypto's trending behavior
  * Price often doesn't return to mean before stop loss
  * Low win rates across all tested parameter combinations
- Tested: Various periods (14-30), std (1.5-3.0), stop losses (2-5%)
- Recommendation: Use GridFutures (range trading) or RSI instead
- Note: Bollinger Bands work better as volatility indicator than
        standalone trading strategy in crypto markets
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from ...bots.bollinger.indicators import BollingerCalculator
from ...core.models import Kline
from ..order import Signal
from ..position import Position
from ..result import Trade
from .base import BacktestContext, BacktestStrategy


class BollingerMode(str, Enum):
    """Bollinger Band strategy modes."""
    TREND_BOUNCE = "trend_bounce"      # Original: Supertrend + BB
    MEAN_REVERSION = "mean_reversion"  # Pure BB mean reversion


@dataclass
class BollingerStrategyConfig:
    """
    Configuration for Bollinger backtest strategy.

    Attributes:
        mode: Strategy mode (TREND_BOUNCE or MEAN_REVERSION)
        bb_period: Bollinger Band period
        bb_std: Standard deviation multiplier
        st_atr_period: Supertrend ATR period (TREND_BOUNCE mode)
        st_atr_multiplier: Supertrend ATR multiplier (TREND_BOUNCE mode)
        atr_stop_multiplier: ATR multiplier for stop loss
        bbw_lookback: BBW lookback for squeeze detection
        bbw_threshold_pct: BBW squeeze threshold percentile
        exit_at_middle: Exit at BB middle band (MEAN_REVERSION mode)
        stop_loss_pct: Stop loss percentage (MEAN_REVERSION mode)
    """

    mode: BollingerMode = BollingerMode.MEAN_REVERSION  # Default to mean reversion
    bb_period: int = 20
    bb_std: Decimal = Decimal("2.0")
    st_atr_period: int = 20
    st_atr_multiplier: Decimal = Decimal("3.5")
    atr_stop_multiplier: Decimal = Decimal("2.0")
    bbw_lookback: int = 200
    bbw_threshold_pct: int = 20
    exit_at_middle: bool = True  # Exit at SMA instead of opposite band
    stop_loss_pct: Decimal = Decimal("0.03")  # 3% stop loss for mean reversion


class SupertrendState:
    """
    Simple Supertrend calculator for backtest.

    Calculates Supertrend indicator without complex dependencies.
    """

    def __init__(self, atr_period: int = 20, atr_multiplier: Decimal = Decimal("3.5")):
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier
        self._trend: int = 0  # 1 = bullish, -1 = bearish
        self._supertrend_up: Optional[Decimal] = None
        self._supertrend_down: Optional[Decimal] = None
        self._prev_close: Optional[Decimal] = None

    @property
    def trend(self) -> int:
        """Get current trend direction."""
        return self._trend

    @property
    def supertrend_value(self) -> Optional[Decimal]:
        """Get current supertrend value."""
        if self._trend == 1:
            return self._supertrend_up
        elif self._trend == -1:
            return self._supertrend_down
        return None

    def calculate_atr(self, klines: list[Kline]) -> Optional[Decimal]:
        """Calculate ATR from klines."""
        if len(klines) < self._atr_period + 1:
            return None

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

        return sum(true_ranges) / Decimal(len(true_ranges))

    def update(self, klines: list[Kline]) -> int:
        """
        Update Supertrend state with latest klines.

        Returns:
            Current trend (1=bullish, -1=bearish, 0=unknown)
        """
        if len(klines) < self._atr_period + 1:
            return 0

        kline = klines[-1]
        atr = self.calculate_atr(klines)

        if atr is None:
            return 0

        # Calculate basic upper and lower bands
        hl2 = (kline.high + kline.low) / Decimal("2")
        basic_upper = hl2 + (self._atr_multiplier * atr)
        basic_lower = hl2 - (self._atr_multiplier * atr)

        # Initialize on first run
        if self._supertrend_up is None:
            self._supertrend_up = basic_lower
            self._supertrend_down = basic_upper
            self._trend = 1 if kline.close > hl2 else -1
            self._prev_close = kline.close
            return self._trend

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
        return self._trend

    def reset(self) -> None:
        """Reset state."""
        self._trend = 0
        self._supertrend_up = None
        self._supertrend_down = None
        self._prev_close = None


class BollingerBacktestStrategy(BacktestStrategy):
    """
    Bollinger Band + Supertrend strategy adapter.

    Implements the BOLLINGER_TREND strategy for the unified backtest framework.

    Example:
        config = BollingerStrategyConfig(bb_std=Decimal("3.0"))
        strategy = BollingerBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[BollingerStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self._config = config or BollingerStrategyConfig()

        # Initialize calculators
        self._bb_calculator = BollingerCalculator(
            period=self._config.bb_period,
            std_multiplier=self._config.bb_std,
            bbw_lookback=self._config.bbw_lookback,
            bbw_threshold_pct=self._config.bbw_threshold_pct,
        )
        self._supertrend = SupertrendState(
            atr_period=self._config.st_atr_period,
            atr_multiplier=self._config.st_atr_multiplier,
        )

        # State
        self._current_atr: Optional[Decimal] = None
        self._prev_trend: int = 0
        self._current_bands: Optional[object] = None  # Store for exit logic

    @property
    def config(self) -> BollingerStrategyConfig:
        """Get strategy configuration."""
        return self._config

    def warmup_period(self) -> int:
        """Return warmup period needed for indicators."""
        return max(
            self._config.bb_period + self._config.bbw_lookback,
            self._config.st_atr_period + 50,
        )

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        TREND_BOUNCE Mode:
        - LONG: Supertrend bullish (trend=1) AND price <= BB lower band
        - SHORT: Supertrend bearish (trend=-1) AND price >= BB upper band

        MEAN_REVERSION Mode:
        - LONG: Price touches/crosses BB lower band
        - SHORT: Price touches/crosses BB upper band

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Signal if entry conditions met, None otherwise
        """
        # Skip if already in position
        if context.has_position:
            return None

        klines = context.klines

        # Calculate indicators
        try:
            bands, bbw = self._bb_calculator.get_all(klines)
        except Exception:
            return None

        # Store bands for exit logic
        self._current_bands = bands

        # Calculate ATR for stop loss
        self._current_atr = self._bb_calculator.calculate_atr(
            klines, self._config.st_atr_period
        )

        # Skip during squeeze (volatility too low)
        if bbw.is_squeeze:
            return None

        current_price = kline.close

        # MEAN_REVERSION Mode: Pure BB mean reversion
        if self._config.mode == BollingerMode.MEAN_REVERSION:
            # Long entry: Price at lower band (expect reversion to mean)
            if current_price <= bands.lower:
                stop_loss = current_price * (Decimal("1") - self._config.stop_loss_pct)
                # Take profit at middle band (SMA) or upper band
                take_profit = bands.middle if self._config.exit_at_middle else bands.upper

                return Signal.long_entry(
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="bb_lower_mean_reversion",
                )

            # Short entry: Price at upper band (expect reversion to mean)
            elif current_price >= bands.upper:
                stop_loss = current_price * (Decimal("1") + self._config.stop_loss_pct)
                # Take profit at middle band (SMA) or lower band
                take_profit = bands.middle if self._config.exit_at_middle else bands.lower

                return Signal.short_entry(
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="bb_upper_mean_reversion",
                )

            return None

        # TREND_BOUNCE Mode: Original Supertrend + BB logic
        # Update Supertrend
        trend = self._supertrend.update(klines)

        # Entry conditions
        if trend == 1 and current_price <= bands.lower:
            # Long entry: Supertrend bullish + price at lower band
            stop_loss = None
            if self._current_atr:
                stop_loss = bands.lower - (self._current_atr * self._config.atr_stop_multiplier)

            return Signal.long_entry(
                price=bands.lower,
                stop_loss=stop_loss,
                take_profit=None,  # Exit on Supertrend flip
                reason="supertrend_bullish_bb_lower",
            )

        elif trend == -1 and current_price >= bands.upper:
            # Short entry: Supertrend bearish + price at upper band
            stop_loss = None
            if self._current_atr:
                stop_loss = bands.upper + (self._current_atr * self._config.atr_stop_multiplier)

            return Signal.short_entry(
                price=bands.upper,
                stop_loss=stop_loss,
                take_profit=None,  # Exit on Supertrend flip
                reason="supertrend_bearish_bb_upper",
            )

        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        TREND_BOUNCE Mode: Exit on Supertrend flip
        MEAN_REVERSION Mode: Exit handled by take_profit/stop_loss

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal if conditions met, None otherwise
        """
        # MEAN_REVERSION mode: exits handled by take_profit/stop_loss
        if self._config.mode == BollingerMode.MEAN_REVERSION:
            # No additional exit logic - rely on TP/SL
            return None

        # TREND_BOUNCE Mode: Exit on Supertrend flip
        trend = self._supertrend.trend

        if position.side == "LONG" and trend == -1:
            return Signal.close_all(reason="supertrend_flip")

        elif position.side == "SHORT" and trend == 1:
            return Signal.close_all(reason="supertrend_flip")

        return None

    def update_trailing_stop(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Decimal]:
        """
        Update trailing stop using ATR.

        Implements trailing stop based on ATR distance from
        max favorable price.
        """
        if self._current_atr is None:
            return None

        # Calculate new stop based on max favorable price
        trail_distance = self._current_atr * self._config.atr_stop_multiplier

        if position.side == "LONG":
            new_stop = position.max_favorable_price - trail_distance
            # Only update if better than current stop
            if position.stop_loss is None or new_stop > position.stop_loss:
                return new_stop
        else:  # SHORT
            new_stop = position.max_favorable_price + trail_distance
            if position.stop_loss is None or new_stop < position.stop_loss:
                return new_stop

        return None

    def on_position_opened(self, position: Position) -> None:
        """Track trend at entry."""
        self._prev_trend = self._supertrend.trend

    def on_position_closed(self, trade: Trade) -> None:
        """Reset tracking after trade."""
        pass

    def reset(self) -> None:
        """Reset strategy state."""
        self._bb_calculator.reset()
        self._supertrend.reset()
        self._current_atr = None
        self._prev_trend = 0
        self._current_bands = None
