"""
Bollinger Band Strategy Adapter.

Adapts the Bollinger Band strategy for the unified backtest framework.

Strategy Modes:
1. TREND_BOUNCE (legacy): Supertrend + BB combination
   - Status: NOT PASSED (0% Walk-Forward consistency)

2. MEAN_REVERSION (legacy): Pure BB mean reversion
   - Status: NOT PASSED (negative returns)

3. SQUEEZE_BREAKOUT (default): BBW squeeze breakout strategy
   - Entry: When BBW is at historical low (squeeze) and price breaks out
   - Logic: Low volatility (squeeze) precedes high volatility (breakout)
   - Exit: Trailing stop or fixed take profit

VALIDATION STATUS: NOT PASSED (2024-01-05 ~ 2026-01-24)
- All modes tested with multiple parameter combinations
- SQUEEZE_BREAKOUT: Too few signals, 0-17% Walk-Forward consistency
- Issue: Bollinger Bands don't provide reliable signals in crypto markets
- Recommendation: Use GridFutures, RSI, or Supertrend TREND_GRID instead

Note: Bollinger Bands may work better as a volatility indicator for
other strategies rather than as a standalone trading signal generator.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from ...bots.bollinger.indicators import BollingerCalculator
from ...core.models import Kline
from ..order import Signal
from ..position import Position
from ..result import Trade
from .base import BacktestContext, BacktestStrategy


class BollingerMode(str, Enum):
    """Bollinger Band strategy modes."""
    TREND_BOUNCE = "trend_bounce"          # Legacy: Supertrend + BB
    MEAN_REVERSION = "mean_reversion"      # Legacy: Pure BB mean reversion
    SQUEEZE_BREAKOUT = "squeeze_breakout"  # New: BBW squeeze breakout


@dataclass
class BollingerStrategyConfig:
    """
    Configuration for Bollinger backtest strategy.

    Attributes:
        mode: Strategy mode
        bb_period: Bollinger Band period
        bb_std: Standard deviation multiplier
        bbw_lookback: BBW lookback for squeeze detection
        bbw_squeeze_pct: Percentile threshold for squeeze (lower = tighter squeeze)
        breakout_confirmation: Bars to confirm breakout
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        use_trailing_stop: Enable trailing stop
        trailing_stop_pct: Trailing stop percentage from peak
    """

    mode: BollingerMode = BollingerMode.SQUEEZE_BREAKOUT
    bb_period: int = 20
    bb_std: Decimal = Decimal("2.0")
    bbw_lookback: int = 100
    bbw_squeeze_pct: int = 20  # Bottom 20% of BBW = squeeze
    breakout_confirmation: int = 1  # Bars to confirm breakout
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.06")
    use_trailing_stop: bool = True
    trailing_stop_pct: Decimal = Decimal("0.015")  # 1.5% trailing


class BollingerBacktestStrategy(BacktestStrategy):
    """
    Bollinger Band strategy adapter with SQUEEZE_BREAKOUT mode.

    SQUEEZE_BREAKOUT Strategy:
    - Monitors Bollinger Band Width (BBW) for squeeze conditions
    - Squeeze = BBW in bottom percentile of lookback period
    - When squeeze detected and price breaks out of BB:
      * Break above upper band = LONG
      * Break below lower band = SHORT
    - Uses trailing stop or fixed TP/SL for exits

    Example:
        # Use default SQUEEZE_BREAKOUT mode
        strategy = BollingerBacktestStrategy()
        result = engine.run(klines, strategy)

        # Custom configuration
        config = BollingerStrategyConfig(
            bb_period=20,
            bbw_squeeze_pct=20,
            take_profit_pct=Decimal("0.08"),
        )
        strategy = BollingerBacktestStrategy(config)
    """

    def __init__(self, config: Optional[BollingerStrategyConfig] = None):
        """Initialize strategy."""
        self._config = config or BollingerStrategyConfig()

        # Initialize BB calculator
        self._bb_calculator = BollingerCalculator(
            period=self._config.bb_period,
            std_multiplier=self._config.bb_std,
            bbw_lookback=self._config.bbw_lookback,
            bbw_threshold_pct=self._config.bbw_squeeze_pct,
        )

        # State
        self._bbw_history: List[Decimal] = []
        self._in_squeeze = False
        self._squeeze_start_bar = 0
        self._bar_count = 0
        self._current_bands = None
        self._entry_price: Optional[Decimal] = None
        self._max_favorable_price: Optional[Decimal] = None

    @property
    def config(self) -> BollingerStrategyConfig:
        """Get strategy configuration."""
        return self._config

    def warmup_period(self) -> int:
        """Return warmup period needed for indicators."""
        return self._config.bb_period + self._config.bbw_lookback + 10

    def _is_squeeze(self, bbw: Decimal) -> bool:
        """Check if current BBW indicates a squeeze."""
        if len(self._bbw_history) < self._config.bbw_lookback:
            return False

        # Calculate percentile
        sorted_bbw = sorted(self._bbw_history[-self._config.bbw_lookback:])
        threshold_idx = int(len(sorted_bbw) * self._config.bbw_squeeze_pct / 100)
        threshold_idx = max(0, min(threshold_idx, len(sorted_bbw) - 1))
        threshold = sorted_bbw[threshold_idx]

        return bbw <= threshold

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        SQUEEZE_BREAKOUT Logic:
        1. Detect squeeze (BBW in bottom percentile)
        2. Wait for breakout (price breaks BB band)
        3. Enter in breakout direction
        """
        self._bar_count += 1
        klines = context.klines

        # Calculate indicators
        try:
            bands, bbw = self._bb_calculator.get_all(klines)
        except Exception:
            return None

        self._current_bands = bands

        # Track BBW history
        if bbw.bbw is not None:
            self._bbw_history.append(bbw.bbw)
            if len(self._bbw_history) > self._config.bbw_lookback + 50:
                self._bbw_history = self._bbw_history[-self._config.bbw_lookback - 50:]

        # Skip if already in position
        if context.has_position:
            return None

        current_price = kline.close

        # Check for squeeze
        is_squeeze = bbw.is_squeeze if bbw.is_squeeze is not None else self._is_squeeze(bbw.bbw)

        if is_squeeze:
            if not self._in_squeeze:
                self._in_squeeze = True
                self._squeeze_start_bar = self._bar_count
        else:
            # Squeeze released - check for breakout
            if self._in_squeeze:
                bars_in_squeeze = self._bar_count - self._squeeze_start_bar

                # Need minimum time in squeeze
                if bars_in_squeeze >= 3:
                    # Check breakout direction
                    if current_price > bands.upper:
                        # Bullish breakout
                        self._in_squeeze = False
                        stop_loss = current_price * (Decimal("1") - self._config.stop_loss_pct)
                        take_profit = current_price * (Decimal("1") + self._config.take_profit_pct)

                        return Signal.long_entry(
                            price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason="squeeze_breakout_long",
                        )

                    elif current_price < bands.lower:
                        # Bearish breakout
                        self._in_squeeze = False
                        stop_loss = current_price * (Decimal("1") + self._config.stop_loss_pct)
                        take_profit = current_price * (Decimal("1") - self._config.take_profit_pct)

                        return Signal.short_entry(
                            price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason="squeeze_breakout_short",
                        )

            self._in_squeeze = False

        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        Uses trailing stop if enabled.
        """
        # Trailing stop is handled by update_trailing_stop
        return None

    def update_trailing_stop(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Decimal]:
        """
        Update trailing stop based on favorable price movement.
        """
        if not self._config.use_trailing_stop:
            return None

        current_price = kline.close

        if position.side == "LONG":
            # Track max favorable price
            if self._max_favorable_price is None or current_price > self._max_favorable_price:
                self._max_favorable_price = current_price

            # Calculate trailing stop
            new_stop = self._max_favorable_price * (Decimal("1") - self._config.trailing_stop_pct)

            # Only update if better than current
            if position.stop_loss is None or new_stop > position.stop_loss:
                return new_stop

        else:  # SHORT
            if self._max_favorable_price is None or current_price < self._max_favorable_price:
                self._max_favorable_price = current_price

            new_stop = self._max_favorable_price * (Decimal("1") + self._config.trailing_stop_pct)

            if position.stop_loss is None or new_stop < position.stop_loss:
                return new_stop

        return None

    def on_position_opened(self, position: Position) -> None:
        """Track state at entry."""
        self._entry_price = position.entry_price
        self._max_favorable_price = position.entry_price

    def on_position_closed(self, trade: Trade) -> None:
        """Reset tracking after trade."""
        self._entry_price = None
        self._max_favorable_price = None

    def reset(self) -> None:
        """Reset strategy state."""
        self._bb_calculator.reset()
        self._bbw_history = []
        self._in_squeeze = False
        self._squeeze_start_bar = 0
        self._bar_count = 0
        self._current_bands = None
        self._entry_price = None
        self._max_favorable_price = None
