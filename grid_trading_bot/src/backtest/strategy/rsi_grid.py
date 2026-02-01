"""
RSI-Grid Hybrid Strategy v2 - Complete Architecture Overhaul.

Key changes from v1:
- Grid level reset when no position (prevents permanent level consumption)
- RSI continuous score via tanh (replaces hard RSI zone cutoffs)
- Timeout exit for stale positions
- Optional trailing stop
- Simplified config (removed unused protective features)

Design Goals:
- Annual return > 40% (2x leverage)
- 3-7 trades per day (2000+ trades in 730 days)
- Pass Walk-Forward OOS validation
- Pass Monte Carlo overfit test
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional
import math

from ...core.models import Kline
from ..order import Signal
from ..position import Position
from .base import BacktestContext, BacktestStrategy


@dataclass
class RSIGridStrategyConfig:
    """
    Configuration for RSI-Grid v2 strategy.

    Simplified from v1: removed oversold/overbought_level, trend_sma_period,
    use_trend_filter, use_hysteresis, hysteresis_pct, use_signal_cooldown,
    cooldown_bars, max_positions, position_size_pct, max_stop_loss_pct.
    """

    # RSI Parameters
    rsi_period: int = 14
    rsi_block_threshold: float = 0.7  # tanh score threshold to block counter-trend

    # ATR / Grid Parameters
    atr_period: int = 14
    grid_count: int = 15
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.5"))

    # Risk Management
    stop_loss_atr_mult: Decimal = field(default_factory=lambda: Decimal("1.5"))
    take_profit_grids: int = 1
    max_hold_bars: int = 24  # timeout exit after N bars (if losing)

    # Trailing Stop (optional)
    use_trailing_stop: bool = False
    trailing_activate_pct: float = 0.01   # activate after 1% profit
    trailing_distance_pct: float = 0.005  # trail by 0.5%

    # Volatility Regime Filter
    use_volatility_filter: bool = True
    vol_atr_baseline_period: int = 200    # 長期 ATR 基線
    vol_ratio_low: float = 0.5           # ATR ratio 下限
    vol_ratio_high: float = 2.0          # ATR ratio 上限

    def __post_init__(self):
        """Normalize Decimal types."""
        for field_name in ['atr_multiplier', 'stop_loss_atr_mult']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))


@dataclass
class GridLevel:
    """Individual grid level."""
    index: int
    price: Decimal
    is_long_filled: bool = False
    is_short_filled: bool = False


class RSIGridBacktestStrategy(BacktestStrategy):
    """
    RSI-Grid Hybrid Strategy v2.

    Architecture changes:
    A. Grid Reset - filled levels reset when no position (like grid_futures.py)
    B. RSI Continuous Score - tanh((rsi-50)/50*1.5) replaces hard zone cutoffs
    C. Entry: grid touch + RSI not blocking + no position
    D. Exit: RSI extreme reversal, timeout, trailing stop, SL/TP by engine
    """

    def __init__(self, config: Optional[RSIGridStrategyConfig] = None):
        self._config = config or RSIGridStrategyConfig()

        # RSI state
        self._current_rsi: Optional[Decimal] = None
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None

        # Grid state
        self._grid_levels: List[GridLevel] = []
        self._upper_price: Optional[Decimal] = None
        self._lower_price: Optional[Decimal] = None
        self._grid_spacing: Optional[Decimal] = None
        self._current_atr: Optional[Decimal] = None
        self._grid_initialized = False

        # Position tracking
        self._entry_bar_index: int = 0

        # Volatility filter state
        self._atr_history: List[Decimal] = []
        self._atr_baseline: Optional[Decimal] = None

    def warmup_period(self) -> int:
        return max(self._config.rsi_period, self._config.atr_period) + 10

    def reset(self) -> None:
        self._current_rsi = None
        self._avg_gain = None
        self._avg_loss = None
        self._grid_levels = []
        self._upper_price = None
        self._lower_price = None
        self._grid_spacing = None
        self._current_atr = None
        self._grid_initialized = False
        self._entry_bar_index = 0
        self._atr_history = []
        self._atr_baseline = None

    def _calculate_rsi(self, closes: List[Decimal]) -> Optional[Decimal]:
        """Calculate RSI using Wilder's smoothing."""
        period = self._config.rsi_period
        if len(closes) < period + 1:
            return None

        if self._avg_gain is None or self._avg_loss is None:
            gains = []
            losses = []
            for i in range(1, period + 1):
                change = closes[i] - closes[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(Decimal("0"))
                else:
                    gains.append(Decimal("0"))
                    losses.append(abs(change))

            self._avg_gain = sum(gains) / Decimal(period)
            self._avg_loss = sum(losses) / Decimal(period)

            for i in range(period + 1, len(closes)):
                change = closes[i] - closes[i - 1]
                if change > 0:
                    current_gain = change
                    current_loss = Decimal("0")
                else:
                    current_gain = Decimal("0")
                    current_loss = abs(change)

                self._avg_gain = (self._avg_gain * (period - 1) + current_gain) / period
                self._avg_loss = (self._avg_loss * (period - 1) + current_loss) / period
        else:
            change = closes[-1] - closes[-2]
            if change > 0:
                current_gain = change
                current_loss = Decimal("0")
            else:
                current_gain = Decimal("0")
                current_loss = abs(change)

            self._avg_gain = (self._avg_gain * (period - 1) + current_gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + current_loss) / period

        if self._avg_loss == 0:
            return Decimal("100") if self._avg_gain > 0 else Decimal("50")

        rs = self._avg_gain / self._avg_loss
        return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

    def _rsi_score(self, rsi: Decimal) -> float:
        """
        Continuous RSI bias score using tanh.

        Returns value in [-1, +1]:
        - Positive = bullish (RSI > 50)
        - Negative = bearish (RSI < 50)
        - Near 0 = neutral
        """
        return math.tanh((float(rsi) - 50) / 50 * 1.5)

    def _calculate_atr(self, klines: List[Kline], period: int) -> Optional[Decimal]:
        """Calculate Average True Range."""
        if len(klines) < period + 1:
            return None

        tr_values = []
        for i in range(-period, 0):
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return sum(tr_values) / Decimal(period)

    def _reset_filled_levels(self) -> None:
        """Reset all filled grid levels when no position is held."""
        for level in self._grid_levels:
            level.is_long_filled = False
            level.is_short_filled = False

    def _initialize_grid(self, klines: List[Kline], current_price: Decimal) -> None:
        """Initialize grid levels around current price."""
        self._current_atr = self._calculate_atr(klines, self._config.atr_period)

        if self._current_atr and self._current_atr > 0:
            range_size = self._current_atr * self._config.atr_multiplier
        else:
            range_size = current_price * Decimal("0.05")

        self._upper_price = current_price + range_size
        self._lower_price = current_price - range_size

        if self._config.grid_count <= 0:
            self._grid_spacing = range_size
        else:
            self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)

        self._grid_levels = []
        for i in range(self._config.grid_count + 1):
            price = self._lower_price + (self._grid_spacing * Decimal(i))
            self._grid_levels.append(GridLevel(index=i, price=price))

        self._grid_initialized = True

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding (price out of range)."""
        if not self._grid_initialized:
            return True
        if current_price > self._upper_price or current_price < self._lower_price:
            return True
        return False

    def _find_current_grid_index(self, price: Decimal) -> Optional[int]:
        """Find the grid level index for the current price."""
        if not self._grid_levels:
            return None

        for i, level in enumerate(self._grid_levels):
            if i < len(self._grid_levels) - 1:
                if level.price <= price < self._grid_levels[i + 1].price:
                    return i

        if price >= self._grid_levels[-1].price:
            return len(self._grid_levels) - 1

        return 0

    def _update_volatility_baseline(self, current_atr: Decimal) -> None:
        """更新 ATR 滾動歷史，計算基線。"""
        self._atr_history.append(current_atr)
        bp = self._config.vol_atr_baseline_period
        if len(self._atr_history) > bp:
            self._atr_history = self._atr_history[-bp:]
        if len(self._atr_history) >= bp:
            self._atr_baseline = sum(self._atr_history) / Decimal(len(self._atr_history))

    def _check_volatility_regime(self) -> bool:
        """檢查當前波動率是否在可交易範圍內。"""
        if not self._config.use_volatility_filter:
            return True
        if self._atr_baseline is None or self._atr_baseline == 0:
            return True  # 數據不足時不過濾
        ratio = float(self._current_atr / self._atr_baseline)
        return self._config.vol_ratio_low <= ratio <= self._config.vol_ratio_high

    def on_kline(self, kline: Kline, context: BacktestContext) -> list[Signal]:
        """
        Process kline and generate signal.

        Entry: grid touch + RSI score doesn't block direction + no position.
        """
        current_price = kline.close
        closes = context.get_closes(self._config.rsi_period + 10)
        klines_window = context.get_klines_window(self._config.atr_period + 5)

        # A. Reset filled levels when no position
        if not context.has_position:
            self._reset_filled_levels()

        # Update RSI
        self._current_rsi = self._calculate_rsi(closes)
        if self._current_rsi is None:
            return []

        # B. Calculate RSI continuous score
        score = self._rsi_score(self._current_rsi)
        threshold = self._config.rsi_block_threshold

        # Initialize or rebuild grid if needed
        if self._should_rebuild_grid(current_price):
            self._initialize_grid(klines_window, current_price)
            # Update volatility baseline even when rebuilding
            if self._current_atr:
                self._update_volatility_baseline(self._current_atr)
            return []

        # Update volatility baseline
        if self._current_atr:
            self._update_volatility_baseline(self._current_atr)

        # Volatility regime filter — block entry if outside range
        if not self._check_volatility_regime():
            return []

        # Already have position - don't generate new signals
        if context.has_position:
            return []

        # Find current grid level
        level_idx = self._find_current_grid_index(current_price)
        if level_idx is None:
            return []

        # Calculate SL/TP distances
        atr_sl = self._current_atr * self._config.stop_loss_atr_mult if self._current_atr else current_price * Decimal("0.015")
        tp_distance = self._grid_spacing * Decimal(self._config.take_profit_grids)

        # Check for long entry: price dips to touch grid level
        # RSI score < +threshold means NOT strongly bullish → allow long (buying the dip)
        # RSI score > +threshold means TOO bullish → block long (overbought)
        can_long = score < threshold   # block long when RSI too high
        can_short = score > -threshold  # block short when RSI too low

        curr_low = kline.low
        curr_high = kline.high

        # Long entry
        if can_long and level_idx > 0:
            entry_level = self._grid_levels[level_idx]
            if not entry_level.is_long_filled and curr_low <= entry_level.price:
                entry_level.is_long_filled = True
                self._entry_bar_index = context.bar_index

                stop_loss = entry_level.price - atr_sl
                take_profit = entry_level.price + tp_distance

                return [Signal.long_entry(
                    price=entry_level.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"rsi_grid_long_score={score:.2f}",
                )]

        # Short entry
        if can_short and level_idx < len(self._grid_levels) - 1:
            entry_level = self._grid_levels[level_idx + 1]
            if not entry_level.is_short_filled and curr_high >= entry_level.price:
                entry_level.is_short_filled = True
                self._entry_bar_index = context.bar_index

                stop_loss = entry_level.price + atr_sl
                take_profit = entry_level.price - tp_distance

                return [Signal.short_entry(
                    price=entry_level.price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"rsi_grid_short_score={score:.2f}",
                )]

        return []

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check for exit conditions.

        Exit triggers:
        - RSI extreme reversal (LONG: RSI>75, SHORT: RSI<25)
        - Timeout exit (held > max_hold_bars and losing)
        - SL/TP handled by engine
        """
        if self._current_rsi is None:
            return None

        rsi_val = float(self._current_rsi)

        # RSI extreme reversal exit
        if position.side == "LONG":
            if rsi_val > 75:
                return Signal.close_all(reason="rsi_extreme_exit_long")
        else:  # SHORT
            if rsi_val < 25:
                return Signal.close_all(reason="rsi_extreme_exit_short")

        # Timeout exit (only if losing)
        bars_held = context.bar_index - self._entry_bar_index
        if bars_held >= self._config.max_hold_bars:
            # Check if position is losing
            if position.side == "LONG":
                if kline.close < position.entry_price:
                    return Signal.close_all(reason=f"timeout_exit_{bars_held}bars")
            else:
                if kline.close > position.entry_price:
                    return Signal.close_all(reason=f"timeout_exit_{bars_held}bars")

        return None

    def update_trailing_stop(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Decimal]:
        """Update trailing stop if enabled."""
        if not self._config.use_trailing_stop:
            return None

        entry = position.entry_price
        activate_pct = Decimal(str(self._config.trailing_activate_pct))
        trail_pct = Decimal(str(self._config.trailing_distance_pct))

        if position.side == "LONG":
            profit_pct = (kline.high - entry) / entry
            if profit_pct >= activate_pct:
                new_stop = kline.high * (Decimal("1") - trail_pct)
                if position.stop_loss is None or new_stop > position.stop_loss:
                    return new_stop
        else:  # SHORT
            profit_pct = (entry - kline.low) / entry
            if profit_pct >= activate_pct:
                new_stop = kline.low * (Decimal("1") + trail_pct)
                if position.stop_loss is None or new_stop < position.stop_loss:
                    return new_stop

        return None

    def on_position_opened(self, position: Position) -> None:
        """Track entry bar for timeout calculation."""
        pass

    def on_position_closed(self, trade) -> None:
        """No-op; grid reset handled in on_kline."""
        pass
