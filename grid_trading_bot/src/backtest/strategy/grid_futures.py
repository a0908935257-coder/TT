"""
Grid Futures Strategy Adapter.

Adapts the Grid Futures trading strategy for the unified backtest framework.

Strategy Logic:
- Uses neutral direction (both long and short)
- Enters when price touches grid levels
- Dynamic ATR-based grid range
- Leverage support with proper PnL calculation

⚠️ Walk-Forward Validated Parameters (2026-01-27) - 高槓桿高風險:
- Direction: NEUTRAL (雙向交易)
- Leverage: 18x (高槓桿)
- Grid Count: 12
- ATR Period: 21
- ATR Multiplier: 6.0 (寬範圍)
- Stop Loss: 0.5% (緊止損)

Validation Results (BTCUSDT 1h, 2024-01 ~ 2026-01):
- 年化報酬: 43.25%
- 回撤: 3.74%
- 勝率: 56.6%
- 交易次數: 4,949
- Walk-Forward 一致性: 100% (9/9)
- Monte Carlo 穩健性: 100%
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from ...bots.grid_futures.models import GridFuturesConfig as LiveGridFuturesConfig
from ...bots.grid_futures.models import GridDirection as LiveGridDirection
from ...core.models import Kline
from ..order import Signal
from ..position import Position
from .base import BacktestContext, BacktestStrategy


class GridDirection(str, Enum):
    """Grid trading direction mode."""
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    NEUTRAL = "neutral"
    TREND_FOLLOW = "trend_follow"


# 使用實戰配置的默認值作為回測默認值 (單一來源)
_DEFAULT_LIVE_CONFIG = LiveGridFuturesConfig(symbol="BTCUSDT")


@dataclass
class GridFuturesStrategyConfig:
    """
    Configuration for Grid Futures backtest strategy.

    所有默認值均來自實戰配置 (GridFuturesConfig)，確保回測與實戰一致。

    Attributes:
        grid_count: Number of grid levels
        direction: Trading direction mode
        leverage: Leverage multiplier
        trend_period: SMA period for trend detection
        atr_period: ATR calculation period
        atr_multiplier: ATR multiplier for range
        fallback_range_pct: Range when ATR unavailable
        stop_loss_pct: Stop loss percentage
        take_profit_grids: Number of grids for take profit
        use_hysteresis: Enable hysteresis buffer zone
        hysteresis_pct: Hysteresis percentage
        use_signal_cooldown: Enable signal cooldown
        cooldown_bars: Minimum bars between signals
    """

    # 所有默認值來自實戰配置 (單一來源，確保一致性)
    grid_count: int = _DEFAULT_LIVE_CONFIG.grid_count
    direction: GridDirection = GridDirection.NEUTRAL  # 映射到本地 enum
    leverage: int = _DEFAULT_LIVE_CONFIG.leverage
    trend_period: int = _DEFAULT_LIVE_CONFIG.trend_period
    atr_period: int = _DEFAULT_LIVE_CONFIG.atr_period
    atr_multiplier: Decimal = _DEFAULT_LIVE_CONFIG.atr_multiplier
    fallback_range_pct: Decimal = _DEFAULT_LIVE_CONFIG.fallback_range_pct
    stop_loss_pct: Decimal = _DEFAULT_LIVE_CONFIG.stop_loss_pct
    take_profit_grids: int = 1  # 回測專用參數，實戰配置無此欄位
    # Protective features
    use_hysteresis: bool = _DEFAULT_LIVE_CONFIG.use_hysteresis
    hysteresis_pct: Decimal = _DEFAULT_LIVE_CONFIG.hysteresis_pct
    use_signal_cooldown: bool = _DEFAULT_LIVE_CONFIG.use_signal_cooldown
    cooldown_bars: int = _DEFAULT_LIVE_CONFIG.cooldown_bars

    def __post_init__(self):
        if not isinstance(self.atr_multiplier, Decimal):
            self.atr_multiplier = Decimal(str(self.atr_multiplier))
        if not isinstance(self.fallback_range_pct, Decimal):
            self.fallback_range_pct = Decimal(str(self.fallback_range_pct))
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))
        if not isinstance(self.hysteresis_pct, Decimal):
            self.hysteresis_pct = Decimal(str(self.hysteresis_pct))


@dataclass
class GridLevel:
    """Individual grid level."""
    index: int
    price: Decimal
    is_filled: bool = False


class GridFuturesBacktestStrategy(BacktestStrategy):
    """
    Grid Futures Trading Strategy Adapter.

    Implements grid trading on futures with:
    - Neutral direction (both long and short positions)
    - Dynamic ATR-based grid range
    - 10x leverage (validated optimal)
    - Automatic grid level management

    Example:
        # Use validated defaults (recommended)
        strategy = GridFuturesBacktestStrategy()
        result = engine.run(klines, strategy)

        # Or customize
        config = GridFuturesStrategyConfig(
            grid_count=10,
            leverage=10,
            direction=GridDirection.NEUTRAL,
        )
        strategy = GridFuturesBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[GridFuturesStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self._config = config or GridFuturesStrategyConfig()

        # Grid state
        self._grid_levels: List[GridLevel] = []
        self._upper_price: Optional[Decimal] = None
        self._lower_price: Optional[Decimal] = None
        self._center_price: Optional[Decimal] = None
        self._grid_spacing: Optional[Decimal] = None
        self._grid_initialized = False

        # Trend state
        self._current_trend: int = 0  # 1=up, -1=down, 0=neutral

        # Hysteresis state (like live bot)
        self._last_triggered_level: Optional[int] = None
        self._last_triggered_side: Optional[str] = None

        # Signal cooldown state (like live bot)
        self._signal_cooldown: int = 0

    def warmup_period(self) -> int:
        """Return warmup period for indicators."""
        return max(self._config.trend_period, self._config.atr_period) + 10

    def reset(self) -> None:
        """Reset strategy state."""
        self._grid_levels = []
        self._upper_price = None
        self._lower_price = None
        self._center_price = None
        self._grid_spacing = None
        self._grid_initialized = False
        self._current_trend = 0
        # Reset hysteresis state
        self._last_triggered_level = None
        self._last_triggered_side = None
        # Reset cooldown state
        self._signal_cooldown = 0

    def _calculate_sma(self, closes: List[Decimal], period: int) -> Optional[Decimal]:
        """Calculate Simple Moving Average."""
        if len(closes) < period:
            return None
        return sum(closes[-period:]) / Decimal(period)

    def _calculate_atr(self, klines: List[Kline], period: int) -> Optional[Decimal]:
        """Calculate Average True Range."""
        if len(klines) < period + 1:
            return None

        tr_values = []
        for i in range(-period, 0):
            high = Decimal(str(klines[i].high))
            low = Decimal(str(klines[i].low))
            prev_close = Decimal(str(klines[i - 1].close))

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        return sum(tr_values) / Decimal(period)

    def _update_trend(self, closes: List[Decimal], current_price: Decimal) -> None:
        """Update trend direction based on SMA."""
        sma = self._calculate_sma(closes, self._config.trend_period)
        if sma is None:
            self._current_trend = 0
            return

        if current_price > sma:
            self._current_trend = 1  # Uptrend
        elif current_price < sma:
            self._current_trend = -1  # Downtrend
        else:
            self._current_trend = 0  # Neutral

    def _initialize_grid(self, klines: List[Kline], current_price: Decimal) -> None:
        """Initialize grid levels around current price."""
        # Calculate ATR for dynamic range
        atr_value = self._calculate_atr(klines, self._config.atr_period)

        if atr_value and atr_value > 0:
            range_size = atr_value * self._config.atr_multiplier
        else:
            range_size = current_price * self._config.fallback_range_pct

        self._center_price = current_price
        self._upper_price = current_price + range_size
        self._lower_price = current_price - range_size

        # Guard against division by zero
        if self._config.grid_count <= 0:
            self._grid_spacing = range_size  # Fallback to single grid
        else:
            self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)

        # Create grid levels
        self._grid_levels = []
        for i in range(self._config.grid_count + 1):
            price = self._lower_price + (self._grid_spacing * Decimal(i))
            self._grid_levels.append(GridLevel(index=i, price=price))

        self._grid_initialized = True

    def _find_current_grid_level(self, price: Decimal) -> Optional[int]:
        """Find the grid level index for the current price."""
        if not self._grid_levels:
            return None

        for i, level in enumerate(self._grid_levels):
            if i < len(self._grid_levels) - 1:
                if level.price <= price < self._grid_levels[i + 1].price:
                    return i

        # Price above all levels
        if price >= self._grid_levels[-1].price:
            return len(self._grid_levels) - 1

        return 0

    def _reset_filled_levels(self) -> None:
        """
        Reset all filled grid levels.

        Called when no position is held, allowing levels to be reused.
        This prevents the grid from becoming unusable after positions close.
        """
        for level in self._grid_levels:
            level.is_filled = False

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding (price out of range)."""
        if not self._grid_initialized:
            return True

        if current_price > self._upper_price or current_price < self._lower_price:
            return True

        return False

    def _check_hysteresis(self, level_idx: int, side: str, level_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if hysteresis allows entry (like live bot).

        Prevents oscillation by requiring price to move beyond the buffer zone
        before re-triggering the same level.
        """
        if not self._config.use_hysteresis:
            return True

        # Different level or different side - always allow
        if self._last_triggered_level != level_idx or self._last_triggered_side != side:
            return True

        # Same level and side - check if price moved beyond hysteresis buffer
        buffer = level_price * self._config.hysteresis_pct

        if side == "long":
            if current_price > level_price + buffer:
                return True
        else:  # short
            if current_price < level_price - buffer:
                return True

        return False

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        Grid Futures Entry Logic (買跌賣漲):
        - 上升趨勢: 價格下跌穿越 grid level 時做多 (buy the dip)
        - 下降趨勢: 價格上漲穿越 grid level 時做空 (sell the rally)

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Entry signal or None
        """
        # Decrement cooldown (like live bot)
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1

        current_price = kline.close
        closes = context.get_closes(self._config.trend_period + 5)
        klines_window = context.get_klines_window(self._config.atr_period + 5)

        # Update trend
        self._update_trend(closes, current_price)

        # Initialize or rebuild grid if needed
        if self._should_rebuild_grid(current_price):
            self._initialize_grid(klines_window, current_price)
            return None

        # Already have position - don't generate new signals
        if context.has_position:
            return None

        # Reset filled levels when no position is held
        # This allows grid levels to be reused after positions close
        self._reset_filled_levels()

        # Check signal cooldown (like live bot)
        if self._config.use_signal_cooldown and self._signal_cooldown > 0:
            return None

        # Determine allowed direction based on trend
        allowed_long = False
        allowed_short = False

        if self._config.direction == GridDirection.LONG_ONLY:
            allowed_long = True
        elif self._config.direction == GridDirection.SHORT_ONLY:
            allowed_short = True
        elif self._config.direction == GridDirection.NEUTRAL:
            allowed_long = True
            allowed_short = True
        elif self._config.direction == GridDirection.TREND_FOLLOW:
            # 趨勢跟隨: 上升趨勢做多，下降趨勢做空
            if self._current_trend > 0:
                allowed_long = True
            elif self._current_trend < 0:
                allowed_short = True

        # Find current grid level
        level_idx = self._find_current_grid_level(current_price)
        if level_idx is None:
            return None

        # Check for entry signals using kline low/high for better detection
        prev_klines = context.get_klines_window(2)
        if len(prev_klines) < 2:
            return None

        prev_low = prev_klines[-2].low
        prev_high = prev_klines[-2].high
        curr_low = kline.low
        curr_high = kline.high

        # Long entry: 上升趨勢中，價格下跌觸及 grid level (買跌)
        # 條件: kline 的 low 觸及或穿越 grid level
        if allowed_long and level_idx > 0:
            entry_level = self._grid_levels[level_idx]
            # 當前 K 線的 low 觸及 grid level (價格下探)
            if not entry_level.is_filled and curr_low <= entry_level.price:
                # Check hysteresis (like live bot)
                if not self._check_hysteresis(level_idx, "long", entry_level.price, current_price):
                    return None

                entry_level.is_filled = True
                # Take profit at next grid level up
                tp_level = min(level_idx + self._config.take_profit_grids, len(self._grid_levels) - 1)
                tp_price = self._grid_levels[tp_level].price
                sl_price = entry_level.price * (Decimal("1") - self._config.stop_loss_pct)

                # Update hysteresis state (like live bot)
                self._last_triggered_level = level_idx
                self._last_triggered_side = "long"
                # Set cooldown (like live bot)
                if self._config.use_signal_cooldown:
                    self._signal_cooldown = self._config.cooldown_bars

                return Signal.long_entry(
                    price=entry_level.price,  # 使用 grid level 價格
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="grid_long_dip_buy",
                )

        # Short entry: 下降趨勢中，價格上漲觸及 grid level (賣漲)
        # 條件: kline 的 high 觸及或穿越 grid level
        if allowed_short and level_idx < len(self._grid_levels) - 1:
            entry_level = self._grid_levels[level_idx + 1]
            # 當前 K 線的 high 觸及 grid level (價格反彈)
            if not entry_level.is_filled and curr_high >= entry_level.price:
                # Check hysteresis (like live bot)
                if not self._check_hysteresis(level_idx + 1, "short", entry_level.price, current_price):
                    return None

                entry_level.is_filled = True
                # Take profit at next grid level down
                tp_level = max(level_idx + 1 - self._config.take_profit_grids, 0)
                tp_price = self._grid_levels[tp_level].price
                sl_price = entry_level.price * (Decimal("1") + self._config.stop_loss_pct)

                # Update hysteresis state (like live bot)
                self._last_triggered_level = level_idx + 1
                self._last_triggered_side = "short"
                # Set cooldown (like live bot)
                if self._config.use_signal_cooldown:
                    self._signal_cooldown = self._config.cooldown_bars

                return Signal.short_entry(
                    price=entry_level.price,  # 使用 grid level 價格
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="grid_short_rally_sell",
                )

        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check for exit conditions.

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal or None
        """
        # Check for trend change (if trend following)
        if self._config.direction == GridDirection.TREND_FOLLOW:
            if position.side == "LONG" and self._current_trend < 0:
                return Signal.close_all(reason="trend_change_bearish")
            elif position.side == "SHORT" and self._current_trend > 0:
                return Signal.close_all(reason="trend_change_bullish")

        return None
