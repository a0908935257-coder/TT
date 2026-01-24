"""
Grid Futures Strategy Adapter.

Adapts the Grid Futures trading strategy for the unified backtest framework.

Strategy Logic:
- Uses trend-following direction (SMA-based)
- Enters when price touches grid levels in trend direction
- Dynamic ATR-based grid range
- Leverage support with proper PnL calculation

Walk-Forward Validated Parameters:
- Leverage: 2x
- Grid Count: 10
- Trend Period: 20 (SMA)
- ATR Multiplier: 3.0
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional

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


@dataclass
class GridFuturesStrategyConfig:
    """
    Configuration for Grid Futures backtest strategy.

    Attributes:
        grid_count: Number of grid levels (default 10, validated)
        direction: Trading direction mode (default TREND_FOLLOW)
        leverage: Leverage multiplier for PnL (default 2)
        trend_period: SMA period for trend detection (default 20)
        atr_period: ATR calculation period (default 14)
        atr_multiplier: ATR multiplier for range (default 3.0)
        fallback_range_pct: Range when ATR unavailable (default 8%)
        stop_loss_pct: Stop loss percentage (default 5%)
        take_profit_grids: Number of grids for take profit (default 1)
    """

    grid_count: int = 10
    direction: GridDirection = GridDirection.TREND_FOLLOW
    leverage: int = 2
    trend_period: int = 20
    atr_period: int = 14
    atr_multiplier: Decimal = Decimal("3.0")
    fallback_range_pct: Decimal = Decimal("0.08")
    stop_loss_pct: Decimal = Decimal("0.05")
    take_profit_grids: int = 1

    def __post_init__(self):
        if not isinstance(self.atr_multiplier, Decimal):
            self.atr_multiplier = Decimal(str(self.atr_multiplier))
        if not isinstance(self.fallback_range_pct, Decimal):
            self.fallback_range_pct = Decimal(str(self.fallback_range_pct))
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))


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
    - Trend-following direction (SMA-based)
    - Dynamic ATR-based grid range
    - Leverage support
    - Automatic grid level management

    Example:
        config = GridFuturesStrategyConfig(
            grid_count=10,
            leverage=2,
            direction=GridDirection.TREND_FOLLOW,
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

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding (price out of range)."""
        if not self._grid_initialized:
            return True

        if current_price > self._upper_price or current_price < self._lower_price:
            return True

        return False

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Entry signal or None
        """
        current_price = kline.close
        closes = context.get_closes(self._config.trend_period + 5)
        klines = context.get_klines_window(self._config.atr_period + 5)

        # Update trend
        self._update_trend(closes, current_price)

        # Initialize or rebuild grid if needed
        if self._should_rebuild_grid(current_price):
            self._initialize_grid(klines, current_price)
            return None

        # Already have position - don't generate new signals
        if context.has_position:
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
            if self._current_trend > 0:
                allowed_long = True
            elif self._current_trend < 0:
                allowed_short = True

        # Find current grid level
        level_idx = self._find_current_grid_level(current_price)
        if level_idx is None:
            return None

        # Check for entry signals
        prev_klines = context.get_klines_window(2)
        if len(prev_klines) < 2:
            return None

        prev_close = prev_klines[-2].close

        # Long entry: price dropped to a grid level
        if allowed_long and level_idx > 0:
            entry_level = self._grid_levels[level_idx]
            if not entry_level.is_filled and prev_close > entry_level.price >= current_price:
                entry_level.is_filled = True
                # Take profit at next grid level up
                tp_level = min(level_idx + self._config.take_profit_grids, len(self._grid_levels) - 1)
                tp_price = self._grid_levels[tp_level].price
                sl_price = current_price * (Decimal("1") - self._config.stop_loss_pct)

                return Signal.long_entry(
                    price=current_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="grid_long_entry",
                )

        # Short entry: price rose to a grid level
        if allowed_short and level_idx < len(self._grid_levels) - 1:
            entry_level = self._grid_levels[level_idx + 1]
            if not entry_level.is_filled and prev_close < entry_level.price <= current_price:
                entry_level.is_filled = True
                # Take profit at next grid level down
                tp_level = max(level_idx + 1 - self._config.take_profit_grids, 0)
                tp_price = self._grid_levels[tp_level].price
                sl_price = current_price * (Decimal("1") + self._config.stop_loss_pct)

                return Signal.short_entry(
                    price=current_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="grid_short_entry",
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
