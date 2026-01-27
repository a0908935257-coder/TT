"""
RSI-Grid Hybrid Strategy Adapter.

Adapts the RSI-Grid hybrid trading strategy for the unified backtest framework.

Strategy Logic:
- Combines RSI mean reversion characteristics with Grid high-frequency entry mechanism
- RSI Zone-based Direction Filter:
  - Oversold (RSI < 30): Only allow LONG
  - Neutral (RSI 30-70): Allow both directions
  - Overbought (RSI > 70): Only allow SHORT
- Trend Filter (SMA):
  - Uptrend (price > SMA): Prefer LONG
  - Downtrend (price < SMA): Prefer SHORT
- Grid Entry: ATR-based grid levels, enter when price touches grid line
- Exit: Take profit at 1 grid distance, stop loss at ATR * 1.5

Design Goals:
- Target Sharpe > 3.0
- Walk-Forward Consistency > 90%
- Win Rate > 70%
- Max Drawdown < 5%
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from ...core.models import Kline
from ..order import Signal
from ..position import Position
from .base import BacktestContext, BacktestStrategy


class RSIZone(str, Enum):
    """RSI zone classification."""
    OVERSOLD = "oversold"        # RSI < 30
    NEUTRAL = "neutral"          # RSI 30-70
    OVERBOUGHT = "overbought"    # RSI > 70


class AllowedDirection(str, Enum):
    """Allowed trading direction."""
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"
    NONE = "none"


@dataclass
class RSIGridStrategyConfig:
    """
    Configuration for RSI-Grid hybrid backtest strategy.

    Attributes:
        # RSI Parameters
        rsi_period: RSI calculation period (default 14)
        oversold_level: RSI oversold threshold (default 30)
        overbought_level: RSI overbought threshold (default 70)

        # Grid Parameters
        grid_count: Number of grid levels (default 10)
        atr_period: ATR calculation period (default 14)
        atr_multiplier: ATR multiplier for grid range (default 3.0)

        # Trend Filter
        trend_sma_period: SMA period for trend detection (default 20)
        use_trend_filter: Enable trend-based direction filter (default True)

        # Risk Management
        position_size_pct: Position size as % of capital (default 10%)
        stop_loss_atr_mult: Stop loss as ATR multiple (default 1.5)
        max_stop_loss_pct: Maximum stop loss percentage (default 3%)
        take_profit_grids: Number of grids for take profit (default 1)
        max_positions: Maximum concurrent positions (default 5)

        # Live-like protective features
        use_hysteresis: Enable hysteresis buffer zone (like live bot)
        hysteresis_pct: Hysteresis percentage (0.2% = 0.002)
        use_signal_cooldown: Enable signal cooldown (like live bot)
        cooldown_bars: Minimum bars between signals
    """

    # RSI Parameters (優化後 2026-01-27)
    rsi_period: int = 14
    oversold_level: int = 33  # 優化後: 33 (原 30)
    overbought_level: int = 66  # 優化後: 66 (原 70)

    # Grid Parameters (優化後)
    grid_count: int = 8  # 優化後: 8 (原 10)
    atr_period: int = 22  # 優化後: 22 (原 14)
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.5"))  # 優化後: 3.5 (原 3.0)

    # Trend Filter (優化後: 關閉)
    trend_sma_period: int = 39  # 優化後: 39 (原 20)
    use_trend_filter: bool = False  # 優化後: 關閉 (原 True)

    # Risk Management (優化後)
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    stop_loss_atr_mult: Decimal = field(default_factory=lambda: Decimal("2.0"))  # 優化後: 2.0 (原 1.5)
    max_stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.03"))
    take_profit_grids: int = 1
    max_positions: int = 5

    # Protective features (優化後: 保持關閉)
    use_hysteresis: bool = False
    hysteresis_pct: Decimal = field(default_factory=lambda: Decimal("0.003"))  # 優化後: 0.3%
    use_signal_cooldown: bool = False
    cooldown_bars: int = 2

    def __post_init__(self):
        """Normalize Decimal types."""
        decimal_fields = [
            'atr_multiplier', 'position_size_pct',
            'stop_loss_atr_mult', 'max_stop_loss_pct', 'hysteresis_pct'
        ]
        for field_name in decimal_fields:
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
    RSI-Grid Hybrid Trading Strategy Adapter.

    Combines RSI mean reversion with Grid entry mechanism:
    - RSI determines allowed direction (oversold=long, overbought=short)
    - SMA trend filter provides additional direction bias
    - Grid levels provide entry points when price touches
    - ATR-based dynamic range adapts to volatility

    Entry Logic:
    1. Calculate RSI zone (oversold/neutral/overbought)
    2. Calculate trend direction (above/below SMA)
    3. Determine allowed direction based on RSI zone + trend
    4. Check if price touches a grid level
    5. Enter in allowed direction

    Exit Logic:
    - Take profit: 1 grid spacing
    - Stop loss: ATR * 1.5 (max 3%)
    - RSI reversal: Exit long when RSI > 70, exit short when RSI < 30

    Example:
        config = RSIGridStrategyConfig(
            rsi_period=14,
            grid_count=10,
            atr_multiplier=Decimal("3.0"),
        )
        strategy = RSIGridBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[RSIGridStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self._config = config or RSIGridStrategyConfig()

        # RSI state
        self._prev_rsi: Optional[Decimal] = None
        self._current_rsi: Optional[Decimal] = None
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None

        # Grid state
        self._grid_levels: List[GridLevel] = []
        self._upper_price: Optional[Decimal] = None
        self._lower_price: Optional[Decimal] = None
        self._center_price: Optional[Decimal] = None
        self._grid_spacing: Optional[Decimal] = None
        self._current_atr: Optional[Decimal] = None
        self._grid_initialized = False

        # Trend state
        self._current_trend: int = 0  # 1=up, -1=down, 0=neutral

        # Position tracking
        self._open_position_count: int = 0

        # Hysteresis state (like live bot)
        self._last_triggered_level: Optional[int] = None
        self._last_triggered_side: Optional[str] = None

        # Signal cooldown state (like live bot)
        self._signal_cooldown: int = 0

    def warmup_period(self) -> int:
        """Return warmup period for indicator initialization."""
        return max(
            self._config.rsi_period + 10,
            self._config.atr_period + 10,
            self._config.trend_sma_period + 10
        )

    def reset(self) -> None:
        """Reset strategy state."""
        self._prev_rsi = None
        self._current_rsi = None
        self._avg_gain = None
        self._avg_loss = None
        self._grid_levels = []
        self._upper_price = None
        self._lower_price = None
        self._center_price = None
        self._grid_spacing = None
        self._current_atr = None
        self._grid_initialized = False
        self._current_trend = 0
        self._open_position_count = 0
        # Reset hysteresis state
        self._last_triggered_level = None
        self._last_triggered_side = None
        # Reset cooldown state
        self._signal_cooldown = 0

    def _calculate_rsi(self, closes: List[Decimal]) -> Optional[Decimal]:
        """Calculate RSI from close prices using Wilder's smoothing."""
        period = self._config.rsi_period
        if len(closes) < period + 1:
            return None

        # Initial average gain/loss
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

            # Process remaining closes with Wilder's smoothing
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
            # Update with new close
            change = closes[-1] - closes[-2]
            if change > 0:
                current_gain = change
                current_loss = Decimal("0")
            else:
                current_gain = Decimal("0")
                current_loss = abs(change)

            self._avg_gain = (self._avg_gain * (period - 1) + current_gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + current_loss) / period

        # Calculate RSI
        if self._avg_loss == 0:
            return Decimal("100") if self._avg_gain > 0 else Decimal("50")

        rs = self._avg_gain / self._avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
        return rsi

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
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        return sum(tr_values) / Decimal(period)

    def _get_rsi_zone(self, rsi: Decimal) -> RSIZone:
        """Determine RSI zone."""
        if rsi < self._config.oversold_level:
            return RSIZone.OVERSOLD
        elif rsi > self._config.overbought_level:
            return RSIZone.OVERBOUGHT
        return RSIZone.NEUTRAL

    def _get_allowed_direction(
        self,
        rsi_zone: RSIZone,
        trend: int
    ) -> AllowedDirection:
        """
        Determine allowed trading direction based on RSI zone and trend.

        RSI Zone Logic:
        - Oversold (< 30): Only LONG
        - Overbought (> 70): Only SHORT
        - Neutral (30-70): Follow trend or BOTH

        Trend Logic (when neutral):
        - Uptrend: Prefer LONG
        - Downtrend: Prefer SHORT
        - Neutral: BOTH
        """
        if rsi_zone == RSIZone.OVERSOLD:
            return AllowedDirection.LONG_ONLY
        elif rsi_zone == RSIZone.OVERBOUGHT:
            return AllowedDirection.SHORT_ONLY
        else:  # NEUTRAL
            if self._config.use_trend_filter:
                if trend > 0:
                    return AllowedDirection.LONG_ONLY
                elif trend < 0:
                    return AllowedDirection.SHORT_ONLY
            return AllowedDirection.BOTH

    def _update_trend(self, closes: List[Decimal], current_price: Decimal) -> None:
        """Update trend direction based on SMA."""
        sma = self._calculate_sma(closes, self._config.trend_sma_period)
        if sma is None:
            self._current_trend = 0
            return

        diff_pct = (current_price - sma) / sma * Decimal("100")

        if diff_pct > Decimal("0.5"):  # > 0.5% above SMA
            self._current_trend = 1  # Uptrend
        elif diff_pct < Decimal("-0.5"):  # < 0.5% below SMA
            self._current_trend = -1  # Downtrend
        else:
            self._current_trend = 0  # Neutral

    def _initialize_grid(self, klines: List[Kline], current_price: Decimal) -> None:
        """Initialize grid levels around current price."""
        # Calculate ATR for dynamic range
        self._current_atr = self._calculate_atr(klines, self._config.atr_period)

        if self._current_atr and self._current_atr > 0:
            range_size = self._current_atr * self._config.atr_multiplier
        else:
            # Fallback to 5% range
            range_size = current_price * Decimal("0.05")

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

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding (price out of range)."""
        if not self._grid_initialized:
            return True

        # Rebuild if price is 10% outside the grid range
        threshold = Decimal("1.1")
        if current_price > self._upper_price * threshold:
            return True
        if current_price < self._lower_price / threshold:
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

        # Price above all levels
        if price >= self._grid_levels[-1].price:
            return len(self._grid_levels) - 1

        return 0

    def _check_hysteresis(self, level_idx: int, side: str, level_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if hysteresis allows entry (like live bot).

        Prevents oscillation by requiring price to move beyond the buffer zone
        before re-triggering the same level.

        Args:
            level_idx: Grid level index
            side: "long" or "short"
            level_price: The grid level price
            current_price: Current market price

        Returns:
            True if entry is allowed, False if blocked by hysteresis
        """
        if not self._config.use_hysteresis:
            return True

        # Different level or different side - always allow
        if self._last_triggered_level != level_idx or self._last_triggered_side != side:
            return True

        # Same level and side - check if price moved beyond hysteresis buffer
        buffer = level_price * self._config.hysteresis_pct

        if side == "long":
            # For long, price must have moved UP beyond buffer before coming back down
            if current_price > level_price + buffer:
                return True
        else:  # short
            # For short, price must have moved DOWN beyond buffer before coming back up
            if current_price < level_price - buffer:
                return True

        return False

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        RSI-Grid Entry Logic:
        1. Calculate RSI and determine zone
        2. Calculate trend direction
        3. Determine allowed direction
        4. Check if price touches grid level
        5. Enter position in allowed direction

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Entry signal or None
        """
        current_price = kline.close
        closes = context.get_closes(max(self._config.rsi_period + 10, self._config.trend_sma_period + 5))
        klines_window = context.get_klines_window(self._config.atr_period + 5)

        # Decrement signal cooldown (like live bot)
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1

        # Update RSI
        self._prev_rsi = self._current_rsi
        self._current_rsi = self._calculate_rsi(closes)

        if self._current_rsi is None:
            return None

        # Update trend
        self._update_trend(closes, current_price)

        # Initialize or rebuild grid if needed
        if self._should_rebuild_grid(current_price):
            self._initialize_grid(klines_window, current_price)
            return None

        # Already have position - don't generate new signals
        if context.has_position:
            return None

        # Check signal cooldown (like live bot)
        if self._config.use_signal_cooldown and self._signal_cooldown > 0:
            return None

        # Check max positions
        if self._open_position_count >= self._config.max_positions:
            return None

        # Determine RSI zone and allowed direction
        rsi_zone = self._get_rsi_zone(self._current_rsi)
        allowed_dir = self._get_allowed_direction(rsi_zone, self._current_trend)

        if allowed_dir == AllowedDirection.NONE:
            return None

        # Find current grid level
        level_idx = self._find_current_grid_index(current_price)
        if level_idx is None:
            return None

        # Check for entry signals using kline low/high for better detection
        prev_klines = context.get_klines_window(2)
        if len(prev_klines) < 2:
            return None

        curr_low = kline.low
        curr_high = kline.high

        # Calculate stop loss and take profit
        atr_sl = self._current_atr * self._config.stop_loss_atr_mult if self._current_atr else current_price * self._config.max_stop_loss_pct
        max_sl = current_price * self._config.max_stop_loss_pct
        sl_distance = min(atr_sl, max_sl)
        tp_distance = self._grid_spacing * Decimal(self._config.take_profit_grids)

        # Long entry: price dips to touch grid level
        if allowed_dir in [AllowedDirection.LONG_ONLY, AllowedDirection.BOTH]:
            if level_idx > 0:
                entry_level = self._grid_levels[level_idx]
                # Current candle's low touches grid level (price dip)
                if not entry_level.is_long_filled and curr_low <= entry_level.price:
                    # Check hysteresis before entry (like live bot)
                    if not self._check_hysteresis(level_idx, "long", entry_level.price, current_price):
                        pass  # Blocked by hysteresis, continue to check short
                    else:
                        entry_level.is_long_filled = True
                        self._open_position_count += 1

                        # Update hysteresis state (like live bot)
                        self._last_triggered_level = level_idx
                        self._last_triggered_side = "long"

                        # Set signal cooldown (like live bot)
                        if self._config.use_signal_cooldown:
                            self._signal_cooldown = self._config.cooldown_bars

                        stop_loss = entry_level.price - sl_distance
                        take_profit = entry_level.price + tp_distance

                        return Signal.long_entry(
                            price=entry_level.price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"rsi_grid_long_zone={rsi_zone.value}_trend={self._current_trend}",
                        )

        # Short entry: price rises to touch grid level
        if allowed_dir in [AllowedDirection.SHORT_ONLY, AllowedDirection.BOTH]:
            if level_idx < len(self._grid_levels) - 1:
                entry_level = self._grid_levels[level_idx + 1]
                # Current candle's high touches grid level (price rally)
                if not entry_level.is_short_filled and curr_high >= entry_level.price:
                    # Check hysteresis before entry (like live bot)
                    if not self._check_hysteresis(level_idx + 1, "short", entry_level.price, current_price):
                        pass  # Blocked by hysteresis
                    else:
                        entry_level.is_short_filled = True
                        self._open_position_count += 1

                        # Update hysteresis state (like live bot)
                        self._last_triggered_level = level_idx + 1
                        self._last_triggered_side = "short"

                        # Set signal cooldown (like live bot)
                        if self._config.use_signal_cooldown:
                            self._signal_cooldown = self._config.cooldown_bars

                        stop_loss = entry_level.price + sl_distance
                        take_profit = entry_level.price - tp_distance

                        return Signal.short_entry(
                            price=entry_level.price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            reason=f"rsi_grid_short_zone={rsi_zone.value}_trend={self._current_trend}",
                        )

        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check for exit conditions.

        Exit triggers:
        - RSI reversal: Long exits when RSI > 70, Short exits when RSI < 30
        - Stop loss and take profit are handled by the engine

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal or None
        """
        if self._current_rsi is None:
            return None

        # RSI-based exit
        if position.side == "LONG":
            # Exit long when RSI enters overbought
            if float(self._current_rsi) > self._config.overbought_level:
                self._open_position_count = max(0, self._open_position_count - 1)
                return Signal.close_all(reason="rsi_overbought_exit")
        else:  # SHORT
            # Exit short when RSI enters oversold
            if float(self._current_rsi) < self._config.oversold_level:
                self._open_position_count = max(0, self._open_position_count - 1)
                return Signal.close_all(reason="rsi_oversold_exit")

        return None

    def on_position_closed(self, trade) -> None:
        """Callback when position is closed."""
        self._open_position_count = max(0, self._open_position_count - 1)
