"""
Bollinger Band Strategy Adapter.

Adapts the Bollinger Band strategy for the unified backtest framework.

Strategy Modes:
1. BB_TREND_GRID (default): BB trend + grid trading
   - Use BB middle (SMA) to determine trend direction
   - In bullish trend (price > SMA): only LONG, buy at grid levels
   - In bearish trend (price < SMA): only SHORT, sell at grid levels
   - Similar to validated Supertrend TREND_GRID strategy

2. BB_NEUTRAL_GRID: Neutral bi-directional grid trading
   - Inspired by Grid Futures strategy (43% annual return)
   - Both LONG and SHORT simultaneously allowed (no trend filter)
   - Uses ATR for dynamic grid range calculation
   - Tighter stop loss (0.5-1%) for quick profit taking
   - Expected to generate 2-3x more trades than BB_TREND_GRID

3. BB_RSI_MOMENTUM: BB + RSI confirmation (legacy, not validated)

VALIDATION STATUS:
- BB_TREND_GRID: PASSED (2024-01-05 ~ 2026-01-24)
- BB_NEUTRAL_GRID: Pending optimization

BB_TREND_GRID Walk-Forward Validated Parameters:
- BB Period: 12, BB Std: 2.0
- Grid Count: 6, Grid Range: 2%
- Take Profit: 2 grids, Stop Loss: 2.5%
- W-F Consistency: 100% (9/9)

Note: This strategy combines BB indicators with grid trading.
BB_TREND_GRID trades only in trend direction, BB_NEUTRAL_GRID trades both directions.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from ...bots.bollinger.indicators import BollingerCalculator
from ...bots.bollinger.models import BollingerConfig as LiveBollingerConfig
from ...core.models import Kline
from ..order import Signal
from ..position import Position
from ..result import Trade
from .base import BacktestContext, BacktestStrategy


class BollingerMode(str, Enum):
    """Bollinger Band strategy modes."""
    BB_TREND_GRID = "bb_trend_grid"      # BB trend + grid trading (validated)
    BB_RSI_MOMENTUM = "bb_rsi_momentum"  # Legacy
    BB_NEUTRAL_GRID = "bb_neutral_grid"  # Neutral bi-directional grid (like Grid Futures)


# 使用實戰配置的默認值作為回測默認值 (單一來源)
_DEFAULT_LIVE_CONFIG = LiveBollingerConfig(symbol="BTCUSDT")


@dataclass
class BollingerStrategyConfig:
    """
    Configuration for Bollinger backtest strategy.

    所有默認值均來自實戰配置 (BollingerConfig)，確保回測與實戰一致。

    Attributes:
        mode: Strategy mode (BB_TREND_GRID, BB_NEUTRAL_GRID, BB_RSI_MOMENTUM)
        bb_period: Bollinger Band period (also used as trend SMA)
        bb_std: Standard deviation multiplier
        grid_count: Number of grid levels
        grid_range_pct: Grid range as percentage of price (used when use_atr_range=False)
        take_profit_grids: Grids for take profit
        stop_loss_pct: Stop loss percentage
        use_hysteresis: Enable hysteresis buffer zone (like live bot)
        hysteresis_pct: Hysteresis percentage (0.2% = 0.002)
        use_signal_cooldown: Enable signal cooldown (like live bot)
        cooldown_bars: Minimum bars between signals
        use_atr_range: Use ATR for dynamic grid range (BB_NEUTRAL_GRID)
        atr_period: ATR calculation period
        atr_multiplier: ATR multiplier for range calculation
        fallback_range_pct: Fallback range when ATR unavailable
    """

    # 所有默認值來自實戰配置 (單一來源，確保一致性)
    mode: BollingerMode = BollingerMode.BB_TREND_GRID
    bb_period: int = _DEFAULT_LIVE_CONFIG.bb_period
    bb_std: Decimal = _DEFAULT_LIVE_CONFIG.bb_std
    grid_count: int = _DEFAULT_LIVE_CONFIG.grid_count
    grid_range_pct: Decimal = _DEFAULT_LIVE_CONFIG.grid_range_pct
    take_profit_grids: int = _DEFAULT_LIVE_CONFIG.take_profit_grids
    stop_loss_pct: Decimal = _DEFAULT_LIVE_CONFIG.stop_loss_pct
    # Protective features
    use_hysteresis: bool = _DEFAULT_LIVE_CONFIG.use_hysteresis
    hysteresis_pct: Decimal = _DEFAULT_LIVE_CONFIG.hysteresis_pct
    use_signal_cooldown: bool = _DEFAULT_LIVE_CONFIG.use_signal_cooldown
    cooldown_bars: int = _DEFAULT_LIVE_CONFIG.cooldown_bars
    # ATR dynamic range (for BB_NEUTRAL_GRID mode)
    use_atr_range: bool = _DEFAULT_LIVE_CONFIG.use_atr_range
    atr_period: int = _DEFAULT_LIVE_CONFIG.atr_period
    atr_multiplier: Decimal = _DEFAULT_LIVE_CONFIG.atr_multiplier
    fallback_range_pct: Decimal = _DEFAULT_LIVE_CONFIG.fallback_range_pct


@dataclass
class GridLevel:
    """Individual grid level."""
    index: int
    price: Decimal
    is_filled: bool = False


class BollingerBacktestStrategy(BacktestStrategy):
    """
    Bollinger Band strategy adapter with multiple modes.

    Modes:
    1. BB_TREND_GRID (default):
       - Uses BB middle band (SMA) to determine trend direction
       - Price > SMA = bullish trend, only LONG positions
       - Price < SMA = bearish trend, only SHORT positions

    2. BB_NEUTRAL_GRID:
       - Bi-directional grid trading (both LONG and SHORT)
       - No trend filter (always allows both directions)
       - Uses ATR for dynamic grid range
       - Inspired by Grid Futures strategy (43% annual)

    Example:
        # Default BB_TREND_GRID mode
        strategy = BollingerBacktestStrategy()
        result = engine.run(klines, strategy)

        # BB_NEUTRAL_GRID mode
        config = BollingerStrategyConfig(
            mode=BollingerMode.BB_NEUTRAL_GRID,
            grid_count=12,
            use_atr_range=True,
            atr_period=21,
            atr_multiplier=Decimal("6.0"),
            stop_loss_pct=Decimal("0.005"),
        )
        strategy = BollingerBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[BollingerStrategyConfig] = None):
        """Initialize strategy."""
        self._config = config or BollingerStrategyConfig()

        # Initialize BB calculator
        self._bb_calculator = BollingerCalculator(
            period=self._config.bb_period,
            std_multiplier=self._config.bb_std,
        )

        # Grid state
        self._grid_levels: List[GridLevel] = []
        self._upper_price: Optional[Decimal] = None
        self._lower_price: Optional[Decimal] = None
        self._grid_spacing: Optional[Decimal] = None
        self._grid_initialized = False

        # Trend state
        self._current_trend: int = 0  # 1=bullish, -1=bearish
        self._current_sma: Optional[Decimal] = None

        # Hysteresis state (like live bot)
        self._last_triggered_level: Optional[int] = None
        self._last_triggered_side: Optional[str] = None

        # Signal cooldown state (like live bot)
        self._signal_cooldown: int = 0

    @property
    def config(self) -> BollingerStrategyConfig:
        """Get strategy configuration."""
        return self._config

    def warmup_period(self) -> int:
        """Return warmup period needed for indicators."""
        base_warmup = self._config.bb_period + 20
        if self._config.use_atr_range:
            return max(base_warmup, self._config.atr_period + 10)
        return base_warmup

    def _calculate_atr(self, klines: List[Kline], period: int) -> Optional[Decimal]:
        """
        Calculate Average True Range.

        Args:
            klines: List of klines
            period: ATR period

        Returns:
            ATR value or None if insufficient data
        """
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

    def _initialize_grid(self, current_price: Decimal, klines: Optional[List[Kline]] = None) -> None:
        """
        Initialize grid levels around current price.

        Args:
            current_price: Current market price
            klines: Optional klines for ATR calculation (required if use_atr_range=True)
        """
        # Calculate range size
        if self._config.use_atr_range and klines:
            atr_value = self._calculate_atr(klines, self._config.atr_period)
            if atr_value and atr_value > 0:
                range_size = atr_value * self._config.atr_multiplier
            else:
                range_size = current_price * self._config.fallback_range_pct
        else:
            range_size = current_price * self._config.grid_range_pct

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

        if price >= self._grid_levels[-1].price:
            return len(self._grid_levels) - 1

        return 0

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding."""
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

    def _reset_filled_levels(self) -> None:
        """
        Reset all filled grid levels.

        Called when no position is held, allowing levels to be reused.
        This prevents the grid from becoming unusable after positions close.
        """
        for level in self._grid_levels:
            level.is_filled = False

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        Dispatches to appropriate mode handler:
        - BB_TREND_GRID: Trend-following grid trading
        - BB_NEUTRAL_GRID: Bi-directional grid trading (like Grid Futures)
        """
        if self._config.mode == BollingerMode.BB_NEUTRAL_GRID:
            return self._on_kline_neutral_grid(kline, context)
        else:
            return self._on_kline_trend_grid(kline, context)

    def _on_kline_trend_grid(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        BB_TREND_GRID Logic:
        - Bullish (price > SMA): Buy dips at grid levels (LONG only)
        - Bearish (price < SMA): Sell rallies at grid levels (SHORT only)
        - Optional hysteresis and cooldown (like live bot)
        """
        klines = context.klines

        # Decrement signal cooldown (like live bot)
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1

        # Calculate Bollinger Bands
        try:
            bands, _ = self._bb_calculator.get_all(klines)
        except Exception:
            return None

        current_price = kline.close
        self._current_sma = bands.middle

        # Determine trend based on price vs SMA
        if current_price > bands.middle:
            self._current_trend = 1  # Bullish
        elif current_price < bands.middle:
            self._current_trend = -1  # Bearish
        else:
            self._current_trend = 0

        # Skip if already in position
        if context.has_position:
            return None

        # Check signal cooldown (like live bot)
        if self._config.use_signal_cooldown and self._signal_cooldown > 0:
            return None

        # Initialize or rebuild grid if needed
        if self._should_rebuild_grid(current_price):
            klines_window = context.get_klines_window(self._config.atr_period + 5) if self._config.use_atr_range else None
            self._initialize_grid(current_price, klines_window)
            return None

        # Need trend to be established
        if self._current_trend == 0:
            return None

        # Find current grid level
        level_idx = self._find_current_grid_level(current_price)
        if level_idx is None:
            return None

        # Bullish trend: LONG only (buy dips)
        if self._current_trend == 1 and level_idx > 0:
            entry_level = self._grid_levels[level_idx]
            # Check if current kline low touched the grid level
            if not entry_level.is_filled and kline.low <= entry_level.price:
                # Check hysteresis before entry (like live bot)
                if not self._check_hysteresis(level_idx, "long", entry_level.price, current_price):
                    return None

                entry_level.is_filled = True

                # Update hysteresis state (like live bot)
                self._last_triggered_level = level_idx
                self._last_triggered_side = "long"

                # Set signal cooldown (like live bot)
                if self._config.use_signal_cooldown:
                    self._signal_cooldown = self._config.cooldown_bars

                # Take profit at next grid level up
                tp_level = min(level_idx + self._config.take_profit_grids, len(self._grid_levels) - 1)
                tp_price = self._grid_levels[tp_level].price
                sl_price = entry_level.price * (Decimal("1") - self._config.stop_loss_pct)

                return Signal.long_entry(
                    price=entry_level.price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="bb_trend_grid_long",
                )

        # Bearish trend: SHORT only (sell rallies)
        elif self._current_trend == -1 and level_idx < len(self._grid_levels) - 1:
            entry_level = self._grid_levels[level_idx + 1]
            # Check if current kline high touched the grid level
            if not entry_level.is_filled and kline.high >= entry_level.price:
                # Check hysteresis before entry (like live bot)
                if not self._check_hysteresis(level_idx + 1, "short", entry_level.price, current_price):
                    return None

                entry_level.is_filled = True

                # Update hysteresis state (like live bot)
                self._last_triggered_level = level_idx + 1
                self._last_triggered_side = "short"

                # Set signal cooldown (like live bot)
                if self._config.use_signal_cooldown:
                    self._signal_cooldown = self._config.cooldown_bars

                # Take profit at next grid level down
                tp_level = max(level_idx + 1 - self._config.take_profit_grids, 0)
                tp_price = self._grid_levels[tp_level].price
                sl_price = entry_level.price * (Decimal("1") + self._config.stop_loss_pct)

                return Signal.short_entry(
                    price=entry_level.price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="bb_trend_grid_short",
                )

        return None

    def _on_kline_neutral_grid(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        BB_NEUTRAL_GRID Logic (inspired by Grid Futures):
        - Bi-directional trading: Both LONG and SHORT allowed simultaneously
        - No trend filter: Always allows both directions
        - Uses ATR for dynamic grid range calculation
        - Expected to generate 2-3x more trades than BB_TREND_GRID

        Entry conditions:
        - LONG: kline.low touches grid level (buy the dip)
        - SHORT: kline.high touches grid level (sell the rally)
        """
        klines = context.klines

        # Decrement signal cooldown (like live bot)
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1

        # Calculate Bollinger Bands (for monitoring, not filtering)
        try:
            bands, _ = self._bb_calculator.get_all(klines)
            self._current_sma = bands.middle
        except Exception:
            pass  # BB not critical for NEUTRAL mode

        current_price = kline.close

        # Skip if already in position
        if context.has_position:
            return None

        # Reset filled levels when no position (allow reuse)
        self._reset_filled_levels()

        # Check signal cooldown (like live bot)
        if self._config.use_signal_cooldown and self._signal_cooldown > 0:
            return None

        # Initialize or rebuild grid if needed
        if self._should_rebuild_grid(current_price):
            klines_window = context.get_klines_window(self._config.atr_period + 5)
            self._initialize_grid(current_price, klines_window)
            return None

        # Find current grid level
        level_idx = self._find_current_grid_level(current_price)
        if level_idx is None:
            return None

        # NEUTRAL mode: Check both LONG and SHORT opportunities

        # LONG entry: price dips to touch lower grid level
        if level_idx > 0:
            entry_level = self._grid_levels[level_idx]
            if not entry_level.is_filled and kline.low <= entry_level.price:
                # Check hysteresis before entry
                if not self._check_hysteresis(level_idx, "long", entry_level.price, current_price):
                    pass  # Don't return, check SHORT too
                else:
                    entry_level.is_filled = True

                    # Update hysteresis state
                    self._last_triggered_level = level_idx
                    self._last_triggered_side = "long"

                    # Set signal cooldown
                    if self._config.use_signal_cooldown:
                        self._signal_cooldown = self._config.cooldown_bars

                    # Take profit at next grid level up
                    tp_level = min(level_idx + self._config.take_profit_grids, len(self._grid_levels) - 1)
                    tp_price = self._grid_levels[tp_level].price
                    sl_price = entry_level.price * (Decimal("1") - self._config.stop_loss_pct)

                    return Signal.long_entry(
                        price=entry_level.price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        reason="bb_neutral_grid_long",
                    )

        # SHORT entry: price rallies to touch upper grid level
        if level_idx < len(self._grid_levels) - 1:
            entry_level = self._grid_levels[level_idx + 1]
            if not entry_level.is_filled and kline.high >= entry_level.price:
                # Check hysteresis before entry
                if not self._check_hysteresis(level_idx + 1, "short", entry_level.price, current_price):
                    return None

                entry_level.is_filled = True

                # Update hysteresis state
                self._last_triggered_level = level_idx + 1
                self._last_triggered_side = "short"

                # Set signal cooldown
                if self._config.use_signal_cooldown:
                    self._signal_cooldown = self._config.cooldown_bars

                # Take profit at next grid level down
                tp_level = max(level_idx + 1 - self._config.take_profit_grids, 0)
                tp_price = self._grid_levels[tp_level].price
                sl_price = entry_level.price * (Decimal("1") + self._config.stop_loss_pct)

                return Signal.short_entry(
                    price=entry_level.price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    reason="bb_neutral_grid_short",
                )

        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        BB_TREND_GRID: Exit if trend flips against position.
        BB_NEUTRAL_GRID: No trend-based exit (rely on TP/SL only).
        """
        # BB_NEUTRAL_GRID: No trend-based exit, rely on TP/SL only
        if self._config.mode == BollingerMode.BB_NEUTRAL_GRID:
            return None

        # BB_TREND_GRID: Exit if trend flipped against position
        if position.side == "LONG" and self._current_trend == -1:
            return Signal.close_all(reason="trend_flip_bearish")

        elif position.side == "SHORT" and self._current_trend == 1:
            return Signal.close_all(reason="trend_flip_bullish")

        return None

    def on_position_opened(self, position: Position) -> None:
        """Track state at entry."""
        pass

    def on_position_closed(self, trade: Trade) -> None:
        """Reset tracking after trade."""
        pass

    def reset(self) -> None:
        """Reset strategy state."""
        self._bb_calculator.reset()
        self._grid_levels = []
        self._upper_price = None
        self._lower_price = None
        self._grid_spacing = None
        self._grid_initialized = False
        self._current_trend = 0
        self._current_sma = None
        # Reset hysteresis state
        self._last_triggered_level = None
        self._last_triggered_side = None
        # Reset cooldown state
        self._signal_cooldown = 0
