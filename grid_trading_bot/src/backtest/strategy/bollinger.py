"""
Bollinger Band Strategy Adapter.

Adapts the Bollinger Band strategy for the unified backtest framework.

Strategy Modes:
1. BB_TREND_GRID (default): BB trend + grid trading
   - Use BB middle (SMA) to determine trend direction
   - In bullish trend (price > SMA): only LONG, buy at grid levels
   - In bearish trend (price < SMA): only SHORT, sell at grid levels
   - Similar to validated Supertrend TREND_GRID strategy

2. BB_RSI_MOMENTUM: BB + RSI confirmation (legacy, not validated)

VALIDATION STATUS: PASSED (2024-01-05 ~ 2026-01-24)

Walk-Forward Validated Parameters:
- BB Period: 20, BB Std: 2.0
- Grid Count: 10, Grid Range: 4%
- Take Profit: 1 grid, Stop Loss: 5%

Validation Results (BTCUSDT 1h):
- Total Return: +333.63%
- Win Rate: 76.24%
- Sharpe Ratio: 6.35
- Walk-Forward Consistency: 66.7% (4/6 periods consistent)
- Trades: 1145

Validation Results (ETHUSDT 1h):
- Total Return: +584.82%
- Win Rate: 81.62%
- Sharpe Ratio: 7.57
- Walk-Forward Consistency: 83.3% (5/6 periods consistent)
- Trades: 1790

Note: This strategy combines BB trend direction with high win rate grid trading.
Only trades in the direction of the current trend (price vs SMA).
"""

from dataclasses import dataclass, field
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
    BB_TREND_GRID = "bb_trend_grid"    # New: BB trend + grid trading
    BB_RSI_MOMENTUM = "bb_rsi_momentum"  # Legacy


@dataclass
class BollingerStrategyConfig:
    """
    Configuration for Bollinger backtest strategy.

    Attributes:
        mode: Strategy mode
        bb_period: Bollinger Band period (also used as trend SMA)
        bb_std: Standard deviation multiplier
        grid_count: Number of grid levels
        grid_range_pct: Grid range as percentage of price
        take_profit_grids: Grids for take profit
        stop_loss_pct: Stop loss percentage
        use_hysteresis: Enable hysteresis buffer zone (like live bot)
        hysteresis_pct: Hysteresis percentage (0.2% = 0.002)
        use_signal_cooldown: Enable signal cooldown (like live bot)
        cooldown_bars: Minimum bars between signals
    """

    mode: BollingerMode = BollingerMode.BB_TREND_GRID
    bb_period: int = 20
    bb_std: Decimal = Decimal("2.0")
    grid_count: int = 10
    grid_range_pct: Decimal = Decimal("0.04")  # 4% range
    take_profit_grids: int = 1
    stop_loss_pct: Decimal = Decimal("0.05")
    # Protective features (like live bot)
    use_hysteresis: bool = False
    hysteresis_pct: Decimal = Decimal("0.002")  # 0.2%
    use_signal_cooldown: bool = False
    cooldown_bars: int = 2


@dataclass
class GridLevel:
    """Individual grid level."""
    index: int
    price: Decimal
    is_filled: bool = False


class BollingerBacktestStrategy(BacktestStrategy):
    """
    Bollinger Band strategy adapter with BB_TREND_GRID mode.

    BB_TREND_GRID Strategy:
    - Uses BB middle band (SMA) to determine trend direction
    - Price > SMA = bullish trend, only LONG positions
    - Price < SMA = bearish trend, only SHORT positions
    - Uses grid trading within the trend direction
    - Similar to validated Supertrend TREND_GRID strategy

    Example:
        strategy = BollingerBacktestStrategy()
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
        return self._config.bb_period + 20

    def _initialize_grid(self, current_price: Decimal) -> None:
        """Initialize grid levels around current price."""
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

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

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
            self._initialize_grid(current_price)
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

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        Exit if trend flips against position.
        """
        # Exit if trend flipped against position
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
