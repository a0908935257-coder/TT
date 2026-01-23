"""
Grid Trading Strategy Adapter.

Adapts the Grid trading strategy for the unified backtest framework.

Grid Strategy Logic:
- Creates a grid of price levels between upper and lower bounds
- Buys when price drops to a grid level
- Sells when price rises to the next grid level above
- Profits from price oscillation within the range

Note: This is a simplified single-position grid for the unified framework.
For full multi-level grid simulation, use the dedicated GridBacktest class
in tests/simulation/test_backtest.py.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from ...core.models import Kline
from ..order import Signal
from ..position import Position
from ..result import Trade
from .base import BacktestContext, BacktestStrategy


@dataclass
class GridStrategyConfig:
    """
    Configuration for Grid backtest strategy.

    Attributes:
        upper_price: Upper bound of grid range
        lower_price: Lower bound of grid range
        grid_count: Number of grid levels
        use_geometric: Use geometric (percentage) spacing vs arithmetic
        take_profit_grids: Number of grid levels for take profit (default 1)
        stop_loss_pct: Stop loss percentage below lower bound
    """

    upper_price: Decimal = field(default_factory=lambda: Decimal("55000"))
    lower_price: Decimal = field(default_factory=lambda: Decimal("45000"))
    grid_count: int = 10
    use_geometric: bool = True
    take_profit_grids: int = 1
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))


class GridBacktestStrategy(BacktestStrategy):
    """
    Simplified Grid trading strategy adapter.

    This strategy implements a simplified version of grid trading:
    - Enters LONG when price touches a grid level from above
    - Exits when price rises to the next grid level (or multiple levels)
    - Uses stop loss below the entry grid level

    For full multi-position grid backtesting, use GridBacktest class.

    Example:
        config = GridStrategyConfig(
            upper_price=Decimal("55000"),
            lower_price=Decimal("45000"),
            grid_count=10,
        )
        strategy = GridBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[GridStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self._config = config or GridStrategyConfig()

        # Calculate grid levels
        self._grid_levels: list[Decimal] = []
        self._calculate_grid_levels()

        # State
        self._entry_level_idx: Optional[int] = None
        self._prev_price: Optional[Decimal] = None

    @property
    def config(self) -> GridStrategyConfig:
        """Get strategy configuration."""
        return self._config

    @property
    def grid_levels(self) -> list[Decimal]:
        """Get calculated grid levels."""
        return self._grid_levels.copy()

    def _calculate_grid_levels(self) -> None:
        """Calculate grid price levels."""
        self._grid_levels = []
        upper = self._config.upper_price
        lower = self._config.lower_price
        count = self._config.grid_count

        if self._config.use_geometric:
            # Geometric spacing (equal percentage between levels)
            ratio = (upper / lower) ** (Decimal("1") / Decimal(count))
            for i in range(count + 1):
                level = lower * (ratio ** Decimal(i))
                self._grid_levels.append(level)
        else:
            # Arithmetic spacing (equal price between levels)
            step = (upper - lower) / Decimal(count)
            for i in range(count + 1):
                level = lower + step * Decimal(i)
                self._grid_levels.append(level)

    def warmup_period(self) -> int:
        """Return warmup period (minimal for grid strategy)."""
        return 10

    def _find_grid_level(self, price: Decimal) -> Optional[int]:
        """
        Find the grid level index at or below the given price.

        Returns:
            Grid level index, or None if below lowest level
        """
        for i in range(len(self._grid_levels) - 1, -1, -1):
            if price >= self._grid_levels[i]:
                return i
        return None

    def _price_crossed_level_down(
        self, prev_price: Decimal, current_price: Decimal, level: Decimal
    ) -> bool:
        """Check if price crossed a level downward."""
        return prev_price > level >= current_price

    def _price_crossed_level_up(
        self, prev_price: Decimal, current_price: Decimal, level: Decimal
    ) -> bool:
        """Check if price crossed a level upward."""
        return prev_price < level <= current_price

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        Entry Logic:
        - Enter LONG when price drops to a grid level from above
        - Only enter if within grid range

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Signal if entry conditions met, None otherwise
        """
        current_price = kline.close

        # Initialize prev_price
        if self._prev_price is None:
            self._prev_price = current_price
            return None

        # Skip if already in position
        if context.has_position:
            self._prev_price = current_price
            return None

        # Check if price is within grid range
        if current_price < self._config.lower_price or current_price > self._config.upper_price:
            self._prev_price = current_price
            return None

        # Check for price crossing grid levels downward (buy signal)
        for i, level in enumerate(self._grid_levels):
            if i == len(self._grid_levels) - 1:
                # Skip top level (no room to sell above)
                continue

            if self._price_crossed_level_down(self._prev_price, current_price, level):
                # Price dropped to this grid level - BUY
                self._entry_level_idx = i

                # Stop loss below entry level
                stop_loss = level * (Decimal("1") - self._config.stop_loss_pct)

                # Take profit at higher grid level(s)
                tp_idx = min(i + self._config.take_profit_grids, len(self._grid_levels) - 1)
                take_profit = self._grid_levels[tp_idx]

                self._prev_price = current_price
                return Signal.long_entry(
                    price=level,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"grid_level_{i}_buy",
                )

        self._prev_price = current_price
        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        Exits when price rises to take profit grid level.
        (Stop loss is handled by the engine via position.stop_loss)

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal if take profit reached, None otherwise
        """
        # Take profit is handled by the engine through position.take_profit
        # This method can be used for additional exit logic

        # Exit if price breaks above upper bound (take remaining profit)
        if kline.close > self._config.upper_price:
            return Signal.close_all(reason="grid_upper_breakout")

        return None

    def on_position_opened(self, position: Position) -> None:
        """Track entry level."""
        pass

    def on_position_closed(self, trade: Trade) -> None:
        """Reset entry level tracking after trade."""
        self._entry_level_idx = None

    def reset(self) -> None:
        """Reset strategy state."""
        self._entry_level_idx = None
        self._prev_price = None
        self._calculate_grid_levels()


class MultiLevelGridStrategy(BacktestStrategy):
    """
    Full multi-level grid strategy.

    This implements the complete grid trading logic with multiple
    concurrent positions at different grid levels.

    Note: Requires BacktestConfig with max_positions > 1.
    """

    def __init__(self, config: Optional[GridStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self._config = config or GridStrategyConfig()

        # Calculate grid levels
        self._grid_levels: list[Decimal] = []
        self._level_states: dict[int, str] = {}  # level_idx -> 'empty', 'filled'
        self._calculate_grid_levels()

        # State
        self._prev_price: Optional[Decimal] = None
        self._positions_by_level: dict[int, Position] = {}

    def _calculate_grid_levels(self) -> None:
        """Calculate grid price levels."""
        self._grid_levels = []
        self._level_states = {}
        upper = self._config.upper_price
        lower = self._config.lower_price
        count = self._config.grid_count

        if self._config.use_geometric:
            ratio = (upper / lower) ** (Decimal("1") / Decimal(count))
            for i in range(count + 1):
                level = lower * (ratio ** Decimal(i))
                self._grid_levels.append(level)
                self._level_states[i] = "empty"
        else:
            step = (upper - lower) / Decimal(count)
            for i in range(count + 1):
                level = lower + step * Decimal(i)
                self._grid_levels.append(level)
                self._level_states[i] = "empty"

    def warmup_period(self) -> int:
        """Return warmup period."""
        return 10

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline for multi-level grid.

        This is a simplified version - for full multi-position support,
        the engine would need to be modified to handle multiple signals
        per kline.
        """
        current_price = kline.close

        if self._prev_price is None:
            self._prev_price = current_price
            return None

        # Check if within range
        if current_price < self._config.lower_price or current_price > self._config.upper_price:
            self._prev_price = current_price
            return None

        # Skip if already at max positions
        if not context.has_position:
            # Look for buy signals at grid levels
            for i, level in enumerate(self._grid_levels[:-1]):
                if (
                    self._prev_price > level >= current_price
                    and self._level_states[i] == "empty"
                ):
                    # Buy at this level
                    self._level_states[i] = "filled"
                    stop_loss = level * (Decimal("1") - self._config.stop_loss_pct)
                    tp_idx = min(i + 1, len(self._grid_levels) - 1)
                    take_profit = self._grid_levels[tp_idx]

                    self._prev_price = current_price
                    return Signal.long_entry(
                        price=level,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reason=f"multi_grid_level_{i}",
                    )

        self._prev_price = current_price
        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """Check exit for multi-level grid."""
        return None

    def on_position_closed(self, trade: Trade) -> None:
        """Reset level state after trade closes."""
        # Reset the level that was filled
        for i, level in enumerate(self._grid_levels):
            if abs(trade.entry_price - level) < level * Decimal("0.001"):
                self._level_states[i] = "empty"
                break

    def reset(self) -> None:
        """Reset strategy state."""
        self._prev_price = None
        self._positions_by_level.clear()
        self._calculate_grid_levels()
