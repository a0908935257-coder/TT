"""
Base Strategy Module.

Defines the abstract interface for backtest strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from ...core.models import Kline
from ..order import Signal
from ..position import Position
from ..result import Trade


@dataclass
class BacktestContext:
    """
    Context provided to strategy on each kline.

    Contains current state and historical data needed for
    decision making.

    Attributes:
        bar_index: Current bar index (0-based)
        klines: All klines up to and including current
        current_position: Current open position (if any)
        realized_pnl: Cumulative realized P&L
        equity: Current account equity
        initial_capital: Starting capital
    """

    bar_index: int
    klines: list[Kline]
    current_position: Optional[Position] = None
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    equity: Decimal = field(default_factory=lambda: Decimal("10000"))
    initial_capital: Decimal = field(default_factory=lambda: Decimal("10000"))

    @property
    def current_kline(self) -> Kline:
        """Get the current kline."""
        return self.klines[self.bar_index]

    @property
    def current_price(self) -> Decimal:
        """Get the current close price."""
        return self.current_kline.close

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.current_position is not None

    @property
    def position_side(self) -> Optional[str]:
        """Get the current position side, if any."""
        if self.current_position:
            return self.current_position.side
        return None

    def get_klines_window(self, lookback: int) -> list[Kline]:
        """
        Get a window of recent klines.

        Args:
            lookback: Number of bars to look back

        Returns:
            List of klines (up to lookback bars)
        """
        start = max(0, self.bar_index - lookback + 1)
        return self.klines[start : self.bar_index + 1]

    def get_closes(self, lookback: int) -> list[Decimal]:
        """
        Get recent close prices.

        Args:
            lookback: Number of bars

        Returns:
            List of close prices
        """
        return [k.close for k in self.get_klines_window(lookback)]

    def get_highs(self, lookback: int) -> list[Decimal]:
        """
        Get recent high prices.

        Args:
            lookback: Number of bars

        Returns:
            List of high prices
        """
        return [k.high for k in self.get_klines_window(lookback)]

    def get_lows(self, lookback: int) -> list[Decimal]:
        """
        Get recent low prices.

        Args:
            lookback: Number of bars

        Returns:
            List of low prices
        """
        return [k.low for k in self.get_klines_window(lookback)]


class BacktestStrategy(ABC):
    """
    Abstract base class for backtest strategies.

    Implement this interface to create a new strategy that can
    be backtested using the unified framework.

    Example:
        class MyStrategy(BacktestStrategy):
            def warmup_period(self) -> int:
                return 20

            def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
                # Strategy logic here
                if some_condition:
                    return Signal.long_entry(stop_loss=..., take_profit=...)
                return None
    """

    @abstractmethod
    def warmup_period(self) -> int:
        """
        Return the number of bars needed for indicator warmup.

        The engine will skip the first `warmup_period()` bars
        before calling `on_kline()`.

        Returns:
            Number of warmup bars required
        """
        pass

    @abstractmethod
    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process a kline and optionally return a trading signal.

        This is the main strategy logic method. It's called once
        per bar after the warmup period.

        Args:
            kline: The current kline
            context: Backtest context with state and history

        Returns:
            Signal if a trade should be made, None otherwise
        """
        pass

    def on_position_opened(self, position: Position) -> None:
        """
        Callback when a position is opened.

        Override to perform custom logic when entering a trade.

        Args:
            position: The newly opened position
        """
        pass

    def on_position_closed(self, trade: Trade) -> None:
        """
        Callback when a position is closed.

        Override to perform custom logic when exiting a trade.

        Args:
            trade: The completed trade record
        """
        pass

    def on_bar_close(self, kline: Kline, context: BacktestContext) -> None:
        """
        Callback at the end of each bar.

        Override for custom end-of-bar logic like tracking
        or updating internal state.

        Args:
            kline: The closing kline
            context: Current backtest context
        """
        pass

    def reset(self) -> None:
        """
        Reset strategy state.

        Override if the strategy maintains internal state
        that needs to be cleared between backtests.
        """
        pass

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if an open position should be exited.

        Default implementation checks stop loss and take profit
        from the position. Override for custom exit logic.

        Args:
            position: Current open position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal if position should close, None otherwise
        """
        return None

    def update_trailing_stop(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Decimal]:
        """
        Update trailing stop price for a position.

        Override to implement trailing stop logic.

        Args:
            position: Current open position
            kline: Current kline
            context: Backtest context

        Returns:
            New stop loss price, or None to keep existing
        """
        return None
