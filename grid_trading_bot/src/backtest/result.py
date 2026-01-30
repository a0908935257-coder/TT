"""
Backtest Result Models.

Provides comprehensive result models for backtest output including
trades, metrics, and walk-forward validation results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional


class ExitReason(str, Enum):
    """Reason for exiting a position."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIMEOUT = "timeout"
    SIGNAL = "signal"
    REVERSAL = "reversal"
    MANUAL = "manual"
    LIQUIDATION = "liquidation"


@dataclass
class Trade:
    """
    Completed trade record.

    Attributes:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        side: Trade direction ('LONG' or 'SHORT')
        entry_price: Entry fill price
        exit_price: Exit fill price
        quantity: Position size
        pnl: Net profit/loss after fees
        pnl_pct: Percentage return
        fees: Total fees paid
        bars_held: Number of bars position was held
        exit_reason: Reason for exit
    """

    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    fees: Decimal
    bars_held: int
    exit_reason: ExitReason

    @property
    def is_win(self) -> bool:
        """Whether this trade was profitable."""
        return self.pnl > 0

    @property
    def holding_time(self) -> timedelta:
        """Duration the position was held."""
        return self.exit_time - self.entry_time


@dataclass
class BacktestResult:
    """
    Comprehensive backtest result containing all metrics.

    Attributes:
        total_profit: Cumulative net profit
        total_profit_pct: Total return percentage
        total_trades: Number of completed trades
        win_rate: Percentage of winning trades
        max_drawdown: Maximum absolute drawdown
        max_drawdown_pct: Maximum percentage drawdown
        sharpe_ratio: Risk-adjusted return (annualized)
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: Return / max drawdown ratio
        profit_factor: Gross profit / gross loss
        avg_win: Average winning trade profit
        avg_loss: Average losing trade loss
        largest_win: Largest single winning trade
        largest_loss: Largest single losing trade
        avg_holding_bars: Average bars held per trade
        long_trades: Number of long trades
        short_trades: Number of short trades
        equity_curve: Time series of equity values
        trades: List of all completed trades
        daily_returns: Daily P&L mapping
    """

    # Core metrics
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_profit_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))

    # Risk metrics
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    sortino_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    calmar_ratio: Decimal = field(default_factory=lambda: Decimal("0"))

    # Trade statistics
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    largest_win: Decimal = field(default_factory=lambda: Decimal("0"))
    largest_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_holding_bars: Decimal = field(default_factory=lambda: Decimal("0"))

    # Breakdown
    long_trades: int = 0
    short_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    gross_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    gross_loss: Decimal = field(default_factory=lambda: Decimal("0"))

    # Futures metrics
    liquidation_count: int = 0
    total_funding_paid: Decimal = field(default_factory=lambda: Decimal("0"))
    max_margin_utilization_pct: Decimal = field(default_factory=lambda: Decimal("0"))

    # Time series
    equity_curve: list[Decimal] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    daily_returns: dict[str, Decimal] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Total Profit: {self.total_profit:.2f} ({self.total_profit_pct:.2f}%)\n"
            f"Total Trades: {self.total_trades} (W: {self.num_wins}, L: {self.num_losses})\n"
            f"Win Rate: {self.win_rate:.2f}%\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2f} ({self.max_drawdown_pct:.2f}%)\n"
            f"Avg Win: {self.avg_win:.2f}, Avg Loss: {self.avg_loss:.2f}"
        )


@dataclass
class WalkForwardPeriod:
    """
    Result from a single walk-forward period.

    Attributes:
        period_num: Period number (1-indexed)
        is_start: Start date of in-sample period
        is_end: End date of in-sample period
        oos_start: Start date of out-of-sample period
        oos_end: End date of out-of-sample period
        is_result: In-sample backtest result
        oos_result: Out-of-sample backtest result
        is_consistent: Whether OOS performance is consistent with IS
    """

    period_num: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_result: BacktestResult
    oos_result: BacktestResult
    is_consistent: bool = False

    @property
    def oos_vs_is_ratio(self) -> Decimal:
        """Ratio of OOS to IS performance."""
        if self.is_result.sharpe_ratio == 0:
            return Decimal("0")
        return self.oos_result.sharpe_ratio / self.is_result.sharpe_ratio


@dataclass
class WalkForwardResult:
    """
    Aggregated walk-forward validation result.

    Attributes:
        periods: List of individual period results
        consistency_pct: Percentage of periods where OOS was consistent with IS
        avg_oos_sharpe: Average out-of-sample Sharpe ratio
        combined_oos_result: Combined result from all OOS periods
    """

    periods: list[WalkForwardPeriod] = field(default_factory=list)
    consistency_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_oos_sharpe: Decimal = field(default_factory=lambda: Decimal("0"))
    combined_oos_result: Optional[BacktestResult] = None

    @property
    def total_periods(self) -> int:
        """Total number of walk-forward periods."""
        return len(self.periods)

    @property
    def consistent_periods(self) -> int:
        """Number of consistent periods."""
        return sum(1 for p in self.periods if p.is_consistent)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Walk-Forward Results ({self.total_periods} periods)\n"
            f"Consistency: {self.consistency_pct:.1f}% ({self.consistent_periods}/{self.total_periods})\n"
            f"Avg OOS Sharpe: {self.avg_oos_sharpe:.2f}\n"
        )
