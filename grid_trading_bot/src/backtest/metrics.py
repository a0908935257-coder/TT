"""
Metrics Calculation Module.

Provides comprehensive performance metrics calculation for backtests.
"""

import math
from decimal import Decimal
from typing import Optional

from .result import BacktestResult, Trade


class MetricsCalculator:
    """
    Calculates performance metrics from trade history and equity curve.

    Computes standard trading metrics including Sharpe ratio, max drawdown,
    profit factor, and more.
    """

    # Constants
    TRADING_DAYS_PER_YEAR = Decimal("365")  # Crypto trades 24/7
    RISK_FREE_RATE = Decimal("0.02")  # 2% annual risk-free rate

    def __init__(self, initial_capital: Decimal = Decimal("10000")) -> None:
        """
        Initialize metrics calculator.

        Args:
            initial_capital: Starting capital for percentage calculations
        """
        self._initial_capital = initial_capital

    def calculate_all(
        self,
        trades: list[Trade],
        equity_curve: list[Decimal],
        daily_returns: dict[str, Decimal],
        total_funding_paid: Decimal = Decimal("0"),
    ) -> BacktestResult:
        """
        Calculate all metrics and return BacktestResult.

        Args:
            trades: List of completed trades
            equity_curve: Time series of equity values
            daily_returns: Daily P&L mapping

        Returns:
            Comprehensive BacktestResult
        """
        result = BacktestResult()
        result.trades = trades
        result.equity_curve = equity_curve
        result.daily_returns = daily_returns

        # Always calculate drawdown from equity curve even without trades
        if equity_curve:
            result.max_drawdown, result.max_drawdown_pct = self._calculate_max_drawdown(
                equity_curve
            )

        if not trades:
            return result

        # Basic statistics
        result.total_trades = len(trades)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)

        # Profit metrics (deduct funding fees)
        result.total_profit = sum(t.pnl for t in trades) - total_funding_paid
        result.total_profit_pct = (
            result.total_profit / self._initial_capital * Decimal("100")
        )

        # Win rate
        result.win_rate = (
            Decimal(len(wins)) / Decimal(len(trades)) * Decimal("100")
            if trades
            else Decimal("0")
        )

        # Gross profit/loss
        result.gross_profit = sum(t.pnl for t in wins) if wins else Decimal("0")
        result.gross_loss = abs(sum(t.pnl for t in losses)) if losses else Decimal("0")

        # Profit factor
        result.profit_factor = self._calculate_profit_factor(
            result.gross_profit, result.gross_loss
        )

        # Average win/loss
        result.avg_win = result.gross_profit / Decimal(len(wins)) if wins else Decimal("0")
        result.avg_loss = result.gross_loss / Decimal(len(losses)) if losses else Decimal("0")

        # Largest win/loss
        result.largest_win = max(t.pnl for t in wins) if wins else Decimal("0")
        result.largest_loss = abs(min(t.pnl for t in losses)) if losses else Decimal("0")

        # Average holding
        result.avg_holding_bars = (
            Decimal(sum(t.bars_held for t in trades)) / Decimal(len(trades))
            if trades
            else Decimal("0")
        )

        # Direction breakdown
        result.long_trades = len([t for t in trades if t.side == "LONG"])
        result.short_trades = len([t for t in trades if t.side == "SHORT"])

        # Risk metrics
        result.max_drawdown, result.max_drawdown_pct = self._calculate_max_drawdown(
            equity_curve
        )
        result.sharpe_ratio = self._calculate_sharpe_ratio(daily_returns, equity_curve)
        result.sortino_ratio = self._calculate_sortino_ratio(daily_returns, equity_curve)
        result.calmar_ratio = self._calculate_calmar_ratio(
            result.total_profit_pct, result.max_drawdown_pct
        )

        return result

    def _calculate_profit_factor(
        self, gross_profit: Decimal, gross_loss: Decimal
    ) -> Decimal:
        """Calculate profit factor (gross profit / gross loss)."""
        if gross_loss == 0:
            return Decimal("999") if gross_profit > 0 else Decimal("0")
        return gross_profit / gross_loss

    def _calculate_max_drawdown(
        self, equity_curve: list[Decimal]
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate maximum drawdown.

        Returns:
            Tuple of (absolute_drawdown, percentage_drawdown)
        """
        if not equity_curve:
            return Decimal("0"), Decimal("0")

        peak = equity_curve[0]
        max_dd = Decimal("0")
        max_dd_pct = Decimal("0")

        for equity in equity_curve:
            if equity > peak:
                peak = equity

            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
                if peak > 0:
                    max_dd_pct = drawdown / peak * Decimal("100")

        return max_dd, max_dd_pct

    def _calculate_sharpe_ratio(
        self,
        daily_returns: dict[str, Decimal],
        equity_curve: list[Decimal] | None = None,
        annualize: bool = True,
    ) -> Decimal:
        """
        Calculate Sharpe ratio.

        Args:
            daily_returns: Daily P&L values
            equity_curve: Equity curve for running-equity-based returns
            annualize: Whether to annualize the ratio

        Returns:
            Sharpe ratio
        """
        if not daily_returns or len(daily_returns) < 2:
            return Decimal("0")

        returns = list(daily_returns.values())

        # Convert to percentage returns using running equity as denominator
        returns_pct = self._compute_equity_based_returns(returns, equity_curve)

        # Mean return
        mean_return = sum(returns_pct) / Decimal(len(returns_pct))

        # Standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns_pct) / Decimal(
            len(returns_pct) - 1
        )

        if variance <= 0:
            return Decimal("0")

        std_dev = Decimal(str(math.sqrt(float(variance))))

        if std_dev == 0:
            return Decimal("0")

        # Daily risk-free rate
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        # Sharpe
        sharpe = (mean_return - daily_rf) / std_dev

        if annualize:
            sharpe *= Decimal(str(math.sqrt(float(self.TRADING_DAYS_PER_YEAR))))

        return sharpe

    def _calculate_sortino_ratio(
        self,
        daily_returns: dict[str, Decimal],
        equity_curve: list[Decimal] | None = None,
        annualize: bool = True,
    ) -> Decimal:
        """
        Calculate Sortino ratio (downside deviation only).

        Args:
            daily_returns: Daily P&L values
            equity_curve: Equity curve for running-equity-based returns
            annualize: Whether to annualize the ratio

        Returns:
            Sortino ratio
        """
        if not daily_returns or len(daily_returns) < 2:
            return Decimal("0")

        returns = list(daily_returns.values())
        returns_pct = self._compute_equity_based_returns(returns, equity_curve)

        # Mean return
        mean_return = sum(returns_pct) / Decimal(len(returns_pct))

        # Downside deviation (only negative returns)
        # Standard Sortino uses total observation count for downside deviation
        negative_returns = [r for r in returns_pct if r < 0]

        if not negative_returns:
            return Decimal("999") if mean_return > 0 else Decimal("0")

        # Use total observation count for proper Sortino calculation
        # This prevents overestimation of downside risk when few negative days exist
        downside_variance = sum(r**2 for r in negative_returns) / Decimal(
            len(returns_pct)
        )

        if downside_variance <= 0:
            return Decimal("0")

        downside_std = Decimal(str(math.sqrt(float(downside_variance))))

        if downside_std == 0:
            return Decimal("0")

        # Daily risk-free rate
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        # Sortino
        sortino = (mean_return - daily_rf) / downside_std

        if annualize:
            sortino *= Decimal(str(math.sqrt(float(self.TRADING_DAYS_PER_YEAR))))

        return sortino

    def _compute_equity_based_returns(
        self,
        returns: list[Decimal],
        equity_curve: list[Decimal] | None = None,
    ) -> list[Decimal]:
        """
        Convert daily P&L to percentage returns using running equity.

        If equity_curve is provided, builds a date-aligned equity series
        to divide each day's PnL by the prior day's equity. Otherwise
        falls back to initial_capital (legacy behaviour).
        """
        if not equity_curve or len(equity_curve) < 2:
            return [r / self._initial_capital for r in returns]

        # equity_curve has one entry per bar; daily_returns has one per calendar day.
        # We approximate daily starting equity by walking equity forward day-by-day.
        # Build running start-of-day equity from cumulative daily PnL
        returns_pct: list[Decimal] = []
        running_equity = self._initial_capital
        for r in returns:
            if running_equity > 0:
                returns_pct.append(r / running_equity)
            else:
                returns_pct.append(Decimal("0"))
            running_equity += r
        return returns_pct

    def _calculate_calmar_ratio(
        self, total_return_pct: Decimal, max_drawdown_pct: Decimal
    ) -> Decimal:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            total_return_pct: Total return percentage
            max_drawdown_pct: Maximum drawdown percentage

        Returns:
            Calmar ratio
        """
        if max_drawdown_pct == 0:
            return Decimal("999") if total_return_pct > 0 else Decimal("0")
        return total_return_pct / max_drawdown_pct

    def combine_results(self, results: list[BacktestResult]) -> BacktestResult:
        """
        Combine multiple backtest results into one.

        Useful for walk-forward validation where OOS results
        need to be aggregated.

        Args:
            results: List of BacktestResult to combine

        Returns:
            Combined BacktestResult
        """
        if not results:
            return BacktestResult()

        # Combine trades
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        # Combine equity curves (append sequentially)
        combined_equity = []
        for r in results:
            if combined_equity and r.equity_curve:
                # Offset by last equity value
                offset = combined_equity[-1] - self._initial_capital
                combined_equity.extend([e + offset for e in r.equity_curve])
            else:
                combined_equity.extend(r.equity_curve)

        # Combine daily returns
        combined_daily = {}
        for r in results:
            combined_daily.update(r.daily_returns)

        # Combine funding paid
        combined_funding = sum(r.total_funding_paid for r in results)

        return self.calculate_all(all_trades, combined_equity, combined_daily, combined_funding)
