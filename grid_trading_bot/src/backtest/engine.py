"""
Backtest Engine Module.

The main orchestrator for running backtests with unified framework.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..core.models import Kline
from .config import BacktestConfig
from .metrics import MetricsCalculator
from .order import OrderSimulator, Signal, SignalType
from .position import Position, PositionManager
from .result import (
    BacktestResult,
    ExitReason,
    WalkForwardPeriod,
    WalkForwardResult,
)
from .strategy.base import BacktestContext, BacktestStrategy


class BacktestEngine:
    """
    Unified backtest engine.

    Orchestrates backtesting with pluggable strategies, handling
    order simulation, position management, and metrics calculation.

    Example:
        config = BacktestConfig(initial_capital=Decimal("10000"), leverage=2)
        engine = BacktestEngine(config)
        result = engine.run(klines, my_strategy)
        print(result.summary())
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self._config = config
        self._position_manager = PositionManager(max_positions=config.max_positions)
        self._order_simulator = OrderSimulator(config)
        self._metrics = MetricsCalculator(initial_capital=config.initial_capital)

        # State
        self._equity_curve: list[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}
        self._liquidation_count: int = 0
        self._last_funding_time: Optional[datetime] = None
        self._max_margin_utilization_pct: Decimal = Decimal("0")

    @property
    def config(self) -> BacktestConfig:
        """Get backtest configuration."""
        return self._config

    def run(
        self, klines: list[Kline], strategy: BacktestStrategy
    ) -> BacktestResult:
        """
        Execute backtest simulation.

        Process:
        1. Skip warmup period
        2. For each bar: check exits, check entries, update equity
        3. Close remaining positions at end
        4. Calculate metrics

        Args:
            klines: Historical kline data
            strategy: Strategy to backtest

        Returns:
            BacktestResult with all metrics
        """
        self._reset()
        strategy.reset()

        if not klines:
            return BacktestResult()

        warmup = strategy.warmup_period()
        if len(klines) <= warmup:
            return BacktestResult()

        # Process each bar after warmup
        for bar_idx in range(warmup, len(klines)):
            kline = klines[bar_idx]
            context = self._create_context(bar_idx, klines)

            # 0a. Check liquidation (before any exit logic)
            if self._config.use_margin and self._config.leverage > 1:
                self._check_liquidation(kline, bar_idx, strategy)

            # 0b. Apply funding rate
            if self._config.use_margin and self._config.leverage > 1:
                self._apply_funding(kline)

            # 1. Check exits for existing positions
            self._check_and_process_exits(kline, bar_idx, strategy, context)

            # 2. Update context after potential exits
            context = self._create_context(bar_idx, klines)

            # 3. Get strategy signal
            signal = strategy.on_kline(kline, context)

            # 4. Process signal
            if signal:
                self._process_signal(signal, kline, bar_idx, strategy)

            # 5. Update tracking prices
            self._update_position_tracking(kline)

            # 6. Update order simulator context for advanced models
            self._order_simulator.update_context(kline)

            # 7. Update equity curve
            self._update_equity(kline)

            # 8. Strategy end-of-bar callback
            context = self._create_context(bar_idx, klines)
            strategy.on_bar_close(kline, context)

        # Close any remaining positions at the end
        if self._position_manager.has_position:
            self._close_all_positions(
                klines[-1], len(klines) - 1, ExitReason.MANUAL
            )

        return self._calculate_result()

    def run_walk_forward(
        self,
        klines: list[Kline],
        strategy: BacktestStrategy,
        periods: int = 6,
        is_ratio: float = 0.7,
        consistency_threshold: float = 0.5,
    ) -> WalkForwardResult:
        """
        Execute walk-forward validation.

        Splits data into multiple periods, each with in-sample (IS)
        and out-of-sample (OOS) sections. Tests consistency between
        IS and OOS performance.

        Args:
            klines: Historical kline data
            strategy: Strategy to backtest
            periods: Number of walk-forward periods
            is_ratio: Ratio of data used for in-sample (0.7 = 70%)
            consistency_threshold: Minimum OOS/IS ratio to be consistent

        Returns:
            WalkForwardResult with all period results
        """
        if not klines or periods < 1:
            return WalkForwardResult()

        total_bars = len(klines)
        bars_per_period = total_bars // periods

        if bars_per_period < strategy.warmup_period() * 2:
            # Not enough data per period
            return WalkForwardResult()

        results = []

        for p in range(periods):
            period_start = p * bars_per_period
            period_end = (p + 1) * bars_per_period if p < periods - 1 else total_bars

            period_klines = klines[period_start:period_end]
            is_bars = int(len(period_klines) * is_ratio)

            is_klines = period_klines[:is_bars]
            oos_klines = period_klines[is_bars:]

            if len(is_klines) <= strategy.warmup_period() or len(oos_klines) <= strategy.warmup_period():
                continue

            # Run IS backtest
            is_result = self.run(is_klines, strategy)

            # Run OOS backtest
            oos_result = self.run(oos_klines, strategy)

            # Check consistency
            is_consistent = False
            if is_result.sharpe_ratio > 0:
                ratio = (
                    oos_result.sharpe_ratio / is_result.sharpe_ratio
                    if is_result.sharpe_ratio != 0
                    else Decimal("0")
                )
                is_consistent = float(ratio) >= consistency_threshold

            period_result = WalkForwardPeriod(
                period_num=p + 1,
                is_start=is_klines[0].open_time,
                is_end=is_klines[-1].close_time,
                oos_start=oos_klines[0].open_time,
                oos_end=oos_klines[-1].close_time,
                is_result=is_result,
                oos_result=oos_result,
                is_consistent=is_consistent,
            )
            results.append(period_result)

        if not results:
            return WalkForwardResult()

        # Calculate aggregate metrics
        consistency_pct = Decimal(
            sum(1 for r in results if r.is_consistent)
        ) / Decimal(len(results)) * Decimal("100")

        avg_oos_sharpe = sum(r.oos_result.sharpe_ratio for r in results) / Decimal(
            len(results)
        )

        # Combine OOS results
        oos_results = [r.oos_result for r in results]
        combined_oos = self._metrics.combine_results(oos_results)

        return WalkForwardResult(
            periods=results,
            consistency_pct=consistency_pct,
            avg_oos_sharpe=avg_oos_sharpe,
            combined_oos_result=combined_oos,
        )

    def _reset(self) -> None:
        """Reset engine state for new backtest."""
        self._position_manager.reset()
        self._equity_curve.clear()
        self._daily_pnl.clear()
        self._liquidation_count = 0
        self._last_funding_time = None
        self._max_margin_utilization_pct = Decimal("0")

    def _create_context(self, bar_idx: int, klines: list[Kline]) -> BacktestContext:
        """Create backtest context for current bar."""
        current_price = klines[bar_idx].close
        return BacktestContext(
            bar_index=bar_idx,
            klines=klines[: bar_idx + 1],
            current_position=self._position_manager.current_position,
            realized_pnl=self._position_manager.realized_pnl,
            equity=self._position_manager.total_equity(
                current_price, self._config.initial_capital, self._config.leverage
            ),
            initial_capital=self._config.initial_capital,
        )

    def _check_and_process_exits(
        self,
        kline: Kline,
        bar_idx: int,
        strategy: BacktestStrategy,
        context: BacktestContext,
    ) -> None:
        """Check and process exits for all open positions."""
        if not self._position_manager.has_position:
            return

        position = self._position_manager.current_position
        if position is None:
            return

        # Check stop loss
        sl_triggered, sl_price = self._order_simulator.check_stop_loss(position, kline)
        if sl_triggered and sl_price is not None:
            self._close_position(position, sl_price, kline, bar_idx, ExitReason.STOP_LOSS, strategy)
            return

        # Check take profit
        tp_triggered, tp_price = self._order_simulator.check_take_profit(position, kline)
        if tp_triggered and tp_price is not None:
            self._close_position(position, tp_price, kline, bar_idx, ExitReason.TAKE_PROFIT, strategy)
            return

        # Update trailing stop
        new_sl = strategy.update_trailing_stop(position, kline, context)
        if new_sl is not None:
            position.stop_loss = new_sl

        # Check strategy exit signal
        exit_signal = strategy.check_exit(position, kline, context)
        if exit_signal:
            exit_reason = ExitReason.SIGNAL
            if exit_signal.reason == "timeout":
                exit_reason = ExitReason.TIMEOUT
            elif exit_signal.reason == "reversal":
                exit_reason = ExitReason.REVERSAL

            exit_price = exit_signal.price if exit_signal.price else kline.close
            self._close_position(position, exit_price, kline, bar_idx, exit_reason, strategy)

    def _process_signal(
        self,
        signal: Signal,
        kline: Kline,
        bar_idx: int,
        strategy: BacktestStrategy,
    ) -> None:
        """Process a trading signal."""
        if signal.signal_type == SignalType.CLOSE_ALL:
            if self._position_manager.has_position:
                self._close_all_positions(kline, bar_idx, ExitReason.SIGNAL)
            return

        if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            if not self._position_manager.can_open_position:
                return

            # Use current equity for position sizing (equity-based sizing)
            if self._config.use_margin:
                current_equity = self._position_manager.total_equity(
                    kline.close, self._config.initial_capital, self._config.leverage
                )
                # Skip if equity depleted
                if current_equity <= 0:
                    return
                available = self._position_manager.available_margin(
                    kline.close, self._config.initial_capital, self._config.leverage
                )
                notional = current_equity * self._config.position_size_pct
                if notional > available:
                    return
                # Temporarily override notional for this trade
                self._order_simulator.override_notional(notional)

            side = "LONG" if signal.signal_type == SignalType.LONG_ENTRY else "SHORT"
            position = self._order_simulator.create_position(
                side=side,
                kline=kline,
                bar_index=bar_idx,
                target_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            if self._position_manager.open_position(position):
                strategy.on_position_opened(position)

    def _close_position(
        self,
        position: Position,
        exit_price: Decimal,
        kline: Kline,
        bar_idx: int,
        exit_reason: ExitReason,
        strategy: BacktestStrategy,
    ) -> None:
        """Close a position and record the trade."""
        _, exit_fee = self._order_simulator.simulate_exit_order(
            position, kline, exit_price
        )

        trade = self._position_manager.close_position(
            position=position,
            exit_price=exit_price,
            exit_time=kline.close_time,
            exit_bar=bar_idx,
            exit_fee=exit_fee,
            exit_reason=exit_reason,
            leverage=self._config.leverage,
        )

        # Update daily P&L
        date_key = kline.close_time.strftime("%Y-%m-%d")
        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += trade.pnl

        strategy.on_position_closed(trade)

    def _close_all_positions(
        self, kline: Kline, bar_idx: int, exit_reason: ExitReason
    ) -> None:
        """Close all open positions."""
        # Calculate effective fee rate (leverage-adjusted for futures)
        effective_fee_rate = self._config.fee_rate
        if self._config.use_margin and self._config.leverage > 1:
            effective_fee_rate = effective_fee_rate * Decimal(self._config.leverage)

        trades = self._position_manager.close_all_positions(
            exit_price=kline.close,
            exit_time=kline.close_time,
            exit_bar=bar_idx,
            fee_rate=effective_fee_rate,
            exit_reason=exit_reason,
            leverage=self._config.leverage,
        )

        date_key = kline.close_time.strftime("%Y-%m-%d")
        for trade in trades:
            if date_key not in self._daily_pnl:
                self._daily_pnl[date_key] = Decimal("0")
            self._daily_pnl[date_key] += trade.pnl

    def _update_position_tracking(self, kline: Kline) -> None:
        """Update tracking prices for all positions."""
        for position in self._position_manager.positions:
            position.update_tracking_prices(kline.close)

    def _update_equity(self, kline: Kline) -> None:
        """Update equity curve with current equity."""
        equity = self._position_manager.total_equity(
            kline.close, self._config.initial_capital, self._config.leverage
        )
        self._equity_curve.append(equity)

        # Track margin utilization
        if self._config.use_margin and equity > 0:
            used = self._position_manager.used_margin()
            utilization = (used / equity) * Decimal("100")
            if utilization > self._max_margin_utilization_pct:
                self._max_margin_utilization_pct = utilization

    def _check_liquidation(
        self, kline: Kline, bar_idx: int, strategy: BacktestStrategy
    ) -> None:
        """Check if any position should be liquidated."""
        if not self._position_manager.has_position:
            return

        for position in self._position_manager.positions:
            liq_price = position.liquidation_price(
                self._config.leverage, self._config.maintenance_margin_pct
            )
            if liq_price is None:
                continue

            triggered = False
            if position.side == "LONG" and kline.low <= liq_price:
                triggered = True
            elif position.side == "SHORT" and kline.high >= liq_price:
                triggered = True

            if triggered:
                # Liquidation fee on notional × leverage
                liq_fee = position.notional * Decimal(self._config.leverage) * self._config.liquidation_fee_pct
                exit_fee = liq_fee + position.entry_price * position.quantity * self._config.fee_rate

                trade = self._position_manager.close_position(
                    position=position,
                    exit_price=liq_price,
                    exit_time=kline.close_time,
                    exit_bar=bar_idx,
                    exit_fee=exit_fee,
                    exit_reason=ExitReason.LIQUIDATION,
                    leverage=self._config.leverage,
                )
                self._liquidation_count += 1

                date_key = kline.close_time.strftime("%Y-%m-%d")
                if date_key not in self._daily_pnl:
                    self._daily_pnl[date_key] = Decimal("0")
                self._daily_pnl[date_key] += trade.pnl

                strategy.on_position_closed(trade)
                break  # Re-check remaining positions on next bar

    def _apply_funding(self, kline: Kline) -> None:
        """Apply funding rate to open positions at funding intervals."""
        if not self._position_manager.has_position:
            return

        current_time = kline.close_time
        if self._last_funding_time is None:
            self._last_funding_time = current_time
            return

        hours_elapsed = (current_time - self._last_funding_time).total_seconds() / 3600
        if hours_elapsed >= self._config.funding_interval_hours:
            funding_periods = int(hours_elapsed // self._config.funding_interval_hours)
            for position in self._position_manager.positions:
                # Funding = notional × leverage × rate × periods
                funding = (
                    position.notional
                    * Decimal(self._config.leverage)
                    * self._config.funding_rate
                    * Decimal(funding_periods)
                )
                self._position_manager.add_funding_payment(funding)
            self._last_funding_time = current_time

    def _calculate_result(self) -> BacktestResult:
        """Calculate final backtest result."""
        result = self._metrics.calculate_all(
            trades=self._position_manager.trades,
            equity_curve=self._equity_curve,
            daily_returns=self._daily_pnl,
        )
        result.liquidation_count = self._liquidation_count
        result.total_funding_paid = self._position_manager.total_funding_paid
        result.max_margin_utilization_pct = self._max_margin_utilization_pct
        return result
