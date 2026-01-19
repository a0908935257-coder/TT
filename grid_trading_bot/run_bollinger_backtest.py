#!/usr/bin/env python3
"""
Bollinger Bot Backtest Runner.

Runs backtest on different market conditions and displays results.

Usage:
    python run_bollinger_backtest.py
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.core.models import Kline
from src.bots.bollinger.indicators import BollingerCalculator
from src.bots.bollinger.models import BollingerConfig, PositionSide

# Import the backtest classes from test file
import sys
sys.path.insert(0, '.')

# Define backtest result and engine here for standalone use
@dataclass
class BollingerBacktestResult:
    """Bollinger Bot backtest result."""
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_holding_time: timedelta = field(default_factory=lambda: timedelta(0))
    num_wins: int = 0
    num_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    filtered_by_squeeze: int = 0
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))


class BollingerBacktest:
    """Bollinger Band mean reversion backtest engine."""

    FEE_RATE = Decimal("0.0004")

    def __init__(self, klines: list[Kline], config: BollingerConfig):
        self._klines = klines
        self._config = config
        self._calculator = BollingerCalculator(
            period=config.bb_period,
            std_multiplier=config.bb_std,
            bbw_lookback=config.bbw_lookback,
            bbw_threshold_pct=config.bbw_threshold_pct,
        )
        self._position: Optional[dict] = None
        self._trades: list[dict] = []
        self._equity_curve: list[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}
        self._filtered_signals: int = 0

    def run(self) -> BollingerBacktestResult:
        if not self._klines or len(self._klines) < self._config.bb_period + 50:
            return BollingerBacktestResult()

        self._calculator.initialize(self._klines[:self._config.bb_period + 50])

        for i in range(self._config.bb_period + 50, len(self._klines)):
            kline = self._klines[i]
            klines_subset = self._klines[:i + 1]
            self._process_kline(kline, klines_subset, i)

        return self._calculate_result()

    def _process_kline(self, kline: Kline, klines_subset: list[Kline], bar_idx: int) -> None:
        current_price = kline.close
        date_key = kline.close_time.strftime("%Y-%m-%d")
        bands, bbw = self._calculator.get_all(klines_subset)

        if self._position:
            self._check_exit(kline, bands, current_price, bar_idx, date_key)
            return

        if bbw.is_squeeze:
            self._filtered_signals += 1
            return

        if current_price <= bands.lower:
            self._enter_position(
                side=PositionSide.LONG,
                entry_price=bands.lower,
                take_profit=bands.middle,
                stop_loss=bands.lower * (Decimal("1") - self._config.stop_loss_pct),
                entry_bar=bar_idx,
                entry_time=kline.close_time,
            )
        elif current_price >= bands.upper:
            self._enter_position(
                side=PositionSide.SHORT,
                entry_price=bands.upper,
                take_profit=bands.middle,
                stop_loss=bands.upper * (Decimal("1") + self._config.stop_loss_pct),
                entry_bar=bar_idx,
                entry_time=kline.close_time,
            )

        self._update_equity(current_price)

    def _enter_position(self, side, entry_price, take_profit, stop_loss, entry_bar, entry_time):
        notional = Decimal("10000") * self._config.position_size_pct
        quantity = notional / entry_price
        self._position = {
            "side": side,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "quantity": quantity,
            "entry_bar": entry_bar,
            "entry_time": entry_time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _check_exit(self, kline, bands, current_price, bar_idx, date_key):
        if not self._position:
            return

        should_exit = False
        exit_price = current_price
        exit_reason = ""
        side = self._position["side"]

        if side == PositionSide.LONG:
            if kline.high >= bands.middle:
                should_exit, exit_price, exit_reason = True, bands.middle, "take_profit"
        else:
            if kline.low <= bands.middle:
                should_exit, exit_price, exit_reason = True, bands.middle, "take_profit"

        if not should_exit:
            if side == PositionSide.LONG:
                if kline.low <= self._position["stop_loss"]:
                    should_exit, exit_price, exit_reason = True, self._position["stop_loss"], "stop_loss"
            else:
                if kline.high >= self._position["stop_loss"]:
                    should_exit, exit_price, exit_reason = True, self._position["stop_loss"], "stop_loss"

        if not should_exit:
            hold_bars = bar_idx - self._position["entry_bar"]
            if hold_bars >= self._config.max_hold_bars:
                should_exit, exit_price, exit_reason = True, current_price, "timeout"

        if should_exit:
            self._exit_position(exit_price, exit_reason, kline.close_time, bar_idx, date_key)

    def _exit_position(self, exit_price, exit_reason, exit_time, exit_bar, date_key):
        if not self._position:
            return

        side = self._position["side"]
        entry_price = self._position["entry_price"]
        quantity = self._position["quantity"]

        if side == PositionSide.LONG:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl *= Decimal(self._config.leverage)
        exit_fee = (exit_price * quantity) * self.FEE_RATE
        total_fee = self._position["entry_fee"] + exit_fee
        net_pnl = pnl - total_fee

        trade = {
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": net_pnl,
            "fee": total_fee,
            "entry_time": self._position["entry_time"],
            "exit_time": exit_time,
            "holding_bars": exit_bar - self._position["entry_bar"],
            "exit_reason": exit_reason,
        }
        self._trades.append(trade)

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        self._position = None

    def _update_equity(self, current_price):
        realized = sum(t["pnl"] for t in self._trades)
        unrealized = Decimal("0")
        if self._position:
            side = self._position["side"]
            entry_price = self._position["entry_price"]
            quantity = self._position["quantity"]
            if side == PositionSide.LONG:
                unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = (entry_price - current_price) * quantity
            unrealized *= Decimal(self._config.leverage)
        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> BollingerBacktestResult:
        result = BollingerBacktestResult()
        if not self._trades:
            return result

        result.total_profit = sum(t["pnl"] for t in self._trades)
        result.total_trades = len(self._trades)

        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)

        if self._trades:
            result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * Decimal("100")

        if wins:
            result.avg_win = sum(t["pnl"] for t in wins) / Decimal(len(wins))
        if losses:
            result.avg_loss = abs(sum(t["pnl"] for t in losses)) / Decimal(len(losses))

        gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.profit_factor = Decimal("999")

        result.long_trades = len([t for t in self._trades if t["side"] == PositionSide.LONG])
        result.short_trades = len([t for t in self._trades if t["side"] == PositionSide.SHORT])
        result.filtered_by_squeeze = self._filtered_signals

        if self._equity_curve:
            peak = self._equity_curve[0]
            max_dd = Decimal("0")
            for equity in self._equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_dd:
                    max_dd = drawdown
            result.max_drawdown = max_dd

        if self._trades:
            total_bars = sum(t["holding_bars"] for t in self._trades)
            avg_bars = total_bars / len(self._trades)
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        return result


def generate_mean_reversion_klines(num_klines: int = 500) -> list[Kline]:
    """Generate oscillating klines for mean reversion."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    random.seed(555)

    for i in range(num_klines):
        cycle_pos = (i % 30) / 30.0
        wave = math.sin(cycle_pos * 2 * math.pi)
        offset = Decimal(str(wave * 0.03))
        noise = Decimal(str(random.uniform(-0.005, 0.005)))
        center = base_price * (Decimal("1") + offset + noise)

        open_price = center * Decimal("0.998")
        close_price = center * Decimal("1.002")
        high_price = max(open_price, close_price) * Decimal("1.005")
        low_price = min(open_price, close_price) * Decimal("0.995")

        kline = Kline(
            symbol="BTCUSDT",
            interval="15m",
            open_time=start_time + timedelta(minutes=i * 15),
            close_time=start_time + timedelta(minutes=(i + 1) * 15 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


def generate_volatile_klines(num_klines: int = 500) -> list[Kline]:
    """Generate high volatility klines."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    random.seed(999)

    for i in range(num_klines):
        swing = Decimal(str(random.uniform(-0.08, 0.08)))
        center = base_price * (Decimal("1") + swing)

        direction = 1 if random.random() > 0.5 else -1
        body = Decimal(str(random.uniform(0.02, 0.04)))

        open_price = center
        close_price = center * (Decimal("1") + body * Decimal(direction))
        high_price = max(open_price, close_price) * Decimal("1.02")
        low_price = min(open_price, close_price) * Decimal("0.98")

        kline = Kline(
            symbol="BTCUSDT",
            interval="15m",
            open_time=start_time + timedelta(minutes=i * 15),
            close_time=start_time + timedelta(minutes=(i + 1) * 15 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("5000"),
            quote_volume=Decimal("250000000"),
            trades=500,
        )
        klines.append(kline)

    return klines


def generate_sideways_klines(num_klines: int = 500) -> list[Kline]:
    """Generate sideways market klines."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    random.seed(123)

    for i in range(num_klines):
        offset = Decimal(str(random.uniform(-0.05, 0.05)))
        center = base_price * (Decimal("1") + offset)

        open_price = center * Decimal("0.999")
        close_price = center * Decimal("1.001")
        high_price = max(open_price, close_price) * Decimal("1.01")
        low_price = min(open_price, close_price) * Decimal("0.99")

        kline = Kline(
            symbol="BTCUSDT",
            interval="15m",
            open_time=start_time + timedelta(minutes=i * 15),
            close_time=start_time + timedelta(minutes=(i + 1) * 15 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


def print_result(name: str, result: BollingerBacktestResult):
    """Print backtest result in formatted output."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  總交易數:        {result.total_trades}")
    print(f"  勝率:            {result.win_rate:.1f}%")
    print(f"  多單交易:        {result.long_trades}")
    print(f"  空單交易:        {result.short_trades}")
    print(f"  獲勝次數:        {result.num_wins}")
    print(f"  虧損次數:        {result.num_losses}")
    print(f"  總盈虧:          {result.total_profit:+.2f} USDT")
    print(f"  平均獲利:        {result.avg_win:+.4f} USDT")
    print(f"  平均虧損:        {result.avg_loss:.4f} USDT")
    print(f"  獲利因子:        {result.profit_factor:.2f}")
    print(f"  最大回撤:        {result.max_drawdown:.2f} USDT")
    print(f"  BBW 壓縮過濾:    {result.filtered_by_squeeze} 次")
    print(f"  平均持倉時間:    {result.avg_holding_time}")
    print(f"{'='*60}")


def main():
    """Run backtest on different market conditions."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Bollinger Bot 歷史數據回測                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    config = BollingerConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=20,
        stop_loss_pct=Decimal("0.015"),
        max_hold_bars=16,
        leverage=2,
        position_size_pct=Decimal("0.1"),
    )

    print("\n配置參數:")
    print(f"  交易對:         {config.symbol}")
    print(f"  時間框架:       {config.timeframe}")
    print(f"  BB 週期:        {config.bb_period}")
    print(f"  BB 標準差:      {config.bb_std}")
    print(f"  BBW 回溯:       {config.bbw_lookback}")
    print(f"  壓縮閾值:       {config.bbw_threshold_pct}%")
    print(f"  止損:           {config.stop_loss_pct * 100}%")
    print(f"  最大持倉 K 線:  {config.max_hold_bars}")
    print(f"  槓桿:           {config.leverage}x")
    print(f"  倉位大小:       {config.position_size_pct * 100}%")

    # 1. Mean Reversion Market
    print("\n正在回測均值回歸行情...")
    mean_reversion_klines = generate_mean_reversion_klines(500)
    backtest1 = BollingerBacktest(klines=mean_reversion_klines, config=config)
    result1 = backtest1.run()
    print_result("均值回歸行情 (Mean Reversion)", result1)

    # 2. Sideways Market
    print("\n正在回測橫盤行情...")
    sideways_klines = generate_sideways_klines(500)
    backtest2 = BollingerBacktest(klines=sideways_klines, config=config)
    result2 = backtest2.run()
    print_result("橫盤行情 (Sideways)", result2)

    # 3. Volatile Market
    print("\n正在回測高波動行情...")
    volatile_klines = generate_volatile_klines(500)
    backtest3 = BollingerBacktest(klines=volatile_klines, config=config)
    result3 = backtest3.run()
    print_result("高波動行情 (Volatile)", result3)

    # Summary
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                      回測總結                                ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    total_profit = result1.total_profit + result2.total_profit + result3.total_profit
    total_trades = result1.total_trades + result2.total_trades + result3.total_trades
    total_wins = result1.num_wins + result2.num_wins + result3.num_wins

    overall_win_rate = Decimal(total_wins) / Decimal(total_trades) * Decimal("100") if total_trades > 0 else Decimal("0")

    print(f"║  總交易次數:       {total_trades:>30}     ║")
    print(f"║  總盈虧:           {total_profit:>+30.2f} USDT ║")
    print(f"║  整體勝率:         {overall_win_rate:>30.1f}%    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n回測完成!")


if __name__ == "__main__":
    main()
