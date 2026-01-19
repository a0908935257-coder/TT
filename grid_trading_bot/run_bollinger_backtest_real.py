#!/usr/bin/env python3
"""
Bollinger Bot Backtest with Real Binance Historical Data.

Fetches real kline data from Binance Futures API and runs backtest.

Usage:
    python run_bollinger_backtest_real.py
    python run_bollinger_backtest_real.py --symbol ETHUSDT --interval 15m --days 30
"""

import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI
from src.bots.bollinger.indicators import BollingerCalculator
from src.bots.bollinger.models import BollingerConfig, PositionSide


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
    take_profit_exits: int = 0
    stop_loss_exits: int = 0
    timeout_exits: int = 0


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
        self._exit_reasons: dict[str, int] = {"take_profit": 0, "stop_loss": 0, "timeout": 0}

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
            self._exit_reasons[exit_reason] += 1
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

        result.take_profit_exits = self._exit_reasons["take_profit"]
        result.stop_loss_exits = self._exit_reasons["stop_loss"]
        result.timeout_exits = self._exit_reasons["timeout"]

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
            # Calculate based on actual interval
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        return result


async def fetch_klines(
    symbol: str,
    interval: str,
    days: int,
) -> list[Kline]:
    """Fetch historical klines from Binance Futures."""
    print(f"\n正在從 Binance 獲取 {symbol} {interval} 歷史數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        # Test connectivity
        await api.ping()
        print("  Binance API 連接成功")

        # Calculate time range
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        # Map interval to KlineInterval
        interval_map = {
            "1m": KlineInterval.m1,
            "5m": KlineInterval.m5,
            "15m": KlineInterval.m15,
            "30m": KlineInterval.m30,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
            "1d": KlineInterval.d1,
        }
        kline_interval = interval_map.get(interval, KlineInterval.m15)

        # Fetch klines in batches (max 1500 per request)
        all_klines = []
        current_start = start_time
        batch_count = 0

        while current_start < end_time:
            klines = await api.get_klines(
                symbol=symbol,
                interval=kline_interval,
                limit=1500,
                start_time=current_start,
                end_time=end_time,
            )

            if not klines:
                break

            all_klines.extend(klines)
            batch_count += 1

            # Update start time for next batch
            if klines:
                last_close_time = int(klines[-1].close_time.timestamp() * 1000)
                current_start = last_close_time + 1

            print(f"  已獲取 {len(all_klines)} 根 K 線 (批次 {batch_count})")

            # Avoid rate limiting
            await asyncio.sleep(0.1)

        print(f"  總共獲取 {len(all_klines)} 根 K 線")
        return all_klines


def print_result(name: str, result: BollingerBacktestResult, symbol: str, interval: str):
    """Print backtest result in formatted output."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  {symbol} | {interval}")
    print(f"{'='*65}")
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
    print(f"{'='*65}")
    print(f"  出場統計:")
    print(f"    止盈出場:      {result.take_profit_exits}")
    print(f"    止損出場:      {result.stop_loss_exits}")
    print(f"    超時出場:      {result.timeout_exits}")
    print(f"  BBW 壓縮過濾:    {result.filtered_by_squeeze} 次")
    print(f"  平均持倉時間:    {result.avg_holding_time}")
    print(f"{'='*65}")


async def run_backtest(symbol: str, interval: str, days: int, config: BollingerConfig):
    """Run backtest for a single symbol/interval."""
    # Fetch real data
    klines = await fetch_klines(symbol, interval, days)

    if len(klines) < config.bb_period + 100:
        print(f"  警告: K 線數量不足 ({len(klines)}), 跳過回測")
        return None

    # Run backtest
    print(f"\n正在回測 {symbol} {interval}...")
    backtest = BollingerBacktest(klines=klines, config=config)
    result = backtest.run()

    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bollinger Bot Backtest with Real Data")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--leverage", type=int, default=2, help="Leverage")
    parser.add_argument("--stop-loss", type=float, default=0.015, help="Stop loss percentage")
    args = parser.parse_args()

    print("\n")
    print("+" + "="*63 + "+")
    print("|" + " "*15 + "Bollinger Bot 真實數據回測" + " "*15 + "|")
    print("+" + "="*63 + "+")

    config = BollingerConfig(
        symbol=args.symbol,
        timeframe=args.interval,
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=20,
        stop_loss_pct=Decimal(str(args.stop_loss)),
        max_hold_bars=16,
        leverage=args.leverage,
        position_size_pct=Decimal("0.1"),
    )

    print("\n配置參數:")
    print(f"  交易對:         {config.symbol}")
    print(f"  時間框架:       {config.timeframe}")
    print(f"  回測天數:       {args.days} 天")
    print(f"  BB 週期:        {config.bb_period}")
    print(f"  BB 標準差:      {config.bb_std}")
    print(f"  BBW 回溯:       {config.bbw_lookback}")
    print(f"  壓縮閾值:       {config.bbw_threshold_pct}%")
    print(f"  止損:           {config.stop_loss_pct * 100}%")
    print(f"  最大持倉 K 線:  {config.max_hold_bars}")
    print(f"  槓桿:           {config.leverage}x")
    print(f"  倉位大小:       {config.position_size_pct * 100}%")

    # Run single symbol backtest
    result = await run_backtest(args.symbol, args.interval, args.days, config)

    if result:
        print_result("回測結果", result, args.symbol, args.interval)

        # Summary
        print("\n")
        print("+" + "="*63 + "+")
        print("|" + " "*25 + "回測總結" + " "*25 + "|")
        print("+" + "="*63 + "+")

        if result.total_trades > 0:
            print(f"|  交易對:         {args.symbol:>40}   |")
            print(f"|  時間框架:       {args.interval:>40}   |")
            print(f"|  回測天數:       {args.days:>40}   |")
            print(f"|  總交易次數:     {result.total_trades:>40}   |")
            print(f"|  總盈虧:         {str(result.total_profit) + ' USDT':>40}   |")
            print(f"|  勝率:           {str(result.win_rate) + '%':>40}   |")
            print(f"|  獲利因子:       {str(result.profit_factor):>40}   |")

            # Calculate annualized return
            if args.days > 0:
                daily_return = result.total_profit / Decimal(args.days)
                annualized = daily_return * Decimal("365")
                print(f"|  日均收益:       {str(daily_return.quantize(Decimal('0.01'))) + ' USDT':>40}   |")
                print(f"|  年化收益(估):   {str(annualized.quantize(Decimal('0.01'))) + ' USDT':>40}   |")
        else:
            print("|  無交易產生                                                   |")

        print("+" + "="*63 + "+")

    print("\n回測完成!")


if __name__ == "__main__":
    asyncio.run(main())
