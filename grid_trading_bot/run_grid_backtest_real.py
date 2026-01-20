#!/usr/bin/env python3
"""
Grid Bot Backtest with Real Binance Historical Data.

Fetches real kline data from Binance Futures API and runs grid bot backtest.

Usage:
    python run_grid_backtest_real.py
    python run_grid_backtest_real.py --symbol ETHUSDT --interval 1h --days 60
"""

import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, List

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


@dataclass
class GridBacktestResult:
    """Grid Bot backtest result."""
    initial_capital: Decimal = field(default_factory=lambda: Decimal("0"))
    final_equity: Decimal = field(default_factory=lambda: Decimal("0"))
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    grid_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_and_hold_return: Decimal = field(default_factory=lambda: Decimal("0"))
    excess_return: Decimal = field(default_factory=lambda: Decimal("0"))
    start_price: Decimal = field(default_factory=lambda: Decimal("0"))
    end_price: Decimal = field(default_factory=lambda: Decimal("0"))
    price_change_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_trade_profit: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class GridConfig:
    """Grid configuration."""
    grid_count: int = 10
    position_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    leverage: int = 1
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))
    # Range settings - percentage from current price
    range_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))  # ±5% from center


class GridBacktest:
    """Grid Bot backtest engine with real data."""

    def __init__(
        self,
        klines: List[Kline],
        initial_capital: Decimal,
        config: GridConfig,
    ):
        self._klines = klines
        self._initial_capital = initial_capital
        self._config = config

        # State
        self._capital = initial_capital
        self._position = Decimal("0")
        self._avg_entry_price = Decimal("0")

        # Grid state
        self._grids: List[dict] = []
        self._upper_price = Decimal("0")
        self._lower_price = Decimal("0")
        self._grid_spacing = Decimal("0")

        # Statistics
        self._trades: List[dict] = []
        self._equity_curve: List[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}

    def _initialize_grid(self, center_price: Decimal) -> None:
        """Initialize grid around center price."""
        range_pct = self._config.range_pct
        self._upper_price = center_price * (Decimal("1") + range_pct)
        self._lower_price = center_price * (Decimal("1") - range_pct)
        self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)

        self._grids = []
        for i in range(self._config.grid_count + 1):
            price = self._lower_price + Decimal(i) * self._grid_spacing
            self._grids.append({
                "price": price,
                "buy_filled": False,
                "sell_filled": False,
            })

    def _check_rebuild_needed(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuild (price out of range)."""
        if current_price > self._upper_price or current_price < self._lower_price:
            return True
        return False

    def run(self) -> GridBacktestResult:
        """Run backtest."""
        if not self._klines or len(self._klines) < 50:
            return GridBacktestResult()

        # Initialize grid at first price
        first_price = self._klines[0].close
        self._initialize_grid(first_price)

        max_equity = self._initial_capital
        max_drawdown = Decimal("0")

        for kline in self._klines:
            current_price = kline.close
            date_key = kline.close_time.strftime("%Y-%m-%d")

            # Check if rebuild needed
            if self._check_rebuild_needed(current_price):
                # Close position before rebuild
                if self._position > 0:
                    pnl = self._position * (current_price - self._avg_entry_price)
                    fee = self._position * current_price * self._config.fee_rate
                    self._capital += self._position * current_price - fee
                    self._position = Decimal("0")

                # Rebuild grid around current price
                self._initialize_grid(current_price)

            # Process grid logic
            self._process_grids(current_price, date_key)

            # Calculate equity
            unrealized_pnl = Decimal("0")
            if self._position > 0:
                unrealized_pnl = self._position * (current_price - self._avg_entry_price)

            current_equity = self._capital + self._position * current_price
            self._equity_curve.append(current_equity)

            # Track max drawdown
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = max_equity - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close final position
        final_price = self._klines[-1].close
        if self._position > 0:
            self._capital += self._position * final_price
            self._position = Decimal("0")

        return self._calculate_result(max_drawdown, max_equity)

    def _process_grids(self, current_price: Decimal, date_key: str) -> None:
        """Process grid buy/sell logic."""
        trade_value = self._capital * self._config.position_pct

        for grid in self._grids:
            grid_price = grid["price"]

            # Buy condition: price at or below grid, not yet bought
            if current_price <= grid_price and not grid["buy_filled"]:
                if trade_value > Decimal("10"):
                    quantity = trade_value / current_price
                    fee = trade_value * self._config.fee_rate

                    # Update average entry price
                    if self._position > 0:
                        total_value = self._position * self._avg_entry_price + quantity * current_price
                        self._position += quantity
                        self._avg_entry_price = total_value / self._position
                    else:
                        self._position = quantity
                        self._avg_entry_price = current_price

                    self._capital -= trade_value + fee
                    grid["buy_filled"] = True
                    grid["sell_filled"] = False

                    self._trades.append({
                        "type": "buy",
                        "price": current_price,
                        "quantity": quantity,
                        "value": trade_value,
                        "fee": fee,
                    })

            # Sell condition: price at or above grid, has bought
            elif current_price >= grid_price and grid["buy_filled"] and not grid["sell_filled"]:
                if self._position > 0:
                    sell_quantity = min(
                        self._position,
                        trade_value / current_price
                    )

                    if sell_quantity > 0:
                        sell_value = sell_quantity * current_price
                        fee = sell_value * self._config.fee_rate
                        pnl = sell_quantity * (current_price - self._avg_entry_price) - fee

                        self._capital += sell_value - fee
                        self._position -= sell_quantity

                        grid["sell_filled"] = True
                        grid["buy_filled"] = False

                        # Track daily PnL
                        if date_key not in self._daily_pnl:
                            self._daily_pnl[date_key] = Decimal("0")
                        self._daily_pnl[date_key] += pnl

                        self._trades.append({
                            "type": "sell",
                            "price": current_price,
                            "quantity": sell_quantity,
                            "value": sell_value,
                            "fee": fee,
                            "pnl": pnl,
                        })

    def _calculate_result(self, max_drawdown: Decimal, max_equity: Decimal) -> GridBacktestResult:
        """Calculate final result."""
        result = GridBacktestResult()

        result.initial_capital = self._initial_capital
        result.final_equity = self._capital
        result.total_profit = self._capital - self._initial_capital
        result.total_return_pct = (result.total_profit / self._initial_capital) * Decimal("100")

        # Count trades
        buy_trades = [t for t in self._trades if t["type"] == "buy"]
        sell_trades = [t for t in self._trades if t["type"] == "sell"]
        result.total_trades = len(self._trades)

        # Win/loss from sell trades
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]
        result.num_wins = len(wins)
        result.num_losses = len(losses)

        if sell_trades:
            result.win_rate = Decimal(len(wins)) / Decimal(len(sell_trades)) * Decimal("100")
            result.grid_profit = sum(t.get("pnl", Decimal("0")) for t in sell_trades)
            result.avg_trade_profit = result.grid_profit / Decimal(len(sell_trades))

        # Drawdown
        result.max_drawdown = max_drawdown
        if max_equity > 0:
            result.max_drawdown_pct = (max_drawdown / max_equity) * Decimal("100")

        # Buy and hold comparison
        if self._klines:
            result.start_price = self._klines[0].close
            result.end_price = self._klines[-1].close
            result.price_change_pct = ((result.end_price - result.start_price) / result.start_price) * Decimal("100")
            result.buy_and_hold_return = result.price_change_pct
            result.excess_return = result.total_return_pct - result.buy_and_hold_return

        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe_ratio()

        return result

    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate annualized Sharpe ratio from daily PnL."""
        if not self._daily_pnl or len(self._daily_pnl) < 2:
            return Decimal("0")

        returns = list(self._daily_pnl.values())
        mean_return = sum(returns) / Decimal(len(returns))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)

        if variance <= 0:
            return Decimal("0")

        std_dev = Decimal(str(math.sqrt(float(variance))))
        if std_dev == 0:
            return Decimal("0")

        # Annualized Sharpe ratio (252 trading days)
        sharpe = (mean_return / std_dev) * Decimal(str(math.sqrt(252)))
        return sharpe


async def fetch_klines(
    symbol: str,
    interval: str,
    days: int,
) -> List[Kline]:
    """Fetch historical klines from Binance Futures."""
    print(f"\n正在從 Binance 獲取 {symbol} {interval} 歷史數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        await api.ping()
        print("  Binance API 連接成功")

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        interval_map = {
            "1m": KlineInterval.m1,
            "5m": KlineInterval.m5,
            "15m": KlineInterval.m15,
            "30m": KlineInterval.m30,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
            "1d": KlineInterval.d1,
        }
        kline_interval = interval_map.get(interval, KlineInterval.h1)

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

            if len(klines) < 1500:
                break

            current_start = int(klines[-1].close_time.timestamp() * 1000) + 1

        print(f"  獲取 {len(all_klines)} 根 K 線 ({batch_count} 批次)")
        return all_klines


def print_result(result: GridBacktestResult, config: GridConfig, symbol: str, interval: str):
    """Print backtest result."""
    print(f"\n{'='*65}")
    print(f"Grid Bot 回測結果 - {symbol} | {interval}")
    print(f"{'='*65}")

    print(f"\n📊 基本資訊:")
    print(f"  網格數量:       {config.grid_count}")
    print(f"  網格範圍:       ±{float(config.range_pct)*100:.1f}%")
    print(f"  單次交易比例:   {float(config.position_pct)*100:.1f}%")
    print(f"  槓桿:           {config.leverage}x")

    print(f"\n📈 價格資訊:")
    print(f"  起始價格:       ${float(result.start_price):,.2f}")
    print(f"  結束價格:       ${float(result.end_price):,.2f}")
    print(f"  價格變化:       {float(result.price_change_pct):+.2f}%")

    print(f"\n💰 績效摘要:")
    print(f"  初始資金:       ${float(result.initial_capital):,.2f}")
    print(f"  最終資金:       ${float(result.final_equity):,.2f}")
    print(f"  總盈虧:         ${float(result.total_profit):+,.2f}")
    print(f"  ROI:            {float(result.total_return_pct):+.2f}%")

    print(f"\n📊 交易統計:")
    print(f"  總交易次數:     {result.total_trades}")
    print(f"  獲勝次數:       {result.num_wins}")
    print(f"  虧損次數:       {result.num_losses}")
    print(f"  勝率:           {float(result.win_rate):.1f}%")
    print(f"  網格利潤:       ${float(result.grid_profit):+,.2f}")
    print(f"  平均交易盈虧:   ${float(result.avg_trade_profit):+,.2f}")

    print(f"\n⚠️  風險指標:")
    print(f"  最大回撤:       ${float(result.max_drawdown):,.2f}")
    print(f"  最大回撤比例:   {float(result.max_drawdown_pct):.2f}%")
    print(f"  Sharpe Ratio:   {float(result.sharpe_ratio):.2f}")

    print(f"\n🔄 策略比較:")
    print(f"  Grid Bot ROI:   {float(result.total_return_pct):+.2f}%")
    print(f"  持有策略 ROI:   {float(result.buy_and_hold_return):+.2f}%")
    print(f"  超額收益:       {float(result.excess_return):+.2f}%")

    # Verdict
    print(f"\n{'='*65}")
    if result.total_profit > 0:
        print(f"  ✅ Grid Bot 結果: 獲利 ${float(result.total_profit):,.2f}")
    else:
        print(f"  ❌ Grid Bot 結果: 虧損 ${float(result.total_profit):,.2f}")

    if result.excess_return > 0:
        print(f"  ✅ Grid Bot 表現優於持有策略 ({float(result.excess_return):+.2f}%)")
    else:
        print(f"  ❌ 持有策略表現優於 Grid Bot ({float(result.excess_return):+.2f}%)")
    print(f"{'='*65}\n")


async def run_backtest(
    symbol: str,
    interval: str,
    days: int,
    initial_capital: float,
    grid_count: int,
    range_pct: float,
) -> GridBacktestResult:
    """Run a single backtest."""
    klines = await fetch_klines(symbol, interval, days)

    if not klines:
        print("無法獲取 K 線數據")
        return GridBacktestResult()

    config = GridConfig(
        grid_count=grid_count,
        range_pct=Decimal(str(range_pct)),
        position_pct=Decimal("0.1"),
        leverage=1,
    )

    backtest = GridBacktest(
        klines=klines,
        initial_capital=Decimal(str(initial_capital)),
        config=config,
    )

    result = backtest.run()
    print_result(result, config, symbol, interval)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Grid Bot Backtest with Real Data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--interval", default="1h", help="Kline interval (1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=90, help="Backtest days")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--grids", type=int, default=10, help="Grid count")
    parser.add_argument("--range", type=float, default=0.05, help="Range percentage (0.05 = ±5%)")
    args = parser.parse_args()

    print("=" * 65)
    print("Grid Bot 回測 - 使用真實 Binance 數據")
    print("=" * 65)

    # Test multiple configurations
    test_cases = [
        {"symbol": args.symbol, "range_pct": 0.05, "grids": 10, "name": "窄幅 ±5%"},
        {"symbol": args.symbol, "range_pct": 0.10, "grids": 15, "name": "中幅 ±10%"},
        {"symbol": args.symbol, "range_pct": 0.15, "grids": 20, "name": "寬幅 ±15%"},
    ]

    results = []
    for case in test_cases:
        print(f"\n{'='*65}")
        print(f"測試配置: {case['name']} ({case['grids']} 網格)")
        print(f"{'='*65}")

        result = await run_backtest(
            symbol=case["symbol"],
            interval=args.interval,
            days=args.days,
            initial_capital=args.capital,
            grid_count=case["grids"],
            range_pct=case["range_pct"],
        )
        results.append((case["name"], result))

    # Summary
    print("\n" + "=" * 65)
    print("總結比較")
    print("=" * 65)
    print(f"{'配置':<15} {'ROI':>10} {'勝率':>10} {'Sharpe':>10} {'超額收益':>12}")
    print("-" * 65)

    for name, result in results:
        print(f"{name:<15} {float(result.total_return_pct):>+9.2f}% {float(result.win_rate):>9.1f}% {float(result.sharpe_ratio):>10.2f} {float(result.excess_return):>+11.2f}%")

    print("=" * 65)

    # Find best strategy
    best_result = max(results, key=lambda x: x[1].sharpe_ratio)
    print(f"\n最佳策略: {best_result[0]} (Sharpe: {float(best_result[1].sharpe_ratio):.2f})")

    print("\n回測完成!")


if __name__ == "__main__":
    asyncio.run(main())
