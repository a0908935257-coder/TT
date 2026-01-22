#!/usr/bin/env python3
"""
RSI Bot Backtest - 測試 RSI Momentum 策略績效

Walk-Forward 驗證通過的參數 (2 年, 12 期, 67% 一致性):
- RSI Period: 25
- Entry Level: 50, Momentum Threshold: 5
- Leverage: 5x
- Stop Loss: 2%, Take Profit: 4%
- Sharpe: 0.65, Return: +12.0%, Max DD: 9.5%
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


@dataclass
class RSIBacktestResult:
    """RSI backtest result."""
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    take_profit_exits: int = 0
    stop_loss_exits: int = 0
    rsi_exits: int = 0


@dataclass
class RSIConfig:
    """RSI Momentum strategy configuration."""
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"

    # RSI Parameters (Walk-Forward validated)
    rsi_period: int = 25  # Walk-Forward validated: RSI=25
    entry_level: int = 50  # Center level for crossover
    momentum_threshold: int = 5  # RSI must cross by this amount

    # Position Management
    leverage: int = 5  # Walk-Forward validated
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))


class RSIBacktest:
    """RSI Momentum Backtest."""

    FEE_RATE = Decimal("0.0004")

    def __init__(self, klines: list[Kline], config: RSIConfig):
        self._klines = klines
        self._config = config
        self._position: Optional[dict] = None
        self._trades: list[dict] = []

        # RSI state (Wilder's smoothing)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._prev_rsi: float = 50.0  # Track previous RSI for crossover detection

    def _calculate_rsi(self, closes: list[float]) -> float:
        """Calculate RSI using Wilder's smoothing method."""
        period = self._config.rsi_period

        if len(closes) < period + 1:
            return 50.0  # Neutral

        # Calculate price changes
        changes = []
        for i in range(1, len(closes)):
            changes.append(closes[i] - closes[i-1])

        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        # Initial averages (SMA for first calculation)
        if self._avg_gain is None:
            self._avg_gain = sum(gains[-period:]) / period
            self._avg_loss = sum(losses[-period:]) / period
        else:
            # Wilder's smoothing for subsequent calculations
            current_gain = gains[-1] if gains else 0
            current_loss = losses[-1] if losses else 0
            self._avg_gain = (self._avg_gain * (period - 1) + current_gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + current_loss) / period

        if self._avg_loss == 0:
            return 100.0

        rs = self._avg_gain / self._avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run(self) -> RSIBacktestResult:
        """Run backtest."""
        result = RSIBacktestResult()

        if len(self._klines) < self._config.rsi_period + 10:
            print(f"  ⚠️ 數據不足: {len(self._klines)} 根 K 線")
            return result

        # Reset state
        self._avg_gain = None
        self._avg_loss = None
        self._position = None
        self._trades = []
        self._prev_rsi = 50.0

        # Track equity for drawdown
        equity_curve = []
        peak_equity = Decimal("10000")
        current_equity = Decimal("10000")
        max_dd = Decimal("0")

        # Daily returns for Sharpe
        daily_returns = []
        prev_equity = current_equity

        # Collect closes for RSI
        closes = []

        entry_level = self._config.entry_level
        momentum_threshold = self._config.momentum_threshold

        for i, kline in enumerate(self._klines):
            close = float(kline.close)
            closes.append(close)

            # Need enough data for RSI
            if len(closes) < self._config.rsi_period + 5:
                continue

            # Calculate RSI
            rsi = self._calculate_rsi(closes)

            # Check position management first
            if self._position:
                entry_price = self._position['entry_price']
                side = self._position['side']
                price_change = (Decimal(str(close)) - entry_price) / entry_price

                exit_reason = None

                if side == 'long':
                    # Stop loss
                    if price_change <= -self._config.stop_loss_pct:
                        exit_reason = 'stop_loss'
                    # Take profit
                    elif price_change >= self._config.take_profit_pct:
                        exit_reason = 'take_profit'
                    # RSI exit: bearish momentum (crosses below entry_level - threshold)
                    elif rsi < entry_level - momentum_threshold:
                        exit_reason = 'rsi_exit'

                elif side == 'short':
                    # Stop loss (price went up)
                    if price_change >= self._config.stop_loss_pct:
                        exit_reason = 'stop_loss'
                    # Take profit (price went down)
                    elif price_change <= -self._config.take_profit_pct:
                        exit_reason = 'take_profit'
                    # RSI exit: bullish momentum (crosses above entry_level + threshold)
                    elif rsi > entry_level + momentum_threshold:
                        exit_reason = 'rsi_exit'

                if exit_reason:
                    # Close position
                    exit_price = Decimal(str(close))

                    if side == 'long':
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price

                    # Apply leverage and fees
                    leveraged_pnl = pnl * self._config.leverage * self._config.position_size_pct
                    fee = self.FEE_RATE * 2 * self._config.leverage * self._config.position_size_pct
                    net_pnl = leveraged_pnl - fee

                    # Record trade
                    trade = {
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': side,
                        'pnl': net_pnl,
                        'exit_reason': exit_reason,
                    }
                    self._trades.append(trade)

                    # Update equity
                    current_equity += current_equity * net_pnl

                    # Clear position
                    self._position = None

                    # Count exits
                    if exit_reason == 'stop_loss':
                        result.stop_loss_exits += 1
                    elif exit_reason == 'take_profit':
                        result.take_profit_exits += 1
                    else:
                        result.rsi_exits += 1

            # Entry signals (only if not in position) - Momentum crossover
            if not self._position:
                # Bullish momentum: RSI crossed above entry_level + threshold
                if self._prev_rsi <= entry_level and rsi > entry_level + momentum_threshold:
                    self._position = {
                        'entry_price': Decimal(str(close)),
                        'side': 'long',
                        'entry_bar': i,
                    }
                    result.long_trades += 1

                # Bearish momentum: RSI crossed below entry_level - threshold
                elif self._prev_rsi >= entry_level and rsi < entry_level - momentum_threshold:
                    self._position = {
                        'entry_price': Decimal(str(close)),
                        'side': 'short',
                        'entry_bar': i,
                    }
                    result.short_trades += 1

            # Track previous RSI for crossover detection
            self._prev_rsi = rsi

            # Track equity
            equity_curve.append(current_equity)

            # Update max drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = (peak_equity - current_equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

            # Daily returns (every 96 bars for 15m = 1 day)
            if i % 96 == 0 and prev_equity > 0:
                daily_return = float((current_equity - prev_equity) / prev_equity)
                daily_returns.append(daily_return)
                prev_equity = current_equity

        # Close any remaining position at last price
        if self._position:
            close = float(self._klines[-1].close)
            exit_price = Decimal(str(close))
            entry_price = self._position['entry_price']
            side = self._position['side']

            if side == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            leveraged_pnl = pnl * self._config.leverage * self._config.position_size_pct
            fee = self.FEE_RATE * 2 * self._config.leverage * self._config.position_size_pct
            net_pnl = leveraged_pnl - fee

            trade = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': side,
                'pnl': net_pnl,
                'exit_reason': 'end_of_period',
            }
            self._trades.append(trade)
            current_equity += current_equity * net_pnl

        # Calculate results
        result.total_trades = len(self._trades)

        if result.total_trades > 0:
            wins = [t for t in self._trades if t['pnl'] > 0]
            losses = [t for t in self._trades if t['pnl'] <= 0]

            result.num_wins = len(wins)
            result.num_losses = len(losses)
            result.win_rate = Decimal(str(len(wins) / result.total_trades))

            if wins:
                result.avg_win = sum(t['pnl'] for t in wins) / len(wins)
            if losses:
                result.avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses))

            total_win = sum(t['pnl'] for t in wins) if wins else Decimal("0")
            total_loss = abs(sum(t['pnl'] for t in losses)) if losses else Decimal("0")

            if total_loss > 0:
                result.profit_factor = total_win / total_loss

        # Total profit
        initial = Decimal("10000")
        result.total_profit = (current_equity - initial) / initial

        # Max drawdown
        result.max_drawdown = max_dd

        # Sharpe ratio
        if len(daily_returns) > 1:
            avg_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_dev = math.sqrt(variance) if variance > 0 else 0.001
            if std_dev > 0:
                result.sharpe_ratio = Decimal(str((avg_return / std_dev) * math.sqrt(365)))

        return result


async def fetch_klines(symbol: str, interval: str, days: int) -> list[Kline]:
    """Fetch historical klines from Binance."""
    print(f"  正在獲取 {symbol} {interval} K 線數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        await api.ping()

        # Use millisecond timestamps
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        # Map interval string to KlineInterval
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

        all_klines = []
        current_start = start_time
        batch_count = 0

        while current_start < end_time:
            try:
                klines = await api.get_klines(
                    symbol=symbol,
                    interval=kline_interval,
                    start_time=current_start,
                    end_time=end_time,
                    limit=1500,
                )

                if not klines:
                    break

                all_klines.extend(klines)
                batch_count += 1

                # Move to next batch
                last_close_time = int(klines[-1].close_time.timestamp() * 1000)
                current_start = last_close_time + 1

                print(f"    已獲取 {len(all_klines)} 根 K 線 (批次 {batch_count})", end='\r')
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"\n    獲取數據錯誤: {e}")
                break

        print(f"    完成: {len(all_klines)} 根 K 線                    ")
        return all_klines


async def run_backtest():
    """Run RSI Bot backtest."""
    print("=" * 70)
    print("       RSI Bot 回測 - Momentum Strategy")
    print("=" * 70)

    # RSI Bot Walk-Forward validated parameters
    config = RSIConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        rsi_period=25,
        entry_level=50,
        momentum_threshold=5,
        leverage=5,
        position_size_pct=Decimal("0.1"),
        stop_loss_pct=Decimal("0.02"),
        take_profit_pct=Decimal("0.04"),
    )

    print(f"\n策略參數 (Walk-Forward Validated):")
    print(f"  RSI Period: {config.rsi_period}")
    print(f"  Entry Level: {config.entry_level} ± {config.momentum_threshold}")
    print(f"  Long: RSI crosses above {config.entry_level + config.momentum_threshold}")
    print(f"  Short: RSI crosses below {config.entry_level - config.momentum_threshold}")
    print(f"  Stop Loss: {float(config.stop_loss_pct)*100:.1f}%")
    print(f"  Take Profit: {float(config.take_profit_pct)*100:.1f}%")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Position Size: {float(config.position_size_pct)*100:.0f}%")

    # Test periods
    periods = [
        ("近三個月", 90),
        ("近半年", 180),
        ("近一年", 365),
    ]

    results = []

    for period_name, days in periods:
        print(f"\n{'='*70}")
        print(f"測試期間: {period_name} ({days} 天)")
        print("=" * 70)

        # Fetch data
        klines = await fetch_klines(config.symbol, config.timeframe, days)

        if len(klines) < 100:
            print(f"  ⚠️ 數據不足，跳過")
            continue

        # Run backtest
        backtest = RSIBacktest(klines, config)
        result = backtest.run()

        # Print results
        print(f"\n📊 績效指標:")
        print(f"  總報酬率: {float(result.total_profit)*100:+.2f}%")
        print(f"  Sharpe Ratio: {float(result.sharpe_ratio):.2f}")
        print(f"  最大回撤: {float(result.max_drawdown)*100:.2f}%")
        print(f"  勝率: {float(result.win_rate)*100:.1f}%")
        print(f"  總交易數: {result.total_trades}")
        print(f"    - 做多: {result.long_trades}")
        print(f"    - 做空: {result.short_trades}")

        if result.total_trades > 0:
            print(f"\n📈 出場統計:")
            print(f"  止盈出場: {result.take_profit_exits}")
            print(f"  止損出場: {result.stop_loss_exits}")
            print(f"  RSI 出場: {result.rsi_exits}")

            if result.profit_factor > 0:
                print(f"\n  盈虧比: {float(result.profit_factor):.2f}")
            if result.avg_win > 0:
                print(f"  平均獲利: {float(result.avg_win)*100:+.2f}%")
            if result.avg_loss > 0:
                print(f"  平均虧損: {float(result.avg_loss)*100:-.2f}%")

        results.append({
            "period": period_name,
            "days": days,
            "return": float(result.total_profit) * 100,
            "sharpe": float(result.sharpe_ratio),
            "max_dd": float(result.max_drawdown) * 100,
            "win_rate": float(result.win_rate) * 100,
            "trades": result.total_trades,
            "profit_factor": float(result.profit_factor),
        })

    # Walk-Forward Validation
    print(f"\n{'='*70}")
    print("Walk-Forward 驗證")
    print("=" * 70)

    # Fetch 1 year of data for walk-forward validation
    print("\n正在獲取完整數據進行 Walk-Forward 驗證...")
    all_klines = await fetch_klines(config.symbol, config.timeframe, 365)

    if len(all_klines) < 5000:
        print("  ⚠️ 數據不足進行 Walk-Forward 驗證")
    else:
        # Split into 6 periods (each ~2 months)
        total_bars = len(all_klines)
        period_size = total_bars // 6

        wf_results = []

        print(f"\n將 {total_bars} 根 K 線分為 6 個測試期間:")

        for i in range(6):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < 5 else total_bars

            period_klines = all_klines[start_idx:end_idx]

            if len(period_klines) < 100:
                continue

            # Determine date range
            start_date = period_klines[0].open_time.strftime('%Y-%m-%d')
            end_date = period_klines[-1].close_time.strftime('%Y-%m-%d')

            # Run backtest for this period
            backtest = RSIBacktest(period_klines, config)
            result = backtest.run()

            ret = float(result.total_profit) * 100
            profitable = ret > 0

            status = "✅" if profitable else "❌"
            print(f"  期間 {i+1}: {start_date} ~ {end_date}")
            print(f"    {status} 報酬: {ret:+.2f}% | 交易: {result.total_trades} | 勝率: {float(result.win_rate)*100:.0f}%")

            wf_results.append({
                "period": i + 1,
                "return": ret,
                "profitable": profitable,
                "trades": result.total_trades,
                "win_rate": float(result.win_rate) * 100,
            })

        # Summary
        if wf_results:
            profitable_periods = sum(1 for r in wf_results if r['profitable'])
            consistency = profitable_periods / len(wf_results) * 100
            avg_return = sum(r['return'] for r in wf_results) / len(wf_results)
            total_trades = sum(r['trades'] for r in wf_results)

            print(f"\n🔄 Walk-Forward 結果:")
            print(f"  獲利期數: {profitable_periods}/{len(wf_results)}")
            print(f"  一致性: {consistency:.0f}%")
            print(f"  平均報酬: {avg_return:+.2f}%")
            print(f"  總交易數: {total_trades}")

    # Final Summary
    print(f"\n{'='*70}")
    print("       回測結果總結")
    print("=" * 70)

    print("\n📈 各期間績效:")
    print("-" * 70)
    print(f"{'期間':<18} {'報酬率':>10} {'Sharpe':>10} {'最大回撤':>10} {'勝率':>8} {'交易數':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['period']:<18} {r['return']:>+9.2f}% {r['sharpe']:>10.2f} {r['max_dd']:>9.2f}% {r['win_rate']:>7.1f}% {r['trades']:>8}")

    print("-" * 70)

    # Overall assessment
    if results:
        avg_return = sum(r['return'] for r in results) / len(results)
        avg_sharpe = sum(r['sharpe'] for r in results) / len(results)
        max_dd = max(r['max_dd'] for r in results)
        avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

        print(f"\n🎯 整體評估:")
        print(f"  平均報酬: {avg_return:+.2f}%")
        print(f"  平均 Sharpe: {avg_sharpe:.2f}")
        print(f"  最大回撤: {max_dd:.2f}%")
        print(f"  平均勝率: {avg_win_rate:.1f}%")

        # Rating
        print(f"\n📋 策略評級:")
        if avg_sharpe >= 1.0 and avg_return > 0:
            print("  ⭐⭐⭐ 優秀 - Sharpe >= 1.0, 正報酬")
        elif avg_sharpe >= 0.5 and avg_return > 0:
            print("  ⭐⭐ 良好 - Sharpe >= 0.5, 正報酬")
        elif avg_return > 0:
            print("  ⭐ 可接受 - 正報酬但風險調整後表現一般")
        else:
            print("  ❌ 需要優化 - 負報酬")

    print("\n" + "=" * 70)
    print("       RSI Bot 回測完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_backtest())
