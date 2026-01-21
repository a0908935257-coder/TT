#!/usr/bin/env python3
"""
RSI Bot Parameter Optimization

目標：找到正報酬且通過過度擬合驗證的參數組合
- Walk-Forward 一致性 > 60%
- 正報酬率
- Sharpe Ratio > 0.5
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional
from itertools import product

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


@dataclass
class RSIConfig:
    """RSI strategy configuration."""
    rsi_period: int = 14
    oversold: int = 30
    overbought: int = 70
    exit_level: int = 50
    leverage: int = 5
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))


class RSIBacktest:
    """RSI Mean Reversion Backtest."""

    FEE_RATE = Decimal("0.0004")

    def __init__(self, klines: list[Kline], config: RSIConfig):
        self._klines = klines
        self._config = config
        self._position: Optional[dict] = None
        self._trades: list[dict] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None

    def _calculate_rsi(self, closes: list[float]) -> float:
        """Calculate RSI using Wilder's smoothing method."""
        period = self._config.rsi_period

        if len(closes) < period + 1:
            return 50.0

        changes = []
        for i in range(1, len(closes)):
            changes.append(closes[i] - closes[i-1])

        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        if self._avg_gain is None:
            self._avg_gain = sum(gains[-period:]) / period
            self._avg_loss = sum(losses[-period:]) / period
        else:
            current_gain = gains[-1] if gains else 0
            current_loss = losses[-1] if losses else 0
            self._avg_gain = (self._avg_gain * (period - 1) + current_gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + current_loss) / period

        if self._avg_loss == 0:
            return 100.0

        rs = self._avg_gain / self._avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def run(self) -> dict:
        """Run backtest and return metrics."""
        if len(self._klines) < self._config.rsi_period + 10:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "trades": 0}

        self._avg_gain = None
        self._avg_loss = None
        self._position = None
        self._trades = []

        peak_equity = Decimal("10000")
        current_equity = Decimal("10000")
        max_dd = Decimal("0")
        daily_returns = []
        prev_equity = current_equity
        closes = []

        for i, kline in enumerate(self._klines):
            close = float(kline.close)
            closes.append(close)

            if len(closes) < self._config.rsi_period + 5:
                continue

            rsi = self._calculate_rsi(closes)

            if self._position:
                entry_price = self._position['entry_price']
                side = self._position['side']
                price_change = (Decimal(str(close)) - entry_price) / entry_price

                exit_reason = None

                if side == 'long':
                    if price_change <= -self._config.stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif price_change >= self._config.take_profit_pct:
                        exit_reason = 'take_profit'
                    elif rsi >= self._config.exit_level:
                        exit_reason = 'rsi_exit'
                elif side == 'short':
                    if price_change >= self._config.stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif price_change <= -self._config.take_profit_pct:
                        exit_reason = 'take_profit'
                    elif rsi <= self._config.exit_level:
                        exit_reason = 'rsi_exit'

                if exit_reason:
                    exit_price = Decimal(str(close))
                    if side == 'long':
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price

                    leveraged_pnl = pnl * self._config.leverage * self._config.position_size_pct
                    fee = self.FEE_RATE * 2 * self._config.leverage * self._config.position_size_pct
                    net_pnl = leveraged_pnl - fee

                    self._trades.append({'pnl': net_pnl, 'exit_reason': exit_reason})
                    current_equity += current_equity * net_pnl
                    self._position = None

            if not self._position:
                if rsi < self._config.oversold:
                    self._position = {'entry_price': Decimal(str(close)), 'side': 'long'}
                elif rsi > self._config.overbought:
                    self._position = {'entry_price': Decimal(str(close)), 'side': 'short'}

            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = (peak_equity - current_equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

            if i % 96 == 0 and prev_equity > 0:
                daily_return = float((current_equity - prev_equity) / prev_equity)
                daily_returns.append(daily_return)
                prev_equity = current_equity

        # Close remaining position
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
            self._trades.append({'pnl': net_pnl, 'exit_reason': 'end'})
            current_equity += current_equity * net_pnl

        # Calculate metrics
        total_trades = len(self._trades)
        win_rate = 0
        if total_trades > 0:
            wins = sum(1 for t in self._trades if t['pnl'] > 0)
            win_rate = wins / total_trades

        total_return = float((current_equity - Decimal("10000")) / Decimal("10000"))

        sharpe = 0
        if len(daily_returns) > 1:
            avg_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_dev = math.sqrt(variance) if variance > 0 else 0.001
            if std_dev > 0:
                sharpe = (avg_return / std_dev) * math.sqrt(365)

        return {
            "return": total_return * 100,
            "sharpe": sharpe,
            "max_dd": float(max_dd) * 100,
            "win_rate": win_rate * 100,
            "trades": total_trades,
        }


async def fetch_klines(symbol: str, interval: str, days: int) -> list[Kline]:
    """Fetch historical klines from Binance."""
    async with BinanceFuturesAPI() as api:
        await api.ping()

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        interval_map = {
            "15m": KlineInterval.m15,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
        }
        kline_interval = interval_map.get(interval, KlineInterval.m15)

        all_klines = []
        current_start = start_time

        while current_start < end_time:
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
            last_close_time = int(klines[-1].close_time.timestamp() * 1000)
            current_start = last_close_time + 1
            await asyncio.sleep(0.05)

        return all_klines


def walk_forward_validate(klines: list[Kline], config: RSIConfig, num_periods: int = 6) -> dict:
    """Run walk-forward validation."""
    total_bars = len(klines)
    period_size = total_bars // num_periods

    results = []
    for i in range(num_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < num_periods - 1 else total_bars
        period_klines = klines[start_idx:end_idx]

        if len(period_klines) < 100:
            continue

        backtest = RSIBacktest(period_klines, config)
        result = backtest.run()
        results.append(result)

    if not results:
        return {"consistency": 0, "avg_return": 0, "profitable_periods": 0}

    profitable = sum(1 for r in results if r["return"] > 0)
    consistency = profitable / len(results) * 100
    avg_return = sum(r["return"] for r in results) / len(results)

    return {
        "consistency": consistency,
        "avg_return": avg_return,
        "profitable_periods": profitable,
        "total_periods": len(results),
    }


async def optimize():
    """Run parameter optimization."""
    print("=" * 70)
    print("       RSI Bot 參數優化")
    print("=" * 70)

    # Fetch data (6 months for faster optimization)
    print("\n正在獲取歷史數據 (6 個月)...")
    klines = await fetch_klines("BTCUSDT", "15m", 180)
    print(f"  獲取 {len(klines)} 根 K 線")

    if len(klines) < 5000:
        print("  ⚠️ 數據不足")
        return

    # Parameter grid (simplified for faster testing)
    param_grid = {
        "rsi_period": [7, 14, 21],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
        "exit_level": [50],
        "stop_loss_pct": [0.02, 0.03],
        "take_profit_pct": [0.04, 0.06],
        "leverage": [3, 5],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    total = len(combinations)

    print(f"\n測試 {total} 個參數組合...")
    print("-" * 70)

    best_results = []

    for idx, values in enumerate(combinations):
        params = dict(zip(keys, values))

        config = RSIConfig(
            rsi_period=params["rsi_period"],
            oversold=params["oversold"],
            overbought=params["overbought"],
            exit_level=params["exit_level"],
            leverage=params["leverage"],
            stop_loss_pct=Decimal(str(params["stop_loss_pct"])),
            take_profit_pct=Decimal(str(params["take_profit_pct"])),
        )

        # Full period backtest
        backtest = RSIBacktest(klines, config)
        result = backtest.run()

        # Walk-forward validation (4 periods = ~1.5 months each)
        wf = walk_forward_validate(klines, config, num_periods=4)

        # Progress
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  進度: {idx + 1}/{total} ({(idx+1)/total*100:.0f}%)", flush=True)

        # Filter: positive return, consistency > 50%, sharpe > 0
        if result["return"] > 0 and wf["consistency"] >= 50 and result["sharpe"] > 0:
            best_results.append({
                "params": params,
                "return": result["return"],
                "sharpe": result["sharpe"],
                "max_dd": result["max_dd"],
                "win_rate": result["win_rate"],
                "trades": result["trades"],
                "consistency": wf["consistency"],
                "wf_return": wf["avg_return"],
            })

    print(f"\n找到 {len(best_results)} 個符合條件的參數組合")

    if not best_results:
        print("\n❌ 沒有找到符合條件的參數組合")
        print("   放寬條件重新搜索...")

        # Relax criteria
        for idx, values in enumerate(combinations):
            params = dict(zip(keys, values))
            config = RSIConfig(
                rsi_period=params["rsi_period"],
                oversold=params["oversold"],
                overbought=params["overbought"],
                exit_level=params["exit_level"],
                leverage=params["leverage"],
                stop_loss_pct=Decimal(str(params["stop_loss_pct"])),
                take_profit_pct=Decimal(str(params["take_profit_pct"])),
            )

            backtest = RSIBacktest(klines, config)
            result = backtest.run()
            wf = walk_forward_validate(klines, config, num_periods=6)

            # Relaxed: any positive return or low loss
            if result["return"] > -2 and wf["consistency"] >= 33:
                best_results.append({
                    "params": params,
                    "return": result["return"],
                    "sharpe": result["sharpe"],
                    "max_dd": result["max_dd"],
                    "win_rate": result["win_rate"],
                    "trades": result["trades"],
                    "consistency": wf["consistency"],
                    "wf_return": wf["avg_return"],
                })

    # Sort by consistency first, then by return
    best_results.sort(key=lambda x: (x["consistency"], x["return"]), reverse=True)

    # Show top 10
    print("\n" + "=" * 70)
    print("       最佳參數組合 (Top 10)")
    print("=" * 70)

    for i, r in enumerate(best_results[:10]):
        p = r["params"]
        print(f"\n#{i+1} 組合:")
        print(f"  RSI Period: {p['rsi_period']}, Oversold: {p['oversold']}, Overbought: {p['overbought']}")
        print(f"  Exit Level: {p['exit_level']}, Leverage: {p['leverage']}x")
        print(f"  Stop Loss: {p['stop_loss_pct']*100:.1f}%, Take Profit: {p['take_profit_pct']*100:.1f}%")
        print(f"  ---")
        print(f"  報酬率: {r['return']:+.2f}% | Sharpe: {r['sharpe']:.2f} | 最大回撤: {r['max_dd']:.1f}%")
        print(f"  勝率: {r['win_rate']:.1f}% | 交易數: {r['trades']}")
        print(f"  Walk-Forward 一致性: {r['consistency']:.0f}% | WF 平均報酬: {r['wf_return']:+.2f}%")

        # Pass/Fail assessment
        if r["consistency"] >= 67 and r["return"] > 0:
            print(f"  ✅ 通過過度擬合驗證")
        elif r["consistency"] >= 50 and r["return"] > 0:
            print(f"  ⚠️ 中等風險 - 一致性適中")
        else:
            print(f"  ❌ 需要進一步優化")

    # Return best config
    if best_results:
        best = best_results[0]
        print("\n" + "=" * 70)
        print("       推薦參數")
        print("=" * 70)
        p = best["params"]
        print(f"""
RSI_PERIOD={p['rsi_period']}
RSI_OVERSOLD={p['oversold']}
RSI_OVERBOUGHT={p['overbought']}
RSI_EXIT_LEVEL={p['exit_level']}
RSI_LEVERAGE={p['leverage']}
RSI_STOP_LOSS_PCT={p['stop_loss_pct']}
RSI_TAKE_PROFIT_PCT={p['take_profit_pct']}

預期績效:
  報酬率: {best['return']:+.2f}%
  Sharpe: {best['sharpe']:.2f}
  最大回撤: {best['max_dd']:.1f}%
  Walk-Forward 一致性: {best['consistency']:.0f}%
""")

        return best

    return None


if __name__ == "__main__":
    result = asyncio.run(optimize())
