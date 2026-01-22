#!/usr/bin/env python3
"""
Supertrend Bot Walk-Forward 驗證優化.

驗證標準 (與其他策略一致):
- 數據: 2 年 (2024-01 ~ 2026-01)
- Walk-Forward: 8 期分割
- 通過標準: 一致性 ≥67% (6/8 時段獲利)
"""

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


@dataclass
class BacktestResult:
    """回測結果"""
    total_return_pct: float
    sharpe: float
    max_dd: float
    trades: int
    win_rate: float
    profit_factor: float


class SupertrendBacktester:
    """Supertrend 回測器"""

    def __init__(self, klines: List[Kline], config: Dict[str, Any]):
        self.klines = klines
        self.config = config
        self.leverage = config.get('leverage', 5)
        self.fee_rate = 0.0004

    def calculate_supertrend(self, highs, lows, closes, atr_period, atr_multiplier):
        """計算 Supertrend"""
        n = len(closes)
        if n < atr_period + 10:
            return []

        # Calculate ATR
        atrs = []
        for i in range(n):
            if i < 1:
                atrs.append(highs[i] - lows[i])
            else:
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1])
                )
                atrs.append(tr)

        # Calculate Supertrend
        upper_band = [0.0] * n
        lower_band = [0.0] * n
        supertrend = [0.0] * n
        trend = [1] * n

        for i in range(atr_period, n):
            atr = sum(atrs[i - atr_period + 1:i + 1]) / atr_period
            hl2 = (highs[i] + lows[i]) / 2
            basic_upper = hl2 + atr_multiplier * atr
            basic_lower = hl2 - atr_multiplier * atr

            if i > atr_period:
                if basic_lower > lower_band[i - 1] or closes[i - 1] < lower_band[i - 1]:
                    lower_band[i] = basic_lower
                else:
                    lower_band[i] = lower_band[i - 1]

                if basic_upper < upper_band[i - 1] or closes[i - 1] > upper_band[i - 1]:
                    upper_band[i] = basic_upper
                else:
                    upper_band[i] = upper_band[i - 1]
            else:
                upper_band[i] = basic_upper
                lower_band[i] = basic_lower

            if i > atr_period:
                if supertrend[i - 1] == upper_band[i - 1]:
                    if closes[i] > upper_band[i]:
                        trend[i] = 1
                        supertrend[i] = lower_band[i]
                    else:
                        trend[i] = -1
                        supertrend[i] = upper_band[i]
                else:
                    if closes[i] < lower_band[i]:
                        trend[i] = -1
                        supertrend[i] = upper_band[i]
                    else:
                        trend[i] = 1
                        supertrend[i] = lower_band[i]
            else:
                if closes[i] > upper_band[i]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]

        return [{'index': i, 'trend': trend[i], 'supertrend': supertrend[i]}
                for i in range(atr_period, n)]

    def run(self) -> BacktestResult:
        """執行回測"""
        atr_period = self.config.get('atr_period', 10)
        atr_multiplier = self.config.get('atr_multiplier', 3.0)
        stop_loss_pct = self.config.get('stop_loss_pct', 0.02)

        closes = [float(k.close) for k in self.klines]
        highs = [float(k.high) for k in self.klines]
        lows = [float(k.low) for k in self.klines]

        st_data = self.calculate_supertrend(highs, lows, closes, atr_period, atr_multiplier)

        if len(st_data) < 10:
            return BacktestResult(0, 0, 0, 0, 0, 0)

        # Backtest
        trades = []
        position = 0  # 1 = long, -1 = short, 0 = none
        entry_price = 0
        initial_capital = 10000.0
        capital = initial_capital
        equity = [capital]
        peak = capital
        max_dd = 0

        for i, st in enumerate(st_data[1:], 1):
            prev_st = st_data[i - 1]
            price = closes[st['index']]

            # Check for signal flip
            if prev_st['trend'] != st['trend']:
                # Close existing position
                if position != 0:
                    if position == 1:
                        pnl_pct = (price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - price) / entry_price

                    pnl = pnl_pct * self.leverage * capital * 0.1
                    fee = capital * 0.1 * self.fee_rate * 2
                    capital += pnl - fee
                    trades.append(pnl - fee)

                # Open new position
                if st['trend'] == 1:
                    position = 1
                    entry_price = price
                else:
                    position = -1
                    entry_price = price

            # Check stop loss
            if position != 0:
                if position == 1:
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - price) / entry_price

                if pnl_pct < -stop_loss_pct:
                    pnl = pnl_pct * self.leverage * capital * 0.1
                    fee = capital * 0.1 * self.fee_rate * 2
                    capital += pnl - fee
                    trades.append(pnl - fee)
                    position = 0
                    entry_price = 0

            equity.append(capital)
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Close remaining position
        if position != 0:
            price = closes[-1]
            if position == 1:
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price
            pnl = pnl_pct * self.leverage * capital * 0.1
            fee = capital * 0.1 * self.fee_rate * 2
            capital += pnl - fee
            trades.append(pnl - fee)

        # Calculate metrics
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0)

        total_return = (capital - initial_capital) / initial_capital * 100
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe
        returns = [(equity[i] - equity[i-1]) / equity[i-1]
                   for i in range(1, len(equity)) if equity[i-1] > 0]
        if len(returns) > 10:
            avg_ret = sum(returns) / len(returns)
            std_ret = math.sqrt(sum((r - avg_ret)**2 for r in returns) / len(returns))
            sharpe = (avg_ret / std_ret) * math.sqrt(365 * 24 * 4) if std_ret > 0 else 0  # 15m bars
        else:
            sharpe = 0

        return BacktestResult(
            total_return_pct=total_return,
            sharpe=sharpe,
            max_dd=max_dd * 100,
            trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
        )


def walk_forward(klines: List[Kline], config: Dict, periods: int = 8) -> Dict:
    """Walk-Forward 驗證"""
    n = len(klines) // periods
    results = []
    period_results = []

    for i in range(periods):
        start = i * n
        end = (i + 1) * n if i < periods - 1 else len(klines)
        period_klines = klines[start:end]

        bt = SupertrendBacktester(period_klines, config)
        r = bt.run()
        results.append(r.total_return_pct > 0)
        period_results.append(r)

    return {
        "consistency": sum(results) / len(results) * 100,
        "profitable": sum(results),
        "total": len(results),
        "periods": period_results,
    }


async def fetch_klines(days: int, timeframe: str = "15m") -> List[Kline]:
    """獲取歷史數據"""
    interval = KlineInterval.m15 if timeframe == "15m" else KlineInterval.h1

    async with BinanceFuturesAPI() as api:
        await api.ping()
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        all_klines = []
        current = start_time

        while current < end_time:
            klines = await api.get_klines(
                symbol="BTCUSDT",
                interval=interval,
                start_time=current,
                end_time=end_time,
                limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            current = int(klines[-1].close_time.timestamp() * 1000) + 1
            await asyncio.sleep(0.05)

        return all_klines


async def main():
    print("=" * 70)
    print("       Supertrend Bot Walk-Forward 驗證優化")
    print("=" * 70)

    print("\n獲取 2 年數據 (15m timeframe)...")
    klines = await fetch_klines(730, "15m")
    print(f"  獲取 {len(klines)} 根 K 線")
    print(f"  時間範圍: {klines[0].open_time.date()} ~ {klines[-1].close_time.date()}")

    # 測試配置 - 擴展參數範圍，包含低槓桿選項
    configs = [
        # === 低槓桿組 (推薦) - 穩定性優先 ===
        ("2x, ATR=10, M=2.5, SL=3%", {'leverage': 2, 'atr_period': 10, 'atr_multiplier': 2.5, 'stop_loss_pct': 0.03}),
        ("2x, ATR=10, M=3.0, SL=3%", {'leverage': 2, 'atr_period': 10, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.03}),
        ("2x, ATR=14, M=2.5, SL=3%", {'leverage': 2, 'atr_period': 14, 'atr_multiplier': 2.5, 'stop_loss_pct': 0.03}),
        ("2x, ATR=14, M=3.0, SL=3%", {'leverage': 2, 'atr_period': 14, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.03}),
        ("2x, ATR=20, M=3.0, SL=3%", {'leverage': 2, 'atr_period': 20, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.03}),
        ("2x, ATR=20, M=3.5, SL=3%", {'leverage': 2, 'atr_period': 20, 'atr_multiplier': 3.5, 'stop_loss_pct': 0.03}),
        # === 中槓桿組 ===
        ("3x, ATR=10, M=2.5, SL=2%", {'leverage': 3, 'atr_period': 10, 'atr_multiplier': 2.5, 'stop_loss_pct': 0.02}),
        ("3x, ATR=14, M=3.0, SL=2%", {'leverage': 3, 'atr_period': 14, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.02}),
        ("3x, ATR=20, M=3.0, SL=2%", {'leverage': 3, 'atr_period': 20, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.02}),
        # === 原始配置對照 ===
        ("5x, ATR=10, M=3.0, SL=2% (原始)", {'leverage': 5, 'atr_period': 10, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.02}),
        # === 更長 ATR 週期 ===
        ("2x, ATR=25, M=3.0, SL=3%", {'leverage': 2, 'atr_period': 25, 'atr_multiplier': 3.0, 'stop_loss_pct': 0.03}),
        ("2x, ATR=30, M=3.5, SL=3%", {'leverage': 2, 'atr_period': 30, 'atr_multiplier': 3.5, 'stop_loss_pct': 0.03}),
    ]

    print(f"\n測試 {len(configs)} 個參數組合...")
    print("-" * 70)

    results = []
    for name, config in configs:
        # Full backtest
        bt = SupertrendBacktester(klines, config)
        full = bt.run()

        # Walk-forward validation (8 periods)
        wf = walk_forward(klines, config, periods=8)

        status = "✅" if full.total_return_pct > 0 and wf["consistency"] >= 67 else "⚠️" if full.total_return_pct > 0 else "❌"

        print(f"\n{status} {name}")
        print(f"   全期回測: {full.total_return_pct:+.1f}% | Sharpe: {full.sharpe:.2f} | 回撤: {full.max_dd:.1f}%")
        print(f"   Walk-Forward: {wf['consistency']:.0f}% ({wf['profitable']}/{wf['total']} 時段獲利)")

        period_str = " | ".join([f"P{i+1}:{r.total_return_pct:+.0f}%" for i, r in enumerate(wf['periods'])])
        print(f"   各時段: {period_str}")

        results.append({
            "name": name,
            "config": config,
            "return": full.total_return_pct,
            "sharpe": full.sharpe,
            "max_dd": full.max_dd,
            "trades": full.trades,
            "win_rate": full.win_rate,
            "consistency": wf["consistency"],
            "periods": wf["periods"],
        })

    # Sort by consistency then return
    results.sort(key=lambda x: (x["consistency"], x["return"]), reverse=True)

    print("\n" + "=" * 70)
    print("       驗證結果總結")
    print("=" * 70)

    # Best passing
    passing = [r for r in results if r["return"] > 0 and r["consistency"] >= 67]

    if passing:
        best = passing[0]
        print(f"\n✅ 通過驗證的最佳策略: {best['name']}")
        print(f"   報酬: {best['return']:+.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   最大回撤: {best['max_dd']:.1f}%")
        print(f"   Walk-Forward 一致性: {best['consistency']:.0f}%")
        print(f"   交易次數: {best['trades']}")
    else:
        print("\n❌ 沒有策略通過 Walk-Forward 驗證 (一致性 ≥67%)")
        print("   Supertrend 策略可能不適合當前市場環境")

        if results:
            best = results[0]
            print(f"\n   最佳策略 (未通過):")
            print(f"   {best['name']}")
            print(f"   報酬: {best['return']:+.1f}%, 一致性: {best['consistency']:.0f}%")

    # Show all results
    print("\n" + "=" * 100)
    print("       所有配置結果 (按一致性排序)")
    print("=" * 100)
    print(f"\n{'配置名稱':<35} {'報酬%':>10} {'Sharpe':>10} {'回撤%':>10} {'一致性':>10} {'狀態':>8}")
    print("-" * 100)

    for r in results:
        status = "✅" if r["return"] > 0 and r["consistency"] >= 67 else "⚠️"
        print(f"{r['name']:<35} {r['return']:>+9.1f}% {r['sharpe']:>10.2f} {r['max_dd']:>9.1f}% {r['consistency']:>9.0f}% {status:>8}")

    print("-" * 100)

    # Recommendation
    if passing:
        best = max(passing, key=lambda x: x["sharpe"])
        print(f"\n🏆 推薦配置: {best['name']}")
        print(f"   報酬: {best['return']:+.1f}% (2 年)")
        print(f"   年化報酬: {best['return']/2:+.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   最大回撤: {best['max_dd']:.1f}%")
        print(f"   Walk-Forward 一致性: {best['consistency']:.0f}% ({best['consistency']/100*8:.0f}/8 時段)")
        print(f"   勝率: {best['win_rate']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
