#!/usr/bin/env python3
"""
RSI Bot Quick Parameter Optimization
Uses 90 days of data for faster testing
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


@dataclass
class RSIConfig:
    rsi_period: int = 14
    oversold: int = 30
    overbought: int = 70
    exit_level: int = 50
    leverage: int = 5
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))


class RSIBacktest:
    FEE_RATE = Decimal("0.0004")

    def __init__(self, klines: list[Kline], config: RSIConfig):
        self._klines = klines
        self._config = config
        self._position = None
        self._trades = []
        self._avg_gain = None
        self._avg_loss = None

    def _calculate_rsi(self, closes: list[float]) -> float:
        period = self._config.rsi_period
        if len(closes) < period + 1:
            return 50.0

        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        if self._avg_gain is None:
            self._avg_gain = sum(gains[-period:]) / period
            self._avg_loss = sum(losses[-period:]) / period
        else:
            self._avg_gain = (self._avg_gain * (period - 1) + gains[-1]) / period
            self._avg_loss = (self._avg_loss * (period - 1) + losses[-1]) / period

        if self._avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + self._avg_gain / self._avg_loss))

    def run(self) -> dict:
        if len(self._klines) < self._config.rsi_period + 10:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "trades": 0}

        self._avg_gain = None
        self._avg_loss = None
        self._position = None
        self._trades = []

        peak = current = Decimal("10000")
        max_dd = Decimal("0")
        closes = []

        for kline in self._klines:
            close = float(kline.close)
            closes.append(close)

            if len(closes) < self._config.rsi_period + 5:
                continue

            rsi = self._calculate_rsi(closes[-50:])

            if self._position:
                entry = self._position['entry']
                side = self._position['side']
                chg = (Decimal(str(close)) - entry) / entry

                exit_reason = None
                if side == 'long':
                    if chg <= -self._config.stop_loss_pct:
                        exit_reason = 'sl'
                    elif chg >= self._config.take_profit_pct:
                        exit_reason = 'tp'
                    elif rsi >= self._config.exit_level:
                        exit_reason = 'rsi'
                else:
                    if chg >= self._config.stop_loss_pct:
                        exit_reason = 'sl'
                    elif chg <= -self._config.take_profit_pct:
                        exit_reason = 'tp'
                    elif rsi <= self._config.exit_level:
                        exit_reason = 'rsi'

                if exit_reason:
                    pnl = chg if side == 'long' else -chg
                    net = pnl * self._config.leverage * self._config.position_size_pct
                    net -= self.FEE_RATE * 2 * self._config.leverage * self._config.position_size_pct
                    self._trades.append(float(net))
                    current += current * net
                    self._position = None

            if not self._position:
                if rsi < self._config.oversold:
                    self._position = {'entry': Decimal(str(close)), 'side': 'long'}
                elif rsi > self._config.overbought:
                    self._position = {'entry': Decimal(str(close)), 'side': 'short'}

            if current > peak:
                peak = current
            dd = (peak - current) / peak
            if dd > max_dd:
                max_dd = dd

        total_trades = len(self._trades)
        win_rate = sum(1 for t in self._trades if t > 0) / total_trades * 100 if total_trades else 0
        total_return = float((current - Decimal("10000")) / Decimal("10000")) * 100

        return {
            "return": total_return,
            "max_dd": float(max_dd) * 100,
            "win_rate": win_rate,
            "trades": total_trades,
        }


async def fetch_klines(days: int) -> list[Kline]:
    async with BinanceFuturesAPI() as api:
        await api.ping()
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        all_klines = []
        current = start_time

        while current < end_time:
            klines = await api.get_klines(
                symbol="BTCUSDT",
                interval=KlineInterval.m15,
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


def test_params(klines: list[Kline], params: dict) -> dict:
    """Test a single parameter set with walk-forward validation."""
    config = RSIConfig(
        rsi_period=params["rsi_period"],
        oversold=params["oversold"],
        overbought=params["overbought"],
        exit_level=params["exit_level"],
        leverage=params["leverage"],
        stop_loss_pct=Decimal(str(params["stop_loss_pct"])),
        take_profit_pct=Decimal(str(params["take_profit_pct"])),
    )

    # Full backtest
    bt = RSIBacktest(klines, config)
    full = bt.run()

    # Walk-forward (3 periods)
    n = len(klines) // 3
    wf_results = []
    for i in range(3):
        start = i * n
        end = (i + 1) * n if i < 2 else len(klines)
        bt = RSIBacktest(klines[start:end], config)
        r = bt.run()
        wf_results.append(r["return"] > 0)

    consistency = sum(wf_results) / 3 * 100

    return {
        "params": params,
        "return": full["return"],
        "max_dd": full["max_dd"],
        "win_rate": full["win_rate"],
        "trades": full["trades"],
        "consistency": consistency,
    }


async def main():
    print("=" * 60)
    print("       RSI Bot 快速參數優化")
    print("=" * 60)

    print("\n獲取 90 天數據...")
    klines = await fetch_klines(90)
    print(f"  獲取 {len(klines)} 根 K 線")

    # Focused parameter combinations
    param_sets = [
        # Original parameters (baseline)
        {"rsi_period": 14, "oversold": 20, "overbought": 80, "exit_level": 50, "leverage": 7, "stop_loss_pct": 0.02, "take_profit_pct": 0.03},
        # Less extreme RSI thresholds
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        {"rsi_period": 14, "oversold": 35, "overbought": 65, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.025, "take_profit_pct": 0.05},
        # Shorter RSI period (more signals)
        {"rsi_period": 7, "oversold": 30, "overbought": 70, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        {"rsi_period": 7, "oversold": 25, "overbought": 75, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        # Longer RSI period (fewer signals, more reliable)
        {"rsi_period": 21, "oversold": 30, "overbought": 70, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        {"rsi_period": 21, "oversold": 35, "overbought": 65, "exit_level": 50, "leverage": 3, "stop_loss_pct": 0.03, "take_profit_pct": 0.06},
        # Different exit levels
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit_level": 45, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit_level": 55, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.04},
        # Higher take profit
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.06},
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit_level": 50, "leverage": 3, "stop_loss_pct": 0.03, "take_profit_pct": 0.08},
        # Lower leverage, wider stops
        {"rsi_period": 14, "oversold": 30, "overbought": 70, "exit_level": 50, "leverage": 3, "stop_loss_pct": 0.04, "take_profit_pct": 0.08},
        {"rsi_period": 14, "oversold": 35, "overbought": 65, "exit_level": 50, "leverage": 3, "stop_loss_pct": 0.05, "take_profit_pct": 0.10},
        # Asymmetric thresholds (trend following)
        {"rsi_period": 14, "oversold": 40, "overbought": 60, "exit_level": 50, "leverage": 3, "stop_loss_pct": 0.03, "take_profit_pct": 0.06},
        {"rsi_period": 14, "oversold": 35, "overbought": 60, "exit_level": 50, "leverage": 5, "stop_loss_pct": 0.02, "take_profit_pct": 0.05},
        # Very conservative
        {"rsi_period": 21, "oversold": 25, "overbought": 75, "exit_level": 50, "leverage": 3, "stop_loss_pct": 0.03, "take_profit_pct": 0.06},
    ]

    print(f"\n測試 {len(param_sets)} 個參數組合...")
    print("-" * 60)

    results = []
    for i, params in enumerate(param_sets):
        result = test_params(klines, params)
        results.append(result)
        status = "✅" if result["return"] > 0 and result["consistency"] >= 67 else "⚠️" if result["return"] > 0 else "❌"
        print(f"  {i+1:2d}. {status} 報酬: {result['return']:+6.2f}% | 一致性: {result['consistency']:.0f}% | 勝率: {result['win_rate']:.0f}%")

    # Sort by consistency then return
    results.sort(key=lambda x: (x["consistency"], x["return"]), reverse=True)

    print("\n" + "=" * 60)
    print("       最佳參數組合")
    print("=" * 60)

    for i, r in enumerate(results[:5]):
        p = r["params"]
        status = "✅ 通過" if r["return"] > 0 and r["consistency"] >= 67 else "⚠️ 中等" if r["return"] > 0 else "❌ 不佳"
        print(f"\n#{i+1} {status}")
        print(f"  RSI: period={p['rsi_period']}, oversold={p['oversold']}, overbought={p['overbought']}")
        print(f"  Exit: {p['exit_level']}, Leverage: {p['leverage']}x")
        print(f"  SL: {p['stop_loss_pct']*100:.1f}%, TP: {p['take_profit_pct']*100:.1f}%")
        print(f"  報酬: {r['return']:+.2f}% | 一致性: {r['consistency']:.0f}% | 勝率: {r['win_rate']:.1f}% | 交易: {r['trades']}")

    # Find best passing params
    passing = [r for r in results if r["return"] > 0 and r["consistency"] >= 67]
    if passing:
        best = passing[0]
        p = best["params"]
        print("\n" + "=" * 60)
        print("       推薦參數 (通過過度擬合驗證)")
        print("=" * 60)
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
  Walk-Forward 一致性: {best['consistency']:.0f}%
  勝率: {best['win_rate']:.1f}%
  最大回撤: {best['max_dd']:.1f}%
""")
        return best
    else:
        print("\n⚠️ 沒有找到通過過度擬合驗證的參數")
        print("   最佳組合可能仍有過擬風險")
        if results:
            best = results[0]
            p = best["params"]
            print(f"\n   最佳組合 (未完全通過):")
            print(f"   RSI: {p['rsi_period']}, {p['oversold']}/{p['overbought']}")
            print(f"   報酬: {best['return']:+.2f}%, 一致性: {best['consistency']:.0f}%")
        return None


if __name__ == "__main__":
    asyncio.run(main())
