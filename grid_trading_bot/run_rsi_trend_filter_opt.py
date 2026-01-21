#!/usr/bin/env python3
"""
RSI Bot with Trend Filter Optimization
Only trade RSI signals in the direction of the trend
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
class RSITrendConfig:
    rsi_period: int = 14
    oversold: int = 30
    overbought: int = 70
    exit_level: int = 50
    leverage: int = 5
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    # Trend filter
    trend_period: int = 50  # EMA period for trend
    use_trend_filter: bool = True


class RSITrendBacktest:
    FEE_RATE = 0.0004

    def __init__(self, klines: list[Kline], config: RSITrendConfig):
        self._klines = klines
        self._config = config
        self._position = None
        self._trades = []
        self._avg_gain = None
        self._avg_loss = None
        self._ema = None

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

    def _calculate_ema(self, close: float) -> float:
        """Calculate EMA for trend."""
        period = self._config.trend_period
        multiplier = 2 / (period + 1)

        if self._ema is None:
            self._ema = close
        else:
            self._ema = (close - self._ema) * multiplier + self._ema

        return self._ema

    def run(self) -> dict:
        if len(self._klines) < max(self._config.rsi_period, self._config.trend_period) + 20:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "trades": 0}

        self._avg_gain = None
        self._avg_loss = None
        self._ema = None
        self._position = None
        self._trades = []

        peak = current = 10000.0
        max_dd = 0.0
        daily_returns = []
        prev_equity = current
        closes = []

        # Warmup EMA
        warmup = min(self._config.trend_period * 2, len(self._klines) // 4)
        for i in range(warmup):
            self._calculate_ema(float(self._klines[i].close))

        for idx, kline in enumerate(self._klines[warmup:], warmup):
            close = float(kline.close)
            closes.append(close)

            if len(closes) < self._config.rsi_period + 5:
                self._calculate_ema(close)
                continue

            rsi = self._calculate_rsi(closes[-50:])
            ema = self._calculate_ema(close)

            # Trend direction
            uptrend = close > ema
            downtrend = close < ema

            if self._position:
                entry = self._position['entry']
                side = self._position['side']
                chg = (close - entry) / entry

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
                    net = pnl * self._config.leverage * 0.1  # 10% position
                    net -= self.FEE_RATE * 2 * self._config.leverage * 0.1
                    self._trades.append(net)
                    current += current * net
                    self._position = None

            # Entry with trend filter
            if not self._position:
                # Long: RSI oversold AND uptrend (buying dip in uptrend)
                if rsi < self._config.oversold:
                    if not self._config.use_trend_filter or uptrend:
                        self._position = {'entry': close, 'side': 'long'}

                # Short: RSI overbought AND downtrend (selling rally in downtrend)
                elif rsi > self._config.overbought:
                    if not self._config.use_trend_filter or downtrend:
                        self._position = {'entry': close, 'side': 'short'}

            if current > peak:
                peak = current
            dd = (peak - current) / peak
            if dd > max_dd:
                max_dd = dd

            if idx % 96 == 0 and prev_equity > 0:
                daily_returns.append((current - prev_equity) / prev_equity)
                prev_equity = current

        total_trades = len(self._trades)
        win_rate = sum(1 for t in self._trades if t > 0) / total_trades * 100 if total_trades else 0
        total_return = (current - 10000) / 10000 * 100

        sharpe = 0
        if len(daily_returns) > 1:
            avg = sum(daily_returns) / len(daily_returns)
            var = sum((r - avg) ** 2 for r in daily_returns) / len(daily_returns)
            std = math.sqrt(var) if var > 0 else 0.001
            sharpe = (avg / std) * math.sqrt(365) if std > 0 else 0

        return {
            "return": total_return,
            "sharpe": sharpe,
            "max_dd": max_dd * 100,
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


def walk_forward(klines: list[Kline], config: RSITrendConfig, periods: int = 6) -> dict:
    n = len(klines) // periods
    results = []
    for i in range(periods):
        start = i * n
        end = (i + 1) * n if i < periods - 1 else len(klines)
        bt = RSITrendBacktest(klines[start:end], config)
        r = bt.run()
        results.append(r["return"] > 0)

    return {
        "consistency": sum(results) / len(results) * 100,
        "profitable": sum(results),
        "total": len(results),
    }


async def main():
    print("=" * 70)
    print("       RSI Bot + Trend Filter 優化")
    print("=" * 70)

    # Fetch 1 year of data
    print("\n獲取 1 年數據...")
    klines = await fetch_klines(365)
    print(f"  獲取 {len(klines)} 根 K 線")

    # Parameter combinations with trend filter
    configs = [
        # With trend filter (only trade with trend)
        RSITrendConfig(rsi_period=14, oversold=30, overbought=70, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=50, use_trend_filter=True),
        RSITrendConfig(rsi_period=14, oversold=35, overbought=65, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=50, use_trend_filter=True),
        RSITrendConfig(rsi_period=21, oversold=30, overbought=70, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=50, use_trend_filter=True),
        RSITrendConfig(rsi_period=14, oversold=30, overbought=70, exit_level=50, leverage=3, stop_loss_pct=0.03, take_profit_pct=0.06, trend_period=50, use_trend_filter=True),
        RSITrendConfig(rsi_period=14, oversold=30, overbought=70, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=100, use_trend_filter=True),
        RSITrendConfig(rsi_period=14, oversold=30, overbought=70, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=20, use_trend_filter=True),
        # More conservative
        RSITrendConfig(rsi_period=14, oversold=25, overbought=75, exit_level=50, leverage=3, stop_loss_pct=0.03, take_profit_pct=0.06, trend_period=50, use_trend_filter=True),
        RSITrendConfig(rsi_period=21, oversold=25, overbought=75, exit_level=50, leverage=3, stop_loss_pct=0.03, take_profit_pct=0.06, trend_period=50, use_trend_filter=True),
        # Without trend filter (baseline)
        RSITrendConfig(rsi_period=14, oversold=30, overbought=70, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=50, use_trend_filter=False),
        RSITrendConfig(rsi_period=21, oversold=30, overbought=70, exit_level=50, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04, trend_period=50, use_trend_filter=False),
    ]

    print(f"\n測試 {len(configs)} 個參數組合 (1 年數據)...")
    print("-" * 70)

    results = []
    for i, config in enumerate(configs):
        bt = RSITrendBacktest(klines, config)
        full = bt.run()
        wf = walk_forward(klines, config, periods=6)

        filter_status = "有" if config.use_trend_filter else "無"
        status = "✅" if full["return"] > 0 and wf["consistency"] >= 67 else "⚠️" if full["return"] > 0 else "❌"

        print(f"  {i+1:2d}. {status} [趨勢過濾: {filter_status}] RSI({config.rsi_period},{config.oversold}/{config.overbought}) EMA({config.trend_period})")
        print(f"      報酬: {full['return']:+6.2f}% | 一致性: {wf['consistency']:.0f}% | 勝率: {full['win_rate']:.0f}% | 交易: {full['trades']}")

        results.append({
            "config": config,
            "return": full["return"],
            "sharpe": full["sharpe"],
            "max_dd": full["max_dd"],
            "win_rate": full["win_rate"],
            "trades": full["trades"],
            "consistency": wf["consistency"],
        })

    # Sort by consistency then return
    results.sort(key=lambda x: (x["consistency"], x["return"]), reverse=True)

    print("\n" + "=" * 70)
    print("       最佳參數 (按一致性排序)")
    print("=" * 70)

    for i, r in enumerate(results[:5]):
        c = r["config"]
        filter_status = "是" if c.use_trend_filter else "否"
        status = "✅ 通過" if r["return"] > 0 and r["consistency"] >= 67 else "⚠️ 部分" if r["return"] > 0 else "❌ 不佳"

        print(f"\n#{i+1} {status}")
        print(f"  趨勢過濾: {filter_status} (EMA {c.trend_period})")
        print(f"  RSI: period={c.rsi_period}, oversold={c.oversold}, overbought={c.overbought}")
        print(f"  Leverage: {c.leverage}x, SL: {c.stop_loss_pct*100:.1f}%, TP: {c.take_profit_pct*100:.1f}%")
        print(f"  報酬: {r['return']:+.2f}% | Sharpe: {r['sharpe']:.2f} | 最大回撤: {r['max_dd']:.1f}%")
        print(f"  勝率: {r['win_rate']:.1f}% | 交易: {r['trades']} | Walk-Forward: {r['consistency']:.0f}%")

    # Best passing
    passing = [r for r in results if r["return"] > 0 and r["consistency"] >= 67]
    if passing:
        best = passing[0]
        c = best["config"]
        print("\n" + "=" * 70)
        print("       推薦參數 (通過過度擬合驗證)")
        print("=" * 70)
        print(f"""
RSI_PERIOD={c.rsi_period}
RSI_OVERSOLD={c.oversold}
RSI_OVERBOUGHT={c.overbought}
RSI_EXIT_LEVEL={c.exit_level}
RSI_LEVERAGE={c.leverage}
RSI_STOP_LOSS_PCT={c.stop_loss_pct}
RSI_TAKE_PROFIT_PCT={c.take_profit_pct}
RSI_USE_TREND_FILTER={'true' if c.use_trend_filter else 'false'}
RSI_TREND_PERIOD={c.trend_period}

預期績效:
  報酬率: {best['return']:+.2f}%
  Sharpe: {best['sharpe']:.2f}
  最大回撤: {best['max_dd']:.1f}%
  Walk-Forward 一致性: {best['consistency']:.0f}%
  勝率: {best['win_rate']:.1f}%
""")
        return best
    else:
        print("\n⚠️ 沒有找到通過過度擬合驗證的參數")
        return None


if __name__ == "__main__":
    asyncio.run(main())
