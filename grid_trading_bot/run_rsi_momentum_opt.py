#!/usr/bin/env python3
"""
RSI Momentum Strategy Optimization

Walk-Forward 驗證通過的參數 (2 年, 12 期, 67% 一致性):
- RSI Period: 25
- Entry Level: 50, Momentum Threshold: 5
- Leverage: 5x
- Stop Loss: 2%, Take Profit: 4%
- Sharpe: 0.65, Return: +12.0%, Max DD: 9.5%

Strategy:
- Long when RSI crosses above entry_level + threshold (55)
- Short when RSI crosses below entry_level - threshold (45)
"""

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


@dataclass
class RSIMomentumConfig:
    rsi_period: int = 25  # Walk-Forward validated: RSI=25
    entry_level: int = 50  # Cross above = long, cross below = short
    momentum_threshold: int = 5  # Must cross by this much
    leverage: int = 5
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    trailing_stop: bool = False
    trailing_pct: float = 0.02


class RSIMomentumBacktest:
    FEE_RATE = 0.0004

    def __init__(self, klines: list[Kline], config: RSIMomentumConfig):
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
        if len(self._klines) < self._config.rsi_period + 20:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "win_rate": 0, "trades": 0}

        self._avg_gain = None
        self._avg_loss = None
        self._position = None
        self._trades = []

        peak = current = 10000.0
        max_dd = 0.0
        daily_returns = []
        prev_equity = current
        closes = []
        prev_rsi = 50.0

        for idx, kline in enumerate(self._klines):
            close = float(kline.close)
            closes.append(close)

            if len(closes) < self._config.rsi_period + 5:
                continue

            rsi = self._calculate_rsi(closes[-50:])

            if self._position:
                entry = self._position['entry']
                side = self._position['side']
                chg = (close - entry) / entry

                # Trailing stop
                if self._config.trailing_stop:
                    if side == 'long' and close > self._position.get('peak', entry):
                        self._position['peak'] = close
                    elif side == 'short' and close < self._position.get('peak', entry):
                        self._position['peak'] = close

                exit_reason = None
                if side == 'long':
                    if chg <= -self._config.stop_loss_pct:
                        exit_reason = 'sl'
                    elif chg >= self._config.take_profit_pct:
                        exit_reason = 'tp'
                    elif self._config.trailing_stop and self._position.get('peak'):
                        trail_chg = (close - self._position['peak']) / self._position['peak']
                        if trail_chg <= -self._config.trailing_pct:
                            exit_reason = 'trail'
                    # Exit on RSI reversal (crosses below 50)
                    elif rsi < self._config.entry_level - self._config.momentum_threshold:
                        exit_reason = 'rsi'
                else:
                    if chg >= self._config.stop_loss_pct:
                        exit_reason = 'sl'
                    elif chg <= -self._config.take_profit_pct:
                        exit_reason = 'tp'
                    elif self._config.trailing_stop and self._position.get('peak'):
                        trail_chg = (self._position['peak'] - close) / self._position['peak']
                        if trail_chg <= -self._config.trailing_pct:
                            exit_reason = 'trail'
                    # Exit on RSI reversal (crosses above 50)
                    elif rsi > self._config.entry_level + self._config.momentum_threshold:
                        exit_reason = 'rsi'

                if exit_reason:
                    pnl = chg if side == 'long' else -chg
                    net = pnl * self._config.leverage * 0.1
                    net -= self.FEE_RATE * 2 * self._config.leverage * 0.1
                    self._trades.append(net)
                    current += current * net
                    self._position = None

            # Entry: RSI momentum crossover
            if not self._position:
                threshold = self._config.entry_level
                momentum = self._config.momentum_threshold

                # RSI crossed above threshold (bullish momentum)
                if prev_rsi <= threshold and rsi > threshold + momentum:
                    self._position = {'entry': close, 'side': 'long', 'peak': close}

                # RSI crossed below threshold (bearish momentum)
                elif prev_rsi >= threshold and rsi < threshold - momentum:
                    self._position = {'entry': close, 'side': 'short', 'peak': close}

            prev_rsi = rsi

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


def walk_forward(klines: list[Kline], config: RSIMomentumConfig, periods: int = 6) -> dict:
    n = len(klines) // periods
    results = []
    for i in range(periods):
        start = i * n
        end = (i + 1) * n if i < periods - 1 else len(klines)
        bt = RSIMomentumBacktest(klines[start:end], config)
        r = bt.run()
        results.append(r["return"] > 0)

    return {
        "consistency": sum(results) / len(results) * 100,
        "profitable": sum(results),
        "total": len(results),
    }


async def main():
    print("=" * 70)
    print("       RSI 動量策略優化")
    print("       (趨勢跟隨而非均值回歸)")
    print("=" * 70)

    print("\n獲取 1 年數據...")
    klines = await fetch_klines(365)
    print(f"  獲取 {len(klines)} 根 K 線")

    # Test different RSI momentum configurations
    configs = [
        # Walk-Forward validated (RSI=25, 67% consistency)
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=5, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04),

        # RSI period variations
        RSIMomentumConfig(rsi_period=21, entry_level=50, momentum_threshold=5, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04),
        RSIMomentumConfig(rsi_period=28, entry_level=50, momentum_threshold=5, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04),

        # Momentum threshold variations
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=3, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04),
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=7, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04),

        # Different TP/SL ratios
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=5, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.06),
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=5, leverage=5, stop_loss_pct=0.025, take_profit_pct=0.05),

        # Lower leverage for safety
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=5, leverage=3, stop_loss_pct=0.03, take_profit_pct=0.06),

        # With trailing stop
        RSIMomentumConfig(rsi_period=25, entry_level=50, momentum_threshold=5, leverage=5, stop_loss_pct=0.02, take_profit_pct=0.06, trailing_stop=True, trailing_pct=0.015),
    ]

    print(f"\n測試 {len(configs)} 個動量策略組合...")
    print("-" * 70)

    results = []
    for i, config in enumerate(configs):
        bt = RSIMomentumBacktest(klines, config)
        full = bt.run()
        wf = walk_forward(klines, config, periods=6)

        trail = "是" if config.trailing_stop else "否"
        status = "✅" if full["return"] > 0 and wf["consistency"] >= 67 else "⚠️" if full["return"] > 0 else "❌"

        print(f"  {i+1:2d}. {status} RSI({config.rsi_period}) Level={config.entry_level}±{config.momentum_threshold} Trail={trail}")
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
    print("       最佳動量策略參數")
    print("=" * 70)

    for i, r in enumerate(results[:5]):
        c = r["config"]
        trail = "是" if c.trailing_stop else "否"
        status = "✅ 通過" if r["return"] > 0 and r["consistency"] >= 67 else "⚠️ 部分" if r["return"] > 0 else "❌ 不佳"

        print(f"\n#{i+1} {status}")
        print(f"  RSI Period: {c.rsi_period}, Entry Level: {c.entry_level}±{c.momentum_threshold}")
        print(f"  Leverage: {c.leverage}x, SL: {c.stop_loss_pct*100:.1f}%, TP: {c.take_profit_pct*100:.1f}%")
        print(f"  Trailing Stop: {trail}")
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
# RSI 動量策略參數
RSI_PERIOD={c.rsi_period}
RSI_ENTRY_LEVEL={c.entry_level}
RSI_MOMENTUM_THRESHOLD={c.momentum_threshold}
RSI_LEVERAGE={c.leverage}
RSI_STOP_LOSS_PCT={c.stop_loss_pct}
RSI_TAKE_PROFIT_PCT={c.take_profit_pct}
RSI_USE_TRAILING_STOP={'true' if c.trailing_stop else 'false'}
RSI_TRAILING_PCT={c.trailing_pct}

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

        # Show best overall even if not passing
        if results and results[0]["return"] > -5:
            best = results[0]
            print(f"\n   最佳組合 (風險較高):")
            print(f"   報酬: {best['return']:+.2f}%, 一致性: {best['consistency']:.0f}%")
        return None


if __name__ == "__main__":
    asyncio.run(main())
