#!/usr/bin/env python3
"""
Bollinger Bot Parameter Optimization.

嘗試找出能通過 Walk-Forward 驗證的參數組合。
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional
from itertools import product

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


class StrategyMode(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


@dataclass
class BollingerConfig:
    bb_period: int = 20
    bb_std: float = 2.0
    leverage: int = 5
    position_pct: float = 0.1
    fee_rate: float = 0.0004
    stop_loss_pct: float = 0.02
    max_hold_bars: int = 48
    strategy_mode: StrategyMode = StrategyMode.BREAKOUT
    use_atr_stop: bool = True
    atr_period: int = 14
    atr_multiplier: float = 2.0
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 2.0
    use_trend_filter: bool = False
    trend_period: int = 50


class BollingerBacktest:
    def __init__(self, klines: list[Kline], config: BollingerConfig):
        self._klines = klines
        self._config = config
        self._position = None
        self._trades = []

    def _calculate_sma(self, prices: list[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def _calculate_std(self, prices: list[float], period: int, sma: float) -> Optional[float]:
        if len(prices) < period:
            return None
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
        return math.sqrt(variance)

    def _calculate_atr(self, idx: int, period: int) -> Optional[float]:
        if idx < period + 1:
            return None
        tr_values = []
        for i in range(idx - period, idx):
            high = float(self._klines[i].high)
            low = float(self._klines[i].low)
            prev_close = float(self._klines[i-1].close)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        return sum(tr_values) / len(tr_values)

    def run(self) -> dict:
        min_bars = max(self._config.bb_period, self._config.atr_period,
                       self._config.trend_period if self._config.use_trend_filter else 0) + 20

        if len(self._klines) < min_bars:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "trades": 0, "win_rate": 0}

        capital = 10000.0
        peak = capital
        max_dd = 0.0
        daily_returns = []
        prev_equity = capital
        closes = []

        self._position = None
        self._trades = []

        for idx, kline in enumerate(self._klines):
            close = float(kline.close)
            high = float(kline.high)
            low = float(kline.low)
            closes.append(close)

            if idx < min_bars:
                continue

            # Calculate indicators
            sma = self._calculate_sma(closes, self._config.bb_period)
            std = self._calculate_std(closes, self._config.bb_period, sma)
            if sma is None or std is None or std == 0:
                continue

            upper_band = sma + std * self._config.bb_std
            lower_band = sma - std * self._config.bb_std
            atr = self._calculate_atr(idx, self._config.atr_period)

            # Trend filter
            trend = 0
            if self._config.use_trend_filter:
                trend_sma = self._calculate_sma(closes, self._config.trend_period)
                if trend_sma:
                    if close > trend_sma * 1.005:
                        trend = 1
                    elif close < trend_sma * 0.995:
                        trend = -1

            # Position management
            if self._position:
                entry = self._position['entry']
                side = self._position['side']
                bars_held = self._position['bars_held']

                # Update trailing stop
                if self._config.use_trailing_stop and atr:
                    if side == 'long':
                        new_trail = close - atr * self._config.trailing_atr_mult
                        current_trail = self._position.get('trailing_stop')
                        if current_trail is None or new_trail > current_trail:
                            self._position['trailing_stop'] = new_trail
                    else:
                        new_trail = close + atr * self._config.trailing_atr_mult
                        current_trail = self._position.get('trailing_stop')
                        if current_trail is None or new_trail < current_trail:
                            self._position['trailing_stop'] = new_trail

                exit_reason = None
                exit_price = close

                if side == 'long':
                    pnl_pct = (close - entry) / entry

                    # Stop loss
                    if self._config.use_atr_stop and atr:
                        stop_price = entry - atr * self._config.atr_multiplier
                        if low <= stop_price:
                            exit_reason = 'sl'
                            exit_price = stop_price
                    elif pnl_pct <= -self._config.stop_loss_pct:
                        exit_reason = 'sl'

                    # Trailing stop
                    if not exit_reason and self._config.use_trailing_stop:
                        trail = self._position.get('trailing_stop')
                        if trail and low <= trail:
                            exit_reason = 'trail'
                            exit_price = trail

                    # Take profit
                    if not exit_reason:
                        if self._config.strategy_mode == StrategyMode.MEAN_REVERSION:
                            if close >= sma:
                                exit_reason = 'tp'
                        else:
                            if close >= upper_band or bars_held >= self._config.max_hold_bars:
                                exit_reason = 'tp'

                else:  # short
                    pnl_pct = (entry - close) / entry

                    if self._config.use_atr_stop and atr:
                        stop_price = entry + atr * self._config.atr_multiplier
                        if high >= stop_price:
                            exit_reason = 'sl'
                            exit_price = stop_price
                    elif pnl_pct <= -self._config.stop_loss_pct:
                        exit_reason = 'sl'

                    if not exit_reason and self._config.use_trailing_stop:
                        trail = self._position.get('trailing_stop')
                        if trail and high >= trail:
                            exit_reason = 'trail'
                            exit_price = trail

                    if not exit_reason:
                        if self._config.strategy_mode == StrategyMode.MEAN_REVERSION:
                            if close <= sma:
                                exit_reason = 'tp'
                        else:
                            if close <= lower_band or bars_held >= self._config.max_hold_bars:
                                exit_reason = 'tp'

                if exit_reason:
                    if side == 'long':
                        pnl = (exit_price - entry) / entry
                    else:
                        pnl = (entry - exit_price) / entry

                    net = pnl * self._config.leverage * self._config.position_pct
                    net -= self._config.fee_rate * 2 * self._config.leverage * self._config.position_pct
                    self._trades.append(net)
                    capital += capital * net
                    self._position = None
                else:
                    self._position['bars_held'] += 1

            # Entry
            if not self._position:
                if self._config.strategy_mode == StrategyMode.BREAKOUT:
                    if high >= upper_band:
                        if not self._config.use_trend_filter or trend >= 0:
                            self._position = {'entry': upper_band, 'side': 'long', 'bars_held': 0, 'trailing_stop': None}
                    elif low <= lower_band:
                        if not self._config.use_trend_filter or trend <= 0:
                            self._position = {'entry': lower_band, 'side': 'short', 'bars_held': 0, 'trailing_stop': None}
                else:
                    if low <= lower_band:
                        if not self._config.use_trend_filter or trend >= 0:
                            self._position = {'entry': lower_band, 'side': 'long', 'bars_held': 0, 'trailing_stop': None}
                    elif high >= upper_band:
                        if not self._config.use_trend_filter or trend <= 0:
                            self._position = {'entry': upper_band, 'side': 'short', 'bars_held': 0, 'trailing_stop': None}

            # Track drawdown
            equity = capital
            if self._position:
                entry = self._position['entry']
                if self._position['side'] == 'long':
                    unrealized = (close - entry) / entry * self._config.leverage * self._config.position_pct
                else:
                    unrealized = (entry - close) / entry * self._config.leverage * self._config.position_pct
                equity += capital * unrealized

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

            if idx % 96 == 0 and prev_equity > 0:
                daily_returns.append((equity - prev_equity) / prev_equity)
                prev_equity = equity

        # Close remaining position
        if self._position:
            close = float(self._klines[-1].close)
            entry = self._position['entry']
            if self._position['side'] == 'long':
                pnl = (close - entry) / entry
            else:
                pnl = (entry - close) / entry
            net = pnl * self._config.leverage * self._config.position_pct
            net -= self._config.fee_rate * 2 * self._config.leverage * self._config.position_pct
            self._trades.append(net)
            capital += capital * net

        total_return = (capital - 10000) / 10000 * 100
        total_trades = len(self._trades)
        win_rate = sum(1 for t in self._trades if t > 0) / total_trades * 100 if total_trades else 0

        sharpe = 0
        if len(daily_returns) > 10:
            avg = sum(daily_returns) / len(daily_returns)
            var = sum((r - avg) ** 2 for r in daily_returns) / len(daily_returns)
            std = math.sqrt(var) if var > 0 else 0.001
            sharpe = (avg / std) * math.sqrt(365) if std > 0 else 0

        return {
            "return": total_return,
            "sharpe": sharpe,
            "max_dd": max_dd * 100,
            "trades": total_trades,
            "win_rate": win_rate,
        }


def walk_forward(klines: list[Kline], config: BollingerConfig, periods: int = 6) -> dict:
    n = len(klines) // periods
    results = []
    period_results = []

    for i in range(periods):
        start = i * n
        end = (i + 1) * n if i < periods - 1 else len(klines)
        bt = BollingerBacktest(klines[start:end], config)
        r = bt.run()
        results.append(r["return"] > 0)
        period_results.append(r)

    return {
        "consistency": sum(results) / len(results) * 100,
        "profitable": sum(results),
        "total": len(results),
        "periods": period_results,
    }


async def fetch_klines(days: int, timeframe: str = "15m") -> list[Kline]:
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
    print("       Bollinger Bot 參數優化")
    print("=" * 70)

    print("\n獲取 1 年數據...")
    klines = await fetch_klines(365, "15m")
    print(f"  獲取 {len(klines)} 根 K 線")

    # Parameter grid
    param_grid = {
        'bb_period': [15, 20, 30],
        'bb_std': [1.5, 2.0, 2.5, 3.0],
        'leverage': [2, 3, 5],
        'strategy_mode': [StrategyMode.MEAN_REVERSION, StrategyMode.BREAKOUT],
        'use_atr_stop': [True, False],
        'atr_multiplier': [1.5, 2.0, 3.0],
        'use_trailing_stop': [True, False],
        'use_trend_filter': [True, False],
        'max_hold_bars': [24, 48, 96],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    total = len(combinations)

    print(f"\n測試 {total} 個參數組合...")
    print("-" * 70)

    results = []
    passing = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Skip invalid combinations
        if not params['use_atr_stop'] and params['atr_multiplier'] != 2.0:
            continue

        config = BollingerConfig(
            bb_period=params['bb_period'],
            bb_std=params['bb_std'],
            leverage=params['leverage'],
            strategy_mode=params['strategy_mode'],
            use_atr_stop=params['use_atr_stop'],
            atr_multiplier=params['atr_multiplier'],
            use_trailing_stop=params['use_trailing_stop'],
            use_trend_filter=params['use_trend_filter'],
            max_hold_bars=params['max_hold_bars'],
            stop_loss_pct=0.03 if not params['use_atr_stop'] else 0.02,
            trend_period=50,
        )

        bt = BollingerBacktest(klines, config)
        full = bt.run()

        # Only do walk-forward if full backtest is profitable
        if full["return"] > 0:
            wf = walk_forward(klines, config, periods=6)

            if wf["consistency"] >= 50:  # Lower threshold first to find candidates
                mode = "MR" if params['strategy_mode'] == StrategyMode.MEAN_REVERSION else "BO"
                trend = "T" if params['use_trend_filter'] else ""
                trail = "Tr" if params['use_trailing_stop'] else ""
                atr = "ATR" if params['use_atr_stop'] else ""

                name = f"BB({params['bb_period']},{params['bb_std']}) {params['leverage']}x {mode} {atr}{trail}{trend}"

                result = {
                    "name": name,
                    "config": config,
                    "params": params,
                    "return": full["return"],
                    "sharpe": full["sharpe"],
                    "max_dd": full["max_dd"],
                    "trades": full["trades"],
                    "win_rate": full["win_rate"],
                    "consistency": wf["consistency"],
                    "periods": wf["periods"],
                }
                results.append(result)

                if wf["consistency"] >= 67:
                    passing.append(result)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  進度: {i+1}/{total} ({len(results)} 候選, {len(passing)} 通過)")

    print(f"\n完成: {total} 組合, {len(results)} 候選, {len(passing)} 通過驗證")

    # Sort by consistency then return
    results.sort(key=lambda x: (x["consistency"], x["return"]), reverse=True)

    print("\n" + "=" * 70)
    print("       最佳結果 (按一致性排序)")
    print("=" * 70)

    for i, r in enumerate(results[:10]):
        status = "✅" if r["consistency"] >= 67 else "⚠️" if r["consistency"] >= 50 else "❌"
        print(f"\n{i+1}. {status} {r['name']}")
        print(f"   報酬: {r['return']:+.1f}% | Sharpe: {r['sharpe']:.2f} | 回撤: {r['max_dd']:.1f}%")
        print(f"   Walk-Forward: {r['consistency']:.0f}% | 交易: {r['trades']} | 勝率: {r['win_rate']:.1f}%")
        period_str = " | ".join([f"P{j+1}:{p['return']:+.0f}%" for j, p in enumerate(r['periods'])])
        print(f"   各時段: {period_str}")

    if passing:
        print("\n" + "=" * 70)
        print("       通過驗證的策略 (一致性 ≥67%)")
        print("=" * 70)

        best = passing[0]
        p = best['params']
        print(f"""
✅ 最佳策略: {best['name']}

參數:
  BB_PERIOD={p['bb_period']}
  BB_STD={p['bb_std']}
  LEVERAGE={p['leverage']}
  STRATEGY={p['strategy_mode'].value}
  USE_ATR_STOP={p['use_atr_stop']}
  ATR_MULTIPLIER={p['atr_multiplier']}
  USE_TRAILING_STOP={p['use_trailing_stop']}
  USE_TREND_FILTER={p['use_trend_filter']}
  MAX_HOLD_BARS={p['max_hold_bars']}

預期績效:
  報酬: {best['return']:+.1f}%
  Sharpe: {best['sharpe']:.2f}
  最大回撤: {best['max_dd']:.1f}%
  Walk-Forward 一致性: {best['consistency']:.0f}%
""")
    else:
        print("\n" + "=" * 70)
        print("       結論")
        print("=" * 70)
        print("\n❌ 沒有參數組合通過 Walk-Forward 驗證 (一致性 ≥67%)")

        if results:
            best = results[0]
            print(f"\n   最接近的策略: {best['name']}")
            print(f"   報酬: {best['return']:+.1f}%, 一致性: {best['consistency']:.0f}%")
            print("\n   建議: Bollinger 策略在當前市場條件下不可靠，建議停用")


if __name__ == "__main__":
    asyncio.run(main())
