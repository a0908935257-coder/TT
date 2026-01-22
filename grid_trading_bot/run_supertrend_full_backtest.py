#!/usr/bin/env python3
"""
Supertrend Bot 完整回測 (2 年數據).

使用足夠長的時間和足夠多的交易來驗證策略，包含追蹤止損。
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

from src.core.models import Kline
from src.exchange import ExchangeClient


@dataclass
class Trade:
    """交易記錄"""
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResult:
    """回測結果"""
    total_pnl: float
    total_return_pct: float
    win_rate: float
    total_trades: int
    long_trades: int
    short_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_hold_hours: float
    trades: List[Trade]
    equity_curve: List[float]


class SupertrendBacktester:
    """Supertrend 回測器"""

    def __init__(self, klines: List[Kline], config: Dict[str, Any]):
        self.klines = klines
        self.config = config
        self.leverage = config.get('leverage', 10)
        self.position_size = config.get('position_size', 0.1)
        self.fee_rate = 0.0004  # 0.04% taker fee

    def run(self) -> BacktestResult:
        """執行回測"""
        c = self.config
        atr_period = c.get('atr_period', 10)
        atr_multiplier = c.get('atr_multiplier', 3.0)
        use_trailing_stop = c.get('use_trailing_stop', True)
        trailing_stop_pct = c.get('trailing_stop_pct', 0.03)

        # State
        position = None
        trades = []
        initial_capital = 10000.0
        equity = [initial_capital]

        closes = [float(k.close) for k in self.klines]
        highs = [float(k.high) for k in self.klines]
        lows = [float(k.low) for k in self.klines]

        warmup = atr_period + 20

        # Supertrend state
        prev_trend = 0  # 1 = bullish, -1 = bearish
        upper_band = None
        lower_band = None

        for i in range(warmup, len(self.klines)):
            price = closes[i]
            high = highs[i]
            low = lows[i]
            kline = self.klines[i]

            # 計算 ATR
            trs = []
            for j in range(i - atr_period, i):
                tr = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1])
                )
                trs.append(tr)
            atr = sum(trs) / atr_period

            # 計算基本上下軌
            hl2 = (high + low) / 2
            basic_upper = hl2 + atr_multiplier * atr
            basic_lower = hl2 - atr_multiplier * atr

            # 更新最終上下軌
            if upper_band is None:
                upper_band = basic_upper
                lower_band = basic_lower
            else:
                prev_close = closes[i - 1]
                # 上軌只能下降或維持
                if basic_upper < upper_band or prev_close > upper_band:
                    upper_band = basic_upper
                # 下軌只能上升或維持
                if basic_lower > lower_band or prev_close < lower_band:
                    lower_band = basic_lower

            # 判斷趨勢
            if price > upper_band:
                current_trend = 1  # Bullish
            elif price < lower_band:
                current_trend = -1  # Bearish
            else:
                current_trend = prev_trend

            # 檢查持倉出場
            if position is not None:
                bars_held = i - position['entry_bar']
                entry_price = position['entry']

                # 更新追蹤價格
                if position['side'] == 'long':
                    position['max_price'] = max(position.get('max_price', entry_price), price)
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    position['min_price'] = min(position.get('min_price', entry_price), price)
                    pnl_pct = (entry_price - price) / entry_price

                exit_reason = None

                # 1. 趨勢反轉出場 (Supertrend 主要信號)
                if current_trend != prev_trend and prev_trend != 0:
                    if position['side'] == 'long' and current_trend == -1:
                        exit_reason = 'trend_reversal'
                    elif position['side'] == 'short' and current_trend == 1:
                        exit_reason = 'trend_reversal'

                # 2. 追蹤止損
                if exit_reason is None and use_trailing_stop:
                    if position['side'] == 'long':
                        max_price = position['max_price']
                        drawdown = (max_price - price) / max_price
                        if drawdown >= trailing_stop_pct and pnl_pct > 0:
                            exit_reason = 'trailing_stop'
                    else:
                        min_price = position['min_price']
                        drawup = (price - min_price) / min_price
                        if drawup >= trailing_stop_pct and pnl_pct > 0:
                            exit_reason = 'trailing_stop'

                if exit_reason:
                    position_value = equity[-1] * self.position_size
                    gross_pnl = pnl_pct * self.leverage * position_value
                    fee = position_value * self.fee_rate * 2
                    net_pnl = gross_pnl - fee

                    trades.append(Trade(
                        entry_time=position['entry_time'],
                        exit_time=kline.close_time,
                        side=position['side'],
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        bars_held=bars_held,
                        exit_reason=exit_reason,
                    ))

                    equity.append(equity[-1] + net_pnl)
                    position = None

            # 趨勢翻轉開倉 (只在沒有持倉且趨勢改變時)
            if position is None and current_trend != prev_trend and prev_trend != 0:
                if current_trend == 1:
                    position = {
                        'side': 'long',
                        'entry': price,
                        'entry_bar': i,
                        'entry_time': kline.close_time,
                        'max_price': price,
                        'min_price': price,
                    }
                elif current_trend == -1:
                    position = {
                        'side': 'short',
                        'entry': price,
                        'entry_bar': i,
                        'entry_time': kline.close_time,
                        'max_price': price,
                        'min_price': price,
                    }

            prev_trend = current_trend

        # Close remaining position
        if position is not None:
            price = closes[-1]
            entry_price = position['entry']
            if position['side'] == 'long':
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price

            position_value = equity[-1] * self.position_size
            gross_pnl = pnl_pct * self.leverage * position_value
            fee = position_value * self.fee_rate * 2
            net_pnl = gross_pnl - fee

            trades.append(Trade(
                entry_time=position['entry_time'],
                exit_time=self.klines[-1].close_time,
                side=position['side'],
                entry_price=entry_price,
                exit_price=price,
                pnl=net_pnl,
                pnl_pct=pnl_pct * 100,
                bars_held=len(self.klines) - position['entry_bar'],
                exit_reason='end',
            ))
            equity.append(equity[-1] + net_pnl)

        # Calculate metrics
        if not trades:
            return BacktestResult(
                total_pnl=0, total_return_pct=0, win_rate=0, total_trades=0,
                long_trades=0, short_trades=0, avg_win=0, avg_loss=0,
                profit_factor=0, sharpe_ratio=0, max_drawdown_pct=0,
                avg_hold_hours=0, trades=[], equity_curve=equity
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        longs = [t for t in trades if t.side == 'long']
        shorts = [t for t in trades if t.side == 'short']

        total_pnl = equity[-1] - initial_capital
        total_return_pct = (equity[-1] / initial_capital - 1) * 100
        win_rate = len(wins) / len(trades) * 100

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe (annualized)
        returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity))]
        if returns:
            avg_ret = sum(returns) / len(returns)
            std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5
            # 15 min bars, 4 per hour, 96 per day, ~35040 per year
            sharpe = (avg_ret / std_ret * (35040 ** 0.5)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

        avg_hold_hours = sum(t.bars_held for t in trades) / len(trades) * 0.25

        return BacktestResult(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            total_trades=len(trades),
            long_trades=len(longs),
            short_trades=len(shorts),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            avg_hold_hours=avg_hold_hours,
            trades=trades,
            equity_curve=equity,
        )


async def fetch_data(days: int = 730) -> List[Kline]:
    """獲取歷史數據"""
    client = ExchangeClient()
    await client.connect()

    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        klines = []
        current_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        batch_count = 0
        while current_ts < end_ts:
            batch = await client.spot.get_klines(
                symbol="BTCUSDT",
                interval="15m",
                start_time=current_ts,
                limit=1500,
            )
            if not batch:
                break
            klines.extend(batch)
            current_ts = int(batch[-1].close_time.timestamp() * 1000) + 1
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  已獲取 {len(klines)} 根 K 線...")

        return klines
    finally:
        await client.close()


async def main():
    print("=" * 70)
    print("       Supertrend Bot 完整回測 (2 年數據)")
    print("=" * 70)

    # 獲取數據
    print("\n正在獲取 BTCUSDT 15m 歷史數據 (2 年)...")
    klines = await fetch_data(days=730)

    print(f"\n數據摘要:")
    print(f"  K 線數量: {len(klines):,}")
    print(f"  時間範圍: {klines[0].open_time.date()} ~ {klines[-1].close_time.date()}")
    closes = [float(k.close) for k in klines]
    print(f"  價格範圍: ${min(closes):,.0f} ~ ${max(closes):,.0f}")
    print(f"  價格變化: {(closes[-1]/closes[0]-1)*100:+.1f}%")

    # 測試配置 (使用驗證後的 5x 槓桿)
    base_config = {
        'leverage': 5,  # Out-of-sample validated
        'position_size': 0.1,
    }

    configs = {
        "🌟 當前配置 (ATR=5, M=2.5) 樣本外驗證": {
            **base_config,
            'atr_period': 5,  # Out-of-sample validated
            'atr_multiplier': 2.5,  # Out-of-sample validated
            'use_trailing_stop': False,
        },
        "當前配置 有追蹤止損": {
            **base_config,
            'atr_period': 5,
            'atr_multiplier': 2.5,
            'use_trailing_stop': True,
            'trailing_stop_pct': 0.03,
        },
        "備選 ATR=18, M=3.5 (樣本外 +16.8%)": {
            **base_config,
            'atr_period': 18,
            'atr_multiplier': 3.5,
            'use_trailing_stop': False,
        },
        "備選 ATR=12, M=3.5 (樣本外 +16.1%)": {
            **base_config,
            'atr_period': 12,
            'atr_multiplier': 3.5,
            'use_trailing_stop': False,
        },
        "舊配置 ATR=10, M=3.0 (樣本外失敗)": {
            **base_config,
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'use_trailing_stop': False,
        },
        "舊配置 ATR=20, M=3.5 (樣本外失敗)": {
            **base_config,
            'atr_period': 20,
            'atr_multiplier': 3.5,
            'use_trailing_stop': False,
        },
    }

    print("\n" + "=" * 70)
    print("  回測結果比較")
    print("=" * 70)

    results = {}
    for name, config in configs.items():
        print(f"\n正在測試: {name}...")
        bt = SupertrendBacktester(klines, config)
        result = bt.run()
        results[name] = result

    # 顯示結果
    print("\n" + "=" * 110)
    print(f"{'策略':<30} {'總報酬%':>10} {'交易數':>8} {'勝率':>8} {'獲利因子':>10} {'Sharpe':>8} {'最大回撤':>10} {'平均持倉':>8}")
    print("-" * 110)

    for name, r in results.items():
        print(f"{name:<30} {r.total_return_pct:>+9.1f}% {r.total_trades:>8} {r.win_rate:>7.1f}% {r.profit_factor:>10.2f} {r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>9.1f}% {r.avg_hold_hours:>7.1f}h")

    print("=" * 110)

    # 最佳策略詳情
    best_name = max(results, key=lambda x: results[x].sharpe_ratio if results[x].total_trades > 50 else -999)
    best = results[best_name]

    print(f"\n最佳策略: {best_name}")
    print("-" * 50)
    print(f"  初始資金: $10,000")
    print(f"  最終資金: ${10000 + best.total_pnl:,.2f}")
    print(f"  總報酬: {best.total_return_pct:+.1f}%")
    print(f"  年化報酬: {best.total_return_pct / 2:+.1f}% (假設 2 年)")
    print(f"  總交易: {best.total_trades} 筆 (多: {best.long_trades}, 空: {best.short_trades})")
    print(f"  勝率: {best.win_rate:.1f}%")
    print(f"  平均獲利: ${best.avg_win:.2f}")
    print(f"  平均虧損: ${best.avg_loss:.2f}")
    print(f"  獲利因子: {best.profit_factor:.2f}")
    print(f"  Sharpe Ratio: {best.sharpe_ratio:.2f}")
    print(f"  最大回撤: {best.max_drawdown_pct:.1f}%")

    # 出場原因統計
    if best.trades:
        exit_reasons = {}
        for t in best.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        print(f"\n  出場原因統計:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count} ({count/len(best.trades)*100:.1f}%)")

    # 追蹤止損效果比較
    print("\n" + "=" * 70)
    print("  追蹤止損效果比較")
    print("=" * 70)

    # Compare current validated config with and without trailing stop
    with_ts = "當前配置 有追蹤止損"
    without_ts = "🌟 當前配置 (ATR=5, M=2.5) 樣本外驗證"

    if with_ts in results and without_ts in results:
        r_with = results[with_ts]
        r_without = results[without_ts]
        print(f"\n  當前配置 (ATR=5, M=2.5):")
        print(f"    {'指標':<15} {'有追蹤':>12} {'無追蹤':>12} {'差異':>12}")
        print(f"    {'-'*50}")
        print(f"    {'總報酬':<15} {r_with.total_return_pct:>+11.1f}% {r_without.total_return_pct:>+11.1f}% {r_with.total_return_pct - r_without.total_return_pct:>+11.1f}%")
        print(f"    {'最大回撤':<15} {r_with.max_drawdown_pct:>11.1f}% {r_without.max_drawdown_pct:>11.1f}% {r_with.max_drawdown_pct - r_without.max_drawdown_pct:>+11.1f}%")
        print(f"    {'Sharpe':<15} {r_with.sharpe_ratio:>12.2f} {r_without.sharpe_ratio:>12.2f} {r_with.sharpe_ratio - r_without.sharpe_ratio:>+12.2f}")

    print("\n" + "=" * 70)
    print("  回測完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
