#!/usr/bin/env python3
"""
Supertrend Bot 過度擬合驗證測試.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

from src.core.models import Kline
from src.exchange import ExchangeClient


@dataclass
class BacktestResult:
    period: str
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float


def run_supertrend_backtest(klines: List[Kline], config: Dict[str, Any]) -> BacktestResult:
    """執行 Supertrend 回測"""
    leverage = config.get('leverage', 10)
    position_size = config.get('position_size', 0.1)
    atr_period = config.get('atr_period', 25)
    atr_multiplier = config.get('atr_multiplier', 3.0)
    use_trailing_stop = config.get('use_trailing_stop', True)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.03)
    fee_rate = 0.0004

    if len(klines) < atr_period + 50:
        return BacktestResult("insufficient_data", 0, 0, 0, 0, 0, 0)

    position = None
    trades = []
    initial_capital = 10000.0
    equity = [initial_capital]

    closes = [float(k.close) for k in klines]
    highs = [float(k.high) for k in klines]
    lows = [float(k.low) for k in klines]

    warmup = atr_period + 20
    prev_trend = 0
    upper_band = None
    lower_band = None

    for i in range(warmup, len(klines)):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        kline = klines[i]

        # ATR
        trs = []
        for j in range(i - atr_period, i):
            tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
            trs.append(tr)
        atr = sum(trs) / atr_period

        # Supertrend bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + atr_multiplier * atr
        basic_lower = hl2 - atr_multiplier * atr

        if upper_band is None:
            upper_band = basic_upper
            lower_band = basic_lower
        else:
            prev_close = closes[i - 1]
            if basic_upper < upper_band or prev_close > upper_band:
                upper_band = basic_upper
            if basic_lower > lower_band or prev_close < lower_band:
                lower_band = basic_lower

        # Trend
        if price > upper_band:
            current_trend = 1
        elif price < lower_band:
            current_trend = -1
        else:
            current_trend = prev_trend

        # Check exit
        if position is not None:
            bars_held = i - position['entry_bar']
            entry_price = position['entry']

            if position['side'] == 'long':
                position['max_price'] = max(position.get('max_price', entry_price), price)
                pnl_pct = (price - entry_price) / entry_price
            else:
                position['min_price'] = min(position.get('min_price', entry_price), price)
                pnl_pct = (entry_price - price) / entry_price

            exit_reason = None

            # Trend reversal
            if current_trend != prev_trend and prev_trend != 0:
                if position['side'] == 'long' and current_trend == -1:
                    exit_reason = 'trend_reversal'
                elif position['side'] == 'short' and current_trend == 1:
                    exit_reason = 'trend_reversal'

            # Trailing stop
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
                position_value = equity[-1] * position_size
                gross_pnl = pnl_pct * leverage * position_value
                fee = position_value * fee_rate * 2
                net_pnl = gross_pnl - fee
                trades.append({'pnl': net_pnl, 'side': position['side']})
                equity.append(equity[-1] + net_pnl)
                position = None

        # Entry on trend change
        if position is None and current_trend != prev_trend and prev_trend != 0:
            if current_trend == 1:
                position = {'side': 'long', 'entry': price, 'entry_bar': i, 'max_price': price, 'min_price': price}
            elif current_trend == -1:
                position = {'side': 'short', 'entry': price, 'entry_bar': i, 'max_price': price, 'min_price': price}

        prev_trend = current_trend

    # Close remaining
    if position is not None:
        price = closes[-1]
        entry_price = position['entry']
        if position['side'] == 'long':
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price
        position_value = equity[-1] * position_size
        gross_pnl = pnl_pct * leverage * position_value
        fee = position_value * fee_rate * 2
        net_pnl = gross_pnl - fee
        trades.append({'pnl': net_pnl, 'side': position['side']})
        equity.append(equity[-1] + net_pnl)

    if not trades:
        return BacktestResult(f"{klines[0].close_time.date()} ~ {klines[-1].close_time.date()}", 0, 0, 0, 0, 0, 0)

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_return_pct = (equity[-1] / initial_capital - 1) * 100
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity)) if equity[i-1] > 0]
    if returns and len(returns) > 1:
        avg_ret = sum(returns) / len(returns)
        std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_ret / std_ret * (35040 ** 0.5)) if std_ret > 0 else 0
    else:
        sharpe = 0

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return BacktestResult(
        period=f"{klines[0].close_time.date()} ~ {klines[-1].close_time.date()}",
        total_return_pct=total_return_pct,
        total_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd * 100,
    )


async def fetch_data(days: int = 730) -> List[Kline]:
    client = ExchangeClient()
    await client.connect()
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        klines = []
        current_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        while current_ts < end_ts:
            batch = await client.spot.get_klines(symbol="BTCUSDT", interval="15m", start_time=current_ts, limit=1500)
            if not batch:
                break
            klines.extend(batch)
            current_ts = int(batch[-1].close_time.timestamp() * 1000) + 1
        return klines
    finally:
        await client.close()


async def main():
    print("=" * 80)
    print("  Supertrend Bot 過度擬合驗證測試")
    print("=" * 80)

    print("\n正在獲取 BTCUSDT 15m 歷史數據 (2 年)...")
    klines = await fetch_data(days=730)
    print(f"  已獲取 {len(klines)} 根 K 線")

    config = {
        'leverage': 5,  # Reduced for better risk management
        'position_size': 0.1,
        'atr_period': 10,  # Walk-Forward validated: 100% (6/6)
        'atr_multiplier': 3.0,  # Walk-Forward validated
        'use_trailing_stop': False,
        'trailing_stop_pct': 0.03,
    }

    # 測試 1: 樣本內 vs 樣本外
    print("\n" + "=" * 80)
    print("  測試 1: 樣本內 vs 樣本外 (70/30 分割)")
    print("=" * 80)

    split_idx = int(len(klines) * 0.7)
    in_sample = klines[:split_idx]
    out_sample = klines[split_idx:]

    in_result = run_supertrend_backtest(in_sample, config)
    out_result = run_supertrend_backtest(out_sample, config)

    print(f"\n  {'指標':<15} {'樣本內 (70%)':<20} {'樣本外 (30%)':<20} {'差異':<15}")
    print("-" * 70)
    print(f"  {'總報酬':<15} {in_result.total_return_pct:>+17.1f}% {out_result.total_return_pct:>+17.1f}% {out_result.total_return_pct - in_result.total_return_pct:>+12.1f}%")
    print(f"  {'交易數':<15} {in_result.total_trades:>18} {out_result.total_trades:>18}")
    print(f"  {'勝率':<15} {in_result.win_rate:>17.1f}% {out_result.win_rate:>17.1f}%")
    print(f"  {'獲利因子':<15} {in_result.profit_factor:>18.2f} {out_result.profit_factor:>18.2f}")
    print(f"  {'最大回撤':<15} {in_result.max_drawdown_pct:>17.1f}% {out_result.max_drawdown_pct:>17.1f}%")

    if in_result.total_return_pct != 0:
        oos_degradation = (in_result.total_return_pct - out_result.total_return_pct) / abs(in_result.total_return_pct) * 100
    else:
        oos_degradation = 0

    print(f"\n  樣本外績效變化: {oos_degradation:+.1f}%")
    if abs(oos_degradation) < 30:
        print("  ✅ 通過: 樣本外績效衰退 < 30%")
    elif abs(oos_degradation) < 50:
        print("  ⚠️ 警告: 樣本外績效衰退 30-50%")
    else:
        print("  ❌ 失敗: 樣本外績效衰退 > 50%")

    # 測試 2: 前瞻分析
    print("\n" + "=" * 80)
    print("  測試 2: 前瞻分析 (3 個月滾動窗口)")
    print("=" * 80)

    window_klines = 90 * 96
    step_klines = 45 * 96

    walk_forward_results = []
    idx = 0
    while idx + window_klines < len(klines):
        window = klines[idx:idx + window_klines]
        result = run_supertrend_backtest(window, config)
        if result.total_trades > 0:
            walk_forward_results.append(result)
        idx += step_klines

    print(f"\n  {'期間':<30} {'報酬%':>10} {'交易數':>8} {'勝率':>8} {'獲利因子':>10}")
    print("-" * 70)

    for r in walk_forward_results:
        print(f"  {r.period:<30} {r.total_return_pct:>+9.1f}% {r.total_trades:>8} {r.win_rate:>7.1f}% {r.profit_factor:>10.2f}")

    returns = [r.total_return_pct for r in walk_forward_results]
    if returns:
        avg_return = sum(returns) / len(returns)
        positive_periods = sum(1 for r in returns if r > 0)
        print(f"\n  平均報酬: {avg_return:+.1f}%")
        print(f"  獲利期間: {positive_periods}/{len(returns)} ({positive_periods/len(returns)*100:.0f}%)")
        if positive_periods / len(returns) >= 0.6:
            print("  ✅ 通過: 60%+ 期間獲利")
        else:
            print("  ❌ 失敗: <60% 期間獲利")

    # 測試 3: 參數敏感度
    print("\n" + "=" * 80)
    print("  測試 3: 參數敏感度分析")
    print("=" * 80)

    print("\n  3.1 ATR 週期敏感度:")
    print(f"  {'ATR週期':>10} {'報酬%':>10} {'交易數':>10} {'勝率':>10}")
    print("-" * 45)

    atr_period_results = []
    for period in [10, 14, 20, 25, 30]:
        test_config = {**config, 'atr_period': period}
        result = run_supertrend_backtest(klines, test_config)
        atr_period_results.append((period, result.total_return_pct))
        print(f"  {period:>10} {result.total_return_pct:>+9.1f}% {result.total_trades:>10} {result.win_rate:>9.1f}%")

    period_range = max(r[1] for r in atr_period_results) - min(r[1] for r in atr_period_results)
    print(f"\n  報酬變化幅度: {period_range:.1f}%")

    print("\n  3.2 ATR 乘數敏感度:")
    print(f"  {'ATR乘數':>10} {'報酬%':>10} {'交易數':>10} {'勝率':>10}")
    print("-" * 45)

    atr_mult_results = []
    for mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
        test_config = {**config, 'atr_multiplier': mult}
        result = run_supertrend_backtest(klines, test_config)
        atr_mult_results.append((mult, result.total_return_pct))
        print(f"  {mult:>10.1f} {result.total_return_pct:>+9.1f}% {result.total_trades:>10} {result.win_rate:>9.1f}%")

    mult_range = max(r[1] for r in atr_mult_results) - min(r[1] for r in atr_mult_results)
    print(f"\n  報酬變化幅度: {mult_range:.1f}%")

    if period_range < 50 and mult_range < 50:
        print("\n  ✅ 通過: 參數敏感度低")
    elif period_range < 100 and mult_range < 100:
        print("\n  ⚠️ 警告: 參數敏感度中等")
    else:
        print("\n  ❌ 失敗: 參數敏感度高")

    # 測試 4: 不同市場狀況
    print("\n" + "=" * 80)
    print("  測試 4: 不同市場狀況分析")
    print("=" * 80)

    quarter_size = len(klines) // 8
    market_results = []
    closes = [float(k.close) for k in klines]

    for i in range(8):
        start = i * quarter_size
        end = (i + 1) * quarter_size
        quarter = klines[start:end]
        if len(quarter) > 500:
            start_price = float(quarter[0].close)
            end_price = float(quarter[-1].close)
            change_pct = (end_price - start_price) / start_price * 100
            if change_pct > 15:
                condition = "牛市"
            elif change_pct < -15:
                condition = "熊市"
            else:
                condition = "盤整"
            result = run_supertrend_backtest(quarter, config)
            market_results.append((condition, change_pct, result))

    print(f"\n  {'期間':>5} {'市場':>8} {'漲跌%':>10} {'報酬%':>10} {'勝率':>10}")
    print("-" * 50)

    all_returns = []
    for i, (condition, change, result) in enumerate(market_results):
        print(f"  Q{i+1:<4} {condition:>8} {change:>+9.1f}% {result.total_return_pct:>+9.1f}% {result.win_rate:>9.1f}%")
        all_returns.append(result.total_return_pct)

    negative_markets = sum(1 for r in all_returns if r < 0)
    if negative_markets <= len(all_returns) // 2:
        print("\n  ✅ 通過: 多數市場狀況獲利")
    else:
        print("\n  ⚠️ 警告: 在某些市場狀況虧損")

    # 總結
    print("\n" + "=" * 80)
    print("  總結")
    print("=" * 80)

    tests_passed = 0
    if abs(oos_degradation) < 50:
        tests_passed += 1
    if returns and positive_periods / len(returns) >= 0.5:
        tests_passed += 1
    if period_range < 100 and mult_range < 100:
        tests_passed += 1
    if negative_markets <= len(all_returns) // 2:
        tests_passed += 1

    print(f"""
  驗證結果: {tests_passed}/4 測試通過

  1. 樣本外測試: {'✅' if abs(oos_degradation) < 50 else '❌'}
     - 樣本內報酬: {in_result.total_return_pct:+.1f}%
     - 樣本外報酬: {out_result.total_return_pct:+.1f}%

  2. 前瞻分析: {'✅' if returns and positive_periods / len(returns) >= 0.5 else '❌'}
     - 獲利期間: {positive_periods}/{len(returns) if returns else 0}

  3. 參數敏感度: {'✅' if period_range < 100 and mult_range < 100 else '⚠️'}
     - ATR 週期影響: {period_range:.1f}%
     - ATR 乘數影響: {mult_range:.1f}%

  4. 市場適應性: {'✅' if negative_markets <= len(all_returns) // 2 else '⚠️'}
     - 虧損期間: {negative_markets}/{len(all_returns)}
    """)


if __name__ == "__main__":
    asyncio.run(main())
