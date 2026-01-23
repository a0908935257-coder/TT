#!/usr/bin/env python3
"""
Supertrend 指標組合測試.

測試不同指標過濾器對 Supertrend 策略的影響:
1. 純 Supertrend (基線)
2. Supertrend + ADX 過濾 (趨勢強度)
3. Supertrend + MA 過濾 (趨勢方向)
4. Supertrend + RSI 過濾 (動量確認)
5. Supertrend + Volume 過濾 (成交量確認)
6. Supertrend + ADX + MA (組合過濾)
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))

from src.core.models import Kline
from src.exchange import ExchangeClient


@dataclass
class BacktestResult:
    name: str
    total_return_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    avg_trade_pct: float


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int, idx: int) -> float:
    """Calculate ATR at index."""
    if idx < period:
        return 0
    trs = []
    for j in range(idx - period + 1, idx + 1):
        tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
        trs.append(tr)
    return sum(trs) / period


def calculate_supertrend(
    highs: List[float], lows: List[float], closes: List[float],
    atr_period: int, atr_multiplier: float
) -> List[Dict]:
    """Calculate Supertrend indicator."""
    results = []
    upper_band = [0.0] * len(closes)
    lower_band = [0.0] * len(closes)
    supertrend = [0.0] * len(closes)
    trend = [0] * len(closes)

    for i in range(atr_period, len(closes)):
        atr = calculate_atr(highs, lows, closes, atr_period, i)
        hl2 = (highs[i] + lows[i]) / 2
        basic_upper = hl2 + atr_multiplier * atr
        basic_lower = hl2 - atr_multiplier * atr

        if i > atr_period:
            if basic_lower > lower_band[i-1] or closes[i-1] < lower_band[i-1]:
                lower_band[i] = basic_lower
            else:
                lower_band[i] = lower_band[i-1]
            if basic_upper < upper_band[i-1] or closes[i-1] > upper_band[i-1]:
                upper_band[i] = basic_upper
            else:
                upper_band[i] = upper_band[i-1]
        else:
            upper_band[i] = basic_upper
            lower_band[i] = basic_lower

        if i > atr_period:
            if supertrend[i-1] == upper_band[i-1]:
                trend[i] = 1 if closes[i] > upper_band[i] else -1
            else:
                trend[i] = -1 if closes[i] < lower_band[i] else 1
            supertrend[i] = lower_band[i] if trend[i] == 1 else upper_band[i]
        else:
            trend[i] = 1 if closes[i] > basic_upper else -1
            supertrend[i] = lower_band[i] if trend[i] == 1 else upper_band[i]

        results.append({'index': i, 'trend': trend[i], 'atr': atr, 'supertrend': supertrend[i]})

    return results


def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """Calculate ADX indicator."""
    adx = [0.0] * len(closes)
    if len(closes) < period * 2:
        return adx

    plus_dm = [0.0] * len(closes)
    minus_dm = [0.0] * len(closes)
    tr = [0.0] * len(closes)

    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))

    # Smoothed values
    smoothed_tr = [0.0] * len(closes)
    smoothed_plus = [0.0] * len(closes)
    smoothed_minus = [0.0] * len(closes)
    dx = [0.0] * len(closes)

    for i in range(period, len(closes)):
        if i == period:
            smoothed_tr[i] = sum(tr[1:period+1])
            smoothed_plus[i] = sum(plus_dm[1:period+1])
            smoothed_minus[i] = sum(minus_dm[1:period+1])
        else:
            smoothed_tr[i] = smoothed_tr[i-1] - smoothed_tr[i-1]/period + tr[i]
            smoothed_plus[i] = smoothed_plus[i-1] - smoothed_plus[i-1]/period + plus_dm[i]
            smoothed_minus[i] = smoothed_minus[i-1] - smoothed_minus[i-1]/period + minus_dm[i]

        plus_di = 100 * smoothed_plus[i] / smoothed_tr[i] if smoothed_tr[i] > 0 else 0
        minus_di = 100 * smoothed_minus[i] / smoothed_tr[i] if smoothed_tr[i] > 0 else 0
        di_sum = plus_di + minus_di
        dx[i] = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0

    # ADX = smoothed DX
    for i in range(period * 2, len(closes)):
        if i == period * 2:
            adx[i] = sum(dx[period:period*2]) / period
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx


def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    """Calculate RSI indicator."""
    rsi = [50.0] * len(closes)
    if len(closes) < period + 1:
        return rsi

    gains = [0.0] * len(closes)
    losses = [0.0] * len(closes)

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains[i] = change if change > 0 else 0
        losses[i] = -change if change < 0 else 0

    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period

    for i in range(period, len(closes)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_ma(closes: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average."""
    ma = [0.0] * len(closes)
    for i in range(period - 1, len(closes)):
        ma[i] = sum(closes[i-period+1:i+1]) / period
    return ma


def calculate_volume_ma(volumes: List[float], period: int) -> List[float]:
    """Calculate Volume Moving Average."""
    vma = [0.0] * len(volumes)
    for i in range(period - 1, len(volumes)):
        vma[i] = sum(volumes[i-period+1:i+1]) / period
    return vma


def run_backtest(
    klines: List[Kline],
    name: str,
    use_adx: bool = False,
    adx_threshold: int = 25,
    use_ma: bool = False,
    ma_period: int = 50,
    use_rsi: bool = False,
    rsi_low: int = 40,
    rsi_high: int = 60,
    use_volume: bool = False,
    atr_period: int = 25,
    atr_multiplier: float = 3.0,
    stop_loss_pct: float = 0.03,
    leverage: int = 2,
) -> BacktestResult:
    """Run backtest with specified filters."""

    closes = [float(k.close) for k in klines]
    highs = [float(k.high) for k in klines]
    lows = [float(k.low) for k in klines]
    volumes = [float(k.volume) for k in klines]

    # Calculate indicators
    st_data = calculate_supertrend(highs, lows, closes, atr_period, atr_multiplier)
    adx = calculate_adx(highs, lows, closes, 14) if use_adx else None
    ma = calculate_ma(closes, ma_period) if use_ma else None
    rsi = calculate_rsi(closes, 14) if use_rsi else None
    vol_ma = calculate_volume_ma(volumes, 20) if use_volume else None

    if len(st_data) < 10:
        return BacktestResult(name, 0, 0, 0, 0, 0, 0)

    # Backtest
    position = None
    trades = []
    equity = [10000.0]
    fee_rate = 0.0004
    prev_trend = st_data[0]['trend']

    for st in st_data[1:]:
        i = st['index']
        price = closes[i]
        current_trend = st['trend']

        # Exit logic
        if position is not None:
            entry_price = position['entry']
            if position['side'] == 'long':
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price

            exit_reason = None

            # Stop loss
            if pnl_pct <= -stop_loss_pct:
                exit_reason = 'stop_loss'

            # Signal flip
            if not exit_reason:
                if position['side'] == 'long' and current_trend == -1:
                    exit_reason = 'signal_flip'
                elif position['side'] == 'short' and current_trend == 1:
                    exit_reason = 'signal_flip'

            if exit_reason:
                gross_pnl = pnl_pct * leverage * equity[-1] * 0.1  # 10% position
                fee = equity[-1] * 0.1 * fee_rate * 2
                net_pnl = gross_pnl - fee
                trades.append({'pnl': net_pnl, 'pnl_pct': pnl_pct})
                equity.append(equity[-1] + net_pnl)
                position = None

        # Entry logic - check filters
        if position is None and current_trend != prev_trend:
            can_enter = True

            # ADX filter: only enter if trend is strong
            if use_adx and adx is not None:
                if adx[i] < adx_threshold:
                    can_enter = False

            # MA filter: only trade in direction of trend
            if use_ma and ma is not None and ma[i] > 0:
                if current_trend == 1 and price < ma[i]:  # Long but below MA
                    can_enter = False
                elif current_trend == -1 and price > ma[i]:  # Short but above MA
                    can_enter = False

            # RSI filter: avoid extreme levels
            if use_rsi and rsi is not None:
                if current_trend == 1 and rsi[i] > rsi_high:  # Overbought for long
                    can_enter = False
                elif current_trend == -1 and rsi[i] < rsi_low:  # Oversold for short
                    can_enter = False

            # Volume filter: only enter on above-average volume
            if use_volume and vol_ma is not None and vol_ma[i] > 0:
                if volumes[i] < vol_ma[i]:
                    can_enter = False

            if can_enter:
                position = {
                    'side': 'long' if current_trend == 1 else 'short',
                    'entry': price,
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
        gross_pnl = pnl_pct * leverage * equity[-1] * 0.1
        fee = equity[-1] * 0.1 * fee_rate * 2
        trades.append({'pnl': gross_pnl - fee, 'pnl_pct': pnl_pct})
        equity.append(equity[-1] + gross_pnl - fee)

    # Calculate metrics
    if not trades:
        return BacktestResult(name, 0, 0, 0, 0, 0, 0)

    total_return = (equity[-1] - 10000) / 10000 * 100
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    avg_trade = sum(t['pnl_pct'] for t in trades) / len(trades) * 100 if trades else 0

    return BacktestResult(name, total_return, len(trades), win_rate, profit_factor, max_dd, avg_trade)


def walk_forward_test(klines: List[Kline], config: Dict[str, Any], n_splits: int = 6) -> Dict:
    """Walk-forward validation."""
    split_size = len(klines) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        test_start = (i + 1) * split_size
        test_end = test_start + split_size
        if test_end > len(klines):
            break

        test_klines = klines[test_start:test_end]
        result = run_backtest(test_klines, config['name'], **{k: v for k, v in config.items() if k != 'name'})
        results.append(result.total_return_pct)

    profitable = sum(1 for r in results if r > 0)
    avg_return = sum(results) / len(results) if results else 0

    return {
        'profitable': profitable,
        'total': len(results),
        'consistency': profitable / len(results) if results else 0,
        'avg_return': avg_return,
    }


async def fetch_data(days: int = 730) -> List[Kline]:
    """Fetch historical data."""
    client = ExchangeClient()
    await client.connect()

    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        klines = []
        current_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        while current_ts < end_ts:
            batch = await client.futures.get_klines(
                symbol="BTCUSDT",
                interval="15m",
                start_time=current_ts,
                limit=1500,
            )
            if not batch:
                break
            klines.extend(batch)
            current_ts = int(batch[-1].close_time.timestamp() * 1000) + 1
            print(f"\r  已獲取 {len(klines)} 根 K 線...", end="")

        print()
        return klines
    finally:
        await client.close()


async def main():
    print("=" * 80)
    print("  Supertrend 指標組合測試")
    print("=" * 80)

    print("\n正在獲取 BTCUSDT 15m 歷史數據 (2 年)...")
    klines = await fetch_data(days=730)
    print(f"  總共 {len(klines)} 根 K 線\n")

    # Test configurations
    configs = [
        {'name': '1. 純 Supertrend (基線)', 'use_adx': False, 'use_ma': False, 'use_rsi': False, 'use_volume': False},
        {'name': '2. ST + ADX(20)', 'use_adx': True, 'adx_threshold': 20, 'use_ma': False, 'use_rsi': False, 'use_volume': False},
        {'name': '3. ST + ADX(25)', 'use_adx': True, 'adx_threshold': 25, 'use_ma': False, 'use_rsi': False, 'use_volume': False},
        {'name': '4. ST + ADX(30)', 'use_adx': True, 'adx_threshold': 30, 'use_ma': False, 'use_rsi': False, 'use_volume': False},
        {'name': '5. ST + MA(50)', 'use_adx': False, 'use_ma': True, 'ma_period': 50, 'use_rsi': False, 'use_volume': False},
        {'name': '6. ST + MA(100)', 'use_adx': False, 'use_ma': True, 'ma_period': 100, 'use_rsi': False, 'use_volume': False},
        {'name': '7. ST + MA(200)', 'use_adx': False, 'use_ma': True, 'ma_period': 200, 'use_rsi': False, 'use_volume': False},
        {'name': '8. ST + RSI(40-60)', 'use_adx': False, 'use_ma': False, 'use_rsi': True, 'rsi_low': 40, 'rsi_high': 60, 'use_volume': False},
        {'name': '9. ST + RSI(30-70)', 'use_adx': False, 'use_ma': False, 'use_rsi': True, 'rsi_low': 30, 'rsi_high': 70, 'use_volume': False},
        {'name': '10. ST + Volume', 'use_adx': False, 'use_ma': False, 'use_rsi': False, 'use_volume': True},
        {'name': '11. ST + ADX(25) + MA(100)', 'use_adx': True, 'adx_threshold': 25, 'use_ma': True, 'ma_period': 100, 'use_rsi': False, 'use_volume': False},
        {'name': '12. ST + ADX(25) + RSI', 'use_adx': True, 'adx_threshold': 25, 'use_ma': False, 'use_rsi': True, 'rsi_low': 40, 'rsi_high': 60, 'use_volume': False},
        {'name': '13. ST + MA(100) + Volume', 'use_adx': False, 'use_ma': True, 'ma_period': 100, 'use_rsi': False, 'use_volume': True},
        {'name': '14. ST + ADX(20) + MA(50)', 'use_adx': True, 'adx_threshold': 20, 'use_ma': True, 'ma_period': 50, 'use_rsi': False, 'use_volume': False},
        {'name': '15. 全部過濾器', 'use_adx': True, 'adx_threshold': 20, 'use_ma': True, 'ma_period': 100, 'use_rsi': True, 'rsi_low': 40, 'rsi_high': 60, 'use_volume': True},
    ]

    # Stage 1: Full period backtest
    print("=" * 80)
    print("  階段 1: 完整期間回測 (2 年)")
    print("=" * 80)
    print(f"\n{'配置':<30} {'報酬%':>10} {'交易數':>8} {'勝率':>8} {'獲利因子':>10} {'回撤%':>8}")
    print("-" * 80)

    results = []
    for config in configs:
        result = run_backtest(klines, **config)
        results.append((config, result))
        print(f"{result.name:<30} {result.total_return_pct:>+9.1f}% {result.total_trades:>8} {result.win_rate:>7.1f}% {result.profit_factor:>10.2f} {result.max_drawdown_pct:>7.1f}%")

    # Stage 2: Walk-forward validation for top performers
    print("\n" + "=" * 80)
    print("  階段 2: Walk-Forward 驗證 (前 8 名配置)")
    print("=" * 80)

    # Sort by return
    results.sort(key=lambda x: x[1].total_return_pct, reverse=True)

    print(f"\n{'配置':<30} {'獲利期':>10} {'一致性':>10} {'平均報酬':>12} {'評價':>8}")
    print("-" * 75)

    validated = []
    for config, result in results[:8]:
        wf = walk_forward_test(klines, config, n_splits=8)
        validated.append((config, result, wf))

        if wf['consistency'] >= 0.6:
            status = "✅ 優"
        elif wf['consistency'] >= 0.5:
            status = "🟡 可"
        else:
            status = "❌ 差"

        print(f"{result.name:<30} {wf['profitable']}/{wf['total']:>7} {wf['consistency']*100:>9.0f}% {wf['avg_return']:>+11.1f}% {status:>8}")

    # Stage 3: Out-of-sample test
    print("\n" + "=" * 80)
    print("  階段 3: 樣本外測試 (最後 30%)")
    print("=" * 80)

    split_idx = int(len(klines) * 0.7)
    oos_klines = klines[split_idx:]

    print(f"\n{'配置':<30} {'IS 報酬%':>12} {'OOS 報酬%':>12} {'衰退率':>10}")
    print("-" * 70)

    best_config = None
    best_score = -999

    for config, result, wf in validated:
        oos_result = run_backtest(oos_klines, **config)
        is_return = result.total_return_pct
        oos_return = oos_result.total_return_pct

        if is_return > 0:
            degradation = (is_return - oos_return) / is_return * 100
        else:
            degradation = 0

        print(f"{result.name:<30} {is_return:>+11.1f}% {oos_return:>+11.1f}% {degradation:>+9.1f}%")

        # Score: consistency * 0.4 + oos_return * 0.4 + (100 - degradation) * 0.2
        score = wf['consistency'] * 40 + oos_return * 0.4 + (100 - abs(degradation)) * 0.2
        if score > best_score:
            best_score = score
            best_config = (config, result, wf, oos_result)

    # Final recommendation
    print("\n" + "=" * 80)
    print("  最佳配置推薦")
    print("=" * 80)

    if best_config:
        config, result, wf, oos = best_config
        print(f"""
  配置: {config['name']}

  參數:
    - ADX 過濾: {'是 (閾值={})'.format(config.get('adx_threshold', '-')) if config.get('use_adx') else '否'}
    - MA 過濾: {'是 (週期={})'.format(config.get('ma_period', '-')) if config.get('use_ma') else '否'}
    - RSI 過濾: {'是 ({}-{})'.format(config.get('rsi_low', '-'), config.get('rsi_high', '-')) if config.get('use_rsi') else '否'}
    - Volume 過濾: {'是' if config.get('use_volume') else '否'}

  績效:
    - 2年總報酬: {result.total_return_pct:+.1f}%
    - 交易數: {result.total_trades} (減少 = 更精選)
    - 勝率: {result.win_rate:.1f}%
    - 獲利因子: {result.profit_factor:.2f}
    - 最大回撤: {result.max_drawdown_pct:.1f}%

  驗證:
    - Walk-Forward 一致性: {wf['consistency']*100:.0f}% ({wf['profitable']}/{wf['total']} 期間獲利)
    - 樣本外報酬: {oos.total_return_pct:+.1f}%

  與純 Supertrend 比較:
    - 交易數變化: {result.total_trades} vs {results[-1][1].total_trades if results else 'N/A'}
    - 報酬變化: {result.total_return_pct:+.1f}% vs {results[-1][1].total_return_pct:+.1f}%
""")

        if wf['consistency'] >= 0.5:
            print("  結論: ✅ 此配置通過驗證，建議採用")
        else:
            print("  結論: ⚠️ 配置仍有改進空間，請謹慎使用")

    print("\n" + "=" * 80)
    print("  測試完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
