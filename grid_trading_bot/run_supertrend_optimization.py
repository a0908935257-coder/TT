#!/usr/bin/env python3
"""
Supertrend Bot 參數優化器.

針對 Supertrend Bot 策略進行參數優化，使用 Walk-Forward 驗證避免過度擬合。
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
    """回測結果"""
    config: Dict[str, Any]
    total_return_pct: float
    win_rate: float
    trades: int
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_bars_held: float


class SupertrendBacktester:
    """Supertrend 策略回測器"""

    def __init__(self, klines: List[Kline], leverage: int = 10):
        self.klines = klines
        self.leverage = leverage
        self.fee_rate = 0.0004  # 0.04% taker fee

    def calculate_supertrend(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        atr_period: int,
        atr_multiplier: float,
    ) -> List[Dict]:
        """計算 Supertrend 指標"""
        results = []

        # Calculate ATR
        atrs = []
        for i in range(len(closes)):
            if i < 1:
                atrs.append(highs[i] - lows[i])
                continue
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            atrs.append(tr)

        # Calculate Supertrend
        upper_band = [0.0] * len(closes)
        lower_band = [0.0] * len(closes)
        supertrend = [0.0] * len(closes)
        trend = [1] * len(closes)  # 1 = bullish, -1 = bearish

        for i in range(atr_period, len(closes)):
            # Calculate ATR (SMA of TR)
            atr = sum(atrs[i - atr_period + 1:i + 1]) / atr_period

            # Calculate basic bands
            hl2 = (highs[i] + lows[i]) / 2
            basic_upper = hl2 + atr_multiplier * atr
            basic_lower = hl2 - atr_multiplier * atr

            # Final bands (use previous band if better)
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

            # Determine trend
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

            results.append({
                'index': i,
                'atr': atr,
                'upper': upper_band[i],
                'lower': lower_band[i],
                'supertrend': supertrend[i],
                'trend': trend[i],
            })

        return results

    def run(self, config: Dict[str, Any]) -> BacktestResult:
        """執行回測"""
        atr_period = config.get('atr_period', 25)
        atr_multiplier = config.get('atr_multiplier', 3.0)
        stop_loss_pct = config.get('stop_loss_pct', 0.02)
        use_trailing_stop = config.get('use_trailing_stop', False)
        trailing_stop_pct = config.get('trailing_stop_pct', 0.03)

        # Extract OHLC data
        closes = [float(k.close) for k in self.klines]
        highs = [float(k.high) for k in self.klines]
        lows = [float(k.low) for k in self.klines]

        # Calculate Supertrend
        st_data = self.calculate_supertrend(highs, lows, closes, atr_period, atr_multiplier)

        if len(st_data) < 10:
            return BacktestResult(
                config=config, total_return_pct=0, win_rate=0, trades=0,
                sharpe=0, max_drawdown=0, profit_factor=0, avg_bars_held=0
            )

        # State
        position = None  # {'side': 'long'/'short', 'entry': price, 'entry_bar': bar, 'highest': price, 'lowest': price}
        trades = []
        equity = [1000.0]  # Start with $1000

        prev_trend = st_data[0]['trend']

        for st in st_data[1:]:
            i = st['index']
            price = closes[i]
            current_trend = st['trend']

            # Check exit conditions
            if position is not None:
                bars_held = i - position['entry_bar']
                entry_price = position['entry']

                # Update trailing stop tracking
                if position['side'] == 'long':
                    if price > position.get('highest', entry_price):
                        position['highest'] = price
                else:
                    if price < position.get('lowest', entry_price):
                        position['lowest'] = price

                # Calculate PnL
                if position['side'] == 'long':
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - price) / entry_price

                exit_reason = None

                # Stop loss check
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = 'stop_loss'

                # Trailing stop check
                if use_trailing_stop and not exit_reason:
                    if position['side'] == 'long':
                        highest = position.get('highest', entry_price)
                        trail_stop = highest * (1 - trailing_stop_pct)
                        if price < trail_stop:
                            exit_reason = 'trailing_stop'
                    else:
                        lowest = position.get('lowest', entry_price)
                        trail_stop = lowest * (1 + trailing_stop_pct)
                        if price > trail_stop:
                            exit_reason = 'trailing_stop'

                # Signal flip - Supertrend changed direction
                if not exit_reason:
                    if position['side'] == 'long' and current_trend == -1:
                        exit_reason = 'signal_flip'
                    elif position['side'] == 'short' and current_trend == 1:
                        exit_reason = 'signal_flip'

                if exit_reason:
                    # Close position
                    gross_pnl = pnl_pct * self.leverage * equity[-1]
                    fee = equity[-1] * self.fee_rate * 2  # entry + exit
                    net_pnl = gross_pnl - fee

                    trades.append({
                        'side': position['side'],
                        'entry': entry_price,
                        'exit': price,
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct,
                        'bars': bars_held,
                        'reason': exit_reason,
                    })

                    equity.append(equity[-1] + net_pnl)
                    position = None

            # Check entry conditions (trend change)
            if position is None and current_trend != prev_trend:
                if current_trend == 1:  # Bullish flip
                    position = {
                        'side': 'long',
                        'entry': price,
                        'entry_bar': i,
                        'highest': price,
                        'lowest': price,
                    }
                elif current_trend == -1:  # Bearish flip
                    position = {
                        'side': 'short',
                        'entry': price,
                        'entry_bar': i,
                        'highest': price,
                        'lowest': price,
                    }

            prev_trend = current_trend

        # Close any remaining position
        if position is not None:
            price = closes[-1]
            entry_price = position['entry']
            if position['side'] == 'long':
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price

            gross_pnl = pnl_pct * self.leverage * equity[-1]
            fee = equity[-1] * self.fee_rate * 2
            net_pnl = gross_pnl - fee

            trades.append({
                'side': position['side'],
                'entry': entry_price,
                'exit': price,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'bars': len(self.klines) - position['entry_bar'],
                'reason': 'end',
            })
            equity.append(equity[-1] + net_pnl)

        # Calculate metrics
        if not trades:
            return BacktestResult(
                config=config, total_return_pct=0, win_rate=0, trades=0,
                sharpe=0, max_drawdown=0, profit_factor=0, avg_bars_held=0
            )

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total_return_pct = (equity[-1] - 1000) / 1000 * 100
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Sharpe ratio
        returns = [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, len(equity))]
        if returns and len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = (avg_return / std_return * (252 * 24 / 0.25) ** 0.5) if std_return > 0 else 0
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

        # Profit factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average bars held
        avg_bars = sum(t['bars'] for t in trades) / len(trades) if trades else 0

        return BacktestResult(
            config=config,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            trades=len(trades),
            sharpe=sharpe,
            max_drawdown=max_dd * 100,
            profit_factor=profit_factor,
            avg_bars_held=avg_bars,
        )


async def fetch_data(days: int = 730) -> List[Kline]:
    """獲取歷史數據 (2年)"""
    client = ExchangeClient()
    await client.connect()

    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        klines = []
        current_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

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
            print(f"\r  已獲取 {len(klines)} 根 K 線...", end="")

        print()
        return klines
    finally:
        await client.close()


def run_grid_search(klines: List[Kline], leverage: int = 10) -> List[BacktestResult]:
    """執行參數網格搜索"""
    configs = []

    # 參數範圍 (針對過度擬合測試中發現的敏感參數)
    atr_periods = [14, 20, 25, 30, 35, 40]
    atr_multipliers = [2.5, 3.0, 3.5, 4.0, 4.5]
    stop_losses = [0.015, 0.02, 0.025, 0.03]
    trailing_configs = [
        {'use_trailing': False, 'trail_pct': 0},
        {'use_trailing': True, 'trail_pct': 0.02},
        {'use_trailing': True, 'trail_pct': 0.03},
    ]

    for atr_p in atr_periods:
        for atr_m in atr_multipliers:
            for sl in stop_losses:
                for trail in trailing_configs:
                    configs.append({
                        'atr_period': atr_p,
                        'atr_multiplier': atr_m,
                        'stop_loss_pct': sl,
                        'use_trailing_stop': trail['use_trailing'],
                        'trailing_stop_pct': trail['trail_pct'],
                    })

    print(f"\n測試 {len(configs)} 種參數組合...")

    backtester = SupertrendBacktester(klines, leverage=leverage)
    results = []

    for i, config in enumerate(configs):
        result = backtester.run(config)
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"\r  進度: {i + 1}/{len(configs)}", end="")

    print()
    return results


def walk_forward_validation(
    klines: List[Kline],
    config: Dict[str, Any],
    n_splits: int = 5,
    leverage: int = 10,
) -> Dict:
    """Walk-Forward 驗證"""
    split_size = len(klines) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        test_start = (i + 1) * split_size
        test_end = test_start + split_size

        if test_end > len(klines):
            break

        test_klines = klines[test_start:test_end]
        backtester = SupertrendBacktester(test_klines, leverage=leverage)
        result = backtester.run(config)
        results.append({
            'period': i + 1,
            'return_pct': result.total_return_pct,
            'trades': result.trades,
            'win_rate': result.win_rate,
            'max_dd': result.max_drawdown,
            'sharpe': result.sharpe,
        })

    profitable_periods = sum(1 for r in results if r['return_pct'] > 0)
    avg_return = sum(r['return_pct'] for r in results) / len(results) if results else 0
    avg_sharpe = sum(r['sharpe'] for r in results) / len(results) if results else 0
    consistency = profitable_periods / len(results) if results else 0

    return {
        'config': config,
        'results': results,
        'profitable_periods': profitable_periods,
        'total_periods': len(results),
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'consistency': consistency,
    }


def in_sample_out_sample_test(
    klines: List[Kline],
    config: Dict[str, Any],
    split_ratio: float = 0.7,
    leverage: int = 10,
) -> Dict:
    """In-Sample / Out-of-Sample 測試"""
    split_idx = int(len(klines) * split_ratio)

    in_sample = klines[:split_idx]
    out_sample = klines[split_idx:]

    bt_in = SupertrendBacktester(in_sample, leverage=leverage)
    bt_out = SupertrendBacktester(out_sample, leverage=leverage)

    result_in = bt_in.run(config)
    result_out = bt_out.run(config)

    degradation = 0
    if result_in.total_return_pct > 0:
        degradation = (result_in.total_return_pct - result_out.total_return_pct) / result_in.total_return_pct * 100

    return {
        'in_sample': {
            'return_pct': result_in.total_return_pct,
            'trades': result_in.trades,
            'win_rate': result_in.win_rate,
            'sharpe': result_in.sharpe,
            'max_dd': result_in.max_drawdown,
        },
        'out_sample': {
            'return_pct': result_out.total_return_pct,
            'trades': result_out.trades,
            'win_rate': result_out.win_rate,
            'sharpe': result_out.sharpe,
            'max_dd': result_out.max_drawdown,
        },
        'degradation_pct': degradation,
    }


async def main():
    print("=" * 70)
    print("           Supertrend Bot 參數優化器")
    print("=" * 70)

    # 獲取數據
    print("\n正在獲取 BTCUSDT 15m 歷史數據 (2 年)...")
    klines = await fetch_data(days=730)
    print(f"  總共獲取 {len(klines)} 根 K 線\n")

    # 階段 1: 網格搜索
    print("=" * 70)
    print("  階段 1: 參數網格搜索")
    print("=" * 70)

    results = run_grid_search(klines, leverage=10)

    # 過濾有效結果 (至少 50 筆交易，Sharpe > -5)
    valid_results = [r for r in results if r.trades >= 50 and r.sharpe > -5]

    # 按綜合評分排序 (Sharpe * 0.5 + Return * 0.3 + (100 - MaxDD) * 0.2)
    def score(r):
        return r.sharpe * 0.5 + r.total_return_pct * 0.003 + (100 - r.max_drawdown) * 0.02

    valid_results.sort(key=score, reverse=True)

    print("\n" + "=" * 70)
    print("  階段 2: Top 20 候選配置")
    print("=" * 70)

    print(f"\n{'#':<3} {'ATR_P':<6} {'ATR_M':<6} {'SL%':<5} {'Trail':<6} | {'Return':>8} {'WinR':>6} {'Sharpe':>7} {'MaxDD':>6} {'Trades':>6}")
    print("-" * 85)

    top_configs = []
    for i, r in enumerate(valid_results[:20]):
        c = r.config
        trail = f"{c['trailing_stop_pct']*100:.0f}%" if c['use_trailing_stop'] else "No"

        print(f"{i+1:<3} {c['atr_period']:<6} {c['atr_multiplier']:<6.1f} {c['stop_loss_pct']*100:<5.1f} {trail:<6} | {r.total_return_pct:>7.1f}% {r.win_rate:>5.1f}% {r.sharpe:>7.2f} {r.max_drawdown:>5.1f}% {r.trades:>6}")

        if i < 10:
            top_configs.append(c)

    # 階段 3: Walk-Forward 驗證
    print("\n" + "=" * 70)
    print("  階段 3: Walk-Forward 驗證 (防止過度擬合)")
    print("=" * 70)

    validated = []
    for i, config in enumerate(top_configs):
        wf_result = walk_forward_validation(klines, config, n_splits=6, leverage=10)
        validated.append(wf_result)
        print(f"\r  驗證配置 {i+1}/{len(top_configs)}...", end="")
    print()

    # 按一致性和平均報酬排序
    validated.sort(key=lambda x: (x['consistency'], x['avg_sharpe']), reverse=True)

    print(f"\n{'#':<3} {'ATR_P':<6} {'ATR_M':<6} {'SL%':<5} {'Trail':<6} | {'獲利期':<8} {'一致性':<8} {'平均報酬':>10} {'評價':<6}")
    print("-" * 80)

    for i, v in enumerate(validated):
        c = v['config']
        trail = f"{c['trailing_stop_pct']*100:.0f}%" if c['use_trailing_stop'] else "No"

        if v['consistency'] >= 0.7:
            status = "✅ 優"
        elif v['consistency'] >= 0.5:
            status = "🟡 可"
        else:
            status = "❌ 差"

        print(f"{i+1:<3} {c['atr_period']:<6} {c['atr_multiplier']:<6.1f} {c['stop_loss_pct']*100:<5.1f} {trail:<6} | {v['profitable_periods']}/{v['total_periods']:<6} {v['consistency']*100:>5.1f}%   {v['avg_return']:>9.1f}%  {status}")

    # 階段 4: In-Sample / Out-of-Sample 測試
    print("\n" + "=" * 70)
    print("  階段 4: In-Sample / Out-of-Sample 測試")
    print("=" * 70)

    best_config = validated[0]['config'] if validated else top_configs[0]
    iso_result = in_sample_out_sample_test(klines, best_config, split_ratio=0.7, leverage=10)

    print(f"""
  最佳配置 In-Sample/Out-of-Sample 測試:

  │ 指標          │ In-Sample (70%)  │ Out-of-Sample (30%) │
  │───────────────│──────────────────│─────────────────────│
  │ 報酬率        │ {iso_result['in_sample']['return_pct']:>14.1f}% │ {iso_result['out_sample']['return_pct']:>17.1f}% │
  │ 交易數        │ {iso_result['in_sample']['trades']:>15} │ {iso_result['out_sample']['trades']:>18} │
  │ 勝率          │ {iso_result['in_sample']['win_rate']:>14.1f}% │ {iso_result['out_sample']['win_rate']:>17.1f}% │
  │ Sharpe        │ {iso_result['in_sample']['sharpe']:>15.2f} │ {iso_result['out_sample']['sharpe']:>18.2f} │
  │ 最大回撤      │ {iso_result['in_sample']['max_dd']:>14.1f}% │ {iso_result['out_sample']['max_dd']:>17.1f}% │

  衰減率: {iso_result['degradation_pct']:.1f}%
""")

    if iso_result['degradation_pct'] < 50:
        print("  ✅ 通過：樣本外衰減 < 50%")
    else:
        print("  ⚠️ 警告：樣本外衰減 >= 50%，存在過度擬合風險")

    # 最終建議
    print("\n" + "=" * 70)
    print("  最佳穩健配置")
    print("=" * 70)

    best = validated[0] if validated and validated[0]['consistency'] >= 0.5 else None

    if best:
        c = best['config']
        print(f"""
  ATR Period: {c['atr_period']}
  ATR Multiplier: {c['atr_multiplier']}
  Stop Loss: {c['stop_loss_pct']*100:.1f}%
  Trailing Stop: {'開啟 (' + str(c['trailing_stop_pct']*100) + '%)' if c['use_trailing_stop'] else '關閉'}

  驗證結果:
    Walk-Forward 獲利期間: {best['profitable_periods']}/{best['total_periods']}
    一致性: {best['consistency']*100:.1f}%
    平均報酬: {best['avg_return']:.1f}%
    平均 Sharpe: {best['avg_sharpe']:.2f}
""")

        # 輸出 .env 配置
        print("  建議 .env 配置:")
        print("-" * 50)
        print(f"  SUPERTREND_ATR_PERIOD={c['atr_period']}")
        print(f"  SUPERTREND_ATR_MULTIPLIER={c['atr_multiplier']}")
        print(f"  SUPERTREND_STOP_LOSS_PCT={c['stop_loss_pct']}")
        print(f"  SUPERTREND_USE_TRAILING_STOP={'true' if c['use_trailing_stop'] else 'false'}")
        if c['use_trailing_stop']:
            print(f"  SUPERTREND_TRAILING_STOP_PCT={c['trailing_stop_pct']}")

        # 與當前配置比較
        print("\n  與當前配置比較:")
        print("-" * 50)
        current = {'atr_period': 5, 'atr_multiplier': 2.5, 'stop_loss_pct': 0.02, 'use_trailing_stop': False, 'trailing_stop_pct': 0.03}
        current_wf = walk_forward_validation(klines, current, n_splits=6, leverage=10)

        print(f"  │ 指標        │ 當前配置     │ 優化配置     │")
        print(f"  │─────────────│──────────────│──────────────│")
        print(f"  │ ATR Period  │ {current['atr_period']:<12} │ {c['atr_period']:<12} │")
        print(f"  │ ATR Mult    │ {current['atr_multiplier']:<12} │ {c['atr_multiplier']:<12} │")
        print(f"  │ Stop Loss   │ {current['stop_loss_pct']*100:<11}% │ {c['stop_loss_pct']*100:<11}% │")
        print(f"  │ 一致性      │ {current_wf['consistency']*100:<11.1f}% │ {best['consistency']*100:<11.1f}% │")
        print(f"  │ 平均報酬    │ {current_wf['avg_return']:<11.1f}% │ {best['avg_return']:<11.1f}% │")

    else:
        print("\n  ⚠️ 沒有找到通過驗證的穩健配置")
        print("  建議: 考慮降低槓桿或調整策略")

    print("\n" + "=" * 70)
    print("  優化完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
