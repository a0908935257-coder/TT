#!/usr/bin/env python3
"""
Bollinger Bot 參數優化器.

針對當前市場重新優化 Bollinger Bot 策略參數。
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import itertools

sys.path.insert(0, os.path.dirname(__file__))

from src.core.models import Kline
from src.exchange import ExchangeClient


@dataclass
class BacktestResult:
    """回測結果"""
    config: Dict[str, Any]
    total_pnl: float
    win_rate: float
    trades: int
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_hold_time: float  # hours


class BollingerBacktester:
    """Bollinger Band 回測器"""

    def __init__(self, klines: List[Kline], leverage: int = 20):
        self.klines = klines
        self.leverage = leverage
        self.fee_rate = 0.0004  # 0.04% taker fee

    def run(self, config: Dict[str, Any]) -> BacktestResult:
        """執行回測"""
        bb_period = config.get('bb_period', 20)
        bb_std = config.get('bb_std', 2.0)
        strategy_mode = config.get('strategy_mode', 'breakout')  # breakout or mean_reversion
        use_trend_filter = config.get('use_trend_filter', False)
        trend_period = config.get('trend_period', 50)
        use_bbw_filter = config.get('use_bbw_filter', True)
        bbw_threshold = config.get('bbw_threshold', 20)
        bbw_lookback = config.get('bbw_lookback', 200)
        stop_loss_pct = config.get('stop_loss_pct', 0.015)
        take_profit_pct = config.get('take_profit_pct', 0.03)
        max_hold_bars = config.get('max_hold_bars', 48)
        use_atr_stop = config.get('use_atr_stop', True)
        atr_period = config.get('atr_period', 14)
        atr_mult = config.get('atr_mult', 2.0)

        # State
        position = None  # {'side': 'long'/'short', 'entry': price, 'entry_bar': bar}
        trades = []
        equity = [1000.0]  # Start with $1000

        closes = [float(k.close) for k in self.klines]
        highs = [float(k.high) for k in self.klines]
        lows = [float(k.low) for k in self.klines]

        # Calculate indicators
        for i in range(max(bb_period, trend_period, bbw_lookback, atr_period) + 10, len(self.klines)):
            price = closes[i]

            # Bollinger Bands
            bb_closes = closes[i-bb_period:i]
            sma = sum(bb_closes) / bb_period
            variance = sum((x - sma) ** 2 for x in bb_closes) / bb_period
            std = variance ** 0.5
            upper = sma + bb_std * std
            lower = sma - bb_std * std

            # BBW (Bollinger Band Width)
            bbw = (upper - lower) / sma * 100 if sma > 0 else 0

            # BBW percentile
            bbw_history = []
            for j in range(i - bbw_lookback, i):
                if j >= bb_period:
                    bb_c = closes[j-bb_period:j]
                    bb_sma = sum(bb_c) / bb_period
                    bb_var = sum((x - bb_sma) ** 2 for x in bb_c) / bb_period
                    bb_s = bb_var ** 0.5
                    bb_u = bb_sma + bb_std * bb_s
                    bb_l = bb_sma - bb_std * bb_s
                    bbw_history.append((bb_u - bb_l) / bb_sma * 100 if bb_sma > 0 else 0)

            bbw_percentile = sum(1 for x in bbw_history if x < bbw) / len(bbw_history) * 100 if bbw_history else 50

            # Trend filter
            if use_trend_filter and trend_period > 0:
                trend_sma = sum(closes[i-trend_period:i]) / trend_period
                trend = 1 if price > trend_sma else -1
            else:
                trend = 0  # No trend filter

            # ATR
            atr = 0
            if use_atr_stop:
                tr_list = []
                for j in range(i - atr_period, i):
                    tr = max(
                        highs[j] - lows[j],
                        abs(highs[j] - closes[j-1]),
                        abs(lows[j] - closes[j-1])
                    )
                    tr_list.append(tr)
                atr = sum(tr_list) / atr_period

            # Check exit conditions
            if position is not None:
                bars_held = i - position['entry_bar']
                entry_price = position['entry']

                if position['side'] == 'long':
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - price) / entry_price

                # Calculate stop loss price
                if use_atr_stop and atr > 0:
                    stop_distance = atr * atr_mult / entry_price
                else:
                    stop_distance = stop_loss_pct

                # Exit conditions
                exit_reason = None

                # Stop loss
                if pnl_pct <= -stop_distance:
                    exit_reason = 'stop_loss'
                # Take profit
                elif pnl_pct >= take_profit_pct:
                    exit_reason = 'take_profit'
                # Max hold time
                elif bars_held >= max_hold_bars:
                    exit_reason = 'timeout'
                # Mean reversion exit: price returns to middle band
                elif strategy_mode == 'mean_reversion':
                    if position['side'] == 'long' and price >= sma:
                        exit_reason = 'target'
                    elif position['side'] == 'short' and price <= sma:
                        exit_reason = 'target'
                # Breakout exit: price returns inside bands
                elif strategy_mode == 'breakout':
                    if position['side'] == 'long' and price < sma:
                        exit_reason = 'reversal'
                    elif position['side'] == 'short' and price > sma:
                        exit_reason = 'reversal'

                if exit_reason:
                    # Close position
                    gross_pnl = pnl_pct * self.leverage * equity[-1]
                    fee = equity[-1] * self.fee_rate * 2  # entry + exit fee
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

            # Check entry conditions (only if no position)
            if position is None:
                # BBW filter
                if use_bbw_filter and bbw_percentile < bbw_threshold:
                    continue

                signal = None

                if strategy_mode == 'breakout':
                    # Breakout: enter on band break
                    if price > upper:
                        if not use_trend_filter or trend >= 0:  # trend == 0 means no filter
                            signal = 'long'
                    elif price < lower:
                        if not use_trend_filter or trend <= 0:
                            signal = 'short'

                elif strategy_mode == 'mean_reversion':
                    # Mean reversion: enter on band touch, exit at middle
                    if price <= lower:
                        if not use_trend_filter or trend >= 0:
                            signal = 'long'
                    elif price >= upper:
                        if not use_trend_filter or trend <= 0:
                            signal = 'short'

                if signal:
                    position = {
                        'side': signal,
                        'entry': price,
                        'entry_bar': i,
                    }

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
                config=config,
                total_pnl=0,
                win_rate=0,
                trades=0,
                sharpe=0,
                max_drawdown=0,
                profit_factor=0,
                avg_hold_time=0,
            )

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]

        total_pnl = equity[-1] - 1000
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Sharpe ratio (simplified)
        returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity))]
        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = (avg_return / std_return * (252 * 24 / 0.25) ** 0.5) if std_return > 0 else 0  # 15min bars
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

        # Average hold time (in hours, assuming 15min bars)
        avg_hold_time = sum(t['bars'] for t in trades) / len(trades) * 0.25 if trades else 0

        return BacktestResult(
            config=config,
            total_pnl=total_pnl,
            win_rate=win_rate,
            trades=len(trades),
            sharpe=sharpe,
            max_drawdown=max_dd * 100,
            profit_factor=profit_factor,
            avg_hold_time=avg_hold_time,
        )


async def fetch_data(days: int = 60) -> List[Kline]:
    """獲取歷史數據"""
    client = ExchangeClient()
    await client.connect()

    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        klines = []
        current_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        while current_ts < end_ts:
            # Use spot API directly for start_time support
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
            print(f"  已獲取 {len(klines)} 根 K 線...")

        return klines
    finally:
        await client.close()


def run_optimization(klines: List[Kline]) -> List[BacktestResult]:
    """執行參數優化"""

    # 精簡的參數組合
    configs = []

    # 測試最關鍵的參數組合 (只測試 36 種)
    for mode in ['breakout', 'mean_reversion']:
        for bb_std in [2.0, 2.5, 3.0]:
            for use_trend in [True, False]:
                for atr_mult in [1.5, 2.0, 2.5]:
                    configs.append({
                        'strategy_mode': mode,
                        'bb_period': 20,  # 固定
                        'bb_std': bb_std,
                        'use_trend_filter': use_trend,
                        'trend_period': 50 if use_trend else 0,
                        'use_bbw_filter': True,
                        'bbw_threshold': 20,
                        'bbw_lookback': 100,  # 減少
                        'stop_loss_pct': 0.015,
                        'take_profit_pct': 0.03,
                        'max_hold_bars': 48,
                        'use_atr_stop': True,
                        'atr_period': 14,
                        'atr_mult': atr_mult,
                    })

    print(f"\n測試 {len(configs)} 種參數組合...\n")

    backtester = BollingerBacktester(klines, leverage=20)
    results = []

    for i, config in enumerate(configs):
        result = backtester.run(config)
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  進度: {i+1}/{len(configs)}")

    return results


def validate_top_configs(klines: List[Kline], top_configs: List[Dict], n_splits: int = 5) -> List[Dict]:
    """使用 walk-forward 驗證最佳配置"""

    split_size = len(klines) // (n_splits + 1)
    validated = []

    for config in top_configs:
        profits = []

        for i in range(n_splits):
            # 訓練集: 前 i+1 個區塊
            # 測試集: 第 i+2 個區塊
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = test_start + split_size

            if test_end > len(klines):
                break

            test_klines = klines[test_start:test_end]
            backtester = BollingerBacktester(test_klines, leverage=20)
            result = backtester.run(config)
            profits.append(result.total_pnl)

        profitable_periods = sum(1 for p in profits if p > 0)
        avg_pnl = sum(profits) / len(profits) if profits else 0

        validated.append({
            'config': config,
            'profitable_periods': profitable_periods,
            'total_periods': len(profits),
            'avg_pnl': avg_pnl,
            'consistency': profitable_periods / len(profits) if profits else 0,
        })

    return validated


async def main():
    print("=" * 70)
    print("           Bollinger Bot 參數優化器")
    print("=" * 70)

    # 獲取數據
    print("\n正在獲取 BTCUSDT 15m 歷史數據 (60 天)...")
    klines = await fetch_data(days=60)
    print(f"  總共獲取 {len(klines)} 根 K 線\n")

    # 執行優化
    print("=" * 70)
    print("  階段 1: 參數網格搜索")
    print("=" * 70)

    results = run_optimization(klines)

    # 過濾有效結果
    valid_results = [r for r in results if r.trades >= 10 and r.sharpe > -10]

    # 按 Sharpe 排序
    valid_results.sort(key=lambda x: x.sharpe, reverse=True)

    print("\n" + "=" * 70)
    print("  階段 2: Top 10 配置")
    print("=" * 70)

    print(f"\n{'排名':<4} {'模式':<12} {'BB週期':<6} {'BB σ':<6} {'趨勢':<4} {'BBW':<4} {'止損':<6} {'ATR':<4} | {'PnL':>10} {'勝率':>6} {'Sharpe':>8} {'回撤':>6} {'交易':>4}")
    print("-" * 110)

    top_configs = []
    for i, r in enumerate(valid_results[:20]):
        c = r.config
        mode = 'BRK' if c['strategy_mode'] == 'breakout' else 'MR'
        trend = 'Y' if c['use_trend_filter'] else 'N'
        bbw = 'Y' if c['use_bbw_filter'] else 'N'

        print(f"{i+1:<4} {mode:<12} {c['bb_period']:<6} {c['bb_std']:<6.1f} {trend:<4} {bbw:<4} {c['stop_loss_pct']*100:<6.1f} {c['atr_mult']:<4.1f} | {r.total_pnl:>10.2f} {r.win_rate:>5.1f}% {r.sharpe:>8.2f} {r.max_drawdown:>5.1f}% {r.trades:>4}")

        if i < 10:
            top_configs.append(c)

    # 驗證最佳配置
    print("\n" + "=" * 70)
    print("  階段 3: Walk-Forward 驗證")
    print("=" * 70)

    validated = validate_top_configs(klines, top_configs)
    validated.sort(key=lambda x: (x['consistency'], x['avg_pnl']), reverse=True)

    print(f"\n{'排名':<4} {'模式':<8} {'BB':<8} {'趨勢':<4} {'獲利期':<10} {'一致性':<10} {'平均PnL':>10}")
    print("-" * 70)

    for i, v in enumerate(validated):
        c = v['config']
        mode = 'BRK' if c['strategy_mode'] == 'breakout' else 'MR'
        trend = 'Y' if c['use_trend_filter'] else 'N'

        status = "✅" if v['consistency'] >= 0.6 else "⚠️" if v['consistency'] >= 0.4 else "❌"

        print(f"{i+1:<4} {mode:<8} {c['bb_period']}/{c['bb_std']:<5.1f} {trend:<4} {v['profitable_periods']}/{v['total_periods']:<8} {v['consistency']*100:>6.1f}% {status}  {v['avg_pnl']:>10.2f}")

    # 最佳配置
    best = validated[0] if validated else None

    print("\n" + "=" * 70)
    print("  最佳穩健配置")
    print("=" * 70)

    if best and best['consistency'] >= 0.4:
        c = best['config']
        print(f"""
  策略模式: {'突破策略' if c['strategy_mode'] == 'breakout' else '均值回歸'}
  布林帶週期: {c['bb_period']}
  布林帶標準差: {c['bb_std']}
  趨勢過濾: {'開啟' if c['use_trend_filter'] else '關閉'}
  BBW 過濾: {'開啟' if c['use_bbw_filter'] else '關閉'}
  止損: {c['stop_loss_pct']*100:.1f}%
  ATR 乘數: {c['atr_mult']}

  驗證結果:
    獲利期間: {best['profitable_periods']}/{best['total_periods']}
    一致性: {best['consistency']*100:.1f}%
    平均 PnL: {best['avg_pnl']:.2f} USDT
""")

        # 輸出 .env 配置
        print("  建議 .env 配置:")
        print("-" * 40)
        mode_str = 'breakout' if c['strategy_mode'] == 'breakout' else 'mean_reversion'
        print(f"  BOLLINGER_STRATEGY_MODE={mode_str}")
        print(f"  BOLLINGER_BB_PERIOD={c['bb_period']}")
        print(f"  BOLLINGER_BB_STD={c['bb_std']}")
        print(f"  BOLLINGER_USE_TREND_FILTER={'true' if c['use_trend_filter'] else 'false'}")
        print(f"  BOLLINGER_USE_BBW_FILTER={'true' if c['use_bbw_filter'] else 'false'}")
        print(f"  BOLLINGER_STOP_LOSS_PCT={c['stop_loss_pct']}")
        print(f"  BOLLINGER_ATR_MULTIPLIER={c['atr_mult']}")
    else:
        print("\n  ⚠️ 沒有找到通過驗證的穩健配置")
        print("  建議: 當前市場可能不適合 Bollinger 策略")

    print("\n" + "=" * 70)
    print("  優化完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
