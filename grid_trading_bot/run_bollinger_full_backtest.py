#!/usr/bin/env python3
"""
Bollinger Bot 完整回測 (2 年數據).

使用足夠長的時間和足夠多的交易來驗證策略。
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


class BollingerBacktester:
    """Bollinger Band 回測器"""

    def __init__(self, klines: List[Kline], config: Dict[str, Any]):
        self.klines = klines
        self.config = config
        self.leverage = config.get('leverage', 20)
        self.position_size = config.get('position_size', 0.1)  # 每次用 10% 資金
        self.fee_rate = 0.0004  # 0.04% taker fee

    def run(self) -> BacktestResult:
        """執行回測"""
        c = self.config
        bb_period = c.get('bb_period', 20)
        bb_std = c.get('bb_std', 2.0)
        strategy_mode = c.get('strategy_mode', 'breakout')
        use_trend_filter = c.get('use_trend_filter', False)
        trend_period = c.get('trend_period', 50)
        use_bbw_filter = c.get('use_bbw_filter', True)
        bbw_threshold = c.get('bbw_threshold', 20)
        bbw_lookback = c.get('bbw_lookback', 100)
        stop_loss_pct = c.get('stop_loss_pct', 0.015)
        take_profit_pct = c.get('take_profit_pct', 0.03)
        max_hold_bars = c.get('max_hold_bars', 48)
        use_atr_stop = c.get('use_atr_stop', True)
        atr_period = c.get('atr_period', 14)
        atr_mult = c.get('atr_mult', 2.0)

        # State
        position = None
        trades = []
        initial_capital = 10000.0
        equity = [initial_capital]

        closes = [float(k.close) for k in self.klines]
        highs = [float(k.high) for k in self.klines]
        lows = [float(k.low) for k in self.klines]

        warmup = max(bb_period, trend_period, bbw_lookback, atr_period) + 10

        for i in range(warmup, len(self.klines)):
            price = closes[i]
            kline = self.klines[i]

            # Bollinger Bands
            bb_closes = closes[i-bb_period:i]
            sma = sum(bb_closes) / bb_period
            variance = sum((x - sma) ** 2 for x in bb_closes) / bb_period
            std = variance ** 0.5
            upper = sma + bb_std * std
            lower = sma - bb_std * std

            # BBW
            bbw = (upper - lower) / sma * 100 if sma > 0 else 0

            # BBW percentile
            bbw_history = []
            for j in range(max(warmup, i - bbw_lookback), i):
                bb_c = closes[j-bb_period:j]
                if len(bb_c) == bb_period:
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
                trend = 0

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

            # Check exit
            if position is not None:
                bars_held = i - position['entry_bar']
                entry_price = position['entry']

                if position['side'] == 'long':
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - price) / entry_price

                if use_atr_stop and atr > 0:
                    stop_distance = atr * atr_mult / entry_price
                else:
                    stop_distance = stop_loss_pct

                exit_reason = None

                if pnl_pct <= -stop_distance:
                    exit_reason = 'stop_loss'
                elif pnl_pct >= take_profit_pct:
                    exit_reason = 'take_profit'
                elif bars_held >= max_hold_bars:
                    exit_reason = 'timeout'
                elif strategy_mode == 'mean_reversion':
                    if position['side'] == 'long' and price >= sma:
                        exit_reason = 'target'
                    elif position['side'] == 'short' and price <= sma:
                        exit_reason = 'target'
                elif strategy_mode == 'breakout':
                    if position['side'] == 'long' and price < sma:
                        exit_reason = 'reversal'
                    elif position['side'] == 'short' and price > sma:
                        exit_reason = 'reversal'

                if exit_reason:
                    # 用 position_size 比例的資金開倉
                    position_value = equity[-1] * self.position_size
                    gross_pnl = pnl_pct * self.leverage * position_value
                    fee = position_value * self.fee_rate * 2  # 開倉 + 平倉手續費
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

            # Check entry
            if position is None:
                if use_bbw_filter and bbw_percentile < bbw_threshold:
                    continue

                signal = None

                if strategy_mode == 'breakout':
                    if price > upper:
                        if not use_trend_filter or trend >= 0:
                            signal = 'long'
                    elif price < lower:
                        if not use_trend_filter or trend <= 0:
                            signal = 'short'
                elif strategy_mode == 'mean_reversion':
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
                        'entry_time': kline.close_time,
                    }

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
    print("       Bollinger Bot 完整回測 (2 年數據)")
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

    # 測試配置 (全部使用 10% 倉位大小)
    base_config = {
        'bb_period': 20,
        'use_bbw_filter': True,
        'bbw_threshold': 20,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'max_hold_bars': 48,
        'use_atr_stop': True,
        'atr_period': 14,
        'leverage': 20,
        'position_size': 0.1,  # 每次用 10% 資金
    }

    configs = {
        "當前配置 (BB=2.0, 無趨勢)": {
            **base_config,
            'strategy_mode': 'breakout',
            'bb_std': 2.0,
            'use_trend_filter': False,
            'trend_period': 50,
            'bbw_lookback': 100,
            'atr_mult': 2.0,
        },
        "舊配置 (BB=3.25)": {
            **base_config,
            'strategy_mode': 'breakout',
            'bb_std': 3.25,
            'use_trend_filter': False,
            'trend_period': 50,
            'bbw_lookback': 200,
            'atr_mult': 2.0,
        },
        "均值回歸 (BB=2.0)": {
            **base_config,
            'strategy_mode': 'mean_reversion',
            'bb_std': 2.0,
            'use_trend_filter': False,
            'trend_period': 50,
            'bbw_lookback': 100,
            'atr_mult': 2.0,
        },
        "突破+趨勢過濾": {
            **base_config,
            'strategy_mode': 'breakout',
            'bb_std': 2.5,
            'use_trend_filter': True,
            'trend_period': 50,
            'bbw_lookback': 100,
            'atr_mult': 2.0,
        },
        "低槓桿穩健 (10x)": {
            **base_config,
            'strategy_mode': 'breakout',
            'bb_std': 2.0,
            'use_trend_filter': False,
            'trend_period': 50,
            'bbw_lookback': 100,
            'atr_mult': 2.5,
            'leverage': 10,
        },
    }

    print("\n" + "=" * 70)
    print("  回測結果比較")
    print("=" * 70)

    results = {}
    for name, config in configs.items():
        print(f"\n正在測試: {name}...")
        bt = BollingerBacktester(klines, config)
        result = bt.run()
        results[name] = result

    # 顯示結果
    print("\n" + "=" * 100)
    print(f"{'策略':<25} {'總報酬%':>10} {'交易數':>8} {'勝率':>8} {'獲利因子':>10} {'Sharpe':>8} {'最大回撤':>10} {'平均持倉':>8}")
    print("-" * 100)

    for name, r in results.items():
        print(f"{name:<25} {r.total_return_pct:>+9.1f}% {r.total_trades:>8} {r.win_rate:>7.1f}% {r.profit_factor:>10.2f} {r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>9.1f}% {r.avg_hold_hours:>7.1f}h")

    print("=" * 100)

    # 最佳策略詳情
    best_name = max(results, key=lambda x: results[x].sharpe_ratio)
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

    print("\n" + "=" * 70)
    print("  回測完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
