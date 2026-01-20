#!/usr/bin/env python3
"""
Grid Futures Bot 完整回測 (2 年數據).

使用足夠長的時間和足夠多的交易來驗證策略。
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

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
    grid_level: int
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
    grid_rebuilds: int
    trades: List[Trade]
    equity_curve: List[float]


class GridFuturesBacktester:
    """Grid Futures 回測器"""

    def __init__(self, klines: List[Kline], config: Dict[str, Any]):
        self.klines = klines
        self.config = config
        self.leverage = config.get('leverage', 3)
        self.position_size = config.get('position_size', 0.1)
        self.fee_rate = 0.0004  # 0.04% taker fee

    def run(self) -> BacktestResult:
        """執行回測"""
        c = self.config
        grid_count = c.get('grid_count', 10)
        range_pct = c.get('range_pct', 0.08)  # 8% 默認範圍
        use_trend_filter = c.get('use_trend_filter', True)
        trend_period = c.get('trend_period', 20)
        use_atr_range = c.get('use_atr_range', True)
        atr_period = c.get('atr_period', 14)
        atr_multiplier = c.get('atr_multiplier', 2.0)
        stop_loss_pct = c.get('stop_loss_pct', 0.05)
        rebuild_threshold = c.get('rebuild_threshold', 0.02)
        direction = c.get('direction', 'trend_follow')  # long_only, short_only, neutral, trend_follow

        # State
        trades = []
        initial_capital = 10000.0
        equity = [initial_capital]
        grid_rebuilds = 0

        closes = [float(k.close) for k in self.klines]
        highs = [float(k.high) for k in self.klines]
        lows = [float(k.low) for k in self.klines]

        warmup = max(trend_period, atr_period) + 20

        # Grid state
        grid_center = None
        grid_upper = None
        grid_lower = None
        grid_levels = []  # List of (price, filled_long, filled_short)
        positions = []  # List of open positions

        def calculate_atr(idx):
            if idx < atr_period + 1:
                return None
            trs = []
            for j in range(idx - atr_period, idx):
                tr = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1])
                )
                trs.append(tr)
            return sum(trs) / atr_period

        def calculate_trend(idx):
            if not use_trend_filter or idx < trend_period:
                return 0
            sma = sum(closes[idx - trend_period:idx]) / trend_period
            price = closes[idx]
            diff_pct = (price - sma) / sma * 100
            if diff_pct > 1:
                return 1  # Uptrend
            elif diff_pct < -1:
                return -1  # Downtrend
            return 0

        def should_trade_direction(side, trend):
            if direction == 'long_only':
                return side == 'long'
            elif direction == 'short_only':
                return side == 'short'
            elif direction == 'neutral':
                return True
            elif direction == 'trend_follow':
                if trend == 1:
                    return side == 'long'
                elif trend == -1:
                    return side == 'short'
                return True
            return True

        def init_grid(center_price, idx):
            nonlocal grid_center, grid_upper, grid_lower, grid_levels

            atr = calculate_atr(idx)

            if use_atr_range and atr:
                range_val = atr * atr_multiplier
                actual_range = range_val / center_price
            else:
                actual_range = range_pct

            grid_center = center_price
            grid_upper = center_price * (1 + actual_range / 2)
            grid_lower = center_price * (1 - actual_range / 2)

            step = (grid_upper - grid_lower) / grid_count
            grid_levels = []
            for i in range(grid_count + 1):
                level_price = grid_lower + step * i
                grid_levels.append({
                    'price': level_price,
                    'filled_long': False,
                    'filled_short': False,
                })

        def check_grid_rebuild(price, idx):
            nonlocal grid_rebuilds
            if grid_center is None:
                return True

            distance = abs(price - grid_center) / grid_center
            if distance > rebuild_threshold:
                grid_rebuilds += 1
                return True
            return False

        for i in range(warmup, len(self.klines)):
            price = closes[i]
            kline = self.klines[i]
            high = highs[i]
            low = lows[i]
            trend = calculate_trend(i)

            # Initialize or rebuild grid
            if grid_center is None or check_grid_rebuild(price, i):
                # Close all positions before rebuild
                for pos in positions:
                    entry_price = pos['entry']
                    if pos['side'] == 'long':
                        pnl_pct = (price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - price) / entry_price

                    position_value = equity[-1] * self.position_size / grid_count
                    gross_pnl = pnl_pct * self.leverage * position_value
                    fee = position_value * self.fee_rate * 2
                    net_pnl = gross_pnl - fee

                    trades.append(Trade(
                        entry_time=pos['entry_time'],
                        exit_time=kline.close_time,
                        side=pos['side'],
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        grid_level=pos['level'],
                        exit_reason='grid_rebuild',
                    ))
                    equity.append(equity[-1] + net_pnl)

                positions = []
                init_grid(price, i)
                continue

            # Check stop loss for existing positions
            for pos in positions[:]:
                entry_price = pos['entry']
                if pos['side'] == 'long':
                    pnl_pct = (price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - price) / entry_price

                if pnl_pct <= -stop_loss_pct:
                    position_value = equity[-1] * self.position_size / grid_count
                    gross_pnl = pnl_pct * self.leverage * position_value
                    fee = position_value * self.fee_rate * 2
                    net_pnl = gross_pnl - fee

                    trades.append(Trade(
                        entry_time=pos['entry_time'],
                        exit_time=kline.close_time,
                        side=pos['side'],
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        grid_level=pos['level'],
                        exit_reason='stop_loss',
                    ))
                    equity.append(equity[-1] + net_pnl)
                    positions.remove(pos)

                    # Reset grid level
                    level = pos['level']
                    if 0 <= level < len(grid_levels):
                        grid_levels[level]['filled_long'] = False
                        grid_levels[level]['filled_short'] = False

            # Check grid levels for entry/exit
            for level_idx, level in enumerate(grid_levels):
                level_price = level['price']

                # Price crossed this level (check both directions using high/low)
                crossed_up = low <= level_price <= high
                crossed_down = low <= level_price <= high

                if crossed_up or crossed_down:
                    # Check for exit (take profit)
                    for pos in positions[:]:
                        if pos['level'] == level_idx:
                            continue  # Same level

                        # Long position exits at higher level
                        if pos['side'] == 'long' and level_idx > pos['level']:
                            if level_price > pos['entry']:
                                entry_price = pos['entry']
                                pnl_pct = (level_price - entry_price) / entry_price

                                position_value = equity[-1] * self.position_size / grid_count
                                gross_pnl = pnl_pct * self.leverage * position_value
                                fee = position_value * self.fee_rate * 2
                                net_pnl = gross_pnl - fee

                                trades.append(Trade(
                                    entry_time=pos['entry_time'],
                                    exit_time=kline.close_time,
                                    side='long',
                                    entry_price=entry_price,
                                    exit_price=level_price,
                                    pnl=net_pnl,
                                    pnl_pct=pnl_pct * 100,
                                    grid_level=pos['level'],
                                    exit_reason='take_profit',
                                ))
                                equity.append(equity[-1] + net_pnl)
                                positions.remove(pos)

                                # Reset original level
                                orig_level = pos['level']
                                if 0 <= orig_level < len(grid_levels):
                                    grid_levels[orig_level]['filled_long'] = False

                        # Short position exits at lower level
                        elif pos['side'] == 'short' and level_idx < pos['level']:
                            if level_price < pos['entry']:
                                entry_price = pos['entry']
                                pnl_pct = (entry_price - level_price) / entry_price

                                position_value = equity[-1] * self.position_size / grid_count
                                gross_pnl = pnl_pct * self.leverage * position_value
                                fee = position_value * self.fee_rate * 2
                                net_pnl = gross_pnl - fee

                                trades.append(Trade(
                                    entry_time=pos['entry_time'],
                                    exit_time=kline.close_time,
                                    side='short',
                                    entry_price=entry_price,
                                    exit_price=level_price,
                                    pnl=net_pnl,
                                    pnl_pct=pnl_pct * 100,
                                    grid_level=pos['level'],
                                    exit_reason='take_profit',
                                ))
                                equity.append(equity[-1] + net_pnl)
                                positions.remove(pos)

                                orig_level = pos['level']
                                if 0 <= orig_level < len(grid_levels):
                                    grid_levels[orig_level]['filled_short'] = False

                    # Check for entry
                    max_positions = grid_count // 2

                    # Long entry (price dropped to this level)
                    if not level['filled_long'] and len([p for p in positions if p['side'] == 'long']) < max_positions:
                        if should_trade_direction('long', trend):
                            level['filled_long'] = True
                            positions.append({
                                'side': 'long',
                                'entry': level_price,
                                'entry_time': kline.close_time,
                                'level': level_idx,
                            })

                    # Short entry (price rose to this level)
                    if not level['filled_short'] and len([p for p in positions if p['side'] == 'short']) < max_positions:
                        if should_trade_direction('short', trend):
                            level['filled_short'] = True
                            positions.append({
                                'side': 'short',
                                'entry': level_price,
                                'entry_time': kline.close_time,
                                'level': level_idx,
                            })

        # Close remaining positions
        price = closes[-1]
        for pos in positions:
            entry_price = pos['entry']
            if pos['side'] == 'long':
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price

            position_value = equity[-1] * self.position_size / grid_count
            gross_pnl = pnl_pct * self.leverage * position_value
            fee = position_value * self.fee_rate * 2
            net_pnl = gross_pnl - fee

            trades.append(Trade(
                entry_time=pos['entry_time'],
                exit_time=self.klines[-1].close_time,
                side=pos['side'],
                entry_price=entry_price,
                exit_price=price,
                pnl=net_pnl,
                pnl_pct=pnl_pct * 100,
                grid_level=pos['level'],
                exit_reason='end',
            ))
            equity.append(equity[-1] + net_pnl)

        # Calculate metrics
        if not trades:
            return BacktestResult(
                total_pnl=0, total_return_pct=0, win_rate=0, total_trades=0,
                long_trades=0, short_trades=0, avg_win=0, avg_loss=0,
                profit_factor=0, sharpe_ratio=0, max_drawdown_pct=0,
                grid_rebuilds=grid_rebuilds, trades=[], equity_curve=equity
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
        returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity)) if equity[i-1] > 0]
        if returns:
            avg_ret = sum(returns) / len(returns)
            std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5
            # Approximate trades per year
            days = (self.klines[-1].close_time - self.klines[0].close_time).days
            trades_per_year = len(trades) / (days / 365) if days > 0 else len(trades)
            sharpe = (avg_ret * trades_per_year) / (std_ret * (trades_per_year ** 0.5)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

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
            grid_rebuilds=grid_rebuilds,
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
                interval="1h",  # Grid 用 1h
                start_time=current_ts,
                limit=1500,
            )
            if not batch:
                break
            klines.extend(batch)
            current_ts = int(batch[-1].close_time.timestamp() * 1000) + 1
            batch_count += 1
            if batch_count % 5 == 0:
                print(f"  已獲取 {len(klines)} 根 K 線...")

        return klines
    finally:
        await client.close()


async def main():
    print("=" * 70)
    print("       Grid Futures Bot 完整回測 (2 年數據)")
    print("=" * 70)

    # 獲取數據
    print("\n正在獲取 BTCUSDT 1h 歷史數據 (2 年)...")
    klines = await fetch_data(days=730)

    print(f"\n數據摘要:")
    print(f"  K 線數量: {len(klines):,}")
    print(f"  時間範圍: {klines[0].open_time.date()} ~ {klines[-1].close_time.date()}")
    closes = [float(k.close) for k in klines]
    print(f"  價格範圍: ${min(closes):,.0f} ~ ${max(closes):,.0f}")
    print(f"  價格變化: {(closes[-1]/closes[0]-1)*100:+.1f}%")

    # 測試配置
    base_config = {
        'leverage': 3,
        'position_size': 0.1,
        'grid_count': 10,
        'stop_loss_pct': 0.05,
        'rebuild_threshold': 0.02,
    }

    configs = {
        "當前配置 (趨勢跟蹤)": {
            **base_config,
            'direction': 'trend_follow',
            'use_trend_filter': True,
            'trend_period': 20,
            'use_atr_range': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'range_pct': 0.08,
        },
        "中性策略 (雙向)": {
            **base_config,
            'direction': 'neutral',
            'use_trend_filter': False,
            'trend_period': 20,
            'use_atr_range': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'range_pct': 0.08,
        },
        "純多策略": {
            **base_config,
            'direction': 'long_only',
            'use_trend_filter': True,
            'trend_period': 20,
            'use_atr_range': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'range_pct': 0.08,
        },
        "寬網格 (15格)": {
            **base_config,
            'grid_count': 15,
            'direction': 'trend_follow',
            'use_trend_filter': True,
            'trend_period': 20,
            'use_atr_range': True,
            'atr_period': 14,
            'atr_multiplier': 2.5,
            'range_pct': 0.10,
        },
        "高槓桿 (5x)": {
            **base_config,
            'leverage': 5,
            'direction': 'trend_follow',
            'use_trend_filter': True,
            'trend_period': 20,
            'use_atr_range': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'range_pct': 0.08,
        },
        "無趨勢過濾": {
            **base_config,
            'direction': 'trend_follow',
            'use_trend_filter': False,
            'trend_period': 20,
            'use_atr_range': True,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'range_pct': 0.08,
        },
    }

    print("\n" + "=" * 70)
    print("  回測結果比較")
    print("=" * 70)

    results = {}
    for name, config in configs.items():
        print(f"\n正在測試: {name}...")
        bt = GridFuturesBacktester(klines, config)
        result = bt.run()
        results[name] = result

    # 顯示結果
    print("\n" + "=" * 120)
    print(f"{'策略':<20} {'總報酬%':>10} {'交易數':>8} {'勝率':>8} {'獲利因子':>10} {'Sharpe':>8} {'最大回撤':>10} {'網格重建':>8}")
    print("-" * 120)

    for name, r in results.items():
        print(f"{name:<20} {r.total_return_pct:>+9.1f}% {r.total_trades:>8} {r.win_rate:>7.1f}% {r.profit_factor:>10.2f} {r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>9.1f}% {r.grid_rebuilds:>8}")

    print("=" * 120)

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
    print(f"  網格重建次數: {best.grid_rebuilds}")

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
