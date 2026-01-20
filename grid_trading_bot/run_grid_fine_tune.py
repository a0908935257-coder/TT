#!/usr/bin/env python3
"""
Grid Bot Fine-Tuning - Optimize for better risk-adjusted returns.

目標：年化 >30%，回撤 <40%，Sharpe >1.0
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional
from enum import Enum

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


class GridDirection(str, Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    NEUTRAL = "neutral"
    TREND_FOLLOW = "trend_follow"


@dataclass
class GridConfig:
    grid_count: int = 15
    range_pct: Decimal = field(default_factory=lambda: Decimal("0.08"))
    position_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    leverage: int = 5
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))
    use_trend_filter: bool = True
    trend_period: int = 50
    direction: GridDirection = GridDirection.TREND_FOLLOW
    use_atr_range: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))


@dataclass
class BacktestResult:
    initial_capital: Decimal = field(default_factory=lambda: Decimal("0"))
    final_equity: Decimal = field(default_factory=lambda: Decimal("0"))
    total_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    annual_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    excess_return: Decimal = field(default_factory=lambda: Decimal("0"))


class GridBacktest:
    def __init__(self, klines: List[Kline], initial_capital: Decimal, config: GridConfig):
        self._klines = klines
        self._initial_capital = initial_capital
        self._config = config
        self._capital = initial_capital
        self._position = Decimal("0")
        self._avg_entry_price = Decimal("0")
        self._grids: List[dict] = []
        self._upper_price = Decimal("0")
        self._lower_price = Decimal("0")
        self._current_trend = 0
        self._trades: List[dict] = []
        self._daily_returns: List[Decimal] = []

    def _calculate_sma(self, prices: List[Decimal], period: int) -> Optional[Decimal]:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / Decimal(period)

    def _calculate_atr(self, klines: List[Kline], period: int) -> Optional[Decimal]:
        if len(klines) < period + 1:
            return None
        tr_values = []
        for i in range(1, min(len(klines), period + 1)):
            high, low = klines[-i].high, klines[-i].low
            prev_close = klines[-i-1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        return sum(tr_values) / Decimal(len(tr_values)) if tr_values else None

    def _determine_trend(self, closes: List[Decimal], current_price: Decimal) -> int:
        if not self._config.use_trend_filter:
            return 0
        sma = self._calculate_sma(closes, self._config.trend_period)
        if sma is None:
            return 0
        diff_pct = (current_price - sma) / sma * Decimal("100")
        if diff_pct > Decimal("1"):
            return 1
        elif diff_pct < Decimal("-1"):
            return -1
        return 0

    def _should_trade_direction(self, direction: str) -> bool:
        if self._config.direction == GridDirection.LONG_ONLY:
            return direction == "long"
        elif self._config.direction == GridDirection.SHORT_ONLY:
            return direction == "short"
        elif self._config.direction == GridDirection.NEUTRAL:
            return True
        elif self._config.direction == GridDirection.TREND_FOLLOW:
            if self._current_trend == 1:
                return direction == "long"
            elif self._current_trend == -1:
                return direction == "short"
            return True
        return True

    def _initialize_grid(self, center_price: Decimal, atr: Optional[Decimal] = None) -> None:
        if self._config.use_atr_range and atr:
            range_value = atr * self._config.atr_multiplier
            range_pct = range_value / center_price
            range_pct = min(max(range_pct, Decimal("0.03")), Decimal("0.20"))
        else:
            range_pct = self._config.range_pct
        self._upper_price = center_price * (Decimal("1") + range_pct)
        self._lower_price = center_price * (Decimal("1") - range_pct)
        self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)
        self._grids = [{"price": self._lower_price + Decimal(i) * self._grid_spacing, "long_filled": False, "short_filled": False} for i in range(self._config.grid_count + 1)]

    def _check_rebuild_needed(self, current_price: Decimal) -> bool:
        return current_price > self._upper_price * Decimal("1.02") or current_price < self._lower_price * Decimal("0.98")

    def run(self) -> BacktestResult:
        if not self._klines or len(self._klines) < 100:
            return BacktestResult()
        closes: List[Decimal] = []
        first_price = self._klines[0].close
        self._initialize_grid(first_price)
        max_equity = self._initial_capital
        max_drawdown = Decimal("0")
        prev_equity = self._initial_capital

        for i, kline in enumerate(self._klines):
            current_price = kline.close
            closes.append(current_price)
            self._current_trend = self._determine_trend(closes, current_price)
            atr = self._calculate_atr(self._klines[:i+1], self._config.atr_period) if i > self._config.atr_period else None

            if self._check_rebuild_needed(current_price):
                if self._position != Decimal("0"):
                    self._close_position(current_price)
                self._initialize_grid(current_price, atr)

            self._process_grids(current_price)
            unrealized_pnl = Decimal("0")
            if self._position > 0:
                unrealized_pnl = self._position * (current_price - self._avg_entry_price) * Decimal(self._config.leverage)
            elif self._position < 0:
                unrealized_pnl = abs(self._position) * (self._avg_entry_price - current_price) * Decimal(self._config.leverage)
            current_equity = self._capital + unrealized_pnl

            if prev_equity > 0:
                self._daily_returns.append((current_equity - prev_equity) / prev_equity)
            prev_equity = current_equity
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = max_equity - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        if self._position != Decimal("0"):
            self._close_position(self._klines[-1].close)

        return self._calculate_result(max_drawdown, max_equity)

    def _close_position(self, current_price: Decimal) -> None:
        if self._position == Decimal("0"):
            return
        if self._position > 0:
            pnl = self._position * (current_price - self._avg_entry_price) * Decimal(self._config.leverage)
        else:
            pnl = abs(self._position) * (self._avg_entry_price - current_price) * Decimal(self._config.leverage)
        fee = abs(self._position) * current_price * self._config.fee_rate
        self._capital += pnl - fee
        self._trades.append({"pnl": pnl - fee})
        self._position = Decimal("0")

    def _process_grids(self, current_price: Decimal) -> None:
        max_position_value = self._capital * self._config.max_position_pct
        trade_value = self._capital * self._config.position_pct

        for grid in self._grids:
            grid_price = grid["price"]
            if current_price <= grid_price and not grid["long_filled"]:
                if self._should_trade_direction("long"):
                    current_pos_val = abs(self._position) * current_price if self._position else Decimal("0")
                    if current_pos_val < max_position_value and trade_value > Decimal("10"):
                        self._open_long(current_price, trade_value)
                        grid["long_filled"] = True
                        grid["short_filled"] = False
            elif current_price >= grid_price:
                if grid["long_filled"] and self._position > 0:
                    self._close_partial_long(current_price, trade_value)
                    grid["long_filled"] = False
                if not grid["short_filled"] and self._should_trade_direction("short"):
                    current_pos_val = abs(self._position) * current_price if self._position else Decimal("0")
                    if current_pos_val < max_position_value and trade_value > Decimal("10"):
                        self._open_short(current_price, trade_value)
                        grid["short_filled"] = True
                        grid["long_filled"] = False
            if current_price <= grid_price and grid["short_filled"] and self._position < 0:
                self._close_partial_short(current_price, trade_value)
                grid["short_filled"] = False

    def _open_long(self, price: Decimal, value: Decimal) -> None:
        quantity = value / price
        fee = value * self._config.fee_rate
        if self._position >= 0:
            if self._position > 0:
                total_value = self._position * self._avg_entry_price + quantity * price
                self._position += quantity
                self._avg_entry_price = total_value / self._position
            else:
                self._position = quantity
                self._avg_entry_price = price
        else:
            self._close_position(price)
            self._position = quantity
            self._avg_entry_price = price
        self._capital -= fee

    def _open_short(self, price: Decimal, value: Decimal) -> None:
        quantity = value / price
        fee = value * self._config.fee_rate
        if self._position <= 0:
            if self._position < 0:
                total_value = abs(self._position) * self._avg_entry_price + quantity * price
                self._position -= quantity
                self._avg_entry_price = total_value / abs(self._position)
            else:
                self._position = -quantity
                self._avg_entry_price = price
        else:
            self._close_position(price)
            self._position = -quantity
            self._avg_entry_price = price
        self._capital -= fee

    def _close_partial_long(self, price: Decimal, value: Decimal) -> None:
        if self._position <= 0:
            return
        quantity = min(self._position, value / price)
        pnl = quantity * (price - self._avg_entry_price) * Decimal(self._config.leverage)
        fee = quantity * price * self._config.fee_rate
        self._capital += pnl - fee
        self._position -= quantity
        self._trades.append({"pnl": pnl - fee})

    def _close_partial_short(self, price: Decimal, value: Decimal) -> None:
        if self._position >= 0:
            return
        quantity = min(abs(self._position), value / price)
        pnl = quantity * (self._avg_entry_price - price) * Decimal(self._config.leverage)
        fee = quantity * price * self._config.fee_rate
        self._capital += pnl - fee
        self._position += quantity
        self._trades.append({"pnl": pnl - fee})

    def _calculate_result(self, max_drawdown: Decimal, max_equity: Decimal) -> BacktestResult:
        result = BacktestResult()
        result.initial_capital = self._initial_capital
        result.final_equity = self._capital
        result.total_return_pct = (self._capital - self._initial_capital) / self._initial_capital * Decimal("100")
        days = len(self._klines) / 24
        result.annual_return_pct = result.total_return_pct / Decimal(str(days / 365)) if days > 0 else Decimal("0")
        result.total_trades = len(self._trades)

        profitable = [t for t in self._trades if t.get("pnl", 0) > 0]
        losing = [t for t in self._trades if t.get("pnl", 0) <= 0]
        result.win_rate = Decimal(len(profitable)) / Decimal(len(self._trades)) * Decimal("100") if self._trades else Decimal("0")

        gross_profit = sum(t.get("pnl", Decimal("0")) for t in profitable)
        gross_loss = abs(sum(t.get("pnl", Decimal("0")) for t in losing))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")
        result.max_drawdown_pct = (max_drawdown / max_equity) * Decimal("100") if max_equity > 0 else Decimal("0")

        if self._klines:
            bh_return = ((self._klines[-1].close - self._klines[0].close) / self._klines[0].close) * Decimal("100")
            result.excess_return = result.total_return_pct - bh_return

        if len(self._daily_returns) > 10:
            returns = [float(r) for r in self._daily_returns]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            result.sharpe_ratio = Decimal(str(round((mean_return / std_dev) * math.sqrt(8760), 2))) if std_dev > 0 else Decimal("0")

        return result


async def fetch_klines(symbol: str, interval: str, days: int) -> List[Kline]:
    print(f"\n正在獲取 {symbol} {interval} 數據 ({days} 天)...")
    async with BinanceFuturesAPI() as api:
        await api.ping()
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        interval_map = {"1h": KlineInterval.h1, "4h": KlineInterval.h4, "15m": KlineInterval.m15}
        kline_interval = interval_map.get(interval, KlineInterval.h1)
        all_klines = []
        current_start = start_time
        while current_start < end_time:
            klines = await api.get_klines(symbol=symbol, interval=kline_interval, limit=1500, start_time=current_start, end_time=end_time)
            if not klines:
                break
            all_klines.extend(klines)
            if len(klines) < 1500:
                break
            current_start = int(klines[-1].close_time.timestamp() * 1000) + 1
        print(f"  獲取 {len(all_klines)} 根 K 線")
        return all_klines


async def run_fine_tuning():
    print("=" * 70)
    print("Grid Bot 參數微調 - 目標: 年化>30%, 回撤<40%, Sharpe>1.0")
    print("=" * 70)

    klines = await fetch_klines("BTCUSDT", "1h", 365)
    if not klines:
        return
    initial_capital = Decimal("10000")

    # Parameter grid search
    leverages = [3, 4, 5, 6]
    grid_counts = [10, 15, 20]
    trend_periods = [30, 50, 70]
    atr_multipliers = [Decimal("1.5"), Decimal("2.0"), Decimal("2.5")]
    position_pcts = [Decimal("0.06"), Decimal("0.08"), Decimal("0.10")]

    results = []
    total_tests = len(leverages) * len(grid_counts) * len(trend_periods) * len(atr_multipliers) * len(position_pcts)
    print(f"\n測試 {total_tests} 種配置組合...\n")

    test_num = 0
    for leverage in leverages:
        for grid_count in grid_counts:
            for trend_period in trend_periods:
                for atr_mult in atr_multipliers:
                    for pos_pct in position_pcts:
                        test_num += 1
                        config = GridConfig(
                            leverage=leverage,
                            grid_count=grid_count,
                            trend_period=trend_period,
                            atr_multiplier=atr_mult,
                            position_pct=pos_pct,
                            use_trend_filter=True,
                            use_atr_range=True,
                            direction=GridDirection.TREND_FOLLOW,
                        )
                        backtest = GridBacktest(klines, initial_capital, config)
                        result = backtest.run()
                        results.append((config, result))

                        if test_num % 50 == 0:
                            print(f"  進度: {test_num}/{total_tests}")

    # Filter by targets
    target_configs = [
        (c, r) for c, r in results
        if float(r.annual_return_pct) >= 30
        and float(r.max_drawdown_pct) <= 50
        and float(r.sharpe_ratio) >= 0.8
    ]

    print(f"\n找到 {len(target_configs)} 個達標配置 (年化≥30%, 回撤≤50%, Sharpe≥0.8)")

    if target_configs:
        # Sort by Sharpe ratio
        target_configs.sort(key=lambda x: float(x[1].sharpe_ratio), reverse=True)

        print("\n" + "=" * 90)
        print("Top 10 最佳配置 (按 Sharpe 排序)")
        print("=" * 90)
        print(f"{'#':<3} {'槓桿':>4} {'網格':>4} {'趨勢':>4} {'ATR':>4} {'倉位':>5} {'年化':>8} {'回撤':>7} {'Sharpe':>7} {'勝率':>6}")
        print("-" * 90)

        for i, (config, result) in enumerate(target_configs[:10], 1):
            print(f"{i:<3} {config.leverage:>4}x {config.grid_count:>4} {config.trend_period:>4} {float(config.atr_multiplier):>4.1f} {float(config.position_pct)*100:>4.0f}% {float(result.annual_return_pct):>+7.1f}% {float(result.max_drawdown_pct):>6.1f}% {float(result.sharpe_ratio):>7.2f} {float(result.win_rate):>5.1f}%")

        best_config, best_result = target_configs[0]
        print("\n" + "=" * 70)
        print("🏆 最佳配置")
        print("=" * 70)
        print(f"  槓桿:           {best_config.leverage}x")
        print(f"  網格數量:       {best_config.grid_count}")
        print(f"  趨勢週期:       {best_config.trend_period}")
        print(f"  ATR 乘數:       {float(best_config.atr_multiplier):.1f}")
        print(f"  倉位比例:       {float(best_config.position_pct)*100:.0f}%")
        print(f"\n  年化回報:       {float(best_result.annual_return_pct):+.1f}%")
        print(f"  最大回撤:       {float(best_result.max_drawdown_pct):.1f}%")
        print(f"  Sharpe Ratio:   {float(best_result.sharpe_ratio):.2f}")
        print(f"  勝率:           {float(best_result.win_rate):.1f}%")
        print(f"  獲利因子:       {float(best_result.profit_factor):.2f}")
        print(f"  超額收益:       {float(best_result.excess_return):+.1f}%")

        print(f"\n📝 建議 .env 配置:")
        print(f"  GRID_LEVERAGE={best_config.leverage}")
        print(f"  GRID_COUNT={best_config.grid_count}")
        print(f"  GRID_DIRECTION=trend_follow")
        print(f"  GRID_USE_TREND_FILTER=true")
        print(f"  GRID_TREND_PERIOD={best_config.trend_period}")
        print(f"  GRID_USE_ATR_RANGE=true")
        print(f"  GRID_ATR_MULTIPLIER={float(best_config.atr_multiplier):.1f}")
        print(f"  GRID_POSITION_SIZE={float(best_config.position_pct)}")
    else:
        # Find best overall
        best = max(results, key=lambda x: float(x[1].sharpe_ratio))
        print(f"\n⚠️  沒有配置完全達標")
        print(f"   最佳 Sharpe: {float(best[1].sharpe_ratio):.2f}")
        print(f"   年化: {float(best[1].annual_return_pct):.1f}%")
        print(f"   回撤: {float(best[1].max_drawdown_pct):.1f}%")

    print("\n微調完成!")


if __name__ == "__main__":
    asyncio.run(run_fine_tuning())
