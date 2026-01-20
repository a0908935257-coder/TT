#!/usr/bin/env python3
"""
Grid Futures Bot Overfitting Validation.

防止過度擬合的驗證方法：
1. 樣本外測試 (Out-of-Sample)
2. 走步前進分析 (Walk-Forward)
3. 參數敏感度分析 (Parameter Sensitivity)
4. 不同市場狀態測試 (Market Regime)

Usage:
    python run_grid_futures_overfitting.py
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Tuple
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
    leverage: int = 3
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))
    use_trend_filter: bool = True
    trend_period: int = 30
    direction: GridDirection = GridDirection.TREND_FOLLOW
    use_atr_range: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))


@dataclass
class BacktestResult:
    total_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    annual_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))


class GridBacktest:
    """Simplified grid backtest for validation."""

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
        self._closes: List[Decimal] = []

    def _calculate_sma(self, period: int) -> Optional[Decimal]:
        if len(self._closes) < period:
            return None
        return sum(self._closes[-period:]) / Decimal(period)

    def _calculate_atr(self, end_idx: int) -> Optional[Decimal]:
        period = self._config.atr_period
        if end_idx < period + 1:
            return None
        tr_values = []
        for i in range(period):
            idx = end_idx - i
            if idx < 1:
                break
            high = self._klines[idx].high
            low = self._klines[idx].low
            prev_close = self._klines[idx - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        return sum(tr_values) / Decimal(len(tr_values)) if tr_values else None

    def _determine_trend(self, current_price: Decimal) -> int:
        if not self._config.use_trend_filter:
            return 0
        sma = self._calculate_sma(self._config.trend_period)
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
        spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)
        self._grids = [{"price": self._lower_price + Decimal(i) * spacing, "long_filled": False, "short_filled": False}
                       for i in range(self._config.grid_count + 1)]

    def _check_rebuild_needed(self, current_price: Decimal) -> bool:
        return current_price > self._upper_price * Decimal("1.02") or current_price < self._lower_price * Decimal("0.98")

    def run(self) -> BacktestResult:
        if not self._klines or len(self._klines) < 100:
            return BacktestResult()

        first_price = self._klines[0].close
        self._initialize_grid(first_price)
        max_equity = self._initial_capital
        max_drawdown = Decimal("0")
        prev_equity = self._initial_capital

        for i, kline in enumerate(self._klines):
            current_price = kline.close
            self._closes.append(current_price)
            self._current_trend = self._determine_trend(current_price)
            atr = self._calculate_atr(i) if i > self._config.atr_period else None

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

        if len(self._daily_returns) > 10:
            returns = [float(r) for r in self._daily_returns]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0
            std_dev = math.sqrt(variance) if variance > 0 else 0
            result.sharpe_ratio = Decimal(str(round((mean_return / std_dev) * math.sqrt(8760), 2))) if std_dev > 0 else Decimal("0")

        return result


async def fetch_klines(symbol: str, interval: str, days: int, end_date: Optional[datetime] = None) -> List[Kline]:
    """Fetch historical klines."""
    async with BinanceFuturesAPI() as api:
        await api.ping()

        if end_date:
            end_time = int(end_date.timestamp() * 1000)
        else:
            end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        interval_map = {"1h": KlineInterval.h1, "4h": KlineInterval.h4}
        kline_interval = interval_map.get(interval, KlineInterval.h1)

        all_klines = []
        current_start = start_time
        while current_start < end_time:
            klines = await api.get_klines(symbol=symbol, interval=kline_interval, limit=1500,
                                          start_time=current_start, end_time=end_time)
            if not klines:
                break
            all_klines.extend(klines)
            if len(klines) < 1500:
                break
            current_start = int(klines[-1].close_time.timestamp() * 1000) + 1

        return all_klines


def run_backtest(klines: List[Kline], config: GridConfig) -> BacktestResult:
    """Run single backtest."""
    backtest = GridBacktest(klines, Decimal("10000"), config)
    return backtest.run()


async def test_out_of_sample():
    """
    Test 1: Out-of-Sample Testing
    訓練期: 前 70% 數據
    測試期: 後 30% 數據
    """
    print("\n" + "=" * 70)
    print("測試 1: 樣本外測試 (Out-of-Sample)")
    print("=" * 70)

    klines = await fetch_klines("BTCUSDT", "1h", 730)  # 2 years
    if len(klines) < 500:
        print("數據不足")
        return None

    split_idx = int(len(klines) * 0.7)
    train_klines = klines[:split_idx]
    test_klines = klines[split_idx:]

    print(f"訓練期: {len(train_klines)} 根 K 線 ({len(train_klines)//24} 天)")
    print(f"測試期: {len(test_klines)} 根 K 線 ({len(test_klines)//24} 天)")

    # Optimized config
    config = GridConfig(
        leverage=3,
        grid_count=15,
        trend_period=30,
        atr_multiplier=Decimal("2.0"),
        direction=GridDirection.TREND_FOLLOW,
        use_trend_filter=True,
        use_atr_range=True,
    )

    train_result = run_backtest(train_klines, config)
    test_result = run_backtest(test_klines, config)

    print(f"\n{'指標':<20} {'訓練期':>15} {'測試期':>15} {'衰減':>12}")
    print("-" * 70)
    print(f"{'年化回報':<20} {float(train_result.annual_return_pct):>+14.1f}% {float(test_result.annual_return_pct):>+14.1f}% {float(test_result.annual_return_pct - train_result.annual_return_pct):>+11.1f}%")
    print(f"{'最大回撤':<20} {float(train_result.max_drawdown_pct):>14.1f}% {float(test_result.max_drawdown_pct):>14.1f}%")
    print(f"{'Sharpe Ratio':<20} {float(train_result.sharpe_ratio):>15.2f} {float(test_result.sharpe_ratio):>15.2f}")
    print(f"{'勝率':<20} {float(train_result.win_rate):>14.1f}% {float(test_result.win_rate):>14.1f}%")

    degradation = float(train_result.annual_return_pct - test_result.annual_return_pct)
    if degradation < 30 and float(test_result.annual_return_pct) > 0:
        print(f"\n✅ 樣本外測試通過: 衰減 {degradation:.1f}% (< 30%)")
        passed = True
    else:
        print(f"\n⚠️  樣本外測試警告: 衰減 {degradation:.1f}%")
        passed = False

    return {"train": train_result, "test": test_result, "passed": passed, "degradation": degradation}


async def test_walk_forward():
    """
    Test 2: Walk-Forward Analysis
    將數據分成多個滾動窗口測試
    """
    print("\n" + "=" * 70)
    print("測試 2: 走步前進分析 (Walk-Forward)")
    print("=" * 70)

    klines = await fetch_klines("BTCUSDT", "1h", 730)  # 2 years
    if len(klines) < 500:
        print("數據不足")
        return None

    # 6 periods of ~4 months each
    period_size = len(klines) // 6
    results = []

    config = GridConfig(
        leverage=3,
        grid_count=15,
        trend_period=30,
        atr_multiplier=Decimal("2.0"),
        direction=GridDirection.TREND_FOLLOW,
        use_trend_filter=True,
        use_atr_range=True,
    )

    print(f"\n{'期間':<8} {'起始日期':<12} {'年化':>10} {'回撤':>10} {'Sharpe':>10} {'結果':>8}")
    print("-" * 70)

    for i in range(6):
        start_idx = i * period_size
        end_idx = min((i + 1) * period_size, len(klines))
        period_klines = klines[start_idx:end_idx]

        if len(period_klines) < 100:
            continue

        result = run_backtest(period_klines, config)
        results.append(result)

        start_date = period_klines[0].open_time.strftime("%Y-%m")
        annual = float(result.annual_return_pct)
        status = "✅" if annual > 0 else "❌"

        print(f"期間 {i+1:<3} {start_date:<12} {annual:>+9.1f}% {float(result.max_drawdown_pct):>9.1f}% {float(result.sharpe_ratio):>10.2f} {status:>8}")

    # Summary
    profitable_periods = sum(1 for r in results if float(r.annual_return_pct) > 0)
    avg_annual = sum(float(r.annual_return_pct) for r in results) / len(results) if results else 0
    avg_sharpe = sum(float(r.sharpe_ratio) for r in results) / len(results) if results else 0

    print("-" * 70)
    print(f"{'平均':<8} {'':<12} {avg_annual:>+9.1f}% {'':<10} {avg_sharpe:>10.2f}")
    print(f"\n獲利期間: {profitable_periods}/{len(results)}")

    if profitable_periods >= len(results) * 0.6:  # 60% profitable
        print(f"✅ 走步前進測試通過: {profitable_periods}/{len(results)} 期間獲利")
        passed = True
    else:
        print(f"⚠️  走步前進測試警告: 僅 {profitable_periods}/{len(results)} 期間獲利")
        passed = False

    return {"results": results, "profitable_periods": profitable_periods, "passed": passed}


async def test_parameter_sensitivity():
    """
    Test 3: Parameter Sensitivity Analysis
    測試參數微調對結果的影響
    """
    print("\n" + "=" * 70)
    print("測試 3: 參數敏感度分析 (Parameter Sensitivity)")
    print("=" * 70)

    klines = await fetch_klines("BTCUSDT", "1h", 365)
    if len(klines) < 500:
        print("數據不足")
        return None

    # Base config (optimized)
    base_config = GridConfig(
        leverage=3,
        grid_count=15,
        trend_period=30,
        atr_multiplier=Decimal("2.0"),
    )
    base_result = run_backtest(klines, base_config)

    print(f"\n基準配置: leverage=3, grid=15, trend=30, atr_mult=2.0")
    print(f"基準年化: {float(base_result.annual_return_pct):+.1f}%")
    print()

    # Test variations
    variations = [
        # Leverage variations
        ("槓桿=2", GridConfig(leverage=2, grid_count=15, trend_period=30, atr_multiplier=Decimal("2.0"))),
        ("槓桿=4", GridConfig(leverage=4, grid_count=15, trend_period=30, atr_multiplier=Decimal("2.0"))),
        ("槓桿=5", GridConfig(leverage=5, grid_count=15, trend_period=30, atr_multiplier=Decimal("2.0"))),
        # Grid count variations
        ("網格=10", GridConfig(leverage=3, grid_count=10, trend_period=30, atr_multiplier=Decimal("2.0"))),
        ("網格=20", GridConfig(leverage=3, grid_count=20, trend_period=30, atr_multiplier=Decimal("2.0"))),
        # Trend period variations
        ("趨勢=20", GridConfig(leverage=3, grid_count=15, trend_period=20, atr_multiplier=Decimal("2.0"))),
        ("趨勢=50", GridConfig(leverage=3, grid_count=15, trend_period=50, atr_multiplier=Decimal("2.0"))),
        # ATR multiplier variations
        ("ATR=1.5", GridConfig(leverage=3, grid_count=15, trend_period=30, atr_multiplier=Decimal("1.5"))),
        ("ATR=2.5", GridConfig(leverage=3, grid_count=15, trend_period=30, atr_multiplier=Decimal("2.5"))),
    ]

    print(f"{'變體':<15} {'年化':>10} {'回撤':>10} {'Sharpe':>10} {'差異':>10}")
    print("-" * 60)

    stable_count = 0
    for name, config in variations:
        result = run_backtest(klines, config)
        diff = float(result.annual_return_pct) - float(base_result.annual_return_pct)
        is_stable = abs(diff) < 30 and float(result.annual_return_pct) > 0

        status = "✅" if is_stable else "⚠️"
        if is_stable:
            stable_count += 1

        print(f"{name:<15} {float(result.annual_return_pct):>+9.1f}% {float(result.max_drawdown_pct):>9.1f}% {float(result.sharpe_ratio):>10.2f} {diff:>+9.1f}% {status}")

    print("-" * 60)

    stability_pct = stable_count / len(variations) * 100
    if stability_pct >= 70:
        print(f"\n✅ 參數敏感度測試通過: {stable_count}/{len(variations)} ({stability_pct:.0f}%) 變體穩定")
        passed = True
    else:
        print(f"\n⚠️  參數敏感度警告: 僅 {stable_count}/{len(variations)} ({stability_pct:.0f}%) 變體穩定")
        passed = False

    return {"stable_count": stable_count, "total": len(variations), "passed": passed}


async def test_market_regimes():
    """
    Test 4: Different Market Regimes
    測試不同市場狀態下的表現
    """
    print("\n" + "=" * 70)
    print("測試 4: 市場狀態分析 (Market Regimes)")
    print("=" * 70)

    # Fetch 2 years of data to capture different market conditions
    klines = await fetch_klines("BTCUSDT", "1h", 730)
    if len(klines) < 500:
        print("數據不足")
        return None

    config = GridConfig(
        leverage=3,
        grid_count=15,
        trend_period=30,
        atr_multiplier=Decimal("2.0"),
        direction=GridDirection.TREND_FOLLOW,
        use_trend_filter=True,
        use_atr_range=True,
    )

    # Split into quarters and classify market regime
    quarter_size = len(klines) // 8
    results = []

    print(f"\n{'期間':<8} {'起始':<12} {'終止':<12} {'價格變化':>10} {'市場狀態':<10} {'年化':>10} {'Sharpe':>8}")
    print("-" * 85)

    for i in range(8):
        start_idx = i * quarter_size
        end_idx = min((i + 1) * quarter_size, len(klines))
        period_klines = klines[start_idx:end_idx]

        if len(period_klines) < 100:
            continue

        start_price = period_klines[0].close
        end_price = period_klines[-1].close
        price_change = (end_price - start_price) / start_price * 100

        # Classify market regime
        if float(price_change) > 15:
            regime = "🐂 牛市"
        elif float(price_change) < -15:
            regime = "🐻 熊市"
        else:
            regime = "↔️ 盤整"

        result = run_backtest(period_klines, config)
        results.append({"regime": regime, "result": result, "price_change": price_change})

        start_date = period_klines[0].open_time.strftime("%Y-%m")
        end_date = period_klines[-1].open_time.strftime("%Y-%m")

        print(f"Q{i+1:<6} {start_date:<12} {end_date:<12} {float(price_change):>+9.1f}% {regime:<10} {float(result.annual_return_pct):>+9.1f}% {float(result.sharpe_ratio):>8.2f}")

    print("-" * 85)

    # Analyze by regime
    bull_results = [r for r in results if "牛" in r["regime"]]
    bear_results = [r for r in results if "熊" in r["regime"]]
    side_results = [r for r in results if "盤" in r["regime"]]

    print(f"\n市場狀態統計:")
    if bull_results:
        avg = sum(float(r["result"].annual_return_pct) for r in bull_results) / len(bull_results)
        print(f"  🐂 牛市 ({len(bull_results)} 期): 平均年化 {avg:+.1f}%")
    if bear_results:
        avg = sum(float(r["result"].annual_return_pct) for r in bear_results) / len(bear_results)
        print(f"  🐻 熊市 ({len(bear_results)} 期): 平均年化 {avg:+.1f}%")
    if side_results:
        avg = sum(float(r["result"].annual_return_pct) for r in side_results) / len(side_results)
        print(f"  ↔️ 盤整 ({len(side_results)} 期): 平均年化 {avg:+.1f}%")

    # Check if strategy works in all regimes
    profitable_regimes = 0
    for regime_results in [bull_results, bear_results, side_results]:
        if regime_results:
            avg = sum(float(r["result"].annual_return_pct) for r in regime_results) / len(regime_results)
            if avg > -10:  # Allow small loss
                profitable_regimes += 1

    regime_count = sum(1 for r in [bull_results, bear_results, side_results] if r)
    if profitable_regimes >= regime_count * 0.6:
        print(f"\n✅ 市場狀態測試通過: {profitable_regimes}/{regime_count} 種市場狀態表現良好")
        passed = True
    else:
        print(f"\n⚠️  市場狀態測試警告: 僅 {profitable_regimes}/{regime_count} 種市場狀態表現良好")
        passed = False

    return {"results": results, "passed": passed}


async def main():
    print("=" * 70)
    print("Grid Futures Bot 過度擬合驗證")
    print("=" * 70)
    print("\n優化配置:")
    print("  槓桿: 3x")
    print("  網格: 15")
    print("  趨勢週期: 30")
    print("  ATR 乘數: 2.0")
    print("  方向: 順勢網格")

    # Run all tests
    results = {}

    print("\n正在獲取數據並執行驗證測試...")

    results["out_of_sample"] = await test_out_of_sample()
    results["walk_forward"] = await test_walk_forward()
    results["parameter_sensitivity"] = await test_parameter_sensitivity()
    results["market_regimes"] = await test_market_regimes()

    # Summary
    print("\n" + "=" * 70)
    print("過度擬合驗證總結")
    print("=" * 70)

    tests_passed = 0
    total_tests = 0

    for test_name, result in results.items():
        if result is None:
            continue
        total_tests += 1
        passed = result.get("passed", False)
        status = "✅ 通過" if passed else "⚠️ 警告"
        if passed:
            tests_passed += 1

        name_map = {
            "out_of_sample": "樣本外測試",
            "walk_forward": "走步前進分析",
            "parameter_sensitivity": "參數敏感度",
            "market_regimes": "市場狀態分析",
        }
        print(f"  {name_map.get(test_name, test_name):<20}: {status}")

    print("-" * 70)
    print(f"  總計: {tests_passed}/{total_tests} 測試通過")

    if tests_passed >= total_tests * 0.75:
        print("\n🎉 策略通過過度擬合驗證！配置可以使用。")
    elif tests_passed >= total_tests * 0.5:
        print("\n⚠️  策略部分通過驗證。建議降低槓桿或調整參數。")
    else:
        print("\n❌ 策略可能過度擬合。建議重新優化參數。")

    print("\n驗證完成!")


if __name__ == "__main__":
    asyncio.run(main())
