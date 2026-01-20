#!/usr/bin/env python3
"""
Optimized Grid Bot Backtest - Target >30% Annual Return.

優化方向：
1. 槓桿交易 (合約)
2. 雙向網格 (做多+做空)
3. 趨勢過濾 (順勢網格)
4. 動態網格寬度 (基於 ATR)
5. 參數優化

Usage:
    python run_grid_optimization.py
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
    """Grid trading direction."""
    LONG_ONLY = "long_only"      # 只做多網格
    SHORT_ONLY = "short_only"    # 只做空網格
    NEUTRAL = "neutral"          # 雙向網格
    TREND_FOLLOW = "trend_follow"  # 順勢網格 (趨勢向上做多, 向下做空)


@dataclass
class OptimizedGridConfig:
    """Optimized grid configuration."""
    grid_count: int = 15
    range_pct: Decimal = field(default_factory=lambda: Decimal("0.08"))  # ±8%
    position_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    leverage: int = 5
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))

    # Trend filter
    use_trend_filter: bool = True
    trend_period: int = 50  # SMA period for trend

    # Direction
    direction: GridDirection = GridDirection.TREND_FOLLOW

    # Dynamic range based on ATR
    use_atr_range: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))

    # Risk management
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))  # Max 50% of capital


@dataclass
class OptimizedBacktestResult:
    """Optimized backtest result."""
    initial_capital: Decimal = field(default_factory=lambda: Decimal("0"))
    final_equity: Decimal = field(default_factory=lambda: Decimal("0"))
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    annual_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_and_hold_return: Decimal = field(default_factory=lambda: Decimal("0"))
    excess_return: Decimal = field(default_factory=lambda: Decimal("0"))
    long_trades: int = 0
    short_trades: int = 0
    grid_rebuilds: int = 0
    avg_trade_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))


class OptimizedGridBacktest:
    """Optimized Grid Bot backtest engine."""

    def __init__(
        self,
        klines: List[Kline],
        initial_capital: Decimal,
        config: OptimizedGridConfig,
    ):
        self._klines = klines
        self._initial_capital = initial_capital
        self._config = config

        # State
        self._capital = initial_capital
        self._position = Decimal("0")  # Positive = long, Negative = short
        self._avg_entry_price = Decimal("0")
        self._current_direction: Optional[str] = None  # "long" or "short"

        # Grid state
        self._grids: List[dict] = []
        self._upper_price = Decimal("0")
        self._lower_price = Decimal("0")
        self._center_price = Decimal("0")
        self._grid_spacing = Decimal("0")
        self._current_trend = 0  # 1 = up, -1 = down, 0 = neutral

        # Statistics
        self._trades: List[dict] = []
        self._equity_curve: List[Decimal] = []
        self._daily_returns: List[Decimal] = []
        self._grid_rebuilds = 0
        self._long_trades = 0
        self._short_trades = 0

    def _calculate_sma(self, prices: List[Decimal], period: int) -> Optional[Decimal]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / Decimal(period)

    def _calculate_atr(self, klines: List[Kline], period: int) -> Optional[Decimal]:
        """Calculate Average True Range."""
        if len(klines) < period + 1:
            return None

        tr_values = []
        for i in range(1, min(len(klines), period + 1)):
            high = klines[-i].high
            low = klines[-i].low
            prev_close = klines[-i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        if not tr_values:
            return None
        return sum(tr_values) / Decimal(len(tr_values))

    def _determine_trend(self, closes: List[Decimal], current_price: Decimal) -> int:
        """Determine market trend. Returns 1 (up), -1 (down), 0 (neutral)."""
        if not self._config.use_trend_filter:
            return 0

        sma = self._calculate_sma(closes, self._config.trend_period)
        if sma is None:
            return 0

        diff_pct = (current_price - sma) / sma * Decimal("100")

        if diff_pct > Decimal("1"):  # >1% above SMA = uptrend
            return 1
        elif diff_pct < Decimal("-1"):  # <1% below SMA = downtrend
            return -1
        return 0

    def _should_trade_direction(self, direction: str) -> bool:
        """Check if we should trade in this direction based on config and trend."""
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
            return True  # Neutral trend, allow both
        return True

    def _initialize_grid(self, center_price: Decimal, atr: Optional[Decimal] = None) -> None:
        """Initialize grid around center price."""
        # Calculate range
        if self._config.use_atr_range and atr:
            range_value = atr * self._config.atr_multiplier
            range_pct = range_value / center_price
            range_pct = min(max(range_pct, Decimal("0.03")), Decimal("0.20"))  # Clamp 3%-20%
        else:
            range_pct = self._config.range_pct

        self._center_price = center_price
        self._upper_price = center_price * (Decimal("1") + range_pct)
        self._lower_price = center_price * (Decimal("1") - range_pct)
        self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)

        self._grids = []
        for i in range(self._config.grid_count + 1):
            price = self._lower_price + Decimal(i) * self._grid_spacing
            self._grids.append({
                "price": price,
                "long_filled": False,
                "short_filled": False,
            })

    def _check_rebuild_needed(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuild."""
        if current_price > self._upper_price * Decimal("1.02"):  # 2% buffer
            return True
        if current_price < self._lower_price * Decimal("0.98"):
            return True
        return False

    def run(self) -> OptimizedBacktestResult:
        """Run backtest."""
        if not self._klines or len(self._klines) < 100:
            return OptimizedBacktestResult()

        # Collect close prices for indicators
        closes: List[Decimal] = []

        # Initialize grid
        first_price = self._klines[0].close
        self._initialize_grid(first_price)

        max_equity = self._initial_capital
        max_drawdown = Decimal("0")
        prev_equity = self._initial_capital

        for i, kline in enumerate(self._klines):
            current_price = kline.close
            closes.append(current_price)

            # Update trend
            self._current_trend = self._determine_trend(closes, current_price)

            # Calculate ATR for dynamic range
            atr = self._calculate_atr(self._klines[:i+1], self._config.atr_period) if i > self._config.atr_period else None

            # Check if rebuild needed
            if self._check_rebuild_needed(current_price):
                # Close position before rebuild
                if self._position != Decimal("0"):
                    self._close_position(current_price)

                # Rebuild grid
                self._initialize_grid(current_price, atr)
                self._grid_rebuilds += 1

            # Process grid logic
            self._process_grids(current_price)

            # Calculate equity
            unrealized_pnl = Decimal("0")
            if self._position != Decimal("0"):
                if self._position > 0:  # Long
                    unrealized_pnl = self._position * (current_price - self._avg_entry_price) * Decimal(self._config.leverage)
                else:  # Short
                    unrealized_pnl = abs(self._position) * (self._avg_entry_price - current_price) * Decimal(self._config.leverage)

            current_equity = self._capital + unrealized_pnl
            self._equity_curve.append(current_equity)

            # Track daily returns
            if prev_equity > 0:
                daily_return = (current_equity - prev_equity) / prev_equity
                self._daily_returns.append(daily_return)
            prev_equity = current_equity

            # Track max drawdown
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = max_equity - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close final position
        final_price = self._klines[-1].close
        if self._position != Decimal("0"):
            self._close_position(final_price)

        return self._calculate_result(max_drawdown, max_equity)

    def _close_position(self, current_price: Decimal) -> None:
        """Close current position."""
        if self._position == Decimal("0"):
            return

        if self._position > 0:  # Long position
            pnl = self._position * (current_price - self._avg_entry_price) * Decimal(self._config.leverage)
            fee = abs(self._position) * current_price * self._config.fee_rate
        else:  # Short position
            pnl = abs(self._position) * (self._avg_entry_price - current_price) * Decimal(self._config.leverage)
            fee = abs(self._position) * current_price * self._config.fee_rate

        self._capital += pnl - fee

        self._trades.append({
            "type": "close",
            "direction": "long" if self._position > 0 else "short",
            "price": current_price,
            "pnl": pnl - fee,
        })

        self._position = Decimal("0")
        self._avg_entry_price = Decimal("0")

    def _process_grids(self, current_price: Decimal) -> None:
        """Process grid buy/sell logic."""
        max_position_value = self._capital * self._config.max_position_pct
        trade_value = self._capital * self._config.position_pct

        for grid in self._grids:
            grid_price = grid["price"]

            # Long entry: price at or below grid (buy low)
            if current_price <= grid_price and not grid["long_filled"]:
                if self._should_trade_direction("long"):
                    current_position_value = abs(self._position) * current_price if self._position else Decimal("0")

                    if current_position_value < max_position_value and trade_value > Decimal("10"):
                        self._open_long(current_price, trade_value)
                        grid["long_filled"] = True
                        grid["short_filled"] = False

            # Long exit / Short entry: price at or above grid (sell high)
            elif current_price >= grid_price:
                # Close long if we have one
                if grid["long_filled"] and self._position > 0:
                    self._close_partial_long(current_price, trade_value)
                    grid["long_filled"] = False

                # Open short if allowed and not filled
                if not grid["short_filled"] and self._should_trade_direction("short"):
                    current_position_value = abs(self._position) * current_price if self._position else Decimal("0")

                    if current_position_value < max_position_value and trade_value > Decimal("10"):
                        self._open_short(current_price, trade_value)
                        grid["short_filled"] = True
                        grid["long_filled"] = False

            # Short exit: price drops below grid
            if current_price <= grid_price and grid["short_filled"] and self._position < 0:
                self._close_partial_short(current_price, trade_value)
                grid["short_filled"] = False

    def _open_long(self, price: Decimal, value: Decimal) -> None:
        """Open or add to long position."""
        quantity = value / price
        fee = value * self._config.fee_rate

        if self._position >= 0:  # Adding to long or new long
            if self._position > 0:
                total_value = self._position * self._avg_entry_price + quantity * price
                self._position += quantity
                self._avg_entry_price = total_value / self._position
            else:
                self._position = quantity
                self._avg_entry_price = price
        else:  # Close short first
            self._close_position(price)
            self._position = quantity
            self._avg_entry_price = price

        self._capital -= fee
        self._long_trades += 1

    def _open_short(self, price: Decimal, value: Decimal) -> None:
        """Open or add to short position."""
        quantity = value / price
        fee = value * self._config.fee_rate

        if self._position <= 0:  # Adding to short or new short
            if self._position < 0:
                total_value = abs(self._position) * self._avg_entry_price + quantity * price
                self._position -= quantity
                self._avg_entry_price = total_value / abs(self._position)
            else:
                self._position = -quantity
                self._avg_entry_price = price
        else:  # Close long first
            self._close_position(price)
            self._position = -quantity
            self._avg_entry_price = price

        self._capital -= fee
        self._short_trades += 1

    def _close_partial_long(self, price: Decimal, value: Decimal) -> None:
        """Close partial long position."""
        if self._position <= 0:
            return

        quantity = min(self._position, value / price)
        pnl = quantity * (price - self._avg_entry_price) * Decimal(self._config.leverage)
        fee = quantity * price * self._config.fee_rate

        self._capital += pnl - fee
        self._position -= quantity

        self._trades.append({
            "type": "close_long",
            "price": price,
            "quantity": quantity,
            "pnl": pnl - fee,
        })

    def _close_partial_short(self, price: Decimal, value: Decimal) -> None:
        """Close partial short position."""
        if self._position >= 0:
            return

        quantity = min(abs(self._position), value / price)
        pnl = quantity * (self._avg_entry_price - price) * Decimal(self._config.leverage)
        fee = quantity * price * self._config.fee_rate

        self._capital += pnl - fee
        self._position += quantity

        self._trades.append({
            "type": "close_short",
            "price": price,
            "quantity": quantity,
            "pnl": pnl - fee,
        })

    def _calculate_result(self, max_drawdown: Decimal, max_equity: Decimal) -> OptimizedBacktestResult:
        """Calculate final result."""
        result = OptimizedBacktestResult()

        result.initial_capital = self._initial_capital
        result.final_equity = self._capital
        result.total_profit = self._capital - self._initial_capital
        result.total_return_pct = (result.total_profit / self._initial_capital) * Decimal("100")

        # Annualize return (assuming 365 days of data)
        days = len(self._klines) / 24 if len(self._klines) > 0 else 365  # Assuming hourly data
        years = days / 365
        if years > 0:
            result.annual_return_pct = result.total_return_pct / Decimal(str(years))

        # Trade statistics
        result.total_trades = len(self._trades)
        result.long_trades = self._long_trades
        result.short_trades = self._short_trades
        result.grid_rebuilds = self._grid_rebuilds

        # Win/loss
        profitable_trades = [t for t in self._trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self._trades if t.get("pnl", 0) <= 0]
        result.num_wins = len(profitable_trades)
        result.num_losses = len(losing_trades)

        if self._trades:
            result.win_rate = Decimal(result.num_wins) / Decimal(len(self._trades)) * Decimal("100")
            total_pnl = sum(t.get("pnl", Decimal("0")) for t in self._trades)
            result.avg_trade_profit = total_pnl / Decimal(len(self._trades))

        # Profit factor
        gross_profit = sum(t.get("pnl", Decimal("0")) for t in profitable_trades)
        gross_loss = abs(sum(t.get("pnl", Decimal("0")) for t in losing_trades))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss

        # Drawdown
        if max_equity > 0:
            result.max_drawdown_pct = (max_drawdown / max_equity) * Decimal("100")

        # Buy and hold comparison
        if self._klines:
            start_price = self._klines[0].close
            end_price = self._klines[-1].close
            result.buy_and_hold_return = ((end_price - start_price) / start_price) * Decimal("100")
            result.excess_return = result.total_return_pct - result.buy_and_hold_return

        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe_ratio()

        return result

    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate annualized Sharpe ratio."""
        if len(self._daily_returns) < 10:
            return Decimal("0")

        returns = [float(r) for r in self._daily_returns]
        mean_return = sum(returns) / len(returns)

        if len(returns) < 2:
            return Decimal("0")

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)

        if variance <= 0:
            return Decimal("0")

        std_dev = math.sqrt(variance)
        if std_dev == 0:
            return Decimal("0")

        # Annualized (assuming hourly data, ~8760 hours/year)
        sharpe = (mean_return / std_dev) * math.sqrt(8760)
        return Decimal(str(round(sharpe, 2)))


async def fetch_klines(symbol: str, interval: str, days: int) -> List[Kline]:
    """Fetch historical klines from Binance Futures."""
    print(f"\n正在從 Binance 獲取 {symbol} {interval} 歷史數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        await api.ping()

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        interval_map = {
            "1m": KlineInterval.m1,
            "5m": KlineInterval.m5,
            "15m": KlineInterval.m15,
            "30m": KlineInterval.m30,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
            "1d": KlineInterval.d1,
        }
        kline_interval = interval_map.get(interval, KlineInterval.h1)

        all_klines = []
        current_start = start_time

        while current_start < end_time:
            klines = await api.get_klines(
                symbol=symbol,
                interval=kline_interval,
                limit=1500,
                start_time=current_start,
                end_time=end_time,
            )

            if not klines:
                break

            all_klines.extend(klines)

            if len(klines) < 1500:
                break

            current_start = int(klines[-1].close_time.timestamp() * 1000) + 1

        print(f"  獲取 {len(all_klines)} 根 K 線")
        return all_klines


def print_result(result: OptimizedBacktestResult, config: OptimizedGridConfig, name: str):
    """Print backtest result."""
    print(f"\n{'='*65}")
    print(f"配置: {name}")
    print(f"{'='*65}")

    print(f"\n📊 配置參數:")
    print(f"  網格數量:       {config.grid_count}")
    print(f"  網格範圍:       ±{float(config.range_pct)*100:.1f}%")
    print(f"  槓桿:           {config.leverage}x")
    print(f"  方向:           {config.direction.value}")
    print(f"  趨勢過濾:       {'開啟' if config.use_trend_filter else '關閉'}")
    print(f"  動態 ATR:       {'開啟' if config.use_atr_range else '關閉'}")

    print(f"\n💰 績效摘要:")
    print(f"  初始資金:       ${float(result.initial_capital):,.2f}")
    print(f"  最終資金:       ${float(result.final_equity):,.2f}")
    print(f"  總盈虧:         ${float(result.total_profit):+,.2f}")
    print(f"  總回報:         {float(result.total_return_pct):+.2f}%")
    print(f"  年化回報:       {float(result.annual_return_pct):+.2f}%")

    print(f"\n📊 交易統計:")
    print(f"  總交易次數:     {result.total_trades}")
    print(f"  做多交易:       {result.long_trades}")
    print(f"  做空交易:       {result.short_trades}")
    print(f"  網格重建:       {result.grid_rebuilds}")
    print(f"  勝率:           {float(result.win_rate):.1f}%")
    print(f"  獲利因子:       {float(result.profit_factor):.2f}")

    print(f"\n⚠️  風險指標:")
    print(f"  最大回撤:       {float(result.max_drawdown_pct):.2f}%")
    print(f"  Sharpe Ratio:   {float(result.sharpe_ratio):.2f}")

    print(f"\n🔄 策略比較:")
    print(f"  Grid Bot ROI:   {float(result.total_return_pct):+.2f}%")
    print(f"  持有策略 ROI:   {float(result.buy_and_hold_return):+.2f}%")
    print(f"  超額收益:       {float(result.excess_return):+.2f}%")

    # Target check
    target_annual = 30
    if float(result.annual_return_pct) >= target_annual:
        print(f"\n  ✅ 達成目標! 年化 {float(result.annual_return_pct):.1f}% >= {target_annual}%")
    else:
        print(f"\n  ❌ 未達目標: 年化 {float(result.annual_return_pct):.1f}% < {target_annual}%")


async def run_optimization():
    """Run parameter optimization to find >30% annual return config."""
    print("=" * 65)
    print("Grid Bot 參數優化 - 目標年化 >30%")
    print("=" * 65)

    # Fetch data once
    klines = await fetch_klines("BTCUSDT", "1h", 365)
    if not klines:
        print("無法獲取數據")
        return

    initial_capital = Decimal("10000")

    # Test configurations
    configs = [
        # 1. Baseline: Long-only spot (like current)
        ("基準: 現貨只做多", OptimizedGridConfig(
            grid_count=15,
            range_pct=Decimal("0.08"),
            leverage=1,
            direction=GridDirection.LONG_ONLY,
            use_trend_filter=False,
            use_atr_range=False,
        )),

        # 2. Leverage 3x, long-only
        ("槓桿 3x 只做多", OptimizedGridConfig(
            grid_count=15,
            range_pct=Decimal("0.08"),
            leverage=3,
            direction=GridDirection.LONG_ONLY,
            use_trend_filter=False,
            use_atr_range=False,
        )),

        # 3. Leverage 5x, neutral (both directions)
        ("槓桿 5x 雙向", OptimizedGridConfig(
            grid_count=15,
            range_pct=Decimal("0.08"),
            leverage=5,
            direction=GridDirection.NEUTRAL,
            use_trend_filter=False,
            use_atr_range=False,
        )),

        # 4. Leverage 5x with trend filter
        ("槓桿 5x 順勢", OptimizedGridConfig(
            grid_count=15,
            range_pct=Decimal("0.08"),
            leverage=5,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=50,
            use_atr_range=False,
        )),

        # 5. Leverage 5x, trend filter, ATR range
        ("槓桿 5x 順勢+ATR", OptimizedGridConfig(
            grid_count=15,
            range_pct=Decimal("0.08"),
            leverage=5,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=50,
            use_atr_range=True,
            atr_multiplier=Decimal("2.0"),
        )),

        # 6. Higher leverage 10x, trend filter
        ("槓桿 10x 順勢", OptimizedGridConfig(
            grid_count=20,
            range_pct=Decimal("0.06"),
            leverage=10,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=30,
            use_atr_range=True,
            atr_multiplier=Decimal("1.5"),
        )),

        # 7. Optimized: 10x, narrow grid, trend filter
        ("優化: 10x 窄格順勢", OptimizedGridConfig(
            grid_count=25,
            range_pct=Decimal("0.05"),
            leverage=10,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=20,
            use_atr_range=True,
            atr_multiplier=Decimal("1.5"),
            position_pct=Decimal("0.08"),
        )),

        # 8. Aggressive: 15x leverage
        ("激進: 15x 順勢", OptimizedGridConfig(
            grid_count=20,
            range_pct=Decimal("0.05"),
            leverage=15,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=20,
            use_atr_range=True,
            atr_multiplier=Decimal("1.5"),
            position_pct=Decimal("0.06"),
            max_position_pct=Decimal("0.4"),
        )),
    ]

    results = []
    for name, config in configs:
        backtest = OptimizedGridBacktest(
            klines=klines,
            initial_capital=initial_capital,
            config=config,
        )
        result = backtest.run()
        results.append((name, config, result))
        print_result(result, config, name)

    # Summary table
    print("\n" + "=" * 80)
    print("總結比較")
    print("=" * 80)
    print(f"{'配置':<22} {'年化':>10} {'總回報':>10} {'Sharpe':>8} {'回撤':>8} {'勝率':>8} {'超額':>10}")
    print("-" * 80)

    for name, config, result in results:
        annual = float(result.annual_return_pct)
        total = float(result.total_return_pct)
        sharpe = float(result.sharpe_ratio)
        dd = float(result.max_drawdown_pct)
        wr = float(result.win_rate)
        excess = float(result.excess_return)

        marker = "✅" if annual >= 30 else "  "
        print(f"{marker}{name:<20} {annual:>+9.1f}% {total:>+9.1f}% {sharpe:>8.2f} {dd:>7.1f}% {wr:>7.1f}% {excess:>+9.1f}%")

    print("=" * 80)

    # Find best strategy meeting target
    target_configs = [(n, c, r) for n, c, r in results if float(r.annual_return_pct) >= 30]

    if target_configs:
        # Sort by Sharpe ratio
        best = max(target_configs, key=lambda x: float(x[2].sharpe_ratio))
        print(f"\n🏆 最佳策略 (年化 ≥30%): {best[0]}")
        print(f"   年化: {float(best[2].annual_return_pct):.1f}%")
        print(f"   Sharpe: {float(best[2].sharpe_ratio):.2f}")
        print(f"   最大回撤: {float(best[2].max_drawdown_pct):.1f}%")

        # Print config for .env
        print(f"\n📝 建議 .env 配置:")
        print(f"   GRID_LEVERAGE={best[1].leverage}")
        print(f"   GRID_COUNT={best[1].grid_count}")
        print(f"   GRID_RANGE_PCT={float(best[1].range_pct)}")
        print(f"   GRID_DIRECTION={best[1].direction.value}")
        print(f"   GRID_USE_TREND_FILTER={'true' if best[1].use_trend_filter else 'false'}")
        print(f"   GRID_TREND_PERIOD={best[1].trend_period}")
        print(f"   GRID_USE_ATR_RANGE={'true' if best[1].use_atr_range else 'false'}")
    else:
        # Find best overall
        best = max(results, key=lambda x: float(x[2].annual_return_pct))
        print(f"\n⚠️  沒有配置達到 30% 年化目標")
        print(f"   最佳策略: {best[0]} ({float(best[2].annual_return_pct):.1f}% 年化)")

    print("\n優化完成!")


if __name__ == "__main__":
    asyncio.run(run_optimization())
