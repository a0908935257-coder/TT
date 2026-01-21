#!/usr/bin/env python3
"""
Grid Futures Bot Walk-Forward Validation.

驗證 Grid Bot 是否過度擬合：
1. 將 1 年數據分成 6 個時段
2. 在每個時段獨立回測
3. 計算一致性 (需要 ≥67% 才算通過)
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


class GridDirection(str, Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    NEUTRAL = "neutral"
    TREND_FOLLOW = "trend_follow"


@dataclass
class GridConfig:
    """Grid configuration."""
    grid_count: int = 15
    leverage: int = 3
    position_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))
    direction: GridDirection = GridDirection.TREND_FOLLOW
    use_trend_filter: bool = True
    trend_period: int = 30
    use_atr_range: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    fallback_range_pct: Decimal = field(default_factory=lambda: Decimal("0.08"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))


class GridBacktest:
    """Grid Bot backtest engine."""

    def __init__(self, klines: list[Kline], config: GridConfig):
        self._klines = klines
        self._config = config
        self._position = Decimal("0")
        self._avg_entry = Decimal("0")
        self._trades = []
        self._grid_levels = []
        self._last_grid_price = None

    def _calculate_sma(self, prices: list[Decimal], period: int) -> Optional[Decimal]:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / Decimal(period)

    def _calculate_atr(self, klines: list[Kline], period: int) -> Optional[Decimal]:
        if len(klines) < period + 1:
            return None
        tr_values = []
        for i in range(1, min(len(klines), period + 1)):
            high = klines[-i].high
            low = klines[-i].low
            prev_close = klines[-i-1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        return sum(tr_values) / Decimal(len(tr_values))

    def _get_trend(self, closes: list[Decimal]) -> int:
        """Get trend direction: 1=up, -1=down, 0=neutral."""
        if not self._config.use_trend_filter:
            return 0
        sma = self._calculate_sma(closes, self._config.trend_period)
        if sma is None:
            return 0
        current = closes[-1]
        if current > sma * Decimal("1.005"):
            return 1
        elif current < sma * Decimal("0.995"):
            return -1
        return 0

    def _build_grid(self, price: Decimal, atr: Optional[Decimal]) -> list[Decimal]:
        """Build grid levels around current price."""
        if self._config.use_atr_range and atr:
            range_size = atr * self._config.atr_multiplier
        else:
            range_size = price * self._config.fallback_range_pct

        upper = price + range_size
        lower = price - range_size
        step = (upper - lower) / Decimal(self._config.grid_count)

        levels = []
        for i in range(self._config.grid_count + 1):
            levels.append(lower + step * Decimal(i))
        return levels

    def run(self) -> dict:
        """Run backtest and return results."""
        if len(self._klines) < self._config.trend_period + 50:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "trades": 0, "win_rate": 0}

        capital = Decimal("10000")
        peak = capital
        max_dd = Decimal("0")
        daily_returns = []
        prev_equity = capital
        closes = []

        self._position = Decimal("0")
        self._avg_entry = Decimal("0")
        self._trades = []
        self._grid_levels = []
        self._last_grid_price = None

        # Warmup
        warmup = max(self._config.trend_period, self._config.atr_period) + 10

        for idx, kline in enumerate(self._klines):
            close = kline.close
            closes.append(close)

            if idx < warmup:
                continue

            # Get trend and ATR
            trend = self._get_trend(closes)
            atr = self._calculate_atr(self._klines[:idx+1], self._config.atr_period)

            # Build/rebuild grid if needed
            if self._last_grid_price is None:
                self._grid_levels = self._build_grid(close, atr)
                self._last_grid_price = close
            else:
                # Rebuild if price moved significantly
                price_change = abs(close - self._last_grid_price) / self._last_grid_price
                if price_change > Decimal("0.05"):
                    self._grid_levels = self._build_grid(close, atr)
                    self._last_grid_price = close

            # Determine allowed direction
            if self._config.direction == GridDirection.TREND_FOLLOW:
                allow_long = trend >= 0
                allow_short = trend <= 0
            elif self._config.direction == GridDirection.LONG_ONLY:
                allow_long = True
                allow_short = False
            elif self._config.direction == GridDirection.SHORT_ONLY:
                allow_long = False
                allow_short = True
            else:  # NEUTRAL
                allow_long = True
                allow_short = True

            # Check grid levels for trades
            for i, level in enumerate(self._grid_levels[:-1]):
                upper_level = self._grid_levels[i + 1]

                # Price crossed up through level - potential long entry or short exit
                if kline.low <= level <= kline.high:
                    if allow_long and self._position <= Decimal("0"):
                        # Enter long at this level
                        trade_size = capital * self._config.position_pct * self._config.leverage / close
                        max_size = capital * self._config.max_position_pct * self._config.leverage / close

                        if abs(self._position + trade_size) <= max_size:
                            fee = trade_size * close * self._config.fee_rate
                            capital -= fee

                            if self._position < 0:
                                # Close short first
                                pnl = (self._avg_entry - level) * abs(self._position)
                                capital += pnl
                                self._trades.append(float(pnl))
                                self._position = Decimal("0")

                            self._position += trade_size
                            self._avg_entry = level

                # Price crossed down through level - potential short entry or long exit
                if kline.low <= upper_level <= kline.high:
                    if allow_short and self._position >= Decimal("0"):
                        trade_size = capital * self._config.position_pct * self._config.leverage / close
                        max_size = capital * self._config.max_position_pct * self._config.leverage / close

                        if abs(self._position - trade_size) <= max_size:
                            fee = trade_size * close * self._config.fee_rate
                            capital -= fee

                            if self._position > 0:
                                # Close long first
                                pnl = (upper_level - self._avg_entry) * self._position
                                capital += pnl
                                self._trades.append(float(pnl))
                                self._position = Decimal("0")

                            self._position -= trade_size
                            self._avg_entry = upper_level

            # Check stop loss
            if self._position != 0:
                if self._position > 0:
                    pnl_pct = (close - self._avg_entry) / self._avg_entry
                else:
                    pnl_pct = (self._avg_entry - close) / self._avg_entry

                if pnl_pct < -self._config.stop_loss_pct:
                    # Stop loss hit
                    if self._position > 0:
                        pnl = (close - self._avg_entry) * self._position
                    else:
                        pnl = (self._avg_entry - close) * abs(self._position)
                    capital += pnl
                    fee = abs(self._position) * close * self._config.fee_rate
                    capital -= fee
                    self._trades.append(float(pnl))
                    self._position = Decimal("0")

            # Calculate equity (including unrealized PnL)
            equity = capital
            if self._position != 0:
                if self._position > 0:
                    equity += (close - self._avg_entry) * self._position
                else:
                    equity += (self._avg_entry - close) * abs(self._position)

            # Track drawdown
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

            # Daily returns (every 24 bars for 1h timeframe)
            if idx % 24 == 0 and prev_equity > 0:
                daily_returns.append(float((equity - prev_equity) / prev_equity))
                prev_equity = equity

        # Close any remaining position
        if self._position != 0:
            close = self._klines[-1].close
            if self._position > 0:
                pnl = (close - self._avg_entry) * self._position
            else:
                pnl = (self._avg_entry - close) * abs(self._position)
            capital += pnl
            self._trades.append(float(pnl))

        # Calculate metrics
        total_return = float((capital - Decimal("10000")) / Decimal("10000") * 100)
        total_trades = len(self._trades)
        win_rate = sum(1 for t in self._trades if t > 0) / total_trades * 100 if total_trades else 0

        # Sharpe ratio
        sharpe = 0
        if len(daily_returns) > 10:
            avg = sum(daily_returns) / len(daily_returns)
            var = sum((r - avg) ** 2 for r in daily_returns) / len(daily_returns)
            std = math.sqrt(var) if var > 0 else 0.001
            sharpe = (avg / std) * math.sqrt(365) if std > 0 else 0

        return {
            "return": total_return,
            "sharpe": sharpe,
            "max_dd": float(max_dd * 100),
            "trades": total_trades,
            "win_rate": win_rate,
        }


def walk_forward(klines: list[Kline], config: GridConfig, periods: int = 6) -> dict:
    """Run walk-forward validation."""
    n = len(klines) // periods
    results = []
    period_results = []

    for i in range(periods):
        start = i * n
        end = (i + 1) * n if i < periods - 1 else len(klines)
        period_klines = klines[start:end]

        bt = GridBacktest(period_klines, config)
        r = bt.run()
        results.append(r["return"] > 0)
        period_results.append(r)

    return {
        "consistency": sum(results) / len(results) * 100,
        "profitable": sum(results),
        "total": len(results),
        "periods": period_results,
    }


async def fetch_klines(days: int, timeframe: str = "1h") -> list[Kline]:
    """Fetch historical klines."""
    interval = KlineInterval.h1 if timeframe == "1h" else KlineInterval.m15

    async with BinanceFuturesAPI() as api:
        await api.ping()
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        all_klines = []
        current = start_time

        while current < end_time:
            klines = await api.get_klines(
                symbol="BTCUSDT",
                interval=interval,
                start_time=current,
                end_time=end_time,
                limit=1500,
            )
            if not klines:
                break
            all_klines.extend(klines)
            current = int(klines[-1].close_time.timestamp() * 1000) + 1
            await asyncio.sleep(0.05)

        return all_klines


async def main():
    print("=" * 70)
    print("       Grid Futures Bot Walk-Forward 驗證")
    print("=" * 70)

    print("\n獲取 1 年數據 (1h timeframe)...")
    klines = await fetch_klines(365, "1h")
    print(f"  獲取 {len(klines)} 根 K 線")

    # Test configurations
    configs = [
        # Current optimized config
        ("優化參數 (3x, 15格, trend=30)", GridConfig(
            grid_count=15,
            leverage=3,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=30,
            use_atr_range=True,
            atr_multiplier=Decimal("2.0"),
        )),
        # More conservative
        ("保守參數 (2x, 12格, trend=50)", GridConfig(
            grid_count=12,
            leverage=2,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=50,
            use_atr_range=True,
            atr_multiplier=Decimal("2.0"),
        )),
        # Neutral direction
        ("雙向網格 (3x, 15格, neutral)", GridConfig(
            grid_count=15,
            leverage=3,
            direction=GridDirection.NEUTRAL,
            use_trend_filter=False,
            use_atr_range=True,
            atr_multiplier=Decimal("2.0"),
        )),
        # Higher leverage
        ("高槓桿 (5x, 15格, trend=30)", GridConfig(
            grid_count=15,
            leverage=5,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=30,
            use_atr_range=True,
            atr_multiplier=Decimal("2.0"),
        )),
        # Different trend period
        ("短趨勢 (3x, 15格, trend=20)", GridConfig(
            grid_count=15,
            leverage=3,
            direction=GridDirection.TREND_FOLLOW,
            use_trend_filter=True,
            trend_period=20,
            use_atr_range=True,
            atr_multiplier=Decimal("1.5"),
        )),
    ]

    print(f"\n測試 {len(configs)} 個參數組合...")
    print("-" * 70)

    results = []
    for name, config in configs:
        # Full backtest
        bt = GridBacktest(klines, config)
        full = bt.run()

        # Walk-forward validation
        wf = walk_forward(klines, config, periods=6)

        status = "✅" if full["return"] > 0 and wf["consistency"] >= 67 else "⚠️" if full["return"] > 0 else "❌"

        print(f"\n{status} {name}")
        print(f"   全期回測: {full['return']:+.1f}% | Sharpe: {full['sharpe']:.2f} | 回撤: {full['max_dd']:.1f}%")
        print(f"   Walk-Forward: {wf['consistency']:.0f}% ({wf['profitable']}/{wf['total']} 時段獲利)")

        # Show period details
        period_str = " | ".join([f"P{i+1}:{r['return']:+.0f}%" for i, r in enumerate(wf['periods'])])
        print(f"   各時段: {period_str}")

        results.append({
            "name": name,
            "config": config,
            "return": full["return"],
            "sharpe": full["sharpe"],
            "max_dd": full["max_dd"],
            "trades": full["trades"],
            "win_rate": full["win_rate"],
            "consistency": wf["consistency"],
            "periods": wf["periods"],
        })

    # Sort by consistency then return
    results.sort(key=lambda x: (x["consistency"], x["return"]), reverse=True)

    print("\n" + "=" * 70)
    print("       驗證結果總結")
    print("=" * 70)

    # Best passing
    passing = [r for r in results if r["return"] > 0 and r["consistency"] >= 67]

    if passing:
        best = passing[0]
        print(f"\n✅ 通過驗證的最佳策略: {best['name']}")
        print(f"   報酬: {best['return']:+.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   最大回撤: {best['max_dd']:.1f}%")
        print(f"   Walk-Forward 一致性: {best['consistency']:.0f}%")
        print(f"   交易次數: {best['trades']}")
    else:
        print("\n❌ 沒有策略通過 Walk-Forward 驗證 (一致性 ≥67%)")
        print("   這表示 Grid Bot 的回測績效可能過度擬合")

        if results:
            best = results[0]
            print(f"\n   最佳策略 (未通過):")
            print(f"   {best['name']}")
            print(f"   報酬: {best['return']:+.1f}%, 一致性: {best['consistency']:.0f}%")

    # Compare original claim vs reality
    print("\n" + "=" * 70)
    print("       原始聲稱 vs 驗證結果")
    print("=" * 70)

    original = results[0]  # First config is the "optimized" one
    print(f"\n原始聲稱: 年化 113.6%, Sharpe 1.54")
    print(f"驗證結果: 報酬 {original['return']:+.1f}%, Sharpe {original['sharpe']:.2f}")
    print(f"Walk-Forward 一致性: {original['consistency']:.0f}%")

    if original['consistency'] < 67:
        print(f"\n⚠️  結論: Grid Bot 可能過度擬合")
        print(f"   建議: 降低預期報酬至 {original['return'] * 0.5:+.1f}% ~ {original['return'] * 0.7:+.1f}%")
    else:
        print(f"\n✅ 結論: Grid Bot 通過驗證，績效可信")


if __name__ == "__main__":
    asyncio.run(main())
