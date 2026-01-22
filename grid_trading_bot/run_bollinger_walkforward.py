#!/usr/bin/env python3
"""
Bollinger Bot Walk-Forward Validation.

驗證 Bollinger Bot 是否過度擬合：
1. 將 1 年數據分成 6 個時段
2. 在每個時段獨立回測
3. 計算一致性 (需要 ≥67% 才算通過)
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


@dataclass
class BollingerConfig:
    """Bollinger + Supertrend configuration."""
    bb_period: int = 20
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.5"))
    leverage: int = 5
    position_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))

    # Supertrend parameters
    st_atr_period: int = 20
    st_atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.5"))

    # ATR Stop Loss
    atr_stop_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))


class BollingerBacktest:
    """Bollinger Bot backtest engine."""

    def __init__(self, klines: list[Kline], config: BollingerConfig):
        self._klines = klines
        self._config = config
        self._position = None
        self._trades = []

        # Supertrend state (for BOLLINGER_TREND mode)
        self._prev_st_upper: Optional[Decimal] = None
        self._prev_st_lower: Optional[Decimal] = None
        self._prev_st_trend: int = 0
        self._prev_close: Optional[Decimal] = None

    def _calculate_sma(self, prices: list[Decimal], period: int) -> Optional[Decimal]:
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / Decimal(period)

    def _calculate_std(self, prices: list[Decimal], period: int, sma: Decimal) -> Optional[Decimal]:
        if len(prices) < period:
            return None
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / Decimal(period)
        return variance.sqrt() if hasattr(variance, 'sqrt') else Decimal(str(math.sqrt(float(variance))))

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

    def _calculate_supertrend(self, klines: list[Kline]) -> tuple[int, Decimal, Decimal]:
        """
        Calculate Supertrend indicator.

        Returns:
            Tuple of (trend, supertrend_value, atr)
            trend: 1 = bullish, -1 = bearish
        """
        period = self._config.st_atr_period
        multiplier = self._config.st_atr_multiplier

        if len(klines) < period + 1:
            return 0, Decimal("0"), Decimal("0")

        # Calculate ATR
        tr_values = []
        for i in range(len(klines) - period, len(klines)):
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close if i > 0 else klines[i].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        atr = sum(tr_values) / Decimal(len(tr_values))

        # Current kline
        kline = klines[-1]
        high = kline.high
        low = kline.low
        close = kline.close

        # Calculate bands
        hl2 = (high + low) / Decimal("2")
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Adjust bands based on previous values
        if self._prev_st_upper is not None and self._prev_close is not None:
            if self._prev_close > self._prev_st_upper:
                lower_band = max(lower_band, self._prev_st_lower)
            if self._prev_close < self._prev_st_lower:
                upper_band = min(upper_band, self._prev_st_upper)

        # Determine trend
        if self._prev_st_trend == 0:
            trend = 1 if close > upper_band else -1
        elif self._prev_st_trend == 1:
            trend = 1 if close > self._prev_st_lower else -1
        else:
            trend = -1 if close < self._prev_st_upper else 1

        # Supertrend value
        supertrend = lower_band if trend == 1 else upper_band

        # Store for next iteration
        self._prev_st_upper = upper_band
        self._prev_st_lower = lower_band
        self._prev_st_trend = trend
        self._prev_close = close

        return trend, supertrend, atr

    def run(self) -> dict:
        """Run backtest and return results."""
        min_bars = max(self._config.bb_period, self._config.st_atr_period) + 20

        if len(self._klines) < min_bars:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "trades": 0, "win_rate": 0}

        capital = Decimal("10000")
        peak = capital
        max_dd = Decimal("0")
        daily_returns = []
        prev_equity = capital
        closes = []

        # Reset state
        self._position = None
        self._trades = []
        self._prev_st_upper = None
        self._prev_st_lower = None
        self._prev_st_trend = 0
        self._prev_close = None

        for idx, kline in enumerate(self._klines):
            close = kline.close
            high = kline.high
            low = kline.low
            closes.append(close)

            if len(closes) < min_bars:
                # Still need to update Supertrend state
                self._calculate_supertrend(self._klines[:idx+1])
                continue

            # Calculate Bollinger Bands
            sma = self._calculate_sma(closes, self._config.bb_period)
            std = self._calculate_std(closes, self._config.bb_period, sma)

            if sma is None or std is None:
                continue

            upper_band = sma + std * self._config.bb_std
            lower_band = sma - std * self._config.bb_std

            # Calculate Supertrend
            st_trend, st_value, st_atr = self._calculate_supertrend(self._klines[:idx+1])

            # Exit logic
            if self._position:
                entry = self._position['entry']
                side = self._position['side']

                exit_reason = None
                exit_price = close

                # 1. Check Supertrend flip (primary exit)
                if side == 'long' and st_trend == -1:
                    exit_reason = 'st_flip'
                elif side == 'short' and st_trend == 1:
                    exit_reason = 'st_flip'

                # 2. Check ATR stop loss (protection)
                if not exit_reason:
                    stop_distance = st_atr * self._config.atr_stop_multiplier
                    if side == 'long':
                        stop_price = entry - stop_distance
                        if low <= stop_price:
                            exit_reason = 'sl'
                            exit_price = stop_price
                    else:
                        stop_price = entry + stop_distance
                        if high >= stop_price:
                            exit_reason = 'sl'
                            exit_price = stop_price

                if exit_reason:
                    # Calculate PnL
                    if side == 'long':
                        pnl = (exit_price - entry) / entry
                    else:
                        pnl = (entry - exit_price) / entry

                    net = pnl * self._config.leverage * self._config.position_pct
                    net -= self._config.fee_rate * 2 * self._config.leverage * self._config.position_pct
                    self._trades.append(float(net))
                    capital += capital * net
                    self._position = None

            # Entry logic: Supertrend direction + BB band touch
            if not self._position:
                if st_trend == 1:  # Bullish Supertrend
                    if low <= lower_band:
                        self._position = {
                            'entry': lower_band,
                            'side': 'long',
                        }
                elif st_trend == -1:  # Bearish Supertrend
                    if high >= upper_band:
                        self._position = {
                            'entry': upper_band,
                            'side': 'short',
                        }

            # Track equity and drawdown
            equity = capital
            if self._position:
                entry = self._position['entry']
                if self._position['side'] == 'long':
                    unrealized = (close - entry) / entry * self._config.leverage * self._config.position_pct
                else:
                    unrealized = (entry - close) / entry * self._config.leverage * self._config.position_pct
                equity += capital * unrealized

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

            # Daily returns (every 96 bars for 15m timeframe)
            if idx % 96 == 0 and prev_equity > 0:
                daily_returns.append(float((equity - prev_equity) / prev_equity))
                prev_equity = equity

        # Close any remaining position
        if self._position:
            close = self._klines[-1].close
            entry = self._position['entry']
            if self._position['side'] == 'long':
                pnl = (close - entry) / entry
            else:
                pnl = (entry - close) / entry
            net = pnl * self._config.leverage * self._config.position_pct
            net -= self._config.fee_rate * 2 * self._config.leverage * self._config.position_pct
            self._trades.append(float(net))
            capital += capital * net

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


def walk_forward(klines: list[Kline], config: BollingerConfig, periods: int = 6) -> dict:
    """Run walk-forward validation."""
    n = len(klines) // periods
    results = []
    period_results = []

    for i in range(periods):
        start = i * n
        end = (i + 1) * n if i < periods - 1 else len(klines)
        period_klines = klines[start:end]

        bt = BollingerBacktest(period_klines, config)
        r = bt.run()
        results.append(r["return"] > 0)
        period_results.append(r)

    return {
        "consistency": sum(results) / len(results) * 100,
        "profitable": sum(results),
        "total": len(results),
        "periods": period_results,
    }


async def fetch_klines(days: int, timeframe: str = "15m") -> list[Kline]:
    """Fetch historical klines."""
    interval = KlineInterval.m15 if timeframe == "15m" else KlineInterval.h1

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
    print("       Bollinger Bot Walk-Forward 驗證")
    print("=" * 70)

    print("\n獲取 1 年數據 (15m timeframe)...")
    klines = await fetch_klines(365, "15m")
    print(f"  獲取 {len(klines)} 根 K 線")

    # Test configurations (BOLLINGER_TREND: Supertrend + BB band touch)
    configs = [
        ("🌟 BB+ST (5x, BB 2.5, ST 20/3.5)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.5"),
            leverage=5,
            st_atr_period=20,
            st_atr_multiplier=Decimal("3.5"),
            atr_stop_multiplier=Decimal("2.0"),
        )),
        ("BB+ST (5x, BB 2.0, ST 20/3.5)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.0"),
            leverage=5,
            st_atr_period=20,
            st_atr_multiplier=Decimal("3.5"),
            atr_stop_multiplier=Decimal("2.0"),
        )),
        ("BB+ST (5x, BB 2.0, ST 25/3.0)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.0"),
            leverage=5,
            st_atr_period=25,
            st_atr_multiplier=Decimal("3.0"),
            atr_stop_multiplier=Decimal("2.0"),
        )),
        ("BB+ST (10x, BB 2.5, ST 20/3.5)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.5"),
            leverage=10,
            st_atr_period=20,
            st_atr_multiplier=Decimal("3.5"),
            atr_stop_multiplier=Decimal("2.0"),
        )),
        ("BB+ST (3x, BB 2.0, ST 20/3.5)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.0"),
            leverage=3,
            st_atr_period=20,
            st_atr_multiplier=Decimal("3.5"),
            atr_stop_multiplier=Decimal("2.5"),
        )),
    ]

    print(f"\n測試 {len(configs)} 個參數組合...")
    print("-" * 70)

    results = []
    for name, config in configs:
        # Full backtest
        bt = BollingerBacktest(klines, config)
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
        print(f"   勝率: {best['win_rate']:.1f}%")
    else:
        print("\n❌ 沒有策略通過 Walk-Forward 驗證 (一致性 ≥67%)")
        print("   這表示 Bollinger Bot 的回測績效可能過度擬合")

        if results:
            best = results[0]
            print(f"\n   最佳策略 (未通過):")
            print(f"   {best['name']}")
            print(f"   報酬: {best['return']:+.1f}%, 一致性: {best['consistency']:.0f}%")

    # Final recommendation
    print("\n" + "=" * 70)
    print("       結論與建議")
    print("=" * 70)

    if passing:
        rec = passing[0]
        print(f"\n✅ 推薦配置: {rec['name']}")
        print(f"   報酬: {rec['return']:+.1f}%")
        print(f"   Sharpe: {rec['sharpe']:.2f}")
        print(f"   一致性: {rec['consistency']:.0f}%")
        print("\n   建議使用此策略配置。")
    else:
        print("\n⚠️ 未通過 Walk-Forward 驗證 (一致性 < 67%)")
        print("   策略仍優於舊的均值回歸策略，但建議謹慎使用。")
        if results:
            best = results[0]
            print(f"\n   最佳配置: {best['name']}")
            print(f"   報酬: {best['return']:+.1f}%")
            print(f"   一致性: {best['consistency']:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
