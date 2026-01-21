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
from enum import Enum
from typing import Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


class StrategyMode(str, Enum):
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


@dataclass
class BollingerConfig:
    """Bollinger configuration."""
    bb_period: int = 20
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.0"))
    leverage: int = 10
    position_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.015"))
    max_hold_bars: int = 48
    strategy_mode: StrategyMode = StrategyMode.BREAKOUT

    # ATR Stop Loss
    use_atr_stop: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))

    # Trailing Stop
    use_trailing_stop: bool = True
    trailing_atr_mult: Decimal = field(default_factory=lambda: Decimal("2.0"))

    # Trend Filter
    use_trend_filter: bool = False
    trend_period: int = 50


class BollingerBacktest:
    """Bollinger Bot backtest engine."""

    def __init__(self, klines: list[Kline], config: BollingerConfig):
        self._klines = klines
        self._config = config
        self._position = None
        self._trades = []

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

    def run(self) -> dict:
        """Run backtest and return results."""
        min_bars = max(self._config.bb_period, self._config.atr_period,
                       self._config.trend_period if self._config.use_trend_filter else 0) + 20

        if len(self._klines) < min_bars:
            return {"return": 0, "sharpe": 0, "max_dd": 0, "trades": 0, "win_rate": 0}

        capital = Decimal("10000")
        peak = capital
        max_dd = Decimal("0")
        daily_returns = []
        prev_equity = capital
        closes = []

        self._position = None
        self._trades = []

        for idx, kline in enumerate(self._klines):
            close = kline.close
            high = kline.high
            low = kline.low
            closes.append(close)

            if len(closes) < min_bars:
                continue

            # Calculate Bollinger Bands
            sma = self._calculate_sma(closes, self._config.bb_period)
            std = self._calculate_std(closes, self._config.bb_period, sma)

            if sma is None or std is None:
                continue

            upper_band = sma + std * self._config.bb_std
            lower_band = sma - std * self._config.bb_std

            # Calculate ATR
            atr = self._calculate_atr(self._klines[:idx+1], self._config.atr_period)

            # Get trend
            trend = self._get_trend(closes)

            # Position management
            if self._position:
                entry = self._position['entry']
                side = self._position['side']
                bars_held = self._position['bars_held']

                # Update trailing stop
                if self._config.use_trailing_stop and atr:
                    if side == 'long':
                        new_trail = close - atr * self._config.trailing_atr_mult
                        current_trail = self._position.get('trailing_stop')
                        if current_trail is None or new_trail > current_trail:
                            self._position['trailing_stop'] = new_trail
                    else:
                        new_trail = close + atr * self._config.trailing_atr_mult
                        current_trail = self._position.get('trailing_stop')
                        if current_trail is None or new_trail < current_trail:
                            self._position['trailing_stop'] = new_trail

                exit_reason = None
                exit_price = close

                if side == 'long':
                    pnl_pct = (close - entry) / entry

                    # Check stop loss (ATR-based or fixed)
                    if self._config.use_atr_stop and atr:
                        stop_price = entry - atr * self._config.atr_multiplier
                        if low <= stop_price:
                            exit_reason = 'sl'
                            exit_price = stop_price
                    elif pnl_pct <= -self._config.stop_loss_pct:
                        exit_reason = 'sl'

                    # Check trailing stop
                    if not exit_reason and self._config.use_trailing_stop:
                        trail = self._position.get('trailing_stop')
                        if trail and low <= trail:
                            exit_reason = 'trail'
                            exit_price = trail

                    # Check take profit (price returns to middle band for mean reversion)
                    if not exit_reason:
                        if self._config.strategy_mode == StrategyMode.MEAN_REVERSION:
                            if close >= sma:
                                exit_reason = 'tp'
                        else:  # BREAKOUT - exit on opposite band touch
                            if close <= lower_band:
                                exit_reason = 'tp'

                    # Check timeout
                    if not exit_reason and bars_held >= self._config.max_hold_bars:
                        exit_reason = 'timeout'

                else:  # short
                    pnl_pct = (entry - close) / entry

                    # Check stop loss
                    if self._config.use_atr_stop and atr:
                        stop_price = entry + atr * self._config.atr_multiplier
                        if high >= stop_price:
                            exit_reason = 'sl'
                            exit_price = stop_price
                    elif pnl_pct <= -self._config.stop_loss_pct:
                        exit_reason = 'sl'

                    # Check trailing stop
                    if not exit_reason and self._config.use_trailing_stop:
                        trail = self._position.get('trailing_stop')
                        if trail and high >= trail:
                            exit_reason = 'trail'
                            exit_price = trail

                    # Check take profit
                    if not exit_reason:
                        if self._config.strategy_mode == StrategyMode.MEAN_REVERSION:
                            if close <= sma:
                                exit_reason = 'tp'
                        else:  # BREAKOUT
                            if close >= upper_band:
                                exit_reason = 'tp'

                    # Check timeout
                    if not exit_reason and bars_held >= self._config.max_hold_bars:
                        exit_reason = 'timeout'

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
                else:
                    self._position['bars_held'] += 1

            # Entry logic
            if not self._position:
                if self._config.strategy_mode == StrategyMode.BREAKOUT:
                    # Breakout: enter when price breaks band
                    if high >= upper_band:
                        if not self._config.use_trend_filter or trend >= 0:
                            self._position = {
                                'entry': upper_band,
                                'side': 'long',
                                'bars_held': 0,
                                'trailing_stop': None,
                            }
                    elif low <= lower_band:
                        if not self._config.use_trend_filter or trend <= 0:
                            self._position = {
                                'entry': lower_band,
                                'side': 'short',
                                'bars_held': 0,
                                'trailing_stop': None,
                            }
                else:
                    # Mean reversion: enter when price touches band
                    if low <= lower_band:
                        if not self._config.use_trend_filter or trend >= 0:
                            self._position = {
                                'entry': lower_band,
                                'side': 'long',
                                'bars_held': 0,
                                'trailing_stop': None,
                            }
                    elif high >= upper_band:
                        if not self._config.use_trend_filter or trend <= 0:
                            self._position = {
                                'entry': upper_band,
                                'side': 'short',
                                'bars_held': 0,
                                'trailing_stop': None,
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

    # Test configurations
    configs = [
        # Current "optimized" config (claims 81% annual)
        ("優化參數 (20x, BB 3.25, ATR止損)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("3.25"),
            leverage=20,
            stop_loss_pct=Decimal("0.015"),
            max_hold_bars=48,
            strategy_mode=StrategyMode.BREAKOUT,
            use_atr_stop=True,
            atr_multiplier=Decimal("2.0"),
            use_trailing_stop=True,
            trailing_atr_mult=Decimal("2.0"),
        )),
        # Standard BB (2.0 std)
        ("標準 BB (20x, BB 2.0)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.0"),
            leverage=20,
            stop_loss_pct=Decimal("0.015"),
            max_hold_bars=48,
            strategy_mode=StrategyMode.BREAKOUT,
            use_atr_stop=True,
            atr_multiplier=Decimal("2.0"),
            use_trailing_stop=True,
        )),
        # Lower leverage
        ("保守槓桿 (10x, BB 3.25)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("3.25"),
            leverage=10,
            stop_loss_pct=Decimal("0.015"),
            max_hold_bars=48,
            strategy_mode=StrategyMode.BREAKOUT,
            use_atr_stop=True,
            atr_multiplier=Decimal("2.0"),
            use_trailing_stop=True,
        )),
        # Even lower leverage
        ("低槓桿 (5x, BB 3.25)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("3.25"),
            leverage=5,
            stop_loss_pct=Decimal("0.02"),
            max_hold_bars=48,
            strategy_mode=StrategyMode.BREAKOUT,
            use_atr_stop=True,
            atr_multiplier=Decimal("2.0"),
            use_trailing_stop=True,
        )),
        # Mean reversion instead of breakout
        ("均值回歸 (10x, BB 2.0)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.0"),
            leverage=10,
            stop_loss_pct=Decimal("0.02"),
            max_hold_bars=24,
            strategy_mode=StrategyMode.MEAN_REVERSION,
            use_atr_stop=True,
            atr_multiplier=Decimal("1.5"),
            use_trailing_stop=False,
        )),
        # With trend filter
        ("趨勢過濾 (10x, BB 2.5, trend)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.5"),
            leverage=10,
            stop_loss_pct=Decimal("0.02"),
            max_hold_bars=48,
            strategy_mode=StrategyMode.BREAKOUT,
            use_atr_stop=True,
            atr_multiplier=Decimal("2.0"),
            use_trailing_stop=True,
            use_trend_filter=True,
            trend_period=50,
        )),
        # Very conservative
        ("超保守 (3x, BB 2.5)", BollingerConfig(
            bb_period=20,
            bb_std=Decimal("2.5"),
            leverage=3,
            stop_loss_pct=Decimal("0.03"),
            max_hold_bars=48,
            strategy_mode=StrategyMode.BREAKOUT,
            use_atr_stop=True,
            atr_multiplier=Decimal("2.5"),
            use_trailing_stop=True,
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

    # Compare original claim vs reality
    print("\n" + "=" * 70)
    print("       原始聲稱 vs 驗證結果")
    print("=" * 70)

    original = results[0] if results[0]["name"].startswith("優化") else None
    for r in results:
        if r["name"].startswith("優化"):
            original = r
            break

    if original:
        print(f"\n原始聲稱: 年化 81.4%, Sharpe 1.13, 回撤 43.6%")
        print(f"驗證結果: 報酬 {original['return']:+.1f}%, Sharpe {original['sharpe']:.2f}, 回撤 {original['max_dd']:.1f}%")
        print(f"Walk-Forward 一致性: {original['consistency']:.0f}%")

        if original['consistency'] < 67:
            print(f"\n⚠️  結論: Bollinger Bot 可能過度擬合")
            if passing:
                rec = passing[0]
                print(f"   建議使用通過驗證的配置: {rec['name']}")
                print(f"   預期報酬: {rec['return']:+.1f}%")
        else:
            print(f"\n✅ 結論: Bollinger Bot 配置通過驗證")


if __name__ == "__main__":
    asyncio.run(main())
