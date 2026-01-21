#!/usr/bin/env python3
"""
Validated Parameters Backtest.

Tests the Walk-Forward validated parameters:
- Bollinger Bot BREAKOUT: BB(15, 1.5), 2x leverage, trend filter, 83% consistency
- RSI Bot: rsi(21), oversold=30, overbought=70, 5x leverage
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
class BacktestResult:
    """Backtest result."""
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0


class BollingerBreakoutBacktest:
    """
    Bollinger BREAKOUT Strategy Backtest.

    Walk-Forward Validated Parameters (83% consistency, Sharpe 5.45):
    - bb_period: 15
    - bb_std: 1.5
    - leverage: 2x
    - use_trend_filter: True
    - max_hold_bars: 24
    """

    FEE_RATE = Decimal("0.0004")

    def __init__(
        self,
        klines: list[Kline],
        bb_period: int = 15,
        bb_std: Decimal = Decimal("1.5"),
        trend_period: int = 50,
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal("2.0"),
        max_hold_bars: int = 24,
        leverage: int = 2,
        bbw_threshold_pct: int = 20,
    ):
        self._klines = klines
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._trend_period = trend_period
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier
        self._max_hold_bars = max_hold_bars
        self._leverage = leverage
        self._bbw_threshold_pct = bbw_threshold_pct
        self._position_size_pct = Decimal("0.1")

        self._position: Optional[dict] = None
        self._trades: list[dict] = []
        self._bbw_history: list[Decimal] = []

    def _calculate_bollinger(self, idx: int) -> Optional[dict]:
        """Calculate Bollinger Bands at index."""
        if idx < self._bb_period:
            return None

        closes = [self._klines[j].close for j in range(idx - self._bb_period + 1, idx + 1)]
        middle = sum(closes) / Decimal(len(closes))
        variance = sum((c - middle) ** 2 for c in closes) / Decimal(len(closes))
        std = Decimal(str(math.sqrt(float(variance))))

        upper = middle + std * self._bb_std
        lower = middle - std * self._bb_std

        return {"upper": upper, "middle": middle, "lower": lower}

    def _calculate_sma(self, idx: int, period: int) -> Optional[Decimal]:
        """Calculate SMA at index."""
        if idx < period:
            return None
        closes = [self._klines[j].close for j in range(idx - period + 1, idx + 1)]
        return sum(closes) / Decimal(len(closes))

    def _calculate_atr(self, idx: int) -> Optional[Decimal]:
        """Calculate ATR at index."""
        if idx < self._atr_period + 1:
            return None

        true_ranges = []
        for j in range(idx - self._atr_period, idx):
            kline = self._klines[j + 1]
            prev_close = self._klines[j].close
            tr1 = kline.high - kline.low
            tr2 = abs(kline.high - prev_close)
            tr3 = abs(kline.low - prev_close)
            true_ranges.append(max(tr1, tr2, tr3))

        return sum(true_ranges) / Decimal(len(true_ranges))

    def _is_squeeze(self, bbw: Decimal) -> bool:
        """Check if in BBW squeeze (for BREAKOUT, we want expansion, not squeeze)."""
        if len(self._bbw_history) < 50:
            return True  # Not enough history, assume squeeze

        sorted_bbw = sorted(self._bbw_history)
        rank = sum(1 for v in sorted_bbw if v < bbw)
        percentile = (rank / len(sorted_bbw)) * 100

        # For BREAKOUT: squeeze = low volatility, skip
        return percentile < self._bbw_threshold_pct

    def run(self) -> BacktestResult:
        """Run backtest."""
        result = BacktestResult()

        min_period = max(self._bb_period, self._trend_period, self._atr_period) + 50

        if len(self._klines) < min_period:
            return result

        self._position = None
        self._trades = []
        self._bbw_history = []

        equity_curve = []
        peak_equity = Decimal("10000")
        current_equity = Decimal("10000")
        max_dd = Decimal("0")
        daily_pnl = {}

        # Initialize BBW history
        for i in range(self._bb_period, min_period):
            bands = self._calculate_bollinger(i)
            if bands:
                bbw = (bands["upper"] - bands["lower"]) / bands["middle"]
                self._bbw_history.append(bbw)

        # Track max/min price for trailing stop
        max_price_since_entry = None
        min_price_since_entry = None

        for i in range(min_period, len(self._klines)):
            kline = self._klines[i]
            current_price = kline.close
            date_key = kline.close_time.strftime("%Y-%m-%d")

            bands = self._calculate_bollinger(i)
            trend_sma = self._calculate_sma(i, self._trend_period)
            atr = self._calculate_atr(i)

            if not all([bands, trend_sma, atr]):
                continue

            # Update BBW history
            bbw = (bands["upper"] - bands["lower"]) / bands["middle"]
            self._bbw_history.append(bbw)
            if len(self._bbw_history) > 200:
                self._bbw_history = self._bbw_history[-200:]

            # Check existing position
            if self._position:
                # Update extremes
                if max_price_since_entry is None or current_price > max_price_since_entry:
                    max_price_since_entry = current_price
                if min_price_since_entry is None or current_price < min_price_since_entry:
                    min_price_since_entry = current_price

                should_exit = False
                exit_price = current_price
                exit_reason = ""
                side = self._position["side"]
                entry_price = self._position["entry_price"]

                # BREAKOUT EXIT LOGIC:
                # 1. Trailing stop (ATR-based)
                trailing_distance = atr * self._atr_multiplier
                if side == "long":
                    trailing_stop = max_price_since_entry - trailing_distance
                    if current_price <= trailing_stop:
                        should_exit, exit_price, exit_reason = True, current_price, "trailing_stop"
                else:
                    trailing_stop = min_price_since_entry + trailing_distance
                    if current_price >= trailing_stop:
                        should_exit, exit_price, exit_reason = True, current_price, "trailing_stop"

                # 2. Reverse signal (price breaks opposite band)
                if not should_exit:
                    if side == "long" and current_price < bands["lower"]:
                        should_exit, exit_price, exit_reason = True, current_price, "reverse_signal"
                    elif side == "short" and current_price > bands["upper"]:
                        should_exit, exit_price, exit_reason = True, current_price, "reverse_signal"

                # 3. Stop loss (fallback)
                if not should_exit:
                    sl = self._position["stop_loss"]
                    if side == "long":
                        if current_price <= sl:
                            should_exit, exit_price, exit_reason = True, sl, "stop_loss"
                    else:
                        if current_price >= sl:
                            should_exit, exit_price, exit_reason = True, sl, "stop_loss"

                # 4. Timeout
                if not should_exit:
                    hold_bars = i - self._position["entry_bar"]
                    if hold_bars >= self._max_hold_bars:
                        should_exit, exit_price, exit_reason = True, current_price, "timeout"

                if should_exit:
                    quantity = self._position["quantity"]
                    if side == "long":
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - exit_price) * quantity

                    pnl *= Decimal(self._leverage)
                    exit_fee = (exit_price * quantity) * self.FEE_RATE
                    total_fee = self._position["entry_fee"] + exit_fee
                    net_pnl = pnl - total_fee

                    self._trades.append({
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": net_pnl,
                        "exit_reason": exit_reason,
                    })

                    if date_key not in daily_pnl:
                        daily_pnl[date_key] = Decimal("0")
                    daily_pnl[date_key] += net_pnl

                    current_equity += net_pnl
                    self._position = None
                    max_price_since_entry = None
                    min_price_since_entry = None
                    continue

            # BREAKOUT ENTRY LOGIC
            if not self._position:
                # Skip if in squeeze (low volatility)
                if self._is_squeeze(bbw):
                    continue

                signal = None
                entry_price = current_price

                # BREAKOUT: price breaks above upper band -> LONG
                if current_price > bands["upper"]:
                    signal = "long"
                # BREAKOUT: price breaks below lower band -> SHORT
                elif current_price < bands["lower"]:
                    signal = "short"
                else:
                    continue

                # Trend filter: skip counter-trend trades
                # For breakout: go with the trend
                if signal == "long" and current_price < trend_sma:
                    continue  # Skip long if below trend
                elif signal == "short" and current_price > trend_sma:
                    continue  # Skip short if above trend

                # Calculate stop loss (ATR-based)
                atr_stop = atr * self._atr_multiplier
                if signal == "long":
                    stop_loss = entry_price - atr_stop
                else:
                    stop_loss = entry_price + atr_stop

                # Enter position
                notional = Decimal("10000") * self._position_size_pct
                quantity = notional / entry_price

                self._position = {
                    "side": signal,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "quantity": quantity,
                    "entry_bar": i,
                    "entry_fee": notional * self.FEE_RATE,
                }
                max_price_since_entry = entry_price
                min_price_since_entry = entry_price

            # Track equity
            realized = sum(t["pnl"] for t in self._trades)
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = (peak_equity - current_equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

        # Calculate results
        result.total_trades = len(self._trades)

        if result.total_trades > 0:
            wins = [t for t in self._trades if t["pnl"] > 0]
            losses = [t for t in self._trades if t["pnl"] <= 0]

            result.num_wins = len(wins)
            result.num_losses = len(losses)
            result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades))

            gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
            gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

            if gross_loss > 0:
                result.profit_factor = gross_profit / gross_loss

        total_pnl = sum(t["pnl"] for t in self._trades)
        result.total_profit = total_pnl / Decimal("10000")  # As percentage
        result.max_drawdown = max_dd

        # Sharpe ratio
        if daily_pnl and len(daily_pnl) >= 2:
            returns = list(daily_pnl.values())
            mean_return = sum(returns) / Decimal(len(returns))
            variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)

            if variance > 0:
                std_dev = Decimal(str(math.sqrt(float(variance))))
                if std_dev > 0:
                    result.sharpe_ratio = (mean_return / std_dev) * Decimal(str(math.sqrt(252)))

        return result


async def fetch_klines(symbol: str, interval: str, days: int) -> list[Kline]:
    """Fetch historical klines."""
    print(f"  正在獲取 {symbol} {interval} K 線數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        await api.ping()

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        interval_map = {
            "15m": KlineInterval.m15,
            "1h": KlineInterval.h1,
        }
        kline_interval = interval_map.get(interval, KlineInterval.m15)

        all_klines = []
        current_start = start_time

        while current_start < end_time:
            klines = await api.get_klines(
                symbol=symbol,
                interval=kline_interval,
                start_time=current_start,
                end_time=end_time,
                limit=1500,
            )

            if not klines:
                break

            all_klines.extend(klines)

            last_close_time = int(klines[-1].close_time.timestamp() * 1000)
            current_start = last_close_time + 1

            print(f"    已獲取 {len(all_klines)} 根 K 線", end='\r')
            await asyncio.sleep(0.1)

        print(f"    完成: {len(all_klines)} 根 K 線                    ")
        return all_klines


async def main():
    """Run validated parameters backtest."""
    print("\n")
    print("=" * 70)
    print("       驗證參數回測 - Walk-Forward Validated")
    print("=" * 70)
    print("  Bollinger Bot BREAKOUT: BB(15, 1.5), 2x, 趨勢過濾")
    print("  預期: 83% 一致性, Sharpe 5.45, 年化 ~109%")
    print("=" * 70)

    # Fetch data
    print("\n正在獲取歷史數據...")
    klines = await fetch_klines("BTCUSDT", "15m", 365)

    if len(klines) < 500:
        print("數據不足")
        return

    # ==========================================================================
    # Walk-Forward Validation (6 periods)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  Walk-Forward 驗證 (6 個期間)")
    print("=" * 70)

    total_bars = len(klines)
    period_size = total_bars // 6

    wf_results = []

    for i in range(6):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 5 else total_bars

        period_klines = klines[start_idx:end_idx]

        if len(period_klines) < 200:
            continue

        start_date = period_klines[0].open_time.strftime('%Y-%m-%d')
        end_date = period_klines[-1].close_time.strftime('%Y-%m-%d')

        # Run backtest with validated parameters
        backtest = BollingerBreakoutBacktest(
            klines=period_klines,
            bb_period=15,
            bb_std=Decimal("1.5"),
            trend_period=50,
            atr_period=14,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=2,
            bbw_threshold_pct=20,
        )
        result = backtest.run()

        ret = float(result.total_profit) * 100
        profitable = ret > 0

        status = "✅" if profitable else "❌"
        print(f"  期間 {i+1}: {start_date} ~ {end_date}")
        print(f"    {status} 報酬: {ret:+.2f}% | 交易: {result.total_trades} | 勝率: {float(result.win_rate)*100:.0f}% | Sharpe: {float(result.sharpe_ratio):.2f}")

        wf_results.append({
            "period": i + 1,
            "return": ret,
            "profitable": profitable,
            "trades": result.total_trades,
            "win_rate": float(result.win_rate) * 100,
            "sharpe": float(result.sharpe_ratio),
        })

    # Summary
    if wf_results:
        profitable_periods = sum(1 for r in wf_results if r['profitable'])
        consistency = profitable_periods / len(wf_results) * 100
        avg_return = sum(r['return'] for r in wf_results) / len(wf_results)
        total_trades = sum(r['trades'] for r in wf_results)
        avg_sharpe = sum(r['sharpe'] for r in wf_results) / len(wf_results)

        print(f"\n{'='*70}")
        print(f"  Walk-Forward 結果總結")
        print(f"{'='*70}")
        print(f"  獲利期數: {profitable_periods}/{len(wf_results)}")
        print(f"  一致性: {consistency:.0f}%")
        print(f"  平均報酬: {avg_return:+.2f}%")
        print(f"  平均 Sharpe: {avg_sharpe:.2f}")
        print(f"  總交易數: {total_trades}")

        # Assessment
        print(f"\n  評估:")
        if consistency >= 67 and avg_sharpe > 0:
            print(f"  ✅ 策略通過驗證 (一致性 >= 67%, Sharpe > 0)")
        else:
            print(f"  ⚠️ 策略未達標準 (需要一致性 >= 67% 且 Sharpe > 0)")

    # ==========================================================================
    # Full Period Test
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"  完整期間測試 (1 年)")
    print(f"{'='*70}")

    backtest = BollingerBreakoutBacktest(
        klines=klines,
        bb_period=15,
        bb_std=Decimal("1.5"),
        trend_period=50,
        atr_period=14,
        atr_multiplier=Decimal("2.0"),
        max_hold_bars=24,
        leverage=2,
        bbw_threshold_pct=20,
    )
    result = backtest.run()

    print(f"  總報酬率: {float(result.total_profit)*100:+.2f}%")
    print(f"  Sharpe Ratio: {float(result.sharpe_ratio):.2f}")
    print(f"  勝率: {float(result.win_rate)*100:.1f}%")
    print(f"  總交易數: {result.total_trades}")
    print(f"  獲利因子: {float(result.profit_factor):.2f}")
    print(f"  最大回撤: {float(result.max_drawdown)*100:.2f}%")

    print("\n" + "=" * 70)
    print("       回測完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
