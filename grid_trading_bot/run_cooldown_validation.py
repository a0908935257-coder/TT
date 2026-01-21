#!/usr/bin/env python3
"""
Cooldown Mechanism Validation Backtest.

Compare performance with and without signal cooldown:
- RSI Bot: cooldown 0 vs cooldown 3
- Bollinger Bot: cooldown 1 vs cooldown 3

This validates that the cooldown changes don't significantly degrade performance.
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


# =============================================================================
# RSI Backtest with Cooldown
# =============================================================================

class RSIBacktestWithCooldown:
    """RSI Backtest with configurable cooldown."""

    FEE_RATE = Decimal("0.0004")

    def __init__(
        self,
        klines: list[Kline],
        cooldown_bars: int = 0,
        rsi_period: int = 21,
        oversold: int = 30,
        overbought: int = 70,
        exit_level: int = 50,
        leverage: int = 5,
        stop_loss_pct: Decimal = Decimal("0.02"),
        take_profit_pct: Decimal = Decimal("0.04"),
    ):
        self._klines = klines
        self._cooldown_bars = cooldown_bars
        self._rsi_period = rsi_period
        self._oversold = oversold
        self._overbought = overbought
        self._exit_level = exit_level
        self._leverage = leverage
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._position_size_pct = Decimal("0.1")

        self._position: Optional[dict] = None
        self._trades: list[dict] = []
        self._current_cooldown: int = 0

        # RSI state
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None

    def _calculate_rsi(self, closes: list[float]) -> float:
        """Calculate RSI using Wilder's smoothing."""
        period = self._rsi_period

        if len(closes) < period + 1:
            return 50.0

        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        if self._avg_gain is None:
            self._avg_gain = sum(gains[-period:]) / period
            self._avg_loss = sum(losses[-period:]) / period
        else:
            current_gain = gains[-1] if gains else 0
            current_loss = losses[-1] if losses else 0
            self._avg_gain = (self._avg_gain * (period - 1) + current_gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + current_loss) / period

        if self._avg_loss == 0:
            return 100.0

        rs = self._avg_gain / self._avg_loss
        return 100 - (100 / (1 + rs))

    def run(self) -> BacktestResult:
        """Run backtest."""
        result = BacktestResult()

        if len(self._klines) < self._rsi_period + 10:
            return result

        # Reset state
        self._avg_gain = None
        self._avg_loss = None
        self._position = None
        self._trades = []
        self._current_cooldown = 0

        equity_curve = []
        peak_equity = Decimal("10000")
        current_equity = Decimal("10000")
        max_dd = Decimal("0")
        daily_returns = []
        prev_equity = current_equity

        closes = []

        for i, kline in enumerate(self._klines):
            close = float(kline.close)
            closes.append(close)

            if len(closes) < self._rsi_period + 5:
                continue

            rsi = self._calculate_rsi(closes)

            # Check position exit
            if self._position:
                entry_price = self._position['entry_price']
                side = self._position['side']
                price_change = (Decimal(str(close)) - entry_price) / entry_price

                exit_reason = None

                if side == 'long':
                    if price_change <= -self._stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif price_change >= self._take_profit_pct:
                        exit_reason = 'take_profit'
                    elif rsi >= self._exit_level:
                        exit_reason = 'rsi_exit'
                else:
                    if price_change >= self._stop_loss_pct:
                        exit_reason = 'stop_loss'
                    elif price_change <= -self._take_profit_pct:
                        exit_reason = 'take_profit'
                    elif rsi <= self._exit_level:
                        exit_reason = 'rsi_exit'

                if exit_reason:
                    exit_price = Decimal(str(close))

                    if side == 'long':
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price

                    leveraged_pnl = pnl * self._leverage * self._position_size_pct
                    fee = self.FEE_RATE * 2 * self._leverage * self._position_size_pct
                    net_pnl = leveraged_pnl - fee

                    self._trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'side': side,
                        'pnl': net_pnl,
                    })

                    current_equity += current_equity * net_pnl
                    self._position = None

                    # Set cooldown after exit
                    self._current_cooldown = self._cooldown_bars

            # Update cooldown
            if self._current_cooldown > 0:
                self._current_cooldown -= 1

            # Entry signals (only if not in position and cooldown expired)
            if not self._position and self._current_cooldown == 0:
                if rsi < self._oversold:
                    self._position = {
                        'entry_price': Decimal(str(close)),
                        'side': 'long',
                    }
                    self._current_cooldown = self._cooldown_bars
                elif rsi > self._overbought:
                    self._position = {
                        'entry_price': Decimal(str(close)),
                        'side': 'short',
                    }
                    self._current_cooldown = self._cooldown_bars

            equity_curve.append(current_equity)

            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = (peak_equity - current_equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

            if i % 96 == 0 and prev_equity > 0:
                daily_return = float((current_equity - prev_equity) / prev_equity)
                daily_returns.append(daily_return)
                prev_equity = current_equity

        # Close remaining position
        if self._position:
            close = float(self._klines[-1].close)
            exit_price = Decimal(str(close))
            entry_price = self._position['entry_price']
            side = self._position['side']

            if side == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            leveraged_pnl = pnl * self._leverage * self._position_size_pct
            fee = self.FEE_RATE * 2 * self._leverage * self._position_size_pct
            net_pnl = leveraged_pnl - fee

            self._trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': side,
                'pnl': net_pnl,
            })
            current_equity += current_equity * net_pnl

        # Calculate results
        result.total_trades = len(self._trades)

        if result.total_trades > 0:
            wins = [t for t in self._trades if t['pnl'] > 0]
            losses = [t for t in self._trades if t['pnl'] <= 0]

            result.win_rate = Decimal(str(len(wins) / result.total_trades))

            total_win = sum(t['pnl'] for t in wins) if wins else Decimal("0")
            total_loss = abs(sum(t['pnl'] for t in losses)) if losses else Decimal("0")

            if total_loss > 0:
                result.profit_factor = total_win / total_loss

        result.total_profit = (current_equity - Decimal("10000")) / Decimal("10000")
        result.max_drawdown = max_dd

        if len(daily_returns) > 1:
            avg_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_dev = math.sqrt(variance) if variance > 0 else 0.001
            if std_dev > 0:
                result.sharpe_ratio = Decimal(str((avg_return / std_dev) * math.sqrt(365)))

        return result


# =============================================================================
# Bollinger Backtest with Cooldown
# =============================================================================

class BollingerBacktestWithCooldown:
    """Bollinger Backtest with configurable cooldown."""

    FEE_RATE = Decimal("0.0004")

    def __init__(
        self,
        klines: list[Kline],
        cooldown_bars: int = 1,
        bb_period: int = 15,
        bb_std: Decimal = Decimal("1.5"),
        trend_period: int = 50,
        use_trend_filter: bool = True,
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal("2.0"),
        max_hold_bars: int = 24,
        leverage: int = 2,
    ):
        self._klines = klines
        self._cooldown_bars = cooldown_bars
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._trend_period = trend_period
        self._use_trend_filter = use_trend_filter
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier
        self._max_hold_bars = max_hold_bars
        self._leverage = leverage
        self._position_size_pct = Decimal("0.1")

        self._position: Optional[dict] = None
        self._trades: list[dict] = []
        self._current_cooldown: int = 0
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
        """Check if in BBW squeeze."""
        if len(self._bbw_history) < 50:
            return False

        sorted_bbw = sorted(self._bbw_history)
        rank = sum(1 for v in sorted_bbw if v < bbw)
        percentile = (rank / len(sorted_bbw)) * 100

        return percentile < 15

    def run(self) -> BacktestResult:
        """Run backtest."""
        result = BacktestResult()

        min_period = max(self._bb_period, self._trend_period, self._atr_period) + 50

        if len(self._klines) < min_period:
            return result

        # Reset state
        self._position = None
        self._trades = []
        self._current_cooldown = 0
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
            if len(self._bbw_history) > 100:
                self._bbw_history = self._bbw_history[-100:]

            # Check existing position
            if self._position:
                should_exit = False
                exit_price = current_price
                exit_reason = ""
                side = self._position["side"]
                entry_price = self._position["entry_price"]

                # Take profit - middle band
                if side == "long":
                    if current_price >= bands["middle"]:
                        should_exit, exit_price, exit_reason = True, bands["middle"], "take_profit"
                else:
                    if current_price <= bands["middle"]:
                        should_exit, exit_price, exit_reason = True, bands["middle"], "take_profit"

                # Stop loss
                if not should_exit:
                    sl = self._position["stop_loss"]
                    if side == "long":
                        if current_price <= sl:
                            should_exit, exit_price, exit_reason = True, sl, "stop_loss"
                    else:
                        if current_price >= sl:
                            should_exit, exit_price, exit_reason = True, sl, "stop_loss"

                # Timeout
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
                    })

                    if date_key not in daily_pnl:
                        daily_pnl[date_key] = Decimal("0")
                    daily_pnl[date_key] += net_pnl

                    self._position = None
                    self._current_cooldown = self._cooldown_bars
                    continue

            # Update cooldown
            if self._current_cooldown > 0:
                self._current_cooldown -= 1

            # Entry signals
            if not self._position and self._current_cooldown == 0:
                # BBW squeeze filter
                if self._is_squeeze(bbw):
                    continue

                signal = None
                entry_price = None

                if current_price <= bands["lower"]:
                    signal = "long"
                    entry_price = bands["lower"]
                elif current_price >= bands["upper"]:
                    signal = "short"
                    entry_price = bands["upper"]
                else:
                    continue

                # Trend filter
                if self._use_trend_filter:
                    if signal == "long" and current_price < trend_sma:
                        continue
                    elif signal == "short" and current_price > trend_sma:
                        continue

                # Calculate stop loss
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
                self._current_cooldown = self._cooldown_bars

            # Track equity
            realized = sum(t["pnl"] for t in self._trades)
            equity_curve.append(realized)

            if realized > peak_equity - Decimal("10000"):
                peak_equity = realized + Decimal("10000")
            dd = (peak_equity - Decimal("10000")) - realized
            if dd > max_dd:
                max_dd = dd

        # Calculate results
        result.total_trades = len(self._trades)

        if result.total_trades > 0:
            wins = [t for t in self._trades if t["pnl"] > 0]
            losses = [t for t in self._trades if t["pnl"] <= 0]

            result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades))

            gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
            gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

            if gross_loss > 0:
                result.profit_factor = gross_profit / gross_loss

        # Calculate total profit as percentage
        total_pnl = sum(t["pnl"] for t in self._trades)
        result.total_profit = total_pnl / Decimal("10000")  # Convert to percentage
        result.max_drawdown = max_dd / Decimal("10000")  # Convert to percentage

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


# =============================================================================
# Main
# =============================================================================

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


def print_comparison(name: str, result_no_cooldown: BacktestResult, result_with_cooldown: BacktestResult, cooldown: int):
    """Print comparison between no cooldown and with cooldown."""
    print(f"\n{'='*70}")
    print(f"  {name} - 冷卻機制影響分析")
    print(f"{'='*70}")
    print(f"  {'指標':<20} {'無冷卻':>18} {'冷卻={0}根K線':>18}".format(cooldown))
    print(f"  {'-'*58}")
    print(f"  {'總交易數':<20} {result_no_cooldown.total_trades:>18} {result_with_cooldown.total_trades:>18}")
    print(f"  {'總報酬率':<20} {float(result_no_cooldown.total_profit)*100:>17.2f}% {float(result_with_cooldown.total_profit)*100:>17.2f}%")
    print(f"  {'勝率':<20} {float(result_no_cooldown.win_rate)*100:>17.1f}% {float(result_with_cooldown.win_rate)*100:>17.1f}%")
    print(f"  {'Sharpe Ratio':<20} {float(result_no_cooldown.sharpe_ratio):>18.2f} {float(result_with_cooldown.sharpe_ratio):>18.2f}")
    print(f"  {'獲利因子':<20} {float(result_no_cooldown.profit_factor):>18.2f} {float(result_with_cooldown.profit_factor):>18.2f}")
    print(f"  {'最大回撤':<20} {float(result_no_cooldown.max_drawdown)*100:>17.2f}% {float(result_with_cooldown.max_drawdown)*100:>17.2f}%")
    print(f"  {'-'*58}")

    # Calculate changes
    trade_change = result_with_cooldown.total_trades - result_no_cooldown.total_trades
    trade_pct = (trade_change / result_no_cooldown.total_trades * 100) if result_no_cooldown.total_trades > 0 else 0

    sharpe_change = float(result_with_cooldown.sharpe_ratio) - float(result_no_cooldown.sharpe_ratio)

    print(f"  交易數變化: {trade_change:+d} ({trade_pct:+.1f}%)")
    print(f"  Sharpe 變化: {sharpe_change:+.2f}")

    # Assessment
    if sharpe_change >= -0.1:
        print(f"\n  ✅ 結論: 冷卻機制對績效影響可接受")
    else:
        print(f"\n  ⚠️  結論: 冷卻機制可能影響績效，建議調整")


async def main():
    """Run cooldown validation."""
    print("\n")
    print("=" * 70)
    print("       冷卻機制驗證回測")
    print("=" * 70)
    print("  比較訊號冷卻機制變更前後的績效差異")
    print("=" * 70)

    # Fetch data
    print("\n正在獲取歷史數據...")
    klines = await fetch_klines("BTCUSDT", "15m", 180)

    if len(klines) < 500:
        print("數據不足")
        return

    # ==========================================================================
    # RSI Bot Validation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  RSI Bot 冷卻機制驗證")
    print("=" * 70)

    print("\n正在運行 RSI Bot 回測 (無冷卻)...")
    rsi_no_cooldown = RSIBacktestWithCooldown(klines, cooldown_bars=0)
    rsi_result_no = rsi_no_cooldown.run()

    print("正在運行 RSI Bot 回測 (冷卻=3)...")
    rsi_with_cooldown = RSIBacktestWithCooldown(klines, cooldown_bars=3)
    rsi_result_with = rsi_with_cooldown.run()

    print_comparison("RSI Bot", rsi_result_no, rsi_result_with, 3)

    # ==========================================================================
    # Bollinger Bot Validation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  Bollinger Bot 冷卻機制驗證")
    print("=" * 70)

    print("\n正在運行 Bollinger Bot 回測 (冷卻=1)...")
    bb_cooldown_1 = BollingerBacktestWithCooldown(klines, cooldown_bars=1)
    bb_result_1 = bb_cooldown_1.run()

    print("正在運行 Bollinger Bot 回測 (冷卻=3)...")
    bb_cooldown_3 = BollingerBacktestWithCooldown(klines, cooldown_bars=3)
    bb_result_3 = bb_cooldown_3.run()

    print_comparison("Bollinger Bot", bb_result_1, bb_result_3, 3)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("       驗證總結")
    print("=" * 70)

    print("\n  修改內容:")
    print("    - RSI Bot: 新增訊號冷卻 (0 → 3 根 K 線)")
    print("    - Bollinger Bot: 調整訊號冷卻 (1 → 3 根 K 線)")

    print("\n  冷卻機制目的:")
    print("    - 防止快速連續進場導致過度交易")
    print("    - 給予市場足夠時間確認趨勢")
    print("    - 減少假突破造成的連續虧損")

    # Overall assessment
    rsi_sharpe_ok = float(rsi_result_with.sharpe_ratio) >= float(rsi_result_no.sharpe_ratio) - 0.2
    bb_sharpe_ok = float(bb_result_3.sharpe_ratio) >= float(bb_result_1.sharpe_ratio) - 0.2

    print("\n  驗證結果:")
    if rsi_sharpe_ok and bb_sharpe_ok:
        print("    ✅ RSI Bot: 冷卻機制變更通過")
        print("    ✅ Bollinger Bot: 冷卻機制變更通過")
        print("\n  🎯 所有變更驗證通過，可安全使用新版本")
    else:
        if not rsi_sharpe_ok:
            print("    ⚠️  RSI Bot: 需要關注績效變化")
        if not bb_sharpe_ok:
            print("    ⚠️  Bollinger Bot: 需要關注績效變化")

    print("\n" + "=" * 70)
    print("       驗證完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
