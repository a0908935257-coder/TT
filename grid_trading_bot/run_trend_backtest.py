#!/usr/bin/env python3
"""
Trend Following Strategy Backtest.

Tests multiple trend following strategies:
1. Supertrend
2. MACD
3. Dual EMA Crossover

Target: High frequency (>100 trades/year) + Sharpe > 1
"""

import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, List
from enum import Enum

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class TrendBacktestResult:
    """Trend strategy backtest result."""
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_holding_time: timedelta = field(default_factory=lambda: timedelta(0))


class SupertrendBacktest:
    """Supertrend strategy backtest."""

    FEE_RATE = Decimal("0.0004")

    def __init__(
        self,
        klines: List[Kline],
        atr_period: int = 10,
        atr_multiplier: Decimal = Decimal("3.0"),
        leverage: int = 1,
        position_size_pct: Decimal = Decimal("0.1"),
    ):
        self._klines = klines
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier
        self._leverage = leverage
        self._position_size_pct = position_size_pct

        self._position: Optional[dict] = None
        self._trades: List[dict] = []
        self._equity_curve: List[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}

    def _calculate_atr(self, idx: int) -> Optional[Decimal]:
        if idx < self._atr_period + 1:
            return None

        true_ranges = []
        for j in range(idx - self._atr_period, idx):
            kline = self._klines[j + 1]
            prev_close = self._klines[j].close
            tr = max(
                kline.high - kline.low,
                abs(kline.high - prev_close),
                abs(kline.low - prev_close)
            )
            true_ranges.append(tr)

        return sum(true_ranges) / Decimal(len(true_ranges))

    def _calculate_supertrend(self, idx: int, prev_supertrend: dict) -> Optional[dict]:
        atr = self._calculate_atr(idx)
        if atr is None:
            return None

        kline = self._klines[idx]
        hl2 = (kline.high + kline.low) / 2

        upper_band = hl2 + self._atr_multiplier * atr
        lower_band = hl2 - self._atr_multiplier * atr

        # Adjust bands based on previous values
        if prev_supertrend:
            prev_upper = prev_supertrend["upper_band"]
            prev_lower = prev_supertrend["lower_band"]
            prev_close = self._klines[idx - 1].close

            if prev_close > prev_upper:
                lower_band = max(lower_band, prev_lower)
            if prev_close < prev_lower:
                upper_band = min(upper_band, prev_upper)

        # Determine trend
        close = kline.close
        if prev_supertrend:
            prev_trend = prev_supertrend["trend"]
            if prev_trend == 1:  # Was bullish
                trend = 1 if close > prev_supertrend["lower_band"] else -1
            else:  # Was bearish
                trend = -1 if close < prev_supertrend["upper_band"] else 1
        else:
            trend = 1 if close > upper_band else -1

        supertrend_value = lower_band if trend == 1 else upper_band

        return {
            "upper_band": upper_band,
            "lower_band": lower_band,
            "trend": trend,
            "value": supertrend_value,
        }

    def run(self) -> TrendBacktestResult:
        min_period = self._atr_period + 10
        if len(self._klines) < min_period:
            return TrendBacktestResult()

        prev_supertrend = None
        prev_trend = 0

        for i in range(min_period, len(self._klines)):
            kline = self._klines[i]
            date_key = kline.close_time.strftime("%Y-%m-%d")

            supertrend = self._calculate_supertrend(i, prev_supertrend)
            if supertrend is None:
                continue

            current_trend = supertrend["trend"]

            # Check for trend change (signal)
            if prev_trend != 0 and current_trend != prev_trend:
                # Close existing position
                if self._position:
                    self._close_position(kline.close, kline.close_time, i, date_key)

                # Open new position
                if current_trend == 1:  # Bullish
                    self._open_position(PositionSide.LONG, kline.close, i, kline.close_time)
                else:  # Bearish
                    self._open_position(PositionSide.SHORT, kline.close, i, kline.close_time)

            prev_supertrend = supertrend
            prev_trend = current_trend
            self._update_equity(kline.close)

        # Close final position
        if self._position:
            last_kline = self._klines[-1]
            self._close_position(
                last_kline.close,
                last_kline.close_time,
                len(self._klines) - 1,
                last_kline.close_time.strftime("%Y-%m-%d")
            )

        return self._calculate_result()

    def _open_position(self, side: PositionSide, price: Decimal, bar_idx: int, time: datetime):
        notional = Decimal("10000") * self._position_size_pct
        quantity = notional / price
        self._position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "entry_bar": bar_idx,
            "entry_time": time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _close_position(self, price: Decimal, time: datetime, bar_idx: int, date_key: str):
        if not self._position:
            return

        side = self._position["side"]
        entry_price = self._position["entry_price"]
        quantity = self._position["quantity"]

        if side == PositionSide.LONG:
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity

        pnl *= Decimal(self._leverage)
        exit_fee = (price * quantity) * self.FEE_RATE
        total_fee = self._position["entry_fee"] + exit_fee
        net_pnl = pnl - total_fee

        self._trades.append({
            "side": side,
            "entry_price": entry_price,
            "exit_price": price,
            "quantity": quantity,
            "pnl": net_pnl,
            "fee": total_fee,
            "entry_time": self._position["entry_time"],
            "exit_time": time,
            "holding_bars": bar_idx - self._position["entry_bar"],
        })

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        self._position = None

    def _update_equity(self, current_price: Decimal):
        realized = sum(t["pnl"] for t in self._trades)
        unrealized = Decimal("0")
        if self._position:
            side = self._position["side"]
            entry_price = self._position["entry_price"]
            quantity = self._position["quantity"]
            if side == PositionSide.LONG:
                unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = (entry_price - current_price) * quantity
            unrealized *= Decimal(self._leverage)
        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> TrendBacktestResult:
        result = TrendBacktestResult()
        if not self._trades:
            return result

        result.total_profit = sum(t["pnl"] for t in self._trades)
        result.total_trades = len(self._trades)

        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)
        result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * 100 if self._trades else Decimal("0")

        if wins:
            result.avg_win = sum(t["pnl"] for t in wins) / Decimal(len(wins))
        if losses:
            result.avg_loss = abs(sum(t["pnl"] for t in losses)) / Decimal(len(losses))

        gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss

        result.long_trades = len([t for t in self._trades if t["side"] == PositionSide.LONG])
        result.short_trades = len([t for t in self._trades if t["side"] == PositionSide.SHORT])

        # Max drawdown
        if self._equity_curve:
            peak = self._equity_curve[0]
            max_dd = Decimal("0")
            for equity in self._equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown = max_dd

        # Avg holding time
        if self._trades:
            total_bars = sum(t["holding_bars"] for t in self._trades)
            avg_bars = total_bars / len(self._trades)
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe()

        return result

    def _calculate_sharpe(self) -> Decimal:
        if not self._daily_pnl or len(self._daily_pnl) < 2:
            return Decimal("0")

        returns = list(self._daily_pnl.values())
        mean_return = sum(returns) / Decimal(len(returns))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)

        if variance <= 0:
            return Decimal("0")

        std_dev = Decimal(str(math.sqrt(float(variance))))
        if std_dev == 0:
            return Decimal("0")

        return (mean_return / std_dev) * Decimal(str(math.sqrt(252)))


class MACDBacktest:
    """MACD crossover strategy backtest."""

    FEE_RATE = Decimal("0.0004")

    def __init__(
        self,
        klines: List[Kline],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        leverage: int = 1,
        position_size_pct: Decimal = Decimal("0.1"),
    ):
        self._klines = klines
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period
        self._leverage = leverage
        self._position_size_pct = position_size_pct

        self._position: Optional[dict] = None
        self._trades: List[dict] = []
        self._equity_curve: List[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}

    def _calculate_ema(self, prices: List[Decimal], period: int) -> List[Decimal]:
        if len(prices) < period:
            return []

        ema = []
        multiplier = Decimal("2") / Decimal(period + 1)

        # First EMA is SMA
        sma = sum(prices[:period]) / Decimal(period)
        ema.append(sma)

        for i in range(period, len(prices)):
            new_ema = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(new_ema)

        return ema

    def _calculate_macd(self) -> tuple[List[Decimal], List[Decimal], List[Decimal]]:
        closes = [k.close for k in self._klines]

        fast_ema = self._calculate_ema(closes, self._fast_period)
        slow_ema = self._calculate_ema(closes, self._slow_period)

        # Align EMAs
        offset = self._slow_period - self._fast_period
        fast_ema = fast_ema[offset:]

        # MACD line
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

        # Signal line
        signal_line = self._calculate_ema(macd_line, self._signal_period)

        # Histogram
        offset2 = len(macd_line) - len(signal_line)
        macd_line = macd_line[offset2:]
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        return macd_line, signal_line, histogram

    def run(self) -> TrendBacktestResult:
        min_period = self._slow_period + self._signal_period + 10
        if len(self._klines) < min_period:
            return TrendBacktestResult()

        macd_line, signal_line, histogram = self._calculate_macd()

        # Start index in klines
        start_idx = len(self._klines) - len(histogram)

        prev_histogram = Decimal("0")

        for i, hist in enumerate(histogram):
            kline_idx = start_idx + i
            kline = self._klines[kline_idx]
            date_key = kline.close_time.strftime("%Y-%m-%d")

            # Detect crossover
            if prev_histogram != 0:
                # Bullish crossover (histogram crosses above 0)
                if prev_histogram < 0 and hist > 0:
                    if self._position:
                        self._close_position(kline.close, kline.close_time, kline_idx, date_key)
                    self._open_position(PositionSide.LONG, kline.close, kline_idx, kline.close_time)

                # Bearish crossover (histogram crosses below 0)
                elif prev_histogram > 0 and hist < 0:
                    if self._position:
                        self._close_position(kline.close, kline.close_time, kline_idx, date_key)
                    self._open_position(PositionSide.SHORT, kline.close, kline_idx, kline.close_time)

            prev_histogram = hist
            self._update_equity(kline.close)

        # Close final position
        if self._position:
            last_kline = self._klines[-1]
            self._close_position(
                last_kline.close,
                last_kline.close_time,
                len(self._klines) - 1,
                last_kline.close_time.strftime("%Y-%m-%d")
            )

        return self._calculate_result()

    def _open_position(self, side: PositionSide, price: Decimal, bar_idx: int, time: datetime):
        notional = Decimal("10000") * self._position_size_pct
        quantity = notional / price
        self._position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "entry_bar": bar_idx,
            "entry_time": time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _close_position(self, price: Decimal, time: datetime, bar_idx: int, date_key: str):
        if not self._position:
            return

        side = self._position["side"]
        entry_price = self._position["entry_price"]
        quantity = self._position["quantity"]

        if side == PositionSide.LONG:
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity

        pnl *= Decimal(self._leverage)
        exit_fee = (price * quantity) * self.FEE_RATE
        total_fee = self._position["entry_fee"] + exit_fee
        net_pnl = pnl - total_fee

        self._trades.append({
            "side": side,
            "entry_price": entry_price,
            "exit_price": price,
            "quantity": quantity,
            "pnl": net_pnl,
            "fee": total_fee,
            "entry_time": self._position["entry_time"],
            "exit_time": time,
            "holding_bars": bar_idx - self._position["entry_bar"],
        })

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        self._position = None

    def _update_equity(self, current_price: Decimal):
        realized = sum(t["pnl"] for t in self._trades)
        unrealized = Decimal("0")
        if self._position:
            side = self._position["side"]
            entry_price = self._position["entry_price"]
            quantity = self._position["quantity"]
            if side == PositionSide.LONG:
                unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = (entry_price - current_price) * quantity
            unrealized *= Decimal(self._leverage)
        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> TrendBacktestResult:
        result = TrendBacktestResult()
        if not self._trades:
            return result

        result.total_profit = sum(t["pnl"] for t in self._trades)
        result.total_trades = len(self._trades)

        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)
        result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * 100 if self._trades else Decimal("0")

        if wins:
            result.avg_win = sum(t["pnl"] for t in wins) / Decimal(len(wins))
        if losses:
            result.avg_loss = abs(sum(t["pnl"] for t in losses)) / Decimal(len(losses))

        gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss

        result.long_trades = len([t for t in self._trades if t["side"] == PositionSide.LONG])
        result.short_trades = len([t for t in self._trades if t["side"] == PositionSide.SHORT])

        if self._equity_curve:
            peak = self._equity_curve[0]
            max_dd = Decimal("0")
            for equity in self._equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown = max_dd

        if self._trades:
            total_bars = sum(t["holding_bars"] for t in self._trades)
            avg_bars = total_bars / len(self._trades)
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        result.sharpe_ratio = self._calculate_sharpe()

        return result

    def _calculate_sharpe(self) -> Decimal:
        if not self._daily_pnl or len(self._daily_pnl) < 2:
            return Decimal("0")

        returns = list(self._daily_pnl.values())
        mean_return = sum(returns) / Decimal(len(returns))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)

        if variance <= 0:
            return Decimal("0")

        std_dev = Decimal(str(math.sqrt(float(variance))))
        if std_dev == 0:
            return Decimal("0")

        return (mean_return / std_dev) * Decimal(str(math.sqrt(252)))


class DualEMABacktest:
    """Dual EMA crossover strategy."""

    FEE_RATE = Decimal("0.0004")

    def __init__(
        self,
        klines: List[Kline],
        fast_period: int = 9,
        slow_period: int = 21,
        leverage: int = 1,
        position_size_pct: Decimal = Decimal("0.1"),
    ):
        self._klines = klines
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._leverage = leverage
        self._position_size_pct = position_size_pct

        self._position: Optional[dict] = None
        self._trades: List[dict] = []
        self._equity_curve: List[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}

    def _calculate_ema(self, idx: int, period: int) -> Optional[Decimal]:
        if idx < period:
            return None

        closes = [self._klines[j].close for j in range(idx - period + 1, idx + 1)]
        multiplier = Decimal("2") / Decimal(period + 1)

        ema = sum(closes[:period]) / Decimal(period)
        for i in range(period, len(closes)):
            ema = (closes[i] - ema) * multiplier + ema

        return ema

    def run(self) -> TrendBacktestResult:
        min_period = self._slow_period + 10
        if len(self._klines) < min_period:
            return TrendBacktestResult()

        prev_fast = None
        prev_slow = None

        for i in range(min_period, len(self._klines)):
            kline = self._klines[i]
            date_key = kline.close_time.strftime("%Y-%m-%d")

            fast_ema = self._calculate_ema(i, self._fast_period)
            slow_ema = self._calculate_ema(i, self._slow_period)

            if fast_ema is None or slow_ema is None:
                continue

            # Detect crossover
            if prev_fast is not None and prev_slow is not None:
                # Bullish crossover
                if prev_fast <= prev_slow and fast_ema > slow_ema:
                    if self._position:
                        self._close_position(kline.close, kline.close_time, i, date_key)
                    self._open_position(PositionSide.LONG, kline.close, i, kline.close_time)

                # Bearish crossover
                elif prev_fast >= prev_slow and fast_ema < slow_ema:
                    if self._position:
                        self._close_position(kline.close, kline.close_time, i, date_key)
                    self._open_position(PositionSide.SHORT, kline.close, i, kline.close_time)

            prev_fast = fast_ema
            prev_slow = slow_ema
            self._update_equity(kline.close)

        # Close final position
        if self._position:
            last_kline = self._klines[-1]
            self._close_position(
                last_kline.close,
                last_kline.close_time,
                len(self._klines) - 1,
                last_kline.close_time.strftime("%Y-%m-%d")
            )

        return self._calculate_result()

    def _open_position(self, side: PositionSide, price: Decimal, bar_idx: int, time: datetime):
        notional = Decimal("10000") * self._position_size_pct
        quantity = notional / price
        self._position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "entry_bar": bar_idx,
            "entry_time": time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _close_position(self, price: Decimal, time: datetime, bar_idx: int, date_key: str):
        if not self._position:
            return

        side = self._position["side"]
        entry_price = self._position["entry_price"]
        quantity = self._position["quantity"]

        if side == PositionSide.LONG:
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity

        pnl *= Decimal(self._leverage)
        exit_fee = (price * quantity) * self.FEE_RATE
        total_fee = self._position["entry_fee"] + exit_fee
        net_pnl = pnl - total_fee

        self._trades.append({
            "side": side,
            "entry_price": entry_price,
            "exit_price": price,
            "quantity": quantity,
            "pnl": net_pnl,
            "fee": total_fee,
            "entry_time": self._position["entry_time"],
            "exit_time": time,
            "holding_bars": bar_idx - self._position["entry_bar"],
        })

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        self._position = None

    def _update_equity(self, current_price: Decimal):
        realized = sum(t["pnl"] for t in self._trades)
        unrealized = Decimal("0")
        if self._position:
            side = self._position["side"]
            entry_price = self._position["entry_price"]
            quantity = self._position["quantity"]
            if side == PositionSide.LONG:
                unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = (entry_price - current_price) * quantity
            unrealized *= Decimal(self._leverage)
        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> TrendBacktestResult:
        result = TrendBacktestResult()
        if not self._trades:
            return result

        result.total_profit = sum(t["pnl"] for t in self._trades)
        result.total_trades = len(self._trades)

        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)
        result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * 100 if self._trades else Decimal("0")

        if wins:
            result.avg_win = sum(t["pnl"] for t in wins) / Decimal(len(wins))
        if losses:
            result.avg_loss = abs(sum(t["pnl"] for t in losses)) / Decimal(len(losses))

        gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss

        result.long_trades = len([t for t in self._trades if t["side"] == PositionSide.LONG])
        result.short_trades = len([t for t in self._trades if t["side"] == PositionSide.SHORT])

        if self._equity_curve:
            peak = self._equity_curve[0]
            max_dd = Decimal("0")
            for equity in self._equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown = max_dd

        if self._trades:
            total_bars = sum(t["holding_bars"] for t in self._trades)
            avg_bars = total_bars / len(self._trades)
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        result.sharpe_ratio = self._calculate_sharpe()

        return result

    def _calculate_sharpe(self) -> Decimal:
        if not self._daily_pnl or len(self._daily_pnl) < 2:
            return Decimal("0")

        returns = list(self._daily_pnl.values())
        mean_return = sum(returns) / Decimal(len(returns))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)

        if variance <= 0:
            return Decimal("0")

        std_dev = Decimal(str(math.sqrt(float(variance))))
        if std_dev == 0:
            return Decimal("0")

        return (mean_return / std_dev) * Decimal(str(math.sqrt(252)))


async def fetch_klines(symbol: str, interval: str, days: int) -> List[Kline]:
    """Fetch historical klines from Binance Futures."""
    print(f"\n正在從 Binance 獲取 {symbol} {interval} 歷史數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        await api.ping()
        print("  Binance API 連接成功")

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        interval_map = {
            "1m": KlineInterval.m1,
            "5m": KlineInterval.m5,
            "15m": KlineInterval.m15,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
        }
        kline_interval = interval_map.get(interval, KlineInterval.m15)

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

            if klines:
                last_close_time = int(klines[-1].close_time.timestamp() * 1000)
                current_start = last_close_time + 1

            await asyncio.sleep(0.1)

        print(f"  總共獲取 {len(all_klines)} 根 K 線")
        return all_klines


def print_result(name: str, result: TrendBacktestResult, symbol: str):
    """Print backtest result."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {symbol}")
    print(f"{'='*60}")
    print(f"  交易次數:     {result.total_trades}")
    print(f"  勝率:         {result.win_rate:.1f}%")
    print(f"  多/空:        {result.long_trades}/{result.short_trades}")
    print(f"  總盈虧:       {result.total_profit:+.2f} USDT")
    print(f"  平均獲利:     {result.avg_win:+.2f} USDT")
    print(f"  平均虧損:     {result.avg_loss:.2f} USDT")
    print(f"  獲利因子:     {result.profit_factor:.2f}")
    print(f"  Sharpe:       {result.sharpe_ratio:.2f}")
    print(f"  最大回撤:     {result.max_drawdown:.2f} USDT")
    print(f"  平均持倉:     {result.avg_holding_time}")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description="Trend Following Strategy Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", default="15m", help="Kline interval")
    parser.add_argument("--days", type=int, default=730, help="Backtest days")
    args = parser.parse_args()

    print("=" * 60)
    print("  趨勢跟蹤策略回測")
    print("  目標: 高頻交易 + Sharpe > 1")
    print("=" * 60)

    klines = await fetch_klines(args.symbol, args.interval, args.days)

    if len(klines) < 100:
        print("K 線數量不足")
        return

    results = []

    # === Supertrend Tests ===
    print("\n" + "=" * 60)
    print("  Supertrend 策略測試")
    print("=" * 60)

    supertrend_configs = [
        ("ST1: 標準 (10, 3.0)", 10, Decimal("3.0")),
        ("ST2: 快速 (7, 2.0)", 7, Decimal("2.0")),
        ("ST3: 慢速 (14, 3.5)", 14, Decimal("3.5")),
        ("ST4: 敏感 (5, 1.5)", 5, Decimal("1.5")),
        ("ST5: 穩定 (20, 4.0)", 20, Decimal("4.0")),
    ]

    for name, period, mult in supertrend_configs:
        bt = SupertrendBacktest(klines, atr_period=period, atr_multiplier=mult)
        result = bt.run()
        print_result(name, result, args.symbol)
        results.append((name, result))

    # === MACD Tests ===
    print("\n" + "=" * 60)
    print("  MACD 策略測試")
    print("=" * 60)

    macd_configs = [
        ("MACD1: 標準 (12,26,9)", 12, 26, 9),
        ("MACD2: 快速 (8,17,9)", 8, 17, 9),
        ("MACD3: 慢速 (19,39,9)", 19, 39, 9),
        ("MACD4: 敏感 (5,13,6)", 5, 13, 6),
    ]

    for name, fast, slow, signal in macd_configs:
        bt = MACDBacktest(klines, fast_period=fast, slow_period=slow, signal_period=signal)
        result = bt.run()
        print_result(name, result, args.symbol)
        results.append((name, result))

    # === Dual EMA Tests ===
    print("\n" + "=" * 60)
    print("  雙 EMA 策略測試")
    print("=" * 60)

    ema_configs = [
        ("EMA1: 9/21", 9, 21),
        ("EMA2: 5/13", 5, 13),
        ("EMA3: 12/26", 12, 26),
        ("EMA4: 20/50", 20, 50),
        ("EMA5: 8/20", 8, 20),
    ]

    for name, fast, slow in ema_configs:
        bt = DualEMABacktest(klines, fast_period=fast, slow_period=slow)
        result = bt.run()
        print_result(name, result, args.symbol)
        results.append((name, result))

    # === Summary ===
    print("\n" + "=" * 60)
    print("  總結")
    print("=" * 60)
    print(f"{'策略':<25} {'交易數':>8} {'Sharpe':>8} {'盈虧':>12} {'勝率':>8}")
    print("-" * 60)

    # Sort by Sharpe
    results.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)

    for name, result in results:
        sharpe_str = f"{result.sharpe_ratio:.2f}"
        if result.sharpe_ratio > 1:
            sharpe_str += " ✅"
        print(f"{name:<25} {result.total_trades:>8} {sharpe_str:>8} {result.total_profit:>+10.2f} {result.win_rate:>7.1f}%")

    # Find best
    best = max(results, key=lambda x: (x[1].sharpe_ratio > 1, x[1].total_trades, x[1].sharpe_ratio))
    print(f"\n最佳策略: {best[0]}")
    print(f"  Sharpe: {best[1].sharpe_ratio:.2f}")
    print(f"  交易數: {best[1].total_trades} (年均 {best[1].total_trades / (args.days / 365):.0f})")
    print(f"  盈虧: {best[1].total_profit:+.2f} USDT")


if __name__ == "__main__":
    asyncio.run(main())
