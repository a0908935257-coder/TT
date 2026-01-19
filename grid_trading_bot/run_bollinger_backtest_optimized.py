#!/usr/bin/env python3
"""
Optimized Bollinger Bot Backtest.

Improvements over basic strategy:
1. Trend Filter: Only trade in direction of 50-period SMA trend
2. RSI Confirmation: RSI < 30 for long, RSI > 70 for short
3. ATR-based Stop Loss: Dynamic stop based on volatility
4. Extended Take Profit: Use opposite band instead of middle
5. Volatility Filter: Skip extremely low/high volatility periods

Usage:
    python run_bollinger_backtest_optimized.py
    python run_bollinger_backtest_optimized.py --symbol BTCUSDT --days 60
"""

import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI
from src.bots.bollinger.models import PositionSide


@dataclass
class OptimizedBacktestResult:
    """Optimized backtest result."""
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_holding_time: timedelta = field(default_factory=lambda: timedelta(0))
    num_wins: int = 0
    num_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    take_profit_exits: int = 0
    stop_loss_exits: int = 0
    timeout_exits: int = 0
    filtered_by_trend: int = 0
    filtered_by_rsi: int = 0
    filtered_by_squeeze: int = 0


@dataclass
class OptimizedConfig:
    """Optimized strategy configuration."""
    symbol: str = "BTCUSDT"

    # Bollinger Bands
    bb_period: int = 20
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.0"))

    # Trend Filter
    trend_period: int = 50  # SMA period for trend
    use_trend_filter: bool = True

    # RSI Filter
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    use_rsi_filter: bool = True

    # ATR Stop Loss
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    use_atr_stop: bool = True

    # Take Profit
    use_opposite_band_tp: bool = True  # Use opposite band instead of middle

    # Position
    max_hold_bars: int = 24
    leverage: int = 2
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # BBW Filter
    bbw_lookback: int = 100
    bbw_threshold_pct: int = 15  # Lower threshold


class OptimizedBollingerBacktest:
    """Optimized Bollinger backtest with multiple filters."""

    FEE_RATE = Decimal("0.0004")

    def __init__(self, klines: list[Kline], config: OptimizedConfig):
        self._klines = klines
        self._config = config
        self._position: Optional[dict] = None
        self._trades: list[dict] = []
        self._equity_curve: list[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}

        # Filter counters
        self._filtered_trend: int = 0
        self._filtered_rsi: int = 0
        self._filtered_squeeze: int = 0

        # Exit counters
        self._exit_reasons: dict[str, int] = {"take_profit": 0, "stop_loss": 0, "timeout": 0}

        # Pre-calculate indicators
        self._bbw_history: list[Decimal] = []

    def run(self) -> OptimizedBacktestResult:
        min_period = max(
            self._config.bb_period,
            self._config.trend_period,
            self._config.rsi_period,
            self._config.atr_period,
        ) + 50

        if not self._klines or len(self._klines) < min_period:
            return OptimizedBacktestResult()

        # Initialize BBW history
        for i in range(self._config.bb_period, min_period):
            bands = self._calculate_bollinger(i)
            if bands:
                bbw = (bands["upper"] - bands["lower"]) / bands["middle"]
                self._bbw_history.append(bbw)

        # Process each kline
        for i in range(min_period, len(self._klines)):
            self._process_kline(i)

        return self._calculate_result()

    def _calculate_bollinger(self, idx: int) -> Optional[dict]:
        """Calculate Bollinger Bands at index."""
        if idx < self._config.bb_period:
            return None

        closes = [self._klines[j].close for j in range(idx - self._config.bb_period + 1, idx + 1)]

        # SMA
        middle = sum(closes) / Decimal(len(closes))

        # Standard deviation
        variance = sum((c - middle) ** 2 for c in closes) / Decimal(len(closes))
        std = Decimal(str(math.sqrt(float(variance))))

        upper = middle + std * self._config.bb_std
        lower = middle - std * self._config.bb_std

        return {"upper": upper, "middle": middle, "lower": lower, "std": std}

    def _calculate_sma(self, idx: int, period: int) -> Optional[Decimal]:
        """Calculate SMA at index."""
        if idx < period:
            return None
        closes = [self._klines[j].close for j in range(idx - period + 1, idx + 1)]
        return sum(closes) / Decimal(len(closes))

    def _calculate_rsi(self, idx: int) -> Optional[Decimal]:
        """Calculate RSI at index."""
        period = self._config.rsi_period
        if idx < period + 1:
            return None

        gains = []
        losses = []

        for j in range(idx - period, idx):
            change = self._klines[j + 1].close - self._klines[j].close
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        avg_gain = sum(gains) / Decimal(period)
        avg_loss = sum(losses) / Decimal(period)

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    def _calculate_atr(self, idx: int) -> Optional[Decimal]:
        """Calculate ATR at index."""
        period = self._config.atr_period
        if idx < period + 1:
            return None

        true_ranges = []

        for j in range(idx - period, idx):
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

        return percentile < self._config.bbw_threshold_pct

    def _process_kline(self, idx: int) -> None:
        """Process a single kline."""
        kline = self._klines[idx]
        current_price = kline.close
        date_key = kline.close_time.strftime("%Y-%m-%d")

        # Calculate indicators
        bands = self._calculate_bollinger(idx)
        trend_sma = self._calculate_sma(idx, self._config.trend_period)
        rsi = self._calculate_rsi(idx)
        atr = self._calculate_atr(idx)

        if not all([bands, trend_sma, rsi, atr]):
            return

        # Update BBW history
        bbw = (bands["upper"] - bands["lower"]) / bands["middle"]
        self._bbw_history.append(bbw)
        if len(self._bbw_history) > self._config.bbw_lookback:
            self._bbw_history = self._bbw_history[-self._config.bbw_lookback:]

        # Check existing position
        if self._position:
            self._check_exit(kline, bands, atr, idx, date_key)
            return

        # === ENTRY FILTERS ===

        # 1. BBW Squeeze Filter
        if self._is_squeeze(bbw):
            self._filtered_squeeze += 1
            return

        # Determine signal
        signal = None
        entry_price = None

        if current_price <= bands["lower"]:
            signal = PositionSide.LONG
            entry_price = bands["lower"]
        elif current_price >= bands["upper"]:
            signal = PositionSide.SHORT
            entry_price = bands["upper"]
        else:
            return

        # 2. Trend Filter
        if self._config.use_trend_filter:
            if signal == PositionSide.LONG and current_price < trend_sma:
                # Downtrend, skip long
                self._filtered_trend += 1
                return
            elif signal == PositionSide.SHORT and current_price > trend_sma:
                # Uptrend, skip short
                self._filtered_trend += 1
                return

        # 3. RSI Filter
        if self._config.use_rsi_filter:
            if signal == PositionSide.LONG and rsi > self._config.rsi_oversold:
                self._filtered_rsi += 1
                return
            elif signal == PositionSide.SHORT and rsi < self._config.rsi_overbought:
                self._filtered_rsi += 1
                return

        # Calculate stop loss and take profit
        if self._config.use_atr_stop:
            atr_stop = atr * self._config.atr_multiplier
            if signal == PositionSide.LONG:
                stop_loss = entry_price - atr_stop
            else:
                stop_loss = entry_price + atr_stop
        else:
            stop_pct = Decimal("0.015")
            if signal == PositionSide.LONG:
                stop_loss = entry_price * (Decimal("1") - stop_pct)
            else:
                stop_loss = entry_price * (Decimal("1") + stop_pct)

        if self._config.use_opposite_band_tp:
            if signal == PositionSide.LONG:
                take_profit = bands["upper"]
            else:
                take_profit = bands["lower"]
        else:
            take_profit = bands["middle"]

        # Enter position
        self._enter_position(
            side=signal,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            entry_bar=idx,
            entry_time=kline.close_time,
        )

        self._update_equity(current_price)

    def _enter_position(self, side, entry_price, take_profit, stop_loss, entry_bar, entry_time):
        notional = Decimal("10000") * self._config.position_size_pct
        quantity = notional / entry_price
        self._position = {
            "side": side,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "quantity": quantity,
            "entry_bar": entry_bar,
            "entry_time": entry_time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _check_exit(self, kline, bands, atr, bar_idx, date_key):
        if not self._position:
            return

        should_exit = False
        exit_price = kline.close
        exit_reason = ""
        side = self._position["side"]

        # Update trailing stop with ATR
        if self._config.use_atr_stop and atr:
            atr_stop = atr * self._config.atr_multiplier
            if side == PositionSide.LONG:
                new_stop = kline.close - atr_stop
                if new_stop > self._position["stop_loss"]:
                    self._position["stop_loss"] = new_stop
            else:
                new_stop = kline.close + atr_stop
                if new_stop < self._position["stop_loss"]:
                    self._position["stop_loss"] = new_stop

        # 1. Take Profit
        tp = self._position["take_profit"]
        if side == PositionSide.LONG:
            if kline.high >= tp:
                should_exit, exit_price, exit_reason = True, tp, "take_profit"
        else:
            if kline.low <= tp:
                should_exit, exit_price, exit_reason = True, tp, "take_profit"

        # 2. Stop Loss
        if not should_exit:
            sl = self._position["stop_loss"]
            if side == PositionSide.LONG:
                if kline.low <= sl:
                    should_exit, exit_price, exit_reason = True, sl, "stop_loss"
            else:
                if kline.high >= sl:
                    should_exit, exit_price, exit_reason = True, sl, "stop_loss"

        # 3. Timeout
        if not should_exit:
            hold_bars = bar_idx - self._position["entry_bar"]
            if hold_bars >= self._config.max_hold_bars:
                should_exit, exit_price, exit_reason = True, kline.close, "timeout"

        if should_exit:
            self._exit_reasons[exit_reason] += 1
            self._exit_position(exit_price, exit_reason, kline.close_time, bar_idx, date_key)

    def _exit_position(self, exit_price, exit_reason, exit_time, exit_bar, date_key):
        if not self._position:
            return

        side = self._position["side"]
        entry_price = self._position["entry_price"]
        quantity = self._position["quantity"]

        if side == PositionSide.LONG:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl *= Decimal(self._config.leverage)
        exit_fee = (exit_price * quantity) * self.FEE_RATE
        total_fee = self._position["entry_fee"] + exit_fee
        net_pnl = pnl - total_fee

        trade = {
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": net_pnl,
            "fee": total_fee,
            "entry_time": self._position["entry_time"],
            "exit_time": exit_time,
            "holding_bars": exit_bar - self._position["entry_bar"],
            "exit_reason": exit_reason,
        }
        self._trades.append(trade)

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        self._position = None

    def _update_equity(self, current_price):
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
            unrealized *= Decimal(self._config.leverage)
        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> OptimizedBacktestResult:
        result = OptimizedBacktestResult()
        if not self._trades:
            return result

        result.total_profit = sum(t["pnl"] for t in self._trades)
        result.total_trades = len(self._trades)

        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)

        if self._trades:
            result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * Decimal("100")

        if wins:
            result.avg_win = sum(t["pnl"] for t in wins) / Decimal(len(wins))
        if losses:
            result.avg_loss = abs(sum(t["pnl"] for t in losses)) / Decimal(len(losses))

        gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.profit_factor = Decimal("999")

        result.long_trades = len([t for t in self._trades if t["side"] == PositionSide.LONG])
        result.short_trades = len([t for t in self._trades if t["side"] == PositionSide.SHORT])

        result.take_profit_exits = self._exit_reasons["take_profit"]
        result.stop_loss_exits = self._exit_reasons["stop_loss"]
        result.timeout_exits = self._exit_reasons["timeout"]

        result.filtered_by_trend = self._filtered_trend
        result.filtered_by_rsi = self._filtered_rsi
        result.filtered_by_squeeze = self._filtered_squeeze

        if self._equity_curve:
            peak = self._equity_curve[0]
            max_dd = Decimal("0")
            for equity in self._equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_dd:
                    max_dd = drawdown
            result.max_drawdown = max_dd

        if self._trades:
            total_bars = sum(t["holding_bars"] for t in self._trades)
            avg_bars = total_bars / len(self._trades)
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe_ratio()

        return result

    def _calculate_sharpe_ratio(self) -> Decimal:
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


async def fetch_klines(symbol: str, interval: str, days: int) -> list[Kline]:
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
            "30m": KlineInterval.m30,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
            "1d": KlineInterval.d1,
        }
        kline_interval = interval_map.get(interval, KlineInterval.m15)

        all_klines = []
        current_start = start_time
        batch_count = 0

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
            batch_count += 1

            if klines:
                last_close_time = int(klines[-1].close_time.timestamp() * 1000)
                current_start = last_close_time + 1

            print(f"  已獲取 {len(all_klines)} 根 K 線 (批次 {batch_count})")
            await asyncio.sleep(0.1)

        print(f"  總共獲取 {len(all_klines)} 根 K 線")
        return all_klines


def print_result(name: str, result, symbol: str, interval: str):
    """Print backtest result."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  {symbol} | {interval}")
    print(f"{'='*65}")
    print(f"  總交易數:        {result.total_trades}")
    print(f"  勝率:            {result.win_rate:.1f}%")
    print(f"  多單交易:        {result.long_trades}")
    print(f"  空單交易:        {result.short_trades}")
    print(f"  獲勝次數:        {result.num_wins}")
    print(f"  虧損次數:        {result.num_losses}")
    print(f"  總盈虧:          {result.total_profit:+.2f} USDT")
    print(f"  平均獲利:        {result.avg_win:+.4f} USDT")
    print(f"  平均虧損:        {result.avg_loss:.4f} USDT")
    print(f"  獲利因子:        {result.profit_factor:.2f}")
    print(f"  夏普率:          {result.sharpe_ratio:.2f}")
    print(f"  最大回撤:        {result.max_drawdown:.2f} USDT")
    print(f"{'='*65}")
    print(f"  出場統計:")
    print(f"    止盈出場:      {result.take_profit_exits}")
    print(f"    止損出場:      {result.stop_loss_exits}")
    print(f"    超時出場:      {result.timeout_exits}")

    # Show filter stats only for optimized result
    if hasattr(result, 'filtered_by_trend'):
        print(f"  過濾統計:")
        print(f"    趨勢過濾:      {result.filtered_by_trend} 次")
        print(f"    RSI 過濾:      {result.filtered_by_rsi} 次")
        print(f"    壓縮過濾:      {result.filtered_by_squeeze} 次")
    else:
        print(f"  壓縮過濾:        {result.filtered_by_squeeze} 次")

    print(f"  平均持倉時間:    {result.avg_holding_time}")
    print(f"{'='*65}")


def print_comparison(basic_result, optimized_result):
    """Print comparison between basic and optimized strategies."""
    print("\n")
    print("+" + "="*63 + "+")
    print("|" + " "*20 + "策略對比" + " "*20 + "|")
    print("+" + "="*63 + "+")
    print(f"|  {'指標':<20} {'基礎策略':>18} {'優化策略':>18}   |")
    print("|" + "-"*63 + "|")
    print(f"|  {'總交易數':<20} {basic_result.total_trades:>18} {optimized_result.total_trades:>18}   |")
    print(f"|  {'勝率':<20} {str(basic_result.win_rate.quantize(Decimal('0.1')))+'%':>18} {str(optimized_result.win_rate.quantize(Decimal('0.1')))+'%':>18}   |")
    print(f"|  {'總盈虧 (USDT)':<20} {str(basic_result.total_profit.quantize(Decimal('0.01'))):>18} {str(optimized_result.total_profit.quantize(Decimal('0.01'))):>18}   |")
    print(f"|  {'獲利因子':<20} {str(basic_result.profit_factor.quantize(Decimal('0.01'))):>18} {str(optimized_result.profit_factor.quantize(Decimal('0.01'))):>18}   |")
    print(f"|  {'夏普率':<20} {str(basic_result.sharpe_ratio.quantize(Decimal('0.01'))):>18} {str(optimized_result.sharpe_ratio.quantize(Decimal('0.01'))):>18}   |")
    print(f"|  {'最大回撤 (USDT)':<20} {str(basic_result.max_drawdown.quantize(Decimal('0.01'))):>18} {str(optimized_result.max_drawdown.quantize(Decimal('0.01'))):>18}   |")
    print("+" + "="*63 + "+")

    # Improvement
    if basic_result.sharpe_ratio != 0:
        improvement = ((optimized_result.sharpe_ratio - basic_result.sharpe_ratio) / abs(basic_result.sharpe_ratio)) * 100
        print(f"|  夏普率改善: {improvement:+.1f}%")
    print("+" + "="*63 + "+")


async def main():
    parser = argparse.ArgumentParser(description="Optimized Bollinger Bot Backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="15m", help="Kline interval")
    parser.add_argument("--days", type=int, default=60, help="Days of historical data")
    args = parser.parse_args()

    print("\n")
    print("+" + "="*63 + "+")
    print("|" + " "*12 + "Bollinger Bot 優化策略回測" + " "*12 + "|")
    print("+" + "="*63 + "+")

    # Fetch data
    klines = await fetch_klines(args.symbol, args.interval, args.days)

    if len(klines) < 150:
        print("K 線數量不足")
        return

    # === Run Basic Strategy ===
    print("\n正在運行基礎策略...")

    # Import basic backtest
    from run_bollinger_backtest_real import BollingerBacktest, BollingerBacktestResult
    from src.bots.bollinger.models import BollingerConfig

    basic_config = BollingerConfig(
        symbol=args.symbol,
        timeframe=args.interval,
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=20,
        stop_loss_pct=Decimal("0.015"),
        max_hold_bars=16,
        leverage=2,
        position_size_pct=Decimal("0.1"),
    )

    basic_backtest = BollingerBacktest(klines=klines, config=basic_config)
    basic_result = basic_backtest.run()

    print_result("基礎策略", basic_result, args.symbol, args.interval)

    # === Run Optimized Strategy ===
    print("\n正在運行優化策略...")

    # Try multiple configurations - focus on practical improvements
    configs = [
        ("v1: 趨勢過濾+寬鬆RSI", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.0"),
            trend_period=50,
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=35,  # More relaxed
            rsi_overbought=65,  # More relaxed
            use_rsi_filter=True,
            atr_period=14,
            atr_multiplier=Decimal("2.0"),
            use_atr_stop=True,
            use_opposite_band_tp=False,  # Use middle band
            max_hold_bars=24,
            leverage=2,
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=20,  # Same as basic
        )),
        ("v2: 只趨勢過濾", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.0"),
            trend_period=50,
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            use_rsi_filter=False,  # No RSI
            atr_period=14,
            atr_multiplier=Decimal("2.0"),
            use_atr_stop=True,
            use_opposite_band_tp=False,
            max_hold_bars=24,
            leverage=2,
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=20,
        )),
        ("v3: ATR止損+中軌止盈", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.0"),
            trend_period=50,
            use_trend_filter=False,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            use_rsi_filter=False,
            atr_period=14,
            atr_multiplier=Decimal("2.5"),  # Wider ATR stop
            use_atr_stop=True,
            use_opposite_band_tp=False,
            max_hold_bars=24,
            leverage=2,
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=20,
        )),
        ("v4: 趨勢+ATR寬止損", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.0"),
            trend_period=50,
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            use_rsi_filter=False,
            atr_period=14,
            atr_multiplier=Decimal("3.0"),  # Even wider
            use_atr_stop=True,
            use_opposite_band_tp=False,
            max_hold_bars=32,  # Longer hold time
            leverage=2,
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=20,
        )),
        ("v5: 較短BB週期", OptimizedConfig(
            symbol=args.symbol,
            bb_period=15,  # Shorter period
            bb_std=Decimal("2.0"),
            trend_period=50,
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            use_rsi_filter=False,
            atr_period=14,
            atr_multiplier=Decimal("2.5"),
            use_atr_stop=True,
            use_opposite_band_tp=False,
            max_hold_bars=24,
            leverage=2,
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=20,
        )),
        ("v6: 低槓桿+緊帶", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.5"),  # Tighter bands - only extreme entries
            trend_period=50,
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            use_rsi_filter=False,
            atr_period=14,
            atr_multiplier=Decimal("2.0"),
            use_atr_stop=True,
            use_opposite_band_tp=True,  # Use opposite band for more profit
            max_hold_bars=32,
            leverage=1,  # Lower leverage for better Sharpe
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=25,  # Only trade when BBW is higher
        )),
        ("v7: 趨勢+RSI嚴格", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.0"),
            trend_period=50,
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=25,  # Very strict RSI
            rsi_overbought=75,  # Very strict RSI
            use_rsi_filter=True,
            atr_period=14,
            atr_multiplier=Decimal("1.5"),  # Tighter stop
            use_atr_stop=True,
            use_opposite_band_tp=False,
            max_hold_bars=16,  # Shorter hold
            leverage=1,
            position_size_pct=Decimal("0.1"),
            bbw_lookback=100,
            bbw_threshold_pct=30,  # Skip squeeze entirely
        )),
        ("v8: 保守設定", OptimizedConfig(
            symbol=args.symbol,
            bb_period=20,
            bb_std=Decimal("2.2"),  # Slightly tighter
            trend_period=100,  # Longer trend filter
            use_trend_filter=True,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            use_rsi_filter=False,
            atr_period=14,
            atr_multiplier=Decimal("1.5"),  # Tighter stop = smaller losses
            use_atr_stop=True,
            use_opposite_band_tp=False,  # Middle band = faster exit
            max_hold_bars=12,  # Very short hold
            leverage=1,
            position_size_pct=Decimal("0.05"),  # Smaller position
            bbw_lookback=100,
            bbw_threshold_pct=30,
        )),
    ]

    best_sharpe = Decimal("-999")
    best_result = None
    best_name = ""

    for name, optimized_config in configs:
        print(f"\n正在運行 {name}...")

        optimized_backtest = OptimizedBollingerBacktest(klines=klines, config=optimized_config)
        optimized_result = optimized_backtest.run()

        print_result(name, optimized_result, args.symbol, args.interval)

        if optimized_result.sharpe_ratio > best_sharpe and optimized_result.total_trades >= 10:
            best_sharpe = optimized_result.sharpe_ratio
            best_result = optimized_result
            best_name = name

    # === Print Summary ===
    print("\n")
    print("+" + "="*63 + "+")
    print("|" + " "*22 + "回測總結" + " "*22 + "|")
    print("+" + "="*63 + "+")

    if best_result:
        print(f"|  最佳策略: {best_name:<50} |")
        print(f"|  夏普率: {best_sharpe:>52.2f} |")
        print(f"|  交易次數: {best_result.total_trades:>50} |")
        print(f"|  總盈虧: {str(best_result.total_profit.quantize(Decimal('0.01')))+' USDT':>51} |")
        print_comparison(basic_result, best_result)
    else:
        print("|  未找到符合條件的優化策略（交易次數需 >= 10）               |")

    print("+" + "="*63 + "+")

    print("\n優化策略改進項目:")
    print("  1. 趨勢過濾: 只順趨勢方向交易")
    print("  2. RSI 確認: 多單 RSI<閾值, 空單 RSI>閾值")
    print("  3. ATR 止損: 動態止損基於波動率")
    print("  4. 擴大止盈: 對側軌道止盈 (非中軌)")
    print("  5. 降低壓縮閾值: 減少過濾")

    print("\n回測完成!")


if __name__ == "__main__":
    asyncio.run(main())
