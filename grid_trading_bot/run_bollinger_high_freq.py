#!/usr/bin/env python3
"""
Bollinger Bot 高頻交易優化回測.

目標:
1. 交易次數: 年化 100+ 筆
2. 年化報酬: 20%+
3. Sharpe > 1

策略調整:
- 增加槓桿 (2x-5x)
- 放寬過濾條件
- 縮短持倉時間
- 調整 BB 參數增加信號
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, List

from src.core.models import Kline, KlineInterval
from src.exchange.binance.futures_api import BinanceFuturesAPI
from src.bots.bollinger.models import PositionSide


@dataclass
class BacktestResult:
    """Backtest result."""
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0
    annual_return_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    trades_per_year: Decimal = field(default_factory=lambda: Decimal("0"))


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str = ""
    bb_period: int = 20
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.0"))
    trend_period: int = 50
    use_trend_filter: bool = True
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    use_rsi_filter: bool = False
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    use_atr_stop: bool = True
    max_hold_bars: int = 16
    leverage: int = 3
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    bbw_threshold_pct: int = 10
    use_opposite_band_tp: bool = False


class BollingerBacktest:
    """Bollinger backtest engine."""

    FEE_RATE = Decimal("0.0004")
    INITIAL_CAPITAL = Decimal("10000")

    def __init__(self, klines: List[Kline], config: StrategyConfig):
        self._klines = klines
        self._config = config
        self._position: Optional[dict] = None
        self._trades: List[dict] = []
        self._equity_curve: List[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}
        self._bbw_history: List[Decimal] = []

    def run(self) -> BacktestResult:
        min_period = max(
            self._config.bb_period,
            self._config.trend_period,
            self._config.rsi_period,
            self._config.atr_period,
        ) + 50

        if not self._klines or len(self._klines) < min_period:
            return BacktestResult()

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
        if idx < self._config.bb_period:
            return None
        closes = [self._klines[j].close for j in range(idx - self._config.bb_period + 1, idx + 1)]
        middle = sum(closes) / Decimal(len(closes))
        variance = sum((c - middle) ** 2 for c in closes) / Decimal(len(closes))
        std = Decimal(str(math.sqrt(float(variance))))
        upper = middle + std * self._config.bb_std
        lower = middle - std * self._config.bb_std
        return {"upper": upper, "middle": middle, "lower": lower, "std": std}

    def _calculate_sma(self, idx: int, period: int) -> Optional[Decimal]:
        if idx < period:
            return None
        closes = [self._klines[j].close for j in range(idx - period + 1, idx + 1)]
        return sum(closes) / Decimal(len(closes))

    def _calculate_rsi(self, idx: int) -> Optional[Decimal]:
        period = self._config.rsi_period
        if idx < period + 1:
            return None
        gains, losses = [], []
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
        return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

    def _calculate_atr(self, idx: int) -> Optional[Decimal]:
        period = self._config.atr_period
        if idx < period + 1:
            return None
        true_ranges = []
        for j in range(idx - period, idx):
            kline = self._klines[j + 1]
            prev_close = self._klines[j].close
            tr = max(kline.high - kline.low, abs(kline.high - prev_close), abs(kline.low - prev_close))
            true_ranges.append(tr)
        return sum(true_ranges) / Decimal(len(true_ranges))

    def _is_squeeze(self, bbw: Decimal) -> bool:
        if len(self._bbw_history) < 50:
            return False
        sorted_bbw = sorted(self._bbw_history)
        rank = sum(1 for v in sorted_bbw if v < bbw)
        percentile = (rank / len(sorted_bbw)) * 100
        return percentile < self._config.bbw_threshold_pct

    def _process_kline(self, idx: int) -> None:
        kline = self._klines[idx]
        current_price = kline.close
        date_key = kline.close_time.strftime("%Y-%m-%d")

        bands = self._calculate_bollinger(idx)
        trend_sma = self._calculate_sma(idx, self._config.trend_period)
        rsi = self._calculate_rsi(idx)
        atr = self._calculate_atr(idx)

        if not all([bands, trend_sma, rsi, atr]):
            return

        bbw = (bands["upper"] - bands["lower"]) / bands["middle"]
        self._bbw_history.append(bbw)
        if len(self._bbw_history) > 100:
            self._bbw_history = self._bbw_history[-100:]

        if self._position:
            self._check_exit(kline, bands, atr, idx, date_key)
            return

        # Squeeze filter
        if self._is_squeeze(bbw):
            return

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

        # Trend filter
        if self._config.use_trend_filter:
            if signal == PositionSide.LONG and current_price < trend_sma:
                return
            elif signal == PositionSide.SHORT and current_price > trend_sma:
                return

        # RSI filter
        if self._config.use_rsi_filter:
            if signal == PositionSide.LONG and rsi > self._config.rsi_oversold:
                return
            elif signal == PositionSide.SHORT and rsi < self._config.rsi_overbought:
                return

        # Calculate SL/TP
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
            take_profit = bands["upper"] if signal == PositionSide.LONG else bands["lower"]
        else:
            take_profit = bands["middle"]

        # Enter position
        notional = self.INITIAL_CAPITAL * self._config.position_size_pct
        quantity = notional / entry_price
        self._position = {
            "side": signal,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "quantity": quantity,
            "entry_bar": idx,
            "entry_time": kline.close_time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _check_exit(self, kline, bands, atr, bar_idx, date_key):
        if not self._position:
            return

        should_exit = False
        exit_price = kline.close
        exit_reason = ""
        side = self._position["side"]

        # Update trailing stop
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

        # Take Profit
        tp = self._position["take_profit"]
        if side == PositionSide.LONG:
            if kline.high >= tp:
                should_exit, exit_price, exit_reason = True, tp, "tp"
        else:
            if kline.low <= tp:
                should_exit, exit_price, exit_reason = True, tp, "tp"

        # Stop Loss
        if not should_exit:
            sl = self._position["stop_loss"]
            if side == PositionSide.LONG:
                if kline.low <= sl:
                    should_exit, exit_price, exit_reason = True, sl, "sl"
            else:
                if kline.high >= sl:
                    should_exit, exit_price, exit_reason = True, sl, "sl"

        # Timeout
        if not should_exit:
            hold_bars = bar_idx - self._position["entry_bar"]
            if hold_bars >= self._config.max_hold_bars:
                should_exit, exit_price, exit_reason = True, kline.close, "timeout"

        if should_exit:
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

        self._trades.append({
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": net_pnl,
            "exit_reason": exit_reason,
        })

        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        self._position = None

    def _calculate_result(self) -> BacktestResult:
        result = BacktestResult()
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

        gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            result.profit_factor = Decimal("999")

        # Max drawdown
        equity = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")
        for t in self._trades:
            equity += t["pnl"]
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd

        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe_ratio()

        # Annual metrics
        if self._klines:
            days = (self._klines[-1].close_time - self._klines[0].close_time).days
            if days > 0:
                years = Decimal(days) / Decimal("365")
                result.annual_return_pct = (result.total_profit / self.INITIAL_CAPITAL) / years * Decimal("100")
                result.trades_per_year = Decimal(result.total_trades) / years

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


async def fetch_klines(symbol: str, interval: str, days: int) -> List[Kline]:
    """Fetch historical klines."""
    print(f"\n正在獲取 {symbol} {interval} 歷史數據 ({days} 天)...")

    async with BinanceFuturesAPI() as api:
        await api.ping()

        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        interval_map = {
            "5m": KlineInterval.m5,
            "15m": KlineInterval.m15,
            "30m": KlineInterval.m30,
            "1h": KlineInterval.h1,
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
                current_start = int(klines[-1].close_time.timestamp() * 1000) + 1
            print(f"  已獲取 {len(all_klines)} 根 K 線")
            await asyncio.sleep(0.1)

        return all_klines


async def main():
    print("\n" + "="*70)
    print("  Bollinger Bot 高報酬優化回測")
    print("  目標: 年化 20%+ | Sharpe > 1")
    print("="*70)

    # Fetch 2 years of data with 15m timeframe
    klines = await fetch_klines("BTCUSDT", "15m", 730)

    if len(klines) < 200:
        print("數據不足")
        return

    # Define strategies - focus on HIGH LEVERAGE + STRICT FILTERS for high returns
    strategies = [
        # === 嚴格過濾 S4 + 高槓桿 ===
        StrategyConfig(
            name="S4-10x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=10,
            bbw_threshold_pct=35,
        ),
        StrategyConfig(
            name="S4-15x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=15,
            bbw_threshold_pct=35,
        ),
        StrategyConfig(
            name="S4-20x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=20,
            bbw_threshold_pct=35,
        ),
        StrategyConfig(
            name="S4-25x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=25,
            bbw_threshold_pct=35,
        ),
        StrategyConfig(
            name="S4-30x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=30,
            bbw_threshold_pct=35,
        ),
        StrategyConfig(
            name="S4-40x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=40,
            bbw_threshold_pct=35,
        ),
        StrategyConfig(
            name="S4-50x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=50,
            bbw_threshold_pct=35,
        ),
        # === 放寬 BBW 增加交易 + 高槓桿 ===
        StrategyConfig(
            name="S4-20x-BBW25",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=20,
            bbw_threshold_pct=25,
        ),
        StrategyConfig(
            name="S4-30x-BBW25",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=30,
            bbw_threshold_pct=25,
        ),
        # === 只趨勢過濾 + 高槓桿 ===
        StrategyConfig(
            name="趨勢-20x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            use_rsi_filter=False,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=20,
            bbw_threshold_pct=25,
        ),
        StrategyConfig(
            name="趨勢-30x",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            use_rsi_filter=False,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=30,
            bbw_threshold_pct=25,
        ),
        # === 原始對照 ===
        StrategyConfig(
            name="S4-1x (原始)",
            bb_period=20, bb_std=Decimal("2.0"),
            trend_period=50, use_trend_filter=True,
            rsi_oversold=30, rsi_overbought=70,
            use_rsi_filter=True,
            atr_multiplier=Decimal("2.0"),
            max_hold_bars=24,
            leverage=1,
            bbw_threshold_pct=35,
        ),
    ]

    # Run backtests
    results = []
    for config in strategies:
        backtest = BollingerBacktest(klines, config)
        result = backtest.run()
        results.append((config.name, result, config))

    # Print results
    print("\n" + "="*100)
    print(f"{'策略':<25} {'交易數':>8} {'年交易':>8} {'年化%':>10} {'Sharpe':>8} {'勝率':>8} {'盈虧':>12} {'回撤':>10}")
    print("="*100)

    # Sort by annual return
    results.sort(key=lambda x: x[1].annual_return_pct, reverse=True)

    qualified = []
    for name, result, config in results:
        annual_ret = float(result.annual_return_pct)
        trades_yr = float(result.trades_per_year)
        sharpe = float(result.sharpe_ratio)

        # Check if meets criteria
        meets = annual_ret >= 20 and trades_yr >= 50 and sharpe >= 1.0
        marker = "✅" if meets else "  "
        if meets:
            qualified.append((name, result, config))

        print(f"{marker}{name:<23} {result.total_trades:>8} {trades_yr:>8.0f} {annual_ret:>9.1f}% "
              f"{sharpe:>8.2f} {float(result.win_rate):>7.1f}% {float(result.total_profit):>+11.0f} "
              f"{float(result.max_drawdown):>10.0f}")

    print("="*100)

    # Summary
    print("\n" + "="*70)
    print("  符合條件的策略 (年化 >= 20% + 年交易 >= 50 + Sharpe >= 1)")
    print("="*70)

    if qualified:
        for name, result, config in qualified:
            print(f"\n  📌 {name}")
            print(f"     年化報酬: {float(result.annual_return_pct):.1f}%")
            print(f"     年交易數: {float(result.trades_per_year):.0f} 筆")
            print(f"     Sharpe:   {float(result.sharpe_ratio):.2f}")
            print(f"     勝率:     {float(result.win_rate):.1f}%")
            print(f"     槓桿:     {config.leverage}x")
            print(f"     2年盈虧:  {float(result.total_profit):+,.0f} USDT")

        # Best strategy
        best = max(qualified, key=lambda x: x[1].sharpe_ratio)
        print(f"\n  🏆 最佳策略: {best[0]}")
        print(f"     建議配置:")
        print(f"       - BB週期: {best[2].bb_period}")
        print(f"       - BB標準差: {best[2].bb_std}")
        print(f"       - 趨勢過濾: {'開啟' if best[2].use_trend_filter else '關閉'}")
        print(f"       - RSI過濾: {'開啟' if best[2].use_rsi_filter else '關閉'}")
        print(f"       - ATR乘數: {best[2].atr_multiplier}")
        print(f"       - 最大持倉: {best[2].max_hold_bars} bars")
        print(f"       - 槓桿: {best[2].leverage}x")
        print(f"       - BBW閾值: {best[2].bbw_threshold_pct}%")
    else:
        print("\n  ⚠️ 沒有策略同時滿足所有條件")
        print("  建議: 嘗試更高槓桿或更激進的參數")

    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
