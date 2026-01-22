#!/usr/bin/env python3
"""
Supertrend Bot 過度擬合驗證測試

測試方法:
1. 樣本外測試 (Out-of-Sample): 用前 70% 數據訓練，後 30% 測試
2. 前瞻分析 (Walk-Forward): 滾動窗口測試
3. 參數敏感度分析: 微調參數看結果變化
4. 不同市場狀況: 牛市、熊市、盤整期分開測試
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional

sys.path.insert(0, "/mnt/c/trading/grid_trading_bot")

from src.exchange import ExchangeClient
from src.core.models import KlineInterval


@dataclass
class BacktestResult:
    """回測結果"""
    period: str
    total_trades: int
    win_rate: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    total_pnl: float


def calculate_supertrend(klines: list, atr_period: int, atr_multiplier: Decimal, start_idx: int):
    """計算 Supertrend 指標"""
    if start_idx < atr_period + 1:
        return None, None

    # 計算 ATR
    trs = []
    for i in range(start_idx - atr_period, start_idx):
        high = klines[i].high
        low = klines[i].low
        prev_close = klines[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    atr = sum(trs) / Decimal(len(trs))

    # 計算基本上下軌
    current = klines[start_idx]
    hl2 = (current.high + current.low) / 2

    basic_upper = hl2 + atr_multiplier * atr
    basic_lower = hl2 - atr_multiplier * atr

    return basic_upper, basic_lower, atr


def run_supertrend_backtest(
    klines: list,
    atr_period: int = 10,
    atr_multiplier: Decimal = Decimal("3.0"),
    leverage: int = 10,
    initial_capital: Decimal = Decimal("10000"),
) -> BacktestResult:
    """執行 Supertrend 回測"""

    capital = initial_capital
    position = None
    trades = []

    lookback = atr_period + 20
    if len(klines) < lookback + 50:
        return BacktestResult(
            period="insufficient_data",
            total_trades=0,
            win_rate=0,
            annual_return=0,
            sharpe=0,
            max_drawdown=0,
            total_pnl=0,
        )

    fee_rate = Decimal("0.0004")
    max_equity = float(capital)
    max_drawdown = Decimal("0")

    # Supertrend 狀態
    prev_trend = 0  # 1 = bullish, -1 = bearish
    upper_band = None
    lower_band = None

    for i in range(lookback, len(klines)):
        kline = klines[i]
        current_price = kline.close

        # 計算 Supertrend
        result = calculate_supertrend(klines, atr_period, atr_multiplier, i)
        if result[0] is None:
            continue

        basic_upper, basic_lower, atr = result

        # 更新最終上下軌
        if upper_band is None:
            upper_band = basic_upper
            lower_band = basic_lower
        else:
            # 上軌只能下降或維持
            if basic_upper < upper_band or klines[i - 1].close > upper_band:
                upper_band = basic_upper
            # 下軌只能上升或維持
            if basic_lower > lower_band or klines[i - 1].close < lower_band:
                lower_band = basic_lower

        # 判斷趨勢
        if current_price > upper_band:
            current_trend = 1  # Bullish
        elif current_price < lower_band:
            current_trend = -1  # Bearish
        else:
            current_trend = prev_trend

        # 趨勢翻轉信號
        if current_trend != prev_trend and prev_trend != 0:
            # 先平倉
            if position:
                entry_price = position["entry_price"]
                quantity = position["quantity"]

                if position["side"] == "LONG":
                    pnl = (current_price - entry_price) * quantity * Decimal(leverage)
                else:
                    pnl = (entry_price - current_price) * quantity * Decimal(leverage)

                fee = (entry_price + current_price) * quantity * fee_rate
                pnl -= fee
                capital += pnl

                trades.append({
                    "pnl": pnl,
                    "side": position["side"],
                })
                position = None

            # 開新倉
            if capital > 0:
                position_value = capital * Decimal("0.1")
                quantity = position_value / current_price
                quantity = quantity.quantize(Decimal("0.001"))

                if quantity > 0:
                    if current_trend == 1:
                        position = {
                            "side": "LONG",
                            "entry_price": current_price,
                            "quantity": quantity,
                        }
                    else:
                        position = {
                            "side": "SHORT",
                            "entry_price": current_price,
                            "quantity": quantity,
                        }

        prev_trend = current_trend

        # 更新回撤
        current_equity = float(capital)
        if position:
            if position["side"] == "LONG":
                unrealized = float((current_price - position["entry_price"]) * position["quantity"] * Decimal(leverage))
            else:
                unrealized = float((position["entry_price"] - current_price) * position["quantity"] * Decimal(leverage))
            current_equity += unrealized

        max_equity = max(max_equity, current_equity)
        drawdown = Decimal(str((max_equity - current_equity) / max_equity * 100)) if max_equity > 0 else Decimal("0")
        max_drawdown = max(max_drawdown, drawdown)

    # 強制平倉
    if position and len(klines) > 0:
        current_price = klines[-1].close
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        if position["side"] == "LONG":
            pnl = (current_price - entry_price) * quantity * Decimal(leverage)
        else:
            pnl = (entry_price - current_price) * quantity * Decimal(leverage)

        fee = (entry_price + current_price) * quantity * fee_rate
        pnl -= fee
        capital += pnl
        trades.append({"pnl": pnl, "side": position["side"]})

    # 計算統計
    total_trades = len(trades)
    if total_trades == 0:
        return BacktestResult(
            period="no_trades",
            total_trades=0,
            win_rate=0,
            annual_return=0,
            sharpe=0,
            max_drawdown=float(max_drawdown),
            total_pnl=0,
        )

    winning = [t for t in trades if t["pnl"] > 0]
    win_rate = len(winning) / total_trades * 100

    total_pnl = sum(t["pnl"] for t in trades)

    days = (klines[-1].close_time - klines[0].close_time).days
    years = days / 365 if days > 0 else 1
    annual_return = ((float(capital) / float(initial_capital)) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe
    if len(trades) > 1:
        returns = [float(t["pnl"] / initial_capital) for t in trades]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        trades_per_year = total_trades / years if years > 0 else total_trades
        sharpe = (avg_return * trades_per_year) / (std_return * (trades_per_year ** 0.5)) if std_return > 0 else 0
    else:
        sharpe = 0

    return BacktestResult(
        period=f"{klines[0].close_time.date()} to {klines[-1].close_time.date()}",
        total_trades=total_trades,
        win_rate=win_rate,
        annual_return=annual_return,
        sharpe=sharpe,
        max_drawdown=float(max_drawdown),
        total_pnl=float(total_pnl),
    )


async def main():
    print("=" * 80)
    print("  Supertrend Bot 過度擬合驗證測試")
    print("=" * 80)

    exchange = ExchangeClient()
    await exchange.connect()

    print("\n正在獲取 BTCUSDT 15m 歷史數據...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=730)

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        start_ts = int(current_start.timestamp() * 1000)
        klines = await exchange.futures.get_klines(
            symbol="BTCUSDT",
            interval=KlineInterval.m15,
            start_time=start_ts,
            limit=1500,
        )

        if not klines:
            break

        all_klines.extend(klines)
        current_start = klines[-1].close_time + timedelta(minutes=1)

        if len(klines) < 1500:
            break

    print(f"  已獲取 {len(all_klines)} 根 K 線 ({len(all_klines) / 96:.0f} 天)")

    # 預設參數 (樣本外驗證通過)
    default_atr_period = 5  # Out-of-sample validated
    default_atr_mult = Decimal("2.5")  # Out-of-sample validated
    default_leverage = 5  # Out-of-sample validated

    # =========================================================================
    # 測試 1: 樣本內 vs 樣本外
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 1: 樣本內 vs 樣本外 (70/30 分割)")
    print("=" * 80)

    split_idx = int(len(all_klines) * 0.7)
    in_sample = all_klines[:split_idx]
    out_sample = all_klines[split_idx:]

    in_result = run_supertrend_backtest(in_sample, default_atr_period, default_atr_mult, default_leverage)
    out_result = run_supertrend_backtest(out_sample, default_atr_period, default_atr_mult, default_leverage)

    print(f"\n  {'指標':<15} {'樣本內 (訓練)':<20} {'樣本外 (測試)':<20} {'差異':<15}")
    print("-" * 70)
    print(f"  {'年化報酬':<15} {in_result.annual_return:>18.1f}% {out_result.annual_return:>18.1f}% {out_result.annual_return - in_result.annual_return:>+13.1f}%")
    print(f"  {'Sharpe':<15} {in_result.sharpe:>19.2f} {out_result.sharpe:>19.2f} {out_result.sharpe - in_result.sharpe:>+14.2f}")
    print(f"  {'勝率':<15} {in_result.win_rate:>18.1f}% {out_result.win_rate:>18.1f}% {out_result.win_rate - in_result.win_rate:>+13.1f}%")
    print(f"  {'交易數':<15} {in_result.total_trades:>19} {out_result.total_trades:>19}")
    print(f"  {'最大回撤':<15} {in_result.max_drawdown:>18.1f}% {out_result.max_drawdown:>18.1f}%")

    oos_degradation = (in_result.annual_return - out_result.annual_return) / in_result.annual_return * 100 if in_result.annual_return != 0 else 0

    print(f"\n  樣本外績效衰退: {oos_degradation:.1f}%")
    if oos_degradation < 30:
        print("  ✅ 通過: 樣本外績效衰退 < 30%，策略穩健")
    elif oos_degradation < 50:
        print("  ⚠️ 警告: 樣本外績效衰退 30-50%，可能輕微過擬合")
    else:
        print("  ❌ 失敗: 樣本外績效衰退 > 50%，嚴重過擬合")

    # =========================================================================
    # 測試 2: 前瞻分析 (Walk-Forward)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 2: 前瞻分析 (6 個月滾動窗口)")
    print("=" * 80)

    window_days = 180
    window_klines = window_days * 96
    step_days = 90
    step_klines = step_days * 96

    walk_forward_results = []
    idx = 0

    while idx + window_klines < len(all_klines):
        window = all_klines[idx:idx + window_klines]
        result = run_supertrend_backtest(window, default_atr_period, default_atr_mult, default_leverage)
        if result.total_trades > 0:
            walk_forward_results.append(result)
        idx += step_klines

    print(f"\n  {'期間':<30} {'年化%':>10} {'Sharpe':>10} {'勝率':>10} {'交易數':>8}")
    print("-" * 70)

    for r in walk_forward_results:
        print(f"  {r.period:<30} {r.annual_return:>9.1f}% {r.sharpe:>10.2f} {r.win_rate:>9.1f}% {r.total_trades:>8}")

    returns = [r.annual_return for r in walk_forward_results]

    if returns:
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        positive_periods = sum(1 for r in returns if r > 0)

        print(f"\n  平均年化報酬: {avg_return:.1f}% (標準差: {std_return:.1f}%)")
        print(f"  獲利期間: {positive_periods}/{len(returns)} ({positive_periods/len(returns)*100:.0f}%)")

        if positive_periods / len(returns) >= 0.6:
            print("  ✅ 通過: 60%+ 期間獲利，策略穩定")
        else:
            print("  ❌ 失敗: <60% 期間獲利，策略不穩定")

    # =========================================================================
    # 測試 3: 參數敏感度分析
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 3: 參數敏感度分析")
    print("=" * 80)

    # 測試不同 ATR 週期
    print("\n  3.1 ATR 週期敏感度:")
    print(f"  {'ATR 週期':>10} {'年化%':>10} {'Sharpe':>10} {'交易數':>10}")
    print("-" * 45)

    atr_period_results = []
    for period in [7, 10, 14, 20, 25]:
        result = run_supertrend_backtest(all_klines, period, default_atr_mult, default_leverage)
        atr_period_results.append((period, result.annual_return))
        print(f"  {period:>10} {result.annual_return:>9.1f}% {result.sharpe:>10.2f} {result.total_trades:>10}")

    period_range = max(r[1] for r in atr_period_results) - min(r[1] for r in atr_period_results)
    print(f"\n  年化報酬變化幅度: {period_range:.1f}%")

    # 測試不同 ATR 乘數
    print("\n  3.2 ATR 乘數敏感度:")
    print(f"  {'ATR 乘數':>10} {'年化%':>10} {'Sharpe':>10} {'交易數':>10}")
    print("-" * 45)

    atr_mult_results = []
    for mult in [Decimal("2.0"), Decimal("2.5"), Decimal("3.0"), Decimal("3.5"), Decimal("4.0")]:
        result = run_supertrend_backtest(all_klines, default_atr_period, mult, default_leverage)
        atr_mult_results.append((float(mult), result.annual_return))
        print(f"  {float(mult):>10.1f} {result.annual_return:>9.1f}% {result.sharpe:>10.2f} {result.total_trades:>10}")

    mult_range = max(r[1] for r in atr_mult_results) - min(r[1] for r in atr_mult_results)
    print(f"\n  年化報酬變化幅度: {mult_range:.1f}%")

    # 測試不同槓桿
    print("\n  3.3 槓桿敏感度:")
    print(f"  {'槓桿':>10} {'年化%':>10} {'Sharpe':>10} {'回撤%':>10}")
    print("-" * 45)

    for lev in [5, 10, 15, 20, 25]:
        result = run_supertrend_backtest(all_klines, default_atr_period, default_atr_mult, lev)
        print(f"  {lev:>10}x {result.annual_return:>9.1f}% {result.sharpe:>10.2f} {result.max_drawdown:>9.1f}%")

    if period_range < 200 and mult_range < 200:
        print("\n  ✅ 通過: 參數變化對結果影響適中，策略穩健")
    else:
        print("\n  ⚠️ 警告: 策略對某些參數敏感，需謹慎調整")

    # =========================================================================
    # 測試 4: 不同市場狀況
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 4: 不同市場狀況分析")
    print("=" * 80)

    quarter_klines = len(all_klines) // 8

    market_conditions = []
    for i in range(8):
        start = i * quarter_klines
        end = (i + 1) * quarter_klines
        quarter = all_klines[start:end]

        if len(quarter) > 0:
            start_price = quarter[0].close
            end_price = quarter[-1].close
            change_pct = (end_price - start_price) / start_price * 100

            if change_pct > 20:
                condition = "牛市"
            elif change_pct < -20:
                condition = "熊市"
            else:
                condition = "盤整"

            result = run_supertrend_backtest(quarter, default_atr_period, default_atr_mult, default_leverage)
            market_conditions.append((condition, float(change_pct), result))

    print(f"\n  {'期間':<15} {'市場':>8} {'漲跌%':>10} {'年化%':>10} {'Sharpe':>10}")
    print("-" * 60)

    bull_returns = []
    bear_returns = []
    sideways_returns = []

    for i, (condition, change, result) in enumerate(market_conditions):
        print(f"  Q{i+1:<14} {condition:>8} {change:>+9.1f}% {result.annual_return:>9.1f}% {result.sharpe:>10.2f}")

        if condition == "牛市":
            bull_returns.append(result.annual_return)
        elif condition == "熊市":
            bear_returns.append(result.annual_return)
        else:
            sideways_returns.append(result.annual_return)

    print(f"\n  市場狀況統計:")
    if bull_returns:
        print(f"    牛市平均年化: {sum(bull_returns)/len(bull_returns):.1f}%")
    if bear_returns:
        print(f"    熊市平均年化: {sum(bear_returns)/len(bear_returns):.1f}%")
    if sideways_returns:
        print(f"    盤整平均年化: {sum(sideways_returns)/len(sideways_returns):.1f}%")

    all_positive = all(r > 0 for r in bull_returns + bear_returns + sideways_returns if r != 0)
    if all_positive:
        print("  ✅ 通過: 在所有市場狀況都能獲利")
    else:
        print("  ⚠️ 警告: 在某些市場狀況可能虧損")

    # =========================================================================
    # 總結
    # =========================================================================
    print("\n" + "=" * 80)
    print("  總結")
    print("=" * 80)

    print(f"""
  原始策略配置:
    - ATR 週期: {default_atr_period}
    - ATR 乘數: {default_atr_mult}
    - 槓桿: {default_leverage}x

  驗證結果:
    1. 樣本外測試: {'✅ 通過' if oos_degradation < 30 else '⚠️ 需注意' if oos_degradation < 50 else '❌ 失敗'}
       - 績效衰退: {oos_degradation:.1f}%

    2. 前瞻分析: {'✅ 通過' if positive_periods / len(returns) >= 0.6 else '❌ 需改進'}
       - 獲利期間: {positive_periods}/{len(returns)}

    3. 參數敏感度: {'✅ 穩健' if period_range < 200 and mult_range < 200 else '⚠️ 敏感'}
       - ATR 週期影響: {period_range:.1f}%
       - ATR 乘數影響: {mult_range:.1f}%

    4. 市場適應性: {'✅ 全市場' if all_positive else '⚠️ 部分市場'}

  Supertrend 是趨勢跟蹤策略:
    - 在趨勢市場表現優異
    - 在盤整市場可能虧損（正常現象）
    - 適合與均值回歸策略搭配使用
    """)

    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
