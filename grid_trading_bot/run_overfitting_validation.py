#!/usr/bin/env python3
"""
過度擬合驗證測試

測試方法:
1. 樣本外測試 (Out-of-Sample): 用前 70% 數據訓練，後 30% 測試
2. 前瞻分析 (Walk-Forward): 滾動窗口測試
3. 參數敏感度分析: 微調參數看結果變化
4. 不同市場狀況: 牛市、熊市、盤整期分開測試
5. 不同時間週期: 測試多個獨立時間段
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional, Tuple
import random

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


def calculate_bollinger_bands(closes: List[Decimal], period: int, std_mult: Decimal):
    """計算布林帶"""
    if len(closes) < period:
        return None, None, None

    recent = closes[-period:]
    sma = sum(recent) / Decimal(period)
    variance = sum((p - sma) ** 2 for p in recent) / Decimal(period)
    std = variance.sqrt()

    upper = sma + std * std_mult
    lower = sma - std * std_mult
    return upper, sma, lower


def calculate_atr(klines: list, period: int) -> Optional[Decimal]:
    """計算 ATR"""
    if len(klines) < period + 1:
        return None

    trs = []
    for i in range(-period, 0):
        high = klines[i].high
        low = klines[i].low
        prev_close = klines[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    return sum(trs) / Decimal(len(trs))


def calculate_bbw_percentile(bbw_history: List[Decimal], current_bbw: Decimal) -> int:
    """計算 BBW 百分位"""
    if not bbw_history:
        return 50
    count_below = sum(1 for b in bbw_history if b < current_bbw)
    return int(count_below / len(bbw_history) * 100)


def run_backtest_on_period(
    klines: list,
    bb_period: int = 20,
    bb_std: Decimal = Decimal("3.0"),
    leverage: int = 30,
    trailing_atr_mult: Decimal = Decimal("2.0"),
    bbw_threshold: int = 20,
    initial_capital: Decimal = Decimal("10000"),
) -> BacktestResult:
    """在指定數據上執行回測"""

    capital = initial_capital
    position = None
    trades = []
    bbw_history: List[Decimal] = []

    lookback = max(bb_period, 14, 200) + 10
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

    for i in range(lookback, len(klines)):
        kline = klines[i]
        closes = [k.close for k in klines[i - bb_period - 50:i + 1]]

        upper, middle, lower = calculate_bollinger_bands(closes, bb_period, bb_std)
        if upper is None:
            continue

        bbw = (upper - lower) / middle if middle > 0 else Decimal("0")
        bbw_history.append(bbw)
        if len(bbw_history) > 200:
            bbw_history.pop(0)

        bbw_pct = calculate_bbw_percentile(bbw_history[:-1], bbw)
        atr = calculate_atr(klines[:i + 1], 14)
        if atr is None:
            continue

        current_price = kline.close

        # 持倉處理
        if position:
            hold_bars = i - position["entry_bar"]

            # 追蹤止損
            if position["side"] == "LONG":
                position["max_price"] = max(position["max_price"], current_price)
                trailing_stop = position["max_price"] - atr * trailing_atr_mult
            else:
                position["min_price"] = min(position["min_price"], current_price)
                trailing_stop = position["min_price"] + atr * trailing_atr_mult

            should_exit = False
            exit_reason = ""

            if position["side"] == "LONG" and current_price <= trailing_stop:
                should_exit = True
                exit_reason = "追蹤止損"
            elif position["side"] == "SHORT" and current_price >= trailing_stop:
                should_exit = True
                exit_reason = "追蹤止損"

            if not should_exit:
                if position["side"] == "LONG" and current_price < lower:
                    should_exit = True
                    exit_reason = "反向訊號"
                elif position["side"] == "SHORT" and current_price > upper:
                    should_exit = True
                    exit_reason = "反向訊號"

            if not should_exit and hold_bars >= 48:
                should_exit = True
                exit_reason = "超時"

            if should_exit:
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

        # 進場條件
        if position is None and capital > 0:
            signal = None

            if bbw_pct >= bbw_threshold:
                if current_price > upper:
                    signal = "LONG"
                elif current_price < lower:
                    signal = "SHORT"

            if signal:
                position_value = capital * Decimal("0.1")
                quantity = position_value / current_price
                quantity = quantity.quantize(Decimal("0.001"))

                if quantity > 0:
                    position = {
                        "side": signal,
                        "entry_price": current_price,
                        "quantity": quantity,
                        "entry_bar": i,
                        "max_price": current_price,
                        "min_price": current_price,
                    }

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
    print("  Bollinger Bot 過度擬合驗證測試")
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

    # =========================================================================
    # 測試 1: 樣本內 vs 樣本外
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 1: 樣本內 vs 樣本外 (70/30 分割)")
    print("=" * 80)

    split_idx = int(len(all_klines) * 0.7)
    in_sample = all_klines[:split_idx]
    out_sample = all_klines[split_idx:]

    in_result = run_backtest_on_period(in_sample)
    out_result = run_backtest_on_period(out_sample)

    print(f"\n  {'指標':<15} {'樣本內 (訓練)':<20} {'樣本外 (測試)':<20} {'差異':<15}")
    print("-" * 70)
    print(f"  {'年化報酬':<15} {in_result.annual_return:>18.1f}% {out_result.annual_return:>18.1f}% {out_result.annual_return - in_result.annual_return:>+13.1f}%")
    print(f"  {'Sharpe':<15} {in_result.sharpe:>19.2f} {out_result.sharpe:>19.2f} {out_result.sharpe - in_result.sharpe:>+14.2f}")
    print(f"  {'勝率':<15} {in_result.win_rate:>18.1f}% {out_result.win_rate:>18.1f}% {out_result.win_rate - in_result.win_rate:>+13.1f}%")
    print(f"  {'交易數':<15} {in_result.total_trades:>19} {out_result.total_trades:>19}")
    print(f"  {'最大回撤':<15} {in_result.max_drawdown:>18.1f}% {out_result.max_drawdown:>18.1f}%")

    # 過度擬合判斷
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
    window_klines = window_days * 96  # 15分鐘 K 線
    step_days = 90
    step_klines = step_days * 96

    walk_forward_results = []
    idx = 0

    while idx + window_klines < len(all_klines):
        window = all_klines[idx:idx + window_klines]
        result = run_backtest_on_period(window)
        if result.total_trades > 0:
            walk_forward_results.append(result)
        idx += step_klines

    print(f"\n  {'期間':<30} {'年化%':>10} {'Sharpe':>10} {'勝率':>10} {'交易數':>8}")
    print("-" * 70)

    for r in walk_forward_results:
        print(f"  {r.period:<30} {r.annual_return:>9.1f}% {r.sharpe:>10.2f} {r.win_rate:>9.1f}% {r.total_trades:>8}")

    # 統計
    returns = [r.annual_return for r in walk_forward_results]
    sharpes = [r.sharpe for r in walk_forward_results]

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

    # 測試不同 bb_std
    print("\n  3.1 BB 標準差敏感度:")
    print(f"  {'BB Std':>10} {'年化%':>10} {'Sharpe':>10} {'交易數':>10}")
    print("-" * 45)

    std_results = []
    for std in [Decimal("2.5"), Decimal("2.75"), Decimal("3.0"), Decimal("3.25"), Decimal("3.5")]:
        result = run_backtest_on_period(all_klines, bb_std=std)
        std_results.append((float(std), result.annual_return))
        print(f"  {float(std):>10.2f} {result.annual_return:>9.1f}% {result.sharpe:>10.2f} {result.total_trades:>10}")

    # 計算敏感度
    std_range = max(r[1] for r in std_results) - min(r[1] for r in std_results)
    print(f"\n  年化報酬變化幅度: {std_range:.1f}%")

    # 測試不同槓桿
    print("\n  3.2 槓桿敏感度:")
    print(f"  {'槓桿':>10} {'年化%':>10} {'Sharpe':>10} {'回撤%':>10}")
    print("-" * 45)

    for lev in [10, 20, 30, 40, 50]:
        result = run_backtest_on_period(all_klines, leverage=lev)
        print(f"  {lev:>10}x {result.annual_return:>9.1f}% {result.sharpe:>10.2f} {result.max_drawdown:>9.1f}%")

    # 測試不同追蹤止損
    print("\n  3.3 追蹤止損 ATR 乘數敏感度:")
    print(f"  {'ATR Mult':>10} {'年化%':>10} {'Sharpe':>10} {'勝率':>10}")
    print("-" * 45)

    atr_results = []
    for atr_mult in [Decimal("1.5"), Decimal("2.0"), Decimal("2.5"), Decimal("3.0")]:
        result = run_backtest_on_period(all_klines, trailing_atr_mult=atr_mult)
        atr_results.append((float(atr_mult), result.annual_return))
        print(f"  {float(atr_mult):>10.1f} {result.annual_return:>9.1f}% {result.sharpe:>10.2f} {result.win_rate:>9.1f}%")

    atr_range = max(r[1] for r in atr_results) - min(r[1] for r in atr_results)
    print(f"\n  年化報酬變化幅度: {atr_range:.1f}%")

    if std_range < 50 and atr_range < 50:
        print("  ✅ 通過: 參數變化對結果影響適中，策略穩健")
    else:
        print("  ⚠️ 警告: 策略對某些參數敏感，需謹慎調整")

    # =========================================================================
    # 測試 4: 不同市場狀況
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 4: 不同市場狀況分析")
    print("=" * 80)

    # 根據價格變化識別市場狀況
    # 計算每個季度的市場類型
    quarter_klines = len(all_klines) // 8  # 分成 8 個季度

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

            result = run_backtest_on_period(quarter)
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

    # 檢查是否在所有市場狀況都能獲利
    all_positive = all(r > 0 for r in bull_returns + bear_returns + sideways_returns if r != 0)
    if all_positive:
        print("  ✅ 通過: 在所有市場狀況都能獲利")
    else:
        print("  ⚠️ 警告: 在某些市場狀況可能虧損")

    # =========================================================================
    # 測試 5: 蒙地卡羅模擬
    # =========================================================================
    print("\n" + "=" * 80)
    print("  測試 5: 蒙地卡羅模擬 (交易順序隨機化)")
    print("=" * 80)

    # 先執行一次回測獲取所有交易
    base_result = run_backtest_on_period(all_klines)

    # 模擬隨機交易順序
    n_simulations = 1000
    simulated_returns = []

    # 假設每筆交易的報酬率
    if base_result.total_trades > 0:
        avg_trade_return = base_result.annual_return / base_result.total_trades

        for _ in range(n_simulations):
            # 隨機化交易順序的報酬累積
            capital = 10000
            for _ in range(base_result.total_trades):
                # 根據勝率決定這筆交易是贏還是輸
                if random.random() < base_result.win_rate / 100:
                    # 贏的交易
                    trade_return = abs(avg_trade_return) * (0.5 + random.random())
                else:
                    # 輸的交易
                    trade_return = -abs(avg_trade_return) * (0.5 + random.random())

                capital *= (1 + trade_return / 100)

            final_return = (capital / 10000 - 1) * 100
            simulated_returns.append(final_return)

        simulated_returns.sort()

        percentile_5 = simulated_returns[int(n_simulations * 0.05)]
        percentile_50 = simulated_returns[int(n_simulations * 0.50)]
        percentile_95 = simulated_returns[int(n_simulations * 0.95)]

        print(f"\n  蒙地卡羅模擬 ({n_simulations} 次):")
        print(f"    5th 百分位:  {percentile_5:>+.1f}%")
        print(f"    50th 百分位: {percentile_50:>+.1f}%")
        print(f"    95th 百分位: {percentile_95:>+.1f}%")
        print(f"    實際結果:    {base_result.annual_return:>+.1f}%")

        if percentile_5 > 0:
            print("  ✅ 通過: 即使在 5% 最差情況下仍獲利")
        elif percentile_50 > 0:
            print("  ⚠️ 警告: 策略有獲利但風險較高")
        else:
            print("  ❌ 失敗: 策略風險過高")

    # =========================================================================
    # 總結
    # =========================================================================
    print("\n" + "=" * 80)
    print("  總結")
    print("=" * 80)

    print(f"""
  原始策略配置:
    - BB 標準差: 3.0
    - 槓桿: 30x
    - 追蹤止損: 2.0 ATR

  驗證結果:
    1. 樣本外測試: {'✅ 通過' if oos_degradation < 30 else '⚠️ 需注意'}
       - 績效衰退: {oos_degradation:.1f}%

    2. 前瞻分析: {'✅ 通過' if positive_periods / len(returns) >= 0.6 else '❌ 需改進'}
       - 獲利期間: {positive_periods}/{len(returns)}

    3. 參數敏感度: {'✅ 穩健' if std_range < 50 and atr_range < 50 else '⚠️ 敏感'}
       - BB Std 影響: {std_range:.1f}%
       - ATR Mult 影響: {atr_range:.1f}%

    4. 市場適應性: {'✅ 全市場' if all_positive else '⚠️ 部分市場'}

  建議:
    - 如果多數測試通過，策略應該是穩健的
    - 如果樣本外績效衰退 > 30%，考慮簡化策略
    - 如果參數敏感度高，避免使用極端參數值
    """)

    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
