#!/usr/bin/env python3
"""
Bollinger Bot 順勢突破策略回測

策略說明:
- 原本均值回歸: 價格觸碰下軌做多、觸碰上軌做空（預期回歸中軌）
- 順勢突破策略: 價格突破上軌做多、突破下軌做空（追逐趨勢）

突破策略邏輯:
1. 價格收盤在上軌之上 → 做多 (強勢突破)
2. 價格收盤在下軌之下 → 做空 (弱勢突破)
3. 用追蹤止損或反向訊號出場
4. BBW 擴張過濾：只在波動率增加時交易
"""

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Optional

# Add project root to path
sys.path.insert(0, "/mnt/c/trading/grid_trading_bot")

from src.exchange import ExchangeClient


@dataclass
class BreakoutConfig:
    """突破策略配置"""
    name: str
    bb_period: int = 20
    bb_std: Decimal = Decimal("2.0")

    # 突破過濾
    breakout_confirm_bars: int = 1  # 需要連續幾根 K 線確認突破
    require_bbw_expansion: bool = True  # 是否要求 BBW 擴張
    bbw_expansion_pct: int = 20  # BBW 需高於此百分位才交易

    # 出場策略
    use_trailing_stop: bool = True
    trailing_atr_mult: Decimal = Decimal("2.0")  # 追蹤止損 ATR 乘數
    atr_period: int = 14
    max_hold_bars: int = 48  # 最大持倉時間

    # 槓桿和倉位
    leverage: int = 10
    position_size_pct: Decimal = Decimal("0.1")


@dataclass
class Trade:
    """交易記錄"""
    side: str  # "LONG" or "SHORT"
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    hold_bars: int


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

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

    return sum(trs) / Decimal(len(trs))


def calculate_bbw_percentile(bbw_history: List[Decimal], current_bbw: Decimal) -> int:
    """計算 BBW 百分位"""
    if not bbw_history:
        return 50

    count_below = sum(1 for b in bbw_history if b < current_bbw)
    return int(count_below / len(bbw_history) * 100)


async def run_breakout_backtest(klines: list, config: BreakoutConfig, initial_capital: Decimal = Decimal("10000")):
    """執行突破策略回測"""

    capital = initial_capital
    position = None
    trades: List[Trade] = []
    bbw_history: List[Decimal] = []

    # 需要足夠的歷史數據
    lookback = max(config.bb_period, config.atr_period, 200) + 10

    equity_curve = [float(capital)]
    max_equity = float(capital)
    max_drawdown = Decimal("0")

    fee_rate = Decimal("0.0004")  # 0.04% taker fee

    for i in range(lookback, len(klines)):
        kline = klines[i]
        closes = [k.close for k in klines[i - config.bb_period - 50:i + 1]]

        # 計算指標
        upper, middle, lower = calculate_bollinger_bands(
            closes, config.bb_period, config.bb_std
        )
        if upper is None:
            continue

        # 計算 BBW
        bbw = (upper - lower) / middle if middle > 0 else Decimal("0")
        bbw_history.append(bbw)
        if len(bbw_history) > 200:
            bbw_history.pop(0)

        bbw_pct = calculate_bbw_percentile(bbw_history[:-1], bbw)

        # 計算 ATR
        atr = calculate_atr(klines[:i + 1], config.atr_period)
        if atr is None:
            continue

        current_price = kline.close

        # 檢查持倉
        if position:
            hold_bars = i - position["entry_bar"]

            # 計算追蹤止損
            trailing_stop = None
            if config.use_trailing_stop:
                if position["side"] == "LONG":
                    # 多頭追蹤止損：最高價 - ATR * mult
                    position["max_price"] = max(position["max_price"], current_price)
                    trailing_stop = position["max_price"] - atr * config.trailing_atr_mult
                else:
                    # 空頭追蹤止損：最低價 + ATR * mult
                    position["min_price"] = min(position["min_price"], current_price)
                    trailing_stop = position["min_price"] + atr * config.trailing_atr_mult

            # 出場條件
            should_exit = False
            exit_reason = ""

            # 1. 追蹤止損觸發
            if trailing_stop:
                if position["side"] == "LONG" and current_price <= trailing_stop:
                    should_exit = True
                    exit_reason = "追蹤止損"
                elif position["side"] == "SHORT" and current_price >= trailing_stop:
                    should_exit = True
                    exit_reason = "追蹤止損"

            # 2. 反向訊號
            if not should_exit:
                if position["side"] == "LONG" and current_price < lower:
                    should_exit = True
                    exit_reason = "反向訊號"
                elif position["side"] == "SHORT" and current_price > upper:
                    should_exit = True
                    exit_reason = "反向訊號"

            # 3. 超時
            if not should_exit and hold_bars >= config.max_hold_bars:
                should_exit = True
                exit_reason = "超時"

            # 執行出場
            if should_exit:
                entry_price = position["entry_price"]
                quantity = position["quantity"]

                if position["side"] == "LONG":
                    pnl = (current_price - entry_price) * quantity * Decimal(config.leverage)
                else:
                    pnl = (entry_price - current_price) * quantity * Decimal(config.leverage)

                # 扣除手續費
                fee = (entry_price + current_price) * quantity * fee_rate
                pnl -= fee

                capital += pnl

                trades.append(Trade(
                    side=position["side"],
                    entry_price=entry_price,
                    exit_price=current_price,
                    quantity=quantity,
                    pnl=pnl,
                    entry_time=position["entry_time"],
                    exit_time=kline.close_time,
                    exit_reason=exit_reason,
                    hold_bars=hold_bars,
                ))

                position = None

        # 檢查進場條件（沒有持倉時）
        if position is None and capital > 0:
            signal = None

            # BBW 擴張過濾
            bbw_ok = True
            if config.require_bbw_expansion:
                bbw_ok = bbw_pct >= config.bbw_expansion_pct

            if bbw_ok:
                # 突破上軌 → 做多
                if current_price > upper:
                    signal = "LONG"
                # 突破下軌 → 做空
                elif current_price < lower:
                    signal = "SHORT"

            if signal:
                # 計算倉位
                position_value = capital * config.position_size_pct
                quantity = position_value / current_price
                quantity = quantity.quantize(Decimal("0.001"))

                if quantity > 0:
                    position = {
                        "side": signal,
                        "entry_price": current_price,
                        "quantity": quantity,
                        "entry_time": kline.close_time,
                        "entry_bar": i,
                        "max_price": current_price,
                        "min_price": current_price,
                    }

        # 更新權益曲線
        current_equity = float(capital)
        if position:
            # 加上未實現損益
            if position["side"] == "LONG":
                unrealized = float((current_price - position["entry_price"]) * position["quantity"] * Decimal(config.leverage))
            else:
                unrealized = float((position["entry_price"] - current_price) * position["quantity"] * Decimal(config.leverage))
            current_equity += unrealized

        equity_curve.append(current_equity)
        max_equity = max(max_equity, current_equity)

        drawdown = Decimal(str((max_equity - current_equity) / max_equity * 100)) if max_equity > 0 else Decimal("0")
        max_drawdown = max(max_drawdown, drawdown)

    # 強制平倉（如果還有持倉）
    if position:
        current_price = klines[-1].close
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        if position["side"] == "LONG":
            pnl = (current_price - entry_price) * quantity * Decimal(config.leverage)
        else:
            pnl = (entry_price - current_price) * quantity * Decimal(config.leverage)

        fee = (entry_price + current_price) * quantity * fee_rate
        pnl -= fee
        capital += pnl

        trades.append(Trade(
            side=position["side"],
            entry_price=entry_price,
            exit_price=current_price,
            quantity=quantity,
            pnl=pnl,
            entry_time=position["entry_time"],
            exit_time=klines[-1].close_time,
            exit_reason="回測結束",
            hold_bars=len(klines) - position["entry_bar"],
        ))

    # 計算統計
    total_trades = len(trades)
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    win_rate = len(winning) / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(t.pnl for t in trades)

    # 計算年化報酬
    days = (klines[-1].close_time - klines[0].close_time).days
    years = days / 365 if days > 0 else 1
    annual_return = ((float(capital) / float(initial_capital)) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 計算 Sharpe Ratio
    if len(trades) > 1:
        returns = [float(t.pnl / initial_capital) for t in trades]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        # 年化 Sharpe（假設每年 trades_per_year 筆交易）
        trades_per_year = total_trades / years if years > 0 else total_trades
        sharpe = (avg_return * trades_per_year) / (std_return * (trades_per_year ** 0.5)) if std_return > 0 else 0
    else:
        sharpe = 0

    return {
        "config": config,
        "total_trades": total_trades,
        "trades_per_year": total_trades / years if years > 0 else 0,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_capital": capital,
        "trades": trades,
    }


async def main():
    print("=" * 70)
    print("  Bollinger Bot 順勢突破策略回測")
    print("  目標: 年化 20%+ | Sharpe > 1")
    print("=" * 70)

    # 連接交易所
    exchange = ExchangeClient()
    await exchange.connect()

    # 獲取歷史數據
    print(f"\n正在獲取 BTCUSDT 15m 歷史數據 (730 天)...")

    from src.core.models import KlineInterval

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=730)

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        # Convert to milliseconds timestamp
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
        print(f"  已獲取 {len(all_klines)} 根 K 線")

        if len(klines) < 1500:
            break

    print(f"  已獲取 {len(all_klines)} 根 K 線")

    # 定義策略配置 - 基於上一輪結果優化
    # 發現: 寬帶(2.5std) 效果最好，但 Sharpe 太低
    # 優化方向: 更寬的帶 + 更嚴格的過濾 + 更好的止損
    strategies = [
        # 寬帶策略優化 - 基礎版
        BreakoutConfig(name="寬帶2.5-20x", leverage=20, bb_std=Decimal("2.5")),
        BreakoutConfig(name="寬帶2.5-30x", leverage=30, bb_std=Decimal("2.5")),
        BreakoutConfig(name="寬帶2.5-40x", leverage=40, bb_std=Decimal("2.5")),

        # 超寬帶 (3.0 std) - 只抓大突破
        BreakoutConfig(name="超寬帶3.0-20x", leverage=20, bb_std=Decimal("3.0")),
        BreakoutConfig(name="超寬帶3.0-30x", leverage=30, bb_std=Decimal("3.0")),
        BreakoutConfig(name="超寬帶3.0-40x", leverage=40, bb_std=Decimal("3.0")),
        BreakoutConfig(name="超寬帶3.0-50x", leverage=50, bb_std=Decimal("3.0")),

        # 寬帶 + 高波動過濾
        BreakoutConfig(name="寬帶+高波動-20x", leverage=20, bb_std=Decimal("2.5"), bbw_expansion_pct=40),
        BreakoutConfig(name="寬帶+高波動-30x", leverage=30, bb_std=Decimal("2.5"), bbw_expansion_pct=40),
        BreakoutConfig(name="寬帶+高波動-40x", leverage=40, bb_std=Decimal("2.5"), bbw_expansion_pct=40),

        # 超寬帶 + 高波動
        BreakoutConfig(name="超寬+高波動-30x", leverage=30, bb_std=Decimal("3.0"), bbw_expansion_pct=40),
        BreakoutConfig(name="超寬+高波動-40x", leverage=40, bb_std=Decimal("3.0"), bbw_expansion_pct=40),
        BreakoutConfig(name="超寬+高波動-50x", leverage=50, bb_std=Decimal("3.0"), bbw_expansion_pct=40),

        # 寬帶 + 寬止損 (讓利潤跑)
        BreakoutConfig(name="寬帶+寬止損-20x", leverage=20, bb_std=Decimal("2.5"), trailing_atr_mult=Decimal("3.0"), max_hold_bars=96),
        BreakoutConfig(name="寬帶+寬止損-30x", leverage=30, bb_std=Decimal("2.5"), trailing_atr_mult=Decimal("3.0"), max_hold_bars=96),
        BreakoutConfig(name="寬帶+寬止損-40x", leverage=40, bb_std=Decimal("2.5"), trailing_atr_mult=Decimal("3.0"), max_hold_bars=96),

        # 超寬帶 + 寬止損
        BreakoutConfig(name="超寬+寬止損-30x", leverage=30, bb_std=Decimal("3.0"), trailing_atr_mult=Decimal("3.0"), max_hold_bars=96),
        BreakoutConfig(name="超寬+寬止損-40x", leverage=40, bb_std=Decimal("3.0"), trailing_atr_mult=Decimal("3.0"), max_hold_bars=96),
        BreakoutConfig(name="超寬+寬止損-50x", leverage=50, bb_std=Decimal("3.0"), trailing_atr_mult=Decimal("3.0"), max_hold_bars=96),

        # 長週期 (30) + 寬帶 - 更穩定的訊號
        BreakoutConfig(name="長週期+寬帶-30x", leverage=30, bb_period=30, bb_std=Decimal("2.5"), max_hold_bars=72),
        BreakoutConfig(name="長週期+寬帶-40x", leverage=40, bb_period=30, bb_std=Decimal("2.5"), max_hold_bars=72),

        # 組合策略: 長週期 + 超寬帶 + 高波動
        BreakoutConfig(name="全嚴格-30x", leverage=30, bb_period=30, bb_std=Decimal("3.0"), bbw_expansion_pct=40, max_hold_bars=72),
        BreakoutConfig(name="全嚴格-40x", leverage=40, bb_period=30, bb_std=Decimal("3.0"), bbw_expansion_pct=40, max_hold_bars=72),
        BreakoutConfig(name="全嚴格-50x", leverage=50, bb_period=30, bb_std=Decimal("3.0"), bbw_expansion_pct=40, max_hold_bars=72),
    ]

    # 執行回測
    print(f"\n正在測試 {len(strategies)} 種策略配置...")

    results = []
    for config in strategies:
        result = await run_breakout_backtest(all_klines, config)
        results.append(result)

    # 按年化報酬排序
    results.sort(key=lambda x: x["annual_return"], reverse=True)

    # 輸出結果
    print("\n" + "=" * 120)
    print(f"{'策略':<20} {'交易數':>8} {'年交易':>8} {'年化%':>8} {'Sharpe':>8} {'勝率':>8} {'盈虧':>12} {'回撤':>8}")
    print("=" * 120)

    for r in results:
        print(
            f"  {r['config'].name:<18} "
            f"{r['total_trades']:>6} "
            f"{r['trades_per_year']:>8.0f} "
            f"{r['annual_return']:>7.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['win_rate']:>7.1f}% "
            f"{float(r['total_pnl']):>+11.0f} "
            f"{float(r['max_drawdown']):>7.0f}"
        )

    print("=" * 120)

    # 篩選符合條件的策略
    print("\n" + "=" * 70)
    print("  符合條件的策略 (年化 >= 20% + Sharpe >= 1)")
    print("=" * 70)

    qualified = [r for r in results if r["annual_return"] >= 20 and r["sharpe"] >= 1]

    if qualified:
        for r in qualified:
            config = r["config"]
            print(f"\n  ✅ {config.name}")
            print(f"     年化報酬: {r['annual_return']:.1f}%")
            print(f"     Sharpe: {r['sharpe']:.2f}")
            print(f"     年交易數: {r['trades_per_year']:.0f}")
            print(f"     勝率: {r['win_rate']:.1f}%")
            print(f"     最大回撤: {float(r['max_drawdown']):.1f}%")
            print(f"     參數: BB({config.bb_period}, {config.bb_std}), "
                  f"ATR止損({config.trailing_atr_mult}), 槓桿={config.leverage}x")
    else:
        print("\n  ⚠️ 沒有策略同時滿足所有條件")

        # 顯示最佳策略
        best = results[0]
        print(f"\n  最佳策略: {best['config'].name}")
        print(f"     年化報酬: {best['annual_return']:.1f}%")
        print(f"     Sharpe: {best['sharpe']:.2f}")

    print("\n" + "=" * 70)

    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
