#!/usr/bin/env python3
"""
Bollinger 策略 Walk-Forward 驗證腳本.

驗證優化後參數的樣本外表現和穩健性。
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.strategy.bollinger import (
    BollingerBacktestStrategy,
    BollingerStrategyConfig,
    BollingerMode,
)
from src.core.models import Kline


def load_klines(filepath: str) -> list[Kline]:
    """從本地 JSON 檔案載入 K 線數據."""
    print(f"載入數據: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    klines = []
    for k in data["klines"]:
        klines.append(Kline(
            symbol=data["metadata"]["symbol"],
            interval=data["metadata"]["interval"],
            open_time=datetime.fromisoformat(k["open_time"]),
            close_time=datetime.fromisoformat(k["close_time"]),
            open=Decimal(k["open"]),
            high=Decimal(k["high"]),
            low=Decimal(k["low"]),
            close=Decimal(k["close"]),
            volume=Decimal(k["volume"]),
        ))

    print(f"載入 {len(klines):,} 根 K 線")
    print(f"範圍: {klines[0].open_time.strftime('%Y-%m-%d')} ~ {klines[-1].open_time.strftime('%Y-%m-%d')}")
    return klines


def run_backtest(klines: list[Kline], leverage: int = 19) -> dict:
    """執行回測並返回結果."""
    # 優化後的參數 (19x 槓桿激進版)
    config = BollingerStrategyConfig(
        mode=BollingerMode.BB_TREND_GRID,
        bb_period=12,
        bb_std=Decimal("2.0"),
        grid_count=6,
        grid_range_pct=Decimal("0.02"),
        take_profit_grids=2,
        stop_loss_pct=Decimal("0.025"),
        use_hysteresis=False,
        hysteresis_pct=Decimal("0.002"),
        use_signal_cooldown=False,
        cooldown_bars=0,
    )

    strategy = BollingerBacktestStrategy(config)

    bt_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        leverage=leverage,
        fee_rate=Decimal("0.0004"),
        slippage_pct=Decimal("0.0001"),
    )

    engine = BacktestEngine(bt_config)
    result = engine.run(klines, strategy)

    return {
        "total_return": float(result.total_profit_pct),
        "max_drawdown": float(result.max_drawdown_pct),
        "sharpe": float(result.sharpe_ratio),
        "win_rate": float(result.win_rate),
        "trades": result.total_trades,
    }


def walk_forward_validation(klines: list[Kline], n_splits: int = 9, leverage: int = 19):
    """
    Walk-Forward 驗證.

    將數據分成多個時間段，每段獨立回測，檢驗參數穩健性。
    """
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 驗證 (Bollinger 優化參數)")
    print(f"  槓桿: {leverage}x, 分割數: {n_splits}")
    print(f"{'=' * 60}")

    total_bars = len(klines)
    split_size = total_bars // n_splits

    results = []
    profitable_periods = 0

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, total_bars)

        if end_idx - start_idx < 100:
            continue

        period_klines = klines[start_idx:end_idx]
        start_date = period_klines[0].open_time.strftime('%Y-%m-%d')
        end_date = period_klines[-1].open_time.strftime('%Y-%m-%d')

        result = run_backtest(period_klines, leverage)
        results.append(result)

        status = "✅" if result["total_return"] > 0 else "❌"
        if result["total_return"] > 0:
            profitable_periods += 1

        print(f"\n期間 {i+1}/{n_splits}: {start_date} ~ {end_date}")
        print(f"  報酬: {result['total_return']:+.2f}% {status}")
        print(f"  回撤: {result['max_drawdown']:.2f}%")
        print(f"  交易: {result['trades']}")

    # 統計
    consistency = profitable_periods / len(results) * 100
    avg_return = sum(r["total_return"] for r in results) / len(results)
    avg_drawdown = sum(r["max_drawdown"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 統計")
    print(f"{'=' * 60}")
    print(f"  一致性: {profitable_periods}/{len(results)} ({consistency:.1f}%)")
    print(f"  平均報酬: {avg_return:.2f}%")
    print(f"  平均回撤: {avg_drawdown:.2f}%")

    if consistency >= 70:
        print(f"\n  ✅ PASS: 一致性 >= 70%")
    else:
        print(f"\n  ⚠️ WARNING: 一致性 < 70%")

    return {
        "consistency": consistency,
        "profitable_periods": profitable_periods,
        "total_periods": len(results),
        "avg_return": avg_return,
        "avg_drawdown": avg_drawdown,
        "results": results,
    }


def monte_carlo_test(klines: list[Kline], n_tests: int = 15, leverage: int = 19):
    """
    Monte Carlo 穩健性測試.

    從不同起始點開始回測，檢驗參數對起始條件的敏感度。
    """
    print(f"\n{'=' * 60}")
    print(f"  Monte Carlo 穩健性測試")
    print(f"  測試數: {n_tests}")
    print(f"{'=' * 60}")

    total_bars = len(klines)
    # 使用不同的起始偏移量
    offsets = [int(total_bars * i / (n_tests + 1)) for i in range(1, n_tests + 1)]

    results = []
    profitable = 0

    for i, offset in enumerate(offsets):
        # 從 offset 開始，使用 70% 的數據
        test_size = int(total_bars * 0.7)
        end_idx = min(offset + test_size, total_bars)

        if end_idx - offset < 500:
            continue

        test_klines = klines[offset:end_idx]
        result = run_backtest(test_klines, leverage)
        results.append(result)

        if result["total_return"] > 0:
            profitable += 1

        status = "✅" if result["total_return"] > 0 else "❌"
        print(f"  測試 {i+1:2d}: 報酬 {result['total_return']:+6.2f}%, 回撤 {result['max_drawdown']:5.2f}% {status}")

    robustness = profitable / len(results) * 100

    print(f"\n  穩健性: {profitable}/{len(results)} ({robustness:.1f}%)")

    if robustness >= 80:
        print(f"  ✅ ROBUST: 穩健性 >= 80%")
    elif robustness >= 60:
        print(f"  ⚠️ MODERATE: 穩健性 60-80%")
    else:
        print(f"  ❌ WEAK: 穩健性 < 60%")

    return {
        "robustness": robustness,
        "profitable": profitable,
        "total_tests": len(results),
        "results": results,
    }


def main():
    # 載入數據
    klines = load_klines("data/historical/BTCUSDT_1h_730d.json")

    leverage = 19

    # 完整回測
    print(f"\n{'=' * 60}")
    print(f"  完整回測 (優化參數, {leverage}x 槓桿)")
    print(f"{'=' * 60}")

    full_result = run_backtest(klines, leverage)
    total_return = full_result["total_return"]
    annual_return = ((1 + total_return/100) ** 0.5 - 1) * 100

    print(f"  總報酬 (2年): {total_return:.2f}%")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  最大回撤: {full_result['max_drawdown']:.2f}%")
    print(f"  Sharpe: {full_result['sharpe']:.2f}")
    print(f"  勝率: {full_result['win_rate']:.1f}%")
    print(f"  交易數: {full_result['trades']}")

    # Walk-Forward 驗證
    wf_result = walk_forward_validation(klines, n_splits=9, leverage=leverage)

    # Monte Carlo 測試
    mc_result = monte_carlo_test(klines, n_tests=15, leverage=leverage)

    # 總結
    print(f"\n{'=' * 60}")
    print(f"  驗證總結")
    print(f"{'=' * 60}")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  W-F 一致性: {wf_result['consistency']:.1f}%")
    print(f"  Monte Carlo: {mc_result['robustness']:.1f}%")

    passed = wf_result['consistency'] >= 50 and mc_result['robustness'] >= 60
    if passed:
        print(f"\n  ✅ 驗證通過 - 可用於實戰")
    else:
        print(f"\n  ⚠️ 驗證未通過 - 建議保守配置")

    return passed


if __name__ == "__main__":
    main()
