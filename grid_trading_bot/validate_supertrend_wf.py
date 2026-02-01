#!/usr/bin/env python3
"""
Supertrend v2 策略 Walk-Forward 驗證腳本 (HYBRID_GRID + vol filter + timeout).

驗證優化後參數的樣本外表現和穩健性。
包含：完整回測、Walk-Forward、Monte Carlo、OOS 測試、參數敏感度、白噪音基準。

通過標準：
- Walk-Forward 一致性 >= 50%
- Monte Carlo 穩健性 >= 60%
- OOS/IS Sharpe 比率 >= 0.5
- 參數穩健性 >= 80%

用法:
    python validate_supertrend_wf.py --leverage 7
    python validate_supertrend_wf.py --leverage 7 --full
"""

import argparse
import json
import random
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.strategy.supertrend import (
    SupertrendBacktestStrategy,
    SupertrendStrategyConfig,
    SupertrendMode,
)
from src.core.models import Kline


# ============================================================
# 優化後最佳參數（優化完成後請更新此處）
# ============================================================
OPTIMIZED_PARAMS = {
    "atr_period": 7,
    "atr_multiplier": 4.5,
    "grid_count": 9,
    "grid_atr_multiplier": 11.0,
    "take_profit_grids": 1,
    "stop_loss_pct": 0.01,
    "rsi_period": 18,
    "rsi_overbought": 73,
    "rsi_oversold": 27,
    "min_trend_bars": 1,
    "use_hysteresis": False,
    "hysteresis_pct": 0.0045,
    "use_signal_cooldown": True,
    "cooldown_bars": 0,
    "trailing_stop_pct": 0.03,
    # v2: Volatility Regime Filter
    "use_volatility_filter": True,
    "vol_ratio_low": 0.6,
    "vol_ratio_high": 2.0,
    # v2: Timeout Exit
    "max_hold_bars": 28,
    # HYBRID_GRID params
    "hybrid_grid_bias_pct": 0.75,
    "hybrid_tp_multiplier_trend": 1.75,
    "hybrid_tp_multiplier_counter": 0.5,
    "hybrid_sl_multiplier_counter": 0.7,
    "hybrid_rsi_asymmetric": False,
}


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


def run_backtest(klines: list[Kline], leverage: int = 7, params: dict | None = None) -> dict:
    """執行回測並返回結果."""
    p = params or OPTIMIZED_PARAMS

    # Determine mode from params
    mode = SupertrendMode.HYBRID_GRID if "hybrid_grid_bias_pct" in p else SupertrendMode.TREND_GRID

    config = SupertrendStrategyConfig(
        mode=mode,
        atr_period=p["atr_period"],
        atr_multiplier=Decimal(str(p["atr_multiplier"])),
        grid_count=p["grid_count"],
        grid_atr_multiplier=Decimal(str(p["grid_atr_multiplier"])),
        take_profit_grids=p["take_profit_grids"],
        stop_loss_pct=Decimal(str(p["stop_loss_pct"])),
        use_rsi_filter=True,
        rsi_period=p["rsi_period"],
        rsi_overbought=p["rsi_overbought"],
        rsi_oversold=p["rsi_oversold"],
        min_trend_bars=p["min_trend_bars"],
        use_hysteresis=p["use_hysteresis"],
        hysteresis_pct=Decimal(str(p["hysteresis_pct"])),
        use_signal_cooldown=p["use_signal_cooldown"],
        cooldown_bars=p["cooldown_bars"],
        use_trailing_stop=True,
        trailing_stop_pct=Decimal(str(p["trailing_stop_pct"])),
        # v2: Volatility Regime Filter
        use_volatility_filter=p.get("use_volatility_filter", True),
        vol_ratio_low=p.get("vol_ratio_low", 0.5),
        vol_ratio_high=p.get("vol_ratio_high", 2.0),
        # v2: Timeout Exit
        max_hold_bars=p.get("max_hold_bars", 16),
        # HYBRID_GRID params
        hybrid_grid_bias_pct=Decimal(str(p.get("hybrid_grid_bias_pct", "0.6"))),
        hybrid_tp_multiplier_trend=Decimal(str(p.get("hybrid_tp_multiplier_trend", "1.0"))),
        hybrid_tp_multiplier_counter=Decimal(str(p.get("hybrid_tp_multiplier_counter", "0.5"))),
        hybrid_sl_multiplier_counter=Decimal(str(p.get("hybrid_sl_multiplier_counter", "0.7"))),
        hybrid_rsi_asymmetric=p.get("hybrid_rsi_asymmetric", True),
    )

    strategy = SupertrendBacktestStrategy(config)

    bt_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0006"),  # 0.04% 手續費 + ~0.02% 資金費率近似
        slippage_pct=Decimal("0.0005"),  # 0.05% 滑價
    ).with_leverage(leverage)

    engine = BacktestEngine(bt_config)
    result = engine.run(klines, strategy)

    return {
        "total_return": float(result.total_profit_pct),
        "max_drawdown": float(result.max_drawdown_pct),
        "sharpe": float(result.sharpe_ratio),
        "win_rate": float(result.win_rate),
        "trades": result.total_trades,
    }


def walk_forward_validation(klines: list[Kline], n_splits: int = 9, leverage: int = 7):
    """Walk-Forward 驗證."""
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 驗證 (Supertrend TREND_GRID)")
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

        status = "PASS" if result["total_return"] > 0 else "FAIL"
        if result["total_return"] > 0:
            profitable_periods += 1

        print(f"\n期間 {i+1}/{n_splits}: {start_date} ~ {end_date}")
        print(f"  報酬: {result['total_return']:+.2f}% [{status}]")
        print(f"  回撤: {result['max_drawdown']:.2f}%")
        print(f"  交易: {result['trades']}")

    consistency = profitable_periods / len(results) * 100
    avg_return = sum(r["total_return"] for r in results) / len(results)
    avg_drawdown = sum(r["max_drawdown"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 統計")
    print(f"{'=' * 60}")
    print(f"  一致性: {profitable_periods}/{len(results)} ({consistency:.1f}%)")
    print(f"  平均報酬: {avg_return:.2f}%")
    print(f"  平均回撤: {avg_drawdown:.2f}%")

    passed = consistency >= 50
    print(f"\n  {'PASS' if passed else 'FAIL'}: 一致性 {'>='}  50% (實際 {consistency:.1f}%)")

    return {
        "consistency": consistency,
        "profitable_periods": profitable_periods,
        "total_periods": len(results),
        "avg_return": avg_return,
        "avg_drawdown": avg_drawdown,
        "passed": passed,
        "results": results,
    }


def sliding_window_robustness_test(klines: list[Kline], n_tests: int = 15, leverage: int = 7):
    """滑動窗口穩健性測試."""
    print(f"\n{'=' * 60}")
    print(f"  滑動窗口穩健性測試")
    print(f"  測試數: {n_tests}")
    print(f"{'=' * 60}")

    total_bars = len(klines)
    offsets = [int(total_bars * i / (n_tests + 1)) for i in range(1, n_tests + 1)]

    results = []
    profitable = 0

    for i, offset in enumerate(offsets):
        test_size = int(total_bars * 0.7)
        end_idx = min(offset + test_size, total_bars)

        if end_idx - offset < 500:
            continue

        test_klines = klines[offset:end_idx]
        result = run_backtest(test_klines, leverage)
        results.append(result)

        if result["total_return"] > 0:
            profitable += 1

        status = "PASS" if result["total_return"] > 0 else "FAIL"
        print(f"  測試 {i+1:2d}: 報酬 {result['total_return']:+6.2f}%, 回撤 {result['max_drawdown']:5.2f}% [{status}]")

    robustness = profitable / len(results) * 100

    print(f"\n  穩健性: {profitable}/{len(results)} ({robustness:.1f}%)")

    passed = robustness >= 60
    if robustness >= 80:
        print(f"  ROBUST: 穩健性 >= 80%")
    elif passed:
        print(f"  MODERATE: 穩健性 60-80%")
    else:
        print(f"  WEAK: 穩健性 < 60%")

    return {
        "robustness": robustness,
        "profitable": profitable,
        "total_tests": len(results),
        "passed": passed,
        "results": results,
    }


def out_of_sample_test(klines: list[Kline], leverage: int = 7):
    """樣本外測試 (Out-of-Sample Test)."""
    print(f"\n{'=' * 60}")
    print(f"  樣本外測試 (Out-of-Sample Test)")
    print(f"  策略: Supertrend TREND_GRID, 槓桿: {leverage}x")
    print(f"{'=' * 60}")

    total_bars = len(klines)
    train_size = int(total_bars * 0.7)
    train_klines = klines[:train_size]
    test_klines = klines[train_size:]

    train_start = train_klines[0].open_time.strftime('%Y-%m-%d')
    train_end = train_klines[-1].open_time.strftime('%Y-%m-%d')
    test_start = test_klines[0].open_time.strftime('%Y-%m-%d')
    test_end = test_klines[-1].open_time.strftime('%Y-%m-%d')

    print(f"\n  訓練集 (In-Sample): {train_start} ~ {train_end}")
    print(f"  測試集 (Out-of-Sample): {test_start} ~ {test_end}")

    train_result = run_backtest(train_klines, leverage)
    print(f"\n  [訓練集結果 - 樣本內 (IS)]")
    print(f"    報酬: {train_result['total_return']:+.2f}%")
    print(f"    回撤: {train_result['max_drawdown']:.2f}%")
    print(f"    Sharpe: {train_result['sharpe']:.2f}")
    print(f"    交易: {train_result['trades']}")

    test_result = run_backtest(test_klines, leverage)
    print(f"\n  [測試集結果 - 樣本外 (OOS)]")
    print(f"    報酬: {test_result['total_return']:+.2f}%")
    print(f"    回撤: {test_result['max_drawdown']:.2f}%")
    print(f"    Sharpe: {test_result['sharpe']:.2f}")
    print(f"    交易: {test_result['trades']}")

    if train_result['total_return'] > 0:
        oos_is_ratio = test_result['total_return'] / train_result['total_return']
    else:
        oos_is_ratio = 0 if test_result['total_return'] <= 0 else float('inf')

    if train_result['sharpe'] > 0:
        sharpe_ratio = test_result['sharpe'] / train_result['sharpe']
    else:
        sharpe_ratio = 0 if test_result['sharpe'] <= 0 else float('inf')

    print(f"\n  [OOS/IS 比率分析]")
    print(f"    報酬比率: {oos_is_ratio:.2f}")
    print(f"    Sharpe比率: {sharpe_ratio:.2f}")

    passed = sharpe_ratio >= 0.5 and test_result['total_return'] > 0
    is_excellent = oos_is_ratio >= 0.8 and test_result['total_return'] > 0

    if is_excellent:
        print(f"\n  PASS (優秀): OOS/IS >= 80%, 無過度擬合跡象")
    elif passed:
        print(f"\n  PASS (良好): OOS/IS Sharpe >= 50%")
    else:
        print(f"\n  FAIL: OOS 表現不足，可能過度擬合")

    return {
        "train_return": train_result['total_return'],
        "test_return": test_result['total_return'],
        "train_sharpe": train_result['sharpe'],
        "test_sharpe": test_result['sharpe'],
        "oos_is_ratio": oos_is_ratio,
        "sharpe_ratio": sharpe_ratio,
        "passed": passed,
        "is_overfitting": not passed,
    }


def parameter_sensitivity_test(klines: list[Kline], leverage: int = 7):
    """參數敏感度測試 (Parameter Sensitivity Analysis)."""
    print(f"\n{'=' * 60}")
    print(f"  參數敏感度測試 (Overfitting Detection)")
    print(f"  策略: Supertrend TREND_GRID")
    print(f"{'=' * 60}")

    # 可擾動的數值參數
    base_params = {
        "atr_period": OPTIMIZED_PARAMS["atr_period"],
        "atr_multiplier": OPTIMIZED_PARAMS["atr_multiplier"],
        "grid_count": OPTIMIZED_PARAMS["grid_count"],
        "grid_atr_multiplier": OPTIMIZED_PARAMS["grid_atr_multiplier"],
        "stop_loss_pct": OPTIMIZED_PARAMS["stop_loss_pct"],
        "trailing_stop_pct": OPTIMIZED_PARAMS["trailing_stop_pct"],
    }

    # 基準結果
    base_result = run_backtest(klines, leverage)
    print(f"\n  基準參數結果:")
    print(f"    報酬: {base_result['total_return']:+.2f}%")
    print(f"    Sharpe: {base_result['sharpe']:.2f}")

    perturbation_pct = 0.20
    n_tests = 20

    results = []

    print(f"\n  執行 {n_tests} 次參數擾動測試 (+/-{perturbation_pct*100:.0f}%)...")

    for i in range(n_tests):
        perturbed = dict(OPTIMIZED_PARAMS)
        for key, value in base_params.items():
            if isinstance(value, int):
                delta = max(1, int(value * perturbation_pct))
                perturbed[key] = max(1, value + random.randint(-delta, delta))
            elif isinstance(value, float):
                delta = value * perturbation_pct
                perturbed[key] = max(0.001, value + random.uniform(-delta, delta))

        result = run_backtest(klines, leverage, params=perturbed)
        results.append(result)

    returns = [r["total_return"] for r in results]
    sharpes = [r["sharpe"] for r in results]

    avg_return = sum(returns) / len(returns)
    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

    avg_sharpe = sum(sharpes) / len(sharpes)
    std_sharpe = (sum((s - avg_sharpe) ** 2 for s in sharpes) / len(sharpes)) ** 0.5

    cv_return = std_return / abs(avg_return) if avg_return != 0 else float('inf')

    positive_returns = sum(1 for r in returns if r > 0)
    robustness = positive_returns / len(returns) * 100

    print(f"\n  [擾動測試結果]")
    print(f"    平均報酬: {avg_return:+.2f}% (std={std_return:.2f}%)")
    print(f"    平均Sharpe: {avg_sharpe:.2f} (std={std_sharpe:.2f})")
    print(f"    報酬穩定度 (1-CV): {max(0, (1-cv_return)*100):.1f}%")
    print(f"    穩健性 (正報酬率): {robustness:.1f}%")
    print(f"    報酬範圍: [{min(returns):+.2f}%, {max(returns):+.2f}%]")

    is_stable = cv_return < 0.5
    is_robust = robustness >= 80
    passed = is_robust

    if is_stable and is_robust:
        print(f"\n  PASS: 參數穩定，無過度擬合")
    elif is_robust:
        print(f"\n  PASS (中等): 報酬變化大但多數仍獲利")
    else:
        print(f"\n  FAIL: 策略可能過度擬合當前參數")

    return {
        "avg_return": avg_return,
        "std_return": std_return,
        "cv_return": cv_return,
        "robustness": robustness,
        "passed": passed,
        "is_overfitting": not (is_stable and is_robust),
    }


def white_noise_benchmark(klines: list[Kline], leverage: int = 7, n_tests: int = 10):
    """白噪音基準測試 (Random Benchmark)."""
    print(f"\n{'=' * 60}")
    print(f"  白噪音基準測試 (Random Benchmark)")
    print(f"{'=' * 60}")

    random_returns = []

    print(f"\n  執行 {n_tests} 次隨機交易測試...")

    for i in range(n_tests):
        capital = 10000.0
        position = 0.0
        entry_price = 0.0

        for kline in klines:
            price = float(kline.close)

            if position == 0 and random.random() < 0.10:
                direction = random.choice([1, -1])
                position = direction * capital * leverage / price
                entry_price = price
            elif position != 0 and random.random() < 0.10:
                if position > 0:
                    pnl = (price - entry_price) / entry_price * capital * leverage
                else:
                    pnl = (entry_price - price) / entry_price * capital * leverage
                capital += pnl
                position = 0

        if position != 0:
            price = float(klines[-1].close)
            if position > 0:
                pnl = (price - entry_price) / entry_price * capital * leverage
            else:
                pnl = (entry_price - price) / entry_price * capital * leverage
            capital += pnl

        ret = (capital - 10000) / 10000 * 100
        random_returns.append(ret)
        print(f"    測試 {i+1:2d}: {ret:+.2f}%")

    avg_random = sum(random_returns) / len(random_returns)
    print(f"\n  隨機策略平均報酬: {avg_random:+.2f}%")

    return {
        "random_returns": random_returns,
        "avg_random_return": avg_random,
    }


def main():
    parser = argparse.ArgumentParser(description="Supertrend TREND_GRID Walk-Forward 驗證")
    parser.add_argument(
        "--leverage", "-l",
        type=int,
        default=7,
        help="槓桿倍數 (default: 7)",
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="執行完整驗證 (包含樣本外測試、參數敏感度、白噪音)",
    )
    parser.add_argument(
        "--data-file",
        default="data/historical/BTCUSDT_1h_730d.json",
        help="歷史數據檔案路徑",
    )
    args = parser.parse_args()

    leverage = args.leverage

    print("=" * 60)
    print("  Supertrend TREND_GRID 策略驗證")
    print(f"  槓桿: {leverage}x")
    print("=" * 60)

    # 載入數據
    klines = load_klines(args.data_file)

    # 1. 完整回測
    print(f"\n{'=' * 60}")
    print(f"  完整回測 (Supertrend TREND_GRID, {leverage}x 槓桿)")
    print(f"{'=' * 60}")

    full_result = run_backtest(klines, leverage)
    total_return = full_result["total_return"]
    annual_return = ((1 + total_return / 100) ** 0.5 - 1) * 100

    print(f"  總報酬 (2年): {total_return:.2f}%")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  最大回撤: {full_result['max_drawdown']:.2f}%")
    print(f"  Sharpe: {full_result['sharpe']:.2f}")
    print(f"  勝率: {full_result['win_rate']:.1f}%")
    print(f"  交易數: {full_result['trades']}")

    # 2. Walk-Forward 驗證
    wf_result = walk_forward_validation(klines, n_splits=9, leverage=leverage)

    # 3. Monte Carlo 測試
    mc_result = sliding_window_robustness_test(klines, n_tests=15, leverage=leverage)

    # 4. 完整驗證（OOS + 敏感度 + 白噪音）
    oos_result = None
    sensitivity_result = None

    if args.full:
        oos_result = out_of_sample_test(klines, leverage)
        sensitivity_result = parameter_sensitivity_test(klines, leverage)
        white_noise_benchmark(klines, leverage)

    # 總結
    print(f"\n{'=' * 60}")
    print(f"  驗證總結 (Supertrend TREND_GRID)")
    print(f"{'=' * 60}")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  W-F 一致性: {wf_result['consistency']:.1f}% ({'PASS' if wf_result['passed'] else 'FAIL'}, 標準 >= 50%)")
    print(f"  Monte Carlo: {mc_result['robustness']:.1f}% ({'PASS' if mc_result['passed'] else 'FAIL'}, 標準 >= 60%)")

    if oos_result:
        print(f"  OOS/IS Sharpe: {oos_result['sharpe_ratio']:.2f} ({'PASS' if oos_result['passed'] else 'FAIL'}, 標準 >= 0.5)")
    if sensitivity_result:
        print(f"  參數穩健性: {sensitivity_result['robustness']:.1f}% ({'PASS' if sensitivity_result['passed'] else 'FAIL'}, 標準 >= 80%)")

    # 判定
    passed = wf_result['passed'] and mc_result['passed']

    overfitting_warning = False
    if oos_result and oos_result['is_overfitting']:
        overfitting_warning = True
    if sensitivity_result and sensitivity_result['is_overfitting']:
        overfitting_warning = True

    if passed and not overfitting_warning:
        print(f"\n  PASS - 驗證通過，可用於實戰")
    elif passed:
        print(f"\n  WARNING - 驗證通過但有過度擬合風險，建議保守配置")
    else:
        print(f"\n  FAIL - 驗證未通過，建議調整參數")

    return passed


if __name__ == "__main__":
    main()
