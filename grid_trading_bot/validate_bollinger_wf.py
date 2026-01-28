#!/usr/bin/env python3
"""
Bollinger 策略 Walk-Forward 驗證腳本.

驗證優化後參數的樣本外表現和穩健性。

用法:
    # 驗證 BB_TREND_GRID (預設)
    python validate_bollinger_wf.py

    # 驗證 BB_NEUTRAL_GRID
    python validate_bollinger_wf.py --mode neutral
"""

import argparse
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


def run_backtest(klines: list[Kline], leverage: int = 19, mode: str = "trend") -> dict:
    """執行回測並返回結果."""
    if mode == "neutral":
        # BB_NEUTRAL_GRID 模式 - 優化後參數 (2026-01-28)
        # IS: 103.39%, OOS: 30.75%, OOS超額報酬: +44.90%
        # Sharpe: IS 12.04, OOS 11.53, 回撤: IS 2.12%, OOS 2.73%
        config = BollingerStrategyConfig(
            mode=BollingerMode.BB_NEUTRAL_GRID,
            bb_period=31,
            bb_std=Decimal("2.0"),
            grid_count=14,
            take_profit_grids=1,
            stop_loss_pct=Decimal("0.002"),  # 0.2% tight SL
            # ATR dynamic range
            use_atr_range=True,
            atr_period=29,
            atr_multiplier=Decimal("9.5"),
            fallback_range_pct=Decimal("0.04"),
            # Protective features
            use_hysteresis=True,
            hysteresis_pct=Decimal("0.0025"),
            use_signal_cooldown=False,
            cooldown_bars=1,
        )
    else:
        # BB_TREND_GRID 模式 (驗證通過)
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


def walk_forward_validation(klines: list[Kline], n_splits: int = 9, leverage: int = 19, mode: str = "trend"):
    """
    Walk-Forward 驗證.

    將數據分成多個時間段，每段獨立回測，檢驗參數穩健性。
    """
    mode_name = "BB_NEUTRAL_GRID" if mode == "neutral" else "BB_TREND_GRID"
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 驗證 ({mode_name})")
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

        result = run_backtest(period_klines, leverage, mode)
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


def monte_carlo_test(klines: list[Kline], n_tests: int = 15, leverage: int = 19, mode: str = "trend"):
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
        result = run_backtest(test_klines, leverage, mode)
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


def out_of_sample_test(klines: list[Kline], leverage: int = 19, mode: str = "trend"):
    """
    樣本外測試 (Out-of-Sample Test).

    將數據分為訓練集(70%)和測試集(30%)，比較樣本內外表現。
    用於檢測策略是否對歷史數據過度擬合。
    """
    mode_name = "BB_NEUTRAL_GRID" if mode == "neutral" else "BB_TREND_GRID"
    print(f"\n{'=' * 60}")
    print(f"  樣本外測試 (Out-of-Sample Test)")
    print(f"  策略: {mode_name}, 槓桿: {leverage}x")
    print(f"{'=' * 60}")

    total_bars = len(klines)
    train_ratio = 0.7  # 70% 訓練, 30% 測試

    train_size = int(total_bars * train_ratio)
    train_klines = klines[:train_size]
    test_klines = klines[train_size:]

    train_start = train_klines[0].open_time.strftime('%Y-%m-%d')
    train_end = train_klines[-1].open_time.strftime('%Y-%m-%d')
    test_start = test_klines[0].open_time.strftime('%Y-%m-%d')
    test_end = test_klines[-1].open_time.strftime('%Y-%m-%d')

    print(f"\n  訓練集 (In-Sample): {train_start} ~ {train_end}")
    print(f"  測試集 (Out-of-Sample): {test_start} ~ {test_end}")

    # 訓練集回測
    train_result = run_backtest(train_klines, leverage, mode)
    print(f"\n  [訓練集結果 - 樣本內 (IS)]")
    print(f"    報酬: {train_result['total_return']:+.2f}%")
    print(f"    回撤: {train_result['max_drawdown']:.2f}%")
    print(f"    Sharpe: {train_result['sharpe']:.2f}")
    print(f"    交易: {train_result['trades']}")

    # 測試集回測
    test_result = run_backtest(test_klines, leverage, mode)
    print(f"\n  [測試集結果 - 樣本外 (OOS)]")
    print(f"    報酬: {test_result['total_return']:+.2f}%")
    print(f"    回撤: {test_result['max_drawdown']:.2f}%")
    print(f"    Sharpe: {test_result['sharpe']:.2f}")
    print(f"    交易: {test_result['trades']}")

    # 計算 OOS/IS 比率
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

    # 判定標準
    # OOS/IS >= 0.5 表示良好（樣本外表現至少是樣本內的50%）
    # OOS/IS >= 0.8 表示優秀（樣本外表現接近樣本內）
    is_good = oos_is_ratio >= 0.5 and test_result['total_return'] > 0
    is_excellent = oos_is_ratio >= 0.8 and test_result['total_return'] > 0

    if is_excellent:
        print(f"\n  ✅ 優秀: OOS/IS >= 80%, 無過度擬合跡象")
    elif is_good:
        print(f"\n  ⚠️ 良好: OOS/IS >= 50%, 輕微過度擬合")
    else:
        print(f"\n  ❌ 警告: OOS/IS < 50%, 可能過度擬合")

    return {
        "train_return": train_result['total_return'],
        "test_return": test_result['total_return'],
        "train_sharpe": train_result['sharpe'],
        "test_sharpe": test_result['sharpe'],
        "oos_is_ratio": oos_is_ratio,
        "sharpe_ratio": sharpe_ratio,
        "is_overfitting": not is_good,
    }


def parameter_sensitivity_test(klines: list[Kline], leverage: int = 19, mode: str = "trend"):
    """
    參數敏感度測試 (Parameter Sensitivity Analysis).

    對關鍵參數進行擾動測試，檢查策略對參數變化的敏感程度。
    高敏感度可能表示過度擬合特定參數值。
    """
    import random

    mode_name = "BB_NEUTRAL_GRID" if mode == "neutral" else "BB_TREND_GRID"
    print(f"\n{'=' * 60}")
    print(f"  參數敏感度測試 (Overfitting Detection)")
    print(f"  策略: {mode_name}")
    print(f"{'=' * 60}")

    # 基準參數 (當前策略參數 - 優化後)
    if mode == "neutral":
        base_params = {
            "bb_period": 31,
            "grid_count": 14,
            "stop_loss_pct": 0.002,
            "atr_period": 29,
            "atr_multiplier": 9.5,
        }
    else:
        base_params = {
            "bb_period": 12,
            "grid_count": 6,
            "stop_loss_pct": 0.025,
            "grid_range_pct": 0.02,
        }

    # 基準結果
    base_result = run_backtest(klines, leverage, mode)
    print(f"\n  基準參數結果:")
    print(f"    報酬: {base_result['total_return']:+.2f}%")
    print(f"    Sharpe: {base_result['sharpe']:.2f}")

    # 參數擾動範圍 (±20%)
    perturbation_pct = 0.20
    n_tests = 20

    results = []

    print(f"\n  執行 {n_tests} 次參數擾動測試 (±{perturbation_pct*100:.0f}%)...")

    for i in range(n_tests):
        # 隨機擾動參數
        perturbed = base_params.copy()
        for key, value in base_params.items():
            if isinstance(value, int):
                delta = max(1, int(value * perturbation_pct))
                perturbed[key] = value + random.randint(-delta, delta)
                perturbed[key] = max(1, perturbed[key])  # 確保正數
            elif isinstance(value, float):
                delta = value * perturbation_pct
                perturbed[key] = value + random.uniform(-delta, delta)
                perturbed[key] = max(0.001, perturbed[key])  # 確保正數

        # 用擾動參數建立策略配置
        if mode == "neutral":
            config = BollingerStrategyConfig(
                mode=BollingerMode.BB_NEUTRAL_GRID,
                bb_period=perturbed["bb_period"],
                bb_std=Decimal("2.0"),
                grid_count=perturbed["grid_count"],
                take_profit_grids=1,
                stop_loss_pct=Decimal(str(perturbed["stop_loss_pct"])),
                use_atr_range=True,
                atr_period=perturbed["atr_period"],
                atr_multiplier=Decimal(str(perturbed["atr_multiplier"])),
                fallback_range_pct=Decimal("0.04"),
                use_hysteresis=True,
                hysteresis_pct=Decimal("0.0025"),
                use_signal_cooldown=False,
                cooldown_bars=1,
            )
        else:
            config = BollingerStrategyConfig(
                mode=BollingerMode.BB_TREND_GRID,
                bb_period=perturbed["bb_period"],
                bb_std=Decimal("2.0"),
                grid_count=perturbed["grid_count"],
                grid_range_pct=Decimal(str(perturbed["grid_range_pct"])),
                take_profit_grids=2,
                stop_loss_pct=Decimal(str(perturbed["stop_loss_pct"])),
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

        results.append({
            "return": float(result.total_profit_pct),
            "sharpe": float(result.sharpe_ratio),
        })

    # 分析敏感度
    returns = [r["return"] for r in results]
    sharpes = [r["sharpe"] for r in results]

    avg_return = sum(returns) / len(returns)
    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

    avg_sharpe = sum(sharpes) / len(sharpes)
    std_sharpe = (sum((s - avg_sharpe) ** 2 for s in sharpes) / len(sharpes)) ** 0.5

    # 穩定性指標：標準差/平均值 (CV: Coefficient of Variation)
    cv_return = std_return / abs(avg_return) if avg_return != 0 else float('inf')
    cv_sharpe = std_sharpe / abs(avg_sharpe) if avg_sharpe != 0 else float('inf')

    # 勝率 (正報酬比例)
    positive_returns = sum(1 for r in returns if r > 0)
    robustness = positive_returns / len(returns) * 100

    print(f"\n  [擾動測試結果]")
    print(f"    平均報酬: {avg_return:+.2f}% (σ={std_return:.2f}%)")
    print(f"    平均Sharpe: {avg_sharpe:.2f} (σ={std_sharpe:.2f})")
    print(f"    報酬穩定度 (1-CV): {max(0, (1-cv_return)*100):.1f}%")
    print(f"    穩健性 (正報酬率): {robustness:.1f}%")
    print(f"    報酬範圍: [{min(returns):+.2f}%, {max(returns):+.2f}%]")

    # 判定標準
    # CV < 0.5 表示穩定 (變異係數小於50%)
    # 穩健性 >= 80% 表示大多數擾動仍獲利
    is_stable = cv_return < 0.5
    is_robust = robustness >= 80

    if is_stable and is_robust:
        print(f"\n  ✅ 參數穩定: 策略對參數變化不敏感，無過度擬合")
    elif is_robust:
        print(f"\n  ⚠️ 中等敏感: 報酬變化大但多數仍獲利")
    else:
        print(f"\n  ❌ 高敏感度: 策略可能過度擬合當前參數")

    return {
        "avg_return": avg_return,
        "std_return": std_return,
        "cv_return": cv_return,
        "robustness": robustness,
        "is_overfitting": not (is_stable and is_robust),
    }


def white_noise_benchmark(klines: list[Kline], leverage: int = 19, n_tests: int = 10):
    """
    白噪音基準測試 (Random Benchmark).

    用隨機進出場策略作為基準，檢查策略是否顯著優於隨機。
    如果策略表現與隨機接近，可能表示過度擬合。
    """
    import random

    print(f"\n{'=' * 60}")
    print(f"  白噪音基準測試 (Random Benchmark)")
    print(f"{'=' * 60}")

    random_returns = []

    print(f"\n  執行 {n_tests} 次隨機交易測試...")

    for i in range(n_tests):
        # 模擬隨機交易
        capital = 10000.0
        position = 0.0
        entry_price = 0.0

        for j, kline in enumerate(klines):
            price = float(kline.close)

            # 隨機進出場 (10% 機率)
            if position == 0 and random.random() < 0.10:
                # 隨機開倉 (多或空)
                direction = random.choice([1, -1])
                position = direction * capital * leverage / price
                entry_price = price
            elif position != 0 and random.random() < 0.10:
                # 平倉
                if position > 0:
                    pnl = (price - entry_price) / entry_price * capital * leverage
                else:
                    pnl = (entry_price - price) / entry_price * capital * leverage
                capital += pnl
                position = 0

        # 強制平倉
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
    parser = argparse.ArgumentParser(description="Bollinger 策略 Walk-Forward 驗證")
    parser.add_argument(
        "--mode", "-m",
        choices=["trend", "neutral"],
        default="trend",
        help="策略模式: trend (BB_TREND_GRID), neutral (BB_NEUTRAL_GRID)"
    )
    parser.add_argument(
        "--leverage", "-l",
        type=int,
        default=18,
        help="槓桿倍數 (default: 18)"
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="執行完整驗證 (包含樣本外測試和過度擬合驗證)"
    )
    args = parser.parse_args()

    mode = args.mode
    leverage = args.leverage
    mode_name = "BB_NEUTRAL_GRID" if mode == "neutral" else "BB_TREND_GRID"

    # 載入數據
    klines = load_klines("data/historical/BTCUSDT_1h_730d.json")

    # 完整回測
    print(f"\n{'=' * 60}")
    print(f"  完整回測 ({mode_name}, {leverage}x 槓桿)")
    print(f"{'=' * 60}")

    full_result = run_backtest(klines, leverage, mode)
    total_return = full_result["total_return"]
    annual_return = ((1 + total_return/100) ** 0.5 - 1) * 100

    print(f"  總報酬 (2年): {total_return:.2f}%")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  最大回撤: {full_result['max_drawdown']:.2f}%")
    print(f"  Sharpe: {full_result['sharpe']:.2f}")
    print(f"  勝率: {full_result['win_rate']:.1f}%")
    print(f"  交易數: {full_result['trades']}")

    # Walk-Forward 驗證
    wf_result = walk_forward_validation(klines, n_splits=9, leverage=leverage, mode=mode)

    # Monte Carlo 測試
    mc_result = monte_carlo_test(klines, n_tests=15, leverage=leverage, mode=mode)

    # 樣本外測試和過度擬合驗證 (使用 --full 參數)
    oos_result = None
    sensitivity_result = None

    if args.full:
        oos_result = out_of_sample_test(klines, leverage, mode)
        sensitivity_result = parameter_sensitivity_test(klines, leverage, mode)
        white_noise_benchmark(klines, leverage)

    # 總結
    print(f"\n{'=' * 60}")
    print(f"  驗證總結 ({mode_name})")
    print(f"{'=' * 60}")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  W-F 一致性: {wf_result['consistency']:.1f}%")
    print(f"  Monte Carlo: {mc_result['robustness']:.1f}%")

    if oos_result:
        print(f"  OOS/IS 比率: {oos_result['oos_is_ratio']:.2f}")
    if sensitivity_result:
        print(f"  參數穩健性: {sensitivity_result['robustness']:.1f}%")

    passed = wf_result['consistency'] >= 50 and mc_result['robustness'] >= 60

    overfitting_warning = False
    if oos_result and oos_result['is_overfitting']:
        overfitting_warning = True
    if sensitivity_result and sensitivity_result['is_overfitting']:
        overfitting_warning = True

    if passed and not overfitting_warning:
        print(f"\n  ✅ 驗證通過 - 可用於實戰")
    elif passed:
        print(f"\n  ⚠️ 驗證通過但有過度擬合風險 - 建議保守配置")
    else:
        print(f"\n  ❌ 驗證未通過 - 建議調整參數")

    return passed


if __name__ == "__main__":
    main()
