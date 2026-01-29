#!/usr/bin/env python3
"""
RSI-Grid v2 策略參數優化腳本.

改進:
- 全 categorical 參數（抗過擬合）
- 內建 70/30 train/test split
- OOS/IS consistency 檢驗
- Overfit penalty

用法:
    python optimize_rsi_grid.py --trials 200
    python optimize_rsi_grid.py --trials 200 --timeout 3600
"""

import argparse
import json
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: 需要安裝 optuna: pip install optuna")

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.strategy.rsi_grid import (
    RSIGridBacktestStrategy,
    RSIGridStrategyConfig,
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


def run_backtest(klines: list[Kline], config: RSIGridStrategyConfig, leverage: int):
    """Run a single backtest and return the result."""
    strategy = RSIGridBacktestStrategy(config)
    bt_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        leverage=leverage,
        fee_rate=Decimal("0.0004"),
        slippage_pct=Decimal("0.0001"),
    )
    engine = BacktestEngine(bt_config)
    return engine.run(klines, strategy)


def create_objective(klines: list[Kline], leverage: int = 2):
    """創建 Optuna 目標函數（內建 train/test split）."""

    # 70/30 split
    split_idx = int(len(klines) * 0.7)
    train_klines = klines[:split_idx]
    test_klines = klines[split_idx:]

    print(f"Train: {len(train_klines)} bars, Test: {len(test_klines)} bars")

    def objective(trial: optuna.Trial) -> float:
        # 全 categorical 參數（11 個離散值）
        rsi_period = trial.suggest_categorical("rsi_period", [7, 10, 14, 21])
        rsi_block_threshold = trial.suggest_categorical("rsi_block_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
        atr_period = trial.suggest_categorical("atr_period", [10, 14, 21])
        grid_count = trial.suggest_categorical("grid_count", [10, 12, 15, 18, 20])
        atr_multiplier = trial.suggest_categorical("atr_multiplier", [1.5, 2.0, 2.5, 3.0])
        stop_loss_atr_mult = trial.suggest_categorical("stop_loss_atr_mult", [1.0, 1.5, 2.0])
        take_profit_grids = trial.suggest_categorical("take_profit_grids", [1, 2])
        max_hold_bars = trial.suggest_categorical("max_hold_bars", [12, 24, 48])
        use_trailing_stop = trial.suggest_categorical("use_trailing_stop", [True, False])
        trailing_activate_pct = trial.suggest_categorical("trailing_activate_pct", [0.005, 0.01, 0.015])
        trailing_distance_pct = trial.suggest_categorical("trailing_distance_pct", [0.003, 0.005, 0.008])

        config = RSIGridStrategyConfig(
            rsi_period=rsi_period,
            rsi_block_threshold=rsi_block_threshold,
            atr_period=atr_period,
            grid_count=grid_count,
            atr_multiplier=Decimal(str(atr_multiplier)),
            stop_loss_atr_mult=Decimal(str(stop_loss_atr_mult)),
            take_profit_grids=take_profit_grids,
            max_hold_bars=max_hold_bars,
            use_trailing_stop=use_trailing_stop,
            trailing_activate_pct=trailing_activate_pct,
            trailing_distance_pct=trailing_distance_pct,
        )

        # Train backtest
        train_result = run_backtest(train_klines, config, leverage)
        train_trades = train_result.total_trades
        train_sharpe = float(train_result.sharpe_ratio)

        # 品質過濾: train 交易太少 → prune
        if train_trades < 100:
            raise optuna.TrialPruned(f"train_trades={train_trades} < 100")

        # Test backtest
        test_result = run_backtest(test_klines, config, leverage)
        test_trades = test_result.total_trades
        test_sharpe = float(test_result.sharpe_ratio)
        test_return = float(test_result.total_profit_pct)
        test_dd = float(test_result.max_drawdown_pct)

        # OOS/IS Sharpe ratio
        if train_sharpe > 0:
            oos_is_ratio = test_sharpe / train_sharpe
        else:
            oos_is_ratio = 0.0

        # Consistency bonus: OOS/IS ratio close to 1.0 is good
        consistency_bonus = min(oos_is_ratio, 1.0) * 2.0 if oos_is_ratio >= 0.5 else 0.0

        # Overfit penalty: large gap between IS and OOS
        overfit_penalty = 0.0
        if oos_is_ratio < 0.5:
            overfit_penalty = 3.0  # heavy penalty

        # Score = weighted combination
        score = (
            test_sharpe * 0.4
            + train_sharpe * 0.2
            + consistency_bonus
            - overfit_penalty
        )

        # Record attrs
        trial.set_user_attr("train_trades", train_trades)
        trial.set_user_attr("train_sharpe", train_sharpe)
        trial.set_user_attr("train_return", float(train_result.total_profit_pct))
        trial.set_user_attr("test_trades", test_trades)
        trial.set_user_attr("test_sharpe", test_sharpe)
        trial.set_user_attr("test_return", test_return)
        trial.set_user_attr("test_dd", test_dd)
        trial.set_user_attr("oos_is_ratio", oos_is_ratio)

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="RSI-Grid v2 參數優化")
    parser.add_argument(
        "--data-file",
        default="data/historical/BTCUSDT_1h_730d.json",
        help="歷史數據檔案路徑"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=200,
        help="優化試驗次數 (default: 200)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="超時秒數"
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=2,
        help="槓桿倍數 (default: 2)"
    )
    parser.add_argument(
        "--output",
        default="optimization_results",
        help="輸出目錄"
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("錯誤: 需要安裝 optuna")
        print("  pip install optuna")
        return

    # 載入數據
    klines = load_klines(args.data_file)

    # 創建 Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="rsi_grid_v2_optimization",
    )

    # 執行優化
    print(f"\n開始優化 ({args.trials} 次試驗, 70/30 train/test split)...")
    print("=" * 60)

    objective = create_objective(klines, args.leverage)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # 顯示結果
    print("\n" + "=" * 60)
    print("  優化完成")
    print("=" * 60)

    best = study.best_trial
    print(f"\n最佳得分: {best.value:.4f}")
    print(f"\nTrain:")
    print(f"  Sharpe: {best.user_attrs.get('train_sharpe', 0):.2f}")
    print(f"  Return: {best.user_attrs.get('train_return', 0):.2f}%")
    print(f"  Trades: {best.user_attrs.get('train_trades', 0)}")
    print(f"\nTest (OOS):")
    print(f"  Sharpe: {best.user_attrs.get('test_sharpe', 0):.2f}")
    print(f"  Return: {best.user_attrs.get('test_return', 0):.2f}%")
    print(f"  Drawdown: {best.user_attrs.get('test_dd', 0):.2f}%")
    print(f"  Trades: {best.user_attrs.get('test_trades', 0)}")
    print(f"\nOOS/IS Sharpe Ratio: {best.user_attrs.get('oos_is_ratio', 0):.2f}")

    print("\n最佳參數:")
    print("-" * 40)
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    # Full backtest with best params
    print("\n使用最佳參數執行完整回測...")
    best_config = RSIGridStrategyConfig(
        rsi_period=best.params["rsi_period"],
        rsi_block_threshold=best.params["rsi_block_threshold"],
        atr_period=best.params["atr_period"],
        grid_count=best.params["grid_count"],
        atr_multiplier=Decimal(str(best.params["atr_multiplier"])),
        stop_loss_atr_mult=Decimal(str(best.params["stop_loss_atr_mult"])),
        take_profit_grids=best.params["take_profit_grids"],
        max_hold_bars=best.params["max_hold_bars"],
        use_trailing_stop=best.params["use_trailing_stop"],
        trailing_activate_pct=best.params["trailing_activate_pct"],
        trailing_distance_pct=best.params["trailing_distance_pct"],
    )

    full_result = run_backtest(klines, best_config, args.leverage)

    total_return = float(full_result.total_profit_pct)
    days = len(klines) / 24  # 1h bars
    annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0
    trades_per_day = full_result.total_trades / days if days > 0 else 0

    print("\n" + "=" * 60)
    print("  最佳參數完整回測結果")
    print("=" * 60)
    print(f"  總報酬: {total_return:.2f}%")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  Sharpe Ratio: {float(full_result.sharpe_ratio):.2f}")
    print(f"  最大回撤: {float(full_result.max_drawdown_pct):.2f}%")
    print(f"  勝率: {float(full_result.win_rate):.1f}%")
    print(f"  總交易數: {full_result.total_trades}")
    print(f"  每日交易: {trades_per_day:.1f}")

    # 儲存結果
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"rsi_grid_v2_optimization_{timestamp}.json"

    results = {
        "optimization_date": datetime.now().isoformat(),
        "data_file": args.data_file,
        "trials": args.trials,
        "leverage": args.leverage,
        "train_test_split": 0.7,
        "best_score": best.value,
        "best_params": best.params,
        "best_trial_attrs": best.user_attrs,
        "full_result": {
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "sharpe_ratio": float(full_result.sharpe_ratio),
            "max_drawdown_pct": float(full_result.max_drawdown_pct),
            "win_rate": float(full_result.win_rate),
            "total_trades": full_result.total_trades,
            "trades_per_day": trades_per_day,
        },
        "top_10_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda x: x.value,
                reverse=True,
            )[:10]
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {output_file}")

    # 輸出可直接使用的配置
    print("\n" + "=" * 60)
    print("  建議配置 (settings.yaml)")
    print("=" * 60)
    for key, value in best.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
