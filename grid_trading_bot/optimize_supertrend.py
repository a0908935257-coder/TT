#!/usr/bin/env python3
"""
Supertrend 策略參數優化腳本.

使用 Optuna Bayesian 優化找到最佳參數組合。
保持與實戰一致的過濾器（hysteresis, cooldown, trailing stop）。

用法:
    python optimize_supertrend.py --trials 100
    python optimize_supertrend.py --trials 200 --timeout 3600
"""

import argparse
import json
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
from src.backtest.strategy.supertrend import (
    SupertrendBacktestStrategy,
    SupertrendStrategyConfig,
    SupertrendMode,
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


def create_objective(klines: list[Kline], leverage: int = 10):
    """創建 Optuna 目標函數."""

    def objective(trial: optuna.Trial) -> float:
        # 優化參數範圍
        atr_period = trial.suggest_int("atr_period", 10, 50)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.5, 5.0, step=0.5)
        grid_count = trial.suggest_int("grid_count", 5, 20)
        grid_atr_multiplier = trial.suggest_float("grid_atr_multiplier", 1.5, 5.0, step=0.5)
        take_profit_grids = trial.suggest_int("take_profit_grids", 1, 3)
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.02, 0.10, step=0.01)

        # RSI 過濾器參數
        rsi_period = trial.suggest_int("rsi_period", 7, 21)
        rsi_overbought = trial.suggest_int("rsi_overbought", 55, 75)
        rsi_oversold = trial.suggest_int("rsi_oversold", 25, 45)

        # 趨勢確認
        min_trend_bars = trial.suggest_int("min_trend_bars", 1, 5)

        # 與實戰一致的過濾器 (保持啟用，但可調整參數)
        hysteresis_pct = trial.suggest_float("hysteresis_pct", 0.001, 0.005, step=0.001)
        cooldown_bars = trial.suggest_int("cooldown_bars", 1, 4)
        trailing_stop_pct = trial.suggest_float("trailing_stop_pct", 0.02, 0.10, step=0.01)

        # 創建策略配置
        config = SupertrendStrategyConfig(
            mode=SupertrendMode.TREND_GRID,
            atr_period=atr_period,
            atr_multiplier=Decimal(str(atr_multiplier)),
            grid_count=grid_count,
            grid_atr_multiplier=Decimal(str(grid_atr_multiplier)),
            take_profit_grids=take_profit_grids,
            stop_loss_pct=Decimal(str(stop_loss_pct)),
            use_rsi_filter=True,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            min_trend_bars=min_trend_bars,
            # 與實戰一致的過濾器
            use_hysteresis=True,
            hysteresis_pct=Decimal(str(hysteresis_pct)),
            use_signal_cooldown=True,
            cooldown_bars=cooldown_bars,
            use_trailing_stop=True,
            trailing_stop_pct=Decimal(str(trailing_stop_pct)),
        )

        strategy = SupertrendBacktestStrategy(config)

        # 回測配置
        bt_config = BacktestConfig(
            initial_capital=Decimal("10000"),
            leverage=leverage,
            fee_rate=Decimal("0.0004"),
            slippage_pct=Decimal("0.0001"),
        )

        engine = BacktestEngine(bt_config)
        result = engine.run(klines, strategy)

        # 多目標優化: Sharpe * sqrt(交易次數權重) - 回撤懲罰
        sharpe = float(result.sharpe_ratio)
        total_return = float(result.total_profit_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)
        trades = result.total_trades

        # 過濾低質量結果
        if trades < 50:
            return -100  # 交易太少
        if max_dd > 20:
            return -100  # 回撤太大
        if win_rate < 50:
            return -100  # 勝率太低

        # 目標函數: 最大化風險調整後報酬
        # Sharpe * (1 + log(trades)/10) - 回撤懲罰
        import math
        trade_bonus = 1 + math.log(trades) / 10
        dd_penalty = max_dd * 0.5

        score = sharpe * trade_bonus - dd_penalty

        # 記錄中間結果
        trial.set_user_attr("total_return", total_return)
        trial.set_user_attr("max_drawdown", max_dd)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("trades", trades)

        return score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Supertrend 參數優化")
    parser.add_argument(
        "--data-file",
        default="data/historical/BTCUSDT_1h_730d.json",
        help="歷史數據檔案路徑"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=100,
        help="優化試驗次數 (default: 100)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="超時秒數 (default: 無限制)"
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=10,
        help="槓桿倍數 (default: 10)"
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
        study_name="supertrend_optimization",
    )

    # 執行優化
    print(f"\n開始優化 ({args.trials} 次試驗)...")
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
    print(f"報酬: {best.user_attrs.get('total_return', 0):.2f}%")
    print(f"回撤: {best.user_attrs.get('max_drawdown', 0):.2f}%")
    print(f"勝率: {best.user_attrs.get('win_rate', 0):.1f}%")
    print(f"交易: {best.user_attrs.get('trades', 0)}")

    print("\n最佳參數:")
    print("-" * 40)
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    # 使用最佳參數執行完整回測
    print("\n使用最佳參數執行完整回測...")
    best_config = SupertrendStrategyConfig(
        mode=SupertrendMode.TREND_GRID,
        atr_period=best.params["atr_period"],
        atr_multiplier=Decimal(str(best.params["atr_multiplier"])),
        grid_count=best.params["grid_count"],
        grid_atr_multiplier=Decimal(str(best.params["grid_atr_multiplier"])),
        take_profit_grids=best.params["take_profit_grids"],
        stop_loss_pct=Decimal(str(best.params["stop_loss_pct"])),
        use_rsi_filter=True,
        rsi_period=best.params["rsi_period"],
        rsi_overbought=best.params["rsi_overbought"],
        rsi_oversold=best.params["rsi_oversold"],
        min_trend_bars=best.params["min_trend_bars"],
        use_hysteresis=True,
        hysteresis_pct=Decimal(str(best.params["hysteresis_pct"])),
        use_signal_cooldown=True,
        cooldown_bars=best.params["cooldown_bars"],
        use_trailing_stop=True,
        trailing_stop_pct=Decimal(str(best.params["trailing_stop_pct"])),
    )

    best_strategy = SupertrendBacktestStrategy(best_config)
    bt_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        leverage=args.leverage,
        fee_rate=Decimal("0.0004"),
        slippage_pct=Decimal("0.0001"),
    )

    engine = BacktestEngine(bt_config)
    final_result = engine.run(klines, best_strategy)

    # 計算年化報酬 (2年數據)
    total_return = float(final_result.total_profit_pct)
    annual_return = ((1 + total_return/100) ** 0.5 - 1) * 100

    print("\n" + "=" * 60)
    print("  最佳參數回測結果")
    print("=" * 60)
    print(f"  總報酬: {total_return:.2f}%")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  Sharpe Ratio: {float(final_result.sharpe_ratio):.2f}")
    print(f"  最大回撤: {float(final_result.max_drawdown_pct):.2f}%")
    print(f"  勝率: {float(final_result.win_rate):.1f}%")
    print(f"  總交易數: {final_result.total_trades}")

    # 儲存結果
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"supertrend_optimization_{timestamp}.json"

    results = {
        "optimization_date": datetime.now().isoformat(),
        "data_file": args.data_file,
        "trials": args.trials,
        "leverage": args.leverage,
        "best_score": best.value,
        "best_params": best.params,
        "final_result": {
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "sharpe_ratio": float(final_result.sharpe_ratio),
            "max_drawdown_pct": float(final_result.max_drawdown_pct),
            "win_rate": float(final_result.win_rate),
            "total_trades": final_result.total_trades,
        },
        "top_10_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in sorted(study.trials, key=lambda x: x.value if x.value else -999, reverse=True)[:10]
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {output_file}")

    # 輸出可直接使用的配置
    print("\n" + "=" * 60)
    print("  建議配置 (可直接複製到實戰)")
    print("=" * 60)
    print(f"""
SupertrendConfig(
    atr_period={best.params["atr_period"]},
    atr_multiplier=Decimal("{best.params["atr_multiplier"]}"),
    grid_count={best.params["grid_count"]},
    grid_atr_multiplier=Decimal("{best.params["grid_atr_multiplier"]}"),
    take_profit_grids={best.params["take_profit_grids"]},
    stop_loss_pct=Decimal("{best.params["stop_loss_pct"]}"),
    rsi_period={best.params["rsi_period"]},
    rsi_overbought={best.params["rsi_overbought"]},
    rsi_oversold={best.params["rsi_oversold"]},
    min_trend_bars={best.params["min_trend_bars"]},
    use_hysteresis=True,
    hysteresis_pct=Decimal("{best.params["hysteresis_pct"]}"),
    use_signal_cooldown=True,
    cooldown_bars={best.params["cooldown_bars"]},
    use_trailing_stop=True,
    trailing_stop_pct=Decimal("{best.params["trailing_stop_pct"]}"),
)
""")


if __name__ == "__main__":
    main()
