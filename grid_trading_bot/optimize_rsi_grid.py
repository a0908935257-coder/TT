#!/usr/bin/env python3
"""
RSI-Grid 策略參數優化腳本.

使用 Optuna Bayesian 優化找到最佳參數組合。
基於 1h 時間框架（比 15m 表現更好）。

用法:
    python optimize_rsi_grid.py --trials 100
    python optimize_rsi_grid.py --trials 200 --timeout 3600
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


def create_objective(klines: list[Kline], leverage: int = 2):
    """創建 Optuna 目標函數."""

    def objective(trial: optuna.Trial) -> float:
        # RSI 參數
        rsi_period = trial.suggest_int("rsi_period", 7, 28)
        oversold_level = trial.suggest_int("oversold_level", 20, 40)
        overbought_level = trial.suggest_int("overbought_level", 60, 80)

        # Grid 參數
        grid_count = trial.suggest_int("grid_count", 5, 20)
        atr_period = trial.suggest_int("atr_period", 10, 30)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.5, 5.0, step=0.5)

        # Trend Filter
        trend_sma_period = trial.suggest_int("trend_sma_period", 10, 50)
        use_trend_filter = trial.suggest_categorical("use_trend_filter", [True, False])

        # Risk Management
        stop_loss_atr_mult = trial.suggest_float("stop_loss_atr_mult", 1.0, 3.0, step=0.5)
        max_stop_loss_pct = trial.suggest_float("max_stop_loss_pct", 0.02, 0.05, step=0.01)
        take_profit_grids = trial.suggest_int("take_profit_grids", 1, 3)

        # Protective features
        use_hysteresis = trial.suggest_categorical("use_hysteresis", [True, False])
        hysteresis_pct = trial.suggest_float("hysteresis_pct", 0.001, 0.005, step=0.001)
        use_signal_cooldown = trial.suggest_categorical("use_signal_cooldown", [True, False])
        cooldown_bars = trial.suggest_int("cooldown_bars", 1, 4)

        # 創建策略配置
        config = RSIGridStrategyConfig(
            rsi_period=rsi_period,
            oversold_level=oversold_level,
            overbought_level=overbought_level,
            grid_count=grid_count,
            atr_period=atr_period,
            atr_multiplier=Decimal(str(atr_multiplier)),
            trend_sma_period=trend_sma_period,
            use_trend_filter=use_trend_filter,
            stop_loss_atr_mult=Decimal(str(stop_loss_atr_mult)),
            max_stop_loss_pct=Decimal(str(max_stop_loss_pct)),
            take_profit_grids=take_profit_grids,
            use_hysteresis=use_hysteresis,
            hysteresis_pct=Decimal(str(hysteresis_pct)),
            use_signal_cooldown=use_signal_cooldown,
            cooldown_bars=cooldown_bars,
        )

        strategy = RSIGridBacktestStrategy(config)

        # 回測配置
        bt_config = BacktestConfig(
            initial_capital=Decimal("10000"),
            leverage=leverage,
            fee_rate=Decimal("0.0004"),
            slippage_pct=Decimal("0.0001"),
        )

        engine = BacktestEngine(bt_config)
        result = engine.run(klines, strategy)

        # 計算指標
        sharpe = float(result.sharpe_ratio)
        total_return = float(result.total_profit_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)
        trades = result.total_trades

        # 過濾低質量結果
        if trades < 30:
            return -100  # 交易太少
        if max_dd > 15:
            return -100  # 回撤太大
        if win_rate < 50:
            return -100  # 勝率太低

        # 目標函數: 最大化風險調整後報酬
        import math
        trade_bonus = 1 + math.log(max(trades, 1)) / 10
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
    parser = argparse.ArgumentParser(description="RSI-Grid 參數優化")
    parser.add_argument(
        "--data-file",
        default="data/historical/BTCUSDT_1h_730d.json",
        help="歷史數據檔案路徑 (建議使用 1h)"
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
        study_name="rsi_grid_optimization",
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
    best_config = RSIGridStrategyConfig(
        rsi_period=best.params["rsi_period"],
        oversold_level=best.params["oversold_level"],
        overbought_level=best.params["overbought_level"],
        grid_count=best.params["grid_count"],
        atr_period=best.params["atr_period"],
        atr_multiplier=Decimal(str(best.params["atr_multiplier"])),
        trend_sma_period=best.params["trend_sma_period"],
        use_trend_filter=best.params["use_trend_filter"],
        stop_loss_atr_mult=Decimal(str(best.params["stop_loss_atr_mult"])),
        max_stop_loss_pct=Decimal(str(best.params["max_stop_loss_pct"])),
        take_profit_grids=best.params["take_profit_grids"],
        use_hysteresis=best.params["use_hysteresis"],
        hysteresis_pct=Decimal(str(best.params["hysteresis_pct"])),
        use_signal_cooldown=best.params["use_signal_cooldown"],
        cooldown_bars=best.params["cooldown_bars"],
    )

    best_strategy = RSIGridBacktestStrategy(best_config)
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
    output_file = output_dir / f"rsi_grid_optimization_{timestamp}.json"

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
RSIGridConfig(
    rsi_period={best.params["rsi_period"]},
    oversold_level={best.params["oversold_level"]},
    overbought_level={best.params["overbought_level"]},
    grid_count={best.params["grid_count"]},
    atr_period={best.params["atr_period"]},
    atr_multiplier=Decimal("{best.params["atr_multiplier"]}"),
    trend_sma_period={best.params["trend_sma_period"]},
    use_trend_filter={best.params["use_trend_filter"]},
    stop_loss_atr_mult=Decimal("{best.params["stop_loss_atr_mult"]}"),
    max_stop_loss_pct=Decimal("{best.params["max_stop_loss_pct"]}"),
    take_profit_grids={best.params["take_profit_grids"]},
    use_hysteresis={best.params["use_hysteresis"]},
    hysteresis_pct=Decimal("{best.params["hysteresis_pct"]}"),
    use_signal_cooldown={best.params["use_signal_cooldown"]},
    cooldown_bars={best.params["cooldown_bars"]},
)
""")


if __name__ == "__main__":
    main()
