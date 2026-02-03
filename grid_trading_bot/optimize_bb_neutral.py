#!/usr/bin/env python3
"""
BB_NEUTRAL_GRID 策略 Walk-Forward 優化腳本 (v3).

使用 N 折 Walk-Forward 作為 Optuna 目標函數。
每個 trial 都在 9 個時間段上回測，確保參數在所有市場環境穩健。

核心優化策略 (v3 - Walk-Forward 穩健性):
1. 將數據分成 N 個等長時段
2. 每個時段獨立回測
3. 優化目標 = 一致性 × 平均 Sharpe × 交易量獎勵
4. 懲罰: 虧損時段、高回撤、低交易量

約束條件：
- 每個時段交易數 >= 10
- 每個時段回撤 <= 30%

用法:
    python optimize_bb_neutral.py --trials 300 --leverage 18
    python optimize_bb_neutral.py --trials 100 --leverage 18 --quick
"""

import argparse
import json
import math
from datetime import datetime, timezone
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


def split_into_folds(klines: list[Kline], n_folds: int = 9) -> list[list[Kline]]:
    """
    將數據分成 N 個等長時段 (用於 Walk-Forward 優化).

    Args:
        klines: 完整的 K 線數據
        n_folds: 折數 (預設 9)

    Returns:
        list of kline segments
    """
    fold_size = len(klines) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(klines)
        folds.append(klines[start:end])

    print(f"\n數據分割為 {n_folds} 折:")
    for i, fold in enumerate(folds):
        market_ret = (float(fold[-1].close) / float(fold[0].close) - 1) * 100
        print(f"  折 {i+1}: {fold[0].open_time.strftime('%Y-%m-%d')} ~ {fold[-1].open_time.strftime('%Y-%m-%d')} ({len(fold):,} 根, 市場 {market_ret:+.1f}%)")

    return folds


def run_backtest(klines: list[Kline], params: dict, leverage: int = 18) -> dict:
    """
    使用指定參數執行回測.

    Args:
        klines: K 線數據
        params: 策略參數字典
        leverage: 槓桿倍數

    Returns:
        回測結果字典
    """
    config = BollingerStrategyConfig(
        mode=BollingerMode.BB_NEUTRAL_GRID,
        bb_period=params["bb_period"],
        bb_std=Decimal("2.0"),
        grid_count=params["grid_count"],
        take_profit_grids=params["take_profit_grids"],
        stop_loss_pct=Decimal(str(params["stop_loss_pct"])),
        use_atr_range=True,
        atr_period=params["atr_period"],
        atr_multiplier=Decimal(str(params["atr_multiplier"])),
        fallback_range_pct=Decimal("0.04"),
        use_hysteresis=params["use_hysteresis"],
        hysteresis_pct=Decimal(str(params["hysteresis_pct"])),
        use_signal_cooldown=params["use_signal_cooldown"],
        cooldown_bars=params["cooldown_bars"],
    )

    strategy = BollingerBacktestStrategy(config)

    bt_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0006"),       # 0.06% (aligned with WF validation)
        slippage_pct=Decimal("0.0005"),   # 0.05% (aligned with WF validation)
    ).with_leverage(leverage)             # use_margin=True for proper futures accounting

    engine = BacktestEngine(bt_config)
    result = engine.run(klines, strategy)

    return {
        "total_return": float(result.total_profit_pct),
        "max_drawdown": float(result.max_drawdown_pct),
        "sharpe": float(result.sharpe_ratio),
        "win_rate": float(result.win_rate),
        "trades": result.total_trades,
    }


def create_wf_objective(
    folds: list[list[Kline]],
    leverage: int = 18
):
    """
    創建 Walk-Forward 目標函數 (v3: 多時段穩健性).

    每個 trial 在所有 N 折上回測，確保參數在不同市場環境都有效。

    優化目標:
    score = consistency_pct * avg_sharpe * 10   # 一致性 × 平均風險調整報酬
          + profitable_folds * 20                # 每個獲利折加 20 分
          + min_sharpe * 5                       # 最差折的 Sharpe 獎勵
          + trade_bonus                          # 交易量獎勵 (鼓勵多交易)
          - worst_dd * 0.5                       # 最差回撤懲罰

    約束條件:
    - 每折交易數 >= 10
    - 每折回撤 <= 30%
    """

    def objective(trial: optuna.Trial) -> float:
        # 參數搜索空間
        params = {
            "bb_period": trial.suggest_int("bb_period", 8, 35),
            "grid_count": trial.suggest_int("grid_count", 4, 24),
            "atr_period": trial.suggest_int("atr_period", 10, 30),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 2.0, 10.0, step=0.5),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.002, 0.02, step=0.001),
            "take_profit_grids": trial.suggest_int("take_profit_grids", 1, 5),
            "hysteresis_pct": trial.suggest_float("hysteresis_pct", 0.0005, 0.006, step=0.0005),
            "use_hysteresis": trial.suggest_categorical("use_hysteresis", [True, False]),
            "use_signal_cooldown": trial.suggest_categorical("use_signal_cooldown", [True, False]),
            "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 6),
        }

        # 在所有折上回測
        fold_results = []
        total_trades = 0
        for i, fold_klines in enumerate(folds):
            try:
                result = run_backtest(fold_klines, params, leverage)
                fold_results.append(result)
                total_trades += result["trades"]
            except Exception:
                return -1000

        n_folds = len(fold_results)

        # ========== 約束條件 ==========
        # 每折至少 10 筆交易
        min_trades = min(r["trades"] for r in fold_results)
        if min_trades < 10:
            return -500 + min_trades

        # 每折回撤 <= 30%
        max_dd = max(r["max_drawdown"] for r in fold_results)
        if max_dd > 30:
            return -100 - max_dd

        # ========== 計算各折指標 ==========
        sharpes = [r["sharpe"] for r in fold_results]
        returns = [r["total_return"] for r in fold_results]
        drawdowns = [r["max_drawdown"] for r in fold_results]
        trades_list = [r["trades"] for r in fold_results]

        profitable_folds = sum(1 for r in returns if r > 0)
        consistency_pct = profitable_folds / n_folds * 100
        avg_sharpe = sum(sharpes) / n_folds
        min_sharpe = min(sharpes)
        avg_return = sum(returns) / n_folds
        worst_dd = max(drawdowns)

        # ========== 目標函數 (v3: WF 穩健性) ==========

        # 1. 一致性 × 平均 Sharpe (核心: 所有折都要獲利)
        if avg_sharpe > 0:
            core_score = (consistency_pct / 100) * avg_sharpe * 10
        else:
            core_score = avg_sharpe * 5  # 負 Sharpe 直接懲罰

        # 2. 每個獲利折加分
        consistency_bonus = profitable_folds * 20

        # 3. 最差折 Sharpe 獎勵 (提高最差情況)
        min_sharpe_score = min_sharpe * 5

        # 4. 交易量獎勵 (鼓勵多交易，每 1000 筆加 5 分，上限 50 分)
        trade_bonus = min(total_trades / 1000 * 5, 50)

        # 5. 回撤懲罰
        dd_penalty = worst_dd * 0.5

        # 6. 全部獲利額外獎勵
        all_profitable_bonus = 50 if profitable_folds == n_folds else 0

        # 總分
        score = core_score + consistency_bonus + min_sharpe_score + trade_bonus - dd_penalty + all_profitable_bonus

        # 記錄詳細指標
        trial.set_user_attr("consistency_pct", consistency_pct)
        trial.set_user_attr("profitable_folds", profitable_folds)
        trial.set_user_attr("avg_sharpe", avg_sharpe)
        trial.set_user_attr("min_sharpe", min_sharpe)
        trial.set_user_attr("avg_return", avg_return)
        trial.set_user_attr("worst_dd", worst_dd)
        trial.set_user_attr("total_trades", total_trades)
        trial.set_user_attr("fold_sharpes", sharpes)
        trial.set_user_attr("fold_returns", returns)
        trial.set_user_attr("fold_trades", trades_list)

        return score

    return objective


def display_results(study: optuna.Study, leverage: int):
    """顯示優化結果."""
    print("\n" + "=" * 70)
    print("  優化完成 - 最佳結果 (Walk-Forward v3)")
    print("=" * 70)

    best = study.best_trial

    print(f"\n最佳得分: {best.value:.4f}")

    consistency = best.user_attrs.get('consistency_pct', 0)
    profitable = best.user_attrs.get('profitable_folds', 0)
    avg_sharpe = best.user_attrs.get('avg_sharpe', 0)
    min_sharpe = best.user_attrs.get('min_sharpe', 0)
    avg_return = best.user_attrs.get('avg_return', 0)
    worst_dd = best.user_attrs.get('worst_dd', 0)
    total_trades = best.user_attrs.get('total_trades', 0)
    fold_sharpes = best.user_attrs.get('fold_sharpes', [])
    fold_returns = best.user_attrs.get('fold_returns', [])
    fold_trades = best.user_attrs.get('fold_trades', [])

    n_folds = len(fold_sharpes) if fold_sharpes else 9

    print(f"\n[Walk-Forward 穩健性]")
    print(f"  一致性: {consistency:.1f}% ({profitable}/{n_folds} 折獲利)")
    print(f"  平均 Sharpe: {avg_sharpe:.2f}")
    print(f"  最差 Sharpe: {min_sharpe:.2f}")
    print(f"  平均報酬: {avg_return:+.2f}%")
    print(f"  最差回撤: {worst_dd:.2f}%")
    print(f"  總交易數: {total_trades:,}")

    if fold_sharpes:
        print(f"\n[各折詳細]")
        for i, (s, r, t) in enumerate(zip(fold_sharpes, fold_returns, fold_trades)):
            status = "✓" if r > 0 else "✗"
            print(f"  折 {i+1}: Sharpe {s:+.2f}, 報酬 {r:+.1f}%, 交易 {t} {status}")

    # 目標達成檢查
    print(f"\n[目標達成檢查]")
    c_target = "PASS" if consistency >= 50 else "FAIL"
    s_target = "PASS" if avg_sharpe > 0 else "FAIL"
    ms_target = "PASS" if min_sharpe > -5 else "FAIL"
    t_target = "PASS" if total_trades >= 3000 else "FAIL"

    print(f"  一致性 >= 50%: {c_target} ({consistency:.1f}%)")
    print(f"  平均 Sharpe > 0: {s_target} ({avg_sharpe:.2f})")
    print(f"  最差 Sharpe > -5: {ms_target} ({min_sharpe:.2f})")
    print(f"  總交易 >= 3000: {t_target} ({total_trades:,})")

    # 綜合評估
    print(f"\n[綜合評估]")
    if consistency == 100 and avg_sharpe > 2:
        print(f"  ★★★ 優秀: 所有折獲利, Sharpe > 2")
    elif consistency >= 70 and avg_sharpe > 0:
        print(f"  ★★☆ 良好: 大部分折獲利")
    elif consistency >= 50 and avg_sharpe > 0:
        print(f"  ★☆☆ 可接受: 半數折獲利")
    else:
        print(f"  ☆☆☆ 需改進: 一致性或 Sharpe 不足")

    print("\n" + "-" * 70)
    print("最佳參數:")
    print("-" * 70)
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    return best.params


def save_results(study: optuna.Study, output_dir: str, leverage: int, data_file: str):
    """儲存優化結果."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    best = study.best_trial
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"bb_neutral_optimization_{timestamp}.json"

    results = {
        "optimization_date": datetime.now(timezone.utc).isoformat(),
        "optimization_version": "v3_walk_forward",
        "data_file": data_file,
        "n_trials": len(study.trials),
        "leverage": leverage,
        "best_score": best.value,
        "best_params": best.params,
        "metrics": {
            "consistency_pct": best.user_attrs.get("consistency_pct", 0),
            "profitable_folds": best.user_attrs.get("profitable_folds", 0),
            "avg_sharpe": best.user_attrs.get("avg_sharpe", 0),
            "min_sharpe": best.user_attrs.get("min_sharpe", 0),
            "avg_return": best.user_attrs.get("avg_return", 0),
            "worst_dd": best.user_attrs.get("worst_dd", 0),
            "total_trades": best.user_attrs.get("total_trades", 0),
            "fold_sharpes": best.user_attrs.get("fold_sharpes", []),
            "fold_returns": best.user_attrs.get("fold_returns", []),
            "fold_trades": best.user_attrs.get("fold_trades", []),
        },
        "top_10_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "metrics": {
                    "consistency_pct": t.user_attrs.get("consistency_pct", 0),
                    "avg_sharpe": t.user_attrs.get("avg_sharpe", 0),
                    "total_trades": t.user_attrs.get("total_trades", 0),
                },
            }
            for t in sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda x: x.value if x.value else -9999,
                reverse=True
            )[:10]
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {output_file}")
    return output_file


def print_config_suggestion(params: dict):
    """輸出建議配置."""
    print("\n" + "=" * 70)
    print("  建議配置 (可直接用於 validate_bollinger_wf.py)")
    print("=" * 70)
    print(f"""
config = BollingerStrategyConfig(
    mode=BollingerMode.BB_NEUTRAL_GRID,
    bb_period={params['bb_period']},
    bb_std=Decimal("2.0"),
    grid_count={params['grid_count']},
    take_profit_grids={params['take_profit_grids']},
    stop_loss_pct=Decimal("{params['stop_loss_pct']}"),
    use_atr_range=True,
    atr_period={params['atr_period']},
    atr_multiplier=Decimal("{params['atr_multiplier']}"),
    fallback_range_pct=Decimal("0.04"),
    use_hysteresis={params['use_hysteresis']},
    hysteresis_pct=Decimal("{params['hysteresis_pct']}"),
    use_signal_cooldown={params['use_signal_cooldown']},
    cooldown_bars={params['cooldown_bars']},
)
""")


def main():
    parser = argparse.ArgumentParser(
        description="BB_NEUTRAL_GRID Walk-Forward 穩健性優化 (v3)"
    )
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
        help="超時秒數 (default: 無限制)"
    )
    parser.add_argument(
        "--leverage", "-l",
        type=int,
        default=18,
        help="槓桿倍數 (default: 18)"
    )
    parser.add_argument(
        "--folds", "-f",
        type=int,
        default=9,
        help="Walk-Forward 折數 (default: 9)"
    )
    parser.add_argument(
        "--output",
        default="optimization_results",
        help="輸出目錄"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="快速模式 (50 trials)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子 (default: 42)"
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("錯誤: 需要安裝 optuna")
        print("  pip install optuna")
        return

    # 快速模式
    if args.quick:
        args.trials = 50

    print("=" * 70)
    print("  BB_NEUTRAL_GRID Walk-Forward 優化 v3 (穩健性)")
    print("  目標: 所有折獲利 + 高 Sharpe + 多交易")
    print("=" * 70)
    print(f"  試驗次數: {args.trials}")
    print(f"  槓桿倍數: {args.leverage}x")
    print(f"  折數: {args.folds}")

    # 載入數據
    klines = load_klines(args.data_file)

    # 分割數據為 N 折
    folds = split_into_folds(klines, args.folds)

    # 創建 Optuna study
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="bb_neutral_wf_v3",
    )

    # 創建 WF 目標函數
    objective = create_wf_objective(folds, args.leverage)

    # 執行優化
    print(f"\n開始優化 ({args.trials} 次試驗)...")
    print("-" * 70)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # 顯示結果
    best_params = display_results(study, args.leverage)

    # 儲存結果
    save_results(study, args.output, args.leverage, args.data_file)

    # 輸出建議配置
    print_config_suggestion(best_params)


if __name__ == "__main__":
    main()
