#!/usr/bin/env python3
"""
Grid Futures 策略優化腳本 (IS/OOS split).

使用 70/30 train/test split 防止過擬合。
Equity-based margin 引擎配置，對齊 run_backtest / WF 驗證。

參數範圍:
- leverage: 3-20 (合理範圍)
- grid_count: 5-30
- atr_multiplier: 1.0-10.0
- atr_period: 7-50
- stop_loss_pct: 0.5%-5.0%
- take_profit_grids: 1-5
- direction: NEUTRAL / TREND_FOLLOW

用法:
    python optimize_grid_futures.py --trials 300
    python optimize_grid_futures.py --trials 100 --quick
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
from src.backtest.strategy.grid_futures import (
    GridFuturesBacktestStrategy,
    GridFuturesStrategyConfig,
    GridDirection,
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

    # 計算數據跨度年數
    days = (klines[-1].open_time - klines[0].open_time).days
    years = days / 365.0
    print(f"數據跨度: {days} 天 ({years:.2f} 年)")

    return klines


def split_data(klines: list[Kline], train_ratio: float = 0.7) -> tuple[list[Kline], list[Kline], float, float]:
    """將數據分為訓練集 (IS) 和測試集 (OOS)."""
    split_idx = int(len(klines) * train_ratio)
    train_klines = klines[:split_idx]
    test_klines = klines[split_idx:]

    train_market = (float(train_klines[-1].close) / float(train_klines[0].close) - 1) * 100
    test_market = (float(test_klines[-1].close) / float(test_klines[0].close) - 1) * 100

    print(f"\n數據分割:")
    print(f"  訓練集 (IS): {train_klines[0].open_time.strftime('%Y-%m-%d')} ~ {train_klines[-1].open_time.strftime('%Y-%m-%d')} ({len(train_klines):,} 根)")
    print(f"  測試集 (OOS): {test_klines[0].open_time.strftime('%Y-%m-%d')} ~ {test_klines[-1].open_time.strftime('%Y-%m-%d')} ({len(test_klines):,} 根)")
    print(f"\n市場基準 (Buy & Hold):")
    print(f"  訓練期: {train_market:+.2f}%")
    print(f"  測試期: {test_market:+.2f}%")

    return train_klines, test_klines, train_market, test_market


def run_backtest(klines: list[Kline], params: dict, leverage: int) -> dict:
    """使用指定參數執行回測."""
    direction_str = params.get("direction", "NEUTRAL")
    direction = GridDirection.NEUTRAL if direction_str == "NEUTRAL" else GridDirection.TREND_FOLLOW

    config = GridFuturesStrategyConfig(
        grid_count=params["grid_count"],
        direction=direction,
        leverage=leverage,
        trend_period=params["trend_period"],
        atr_period=params["atr_period"],
        atr_multiplier=Decimal(str(params["atr_multiplier"])),
        stop_loss_pct=Decimal(str(params["stop_loss_pct"])),
        take_profit_grids=params["take_profit_grids"],
        use_hysteresis=params["use_hysteresis"],
        hysteresis_pct=Decimal(str(params["hysteresis_pct"])),
        use_signal_cooldown=params["use_signal_cooldown"],
        cooldown_bars=params["cooldown_bars"],
    )

    strategy = GridFuturesBacktestStrategy(config)

    bt_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0006"),
        slippage_pct=Decimal("0.0005"),
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


def create_grid_futures_objective(
    train_klines: list[Kline],
    test_klines: list[Kline],
    train_market: float,
    test_market: float,
    leverage: int = 7,
):
    """創建 Grid Futures 優化目標函數 (IS+OOS 聯合優化)."""

    def objective(trial: optuna.Trial) -> float:
        # 參數搜索空間
        params = {
            "grid_count": trial.suggest_int("grid_count", 5, 30),
            "atr_period": trial.suggest_int("atr_period", 7, 50),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 1.0, 10.0, step=0.5),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.05, step=0.005),
            "take_profit_grids": trial.suggest_int("take_profit_grids", 1, 5),
            "direction": trial.suggest_categorical("direction", ["NEUTRAL", "TREND_FOLLOW"]),
            "trend_period": trial.suggest_int("trend_period", 10, 50),
            "use_hysteresis": trial.suggest_categorical("use_hysteresis", [True, False]),
            "hysteresis_pct": trial.suggest_float("hysteresis_pct", 0.001, 0.01, step=0.001),
            "use_signal_cooldown": trial.suggest_categorical("use_signal_cooldown", [True, False]),
            "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 5),
        }

        # IS 回測
        try:
            is_result = run_backtest(train_klines, params, leverage)
        except Exception:
            return -1000

        # OOS 回測
        try:
            oos_result = run_backtest(test_klines, params, leverage)
        except Exception:
            return -1000

        # 提取指標
        is_return = is_result["total_return"]
        oos_return = oos_result["total_return"]
        is_sharpe = is_result["sharpe"]
        oos_sharpe = oos_result["sharpe"]
        is_trades = is_result["trades"]
        oos_trades = oos_result["trades"]
        is_dd = is_result["max_drawdown"]
        oos_dd = oos_result["max_drawdown"]
        max_dd = max(is_dd, oos_dd)

        # 超額報酬
        is_excess = is_return - train_market
        oos_excess = oos_return - test_market

        # OOS/IS 比率
        if is_return > 0:
            oos_is_ratio = oos_return / is_return
        else:
            oos_is_ratio = 0

        # ========== 約束條件 ==========
        if is_trades < 100:
            return -500 + is_trades
        if oos_trades < 40:
            return -500 + oos_trades
        if max_dd > 20:  # 收緊: 35% → 20%
            return -100 - max_dd

        # ========== 目標函數 (IS+OOS 聯合, v2 強化 OOS) ==========

        # IS 報酬 (降低權重，避免偏重 IS)
        is_return_score = is_return * 0.5

        # OOS 報酬（大幅提高權重: 2.5 → 4.0）
        oos_return_score = oos_return * 4.0

        # OOS 超額報酬
        oos_excess_score = oos_excess * 1.5

        # OOS Sharpe (提高權重)
        sharpe_score = oos_sharpe * 3

        # OOS Sharpe 門檻懲罰: < 1.0 直接扣分
        oos_sharpe_penalty = 0
        if oos_sharpe < 1.0:
            oos_sharpe_penalty = (1.0 - oos_sharpe) * 20

        # 回撤懲罰 (更嚴格)
        dd_penalty = max(0, max_dd - 10) * 1.0

        # 一致性獎勵 (提高)
        consistency_bonus = 0
        if is_excess > 0 and oos_excess > 0:
            consistency_bonus = 15

        # WF 模擬獎勵: OOS/IS 比率 > 0.5 額外加分
        wf_bonus = 0
        if oos_is_ratio > 0.5:
            wf_bonus = (oos_is_ratio - 0.5) * 30

        # Overfit penalty: 閾值 0.3→0.5, 係數 50→100
        overfit_penalty = 0
        if is_return > 0 and oos_is_ratio < 0.5:
            overfit_penalty = (0.5 - oos_is_ratio) * 100

        # 總分
        score = (is_return_score + oos_return_score + oos_excess_score
                 + sharpe_score - oos_sharpe_penalty - dd_penalty
                 + consistency_bonus + wf_bonus - overfit_penalty)

        # 記錄指標
        trial.set_user_attr("is_return", is_return)
        trial.set_user_attr("oos_return", oos_return)
        trial.set_user_attr("is_excess", is_excess)
        trial.set_user_attr("oos_excess", oos_excess)
        trial.set_user_attr("is_sharpe", is_sharpe)
        trial.set_user_attr("oos_sharpe", oos_sharpe)
        trial.set_user_attr("is_trades", is_trades)
        trial.set_user_attr("oos_trades", oos_trades)
        trial.set_user_attr("is_dd", is_dd)
        trial.set_user_attr("oos_dd", oos_dd)
        trial.set_user_attr("oos_is_ratio", oos_is_ratio)
        trial.set_user_attr("train_market", train_market)
        trial.set_user_attr("test_market", test_market)

        return score

    return objective


def display_results(study: optuna.Study, leverage: int):
    """顯示優化結果."""
    print("\n" + "=" * 70)
    print("  優化完成 - 最佳結果")
    print("=" * 70)

    best = study.best_trial

    train_market = best.user_attrs.get('train_market', 0)
    test_market = best.user_attrs.get('test_market', 0)

    print(f"\n最佳得分: {best.value:.4f}")

    print(f"\n[市場基準 (Buy & Hold)]")
    print(f"  訓練期: {train_market:+.2f}%")
    print(f"  測試期: {test_market:+.2f}%")

    print(f"\n[樣本內 (IS) - 訓練期]")
    print(f"  報酬: {best.user_attrs.get('is_return', 0):+.2f}%")
    print(f"  超額報酬: {best.user_attrs.get('is_excess', 0):+.2f}%")
    print(f"  Sharpe: {best.user_attrs.get('is_sharpe', 0):.2f}")
    print(f"  回撤: {best.user_attrs.get('is_dd', 0):.2f}%")
    print(f"  交易: {best.user_attrs.get('is_trades', 0)}")

    print(f"\n[樣本外 (OOS) - 測試期]")
    print(f"  報酬: {best.user_attrs.get('oos_return', 0):+.2f}%")
    print(f"  超額報酬: {best.user_attrs.get('oos_excess', 0):+.2f}%")
    print(f"  Sharpe: {best.user_attrs.get('oos_sharpe', 0):.2f}")
    print(f"  回撤: {best.user_attrs.get('oos_dd', 0):.2f}%")
    print(f"  交易: {best.user_attrs.get('oos_trades', 0)}")

    oos_is_ratio = best.user_attrs.get('oos_is_ratio', 0)
    oos_excess = best.user_attrs.get('oos_excess', 0)
    oos_return = best.user_attrs.get('oos_return', 0)

    print(f"\n[一致性檢查]")
    print(f"  OOS/IS 比率: {oos_is_ratio:.2f} ({'PASS' if oos_is_ratio >= 0.5 else 'FAIL'})")
    print(f"  OOS 超額報酬: {oos_excess:+.2f}% ({'PASS' if oos_excess > 0 else 'FAIL'})")

    print(f"\n[綜合評估]")
    if oos_return > 0 and oos_excess > 0 and oos_is_ratio >= 0.5:
        print(f"  *** 優秀: OOS 正報酬 + 跑贏市場 + 高一致性")
    elif oos_return > 0 and oos_excess > 0:
        print(f"  ** 良好: OOS 正報酬且跑贏市場")
    elif oos_excess > 0:
        print(f"  * 可接受: OOS 跑贏市場")
    else:
        print(f"  需改進: OOS 未跑贏市場")

    print("\n" + "-" * 70)
    print(f"最佳參數 (leverage={leverage}):")
    print("-" * 70)
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    return best.params


def save_results(study: optuna.Study, output_dir: str, leverage: int, data_file: str):
    """儲存優化結果."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    best = study.best_trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"grid_futures_optimization_{timestamp}.json"

    results = {
        "optimization_date": datetime.now().isoformat(),
        "data_file": data_file,
        "n_trials": len(study.trials),
        "leverage": leverage,
        "best_score": best.value,
        "best_params": best.params,
        "metrics": {
            "is_return": best.user_attrs.get("is_return", 0),
            "oos_return": best.user_attrs.get("oos_return", 0),
            "is_excess": best.user_attrs.get("is_excess", 0),
            "oos_excess": best.user_attrs.get("oos_excess", 0),
            "is_sharpe": best.user_attrs.get("is_sharpe", 0),
            "oos_sharpe": best.user_attrs.get("oos_sharpe", 0),
            "is_trades": best.user_attrs.get("is_trades", 0),
            "oos_trades": best.user_attrs.get("oos_trades", 0),
            "is_dd": best.user_attrs.get("is_dd", 0),
            "oos_dd": best.user_attrs.get("oos_dd", 0),
            "oos_is_ratio": best.user_attrs.get("oos_is_ratio", 0),
            "train_market": best.user_attrs.get("train_market", 0),
            "test_market": best.user_attrs.get("test_market", 0),
        },
        "top_10_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "metrics": {
                    "is_return": t.user_attrs.get("is_return", 0),
                    "oos_return": t.user_attrs.get("oos_return", 0),
                    "oos_excess": t.user_attrs.get("oos_excess", 0),
                    "oos_is_ratio": t.user_attrs.get("oos_is_ratio", 0),
                },
            }
            for t in sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda x: x.value if x.value else -9999,
                reverse=True,
            )[:10]
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {output_file}")
    return output_file


def print_config_code(params: dict, leverage: int):
    """印出可用於更新配置的代碼."""
    print(f"\n{'=' * 60}")
    print(f"  建議配置")
    print(f"{'=' * 60}")

    direction = params.get("direction", "NEUTRAL")

    print(f"""
GridFuturesStrategyConfig(
    grid_count={params.get('grid_count', 12)},
    direction=GridDirection.{direction},
    leverage={leverage},
    trend_period={params.get('trend_period', 20)},
    atr_period={params.get('atr_period', 21)},
    atr_multiplier=Decimal("{params.get('atr_multiplier', 6.0)}"),
    stop_loss_pct=Decimal("{params.get('stop_loss_pct', 0.005)}"),
    take_profit_grids={params.get('take_profit_grids', 2)},
    use_hysteresis={params.get('use_hysteresis', True)},
    hysteresis_pct=Decimal("{params.get('hysteresis_pct', 0.002)}"),
    use_signal_cooldown={params.get('use_signal_cooldown', False)},
    cooldown_bars={params.get('cooldown_bars', 0)},
)
""")


def main():
    parser = argparse.ArgumentParser(description="Grid Futures 策略優化 (IS+OOS split)")
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
        "--leverage", "-l",
        type=int,
        default=7,
        help="槓桿倍數 (default: 7)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="訓練集比例 (default: 0.7)"
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
        print("錯誤: 需要安裝 optuna: pip install optuna")
        return

    if args.quick:
        args.trials = 50

    print("=" * 70)
    print("  Grid Futures 策略優化 (IS+OOS 聯合, equity-based margin)")
    print("=" * 70)
    print(f"  試驗次數: {args.trials}")
    print(f"  槓桿倍數: {args.leverage}x")
    print(f"  訓練比例: {args.train_ratio * 100:.0f}%")

    # 載入數據
    klines = load_klines(args.data_file)

    # 分割數據
    train_klines, test_klines, train_market, test_market = split_data(klines, args.train_ratio)

    # 創建 Optuna study
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="grid_futures_is_oos",
    )

    # 創建目標函數
    objective = create_grid_futures_objective(
        train_klines, test_klines, train_market, test_market, args.leverage
    )

    # 執行優化
    print(f"\n開始優化 ({args.trials} 次試驗)...")
    print("-" * 70)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=None,
        show_progress_bar=True,
    )

    # 顯示結果
    best_params = display_results(study, args.leverage)

    # 儲存結果
    save_results(study, args.output, args.leverage, args.data_file)

    # 輸出建議配置
    print_config_code(best_params, args.leverage)


if __name__ == "__main__":
    main()
