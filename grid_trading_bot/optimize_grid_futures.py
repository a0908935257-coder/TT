#!/usr/bin/env python3
"""
Grid Futures 積極策略優化腳本.

目標：年化報酬 40%+
優化目標函數：最大化年化報酬
約束：最大回撤 <= 30%，交易次數 >= 100

參數範圍 (積極優化):
- leverage: 1-125 (不限制)
- grid_count: 5-30
- atr_multiplier: 1.0-10.0
- atr_period: 7-50
- stop_loss_pct: 0.5%-5.0%
- position_size_pct: 10%-50%
- take_profit_grids: 1-5
- direction: NEUTRAL / TREND_FOLLOW

用法:
    python optimize_grid_futures.py --trials 200
    python optimize_grid_futures.py --trials 200 --target-annual-return 40
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


def create_grid_futures_objective(klines: list[Kline], target_annual_return: float = 40.0):
    """創建 Grid Futures 優化目標函數 - 積極優化年化報酬."""

    # 計算數據跨度年數
    days = (klines[-1].open_time - klines[0].open_time).days
    years = days / 365.0

    def objective(trial: optuna.Trial) -> float:
        # ======================================
        # 參數搜索空間 (積極優化)
        # ======================================

        # 核心參數
        leverage = trial.suggest_int("leverage", 1, 125)
        grid_count = trial.suggest_int("grid_count", 5, 30)
        atr_period = trial.suggest_int("atr_period", 7, 50)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 10.0, step=0.5)

        # 風險控制
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.005, 0.05, step=0.005)
        take_profit_grids = trial.suggest_int("take_profit_grids", 1, 5)

        # 方向
        direction_str = trial.suggest_categorical("direction", ["NEUTRAL", "TREND_FOLLOW"])
        direction = GridDirection.NEUTRAL if direction_str == "NEUTRAL" else GridDirection.TREND_FOLLOW

        # 趨勢週期 (僅 TREND_FOLLOW 使用)
        trend_period = trial.suggest_int("trend_period", 10, 50)

        # 保護機制
        use_hysteresis = trial.suggest_categorical("use_hysteresis", [True, False])
        hysteresis_pct = trial.suggest_float("hysteresis_pct", 0.001, 0.01, step=0.001)
        use_signal_cooldown = trial.suggest_categorical("use_signal_cooldown", [True, False])
        cooldown_bars = trial.suggest_int("cooldown_bars", 0, 5)

        # ======================================
        # 建立策略配置
        # ======================================
        config = GridFuturesStrategyConfig(
            grid_count=grid_count,
            direction=direction,
            leverage=leverage,
            trend_period=trend_period,
            atr_period=atr_period,
            atr_multiplier=Decimal(str(atr_multiplier)),
            stop_loss_pct=Decimal(str(stop_loss_pct)),
            take_profit_grids=take_profit_grids,
            use_hysteresis=use_hysteresis,
            hysteresis_pct=Decimal(str(hysteresis_pct)),
            use_signal_cooldown=use_signal_cooldown,
            cooldown_bars=cooldown_bars,
        )

        strategy = GridFuturesBacktestStrategy(config)

        # ======================================
        # 執行回測
        # ======================================
        bt_config = BacktestConfig(
            initial_capital=Decimal("10000"),
            leverage=leverage,
            fee_rate=Decimal("0.0004"),
            slippage_pct=Decimal("0.0001"),
        )

        engine = BacktestEngine(bt_config)
        result = engine.run(klines, strategy)

        # ======================================
        # 計算指標
        # ======================================
        total_return = float(result.total_profit_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)
        trades = result.total_trades
        sharpe = float(result.sharpe_ratio)

        # 計算年化報酬
        if total_return > -100:  # 避免負值無法計算
            annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        else:
            annual_return = -100

        # ======================================
        # 約束條件 (寬鬆以探索積極策略)
        # ======================================
        if trades < 100:
            return -1000  # 交易次數太少
        if max_dd > 30:
            return -1000  # 回撤超過 30%
        if win_rate < 45:
            return -1000  # 勝率太低

        # ======================================
        # 目標函數: 最大化年化報酬，輕微懲罰回撤
        # ======================================
        # 報酬主導，回撤懲罰係數 0.5 (容忍較高回撤)
        score = annual_return - (max_dd * 0.5)

        # 記錄指標
        trial.set_user_attr("total_return", total_return)
        trial.set_user_attr("annual_return", annual_return)
        trial.set_user_attr("max_drawdown", max_dd)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("trades", trades)
        trial.set_user_attr("sharpe", sharpe)
        trial.set_user_attr("years", years)

        return score

    return objective


def run_optimization(klines: list[Kline], trials: int, target_annual_return: float) -> dict:
    """執行 Grid Futures 優化."""
    print(f"\n{'=' * 60}")
    print(f"  優化 GRID FUTURES (目標: 年化 {target_annual_return}%+)")
    print(f"  參數範圍: 槓桿 1-125x (不限制)")
    print(f"  試驗次數: {trials}")
    print(f"{'=' * 60}")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="grid_futures_aggressive",
    )

    objective = create_grid_futures_objective(klines, target_annual_return)

    study.optimize(
        objective,
        n_trials=trials,
        timeout=None,
        show_progress_bar=True,
    )

    best = study.best_trial

    # 提取最佳結果
    annual_return = best.user_attrs.get('annual_return', 0)
    total_return = best.user_attrs.get('total_return', 0)
    max_drawdown = best.user_attrs.get('max_drawdown', 0)
    win_rate = best.user_attrs.get('win_rate', 0)
    trades = best.user_attrs.get('trades', 0)
    sharpe = best.user_attrs.get('sharpe', 0)

    # 判斷是否達成目標
    target_met = annual_return >= target_annual_return

    print(f"\n{'=' * 60}")
    print(f"  最佳結果")
    print(f"{'=' * 60}")
    print(f"\n績效指標:")
    print(f"  年化報酬: {annual_return:.2f}% {'✅' if target_met else '⚠️'}")
    print(f"  總報酬 ({best.user_attrs.get('years', 2):.1f}年): {total_return:.2f}%")
    print(f"  最大回撤: {max_drawdown:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  勝率: {win_rate:.1f}%")
    print(f"  交易數: {trades}")

    print(f"\n最佳參數:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    if target_met:
        print(f"\n✅ 達成目標年化報酬 {target_annual_return}%!")
    else:
        print(f"\n⚠️  未達成目標 (差距: {target_annual_return - annual_return:.2f}%)")

    return {
        "strategy": "grid_futures",
        "target_annual_return": target_annual_return,
        "target_met": target_met,
        "best_params": best.params,
        "metrics": {
            "annual_return": annual_return,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "trades": trades,
        },
    }


def print_config_code(params: dict):
    """印出可用於更新 models.py 的配置代碼."""
    print(f"\n{'=' * 60}")
    print(f"  更新配置代碼 (models.py)")
    print(f"{'=' * 60}")

    direction = params.get("direction", "NEUTRAL")
    leverage = params.get("leverage", 18)
    grid_count = params.get("grid_count", 12)
    atr_period = params.get("atr_period", 21)
    atr_multiplier = params.get("atr_multiplier", 6.0)
    stop_loss_pct = params.get("stop_loss_pct", 0.005)
    trend_period = params.get("trend_period", 20)
    use_hysteresis = params.get("use_hysteresis", True)
    hysteresis_pct = params.get("hysteresis_pct", 0.002)
    use_signal_cooldown = params.get("use_signal_cooldown", False)
    cooldown_bars = params.get("cooldown_bars", 0)

    print(f"""
# GridFuturesConfig 更新配置:
leverage: int = {leverage}  # 優化結果
grid_count: int = {grid_count}  # 優化結果
direction: GridDirection = GridDirection.{direction}
trend_period: int = {trend_period}  # 優化結果
atr_period: int = {atr_period}  # 優化結果
atr_multiplier: Decimal = field(default_factory=lambda: Decimal("{atr_multiplier}"))
stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("{stop_loss_pct}"))
use_hysteresis: bool = {use_hysteresis}
hysteresis_pct: Decimal = field(default_factory=lambda: Decimal("{hysteresis_pct}"))
use_signal_cooldown: bool = {use_signal_cooldown}
cooldown_bars: int = {cooldown_bars}
""")


def main():
    parser = argparse.ArgumentParser(description="Grid Futures 積極策略優化")
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
        "--target-annual-return",
        type=float,
        default=40.0,
        help="目標年化報酬 (default: 40%%)"
    )
    parser.add_argument(
        "--output",
        default="optimization_results",
        help="輸出目錄"
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("錯誤: 需要安裝 optuna: pip install optuna")
        return

    klines = load_klines(args.data_file)

    result = run_optimization(klines, args.trials, args.target_annual_return)

    # 儲存結果
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"grid_futures_aggressive_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "optimization_date": datetime.now().isoformat(),
            "target": f"maximize_annual_return_{args.target_annual_return}%",
            "data_file": args.data_file,
            "trials": args.trials,
            "result": result,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {output_file}")

    # 印出配置代碼
    print_config_code(result["best_params"])

    # 提示下一步
    print(f"\n{'=' * 60}")
    print(f"  下一步: Walk-Forward 驗證")
    print(f"{'=' * 60}")
    print(f"""
請執行以下命令進行 Walk-Forward 驗證:

python run_walk_forward_validation.py \\
    --strategy grid_futures \\
    --data-file {args.data_file} \\
    --periods 9 \\
    --monte-carlo

驗證標準:
- OOS/IS 比率 >= 0.5
- 一致性 >= 50%
- OOS Sharpe > 0
""")


if __name__ == "__main__":
    main()
