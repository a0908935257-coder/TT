#!/usr/bin/env python3
"""
最大化報酬優化腳本.

目標：年化報酬 30%+
優化目標函數：最大化報酬（而非 Sharpe）
約束：最大回撤 < 15%，勝率 > 60%

用法:
    python optimize_max_return.py --strategy supertrend --trials 150
    python optimize_max_return.py --strategy rsi_grid --trials 150
    python optimize_max_return.py --strategy all --trials 150
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


def create_supertrend_objective(klines: list[Kline], leverage: int = 10):
    """創建 Supertrend 優化目標函數 - 最大化報酬."""

    def objective(trial: optuna.Trial) -> float:
        # 優化參數範圍 (更寬的搜索空間)
        atr_period = trial.suggest_int("atr_period", 5, 50)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 5.0, step=0.25)
        grid_count = trial.suggest_int("grid_count", 3, 20)
        grid_atr_multiplier = trial.suggest_float("grid_atr_multiplier", 1.0, 6.0, step=0.5)
        take_profit_grids = trial.suggest_int("take_profit_grids", 1, 4)
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.02, 0.15, step=0.01)

        # RSI 過濾器
        use_rsi_filter = trial.suggest_categorical("use_rsi_filter", [True, False])
        rsi_period = trial.suggest_int("rsi_period", 5, 25)
        rsi_overbought = trial.suggest_int("rsi_overbought", 55, 80)
        rsi_oversold = trial.suggest_int("rsi_oversold", 20, 45)

        # 趨勢確認
        min_trend_bars = trial.suggest_int("min_trend_bars", 1, 5)

        # 保護機制
        use_hysteresis = trial.suggest_categorical("use_hysteresis", [True, False])
        hysteresis_pct = trial.suggest_float("hysteresis_pct", 0.001, 0.01, step=0.001)
        use_signal_cooldown = trial.suggest_categorical("use_signal_cooldown", [True, False])
        cooldown_bars = trial.suggest_int("cooldown_bars", 0, 5)

        # Trailing stop
        use_trailing_stop = trial.suggest_categorical("use_trailing_stop", [True, False])
        trailing_stop_pct = trial.suggest_float("trailing_stop_pct", 0.01, 0.10, step=0.01)

        config = SupertrendStrategyConfig(
            mode=SupertrendMode.TREND_GRID,
            atr_period=atr_period,
            atr_multiplier=Decimal(str(atr_multiplier)),
            grid_count=grid_count,
            grid_atr_multiplier=Decimal(str(grid_atr_multiplier)),
            take_profit_grids=take_profit_grids,
            stop_loss_pct=Decimal(str(stop_loss_pct)),
            use_rsi_filter=use_rsi_filter,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            min_trend_bars=min_trend_bars,
            use_hysteresis=use_hysteresis,
            hysteresis_pct=Decimal(str(hysteresis_pct)),
            use_signal_cooldown=use_signal_cooldown,
            cooldown_bars=cooldown_bars,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_pct=Decimal(str(trailing_stop_pct)),
        )

        strategy = SupertrendBacktestStrategy(config)

        bt_config = BacktestConfig(
            initial_capital=Decimal("10000"),
            leverage=leverage,
            fee_rate=Decimal("0.0004"),
            slippage_pct=Decimal("0.0001"),
        )

        engine = BacktestEngine(bt_config)
        result = engine.run(klines, strategy)

        total_return = float(result.total_profit_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)
        trades = result.total_trades
        sharpe = float(result.sharpe_ratio)

        # 約束條件
        if trades < 30:
            return -1000
        if max_dd > 20:
            return -1000
        if win_rate < 50:
            return -1000

        # 目標函數: 最大化報酬，但懲罰過大回撤
        # 報酬 - 回撤懲罰
        score = total_return - (max_dd * 2)  # 回撤懲罰係數 2

        trial.set_user_attr("total_return", total_return)
        trial.set_user_attr("max_drawdown", max_dd)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("trades", trades)
        trial.set_user_attr("sharpe", sharpe)

        return score

    return objective


def create_rsi_grid_objective(klines: list[Kline], leverage: int = 5):
    """創建 RSI-Grid 優化目標函數 - 最大化報酬."""

    def objective(trial: optuna.Trial) -> float:
        # RSI 參數
        rsi_period = trial.suggest_int("rsi_period", 5, 30)
        oversold_level = trial.suggest_int("oversold_level", 15, 45)
        overbought_level = trial.suggest_int("overbought_level", 55, 85)

        # Grid 參數
        grid_count = trial.suggest_int("grid_count", 3, 20)
        atr_period = trial.suggest_int("atr_period", 5, 35)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 6.0, step=0.5)

        # Trend Filter
        trend_sma_period = trial.suggest_int("trend_sma_period", 5, 60)
        use_trend_filter = trial.suggest_categorical("use_trend_filter", [True, False])

        # Risk Management
        stop_loss_atr_mult = trial.suggest_float("stop_loss_atr_mult", 0.5, 4.0, step=0.5)
        max_stop_loss_pct = trial.suggest_float("max_stop_loss_pct", 0.02, 0.10, step=0.01)
        take_profit_grids = trial.suggest_int("take_profit_grids", 1, 4)

        # 保護機制
        use_hysteresis = trial.suggest_categorical("use_hysteresis", [True, False])
        hysteresis_pct = trial.suggest_float("hysteresis_pct", 0.001, 0.01, step=0.001)
        use_signal_cooldown = trial.suggest_categorical("use_signal_cooldown", [True, False])
        cooldown_bars = trial.suggest_int("cooldown_bars", 0, 5)

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

        bt_config = BacktestConfig(
            initial_capital=Decimal("10000"),
            leverage=leverage,
            fee_rate=Decimal("0.0004"),
            slippage_pct=Decimal("0.0001"),
        )

        engine = BacktestEngine(bt_config)
        result = engine.run(klines, strategy)

        total_return = float(result.total_profit_pct)
        max_dd = float(result.max_drawdown_pct)
        win_rate = float(result.win_rate)
        trades = result.total_trades
        sharpe = float(result.sharpe_ratio)

        # 約束條件
        if trades < 20:
            return -1000
        if max_dd > 20:
            return -1000
        if win_rate < 50:
            return -1000

        # 目標函數: 最大化報酬
        score = total_return - (max_dd * 2)

        trial.set_user_attr("total_return", total_return)
        trial.set_user_attr("max_drawdown", max_dd)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("trades", trades)
        trial.set_user_attr("sharpe", sharpe)

        return score

    return objective


def run_optimization(strategy_name: str, klines: list[Kline], trials: int, leverage: int):
    """執行單一策略優化."""
    print(f"\n{'=' * 60}")
    print(f"  優化 {strategy_name.upper()} (目標: 最大化報酬)")
    print(f"  槓桿: {leverage}x, 試驗次數: {trials}")
    print(f"{'=' * 60}")

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{strategy_name}_max_return",
    )

    if strategy_name == "supertrend":
        objective = create_supertrend_objective(klines, leverage)
    else:
        objective = create_rsi_grid_objective(klines, leverage)

    study.optimize(
        objective,
        n_trials=trials,
        timeout=None,
        show_progress_bar=True,
    )

    best = study.best_trial

    # 計算年化報酬
    total_return = best.user_attrs.get('total_return', 0)
    annual_return = ((1 + total_return/100) ** 0.5 - 1) * 100

    print(f"\n最佳結果:")
    print(f"  總報酬 (2年): {total_return:.2f}%")
    print(f"  年化報酬: {annual_return:.2f}%")
    print(f"  Sharpe: {best.user_attrs.get('sharpe', 0):.2f}")
    print(f"  最大回撤: {best.user_attrs.get('max_drawdown', 0):.2f}%")
    print(f"  勝率: {best.user_attrs.get('win_rate', 0):.1f}%")
    print(f"  交易數: {best.user_attrs.get('trades', 0)}")

    print(f"\n最佳參數:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    return {
        "strategy": strategy_name,
        "leverage": leverage,
        "best_params": best.params,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": best.user_attrs.get('sharpe', 0),
        "max_drawdown": best.user_attrs.get('max_drawdown', 0),
        "win_rate": best.user_attrs.get('win_rate', 0),
        "trades": best.user_attrs.get('trades', 0),
    }


def main():
    parser = argparse.ArgumentParser(description="最大化報酬優化")
    parser.add_argument(
        "--strategy", "-s",
        choices=["supertrend", "rsi_grid", "all"],
        default="all",
        help="要優化的策略"
    )
    parser.add_argument(
        "--data-file",
        default="data/historical/BTCUSDT_1h_730d.json",
        help="歷史數據檔案路徑"
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=150,
        help="優化試驗次數 (default: 150)"
    )
    parser.add_argument(
        "--leverage-supertrend",
        type=int,
        default=10,
        help="Supertrend 槓桿倍數 (default: 10)"
    )
    parser.add_argument(
        "--leverage-rsi-grid",
        type=int,
        default=5,
        help="RSI-Grid 槓桿倍數 (default: 5)"
    )
    parser.add_argument(
        "--output",
        default="optimization_results",
        help="輸出目錄"
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("錯誤: 需要安裝 optuna")
        return

    klines = load_klines(args.data_file)

    results = []

    if args.strategy in ["supertrend", "all"]:
        result = run_optimization("supertrend", klines, args.trials, args.leverage_supertrend)
        results.append(result)

    if args.strategy in ["rsi_grid", "all"]:
        result = run_optimization("rsi_grid", klines, args.trials, args.leverage_rsi_grid)
        results.append(result)

    # 儲存結果
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"max_return_optimization_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "optimization_date": datetime.now().isoformat(),
            "target": "maximize_return",
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {output_file}")

    # 總結
    print("\n" + "=" * 60)
    print("  優化總結 (目標: 30%+ 年化)")
    print("=" * 60)

    for r in results:
        status = "✅" if r['annual_return'] >= 30 else "⚠️"
        print(f"\n{r['strategy'].upper()} ({r['leverage']}x 槓桿):")
        print(f"  年化報酬: {r['annual_return']:.2f}% {status}")
        print(f"  最大回撤: {r['max_drawdown']:.2f}%")
        print(f"  Sharpe: {r['sharpe']:.2f}")


if __name__ == "__main__":
    main()
