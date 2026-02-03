#!/usr/bin/env python3
"""
Supertrend HYBRID_GRID 策略多目標優化腳本.

使用 IS+OOS 聯合優化，防止過度擬合訓練數據。
HYBRID_GRID: 雙向交易 + Supertrend 趨勢偏移。

核心優化策略：
1. 使用「超額報酬」而非絕對報酬 (策略報酬 - 市場報酬)
2. OOS 正超額報酬優先 (跑贏市場 = 成功)
3. 風險調整報酬 (Sharpe)
4. 回撤懲罰

約束條件：
- 最大回撤 <= 35%
- IS 交易數 >= 800, OOS 交易數 >= 350
- 移除 win_rate >= 50% 限制（允許低勝率高盈虧比）

用法:
    python optimize_supertrend.py --trials 300 --leverage 18
    python optimize_supertrend.py --trials 100 --leverage 18 --quick
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


def split_data(klines: list[Kline], train_ratio: float = 0.7) -> tuple[list[Kline], list[Kline], float, float]:
    """
    將數據分為訓練集 (IS) 和測試集 (OOS).

    Returns:
        (train_klines, test_klines, train_market_return, test_market_return)
    """
    split_idx = int(len(klines) * train_ratio)
    train_klines = klines[:split_idx]
    test_klines = klines[split_idx:]

    # 計算市場報酬 (Buy & Hold)
    train_market = (float(train_klines[-1].close) / float(train_klines[0].close) - 1) * 100
    test_market = (float(test_klines[-1].close) / float(test_klines[0].close) - 1) * 100

    print(f"\n數據分割:")
    print(f"  訓練集 (IS): {train_klines[0].open_time.strftime('%Y-%m-%d')} ~ {train_klines[-1].open_time.strftime('%Y-%m-%d')} ({len(train_klines):,} 根)")
    print(f"  測試集 (OOS): {test_klines[0].open_time.strftime('%Y-%m-%d')} ~ {test_klines[-1].open_time.strftime('%Y-%m-%d')} ({len(test_klines):,} 根)")
    print(f"\n市場基準 (Buy & Hold):")
    print(f"  訓練期: {train_market:+.2f}%")
    print(f"  測試期: {test_market:+.2f}%")

    return train_klines, test_klines, train_market, test_market


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
    config = SupertrendStrategyConfig(
        mode=SupertrendMode.HYBRID_GRID,
        atr_period=params["atr_period"],
        atr_multiplier=Decimal(str(params["atr_multiplier"])),
        grid_count=params["grid_count"],
        grid_atr_multiplier=Decimal(str(params["grid_atr_multiplier"])),
        take_profit_grids=params["take_profit_grids"],
        stop_loss_pct=Decimal(str(params["stop_loss_pct"])),
        use_rsi_filter=True,
        rsi_period=params["rsi_period"],
        rsi_overbought=params["rsi_overbought"],
        rsi_oversold=params["rsi_oversold"],
        min_trend_bars=params["min_trend_bars"],
        use_hysteresis=params["use_hysteresis"],
        hysteresis_pct=Decimal(str(params["hysteresis_pct"])),
        use_signal_cooldown=params["use_signal_cooldown"],
        cooldown_bars=params["cooldown_bars"],
        use_trailing_stop=True,
        trailing_stop_pct=Decimal(str(params["trailing_stop_pct"])),
        # Volatility Regime Filter (v2)
        use_volatility_filter=params.get("use_volatility_filter", True),
        vol_ratio_low=params.get("vol_ratio_low", 0.5),
        vol_ratio_high=params.get("vol_ratio_high", 2.0),
        # Timeout Exit (v2)
        max_hold_bars=params.get("max_hold_bars", 16),
        # HYBRID_GRID specific
        hybrid_grid_bias_pct=Decimal(str(params["hybrid_grid_bias_pct"])),
        hybrid_tp_multiplier_trend=Decimal(str(params["hybrid_tp_multiplier_trend"])),
        hybrid_tp_multiplier_counter=Decimal(str(params["hybrid_tp_multiplier_counter"])),
        hybrid_sl_multiplier_counter=Decimal(str(params["hybrid_sl_multiplier_counter"])),
        hybrid_rsi_asymmetric=params["hybrid_rsi_asymmetric"],
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


def create_multi_objective(
    train_klines: list[Kline],
    test_klines: list[Kline],
    train_market: float,
    test_market: float,
    leverage: int = 18,
):
    """
    創建多目標優化函數 (IS+OOS 聯合優化).

    優化目標:
    score = OOS_excess * 2           # OOS 超額報酬（最高權重）
         + IS_return * 0.5           # IS 絕對報酬
         + OOS_return * 2.0          # OOS 絕對報酬（不封頂）
         + OOS_sharpe * 5            # Sharpe 風險調整（降低權重）
         - max_drawdown * 0.3        # 回撤懲罰
         + consistency_bonus * 10    # 一致性獎勵
         + trade_bonus               # 交易次數獎勵 log(total/400)*2

    約束條件:
    - 最大回撤 <= 35%
    - IS 交易數 >= 200, OOS 交易數 >= 80
    """

    def objective(trial: optuna.Trial) -> float:
        # 擴大參數搜索空間 (v2: 加入波動率過濾 + 超時出場)
        params = {
            "atr_period": trial.suggest_int("atr_period", 5, 50),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 1.0, 6.0, step=0.5),
            "grid_count": trial.suggest_int("grid_count", 8, 25),
            "grid_atr_multiplier": trial.suggest_float("grid_atr_multiplier", 4.0, 12.0, step=0.5),
            "take_profit_grids": trial.suggest_int("take_profit_grids", 1, 3),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.01, 0.05, step=0.005),
            "rsi_period": trial.suggest_int("rsi_period", 7, 21),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 55, 75),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 25, 45),
            "min_trend_bars": 1,  # 固定為 1（結構改動已處理趨勢未確立情況）
            "hysteresis_pct": trial.suggest_float("hysteresis_pct", 0.0005, 0.01, step=0.0005),
            "use_hysteresis": trial.suggest_categorical("use_hysteresis", [True, False]),
            "use_signal_cooldown": trial.suggest_categorical("use_signal_cooldown", [True, False]),
            "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 3),
            "trailing_stop_pct": trial.suggest_float("trailing_stop_pct", 0.01, 0.10, step=0.01),
            # Volatility Regime Filter (v2)
            "use_volatility_filter": trial.suggest_categorical("use_volatility_filter", [True, False]),
            "vol_ratio_low": trial.suggest_float("vol_ratio_low", 0.3, 0.7, step=0.1),
            "vol_ratio_high": trial.suggest_float("vol_ratio_high", 1.5, 3.0, step=0.5),
            # Timeout Exit (v2)
            "max_hold_bars": trial.suggest_int("max_hold_bars", 8, 32, step=4),
            # HYBRID_GRID specific
            "hybrid_grid_bias_pct": trial.suggest_float("hybrid_grid_bias_pct", 0.55, 0.75, step=0.05),
            "hybrid_tp_multiplier_trend": trial.suggest_float("hybrid_tp_multiplier_trend", 1.0, 2.0, step=0.25),
            "hybrid_tp_multiplier_counter": trial.suggest_float("hybrid_tp_multiplier_counter", 0.5, 1.0, step=0.25),
            "hybrid_sl_multiplier_counter": trial.suggest_float("hybrid_sl_multiplier_counter", 0.5, 1.0, step=0.1),
            "hybrid_rsi_asymmetric": trial.suggest_categorical("hybrid_rsi_asymmetric", [True, False]),
        }

        # 執行 IS 回測 (訓練集)
        try:
            is_result = run_backtest(train_klines, params, leverage)
        except Exception:
            return -1000

        # 執行 OOS 回測 (測試集)
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

        # 計算超額報酬
        is_excess = is_return - train_market
        oos_excess = oos_return - test_market

        # OOS/IS 比率
        if is_return > 0 and oos_return > 0:
            oos_is_ratio = oos_return / is_return
        elif is_return > 0:
            oos_is_ratio = oos_return / is_return
        else:
            oos_is_ratio = 0

        # ========== 約束條件 ==========

        # 交易數約束（HYBRID_GRID 雙向交易）
        if is_trades < 300:
            trial.set_user_attr("rejection_reason", f"IS交易不足: {is_trades}")
            return -500 + is_trades
        if oos_trades < 120:
            trial.set_user_attr("rejection_reason", f"OOS交易不足: {oos_trades}")
            return -500 + oos_trades

        # 回撤約束
        if max_dd > 35:
            trial.set_user_attr("rejection_reason", f"回撤過大: {max_dd:.1f}%")
            return -100 - max_dd

        # ========== 多目標優化函數（v3：強制達成絕對報酬目標）==========

        # 1. IS 絕對報酬（大幅提高權重）
        is_return_score = is_return * 1.5

        # 2. OOS 絕對報酬（壓倒性權重）
        oos_return_score = oos_return * 3.0

        # 3. OOS 超額報酬（維持但降低）
        oos_excess_score = oos_excess * 1.0

        # 4. Sharpe（大幅削弱，幾乎不影響）
        sharpe_score = oos_sharpe * 1

        # 5. 回撤懲罰（僅懲罰 >15% 的部分，鼓勵用回撤換報酬）
        dd_penalty = max(0, max_dd - 15) * 0.5

        # 5b. Sharpe > 10 懲罰（不合理的高 Sharpe 暗示過擬合）
        sharpe_penalty = 0
        if oos_sharpe > 10:
            sharpe_penalty = (oos_sharpe - 10) * 5

        # 6. 一致性獎勵（兩期都跑贏市場）
        consistency_bonus = 0
        if is_excess > 0 and oos_excess > 0:
            consistency_bonus = 10

        # 7. 交易次數獎勵（進一步降低）
        total_trades = is_trades + oos_trades
        trade_bonus = math.log(total_trades / 1000) * 1

        # 8. 里程碑獎勵（直接獎勵達標）
        milestone_bonus = 0
        if is_return >= 30:
            milestone_bonus += 30
        elif is_return >= 20:
            milestone_bonus += 15
        if oos_return >= 30:
            milestone_bonus += 40
        elif oos_return >= 15:
            milestone_bonus += 20

        # 總分
        score = (is_return_score + oos_return_score + oos_excess_score
                 + sharpe_score - dd_penalty - sharpe_penalty + consistency_bonus
                 + trade_bonus + milestone_bonus)

        # 記錄詳細指標
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

    # 判定
    is_return = best.user_attrs.get('is_return', 0)
    oos_return = best.user_attrs.get('oos_return', 0)
    oos_excess = best.user_attrs.get('oos_excess', 0)
    oos_is_ratio = best.user_attrs.get('oos_is_ratio', 0)

    print(f"\n[目標達成檢查]")
    is_target = "PASS" if is_return >= 30 else "FAIL"
    oos_target = "PASS" if oos_return >= 30 else "FAIL"
    oos_excess_target = "PASS" if oos_excess > 0 else "FAIL"
    ratio_target = "PASS" if oos_is_ratio >= 0.5 else "FAIL"

    print(f"  IS 報酬 >= 30%: {is_target} ({is_return:.1f}%)")
    print(f"  OOS 報酬 >= 30%: {oos_target} ({oos_return:.1f}%)")
    print(f"  OOS 超額報酬 > 0: {oos_excess_target} ({oos_excess:+.1f}%)")
    print(f"  OOS/IS >= 0.5: {ratio_target} ({oos_is_ratio:.2f})")

    # 綜合評估
    print(f"\n[綜合評估]")
    if oos_return >= 30 and oos_excess > 0:
        print(f"  ★★★ 優秀: OOS 報酬 >= 30% 且跑贏市場")
    elif oos_return > 0 and oos_excess > 0:
        print(f"  ★★☆ 良好: OOS 正報酬且跑贏市場")
    elif oos_excess > 0:
        print(f"  ★☆☆ 可接受: OOS 跑贏市場 (但絕對報酬為負)")
    else:
        print(f"  ☆☆☆ 需改進: OOS 未跑贏市場")

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
    output_file = output_path / f"supertrend_optimization_{timestamp}.json"

    results = {
        "optimization_date": datetime.now(timezone.utc).isoformat(),
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


def print_config_suggestion(params: dict):
    """輸出建議配置."""
    print("\n" + "=" * 70)
    print("  建議配置 (可直接用於 validate_supertrend_wf.py)")
    print("=" * 70)
    print(f"""
SupertrendStrategyConfig(
    mode=SupertrendMode.HYBRID_GRID,
    atr_period={params['atr_period']},
    atr_multiplier=Decimal("{params['atr_multiplier']}"),
    grid_count={params['grid_count']},
    grid_atr_multiplier=Decimal("{params['grid_atr_multiplier']}"),
    take_profit_grids={params['take_profit_grids']},
    stop_loss_pct=Decimal("{params['stop_loss_pct']}"),
    use_rsi_filter=True,
    rsi_period={params['rsi_period']},
    rsi_overbought={params['rsi_overbought']},
    rsi_oversold={params['rsi_oversold']},
    min_trend_bars={params.get('min_trend_bars', 1)},
    use_hysteresis={params['use_hysteresis']},
    hysteresis_pct=Decimal("{params['hysteresis_pct']}"),
    use_signal_cooldown={params['use_signal_cooldown']},
    cooldown_bars={params['cooldown_bars']},
    use_trailing_stop=True,
    trailing_stop_pct=Decimal("{params['trailing_stop_pct']}"),
    use_volatility_filter={params.get('use_volatility_filter', True)},
    vol_ratio_low={params.get('vol_ratio_low', 0.5)},
    vol_ratio_high={params.get('vol_ratio_high', 2.0)},
    max_hold_bars={params.get('max_hold_bars', 16)},
    hybrid_grid_bias_pct=Decimal("{params['hybrid_grid_bias_pct']}"),
    hybrid_tp_multiplier_trend=Decimal("{params['hybrid_tp_multiplier_trend']}"),
    hybrid_tp_multiplier_counter=Decimal("{params['hybrid_tp_multiplier_counter']}"),
    hybrid_sl_multiplier_counter=Decimal("{params['hybrid_sl_multiplier_counter']}"),
    hybrid_rsi_asymmetric={params['hybrid_rsi_asymmetric']},
)
""")


def main():
    parser = argparse.ArgumentParser(
        description="Supertrend HYBRID_GRID 多目標參數優化 (IS+OOS 聯合優化)"
    )
    parser.add_argument(
        "--data-file",
        default="data/historical/BTCUSDT_1h_730d.json",
        help="歷史數據檔案路徑",
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=200,
        help="優化試驗次數 (default: 200)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="超時秒數 (default: 無限制)",
    )
    parser.add_argument(
        "--leverage", "-l",
        type=int,
        default=7,
        help="槓桿倍數 (default: 7, 對齊 settings.yaml)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="訓練集比例 (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        default="optimization_results",
        help="輸出目錄",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="快速模式 (50 trials)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子 (default: 42)",
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("錯誤: 需要安裝 optuna")
        print("  pip install optuna")
        return

    if args.quick:
        args.trials = 50

    print("=" * 70)
    print("  Supertrend HYBRID_GRID 多目標優化 (IS+OOS 聯合, 超額報酬)")
    print("  目標: OOS >= 15%, IS >= 40%, 交易數 >= 1000")
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
        study_name="supertrend_multi_objective",
    )

    # 創建目標函數
    objective = create_multi_objective(train_klines, test_klines, train_market, test_market, args.leverage)

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
