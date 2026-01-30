#!/usr/bin/env python3
"""
Walk-Forward Validation Runner.

對所有策略執行 Walk-Forward 驗證，檢測過度擬合。

驗證流程:
1. 獲取足夠長的歷史數據 (180天)
2. 分割成 6 個期間，每期間 70% IS / 30% OOS
3. 計算一致性百分比和 OOS/IS 比率
4. Monte Carlo 模擬驗證穩健性

用法:
    python run_walk_forward_validation.py
    python run_walk_forward_validation.py --strategy rsi
    python run_walk_forward_validation.py --symbol ETHUSDT --days 180
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.result import WalkForwardResult
from src.backtest.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloMethod,
    run_monte_carlo_validation as mc_validate,
)
from src.backtest.strategy.bollinger import BollingerMode
from src.backtest.strategy import (
    BollingerBacktestStrategy,
    BollingerStrategyConfig,
    SupertrendBacktestStrategy,
    SupertrendStrategyConfig,
    GridBacktestStrategy,
    GridStrategyConfig,
    GridFuturesBacktestStrategy,
    GridFuturesStrategyConfig,
    GridDirection,
)
from src.backtest.strategy.supertrend import SupertrendMode
from src.backtest.strategy.rsi_grid import (
    RSIGridBacktestStrategy,
    RSIGridStrategyConfig,
)
from src.core.models import Kline
from src.exchange import ExchangeClient


# 驗證結果資料夾
VALIDATION_OUTPUT_DIR = Path("validation_results")


def load_klines_from_file(filepath: Path) -> list[Kline]:
    """從本地 JSON 檔案載入 K 線數據"""
    from decimal import Decimal

    print(f"正在從檔案載入數據: {filepath}")

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
    if klines:
        print(f"  範圍: {klines[0].open_time.strftime('%Y-%m-%d')} ~ {klines[-1].open_time.strftime('%Y-%m-%d')}")

    return klines


async def fetch_klines(symbol: str, interval: str, days: int, data_file: str = None) -> list[Kline]:
    """獲取歷史 K 線數據 (從檔案或 API)"""

    # 優先使用本地檔案
    if data_file:
        filepath = Path(data_file)
        if filepath.exists():
            return load_klines_from_file(filepath)
        else:
            print(f"警告: 檔案不存在 {filepath}")

    # 檢查預設路徑
    default_file = Path(f"data/historical/{symbol}_{interval}_{days}d.json")
    if default_file.exists():
        return load_klines_from_file(default_file)

    # 從 API 獲取
    print(f"正在從 API 獲取 {symbol} {interval} 數據 ({days} 天)...")
    print(f"  提示: 使用 scripts/download_historical_data.py 下載完整歷史數據")

    client = ExchangeClient(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        testnet=False,
    )

    try:
        await client.connect()

        interval_hours = {
            "1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
            "1h": 1, "2h": 2, "4h": 4, "1d": 24
        }
        hours_per_bar = interval_hours.get(interval, 1)
        limit = min(int(days * 24 / hours_per_bar), 1000)

        klines = await client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        print(f"獲取 {len(klines)} 根 K 線")
        return klines

    finally:
        await client.close()


def create_all_strategies() -> dict:
    """建立所有策略實例"""
    # Grid Futures: 使用優化後的積極策略參數 (2026-01-28)
    # 目標: 年化 40%+, 回撤容忍 30%
    # 優化結果: 年化 45.18%, 回撤 3.79%, Sharpe 10.30
    grid_futures_config = GridFuturesStrategyConfig(
        leverage=7,  # 優化結果 2026-01-31 (equity-based margin)
        direction=GridDirection.NEUTRAL,  # 優化結果: 中性雙向
        grid_count=12,  # 優化結果
        atr_multiplier=Decimal("10.0"),  # 優化結果: 寬範圍
        trend_period=27,  # 優化結果
        atr_period=41,  # 優化結果
        stop_loss_pct=Decimal("0.005"),  # 優化結果: 0.5% 緊止損
        take_profit_grids=1,  # 優化結果
        use_hysteresis=False,  # 優化結果: 關閉
        hysteresis_pct=Decimal("0.005"),  # 優化結果
        use_signal_cooldown=False,  # 優化結果: 關閉
        cooldown_bars=2,  # 優化結果
    )

    # Supertrend: HYBRID_GRID 優化參數 (2026-01-31, equity-based margin)
    supertrend_config = SupertrendStrategyConfig(
        mode=SupertrendMode.HYBRID_GRID,
        atr_period=11,
        atr_multiplier=Decimal("1.5"),
        grid_count=8,
        grid_atr_multiplier=Decimal("7.5"),
        take_profit_grids=1,
        stop_loss_pct=Decimal("0.05"),
        use_rsi_filter=True,
        rsi_period=21,
        rsi_overbought=75,
        rsi_oversold=37,
        min_trend_bars=1,
        use_hysteresis=False,
        hysteresis_pct=Decimal("0.0085"),
        use_signal_cooldown=False,
        cooldown_bars=3,
        use_trailing_stop=True,
        trailing_stop_pct=Decimal("0.01"),
        use_volatility_filter=False,
        vol_ratio_low=0.3,
        vol_ratio_high=3.0,
        max_hold_bars=8,
        hybrid_grid_bias_pct=Decimal("0.65"),
        hybrid_tp_multiplier_trend=Decimal("1.75"),
        hybrid_tp_multiplier_counter=Decimal("0.5"),
        hybrid_sl_multiplier_counter=Decimal("0.9"),
        hybrid_rsi_asymmetric=False,
    )

    return {
        "bollinger": BollingerBacktestStrategy(BollingerStrategyConfig(
            mode=BollingerMode.BB_NEUTRAL_GRID,
            bb_period=31,
            bb_std=Decimal("2.0"),
            grid_count=14,
            take_profit_grids=1,
            stop_loss_pct=Decimal("0.002"),
            use_atr_range=True,
            atr_period=29,
            atr_multiplier=Decimal("9.5"),
            fallback_range_pct=Decimal("0.04"),
            use_hysteresis=True,
            hysteresis_pct=Decimal("0.0025"),
            use_signal_cooldown=False,
            cooldown_bars=1,
        )),
        "supertrend": SupertrendBacktestStrategy(supertrend_config),
        "grid": GridBacktestStrategy(GridStrategyConfig()),
        "rsi_grid": RSIGridBacktestStrategy(RSIGridStrategyConfig(
            rsi_period=5,
            rsi_block_threshold=0.9,
            atr_period=10,
            grid_count=8,
            atr_multiplier=Decimal("4.0"),
            stop_loss_atr_mult=Decimal("2.0"),
            take_profit_grids=2,
            max_hold_bars=8,
            use_trailing_stop=True,
            trailing_activate_pct=0.01,
            trailing_distance_pct=0.003,
            use_volatility_filter=True,
            vol_atr_baseline_period=100,
            vol_ratio_low=0.5,
            vol_ratio_high=2.5,
        )),
        "grid_futures": GridFuturesBacktestStrategy(grid_futures_config),
    }


def run_walk_forward_validation(
    klines: list[Kline],
    strategy,
    strategy_name: str,
    periods: int = 6,
    is_ratio: float = 0.7,
    leverage: int = 10,
) -> dict:
    """執行 Walk-Forward 驗證"""

    print(f"\n{'='*60}")
    print(f"  Walk-Forward 驗證: {strategy_name.upper()}")
    print(f"{'='*60}")

    config = BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0006"),       # 0.06% (aligned with run_backtest)
        slippage_pct=Decimal("0.0005"),   # 0.05% (aligned with run_backtest)
    ).with_leverage(leverage)

    engine = BacktestEngine(config)

    # 執行 Walk-Forward 驗證
    wf_result = engine.run_walk_forward(
        klines=klines,
        strategy=strategy,
        periods=periods,
        is_ratio=is_ratio,
        consistency_threshold=0.5,
    )

    # 收集結果
    result_data = {
        "strategy": strategy_name,
        "validation_date": datetime.now().isoformat(),
        "data_bars": len(klines),
        "periods": periods,
        "is_ratio": is_ratio,
        "walk_forward": {
            "total_periods": wf_result.total_periods,
            "consistent_periods": wf_result.consistent_periods,
            "consistency_pct": float(wf_result.consistency_pct),
            "avg_oos_sharpe": float(wf_result.avg_oos_sharpe),
            "period_details": [],
        },
        "overfitting_analysis": {},
    }

    # 計算 IS 和 OOS 平均值
    is_sharpes = []
    oos_sharpes = []
    is_returns = []
    oos_returns = []
    is_win_rates = []
    oos_win_rates = []

    print(f"\n期間分析:")
    print(f"{'-'*60}")

    for p in wf_result.periods:
        is_sharpes.append(float(p.is_result.sharpe_ratio))
        oos_sharpes.append(float(p.oos_result.sharpe_ratio))
        is_returns.append(float(p.is_result.total_profit_pct))
        oos_returns.append(float(p.oos_result.total_profit_pct))
        is_win_rates.append(float(p.is_result.win_rate))
        oos_win_rates.append(float(p.oos_result.win_rate))

        status = "✅ 一致" if p.is_consistent else "❌ 不一致"

        print(f"\n期間 {p.period_num}:")
        print(f"  IS  期間: {p.is_start.strftime('%Y-%m-%d')} ~ {p.is_end.strftime('%Y-%m-%d')}")
        print(f"  OOS 期間: {p.oos_start.strftime('%Y-%m-%d')} ~ {p.oos_end.strftime('%Y-%m-%d')}")
        print(f"  IS  Sharpe: {p.is_result.sharpe_ratio:>8.2f} | 報酬: {p.is_result.total_profit_pct:>7.2f}% | 勝率: {p.is_result.win_rate:>5.1f}% | 交易: {p.is_result.total_trades:>3}")
        print(f"  OOS Sharpe: {p.oos_result.sharpe_ratio:>8.2f} | 報酬: {p.oos_result.total_profit_pct:>7.2f}% | 勝率: {p.oos_result.win_rate:>5.1f}% | 交易: {p.oos_result.total_trades:>3}")
        print(f"  OOS/IS 比率: {float(p.oos_vs_is_ratio):>6.2f} {status}")

        result_data["walk_forward"]["period_details"].append({
            "period": p.period_num,
            "is_start": p.is_start.isoformat(),
            "is_end": p.is_end.isoformat(),
            "oos_start": p.oos_start.isoformat(),
            "oos_end": p.oos_end.isoformat(),
            "is_sharpe": float(p.is_result.sharpe_ratio),
            "oos_sharpe": float(p.oos_result.sharpe_ratio),
            "is_return_pct": float(p.is_result.total_profit_pct),
            "oos_return_pct": float(p.oos_result.total_profit_pct),
            "is_win_rate": float(p.is_result.win_rate),
            "oos_win_rate": float(p.oos_result.win_rate),
            "is_trades": p.is_result.total_trades,
            "oos_trades": p.oos_result.total_trades,
            "oos_is_ratio": float(p.oos_vs_is_ratio),
            "is_consistent": p.is_consistent,
        })

    # 過度擬合分析
    avg_is_sharpe = sum(is_sharpes) / len(is_sharpes) if is_sharpes else 0
    avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0
    avg_is_return = sum(is_returns) / len(is_returns) if is_returns else 0
    avg_oos_return = sum(oos_returns) / len(oos_returns) if oos_returns else 0

    oos_is_sharpe_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe != 0 else 0
    oos_is_return_ratio = avg_oos_return / avg_is_return if avg_is_return != 0 else 0

    # 判斷過度擬合
    is_overfit = (
        oos_is_sharpe_ratio < 0.5 or  # OOS Sharpe 低於 IS 的 50%
        float(wf_result.consistency_pct) < 50 or  # 一致性低於 50%
        avg_oos_sharpe < 0  # OOS Sharpe 為負
    )

    result_data["overfitting_analysis"] = {
        "avg_is_sharpe": avg_is_sharpe,
        "avg_oos_sharpe": avg_oos_sharpe,
        "avg_is_return_pct": avg_is_return,
        "avg_oos_return_pct": avg_oos_return,
        "oos_is_sharpe_ratio": oos_is_sharpe_ratio,
        "oos_is_return_ratio": oos_is_return_ratio,
        "is_overfit": is_overfit,
        "overfit_severity": "HIGH" if oos_is_sharpe_ratio < 0.3 else ("MEDIUM" if oos_is_sharpe_ratio < 0.5 else "LOW"),
    }

    print(f"\n{'='*60}")
    print(f"  過度擬合分析")
    print(f"{'='*60}")
    print(f"\n平均 IS Sharpe:  {avg_is_sharpe:>8.2f}")
    print(f"平均 OOS Sharpe: {avg_oos_sharpe:>8.2f}")
    print(f"OOS/IS Sharpe 比率: {oos_is_sharpe_ratio:>6.2f}")
    print(f"\n平均 IS 報酬:  {avg_is_return:>8.2f}%")
    print(f"平均 OOS 報酬: {avg_oos_return:>8.2f}%")
    print(f"OOS/IS 報酬比率: {oos_is_return_ratio:>6.2f}")
    print(f"\n一致性百分比: {wf_result.consistency_pct:.1f}% ({wf_result.consistent_periods}/{wf_result.total_periods})")

    if is_overfit:
        print(f"\n⚠️  警告: 檢測到過度擬合風險 (嚴重程度: {result_data['overfitting_analysis']['overfit_severity']})")
    else:
        print(f"\n✅ 通過過度擬合檢測")

    return result_data


def run_monte_carlo_validation_for_strategy(
    klines: list[Kline],
    strategy,
    strategy_name: str,
    n_simulations: int = 500,
    leverage: int = 10,
) -> dict:
    """執行 Monte Carlo 穩健性驗證"""

    print(f"\n{'='*60}")
    print(f"  Monte Carlo 模擬: {strategy_name.upper()}")
    print(f"{'='*60}")

    config = BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0006"),       # 0.06% (aligned with run_backtest)
        slippage_pct=Decimal("0.0005"),   # 0.05% (aligned with run_backtest)
    ).with_leverage(leverage)

    engine = BacktestEngine(config)

    # 先執行基準回測
    base_result = engine.run(klines, strategy)

    if base_result.total_trades < 10:
        print(f"警告: 交易次數過少 ({base_result.total_trades})，無法進行有效的 Monte Carlo 模擬")
        return {
            "strategy": strategy_name,
            "error": "insufficient_trades",
            "total_trades": base_result.total_trades,
        }

    # Monte Carlo 配置
    mc_config = MonteCarloConfig(
        n_simulations=n_simulations,
        confidence_level=0.95,
        seed=42,
    )

    # 使用便利函數執行多種 Monte Carlo 方法
    print(f"\n執行 Monte Carlo 模擬 ({n_simulations} 次)...")
    mc_results = mc_validate(
        result=base_result,
        methods=[MonteCarloMethod.TRADE_SHUFFLE, MonteCarloMethod.RETURNS_BOOTSTRAP],
        config=mc_config,
        initial_capital=Decimal("10000"),
    )

    shuffle_result = mc_results.get(MonteCarloMethod.TRADE_SHUFFLE)
    bootstrap_result = mc_results.get(MonteCarloMethod.RETURNS_BOOTSTRAP)

    result_data = {
        "strategy": strategy_name,
        "base_result": {
            "sharpe_ratio": float(base_result.sharpe_ratio),
            "total_return_pct": float(base_result.total_profit_pct),
            "max_drawdown_pct": float(base_result.max_drawdown_pct),
            "win_rate": float(base_result.win_rate),
            "total_trades": base_result.total_trades,
        },
    }

    if shuffle_result:
        worst_return = shuffle_result.worst_case_metrics.get("total_return", 0)
        result_data["trade_shuffle"] = {
            "probability_of_profit": shuffle_result.probability_of_profit,
            "probability_of_ruin": shuffle_result.probability_of_ruin,
            "worst_case_metrics": shuffle_result.worst_case_metrics,
        }
        print(f"\n交易洗牌結果:")
        print(f"  獲利機率: {shuffle_result.probability_of_profit*100:.1f}%")
        print(f"  破產機率: {shuffle_result.probability_of_ruin*100:.1f}%")
        print(f"  最差情況報酬: {worst_return:.2f}%")

    if bootstrap_result:
        worst_return = bootstrap_result.worst_case_metrics.get("total_return", 0)
        result_data["returns_bootstrap"] = {
            "probability_of_profit": bootstrap_result.probability_of_profit,
            "probability_of_ruin": bootstrap_result.probability_of_ruin,
            "worst_case_metrics": bootstrap_result.worst_case_metrics,
        }
        print(f"\nBootstrap 結果:")
        print(f"  獲利機率: {bootstrap_result.probability_of_profit*100:.1f}%")
        print(f"  破產機率: {bootstrap_result.probability_of_ruin*100:.1f}%")
        print(f"  最差情況報酬: {worst_return:.2f}%")

    # 穩健性判斷
    is_robust = False
    if shuffle_result and bootstrap_result:
        is_robust = (
            shuffle_result.probability_of_profit > 0.6 and
            bootstrap_result.probability_of_profit > 0.6 and
            shuffle_result.probability_of_ruin < 0.1
        )

    result_data["robustness"] = {
        "is_robust": is_robust,
        "assessment": "ROBUST" if is_robust else "FRAGILE",
    }

    if is_robust:
        print(f"\n✅ 策略通過穩健性測試")
    else:
        print(f"\n⚠️  警告: 策略穩健性不足")

    return result_data


def save_validation_results(results: dict, filename: str):
    """儲存驗證結果"""
    VALIDATION_OUTPUT_DIR.mkdir(exist_ok=True)

    filepath = VALIDATION_OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果已儲存至: {filepath}")


def print_summary(all_results: list[dict]):
    """印出所有策略的摘要"""

    print(f"\n\n{'='*80}")
    print(f"  Walk-Forward 驗證總結")
    print(f"{'='*80}")

    print(f"\n{'策略':<15} {'一致性%':<10} {'OOS/IS比':<10} {'OOS Sharpe':<12} {'過度擬合':<10} {'穩健性':<10}")
    print(f"{'-'*80}")

    for r in all_results:
        if "error" in r.get("walk_forward", {}):
            continue

        wf = r.get("walk_forward", {})
        oa = r.get("overfitting_analysis", {})
        mc = r.get("monte_carlo", {})

        consistency = wf.get("consistency_pct", 0)
        oos_is_ratio = oa.get("oos_is_sharpe_ratio", 0)
        avg_oos_sharpe = wf.get("avg_oos_sharpe", 0)
        is_overfit = "❌ YES" if oa.get("is_overfit", True) else "✅ NO"
        robustness = mc.get("robustness", {}).get("assessment", "N/A")

        print(f"{r['strategy']:<15} {consistency:>8.1f}% {oos_is_ratio:>9.2f} {avg_oos_sharpe:>11.2f} {is_overfit:<10} {robustness:<10}")

    print(f"\n{'='*80}")
    print(f"\n判定標準:")
    print(f"  - 一致性 >= 50%: OOS 表現與 IS 一致的期間百分比")
    print(f"  - OOS/IS 比率 >= 0.5: OOS Sharpe 至少為 IS 的 50%")
    print(f"  - 過度擬合: 當一致性 < 50% 或 OOS/IS < 0.5 或 OOS Sharpe < 0")
    print(f"  - 穩健性: Monte Carlo 獲利機率 > 60% 且破產機率 < 10%")


async def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 驗證系統")
    parser.add_argument(
        "--strategy", "-s",
        choices=["all", "bollinger", "supertrend", "grid", "rsi_grid", "grid_futures"],
        default="all",
        help="要驗證的策略 (default: all)"
    )
    parser.add_argument(
        "--symbol", "-p",
        default="BTCUSDT",
        help="交易對 (default: BTCUSDT)"
    )
    parser.add_argument(
        "--interval", "-i",
        default="1h",
        help="K 線週期 (default: 1h)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=42,  # 約 1000 根 1h K 線
        help="歷史數據天數 (default: 42)"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=3,
        help="Walk-Forward 期間數 (default: 3，數據越多可用更多期間)"
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="執行 Monte Carlo 模擬"
    )
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=500,
        help="Monte Carlo 迭代次數 (default: 500)"
    )
    parser.add_argument(
        "--data-file",
        help="本地數據檔案路徑 (優先使用)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Walk-Forward 驗證系統")
    print(f"{'='*60}")
    print(f"\n配置:")
    print(f"  交易對: {args.symbol}")
    print(f"  週期: {args.interval}")
    print(f"  天數: {args.days}")
    print(f"  Walk-Forward 期間: {args.periods}")
    print(f"  Monte Carlo: {'啟用' if args.monte_carlo else '停用'}")

    try:
        # 獲取數據
        klines = await fetch_klines(args.symbol, args.interval, args.days, args.data_file)

        if len(klines) < 100:
            print("錯誤: K 線數據不足")
            return

        # 建立策略
        all_strategies = create_all_strategies()

        if args.strategy == "all":
            strategies_to_test = all_strategies
        else:
            strategies_to_test = {args.strategy: all_strategies[args.strategy]}

        # 策略對應的槓桿倍數
        strategy_leverage = {
            "rsi_grid": 7,
            "grid_futures": 7,
            "bollinger": 7,
            "supertrend": 7,
            "grid": 7,
        }

        all_results = []

        for name, strategy in strategies_to_test.items():
            lev = strategy_leverage.get(name, 10)

            # Walk-Forward 驗證
            wf_result = run_walk_forward_validation(
                klines=klines,
                strategy=strategy,
                strategy_name=name,
                periods=args.periods,
                leverage=lev,
            )

            result = {
                "strategy": name,
                "symbol": args.symbol,
                "interval": args.interval,
                "leverage": lev,
                "walk_forward": wf_result.get("walk_forward", {}),
                "overfitting_analysis": wf_result.get("overfitting_analysis", {}),
            }

            # Monte Carlo 驗證 (如果啟用)
            if args.monte_carlo:
                # 重新建立策略實例 (reset state)
                strategy = create_all_strategies()[name]
                mc_result = run_monte_carlo_validation_for_strategy(
                    klines=klines,
                    strategy=strategy,
                    strategy_name=name,
                    n_simulations=args.mc_iterations,
                    leverage=lev,
                )
                result["monte_carlo"] = mc_result

            all_results.append(result)

        # 印出摘要
        print_summary(all_results)

        # 儲存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{args.symbol}_{timestamp}.json"
        save_validation_results({
            "metadata": {
                "symbol": args.symbol,
                "interval": args.interval,
                "days": args.days,
                "periods": args.periods,
                "data_bars": len(klines),
                "timestamp": datetime.now().isoformat(),
            },
            "results": all_results,
        }, filename)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
