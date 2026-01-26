#!/usr/bin/env python3
"""
Compare RSI Grid backtest results: Original vs Live-like strategy.

This script compares:
1. Original backtest strategy (no hysteresis, no cooldown)
2. Live-like strategy (with hysteresis and signal cooldown)

Purpose: Quantify the impact of live bot's protective features on backtest results.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.strategy.rsi_grid import (
    RSIGridBacktestStrategy,
    RSIGridStrategyConfig,
)
from src.core.models import Kline
from src.exchange.binance.futures_api import BinanceFuturesAPI


async def fetch_historical_klines(
    symbol: str,
    interval: str,
    days: int,
) -> list[Kline]:
    """Fetch historical klines from Binance."""
    api = BinanceFuturesAPI()

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        # Convert datetime to timestamp (milliseconds)
        start_ts = int(current_start.timestamp() * 1000)

        klines = await api.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_ts,
            limit=1000,
        )

        if not klines:
            break

        all_klines.extend(klines)
        current_start = klines[-1].close_time + timedelta(milliseconds=1)

        # Progress
        progress = (current_start - start_time) / (end_time - start_time) * 100
        print(f"\rFetching data: {progress:.1f}%", end="", flush=True)

    await api.close()
    print(f"\nFetched {len(all_klines)} klines")
    return all_klines


def run_backtest(klines: list[Kline], config: RSIGridStrategyConfig, name: str) -> dict:
    """Run backtest with given strategy config."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    strategy = RSIGridBacktestStrategy(config)

    engine_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0004"),      # 0.04% taker fee
        slippage_pct=Decimal("0.0001"),  # 0.01% slippage
    )
    engine = BacktestEngine(config=engine_config)

    result = engine.run(klines, strategy)

    # Extract metrics
    metrics = {
        "name": name,
        "total_trades": result.total_trades,
        "win_rate": float(result.win_rate),
        "total_return_pct": float(result.total_profit_pct),
        "sharpe_ratio": float(result.sharpe_ratio),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "profit_factor": float(result.profit_factor),
        "avg_win": float(result.avg_win) if result.avg_win else 0,
        "num_wins": result.num_wins,
        "num_losses": result.num_losses,
    }

    # Print results
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    return metrics


def compare_results(original: dict, live_like: dict) -> None:
    """Compare and display the difference between two strategy results."""
    print("\n" + "="*80)
    print("COMPARISON: Original vs Live-Like Strategy")
    print("="*80)

    metrics_to_compare = [
        ("Total Trades", "total_trades", "d"),
        ("Win Rate (%)", "win_rate", ".2f"),
        ("Total Return (%)", "total_return_pct", ".2f"),
        ("Sharpe Ratio", "sharpe_ratio", ".2f"),
        ("Max Drawdown (%)", "max_drawdown_pct", ".2f"),
        ("Profit Factor", "profit_factor", ".2f"),
    ]

    print(f"\n{'Metric':<25} {'Original':>15} {'Live-Like':>15} {'Difference':>15}")
    print("-" * 70)

    for label, key, fmt in metrics_to_compare:
        orig_val = original[key]
        live_val = live_like[key]

        if isinstance(orig_val, int):
            diff = live_val - orig_val
            diff_str = f"{diff:+d}"
        else:
            diff = live_val - orig_val
            diff_str = f"{diff:+.2f}"

        print(f"{label:<25} {format(orig_val, fmt):>15} {format(live_val, fmt):>15} {diff_str:>15}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    trade_reduction = (1 - live_like["total_trades"] / original["total_trades"]) * 100 if original["total_trades"] > 0 else 0
    print(f"\n• Trade Reduction: {trade_reduction:.1f}% (from {original['total_trades']} to {live_like['total_trades']})")

    if live_like["win_rate"] > original["win_rate"]:
        print(f"• Win Rate: IMPROVED by {live_like['win_rate'] - original['win_rate']:.2f}%")
    else:
        print(f"• Win Rate: DECREASED by {original['win_rate'] - live_like['win_rate']:.2f}%")

    if live_like["sharpe_ratio"] > original["sharpe_ratio"]:
        print(f"• Sharpe Ratio: IMPROVED by {live_like['sharpe_ratio'] - original['sharpe_ratio']:.2f}")
    else:
        print(f"• Sharpe Ratio: DECREASED by {original['sharpe_ratio'] - live_like['sharpe_ratio']:.2f}")

    if live_like["max_drawdown_pct"] < original["max_drawdown_pct"]:
        print(f"• Max Drawdown: IMPROVED (lower is better) by {original['max_drawdown_pct'] - live_like['max_drawdown_pct']:.2f}%")
    else:
        print(f"• Max Drawdown: WORSENED by {live_like['max_drawdown_pct'] - original['max_drawdown_pct']:.2f}%")

    # Overall assessment
    print("\n" + "-"*80)
    improvements = 0
    if live_like["win_rate"] >= original["win_rate"]:
        improvements += 1
    if live_like["sharpe_ratio"] >= original["sharpe_ratio"]:
        improvements += 1
    if live_like["max_drawdown_pct"] <= original["max_drawdown_pct"]:
        improvements += 1
    if live_like["profit_factor"] >= original["profit_factor"]:
        improvements += 1

    if improvements >= 3:
        print("Overall: Live-like strategy shows BETTER risk-adjusted performance")
        print("The protective features (hysteresis + cooldown) IMPROVE the strategy")
    elif improvements >= 2:
        print("Overall: Live-like strategy shows SIMILAR performance with fewer trades")
        print("The protective features provide comparable returns with less risk")
    else:
        print("Overall: Live-like strategy shows REDUCED performance")
        print("The protective features may be too conservative for this market regime")


async def main():
    """Main entry point."""
    print("="*80)
    print("RSI Grid Strategy Comparison: Original vs Live-Like")
    print("="*80)
    print("\nFetching historical data...")

    # Parameters
    symbol = "BTCUSDT"
    interval = "1h"
    days = 730  # 2 years

    # Fetch data
    klines = await fetch_historical_klines(symbol, interval, days)

    if len(klines) < 1000:
        print(f"Error: Not enough data ({len(klines)} klines)")
        return

    # Strategy 1: Original (no hysteresis, no cooldown)
    original_config = RSIGridStrategyConfig(
        rsi_period=14,
        oversold_level=30,
        overbought_level=70,
        grid_count=10,
        atr_period=14,
        atr_multiplier=Decimal("3.0"),
        trend_sma_period=20,
        use_trend_filter=True,
        stop_loss_atr_mult=Decimal("1.5"),
        max_stop_loss_pct=Decimal("0.03"),
        take_profit_grids=1,
        max_positions=5,
        # Original: NO hysteresis, NO cooldown
        use_hysteresis=False,
        use_signal_cooldown=False,
    )

    original_results = run_backtest(klines, original_config, "Original Strategy (No Protective Features)")

    # Strategy 2: Live-like (with hysteresis and cooldown)
    live_like_config = RSIGridStrategyConfig(
        rsi_period=14,
        oversold_level=30,
        overbought_level=70,
        grid_count=10,
        atr_period=14,
        atr_multiplier=Decimal("3.0"),
        trend_sma_period=20,
        use_trend_filter=True,
        stop_loss_atr_mult=Decimal("1.5"),
        max_stop_loss_pct=Decimal("0.03"),
        take_profit_grids=1,
        max_positions=5,
        # Live-like: WITH hysteresis and cooldown (same as live bot)
        use_hysteresis=True,
        hysteresis_pct=Decimal("0.002"),  # 0.2% buffer zone
        use_signal_cooldown=True,
        cooldown_bars=2,
    )

    live_like_results = run_backtest(klines, live_like_config, "Live-Like Strategy (With Hysteresis + Cooldown)")

    # Compare results
    compare_results(original_results, live_like_results)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
