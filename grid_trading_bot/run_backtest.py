#!/usr/bin/env python3
"""
Unified Backtest Runner.

使用統一回測系統測試任一交易策略。

用法:
    python run_backtest.py --strategy bollinger --symbol BTCUSDT --days 30
    python run_backtest.py --strategy supertrend --symbol ETHUSDT --days 60
    python run_backtest.py --strategy grid --symbol BTCUSDT --days 30
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.strategy import (
    BollingerBacktestStrategy,
    BollingerStrategyConfig,
    SupertrendBacktestStrategy,
    SupertrendStrategyConfig,
    GridBacktestStrategy,
    GridStrategyConfig,
)
from src.core.models import Kline
from src.exchange import ExchangeClient


async def fetch_klines(
    symbol: str,
    interval: str,
    days: int,
) -> list[Kline]:
    """從交易所獲取歷史 K 線數據"""
    print(f"正在獲取 {symbol} {interval} 數據 ({days} 天)...")

    client = ExchangeClient(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        testnet=False,
    )

    try:
        await client.connect()

        # 計算需要的 K 線數量
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


def create_strategy(strategy_name: str, params: dict):
    """根據名稱創建策略實例"""

    if strategy_name == "bollinger":
        config = BollingerStrategyConfig(
            bb_period=params.get("bb_period", 20),
            bb_std=Decimal(str(params.get("bb_std", "3.0"))),
            st_atr_period=params.get("st_atr_period", 20),
            st_atr_multiplier=Decimal(str(params.get("st_atr_multiplier", "3.5"))),
            atr_stop_multiplier=Decimal(str(params.get("atr_stop_multiplier", "2.0"))),
        )
        return BollingerBacktestStrategy(config)

    elif strategy_name == "supertrend":
        config = SupertrendStrategyConfig(
            atr_period=params.get("atr_period", 14),
            atr_multiplier=Decimal(str(params.get("atr_multiplier", "3.0"))),
            use_rsi_filter=params.get("use_rsi_filter", False),
        )
        return SupertrendBacktestStrategy(config)

    elif strategy_name == "grid":
        # Grid 策略需要根據當前價格動態設定範圍
        # 預設使用 ±10% 的價格區間
        config = GridStrategyConfig(
            grid_count=params.get("grid_count", 10),
            use_geometric=params.get("use_geometric", True),
            take_profit_grids=params.get("take_profit_grids", 1),
            stop_loss_pct=Decimal(str(params.get("stop_loss_pct", "0.02"))),
        )
        return GridBacktestStrategy(config)

    else:
        raise ValueError(f"未知策略: {strategy_name}")


def print_result(result, strategy_name: str, symbol: str):
    """格式化輸出回測結果"""

    print("\n" + "=" * 60)
    print(f"  回測結果 - {strategy_name.upper()} on {symbol}")
    print("=" * 60)

    print(f"\n📊 績效摘要:")
    print(f"   總損益:        {float(result.total_profit):>12.2f} USDT")
    print(f"   報酬率:        {float(result.total_profit_pct):>12.2f}%")
    print(f"   最大回撤:      {float(result.max_drawdown_pct):>12.2f}%")
    print(f"   Sharpe Ratio:  {float(result.sharpe_ratio):>12.2f}")

    print(f"\n📈 交易統計:")
    print(f"   總交易次數:    {result.total_trades:>12}")
    print(f"   勝率:          {float(result.win_rate):>12.2f}%")
    print(f"   獲利因子:      {float(result.profit_factor):>12.2f}")
    print(f"   平均獲利:      {float(result.avg_win):>12.2f} USDT")
    print(f"   平均虧損:      {float(result.avg_loss):>12.2f} USDT")

    print(f"\n📉 風險指標:")
    print(f"   勝場 / 敗場:   {result.num_wins:>5} / {result.num_losses}")
    print(f"   最大回撤金額:  {float(result.max_drawdown):>12.2f} USDT")

    print("\n" + "=" * 60)

    # 顯示最近交易
    if result.trades:
        print(f"\n最近 5 筆交易:")
        for trade in result.trades[-5:]:
            pnl_sign = "+" if trade.pnl >= 0 else ""
            print(f"   {trade.side:5} | 入場: {float(trade.entry_price):.2f} | "
                  f"出場: {float(trade.exit_price):.2f} | "
                  f"損益: {pnl_sign}{float(trade.pnl):.2f}")


async def main():
    parser = argparse.ArgumentParser(description="統一回測系統")
    parser.add_argument(
        "--strategy", "-s",
        choices=["bollinger", "supertrend", "grid"],
        default="bollinger",
        help="策略名稱 (default: bollinger)"
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
        default=30,
        help="回測天數 (default: 30)"
    )
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=10000,
        help="初始資金 (default: 10000)"
    )
    parser.add_argument(
        "--leverage", "-l",
        type=int,
        default=10,
        help="槓桿倍數 (default: 10)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("         統一回測系統 - Unified Backtest Runner")
    print("=" * 60)
    print(f"\n配置:")
    print(f"   策略:     {args.strategy}")
    print(f"   交易對:   {args.symbol}")
    print(f"   週期:     {args.interval}")
    print(f"   天數:     {args.days}")
    print(f"   資金:     {args.capital} USDT")
    print(f"   槓桿:     {args.leverage}x")

    try:
        # 1. 獲取數據
        klines = await fetch_klines(args.symbol, args.interval, args.days)

        if len(klines) < 50:
            print("錯誤: K 線數據不足 (需要至少 50 根)")
            return

        # 2. 創建策略
        strategy = create_strategy(args.strategy, {})
        print(f"策略 {args.strategy} 創建完成")

        # 3. 配置回測引擎
        config = BacktestConfig(
            initial_capital=Decimal(str(args.capital)),
            leverage=args.leverage,
            fee_rate=Decimal("0.0004"),  # 0.04% taker fee
            slippage_pct=Decimal("0.0001"),  # 0.01% slippage
        )

        engine = BacktestEngine(config)

        # 4. 執行回測
        print("\n正在執行回測...")
        result = engine.run(klines, strategy)

        # 5. 輸出結果
        print_result(result, args.strategy, args.symbol)

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
