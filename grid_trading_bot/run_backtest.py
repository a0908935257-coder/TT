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
    BollingerMode,
    SupertrendBacktestStrategy,
    SupertrendStrategyConfig,
    GridBacktestStrategy,
    GridStrategyConfig,
    GridFuturesBacktestStrategy,
    GridFuturesStrategyConfig,
    GridDirection,
)
from src.backtest.strategy.rsi_grid import RSIGridBacktestStrategy, RSIGridStrategyConfig
from src.config import load_strategy_config
from src.core.models import Kline
from src.exchange import ExchangeClient


async def fetch_klines(
    symbol: str,
    interval: str,
    days: int,
) -> list[Kline]:
    """從交易所分批獲取歷史 K 線數據（突破 1000 根限制）"""
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
        total_needed = int(days * 24 / hours_per_bar)

        if total_needed <= 1000:
            klines = await client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=total_needed,
            )
            print(f"獲取 {len(klines)} 根 K 線")
            return klines

        # 分批取得：從 start_time 往後每次取 1000 根
        interval_ms = int(hours_per_bar * 3600 * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - int(days * 24 * 3600 * 1000)

        all_klines: list[Kline] = []
        batch_start = start_ms
        batch_num = 0

        while batch_start < now_ms:
            batch_num += 1
            batch = await client.spot.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=batch_start,
            )
            if not batch:
                break

            all_klines.extend(batch)
            # 下一批從最後一根 K 線的下一個 interval 開始
            last_open_ms = int(batch[-1].open_time.timestamp() * 1000)
            batch_start = last_open_ms + interval_ms
            print(f"  批次 {batch_num}: +{len(batch)} 根 (累計 {len(all_klines)})")

            if len(batch) < 1000:
                break

        # 去重（按 open_time）
        seen = set()
        unique_klines = []
        for k in all_klines:
            ot = k.open_time
            if ot not in seen:
                seen.add(ot)
                unique_klines.append(k)

        print(f"獲取 {len(unique_klines)} 根 K 線 (共 {batch_num} 批)")
        return unique_klines

    finally:
        await client.close()


def create_strategy(strategy_name: str, params: dict):
    """根據名稱創建策略實例"""

    if strategy_name == "bollinger":
        # BB_TREND_GRID 模式 (驗證通過)
        config = BollingerStrategyConfig(
            mode=BollingerMode.BB_TREND_GRID,
            bb_period=params.get("bb_period", 12),
            bb_std=Decimal(str(params.get("bb_std", "2.0"))),
            grid_count=params.get("grid_count", 6),
            grid_range_pct=Decimal(str(params.get("grid_range_pct", "0.02"))),
            take_profit_grids=params.get("take_profit_grids", 2),
            stop_loss_pct=Decimal(str(params.get("stop_loss_pct", "0.025"))),
        )
        return BollingerBacktestStrategy(config)

    elif strategy_name == "bollinger_neutral":
        # BB_NEUTRAL_GRID 模式 (新增 - 待優化)
        config = BollingerStrategyConfig(
            mode=BollingerMode.BB_NEUTRAL_GRID,
            bb_period=params.get("bb_period", 20),
            bb_std=Decimal(str(params.get("bb_std", "2.0"))),
            grid_count=params.get("grid_count", 12),
            take_profit_grids=params.get("take_profit_grids", 1),
            stop_loss_pct=Decimal(str(params.get("stop_loss_pct", "0.005"))),  # 0.5% tight SL
            # ATR dynamic range
            use_atr_range=True,
            atr_period=params.get("atr_period", 21),
            atr_multiplier=Decimal(str(params.get("atr_multiplier", "6.0"))),
            fallback_range_pct=Decimal(str(params.get("fallback_range_pct", "0.04"))),
            # Protective features
            use_hysteresis=params.get("use_hysteresis", True),
            hysteresis_pct=Decimal(str(params.get("hysteresis_pct", "0.002"))),
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
        # Grid 策略 (現貨)
        config = GridStrategyConfig(
            grid_count=params.get("grid_count", 10),
            use_geometric=params.get("use_geometric", True),
            take_profit_grids=params.get("take_profit_grids", 1),
            stop_loss_pct=Decimal(str(params.get("stop_loss_pct", "0.02"))),
        )
        return GridBacktestStrategy(config)

    elif strategy_name == "grid_futures":
        # Grid Futures 策略 (合約)
        config = GridFuturesStrategyConfig(
            grid_count=params.get("grid_count", 10),
            direction=GridDirection.TREND_FOLLOW,
            leverage=params.get("leverage", 2),
            trend_period=params.get("trend_period", 20),
            atr_multiplier=Decimal(str(params.get("atr_multiplier", "3.0"))),
            stop_loss_pct=Decimal(str(params.get("stop_loss_pct", "0.05"))),
        )
        return GridFuturesBacktestStrategy(config)

    elif strategy_name == "rsi":
        # RSI-Grid 策略 (從 settings.yaml 讀取實戰參數)
        yaml_params = load_strategy_config("rsi_grid")
        config = RSIGridStrategyConfig(
            rsi_period=yaml_params.get("rsi_period", 14),
            oversold_level=yaml_params.get("oversold_level", 33),
            overbought_level=yaml_params.get("overbought_level", 66),
            grid_count=yaml_params.get("grid_count", 8),
            atr_period=yaml_params.get("atr_period", 22),
            atr_multiplier=Decimal(str(yaml_params.get("atr_multiplier", "3.5"))),
            trend_sma_period=yaml_params.get("trend_sma_period", 39),
            use_trend_filter=yaml_params.get("use_trend_filter", False),
            position_size_pct=Decimal(str(yaml_params.get("position_size_pct", "0.1"))),
            stop_loss_atr_mult=Decimal(str(yaml_params.get("stop_loss_atr_mult", "2.0"))),
            max_stop_loss_pct=Decimal(str(yaml_params.get("max_stop_loss_pct", "0.03"))),
            take_profit_grids=yaml_params.get("take_profit_grids", 1),
            max_positions=yaml_params.get("max_positions", 5),
            use_hysteresis=yaml_params.get("use_hysteresis", False),
            hysteresis_pct=Decimal(str(yaml_params.get("hysteresis_pct", "0.003"))),
            use_signal_cooldown=yaml_params.get("use_signal_cooldown", False),
            cooldown_bars=yaml_params.get("cooldown_bars", 2),
        )
        return RSIGridBacktestStrategy(config)

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
        choices=["bollinger", "bollinger_neutral", "supertrend", "grid", "rsi", "grid_futures"],
        default="bollinger",
        help="策略名稱: bollinger, bollinger_neutral, supertrend, grid, rsi, grid_futures (default: bollinger)"
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

    # RSI 策略：用 yaml leverage (2) 覆蓋 CLI 預設 (10)
    if args.strategy == "rsi" and args.leverage == 10:
        yaml_params = load_strategy_config("rsi_grid")
        args.leverage = yaml_params.get("leverage", 2)

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
