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
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

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


def _load_klines_from_file(filepath: str) -> list[Kline]:
    """從本地 JSON 檔案載入 K 線數據."""
    print(f"從檔案載入數據: {filepath}")
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


def create_strategy(strategy_name: str, params: dict):
    """根據名稱創建策略實例（從 settings.yaml 讀取基礎參數，params 可覆蓋）"""

    # 策略名稱 -> YAML bot_type 映射
    strategy_yaml_key = {
        "bollinger": "bollinger",
        "bollinger_neutral": "bollinger",
        "supertrend": "supertrend",
        "grid_futures": "grid_futures",
        "rsi": "rsi_grid",
    }

    # 從 YAML 載入基礎參數，CLI params 覆蓋
    if strategy_name in strategy_yaml_key:
        p = load_strategy_config(strategy_yaml_key[strategy_name])
        p.update(params)  # CLI 參數覆蓋 YAML
    else:
        p = params  # grid (spot) 無 YAML 定義

    if strategy_name == "bollinger":
        # BB_TREND_GRID 模式：強制覆蓋 mode
        config = BollingerStrategyConfig(
            mode=BollingerMode.BB_TREND_GRID,
            bb_period=p.get("bb_period", 12),
            bb_std=Decimal(str(p.get("bb_std", "2.0"))),
            grid_count=p.get("grid_count", 6),
            grid_range_pct=Decimal(str(p.get("grid_range_pct", "0.02"))),
            take_profit_grids=p.get("take_profit_grids", 2),
            stop_loss_pct=Decimal(str(p.get("stop_loss_pct", "0.025"))),
        )
        return BollingerBacktestStrategy(config)

    elif strategy_name == "bollinger_neutral":
        # BB_NEUTRAL_GRID 模式：強制覆蓋 mode
        config = BollingerStrategyConfig(
            mode=BollingerMode.BB_NEUTRAL_GRID,
            bb_period=p.get("bb_period", 20),
            bb_std=Decimal(str(p.get("bb_std", "2.0"))),
            grid_count=p.get("grid_count", 12),
            take_profit_grids=p.get("take_profit_grids", 1),
            stop_loss_pct=Decimal(str(p.get("stop_loss_pct", "0.005"))),
            use_atr_range=p.get("use_atr_range", True),
            atr_period=p.get("atr_period", 21),
            atr_multiplier=Decimal(str(p.get("atr_multiplier", "6.0"))),
            fallback_range_pct=Decimal(str(p.get("fallback_range_pct", "0.04"))),
            use_hysteresis=p.get("use_hysteresis", True),
            hysteresis_pct=Decimal(str(p.get("hysteresis_pct", "0.002"))),
        )
        return BollingerBacktestStrategy(config)

    elif strategy_name == "supertrend":
        from src.backtest.strategy.supertrend import SupertrendMode
        config = SupertrendStrategyConfig(
            mode=SupertrendMode(p.get("mode", "hybrid_grid")),
            atr_period=p.get("atr_period", 14),
            atr_multiplier=Decimal(str(p.get("atr_multiplier", "3.0"))),
            grid_count=p.get("grid_count", 8),
            grid_atr_multiplier=Decimal(str(p.get("grid_atr_multiplier", "7.5"))),
            take_profit_grids=p.get("take_profit_grids", 1),
            stop_loss_pct=Decimal(str(p.get("stop_loss_pct", "0.05"))),
            use_rsi_filter=p.get("use_rsi_filter", True),
            rsi_period=p.get("rsi_period", 21),
            rsi_overbought=p.get("rsi_overbought", 75),
            rsi_oversold=p.get("rsi_oversold", 37),
            min_trend_bars=p.get("min_trend_bars", 1),
            use_hysteresis=p.get("use_hysteresis", False),
            hysteresis_pct=Decimal(str(p.get("hysteresis_pct", "0.0085"))),
            use_signal_cooldown=p.get("use_signal_cooldown", False),
            cooldown_bars=p.get("cooldown_bars", 3),
            use_trailing_stop=p.get("use_trailing_stop", True),
            trailing_stop_pct=Decimal(str(p.get("trailing_stop_pct", "0.01"))),
            use_volatility_filter=p.get("use_volatility_filter", False),
            vol_ratio_low=p.get("vol_ratio_low", 0.3),
            vol_ratio_high=p.get("vol_ratio_high", 3.0),
            max_hold_bars=p.get("max_hold_bars", 8),
            hybrid_grid_bias_pct=Decimal(str(p.get("hybrid_grid_bias_pct", "0.65"))),
            hybrid_tp_multiplier_trend=Decimal(str(p.get("hybrid_tp_multiplier_trend", "1.75"))),
            hybrid_tp_multiplier_counter=Decimal(str(p.get("hybrid_tp_multiplier_counter", "0.5"))),
            hybrid_sl_multiplier_counter=Decimal(str(p.get("hybrid_sl_multiplier_counter", "0.9"))),
            hybrid_rsi_asymmetric=p.get("hybrid_rsi_asymmetric", False),
        )
        return SupertrendBacktestStrategy(config)

    elif strategy_name == "grid":
        # Grid 策略 (現貨) — 無 YAML 定義
        config = GridStrategyConfig(
            grid_count=p.get("grid_count", 10),
            use_geometric=p.get("use_geometric", True),
            take_profit_grids=p.get("take_profit_grids", 1),
            stop_loss_pct=Decimal(str(p.get("stop_loss_pct", "0.02"))),
        )
        return GridBacktestStrategy(config)

    elif strategy_name == "grid_futures":
        config = GridFuturesStrategyConfig(
            grid_count=p.get("grid_count", 8),
            direction=GridDirection.NEUTRAL,
            leverage=p.get("leverage", 7),
            trend_period=p.get("trend_period", 48),
            atr_period=p.get("atr_period", 46),
            atr_multiplier=Decimal(str(p.get("atr_multiplier", "6.5"))),
            stop_loss_pct=Decimal(str(p.get("stop_loss_pct", "0.005"))),
            take_profit_grids=p.get("take_profit_grids", 1),
            use_hysteresis=p.get("use_hysteresis", True),
            hysteresis_pct=Decimal(str(p.get("hysteresis_pct", "0.001"))),
            use_signal_cooldown=p.get("use_signal_cooldown", True),
            cooldown_bars=p.get("cooldown_bars", 0),
        )
        return GridFuturesBacktestStrategy(config)

    elif strategy_name == "rsi":
        config = RSIGridStrategyConfig(
            rsi_period=p.get("rsi_period", 14),
            rsi_block_threshold=p.get("rsi_block_threshold", 0.7),
            atr_period=p.get("atr_period", 14),
            grid_count=p.get("grid_count", 15),
            atr_multiplier=Decimal(str(p.get("atr_multiplier", "2.5"))),
            stop_loss_atr_mult=Decimal(str(p.get("stop_loss_atr_mult", "1.5"))),
            take_profit_grids=p.get("take_profit_grids", 1),
            max_hold_bars=p.get("max_hold_bars", 24),
            use_trailing_stop=p.get("use_trailing_stop", False),
            trailing_activate_pct=p.get("trailing_activate_pct", 0.01),
            trailing_distance_pct=p.get("trailing_distance_pct", 0.005),
            use_volatility_filter=p.get("use_volatility_filter", True),
            vol_atr_baseline_period=p.get("vol_atr_baseline_period", 200),
            vol_ratio_low=p.get("vol_ratio_low", 0.5),
            vol_ratio_high=p.get("vol_ratio_high", 2.0),
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

    # 合約相關指標
    if hasattr(result, 'liquidation_count') and result.liquidation_count > 0:
        print(f"\n⚠️  合約指標:")
        print(f"   爆倉次數:      {result.liquidation_count:>12}")
    if hasattr(result, 'total_funding_paid') and result.total_funding_paid:
        if not (hasattr(result, 'liquidation_count') and result.liquidation_count > 0):
            print(f"\n⚠️  合約指標:")
        print(f"   資金費率總付:  {float(result.total_funding_paid):>12.2f} USDT")
    if hasattr(result, 'max_margin_utilization_pct') and result.max_margin_utilization_pct > 0:
        print(f"   最大保證金使用: {float(result.max_margin_utilization_pct):>11.1f}%")

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
        default=7,
        help="槓桿倍數 (default: 7)"
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="本地數據檔案路徑（跳過 API 取資料）"
    )

    args = parser.parse_args()

    # 統一 leverage：從 YAML 讀取，CLI --leverage 只在明確指定時覆蓋
    yaml_key_map = {
        "bollinger": "bollinger", "bollinger_neutral": "bollinger",
        "supertrend": "supertrend", "grid_futures": "grid_futures", "rsi": "rsi_grid",
    }
    if args.strategy in yaml_key_map and args.leverage == 7:
        yaml_params = load_strategy_config(yaml_key_map[args.strategy])
        args.leverage = yaml_params.get("leverage", 7)

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
        if args.data_file:
            klines = _load_klines_from_file(args.data_file)
        else:
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
            fee_rate=Decimal("0.0006"),       # 0.06% (嚴格成本約束)
            slippage_pct=Decimal("0.0005"),   # 0.05%
        ).with_leverage(args.leverage)

        # NEUTRAL 網格策略需要多持倉支援
        if args.strategy in ("bollinger_neutral", "grid_futures"):
            # 從策略配置取得 grid_count 作為 max_positions
            grid_count = getattr(strategy, '_config', None)
            if grid_count:
                grid_count = getattr(grid_count, 'grid_count', 12)
            else:
                grid_count = 12
            config = config.with_max_positions(grid_count)
            print(f"多持倉模式: max_positions={grid_count}")

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
