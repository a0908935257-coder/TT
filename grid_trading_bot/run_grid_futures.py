#!/usr/bin/env python3
"""
Grid Futures Bot Runner.

啟動合約版網格交易機器人。
支援槓桿、雙向交易、趨勢跟蹤。

優化配置 (通過回測驗證):
- 年化 113.6%
- 回撤 24.7%
- Sharpe 1.54

Usage:
    python run_grid_futures.py
"""

import asyncio
import os
import signal
import sys
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

from src.core import get_logger
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.notification import NotificationManager
from src.bots.grid_futures import GridFuturesBot, GridFuturesConfig, GridDirection

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global bot reference for signal handling
_bot: GridFuturesBot | None = None


def get_config_from_env() -> GridFuturesConfig:
    """從環境變數讀取配置"""
    # 資金分配
    max_capital_str = os.getenv('GRID_FUTURES_MAX_CAPITAL', '')
    max_capital = Decimal(max_capital_str) if max_capital_str else None

    # 方向模式
    direction_str = os.getenv('GRID_FUTURES_DIRECTION', 'trend_follow').lower()
    direction_map = {
        'long_only': GridDirection.LONG_ONLY,
        'short_only': GridDirection.SHORT_ONLY,
        'neutral': GridDirection.NEUTRAL,
        'trend_follow': GridDirection.TREND_FOLLOW,
    }
    direction = direction_map.get(direction_str, GridDirection.TREND_FOLLOW)

    return GridFuturesConfig(
        # 基本設定
        symbol=os.getenv('GRID_FUTURES_SYMBOL', 'BTCUSDT'),
        timeframe=os.getenv('GRID_FUTURES_TIMEFRAME', '1h'),
        leverage=int(os.getenv('GRID_FUTURES_LEVERAGE', '3')),
        margin_type=os.getenv('GRID_FUTURES_MARGIN_TYPE', 'ISOLATED'),

        # 網格設定 (優化: 15 格)
        grid_count=int(os.getenv('GRID_FUTURES_COUNT', '15')),
        direction=direction,

        # 趨勢過濾 (優化: 週期 30)
        use_trend_filter=os.getenv('GRID_FUTURES_USE_TREND_FILTER', 'true').lower() == 'true',
        trend_period=int(os.getenv('GRID_FUTURES_TREND_PERIOD', '30')),

        # 動態 ATR 範圍 (優化: 乘數 2.0)
        use_atr_range=os.getenv('GRID_FUTURES_USE_ATR_RANGE', 'true').lower() == 'true',
        atr_period=int(os.getenv('GRID_FUTURES_ATR_PERIOD', '14')),
        atr_multiplier=Decimal(os.getenv('GRID_FUTURES_ATR_MULTIPLIER', '2.0')),
        fallback_range_pct=Decimal(os.getenv('GRID_FUTURES_RANGE_PCT', '0.08')),

        # 倉位管理 (優化: 10% 單次)
        max_capital=max_capital,
        position_size_pct=Decimal(os.getenv('GRID_FUTURES_POSITION_SIZE', '0.1')),
        max_position_pct=Decimal(os.getenv('GRID_FUTURES_MAX_POSITION', '0.5')),

        # 風險管理
        stop_loss_pct=Decimal(os.getenv('GRID_FUTURES_STOP_LOSS', '0.05')),
        rebuild_threshold_pct=Decimal(os.getenv('GRID_FUTURES_REBUILD_THRESHOLD', '0.02')),
    )


async def create_exchange_client() -> ExchangeClient:
    """建立交易所客戶端"""
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    client = ExchangeClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    await client.connect()
    return client


async def create_data_manager(exchange: ExchangeClient) -> MarketDataManager:
    """建立數據管理器"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'trading_bot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'pool_size': 10,
    }

    redis_config = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', '6379')),
        'db': 0,
        'key_prefix': 'trading:',
    }

    manager = MarketDataManager(
        db_config=db_config,
        redis_config=redis_config,
        exchange=exchange,
    )
    await manager.connect()
    return manager


def create_notifier() -> NotificationManager:
    """建立通知管理器"""
    return NotificationManager.from_env()


async def shutdown(sig: signal.Signals) -> None:
    """優雅關閉"""
    global _bot

    logger.info(f"收到信號 {sig.name}，正在關閉...")

    if _bot:
        await _bot.stop(clear_position=False)

    logger.info("機器人已關閉")


async def main() -> None:
    """主程式"""
    global _bot

    print("=" * 60)
    print("    Grid Futures Bot - 合約網格交易機器人")
    print("=" * 60)

    # 讀取配置
    config = get_config_from_env()

    # 方向模式顯示
    direction_names = {
        GridDirection.LONG_ONLY: "只做多",
        GridDirection.SHORT_ONLY: "只做空",
        GridDirection.NEUTRAL: "雙向網格",
        GridDirection.TREND_FOLLOW: "順勢網格",
    }
    direction_name = direction_names.get(config.direction, "順勢網格")

    print(f"\n配置資訊：")
    print(f"  交易對: {config.symbol}")
    print(f"  時間框架: {config.timeframe}")
    print(f"  槓桿: {config.leverage}x")
    print(f"  保證金模式: {config.margin_type} (逐倉)")
    print(f"  方向模式: {direction_name}")

    # 資金分配顯示
    if config.max_capital:
        print(f"  分配資金: {config.max_capital:,.0f} USDT")
        print(f"  單次倉位: {config.position_size_pct*100}% = {config.max_capital * config.position_size_pct:,.0f} USDT")
    else:
        print(f"  分配資金: 全部餘額")
        print(f"  單次倉位: {config.position_size_pct*100}%")

    print(f"  最大總倉位: {config.max_position_pct*100}%")

    print(f"\n  網格數量: {config.grid_count}")
    print(f"  趨勢過濾: {'開啟' if config.use_trend_filter else '關閉'} (SMA {config.trend_period})")
    print(f"  動態範圍: {'開啟' if config.use_atr_range else '關閉'} ({config.atr_multiplier}x ATR)")
    print(f"  備用範圍: ±{config.fallback_range_pct*100}%")

    print(f"\n  策略: 穩健版 @ {config.leverage}x")
    print(f"  回測績效: 年化 ~20%, 通過過度擬合驗證")
    print()

    try:
        # 建立元件
        print("正在連接交易所...")
        exchange = await create_exchange_client()

        print("正在連接數據庫...")
        data_manager = await create_data_manager(exchange)

        print("正在設定通知...")
        notifier = create_notifier()

        # 建立機器人
        print("正在初始化機器人...")
        _bot = GridFuturesBot(
            bot_id=f"grid_futures_{config.symbol.lower()}",
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        # 設定信號處理
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s))
            )

        # 啟動機器人
        print("\n正在啟動交易機器人...")
        success = await _bot.start()

        if success:
            status = _bot.get_status()
            print("\n" + "=" * 60)
            print("機器人啟動成功！")
            print("=" * 60)
            print(f"  交易對: {status.get('symbol')}")
            print(f"  狀態: {status.get('state')}")
            print(f"  槓桿: {status.get('leverage')}x")
            print(f"  方向: {status.get('direction')}")
            print(f"  資金: ${float(status.get('capital', 0)):,.2f}")

            grid = status.get('grid', {})
            if grid:
                print(f"  網格中心: ${float(grid.get('center', 0)):,.2f}")
                print(f"  網格範圍: ${float(grid.get('lower', 0)):,.2f} - ${float(grid.get('upper', 0)):,.2f}")
                print(f"  網格數量: {grid.get('count')}")

            print(f"  趨勢: {'看漲' if status.get('current_trend') == 1 else '看跌' if status.get('current_trend') == -1 else '中性'}")
            print(f"  有持倉: {'是' if status.get('has_position') else '否'}")
        else:
            print("\n交易機器人啟動失敗！請檢查日誌。")

        print("\n按 Ctrl+C 停止程式")
        print("=" * 60)

        # 保持運行
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n收到中斷信號...")
        if _bot:
            await _bot.stop(clear_position=False)
    except Exception as e:
        logger.error(f"錯誤: {e}")
        print(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        if _bot:
            await _bot.stop(clear_position=False)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
