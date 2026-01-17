#!/usr/bin/env python3
"""
Grid Trading Bot Runner.

啟動網格交易機器人的主程式。
"""

import asyncio
import os
import signal
import sys
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

from core import get_logger
from core.models import MarketType
from data import MarketDataManager
from exchange import ExchangeClient
from notification import NotificationManager
from strategy.grid.bot import GridBot, GridBotConfig
from strategy.grid.models import GridType, RiskLevel
from strategy.grid.risk_manager import RiskConfig, BreakoutAction

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global bot reference for signal handling
_bot: GridBot | None = None


def get_config_from_env() -> GridBotConfig:
    """從環境變數讀取配置"""

    # 讀取網格設定
    symbol = os.getenv('GRID_SYMBOL', 'BTCUSDT')
    investment = Decimal(os.getenv('GRID_INVESTMENT', '50'))
    risk_level_str = os.getenv('GRID_RISK_LEVEL', 'medium')
    grid_type_str = os.getenv('GRID_TYPE', 'geometric')

    # 對應風險等級
    risk_level_map = {
        'conservative': RiskLevel.CONSERVATIVE,
        'medium': RiskLevel.MEDIUM,
        'aggressive': RiskLevel.AGGRESSIVE,
    }
    risk_level = risk_level_map.get(risk_level_str, RiskLevel.MEDIUM)

    # 對應網格類型
    grid_type_map = {
        'arithmetic': GridType.ARITHMETIC,
        'geometric': GridType.GEOMETRIC,
    }
    grid_type = grid_type_map.get(grid_type_str, GridType.GEOMETRIC)

    # 風險管理設定
    risk_config = RiskConfig(
        stop_loss_percent=Decimal(os.getenv('STOP_LOSS_PERCENT', '20')),
        daily_loss_limit=Decimal(os.getenv('DAILY_LOSS_LIMIT', '5')),
        max_consecutive_losses=int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5')),
        upper_breakout_action=BreakoutAction.PAUSE,
        lower_breakout_action=BreakoutAction.PAUSE,
        auto_reset_enabled=os.getenv('AUTO_RESET_ENABLED', 'false').lower() == 'true',
        reset_cooldown_minutes=int(os.getenv('RESET_COOLDOWN_MINUTES', '60')),
    )

    return GridBotConfig(
        symbol=symbol,
        market_type=MarketType.SPOT,
        total_investment=investment,
        risk_level=risk_level,
        grid_type=grid_type,
        risk_config=risk_config,
        atr_period=int(os.getenv('ATR_PERIOD', '14')),
        kline_timeframe=os.getenv('ATR_TIMEFRAME', '4h'),
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
    print("         Grid Trading Bot - 網格交易機器人")
    print("=" * 60)

    # 讀取配置
    config = get_config_from_env()

    print(f"\n配置資訊：")
    print(f"  交易對: {config.symbol}")
    print(f"  投資金額: {config.total_investment} USDT")
    print(f"  風險等級: {config.risk_level.value}")
    print(f"  網格類型: {config.grid_type.value}")
    print(f"  ATR 週期: {config.atr_period}")
    print(f"  時間框架: {config.kline_timeframe}")
    print(f"  止損: {config.risk_config.stop_loss_percent}%")
    print(f"  日虧損限制: {config.risk_config.daily_loss_limit}%")
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
        _bot = GridBot(
            bot_id=f"grid_{config.symbol.lower()}",
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
        print("\n正在啟動機器人...")
        success = await _bot.start()

        if success:
            status = _bot.get_status()
            print("\n" + "=" * 60)
            print("機器人啟動成功！")
            print("=" * 60)
            print(f"  網格範圍: {status.get('lower_price')} - {status.get('upper_price')}")
            print(f"  網格數量: {status.get('grid_count')} 格")
            print(f"  網格間距: {status.get('grid_spacing')}")
            print(f"  買單數量: {status.get('pending_buy_orders')}")
            print(f"  賣單數量: {status.get('pending_sell_orders')}")
            print("\n按 Ctrl+C 停止機器人")
            print("=" * 60)

            # 保持運行
            while _bot.is_running:
                await asyncio.sleep(1)
        else:
            print("\n機器人啟動失敗！請檢查日誌。")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n收到中斷信號...")
        if _bot:
            await _bot.stop(clear_position=False)
    except Exception as e:
        logger.error(f"錯誤: {e}")
        print(f"\n發生錯誤: {e}")
        if _bot:
            await _bot.stop(clear_position=False)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
