#!/usr/bin/env python3
"""
Supertrend Bot Runner.

啟動 Supertrend 趨勢跟蹤交易機器人。

⚠️ 警告：此策略未通過樣本外驗證，不建議用於實盤交易！
策略：ATR 10, 乘數 3.0 @ 5x 逐倉
樣本外測試結果：所有參數組合均為負報酬
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
from src.bots.supertrend import SupertrendBot, SupertrendConfig

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global bot reference for signal handling
_bot: SupertrendBot | None = None


def get_config_from_env() -> SupertrendConfig:
    """從環境變數讀取配置"""
    # 資金分配：如果設定了 MAX_CAPITAL，則限制該 Bot 最大可用資金
    max_capital_str = os.getenv('SUPERTREND_MAX_CAPITAL', '')
    max_capital = Decimal(max_capital_str) if max_capital_str else None

    return SupertrendConfig(
        # 基本設定
        symbol=os.getenv('SUPERTREND_SYMBOL', 'BTCUSDT'),
        timeframe=os.getenv('SUPERTREND_TIMEFRAME', '15m'),
        leverage=int(os.getenv('SUPERTREND_LEVERAGE', '5')),
        margin_type=os.getenv('SUPERTREND_MARGIN_TYPE', 'ISOLATED'),

        # 資金分配
        max_capital=max_capital,
        position_size_pct=Decimal(os.getenv('SUPERTREND_POSITION_SIZE', '0.1')),

        # Supertrend 設定 (⚠️ 未通過樣本外驗證)
        atr_period=int(os.getenv('SUPERTREND_ATR_PERIOD', '10')),  # ⚠️ 未通過驗證
        atr_multiplier=Decimal(os.getenv('SUPERTREND_ATR_MULTIPLIER', '3.0')),  # ⚠️ 未通過驗證

        # 可選：追蹤止損
        use_trailing_stop=os.getenv('SUPERTREND_USE_TRAILING_STOP', 'false').lower() == 'true',
        trailing_stop_pct=Decimal(os.getenv('SUPERTREND_TRAILING_STOP_PCT', '0.02')),
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
    print("    Supertrend Bot - 趨勢跟蹤交易機器人")
    print("=" * 60)

    # 讀取配置
    config = get_config_from_env()

    print(f"\n配置資訊：")
    print(f"  交易對: {config.symbol}")
    print(f"  時間框架: {config.timeframe}")
    print(f"  槓桿: {config.leverage}x")
    print(f"  保證金模式: {config.margin_type} (逐倉)")

    # 資金分配顯示
    if config.max_capital:
        print(f"  分配資金: {config.max_capital:,.0f} USDT")
        print(f"  單次倉位: {config.position_size_pct*100}% = {config.max_capital * config.position_size_pct:,.0f} USDT")
    else:
        print(f"  分配資金: 全部餘額")
        print(f"  單次倉位: {config.position_size_pct*100}%")

    print(f"  ATR 週期: {config.atr_period}")
    print(f"  ATR 乘數: {config.atr_multiplier}")
    print(f"  追蹤止損: {'開啟' if config.use_trailing_stop else '關閉'}")
    if config.use_trailing_stop:
        print(f"  追蹤止損比例: {config.trailing_stop_pct*100}%")
    print(f"\n  ⚠️ 警告: 此策略未通過樣本外驗證!")
    print(f"  策略: Supertrend @ {config.leverage}x (不建議實盤使用)")
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
        _bot = SupertrendBot(
            bot_id=f"supertrend_{config.symbol.lower()}",
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
            supertrend = status.get('supertrend', {})
            if supertrend:
                print(f"  當前趨勢: {supertrend.get('trend', 'N/A')}")
                print(f"  Supertrend 值: {supertrend.get('value', 0):.2f}")
                print(f"  ATR: {supertrend.get('atr', 0):.2f}")
            print(f"  有持倉: {'是' if status.get('position') else '否'}")
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
