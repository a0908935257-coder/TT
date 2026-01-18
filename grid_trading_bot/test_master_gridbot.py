#!/usr/bin/env python3
"""
Test Master with Real GridBot.

測試透過 Master 主控台啟動真實的 GridBot。
"""

import asyncio
import os
import sys
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

from src.core import get_logger
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.notification import NotificationManager
from src.master import Master, MasterConfig, BotType, BotState

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


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


async def main() -> None:
    """主程式"""
    print("=" * 60)
    print("    Master Control Console - GridBot 整合測試")
    print("=" * 60)

    exchange = None
    data_manager = None
    master = None

    try:
        # 建立元件
        print("\n[1/4] 正在連接交易所...")
        exchange = await create_exchange_client()
        print("      ✓ 交易所連接成功")

        print("\n[2/4] 正在連接數據庫...")
        data_manager = await create_data_manager(exchange)
        print("      ✓ 數據庫連接成功")

        print("\n[3/4] 正在設定通知...")
        notifier = create_notifier()
        print("      ✓ 通知設定完成")

        # 建立 Master
        print("\n[4/4] 正在初始化 Master...")
        master_config = MasterConfig(
            auto_restart=False,
            max_bots=10,
            snapshot_interval=0,  # Disable for test
            restore_on_start=False,
        )
        master = Master(
            exchange=exchange,
            data_manager=data_manager,
            db_manager=None,  # Skip DB for this test
            notifier=notifier,
            config=master_config,
        )

        # 啟動 Master
        await master.start()
        print("      ✓ Master 啟動成功")

        print("\n" + "=" * 60)
        print("  開始測試 GridBot 建立與啟動")
        print("=" * 60)

        # 讀取 GridBot 配置
        symbol = os.getenv('GRID_SYMBOL', 'BTCUSDT')
        investment = os.getenv('GRID_INVESTMENT', '50')
        risk_level = os.getenv('GRID_RISK_LEVEL', 'medium')
        grid_type = os.getenv('GRID_TYPE', 'geometric')

        bot_config = {
            "symbol": symbol,
            "market_type": "spot",
            "total_investment": investment,
            "risk_level": risk_level if risk_level != 'medium' else 'moderate',
            "grid_type": grid_type,
        }

        print(f"\n  配置資訊:")
        print(f"    交易對: {symbol}")
        print(f"    投資金額: {investment} USDT")
        print(f"    風險等級: {risk_level}")
        print(f"    網格類型: {grid_type}")

        # 建立 GridBot
        print("\n  [Step 1] 透過 Master 建立 GridBot...")
        create_result = await master.create_bot(
            bot_type=BotType.GRID,
            config=bot_config,
            bot_id="test_grid_001",
        )

        if create_result.success:
            print(f"      ✓ Bot 建立成功: {create_result.bot_id}")
        else:
            print(f"      ✗ Bot 建立失敗: {create_result.message}")
            return

        # 檢查 Bot 狀態
        bot_info = master.get_bot("test_grid_001")
        print(f"      狀態: {bot_info.state.value}")

        # 啟動 GridBot
        print("\n  [Step 2] 透過 Master 啟動 GridBot...")
        start_result = await master.start_bot("test_grid_001")

        if start_result.success:
            print(f"      ✓ Bot 啟動成功!")

            # 取得 Bot 狀態
            bot_info = master.get_bot("test_grid_001")
            print(f"      狀態: {bot_info.state.value}")

            # 取得 Dashboard 數據
            dashboard = master.get_dashboard_data()
            print(f"\n  Dashboard 摘要:")
            print(f"    總機器人數: {dashboard.summary.total_bots}")
            print(f"    運行中: {dashboard.summary.running_bots}")

            # 取得 Bot 統計 (如果可用)
            instance = master.registry.get_bot_instance("test_grid_001")
            if instance and hasattr(instance, 'get_status'):
                status = instance.get_status()
                print(f"\n  GridBot 狀態:")
                print(f"    網格範圍: {status.get('lower_price')} - {status.get('upper_price')}")
                print(f"    網格數量: {status.get('grid_count')} 格")
                print(f"    待成交買單: {status.get('pending_buy_orders')}")
                print(f"    待成交賣單: {status.get('pending_sell_orders')}")

            # 等待幾秒觀察
            print("\n  等待 10 秒觀察...")
            await asyncio.sleep(10)

            # 停止 Bot
            print("\n  [Step 3] 透過 Master 停止 GridBot...")
            stop_result = await master.stop_bot("test_grid_001")
            if stop_result.success:
                print(f"      ✓ Bot 已停止")
            else:
                print(f"      ✗ 停止失敗: {stop_result.message}")

        else:
            print(f"      ✗ Bot 啟動失敗: {start_result.message}")

        # 關閉 Master
        print("\n  [Step 4] 關閉 Master...")
        await master.stop()
        print("      ✓ Master 已關閉")

        print("\n" + "=" * 60)
        print("  測試完成!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"錯誤: {e}")
        print(f"\n✗ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if master:
            try:
                await master.stop()
            except:
                pass
            Master.reset_instance()


if __name__ == "__main__":
    asyncio.run(main())
