#!/usr/bin/env python3
"""
Master Control Console Runner.

透過主控台管理多個交易機器人。
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
from src.core.models import MarketType
from src.config import load_strategy_config
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.notification import NotificationManager
from src.master import Master, MasterConfig, BotType, BotState

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global references
_master: Master | None = None
_exchange: ExchangeClient | None = None
_data_manager: MarketDataManager | None = None
_notifier: NotificationManager | None = None


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
        market_type=MarketType.FUTURES,
    )
    await manager.connect()
    return manager


def create_notifier() -> NotificationManager:
    """建立通知管理器"""
    return NotificationManager.from_env()


async def shutdown(sig: signal.Signals) -> None:
    """優雅關閉"""
    global _master, _exchange

    print(f"\n收到信號 {sig.name}，正在關閉...")

    if _master:
        await _master.stop()
        Master.reset_instance()

    if _exchange:
        await _exchange.disconnect()

    print("主控台已關閉")


def print_help():
    """顯示幫助訊息"""
    print("""
可用指令:
  list          - 列出所有機器人
  status        - 顯示 Dashboard 摘要

  建立機器人:
    create grid <symbol> [investment]         - 建立 Grid Bot (現貨)
    create bollinger <symbol>                 - 建立 Bollinger Bot (合約)
    create supertrend <symbol>                - 建立 Supertrend Bot (合約)
    create grid_futures <symbol>              - 建立 Grid Futures Bot (合約)

  機器人控制:
    start <bot_id>  - 啟動機器人
    stop <bot_id>   - 停止機器人
    pause <bot_id>  - 暫停機器人
    resume <bot_id> - 恢復機器人
    delete <bot_id> - 刪除機器人

  其他:
    info <bot_id>   - 顯示機器人詳情
    health          - 健康檢查
    help            - 顯示此幫助
    quit / exit     - 退出程式
""")


async def handle_command(master: Master, cmd: str) -> bool:
    """
    處理使用者指令

    Returns:
        False if should exit, True otherwise
    """
    parts = cmd.strip().split()
    if not parts:
        return True

    command = parts[0].lower()

    try:
        if command in ('quit', 'exit', 'q'):
            return False

        elif command == 'help':
            print_help()

        elif command == 'list':
            bots = master.get_all_bots()
            if not bots:
                print("目前沒有任何機器人")
            else:
                print(f"\n{'Bot ID':<20} {'類型':<8} {'交易對':<10} {'狀態':<12}")
                print("-" * 55)
                for bot in bots:
                    print(f"{bot.bot_id:<20} {bot.bot_type.value:<8} {bot.symbol:<10} {bot.state.value:<12}")
                print()

        elif command == 'status':
            dashboard = master.get_dashboard_data()
            summary = dashboard.summary
            print(f"""
Dashboard 摘要:
  總機器人數: {summary.total_bots}
  運行中: {summary.running_bots}
  暫停中: {summary.paused_bots}
  已停止: {summary.stopped_bots}
  錯誤: {summary.error_bots}
  總投資: {summary.total_investment} USDT
  今日損益: {summary.today_profit} USDT
  今日交易: {summary.today_trades} 筆
""")

        elif command == 'create':
            if len(parts) < 3:
                print("用法: create <type> <symbol> [options]")
                print("類型: grid, bollinger, supertrend, grid_futures")
                print("例如: create grid BTCUSDT 100")
                print("      create bollinger BTCUSDT")
                print("      create supertrend BTCUSDT")
                print("      create grid_futures BTCUSDT")
                return True

            bot_type_str = parts[1].lower()
            symbol = parts[2].upper()

            # Parse bot type
            if bot_type_str == 'grid':
                bot_type = BotType.GRID
                investment = parts[3] if len(parts) > 3 else os.getenv('GRID_INVESTMENT', '70')
                risk_level = os.getenv('GRID_RISK_LEVEL', 'medium')
                if risk_level == 'medium':
                    risk_level = 'moderate'
                bot_config = {
                    "symbol": symbol,
                    "market_type": "spot",
                    "total_investment": investment,
                    "risk_level": risk_level,
                }

            elif bot_type_str == 'bollinger':
                bot_type = BotType.BOLLINGER
                bot_config = load_strategy_config("bollinger")
                bot_config["symbol"] = symbol

            elif bot_type_str == 'supertrend':
                bot_type = BotType.SUPERTREND
                bot_config = load_strategy_config("supertrend")
                bot_config["symbol"] = symbol

            elif bot_type_str in ('grid_futures', 'gridfutures'):
                bot_type = BotType.GRID_FUTURES
                bot_config = load_strategy_config("grid_futures")
                bot_config["symbol"] = symbol

            else:
                print(f"未知的機器人類型: {bot_type_str}")
                print("支援類型: grid, bollinger, supertrend, grid_futures")
                return True

            result = await master.create_bot(bot_type, bot_config)
            if result.success:
                print(f"✓ 機器人建立成功: {result.bot_id}")
            else:
                print(f"✗ 建立失敗: {result.message}")

        elif command == 'start':
            if len(parts) < 2:
                print("用法: start <bot_id>")
                return True

            bot_id = parts[1]
            result = await master.start_bot(bot_id)
            if result.success:
                print(f"✓ 機器人 {bot_id} 已啟動")
            else:
                print(f"✗ 啟動失敗: {result.message}")

        elif command == 'stop':
            if len(parts) < 2:
                print("用法: stop <bot_id>")
                return True

            bot_id = parts[1]
            result = await master.stop_bot(bot_id)
            if result.success:
                print(f"✓ 機器人 {bot_id} 已停止")
            else:
                print(f"✗ 停止失敗: {result.message}")

        elif command == 'pause':
            if len(parts) < 2:
                print("用法: pause <bot_id>")
                return True

            bot_id = parts[1]
            result = await master.pause_bot(bot_id)
            if result.success:
                print(f"✓ 機器人 {bot_id} 已暫停")
            else:
                print(f"✗ 暫停失敗: {result.message}")

        elif command == 'resume':
            if len(parts) < 2:
                print("用法: resume <bot_id>")
                return True

            bot_id = parts[1]
            result = await master.resume_bot(bot_id)
            if result.success:
                print(f"✓ 機器人 {bot_id} 已恢復")
            else:
                print(f"✗ 恢復失敗: {result.message}")

        elif command == 'delete':
            if len(parts) < 2:
                print("用法: delete <bot_id>")
                return True

            bot_id = parts[1]
            result = await master.delete_bot(bot_id)
            if result.success:
                print(f"✓ 機器人 {bot_id} 已刪除")
            else:
                print(f"✗ 刪除失敗: {result.message}")

        elif command == 'info':
            if len(parts) < 2:
                print("用法: info <bot_id>")
                return True

            bot_id = parts[1]
            bot_info = master.get_bot(bot_id)
            if not bot_info:
                print(f"找不到機器人: {bot_id}")
                return True

            print(f"""
機器人資訊:
  Bot ID: {bot_info.bot_id}
  類型: {bot_info.bot_type.value}
  交易對: {bot_info.symbol}
  狀態: {bot_info.state.value}
  建立時間: {bot_info.created_at}
""")
            # Try to get more details from instance
            instance = master.registry.get_bot_instance(bot_id)
            if instance and hasattr(instance, 'get_status'):
                status = instance.get_status()
                print(f"  網格範圍: {status.get('lower_price')} - {status.get('upper_price')}")
                print(f"  網格數量: {status.get('grid_count')} 格")
                print(f"  買單數量: {status.get('pending_buy_orders')}")
                print(f"  賣單數量: {status.get('pending_sell_orders')}")
                print()

        elif command == 'health':
            results = await master.health_check_all()
            if not results:
                print("沒有機器人可檢查")
            else:
                print(f"\n{'Bot ID':<20} {'狀態':<10} {'訊息':<30}")
                print("-" * 65)
                for result in results:
                    print(f"{result.bot_id:<20} {result.status.value:<10} {result.message[:30]:<30}")
                print()

        else:
            print(f"未知指令: {command}")
            print("輸入 'help' 查看可用指令")

    except Exception as e:
        print(f"錯誤: {e}")

    return True


async def main() -> None:
    """主程式"""
    global _master, _exchange, _data_manager, _notifier

    print("=" * 60)
    print("       Master Control Console - 主控台")
    print("=" * 60)

    try:
        # 建立元件
        print("\n正在連接交易所...")
        _exchange = await create_exchange_client()
        print("✓ 交易所連接成功")

        print("正在連接數據庫...")
        _data_manager = await create_data_manager(_exchange)
        print("✓ 數據庫連接成功")

        print("正在設定通知...")
        _notifier = create_notifier()
        print("✓ 通知設定完成")

        # 建立 Master
        print("正在初始化主控台...")
        master_config = MasterConfig(
            auto_restart=False,
            max_bots=10,
            snapshot_interval=3600,
            restore_on_start=False,
        )
        _master = Master(
            exchange=_exchange,
            data_manager=_data_manager,
            db_manager=None,
            notifier=_notifier,
            config=master_config,
        )

        await _master.start()
        print("✓ 主控台啟動成功")

        print("\n" + "=" * 60)
        print("  輸入 'help' 查看可用指令")
        print("  輸入 'quit' 或 Ctrl+C 退出")
        print("=" * 60 + "\n")

        # 互動式指令迴圈
        while True:
            try:
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("master> ")
                )
                if not await handle_command(_master, cmd):
                    break
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break

    except Exception as e:
        logger.error(f"錯誤: {e}")
        print(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        print("\n正在關閉...")
        if _master:
            try:
                await _master.stop()
            except Exception:
                pass
            Master.reset_instance()

        if _exchange:
            try:
                await _exchange.disconnect()
            except Exception:
                pass

        if _notifier:
            try:
                await _notifier.close()
            except Exception:
                pass

        print("主控台已關閉")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
