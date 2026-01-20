#!/usr/bin/env python3
"""
Integrated Trading System Runner.

一鍵啟動整合交易系統：
- Master 主控台
- Bollinger Bot (合約)
- Supertrend Bot (合約)
- Grid Futures Bot (合約)
- Discord Bot (可管理所有機器人)

使用方式: python run_all.py
"""

import asyncio
import os
import signal
import sys
from decimal import Decimal

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.core import get_logger
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.master import Master, MasterConfig, BotType
from src.notification import NotificationManager

logger = get_logger(__name__)

# Global references for cleanup
_master: Master | None = None
_exchange: ExchangeClient | None = None
_data_manager: MarketDataManager | None = None
_notifier: NotificationManager | None = None
_discord_bot = None
_shutdown_event: asyncio.Event | None = None


def print_banner():
    """Print startup banner."""
    print("""
============================================================
       整合交易系統 - Integrated Trading System
============================================================

啟動項目:
  ✓ Master 主控台
  ✓ Bollinger Bot (合約 20x)
  ✓ RSI Bot (合約 7x) - 取代 Supertrend
  ✓ Grid Futures Bot (合約 3x)
  ✓ Discord Bot (遠端管理)

Discord 指令:
  /bot list     - 列出所有機器人
  /bot stop     - 停止機器人
  /bot start    - 啟動機器人
  /status       - 系統狀態
  /balance      - 帳戶餘額

============================================================
""")


async def create_exchange_client() -> ExchangeClient:
    """Create and connect exchange client."""
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'

    client = ExchangeClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    await client.connect()
    return client


async def create_data_manager(exchange: ExchangeClient) -> MarketDataManager:
    """Create and connect data manager."""
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
    """Create notification manager."""
    return NotificationManager.from_env()


def get_bollinger_config() -> dict:
    """Get Bollinger Bot config from .env."""
    return {
        "symbol": os.getenv('BOLLINGER_SYMBOL', 'BTCUSDT'),
        "timeframe": os.getenv('BOLLINGER_TIMEFRAME', '15m'),
        "leverage": int(os.getenv('BOLLINGER_LEVERAGE', '20')),
        "position_size_pct": os.getenv('BOLLINGER_POSITION_SIZE', '0.1'),
        "bb_period": int(os.getenv('BOLLINGER_BB_PERIOD', '20')),
        "bb_std": os.getenv('BOLLINGER_BB_STD', '3.25'),
        "bbw_lookback": int(os.getenv('BOLLINGER_BBW_LOOKBACK', '200')),
        "bbw_threshold_pct": int(os.getenv('BOLLINGER_BBW_THRESHOLD', '20')),
        "use_trend_filter": os.getenv('BOLLINGER_USE_TREND_FILTER', 'false').lower() == 'true',
        "trend_period": int(os.getenv('BOLLINGER_TREND_PERIOD', '50')),
        "use_atr_stop": os.getenv('BOLLINGER_USE_ATR_STOP', 'true').lower() == 'true',
        "atr_period": int(os.getenv('BOLLINGER_ATR_PERIOD', '14')),
        "atr_multiplier": os.getenv('BOLLINGER_ATR_MULTIPLIER', '2.0'),
        "use_trailing_stop": os.getenv('BOLLINGER_USE_TRAILING_STOP', 'true').lower() == 'true',
        "trailing_atr_mult": os.getenv('BOLLINGER_TRAILING_ATR_MULT', '2.0'),
        "stop_loss_pct": os.getenv('BOLLINGER_STOP_LOSS_PCT', '0.015'),
        "max_hold_bars": int(os.getenv('BOLLINGER_MAX_HOLD_BARS', '48')),
        "max_capital": os.getenv('BOLLINGER_MAX_CAPITAL', '7'),
    }


def get_rsi_config() -> dict:
    """Get RSI Bot config from .env."""
    return {
        "symbol": os.getenv('RSI_SYMBOL', 'BTCUSDT'),
        "timeframe": os.getenv('RSI_TIMEFRAME', '15m'),
        "rsi_period": int(os.getenv('RSI_PERIOD', '14')),
        "oversold": int(os.getenv('RSI_OVERSOLD', '20')),
        "overbought": int(os.getenv('RSI_OVERBOUGHT', '80')),
        "exit_level": int(os.getenv('RSI_EXIT_LEVEL', '50')),
        "leverage": int(os.getenv('RSI_LEVERAGE', '7')),
        "margin_type": os.getenv('RSI_MARGIN_TYPE', 'ISOLATED'),
        "max_capital": os.getenv('RSI_MAX_CAPITAL', '7'),
        "position_size_pct": os.getenv('RSI_POSITION_SIZE', '0.1'),
        "stop_loss_pct": os.getenv('RSI_STOP_LOSS_PCT', '0.02'),
        "take_profit_pct": os.getenv('RSI_TAKE_PROFIT_PCT', '0.03'),
    }


def get_grid_futures_config() -> dict:
    """Get Grid Futures Bot config from .env."""
    return {
        "symbol": os.getenv('GRID_FUTURES_SYMBOL', 'BTCUSDT'),
        "timeframe": os.getenv('GRID_FUTURES_TIMEFRAME', '1h'),
        "leverage": int(os.getenv('GRID_FUTURES_LEVERAGE', '3')),
        "margin_type": os.getenv('GRID_FUTURES_MARGIN_TYPE', 'ISOLATED'),
        "grid_count": int(os.getenv('GRID_FUTURES_COUNT', '12')),
        "direction": os.getenv('GRID_FUTURES_DIRECTION', 'neutral'),
        "use_trend_filter": os.getenv('GRID_FUTURES_USE_TREND_FILTER', 'false').lower() == 'true',
        "trend_period": int(os.getenv('GRID_FUTURES_TREND_PERIOD', '20')),
        "use_atr_range": os.getenv('GRID_FUTURES_USE_ATR_RANGE', 'true').lower() == 'true',
        "atr_period": int(os.getenv('GRID_FUTURES_ATR_PERIOD', '14')),
        "atr_multiplier": os.getenv('GRID_FUTURES_ATR_MULTIPLIER', '2.0'),
        "fallback_range_pct": os.getenv('GRID_FUTURES_RANGE_PCT', '0.08'),
        "max_capital": os.getenv('GRID_FUTURES_MAX_CAPITAL', '5'),
        "position_size_pct": os.getenv('GRID_FUTURES_POSITION_SIZE', '0.1'),
        "max_position_pct": os.getenv('GRID_FUTURES_MAX_POSITION', '0.5'),
        "stop_loss_pct": os.getenv('GRID_FUTURES_STOP_LOSS', '0.05'),
        "rebuild_threshold_pct": os.getenv('GRID_FUTURES_REBUILD_THRESHOLD', '0.02'),
    }


async def create_and_start_bots(master: Master) -> list[str]:
    """Create and start all bots through Master."""
    bot_ids = []

    # 1. Create Bollinger Bot
    print("  創建 Bollinger Bot...")
    bollinger_config = get_bollinger_config()
    result = await master.create_bot(BotType.BOLLINGER, bollinger_config)
    if result.success:
        bot_ids.append(result.bot_id)
        print(f"    ✓ 已創建: {result.bot_id}")
        # Start the bot
        start_result = await master.start_bot(result.bot_id)
        if start_result.success:
            print(f"    ✓ 已啟動: {result.bot_id}")
        else:
            print(f"    ✗ 啟動失敗: {start_result.message}")
    else:
        print(f"    ✗ 創建失敗: {result.message}")

    # 2. Create RSI Bot (replaces Supertrend)
    print("  創建 RSI Bot...")
    rsi_config = get_rsi_config()
    result = await master.create_bot(BotType.RSI, rsi_config)
    if result.success:
        bot_ids.append(result.bot_id)
        print(f"    ✓ 已創建: {result.bot_id}")
        # Start the bot
        start_result = await master.start_bot(result.bot_id)
        if start_result.success:
            print(f"    ✓ 已啟動: {result.bot_id}")
        else:
            print(f"    ✗ 啟動失敗: {start_result.message}")
    else:
        print(f"    ✗ 創建失敗: {result.message}")

    # 3. Create Grid Futures Bot
    print("  創建 Grid Futures Bot...")
    grid_futures_config = get_grid_futures_config()
    result = await master.create_bot(BotType.GRID_FUTURES, grid_futures_config)
    if result.success:
        bot_ids.append(result.bot_id)
        print(f"    ✓ 已創建: {result.bot_id}")
        # Start the bot
        start_result = await master.start_bot(result.bot_id)
        if start_result.success:
            print(f"    ✓ 已啟動: {result.bot_id}")
        else:
            print(f"    ✗ 啟動失敗: {start_result.message}")
    else:
        print(f"    ✗ 創建失敗: {result.message}")

    return bot_ids


async def start_discord_bot(master: Master):
    """Start Discord bot connected to Master."""
    global _discord_bot

    try:
        from src.discord_bot import TradingBot, load_discord_config

        config = load_discord_config()
        errors = config.validate()
        if errors:
            print("  ⚠️ Discord Bot 配置不完整，跳過")
            for error in errors:
                print(f"     - {error}")
            return None

        _discord_bot = TradingBot(config, master=master, risk_engine=None)

        # Start in background
        asyncio.create_task(_discord_bot.start_bot())
        print("  ✓ Discord Bot 啟動中...")

        return _discord_bot

    except Exception as e:
        print(f"  ⚠️ Discord Bot 啟動失敗: {e}")
        return None


async def shutdown():
    """Graceful shutdown."""
    global _master, _exchange, _data_manager, _notifier, _discord_bot

    print("\n正在關閉系統...")

    # Stop Discord bot
    if _discord_bot:
        try:
            await _discord_bot.stop_bot()
            print("  ✓ Discord Bot 已關閉")
        except:
            pass

    # Stop Master (this stops all bots)
    if _master:
        try:
            await _master.stop()
            Master.reset_instance()
            print("  ✓ Master 已關閉")
        except:
            pass

    # Disconnect exchange
    if _exchange:
        try:
            await _exchange.disconnect()
            print("  ✓ 交易所連接已關閉")
        except:
            pass

    # Close notifier
    if _notifier:
        try:
            await _notifier.close()
        except:
            pass

    print("\n系統已完全關閉")


async def main():
    """Main entry point."""
    global _master, _exchange, _data_manager, _notifier, _shutdown_event

    print_banner()
    _shutdown_event = asyncio.Event()

    # Setup signal handlers
    def handle_signal(sig):
        print(f"\n收到信號 {sig.name}...")
        _shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        # 1. Connect to exchange
        print("正在連接交易所...")
        _exchange = await create_exchange_client()
        print("  ✓ 交易所連接成功")

        # 2. Connect to data sources
        print("正在連接數據庫...")
        _data_manager = await create_data_manager(_exchange)
        print("  ✓ 數據庫連接成功")

        # 3. Setup notifications
        print("正在設定通知...")
        _notifier = create_notifier()
        print("  ✓ 通知設定完成")

        # 4. Initialize Master
        print("正在初始化 Master...")
        master_config = MasterConfig(
            auto_restart=False,  # Disable auto-restart (HeartbeatMonitor missing on_timeout)
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
        print("  ✓ Master 啟動成功")

        # 5. Create and start bots
        print("\n正在創建機器人...")
        bot_ids = await create_and_start_bots(_master)

        # 6. Start Discord bot
        print("\n正在啟動 Discord Bot...")
        await start_discord_bot(_master)

        # Wait a moment for Discord to connect
        await asyncio.sleep(3)

        # Print summary
        print("\n" + "=" * 60)
        print("       系統啟動完成！")
        print("=" * 60)

        bots = _master.get_all_bots()
        print(f"\n運行中的機器人: {len(bots)}")
        for bot in bots:
            state = bot.state.value if hasattr(bot.state, 'value') else str(bot.state)
            print(f"  • {bot.bot_id} [{bot.bot_type.value}] - {state}")

        print(f"""
Discord 指令:
  /bot list     - 列出機器人
  /bot stop     - 停止機器人
  /status       - 系統狀態

按 Ctrl+C 停止系統
""")
        print("=" * 60)

        # Send notification
        try:
            await _notifier.send_info(
                title="🚀 交易系統已啟動",
                message=f"已啟動 {len(bots)} 隻機器人\n\n"
                        f"使用 /bot list 查看狀態",
            )
        except:
            pass

        # Wait for shutdown signal
        await _shutdown_event.wait()

    except Exception as e:
        logger.error(f"系統錯誤: {e}")
        print(f"\n❌ 系統錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
