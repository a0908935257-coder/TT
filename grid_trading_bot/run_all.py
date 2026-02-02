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
from src.core.models import MarketType
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.master import Master, MasterConfig, BotType
from src.notification import NotificationManager
from src.fund_manager import FundManager, SignalCoordinator, ConflictResolution
from src.fund_manager.models.config import FundManagerConfig
from src.config import load_strategy_config

logger = get_logger(__name__)

# Cache for YAML configuration
_yaml_config: dict | None = None


def _load_yaml_config() -> dict:
    """Load and cache the settings.yaml configuration."""
    global _yaml_config
    if _yaml_config is None:
        import yaml
        with open("src/fund_manager/config/settings.yaml", "r") as f:
            _yaml_config = yaml.safe_load(f)
    return _yaml_config

# Global references for cleanup
_master: Master | None = None
_exchange: ExchangeClient | None = None
_data_manager: MarketDataManager | None = None
_notifier: NotificationManager | None = None
_fund_manager: FundManager | None = None
_signal_coordinator: SignalCoordinator | None = None
_discord_bot = None
_shutdown_event: asyncio.Event | None = None


async def ensure_hedge_mode(exchange: ExchangeClient) -> bool:
    """
    Ensure Binance Futures is in Hedge Mode (dual side position).

    Hedge Mode allows holding both LONG and SHORT positions simultaneously
    on the same symbol, which is required for multi-bot trading.

    Returns:
        True if hedge mode is enabled, False otherwise
    """
    try:
        # Check current position mode
        mode_info = await exchange.futures.get_position_mode()
        is_hedge_mode = mode_info.get("dual_side_position", False)

        if is_hedge_mode:
            logger.info("Position mode: Hedge Mode (已啟用雙向持倉)")
            return True

        # Need to enable hedge mode
        logger.info("Position mode: One-Way Mode - 正在切換到 Hedge Mode...")

        # Note: Cannot change position mode if there are open positions
        success = await exchange.futures.set_position_mode(dual_side=True)

        if success:
            logger.info("Successfully enabled Hedge Mode (雙向持倉)")
            return True
        else:
            logger.error("Failed to enable Hedge Mode")
            return False

    except Exception as e:
        error_msg = str(e)
        if "-4059" in error_msg:
            # Already in target mode
            logger.info("Position mode already set to Hedge Mode")
            return True
        elif "-4068" in error_msg:
            # Cannot change mode with open positions
            logger.error(
                "無法切換倉位模式：有未平倉位！\n"
                "請先在 Binance 手動平倉，或在網頁版設定中切換到「雙向持倉」模式"
            )
            return False
        else:
            logger.error(f"Error checking/setting position mode: {e}")
            return False


def initialize_signal_coordinator() -> SignalCoordinator:
    """
    Initialize SignalCoordinator from settings.yaml configuration.

    Reads signal_coordinator section from settings.yaml and configures:
    - Conflict resolution strategy
    - Allowed hedge symbols
    - Bot priorities
    """
    config = _load_yaml_config()
    sc_config = config.get("signal_coordinator", {})

    # Parse resolution strategy
    resolution_str = sc_config.get("resolution", "block_newer")
    resolution_map = {
        "block_newer": ConflictResolution.BLOCK_NEWER,
        "block_smaller": ConflictResolution.BLOCK_SMALLER,
        "warn_only": ConflictResolution.WARN_ONLY,
        "allow_hedge": ConflictResolution.ALLOW_HEDGE,
        "priority_based": ConflictResolution.PRIORITY_BASED,
    }
    resolution = resolution_map.get(resolution_str, ConflictResolution.BLOCK_NEWER)

    # Get other settings
    signal_ttl = sc_config.get("signal_ttl_seconds", 60.0)
    allowed_hedge_symbols = set(sc_config.get("allowed_hedge_symbols", []))
    bot_priorities = sc_config.get("bot_priorities", {})

    # Create and configure SignalCoordinator
    coordinator = SignalCoordinator(
        resolution=resolution,
        signal_ttl_seconds=signal_ttl,
        bot_priorities=bot_priorities,
        allowed_hedge_symbols=allowed_hedge_symbols,
    )

    logger.info(
        f"SignalCoordinator initialized: resolution={resolution.value}, "
        f"hedge_symbols={allowed_hedge_symbols}"
    )

    return coordinator


def print_banner():
    """Print startup banner."""
    print("""
============================================================
       整合交易系統 - Integrated Trading System
============================================================

啟動項目:
  ✓ Master 主控台
  ✓ Fund Manager (中央資金分配系統)
  ✓ Bollinger Bot (合約 18x) - BB_NEUTRAL_GRID 策略
  ✓ RSI-Grid Bot (合約 10x) - RSI 區域 + 網格進場
  ✓ Grid Futures Bot (合約 7x) - NEUTRAL 雙向網格
  ✓ Supertrend Bot (合約 10x) - HYBRID_GRID + RSI 過濾
  ✓ Discord Bot (遠端管理)

Walk-Forward 驗證通過的策略:
  Bollinger: BB(11,2.0)+Grid, 18x (100% 一致性, Sharpe 13.10)
  RSI-Grid: RSI(5)+Grid(6), 10x (Sharpe 7.47)
  Grid Futures: NEUTRAL 7x, hysteresis (100% 一致性, Sharpe 5.57)
  Supertrend: HYBRID_GRID+RSI filter, 10x (Sharpe 5.60)

資金分配 (中央管理):
  Grid Futures: 30%  |  Bollinger: 30%
  RSI-Grid: 15%      |  Supertrend: 15%
  保留金: 10%

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


async def create_data_manager(
    exchange: ExchangeClient,
    market_type: MarketType = MarketType.FUTURES,
) -> MarketDataManager:
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
        market_type=market_type,
    )
    await manager.connect()
    return manager


def create_notifier() -> NotificationManager:
    """Create notification manager."""
    return NotificationManager.from_env()


def get_bollinger_config() -> dict:
    """
    Get Bollinger Bot config from settings.yaml (單一來源).

    使用 strategy_loader 從 settings.yaml 讀取參數，支援環境變數覆蓋。
    """
    return load_strategy_config("bollinger")


def get_rsi_config() -> dict:
    """
    Get RSI Bot config.

    NOTE: This bot is replaced by RSI-Grid. Kept for backward compatibility.
    Returns RSI-Grid config instead.
    """
    return get_rsi_grid_config()


def get_rsi_grid_config() -> dict:
    """
    Get RSI-Grid Hybrid Bot config from settings.yaml (單一來源).

    使用 strategy_loader 從 settings.yaml 讀取參數，支援環境變數覆蓋。
    """
    return load_strategy_config("rsi_grid")


def get_grid_futures_config() -> dict:
    """
    Get Grid Futures Bot config from settings.yaml (單一來源).

    使用 strategy_loader 從 settings.yaml 讀取參數，支援環境變數覆蓋。
    """
    return load_strategy_config("grid_futures")


def get_supertrend_config() -> dict:
    """
    Get Supertrend Bot config from settings.yaml (單一來源).

    使用 strategy_loader 從 settings.yaml 讀取參數，支援環境變數覆蓋。
    """
    return load_strategy_config("supertrend")


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

    # 2. Create RSI-Grid Hybrid Bot (replaces RSI Momentum Bot)
    print("  創建 RSI-Grid Hybrid Bot...")
    rsi_grid_config = get_rsi_grid_config()
    result = await master.create_bot(BotType.RSI_GRID, rsi_grid_config)
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

    # 4. Create Supertrend Bot
    print("  創建 Supertrend Bot...")
    supertrend_config = get_supertrend_config()
    result = await master.create_bot(BotType.SUPERTREND, supertrend_config)
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
    global _master, _exchange, _data_manager, _notifier, _fund_manager, _signal_coordinator, _discord_bot

    print("\n正在關閉系統...")

    # Stop Discord bot
    if _discord_bot:
        try:
            await _discord_bot.stop_bot()
            print("  ✓ Discord Bot 已關閉")
        except Exception:
            pass

    # Stop Signal Coordinator
    if _signal_coordinator:
        try:
            await _signal_coordinator.stop()
            SignalCoordinator.reset_instance()
            print("  ✓ 訊號協調器已關閉")
        except Exception:
            pass

    # Stop Fund Manager
    if _fund_manager:
        try:
            await _fund_manager.stop()
            FundManager.reset_instance()
            print("  ✓ 資金管理系統已關閉")
        except Exception:
            pass

    # Stop Master (this stops all bots)
    if _master:
        try:
            await _master.stop()
            Master.reset_instance()
            print("  ✓ Master 已關閉")
        except Exception:
            pass

    # Disconnect exchange
    if _exchange:
        try:
            await _exchange.disconnect()
            print("  ✓ 交易所連接已關閉")
        except Exception:
            pass

    # Close notifier
    if _notifier:
        try:
            await _notifier.close()
        except Exception:
            pass

    print("\n系統已完全關閉")


async def main():
    """Main entry point."""
    global _master, _exchange, _data_manager, _notifier, _fund_manager, _signal_coordinator, _shutdown_event

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

        # 1.5. Ensure Hedge Mode is enabled (required for multi-bot trading)
        print("正在檢查倉位模式...")
        hedge_ok = await ensure_hedge_mode(_exchange)
        if hedge_ok:
            print("  ✓ 雙向持倉模式已啟用 (Hedge Mode)")
        else:
            raise RuntimeError(
                "無法啟用雙向持倉模式 (Hedge Mode)。"
                "多機器人同時做多做空需要雙向持倉，否則 Binance 會拒絕下單 (-4061)。"
                "請手動在 Binance Futures 設定中啟用「雙向持倉」後重試。"
            )

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

        # 5. Initialize Fund Manager (centralized capital allocation)
        print("正在初始化資金管理系統...")
        yaml_config = _load_yaml_config()
        fund_config = FundManagerConfig.from_yaml(yaml_config)
        _fund_manager = FundManager(
            exchange=_exchange,
            registry=_master.registry,
            notifier=_notifier,
            config=fund_config,
            market_type=MarketType.FUTURES,
        )
        await _fund_manager.start()
        print("  ✓ 資金管理系統啟動成功")

        # 6. Initialize Signal Coordinator (multi-bot conflict prevention)
        print("正在初始化訊號協調器...")
        _signal_coordinator = initialize_signal_coordinator()
        await _signal_coordinator.start()
        print(f"  ✓ 訊號協調器啟動成功 (對沖模式: BTCUSDT)")

        # 7. Create and start bots
        print("\n正在創建機器人...")
        bot_ids = await create_and_start_bots(_master)

        # 7.5. Register WebSocket reconnect callback to trigger position resync
        async def _on_ws_reconnect():
            """Trigger position reconciliation on all running bots after WS reconnect."""
            for bot_info in _master.get_all_bots():
                bot = bot_info.instance if hasattr(bot_info, 'instance') else None
                if bot and hasattr(bot, '_reconcile_position'):
                    try:
                        await bot._reconcile_position()
                    except Exception as e:
                        logger.warning(f"Post-reconnect reconcile failed for {bot_info.bot_id}: {e}")

        _exchange.register_reconnect_callback(_on_ws_reconnect)
        print("  ✓ WebSocket 重連回調已註冊（自動同步持倉）")

        # 8. Dispatch initial funds to bots
        print("\n正在分配資金...")
        dispatch_result = await _fund_manager.dispatch_funds(trigger="startup")
        if dispatch_result.success:
            print(f"  ✓ 資金分配成功")
            for allocation in dispatch_result.allocations:
                print(f"    • {allocation.bot_id}: {allocation.amount:.2f} USDT")
        else:
            print(f"  ⚠️ 資金分配警告: {dispatch_result.errors}")

        # 9. Start Discord bot
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
        except Exception:
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
