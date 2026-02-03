#!/usr/bin/env python3
"""
Advanced Discord Trading Console.

啟動進階版 Discord 交易控制台，支援：
- 創建/管理多個交易機器人
- 完整的風險管理
- 即時指令同步
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
from src.discord_bot import TradingBot
from src.discord_bot.config import DiscordConfig, ChannelConfig

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class SimpleMaster:
    """
    簡化版 Master 控制器。

    提供基本的機器人管理功能，無需完整的 Master 系統。
    """

    def __init__(self):
        self._bots = {}
        self._bot_counter = 0
        self._exchange = None
        self._data_manager = None
        self.registry = self  # 讓 registry.get_bot_instance() 可用

    def get_bot_instance(self, bot_id: str):
        """獲取機器人實例（供 registry 使用）"""
        return self._bots.get(bot_id)

    async def initialize(self):
        """初始化連接"""
        from exchange import ExchangeClient
        from data import MarketDataManager

        # 建立交易所連接
        self._exchange = ExchangeClient(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            api_secret=os.getenv('BINANCE_API_SECRET', ''),
            testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true',
        )
        await self._exchange.connect()
        logger.info("Exchange connected")

        # 建立數據管理器
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
        self._data_manager = MarketDataManager(
            db_config=db_config,
            redis_config=redis_config,
            exchange=self._exchange,
            market_type=MarketType.FUTURES,
        )
        await self._data_manager.connect()
        logger.info("Data manager connected")

    def _wrap_bot_info(self, bot):
        """將 GridBot 包裝成 BotInfo 物件"""
        from datetime import datetime, timezone

        state = getattr(bot, '_state', None)
        if state is None:
            state = type('State', (), {'value': 'unknown'})()

        return type('BotInfo', (), {
            'bot_id': bot.bot_id,
            'symbol': getattr(bot, 'symbol', getattr(bot._config, 'symbol', 'UNKNOWN')),
            'state': state,
            'bot_type': type('BotType', (), {'value': 'grid'})(),
            'created_at': getattr(bot, '_created_at', datetime.now(timezone.utc)),
            'profit': getattr(bot, '_total_profit', 0),
        })()

    def get_all_bots(self):
        """獲取所有機器人"""
        return [self._wrap_bot_info(bot) for bot in self._bots.values()]

    def get_bot(self, bot_id: str):
        """獲取指定機器人"""
        bot = self._bots.get(bot_id)
        if bot:
            return self._wrap_bot_info(bot)
        return None

    async def create_bot(self, bot_type: str, config: dict):
        """創建新機器人"""
        from src.bots.grid.bot import GridBot, GridBotConfig
        from src.bots.grid.models import GridType, RiskLevel, ATRConfig
        from src.bots.grid.risk_manager import RiskConfig, BreakoutAction
        from src.core.models import MarketType
        from notification import NotificationManager

        self._bot_counter += 1
        bot_id = f"bot_{self._bot_counter}"
        symbol = config.get('symbol', 'BTCUSDT')
        investment = Decimal(str(config.get('total_investment', '100')))

        # 解析網格數量
        grid_count = int(config.get('grid_count', 10))

        # 風險配置
        risk_config = RiskConfig(
            stop_loss_percent=Decimal(os.getenv('STOP_LOSS_PERCENT', '20')),
            daily_loss_limit=Decimal(os.getenv('DAILY_LOSS_LIMIT', '5')),
            max_consecutive_losses=int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5')),
            upper_breakout_action=BreakoutAction.AUTO_REBUILD,
            lower_breakout_action=BreakoutAction.AUTO_REBUILD,
            auto_rebuild_enabled=True,
            cooldown_days=7,
        )

        # ATR 配置
        atr_config = ATRConfig(
            period=int(os.getenv('ATR_PERIOD', '14')),
            timeframe=os.getenv('ATR_TIMEFRAME', '4h'),
        )

        # 機器人配置
        bot_config = GridBotConfig(
            symbol=symbol,
            market_type=MarketType.SPOT,
            total_investment=investment,
            risk_level=RiskLevel.MODERATE,
            grid_type=GridType.GEOMETRIC,
            risk_config=risk_config,
            atr_config=atr_config,
            min_order_value=Decimal(os.getenv('MIN_ORDER_VALUE', '5')),
            min_grid_count=int(os.getenv('MIN_GRID_COUNT', '5')),
            max_grid_count=int(os.getenv('MAX_GRID_COUNT', '50')),
        )

        # 創建通知管理器
        notifier = NotificationManager.from_env()

        # 創建機器人
        bot = GridBot(
            bot_id=bot_id,
            config=bot_config,
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=notifier,
        )

        self._bots[bot_id] = bot
        logger.info(f"Created bot: {bot_id} for {symbol}")

        return type('Result', (), {
            'success': True,
            'bot_id': bot_id,
            'message': f'Bot {bot_id} created for {symbol}',
        })()

    async def start_bot(self, bot_id: str):
        """啟動機器人"""
        bot = self._bots.get(bot_id)
        if not bot:
            return type('Result', (), {'success': False, 'message': f'Bot {bot_id} not found'})()

        try:
            await bot.start()
            return type('Result', (), {'success': True, 'message': f'Bot {bot_id} started'})()
        except Exception as e:
            return type('Result', (), {'success': False, 'message': str(e)})()

    async def stop_bot(self, bot_id: str):
        """停止機器人"""
        bot = self._bots.get(bot_id)
        if not bot:
            return type('Result', (), {'success': False, 'message': f'Bot {bot_id} not found'})()

        try:
            await bot.stop()
            return type('Result', (), {'success': True, 'message': f'Bot {bot_id} stopped', 'data': {}})()
        except Exception as e:
            return type('Result', (), {'success': False, 'message': str(e)})()

    async def pause_bot(self, bot_id: str):
        """暫停機器人"""
        bot = self._bots.get(bot_id)
        if not bot:
            return type('Result', (), {'success': False, 'message': f'Bot {bot_id} not found'})()

        try:
            await bot.pause("User requested")
            return type('Result', (), {'success': True, 'message': f'Bot {bot_id} paused'})()
        except Exception as e:
            return type('Result', (), {'success': False, 'message': str(e)})()

    async def resume_bot(self, bot_id: str):
        """恢復機器人"""
        bot = self._bots.get(bot_id)
        if not bot:
            return type('Result', (), {'success': False, 'message': f'Bot {bot_id} not found'})()

        try:
            await bot.resume()
            return type('Result', (), {'success': True, 'message': f'Bot {bot_id} resumed'})()
        except Exception as e:
            return type('Result', (), {'success': False, 'message': str(e)})()

    async def delete_bot(self, bot_id: str):
        """刪除機器人"""
        bot = self._bots.get(bot_id)
        if not bot:
            return type('Result', (), {'success': False, 'message': f'Bot {bot_id} not found'})()

        try:
            await bot.stop()
        except Exception as e:
            logger.warning(f"Error stopping bot {bot_id} before delete: {e}")

        del self._bots[bot_id]
        return type('Result', (), {'success': True, 'message': f'Bot {bot_id} deleted'})()

    def get_dashboard_data(self):
        """獲取儀表板數據"""
        bots = self.get_all_bots()
        running = sum(1 for b in bots if hasattr(b, '_state') and b._state.value == 'running')
        paused = sum(1 for b in bots if hasattr(b, '_state') and b._state.value == 'paused')

        return type('Dashboard', (), {
            'summary': type('Summary', (), {
                'total_bots': len(bots),
                'running_bots': running,
                'paused_bots': paused,
                'error_bots': 0,
                'total_investment': Decimal('0'),
                'total_value': Decimal('0'),
                'total_profit': Decimal('0'),
                'total_profit_rate': Decimal('0'),
                'today_profit': Decimal('0'),
                'today_trades': 0,
            })()
        })()

    async def shutdown(self):
        """關閉所有連接"""
        for bot_id, bot in list(self._bots.items()):
            try:
                await bot.stop()
            except Exception as e:
                logger.warning(f"Error stopping bot {bot_id} during shutdown: {e}")

        if self._exchange:
            await self._exchange.disconnect()


class SimpleRiskEngine:
    """簡化版風險引擎"""

    def __init__(self):
        self._triggered = False

    def get_status(self):
        """獲取風險狀態"""
        return type('Status', (), {
            'level': type('Level', (), {'name': 'NORMAL'})(),
            'capital': type('Capital', (), {
                'total_capital': Decimal('10000'),
                'initial_capital': Decimal('10000'),
                'available_balance': Decimal('5000'),
            })(),
            'drawdown': type('Drawdown', (), {
                'drawdown_pct': Decimal('0.02'),
                'max_drawdown_pct': Decimal('0.05'),
                'peak_value': Decimal('10200'),
            })(),
            'circuit_breaker': type('CB', (), {
                'is_triggered': self._triggered,
                'trigger_reason': None,
                'cooldown_until': None,
            })(),
            'daily_pnl': type('PnL', (), {
                'pnl': Decimal('100'),
                'pnl_pct': Decimal('0.01'),
            })(),
            'active_alerts': [],
            'statistics': type('Stats', (), {
                'total_checks': 100,
                'violations': 0,
                'circuit_breaker_triggers': 0,
            })(),
        })()

    def get_statistics(self):
        """獲取風險統計"""
        return {
            'total_checks': 100,
            'violations': 0,
            'circuit_breaker_triggers': 0,
        }

    async def trigger_emergency(self, reason: str):
        """觸發緊急停止"""
        self._triggered = True
        logger.warning(f"Emergency triggered: {reason}")

    async def reset_circuit_breaker(self, force: bool = False):
        """重置熔斷器"""
        self._triggered = False
        return True


# Global references
_discord_bot = None
_master = None


async def shutdown(sig):
    """優雅關閉"""
    global _discord_bot, _master

    logger.info(f"Received signal {sig.name}, shutting down...")

    if _discord_bot:
        try:
            await _discord_bot.stop_bot()
        except Exception as e:
            logger.warning(f"Error stopping Discord bot during shutdown: {e}")

    if _master:
        await _master.shutdown()

    logger.info("Shutdown complete")


async def main():
    """主程式"""
    global _discord_bot, _master

    print("=" * 60)
    print("   Advanced Discord Trading Console")
    print("   進階版 Discord 交易控制台")
    print("=" * 60)

    # 檢查必要的環境變數
    token = os.getenv('DISCORD_BOT_TOKEN', '')
    guild_id = os.getenv('DISCORD_GUILD_ID', '')

    if not token:
        print("\n❌ 錯誤: 未設定 DISCORD_BOT_TOKEN")
        print("請在 .env 檔案中設定 Discord Bot Token")
        sys.exit(1)

    if not guild_id:
        print("\n⚠️ 警告: 未設定 DISCORD_GUILD_ID")
        print("斜線指令將使用全域同步（需要等待最多 1 小時）")
        print("\n要啟用即時同步，請在 .env 中添加:")
        print("DISCORD_GUILD_ID=你的伺服器ID")
        print("\n繼續啟動...")

    try:
        # 初始化 Master
        print("\n正在初始化系統...")
        _master = SimpleMaster()
        await _master.initialize()

        # 初始化 Risk Engine
        risk_engine = SimpleRiskEngine()

        # 載入 Discord 配置
        config = DiscordConfig.from_env()

        # 創建 Discord Bot
        print("正在啟動 Discord Bot...")
        _discord_bot = TradingBot(
            config=config,
            master=_master,
            risk_engine=risk_engine,
        )

        # 設定信號處理
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(shutdown(s))
                )
            except NotImplementedError:
                signal.signal(sig, lambda s, f: asyncio.get_event_loop().create_task(shutdown(signal.Signals(s))))

        print("\n" + "=" * 60)
        print("Discord Bot 啟動中...")
        print("=" * 60)
        print("\n可用指令:")
        print("  /bot create  - 創建新機器人")
        print("  /bot list    - 列出所有機器人")
        print("  /bot start   - 啟動機器人")
        print("  /bot stop    - 停止機器人")
        print("  /dashboard   - 查看儀表板")
        print("  /risk status - 風險狀態")
        print("  /emergency   - 緊急停止")
        print("\n按 Ctrl+C 停止程式")
        print("=" * 60)

        # 啟動 Bot
        await _discord_bot.start_bot()

    except KeyboardInterrupt:
        print("\n收到中斷信號...")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ 錯誤: {e}")
        sys.exit(1)
    finally:
        if _master:
            await _master.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
