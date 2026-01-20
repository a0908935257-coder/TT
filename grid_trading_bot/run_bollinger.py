#!/usr/bin/env python3
"""
Bollinger Bot Runner.

啟動布林帶均值回歸交易機器人。
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
from src.bots.bollinger.bot import BollingerBot
from src.bots.bollinger.models import BollingerConfig

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global bot reference for signal handling
_bot: BollingerBot | None = None


def get_config_from_env() -> BollingerConfig:
    """從環境變數讀取配置"""
    # 資金分配：如果設定了 MAX_CAPITAL，則限制該 Bot 最大可用資金
    max_capital_str = os.getenv('BOLLINGER_MAX_CAPITAL', '')
    max_capital = Decimal(max_capital_str) if max_capital_str else None

    return BollingerConfig(
        # 基本設定
        symbol=os.getenv('BOLLINGER_SYMBOL', 'BTCUSDT'),
        timeframe=os.getenv('BOLLINGER_TIMEFRAME', '15m'),
        leverage=int(os.getenv('BOLLINGER_LEVERAGE', '1')),
        max_capital=max_capital,
        position_size_pct=Decimal(os.getenv('BOLLINGER_POSITION_SIZE', '0.1')),

        # 布林帶設定
        bb_period=int(os.getenv('BOLLINGER_BB_PERIOD', '20')),
        bb_std=Decimal(os.getenv('BOLLINGER_BB_STD', '2.0')),

        # BBW 壓縮過濾 (35% = 更嚴格過濾)
        bbw_lookback=int(os.getenv('BOLLINGER_BBW_LOOKBACK', '100')),
        bbw_threshold_pct=int(os.getenv('BOLLINGER_BBW_THRESHOLD', '35')),

        # 趨勢過濾
        use_trend_filter=os.getenv('BOLLINGER_USE_TREND_FILTER', 'true').lower() == 'true',
        trend_period=int(os.getenv('BOLLINGER_TREND_PERIOD', '50')),

        # RSI 過濾 (Sharpe > 1 優化)
        use_rsi_filter=os.getenv('BOLLINGER_USE_RSI_FILTER', 'true').lower() == 'true',
        rsi_period=int(os.getenv('BOLLINGER_RSI_PERIOD', '14')),
        rsi_oversold=int(os.getenv('BOLLINGER_RSI_OVERSOLD', '30')),
        rsi_overbought=int(os.getenv('BOLLINGER_RSI_OVERBOUGHT', '70')),

        # ATR 動態止損
        use_atr_stop=os.getenv('BOLLINGER_USE_ATR_STOP', 'true').lower() == 'true',
        atr_period=int(os.getenv('BOLLINGER_ATR_PERIOD', '14')),
        atr_multiplier=Decimal(os.getenv('BOLLINGER_ATR_MULTIPLIER', '2.0')),

        # 止損/持倉
        stop_loss_pct=Decimal(os.getenv('BOLLINGER_STOP_LOSS_PCT', '0.02')),
        max_hold_bars=int(os.getenv('BOLLINGER_MAX_HOLD_BARS', '24')),
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
    print("    Bollinger Bot - 布林帶均值回歸交易機器人")
    print("=" * 60)

    # 讀取配置
    config = get_config_from_env()

    print(f"\n配置資訊：")
    print(f"  交易對: {config.symbol}")
    print(f"  時間框架: {config.timeframe}")
    print(f"  槓桿: {config.leverage}x")

    # 資金分配顯示
    if config.max_capital:
        print(f"  分配資金: {config.max_capital:,.0f} USDT")
        print(f"  單次倉位: {config.position_size_pct*100}% = {config.max_capital * config.position_size_pct:,.0f} USDT")
    else:
        print(f"  分配資金: 全部餘額")
        print(f"  單次倉位: {config.position_size_pct*100}%")

    print(f"  布林帶週期: {config.bb_period}")
    print(f"  布林帶標準差: {config.bb_std}")
    print(f"  BBW 過濾閾值: {config.bbw_threshold_pct}%")
    print(f"  趨勢過濾: {'開啟' if config.use_trend_filter else '關閉'} (SMA {config.trend_period})")
    print(f"  RSI 過濾: {'開啟' if config.use_rsi_filter else '關閉'} ({config.rsi_oversold}/{config.rsi_overbought})")
    print(f"  ATR 止損: {'開啟' if config.use_atr_stop else '關閉'} ({config.atr_multiplier}x ATR)")
    print(f"  最大持倉: {config.max_hold_bars} 根 K 線")
    print(f"\n  策略: S4 雙重過濾 (2年回測 Sharpe 2.05)")
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
        _bot = BollingerBot(
            bot_id=f"bollinger_{config.symbol.lower()}",
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
            print(f"  當前 K 線: {status.get('current_bar')}")
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
