#!/usr/bin/env python3
"""
Bollinger Bot Runner.

啟動布林帶趨勢交易機器人 (BOLLINGER_TREND 策略)。

策略邏輯:
- 進場: Supertrend 看多時在 BB 下軌買入，看空時在 BB 上軌賣出
- 出場: Supertrend 翻轉（主要）或 ATR 止損（保護）

Walk-Forward 驗證結果 (2024-01 ~ 2026-01, 2 年數據, 8 期分割):
- 報酬: +35.1% (2 年)
- Sharpe: 1.81
- 最大回撤: 6.7%
- Walk-Forward 一致性: 75%
- OOS 效率: 96%
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
    """
    從環境變數讀取配置。

    Walk-Forward 驗證通過的默認參數:
    - bb_std: 3.0
    - st_atr_multiplier: 3.5
    - leverage: 2
    - atr_stop_multiplier: 2.0
    """
    # 資金分配：如果設定了 MAX_CAPITAL，則限制該 Bot 最大可用資金
    max_capital_str = os.getenv('BOLLINGER_MAX_CAPITAL', '')
    max_capital = Decimal(max_capital_str) if max_capital_str else None

    return BollingerConfig(
        # 基本設定
        symbol=os.getenv('BOLLINGER_SYMBOL', 'BTCUSDT'),
        timeframe=os.getenv('BOLLINGER_TIMEFRAME', '15m'),
        max_capital=max_capital,
        position_size_pct=Decimal(os.getenv('BOLLINGER_POSITION_SIZE', '0.1')),

        # Walk-Forward 驗證通過的參數 (2x, BB 3.0, ST 3.5)
        leverage=int(os.getenv('BOLLINGER_LEVERAGE', '2')),  # 降低槓桿提高穩定性
        bb_period=int(os.getenv('BOLLINGER_BB_PERIOD', '20')),
        bb_std=Decimal(os.getenv('BOLLINGER_BB_STD', '3.0')),  # Walk-Forward 最佳值

        # Supertrend 參數
        st_atr_period=int(os.getenv('BOLLINGER_ST_ATR_PERIOD', '20')),
        st_atr_multiplier=Decimal(os.getenv('BOLLINGER_ST_ATR_MULTIPLIER', '3.5')),  # Walk-Forward 最佳值

        # ATR Stop Loss
        atr_stop_multiplier=Decimal(os.getenv('BOLLINGER_ATR_STOP_MULTIPLIER', '2.0')),

        # BBW 過濾 (保留用於指標兼容)
        bbw_lookback=int(os.getenv('BOLLINGER_BBW_LOOKBACK', '200')),
        bbw_threshold_pct=int(os.getenv('BOLLINGER_BBW_THRESHOLD', '20')),
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
    print("    Bollinger Trend Bot - 布林帶趨勢交易機器人")
    print("=" * 60)

    # 讀取配置
    config = get_config_from_env()

    print(f"\n配置資訊：")
    print(f"  交易對: {config.symbol}")
    print(f"  時間框架: {config.timeframe}")
    print(f"  策略模式: BOLLINGER_TREND (Supertrend + BB)")
    print(f"  槓桿: {config.leverage}x")
    print(f"  保證金模式: ISOLATED (逐倉)")

    # 資金分配顯示
    if config.max_capital:
        print(f"  分配資金: {config.max_capital:,.0f} USDT")
        print(f"  單次倉位: {config.position_size_pct*100}% = {config.max_capital * config.position_size_pct:,.0f} USDT")
    else:
        print(f"  分配資金: 全部餘額")
        print(f"  單次倉位: {config.position_size_pct*100}%")

    print(f"\n  Bollinger Bands:")
    print(f"    週期: {config.bb_period}")
    print(f"    標準差: {config.bb_std}σ")

    print(f"\n  Supertrend:")
    print(f"    ATR 週期: {config.st_atr_period}")
    print(f"    ATR 乘數: {config.st_atr_multiplier}")

    print(f"\n  ATR Stop Loss:")
    print(f"    乘數: {config.atr_stop_multiplier}")

    print(f"\n  Walk-Forward 驗證結果:")
    print(f"    報酬: +35.1% (2年), Sharpe: 1.81")
    print(f"    回撤: 6.7%, 一致性: 75%")
    print(f"    OOS 效率: 96%")
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
