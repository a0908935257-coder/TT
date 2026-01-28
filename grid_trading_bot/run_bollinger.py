#!/usr/bin/env python3
"""
Bollinger BB_NEUTRAL_GRID Bot Runner.

啟動布林帶中性網格交易機器人。

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據):
- 年化報酬: 44.30%
- Sharpe: 9.74
- 回撤: 3.76%
- W-F 一致性: 100% (9/9)
- Monte Carlo: 100% (15/15)

⚠️ 參數來自 settings.yaml (單一來源):
- 所有策略參數從 settings.yaml 讀取
- 支援環境變數覆蓋 (例: BOLLINGER_LEVERAGE=10)

Usage:
    python run_bollinger.py

    # 覆蓋特定參數
    BOLLINGER_LEVERAGE=10 python run_bollinger.py
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
from src.config import load_strategy_config

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global bot reference for signal handling
_bot: BollingerBot | None = None


def get_config_from_env() -> BollingerConfig:
    """
    從 settings.yaml 讀取配置，支援環境變數覆蓋。

    單一來源設計：
    - 所有策略參數從 settings.yaml 讀取
    - 支援環境變數覆蓋 (格式: BOLLINGER_{PARAM_NAME})

    Example:
        # 使用 YAML 參數
        python run_bollinger.py

        # 覆蓋特定參數
        BOLLINGER_LEVERAGE=10 python run_bollinger.py
    """
    # 從 settings.yaml 加載參數（支援環境變數覆蓋）
    params = load_strategy_config("bollinger")

    # 資金分配（環境變數覆蓋）
    max_capital_str = os.getenv('BOLLINGER_MAX_CAPITAL', '')
    max_capital = Decimal(max_capital_str) if max_capital_str else None

    return BollingerConfig(
        # 基本設定
        symbol=params.get('symbol', 'BTCUSDT'),
        timeframe=params.get('timeframe', '1h'),
        leverage=int(params.get('leverage', 18)),
        margin_type=params.get('margin_type', 'ISOLATED'),

        # Strategy Mode
        mode=params.get('mode', 'bb_neutral_grid'),

        # Bollinger Bands 參數
        bb_period=int(params.get('bb_period', 20)),
        bb_std=Decimal(str(params.get('bb_std', 2.0))),

        # Grid 參數
        grid_count=int(params.get('grid_count', 12)),
        take_profit_grids=int(params.get('take_profit_grids', 1)),

        # ATR Dynamic Range
        use_atr_range=bool(params.get('use_atr_range', True)),
        atr_period=int(params.get('atr_period', 21)),
        atr_multiplier=Decimal(str(params.get('atr_multiplier', 6.0))),
        fallback_range_pct=Decimal(str(params.get('fallback_range_pct', 0.04))),

        # 倉位管理
        max_capital=max_capital,
        position_size_pct=Decimal(str(params.get('position_size_pct', 0.1))),
        max_position_pct=Decimal(str(params.get('max_position_pct', 0.5))),

        # 風險管理
        stop_loss_pct=Decimal(str(params.get('stop_loss_pct', 0.005))),
        rebuild_threshold_pct=Decimal(str(params.get('rebuild_threshold_pct', 0.02))),

        # BBW 過濾
        bbw_lookback=int(params.get('bbw_lookback', 200)),
        bbw_threshold_pct=int(params.get('bbw_threshold_pct', 20)),

        # Protective features
        use_hysteresis=bool(params.get('use_hysteresis', True)),
        hysteresis_pct=Decimal(str(params.get('hysteresis_pct', 0.002))),
        use_signal_cooldown=bool(params.get('use_signal_cooldown', False)),
        cooldown_bars=int(params.get('cooldown_bars', 0)),
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
    print("    Bollinger BB_TREND_GRID Bot - 趨勢網格交易機器人")
    print("=" * 60)

    print("\n✅ Walk-Forward 驗證通過 (80% 一致性, OOS Sharpe 6.56)")

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
        print(f"  單次倉位: {config.position_size_pct*100}%")
    else:
        print(f"  分配資金: 全部餘額")
        print(f"  單次倉位: {config.position_size_pct*100}%")

    print(f"\nBollinger Bands 設定:")
    print(f"  BB 週期: {config.bb_period}")
    print(f"  BB 標準差: {config.bb_std}σ")

    print(f"\nGrid 設定:")
    print(f"  網格數量: {config.grid_count}")
    print(f"  網格範圍: ±{config.grid_range_pct*100}%")
    print(f"  止盈網格: {config.take_profit_grids}")

    print(f"\n風險控制:")
    print(f"  止損: {config.stop_loss_pct*100:.2f}%")
    print(f"  網格重建閾值: {config.rebuild_threshold_pct*100:.1f}%")
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
            print(f"  槓桿: {status.get('leverage')}x")
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
